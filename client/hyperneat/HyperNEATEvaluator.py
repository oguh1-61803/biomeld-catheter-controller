# All necessary libraries and imports from other files.
from client.hyperneat.Substrate import Substrate
from concurrent.futures import ProcessPoolExecutor
import requests
import base64
import random
import json
import math
import neat


# This class defines the fitness function. It evaluates individuals (phase offsets) through Voxelyze instances which are
# deployed in the server. It receives the configuration data of: substrate data, fitness function, server, CPNNs, and
# the number of individuals.
class HyperNEATEvaluator:

    # Constant values required for evolving individuals.
    X_BIAS = Y_BIAS = 0.0
    FITTEST_PATH = "fittest/"
    MAPPING_REFERENCE = 0.5

    POSITIVE_MAPPING_REFERENCE = math.pi * 2
    NEGATIVE_MAPPING_REFERENCE = POSITIVE_MAPPING_REFERENCE * -1.0

    # Constructor
    def __init__(self, catheter_path, substrate_data, fitness_data, server_data, cppn_data, number_of_individuals):

        self.substrate_data = substrate_data
        self.coordinates_and_material_id_list = []
        self.target_url = None
        self.number_of_workers = None
        self.is_recurrent = cppn_data.get("recurrent_topology")
        self.fitness_data = fitness_data
        self.catheter_reference_list = []
        self.list_of_catheter_morphologies = []

        self.__initialise_layout(catheter_path, number_of_individuals)
        self.__initialise_server(server_data)

    # This function evaluates the population of phase offsets through the Voxelyze instances deployed in the server.
    def evaluate_individuals(self, genomes, config):

        list_of_cppns = []

        for genome_id, genome in genomes:

            if self.is_recurrent:

                cppn = neat.nn.RecurrentNetwork.create(genome, config)
                list_of_cppns.append(cppn)

            else:

                cppn = neat.nn.FeedForwardNetwork.create(genome, config)
                list_of_cppns.append(cppn)

        list_of_substrates = []

        with ProcessPoolExecutor(max_workers=self.number_of_workers) as executor:

            for substrate_neural_network in executor.map(self.build_substrate, list_of_cppns, chunksize=2):
                list_of_substrates.append(substrate_neural_network)

        list_of_offsets = []

        with ProcessPoolExecutor(max_workers=self.number_of_workers) as executor:

            for offsets in executor.map(self.build_offsets, list_of_substrates, chunksize=2):

                list_of_offsets.append(offsets)

        list_of_fitness_values = []

        with ProcessPoolExecutor(max_workers=self.number_of_workers) as executor:

            for fitness_value in executor.map(self.evaluate_offsets_in_server, self.list_of_catheter_morphologies,
                                              list_of_offsets, chunksize=2):
                list_of_fitness_values.append(fitness_value)

        if len(genomes) > len(list_of_fitness_values):

            missed_aptitudes = len(genomes) - len(list_of_fitness_values)

            for _ in range(0, missed_aptitudes):

                list_of_fitness_values.append(random.uniform(0.0015, 0.0025))

        index = 0

        for genome_id, genome in genomes:
            genome.fitness = list_of_fitness_values[index]
            index += 1

    # This function returns the substrate built by a CPPN, which is received as a parameter.
    def build_substrate(self, cppn):

        substrate = Substrate()
        substrate.initialise_substrate(self.substrate_data)

        for layer in substrate.hidden_layers:

            for neuron in layer:

                coordinates_b = neuron.coordinates

                for key in neuron.links.keys():

                    link = neuron.links.get(key)
                    coordinates_a = link[0]
                    weight_cppn_input = (coordinates_a[0], coordinates_a[1], coordinates_b[0], coordinates_b[1])
                    link[1] = cppn.activate(weight_cppn_input).pop()

                bias_input = (coordinates_b[0], coordinates_b[0], self.X_BIAS, self.Y_BIAS)
                neuron.bias = cppn.activate(bias_input).pop()

        for output_neuron in substrate.output_layer:

            coordinates_b = output_neuron.coordinates

            for key in output_neuron.links.keys():

                link = output_neuron.links.get(key)
                coordinates_a = link[0]
                weight_cppn_input = (coordinates_a[0], coordinates_a[1], coordinates_b[0], coordinates_b[1])
                link[1] = cppn.activate(weight_cppn_input).pop()

            bias_input = (coordinates_b[0], coordinates_b[0], self.X_BIAS, self.Y_BIAS)
            output_neuron.bias = cppn.activate(bias_input).pop()

        input_neurons = substrate.get_input_neurons()
        output_neurons = substrate.get_output_neurons()
        neurons_evals = substrate.get_neurons_evals()
        substrate_neural_network = neat.nn.FeedForwardNetwork(input_neurons, output_neurons, neurons_evals)

        return substrate_neural_network

    # This function returns phase offsets built by a substrate, which is received as a parameter.
    def build_offsets(self, substrate):

        list_of_offsets = []

        for coor_mat_id in self.coordinates_and_material_id_list:

            offset = []

            for layer in coor_mat_id.keys():

                layer_data = coor_mat_id.get(layer)
                layer_offset = ""

                for coordinate in layer_data.keys():

                    material_id = layer_data.get(coordinate)
                    substrate_input = [coordinate[0], coordinate[1], coordinate[2], material_id]
                    offset_value = substrate.activate(substrate_input)

                    if offset_value[0] >= self.POSITIVE_MAPPING_REFERENCE:

                        layer_offset += str(self.POSITIVE_MAPPING_REFERENCE) + ", "

                    elif offset_value[0] <= self.NEGATIVE_MAPPING_REFERENCE:

                        layer_offset += str(self.NEGATIVE_MAPPING_REFERENCE) + ", "

                    else:

                        layer_offset += str(offset_value[0]) + ", "

                layer_offset = layer_offset[:-2]
                offset.append(layer_offset)

            list_of_offsets.append(offset)

        return list_of_offsets

    # This function sends a phase offset (and morphologies) to the server to be evaluated and returns the evaluation
    # provided by the server.
    def evaluate_offsets_in_server(self, catheter_morphology_list, offset_list):

        accumulated_displacement = 0.0

        for catheter_morphology, offset in zip(catheter_morphology_list, offset_list):

            catheter = {

                "evaluation": True,
                "layers": catheter_morphology,
                "offsets": offset

            }

            r = requests.get(url=self.target_url + "/biomeld-hn", json=catheter).json()
            z_trace = r.get("trace").get("z")
            accumulated_displacement += float(z_trace.get("final")) - float(z_trace.get("initial"))

        return accumulated_displacement / len(catheter_morphology_list)

    # This function generates a .vxa file of the fittest phase offset found by HyperNEAT algorithm.
    def get_fittest_individual_file(self, substrate, experiment_id):

        offset_list = self.build_offsets(substrate)

        catheter = {

            "evaluation": False,
            "layers": self.catheter_reference_list[0],
            "offsets": offset_list[0]

        }

        r = requests.get(url=self.target_url + "/biomeld-hn", json=catheter).json()
        raw_file = r.get("individual_file")
        bytes_file = raw_file.encode()
        final_file = base64.b64decode(bytes_file)
        path = self.FITTEST_PATH + "fittest_" + str(experiment_id) + ".vxa"
        fitness_values = r.get("fitness_values")
        print("Number of voxels: ", fitness_values.get("voxels"))
        print("Displacement: ", fitness_values.get("displacement"))
        print("Instance used:", r.get("instance"))

        with open(path, "wb") as file:

            file.write(final_file)
            file.close()

    # This function initialises all the data necessary for the GA and the morphologies used to evaluate the
    # phase offsets.
    def __initialise_layout(self, catheter_path, number_of_individuals):

        with open(catheter_path, "r") as file:

            catheter_file = file.read()
            catheter_dictionary = json.loads(catheter_file)

        for key in catheter_dictionary.keys():

            catheter_morphology = catheter_dictionary.get(key)
            coordinates_and_material_id = {}

            z_counter = 0
            x_reference = 20

            for layer in catheter_morphology:

                layer_dictionary = {}
                y_counter = 0
                x_counter = 0

                for material_id in layer:

                    if material_id == "0":

                        layer_dictionary[(float(x_counter), float(y_counter), float(z_counter))] = 0.0

                    elif material_id == "1":

                        layer_dictionary[(float(x_counter), float(y_counter), float(z_counter))] = 1.0

                    else:

                        layer_dictionary[(float(x_counter), float(y_counter), float(z_counter))] = 3.0

                    x_counter += 1

                    if x_counter == x_reference:

                        x_counter = 0
                        y_counter += 1

                coordinates_and_material_id["layer_" + str(z_counter)] = layer_dictionary
                z_counter += 1

            self.coordinates_and_material_id_list.append(coordinates_and_material_id)
            self.catheter_reference_list.append(catheter_morphology)

        for _ in range(0, number_of_individuals):

            self.list_of_catheter_morphologies.append(self.catheter_reference_list)

    # This function initialises the Voxelyze instances deployed in the server.
    def __initialise_server(self, server_config):

        self.target_url = server_config.get("target_ip_address") + "8000"
        self.number_of_workers = server_config.get("simulator_instances")
        initial_port = server_config.get("initial_port")

        for _ in range(0, self.number_of_workers):

            initialisation = {

                "voxels": (float(self.fitness_data.get("layout").get("x")),
                           float(self.fitness_data.get("layout").get("y")),
                           float(self.fitness_data.get("layout").get("z")))
            }

            print(initialisation)
            r = requests.post(url=server_config.get("target_ip_address") + str(initial_port) + "/biomeld-hn",
                              json=initialisation)
            print(r.json())
            initial_port += 1
