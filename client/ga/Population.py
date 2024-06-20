# All necessary libraries and imports from other files.
from concurrent.futures import ProcessPoolExecutor
from client.ga.Individual import Individual
import requests
import base64
import random
import json
import math
import copy


# This class represent the population of solutions. It has all the method to evolve solutions.
class Population:

    # Constant values required for evolving individuals.
    FITTEST_PATH = "fittest/"

    CROSSOVER_PROBABILITY = 0.9
    MUTATION_PROBABILITY = 0.1

    POSITIVE_MAPPING_REFERENCE = math.pi * 2
    NEGATIVE_MAPPING_REFERENCE = POSITIVE_MAPPING_REFERENCE * -1.0

    # Constructor
    def __init__(self, individuals, server_configuration, catheters_path):

        self.number_of_individuals = individuals
        self.initial_population = []
        self.intermediate_population = []
        self.best_individual = Individual()
        self.catheter_reference_list = []
        self.list_of_catheter_morphologies = []

        self.number_of_workers = None
        self.target_url = None

        self.__initialise_morphologies_and_server(server_configuration, catheters_path)

    # This method initialises individuals.
    def create_population(self):

        for _ in range(self.number_of_individuals):

            individual = Individual()
            individual.initialise()
            self.initial_population.append(individual)

    # This method evaluates individuals. All the subtasks are executed concurrently.
    def evaluate_population(self):

        list_of_offsets = []

        with ProcessPoolExecutor(max_workers=self.number_of_workers) as executor:

            for offsets in executor.map(self.build_offset, self.initial_population, chunksize=2):

                list_of_offsets.append(offsets)

        list_of_fitness_values = []

        with ProcessPoolExecutor(max_workers=self.number_of_workers) as executor:

            for fitness_value in executor.map(self.evaluate_offset_in_server, self.list_of_catheter_morphologies,
                                              list_of_offsets, chunksize=2):

                list_of_fitness_values.append(fitness_value)

        for individual, fitness_value in zip(self.initial_population, list_of_fitness_values):

            individual.fitness = fitness_value

    # This method configures the phase offset (i.e., the genotype of the individual) to be written in the .vxa file.
    def build_offset(self, individual):

        genotype = individual.genotype
        first_layer = genotype[:160]
        second_layer = genotype[160:320]
        third_layer = genotype[320:480]
        fourth_layer = genotype[480:640]
        fifth_layer = genotype[640:800]
        sixth_layer = genotype[800:960]
        seventh_layer = genotype[960:1120]
        eighth_layer = genotype[1120:1280]

        raw_layers = [first_layer, second_layer, third_layer, fourth_layer, fifth_layer, sixth_layer, seventh_layer, eighth_layer]
        processed_layers = []

        for layer in raw_layers:

            layer_offset = ""

            for value in layer:

                layer_offset += str(value) + ", "

            layer_offset = layer_offset[:-2]
            processed_layers.append(layer_offset)

        return processed_layers

    # This method sends to the server the phase offset and morphologies to be evaluated.
    def evaluate_offset_in_server(self, catheter_morphology_list, offset):

        accumulated_displacement = 0.0

        for catheter_morphology in catheter_morphology_list:

            catheter = {

                "evaluation": True,
                "layers": catheter_morphology,
                "offsets": offset

            }

            r = requests.get(url=self.target_url + "/biomeld-hn", json=catheter).json()
            z_trace = r.get("trace").get("z")
            accumulated_displacement += float(z_trace.get("final")) - float(z_trace.get("initial"))

        return accumulated_displacement / len(catheter_morphology_list)

    # This method saves the best individual of the generation.
    def get_best_individual(self):

        best_index = 0
        best_fitness = 0.0

        for index in range(0, len(self.initial_population)):

            if self.initial_population[index].fitness > best_fitness:

                best_index = index
                best_fitness = self.initial_population[index].fitness

        if best_fitness > self.best_individual.fitness:

            self.best_individual = copy.deepcopy(self.initial_population[best_index])

        print(self.best_individual.fitness)

    # This method generates a new population through the tournament selection (T=2).
    def execute_tournament(self):

        self.intermediate_population = []
        adversaries = random.sample(range(0, self.number_of_individuals), self.number_of_individuals)

        for index in range(0, self.number_of_individuals):

            if self.initial_population[index].fitness < self.initial_population[adversaries[index]].fitness:

                self.intermediate_population.insert(index, self.initial_population[adversaries[index]])

            else:

                self.intermediate_population.insert(index, self.initial_population[index])

    # This method executes crossover and mutation.
    def execute_genetic_operators(self):

        self.__execute_crossover()
        self.__execute_mutation()

    # This method copies the population generated by tournament and genetic operators to the initial population.
    def build_new_population(self):

        self.initial_population = []
        self.initial_population.insert(0, self.best_individual)

        for index in range(1, self.number_of_individuals):

            self.initial_population.insert(index, self.intermediate_population[index])

    # This method combines two individuals to generate two new individuals.
    def __execute_crossover(self):

        crossover_aux = random.sample(range(0, self.number_of_individuals), self.number_of_individuals)
        children = []
        crossover_aux_index = 0

        for index in range(0, int(self.number_of_individuals / 2)):

            crossover_probability = random.uniform(0.0, 1.0)

            if crossover_probability <= self.CROSSOVER_PROBABILITY:

                mother_index = crossover_aux[index]
                father_index = crossover_aux[index + 1]
                mother = copy.deepcopy(self.intermediate_population[mother_index].genotype)
                father = copy.deepcopy(self.intermediate_population[father_index].genotype)
                child_genotype_1 = []
                child_genotype_2 = []
                self.__recombine_arrays(child_genotype_1, mother, father)
                self.__recombine_arrays(child_genotype_2, father, mother)
                child_1 = Individual()
                child_1.genotype = child_genotype_1
                children.insert(crossover_aux_index, child_1)
                crossover_aux_index += 1
                child_2 = Individual()
                child_2.genotype = child_genotype_2
                children.insert(crossover_aux_index, child_2)
                crossover_aux_index += 1

            else:

                mother_index = crossover_aux[index]
                father_index = crossover_aux[index + 1]
                child_1 = Individual()
                child_1.genotype = copy.deepcopy(self.intermediate_population[mother_index].genotype)
                children.insert(crossover_aux_index, child_1)
                crossover_aux_index += 1
                child_2 = Individual()
                child_2.genotype = copy.deepcopy(self.intermediate_population[father_index].genotype)
                children.insert(crossover_aux_index, child_2)
                crossover_aux_index += 1

        for index in range(0, len(children)):

            self.intermediate_population[index] = children[index]

    # This method generates the .vxa file containing the best phase offset.
    def get_fittest_individual_and_offest_files(self, experiment_id):

        offset = self.build_offset(self.best_individual)

        catheter = {

            "evaluation": False,
            "layers": self.catheter_reference_list[0],
            "offsets": offset

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

        with open(self.FITTEST_PATH + "offsets_" + str(experiment_id) + ".txt", "w") as file:

            for layer in offset:

                print(layer, file=file)

            file.close()

    # Core routine of the crossover operator.
    def __recombine_arrays(self, child_genotype, receiver, heritage):

        points = random.sample(range(0, len(receiver)), 2)
        points.sort()

        for index in range(0, len(receiver)):

            if points[0] <= index <= points[1]:

                child_genotype.append(heritage[index])

            else:

                child_genotype.append(receiver[index])

    # This method executes the mutation crossover.
    def __execute_mutation(self):

        for individual in self.intermediate_population:

            mutation = random.uniform(0.0, 1.0)

            if mutation <= self.MUTATION_PROBABILITY:

                point = random.sample(range(0, len(individual.genotype)), 1)

                individual.genotype[point[0]] = random.uniform(self.NEGATIVE_MAPPING_REFERENCE, self.POSITIVE_MAPPING_REFERENCE)

    # This method initialises the server and the morphologies used to evaluate phase offsets.
    def __initialise_morphologies_and_server(self, server_configuration, catheters_path):

        self.target_url = server_configuration.get("target_ip_address") + "8000"
        self.number_of_workers = server_configuration.get("simulator_instances")
        initial_port = server_configuration.get("initial_port")

        for _ in range(0, self.number_of_workers):

            initialisation = {

                "voxels": (float(server_configuration.get("layout").get("x")),
                           float(server_configuration.get("layout").get("y")),
                           float(server_configuration.get("layout").get("z")))
            }

            print(initialisation)
            r = requests.post(url=server_configuration.get("target_ip_address") + str(initial_port) + "/biomeld-hn",
                              json=initialisation)
            print(r.json())
            initial_port += 1

        with open(catheters_path, "r") as file:

            catheters_file = file.read()
            catheters_dictionary = json.loads(catheters_file)

        for key in catheters_dictionary.keys():

            catheter_morphology = catheters_dictionary.get(key)
            self.catheter_reference_list.append(catheter_morphology)

        for _ in range(self.number_of_individuals):

            self.list_of_catheter_morphologies.append(self.catheter_reference_list)
