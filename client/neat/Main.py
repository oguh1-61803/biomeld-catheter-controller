# All necessary libraries and imports from other files.
from client.neat.ControllerNEAT import ControllerNEAT
import time


if __name__ == '__main__':

    # If needed, don't forget to update the path of the files!
    file_paths = {

        "configuration_path": "controller_NEAT.cfg",
        "catheters_path": "../catheters/catheter_2.json"
    }

    # Configuration of the CPPNs.
    cppn_data = {

        "input_neurons": 4,
        "output_neurons": 1,
        "recurrent_topology": False
    }

    # Configuration of the GA.
    ga_data = {

        "number_of_individuals": 50,
        "number_of_generations": 200,
        "reproduction_ratio": 0.2,
        "individuals_in_elitism": 1
    }

    # Configuration of the fitness function and the voxel layout.
    fitness_metrics = {

        "threshold": 0.5,
        "layout": {"x": 20, "y": 8, "z": 8}
    }

    # Configuration of the server.
    server_configuration = {

        "simulator_instances": 18,
        "initial_port": 8081,
        "target_ip_address": "http://192.168.0.89:"
    }

    for exp_id in range(0, 1):

        neat = ControllerNEAT(file_paths, cppn_data, ga_data, fitness_metrics, server_configuration)
        start = time.time()
        neat.execute_algorithm(exp_id)
        end = time.time()
        print("Elapsed time: " + str(end - start))

        path = "fittest/" + "time_" + str(exp_id) + ".txt"

        with open(path, "w") as file:

            file.write(str(end - start))
            file.close()
