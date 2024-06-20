# All necessary libraries and imports from other files.
from client.ga.Population import Population
import time

if __name__ == '__main__':

    # Required parameters for the GA.
    number_of_individuals = 50
    number_of_generations = 15

    # Configuration of the server.
    server_configuration = {

        # This number should be the same as the number of Voxelyze instances deployed in the server side.
        "simulator_instances": 18,
        "initial_port": 8081,
        "layout": {"x": 20, "y": 8, "z": 8},
        "target_ip_address": "http://192.168.0.89:"  # This is the server's ip.
    }

    # If needed, don't forget to update the path of the configuration file!
    catheters_path = "../catheters/catheter_2.json"

    for exp_id in range(0, 1):

        start_time = time.time()
        generation_counter = 0

        population = Population(number_of_individuals, server_configuration, catheters_path)
        population.create_population()
        population.evaluate_population()
        population.get_best_individual()

        while generation_counter < number_of_generations:

            population.execute_tournament()
            population.execute_genetic_operators()
            population.build_new_population()
            population.evaluate_population()
            population.get_best_individual()
            generation_counter += 1

        end_time = time.time()
        print("Elapsed time: " + str(end_time - start_time))
        path = "fittest/time_" + str(exp_id) + ".txt"

        with open(path, "w") as file:

            file.write(str(end_time - start_time))
            file.close()

        population.get_fittest_individual_and_offest_files(exp_id)
