# All necessary libraries and imports from other files.
from client.ActivationFunctionBank import ActivationFunctionBank
from client.neat.NEATEvaluator import NEATEvaluator
from configupdater import ConfigUpdater
import pickle
import neat


# This is the main class. It orchestrates the execution of NEAT. It receives the path of the NEAT configuration file.
# It also receives the configuration of: CPPNs, the GA, the fitness function, the server data.
class ControllerNEAT:

    FITTEST_PATH = "fittest/"

    # Constructor
    def __init__(self, path_data, cppn_data, ga_data, fitness_data, server_data):

        self.configuration_file_path = path_data.get("configuration_path")
        self.function_bank = ActivationFunctionBank()
        self.number_of_generations = ga_data.get("number_of_generations")
        self.fitness_function = None
        self.preview = ga_data.get("preview")

        self.__configure_file(cppn_data, ga_data, fitness_data)
        self.__configure_fitness_function(path_data, fitness_data, server_data, cppn_data,
                                          ga_data.get("number_of_individuals"))

    # This is the main function; it triggers the mechanism of NEAT.
    def execute_algorithm(self, experiment_id):

        configuration = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                    neat.DefaultStagnation, self.configuration_file_path)

        configuration.genome_config.add_activation("neg_abs", self.function_bank.negative_abs)
        configuration.genome_config.add_activation("neg_square", self.function_bank.negative_square)
        configuration.genome_config.add_activation("sqrt_abs", self.function_bank.square_abs)
        configuration.genome_config.add_activation("neg_sqrt_abs", self.function_bank.negative_square_abs)
        configuration.genome_config.add_activation("neg_sin", self.function_bank.negative_sin)

        population = neat.Population(configuration)
        reporter = neat.StdOutReporter(True)
        stats = neat.StatisticsReporter()
        population.add_reporter(reporter)
        population.add_reporter(stats)
        fittest = population.run(self.fitness_function.evaluate_individuals, self.number_of_generations)

        print("*-" * 97)
        print("Best individual:")
        print("{!s}".format(fittest))
        print("%s nodes with %s connections." % (fittest.size()[0], fittest.size()[1]))
        print("*-" * 97)

        if self.fitness_function.is_recurrent:

            fittest_cppn = neat.nn.RecurrentNetwork.create(fittest, configuration)

        else:

            fittest_cppn = neat.nn.FeedForwardNetwork.create(fittest, configuration)

        self.fitness_function.get_fittest_individual_file(fittest_cppn, experiment_id)

        with open(self.FITTEST_PATH + "fittest_cppn_" + str(experiment_id) + ".pickle", "wb") as file:

            pickle.dump(fittest_cppn, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.close()

    # This function configures the input file for NEAT algorithm.
    def __configure_file(self, cppn_data, ga_data, fitness_data):

        updater = ConfigUpdater()
        updater.read(self.configuration_file_path)
        updater["NEAT"]["pop_size"].value = ga_data.get("number_of_individuals")
        updater["NEAT"]["fitness_threshold"].value = fitness_data.get("threshold")
        updater["DefaultReproduction"]["elitism"].value = ga_data.get("individuals_in_elitism")
        updater["DefaultGenome"]["num_inputs"].value = cppn_data.get("input_neurons")
        updater["DefaultGenome"]["num_outputs"].value = cppn_data.get("output_neurons")
        updater["DefaultReproduction"]["survival_threshold"].value = ga_data.get("reproduction_ratio")
        updater.update_file()

    # This function initialises the fitness function.
    def __configure_fitness_function(self, path_data, fitness_data, server_data, cppn_data, number_of_individuals):

        catheter_path = path_data.get("catheters_path")
        self.fitness_function = NEATEvaluator(catheter_path, fitness_data, server_data, cppn_data, number_of_individuals)


