# Libraries needed.
import random
import math


# This class represents an individual of the GA. It has the fitness value and the genotype (the phase offset)
# as attributes.
class Individual:

    # Constant values required to initialise the individual.
    X_LENGTH = 20
    Y_LENGTH = Z_LENGTH = 8

    POSITIVE_MAPPING_REFERENCE = math.pi * 2
    NEGATIVE_MAPPING_REFERENCE = POSITIVE_MAPPING_REFERENCE * -1.0

    # Constructor
    def __init__(self):

        self._fitness = 0.0
        self._genotype = []

    @property
    def fitness(self):

        return self._fitness

    @fitness.setter
    def fitness(self, value):

        self._fitness = value

    @property
    def genotype(self):

        return self._genotype

    @genotype.setter
    def genotype(self, gen):

        self._genotype = gen

    # This method initialises the genotype of individuals at random.
    def initialise(self):

        for _ in range(self.X_LENGTH * self.Y_LENGTH * self.Z_LENGTH):

            self._genotype.append(random.uniform(self.NEGATIVE_MAPPING_REFERENCE, self.POSITIVE_MAPPING_REFERENCE))
