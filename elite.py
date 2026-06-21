import numpy as np

def elitism(nElite, population, fitness):
    order = np.argsort(fitness[:, 0])[::-1]
    elites = population[order[:nElite], :]

    return elites
