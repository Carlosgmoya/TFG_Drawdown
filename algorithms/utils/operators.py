import numpy as np
import random
from .projection import projection_simplex

# This module contains functions to perform binary_tournament, crossover, and mutation operations for evolutionary portfolio optimization.

def binary_tournament(population, fitness):
    """
    Perform binary tournament selection.
    
    Parameters:
    - population: The current population.
    - fitness: A 1D array representing the fitness of each individual in the population.
    
    Returns:
    - selected: The selected individual.
    """
    ind1, ind2 = random.sample(range(len(population)), 2)
    return population[ind1] if fitness[ind1] < fitness[ind2] else population[ind2]


def crossover(parent1, parent2, num_assets, cardinality, crossover_rate):
    """
    Perform crossover between two parents to create one child.
    
    Parameters:
    - parent1: First parent (portfolio).
    - parent2: Second parent (portfolio).
    
    Returns:
    - child: Child (portfolio) created from the parents.

    """
    # If crossover is not performed, return a copy of one of the parents
    if random.random() > crossover_rate:
        return np.copy(parent1) if random.random() > 0.5 else np.copy(parent2)

    child = np.zeros(num_assets)
    # Posive indexes of parent1 and parent2
    parent1_indexes = np.where(parent1 > 0)[0]
    parent2_indexes = np.where(parent2 > 0)[0]

    # Find the indexes of the assets that are present in both parents
    equal_indexes = np.intersect1d(parent1_indexes, parent2_indexes)

    # Number of maximum assets in the child
    max_cardinality = len(parent1_indexes) + len(parent2_indexes) - len(equal_indexes)
    if(max_cardinality > cardinality):
        max_cardinality = cardinality

    # Count the number of assets in the child
    asset_count = 0

    # Fill the child with the assets that are present in both parents
    for index in equal_indexes:
        if asset_count >= max_cardinality:
            break
        if random.random() > 0.5:
            child[index] = parent1[index]
        else:
            child[index] = parent2[index]
        asset_count += 1
    
    # Remove the indexes of the assets that are already in the child
    parent1_indexes = np.setdiff1d(parent1_indexes, equal_indexes)
    parent2_indexes = np.setdiff1d(parent2_indexes, equal_indexes)

    # Create lists of indexes for the remaining assets in each parent
    parent1_index_left = list(range(len(parent1_indexes)))
    parent2_index_left = list(range(len(parent2_indexes)))

    # Add the remaining assets from both parents to the child until it reaches the maximum cardinality
    for _ in range(max_cardinality - len(equal_indexes)):
        if asset_count >= max_cardinality:
            break
        
        # If both parents have remaining assets, randomly select one to add to the child
        if len(parent1_index_left) > 0 and len(parent2_index_left) > 0:
            if random.random() > 0.5:
                child[parent1_indexes[parent1_index_left[0]]] = parent1[parent1_indexes[parent1_index_left[0]]]
                parent1_index_left.pop(0)
            else:
                child[parent2_indexes[parent2_index_left[0]]] = parent2[parent2_indexes[parent2_index_left[0]]]
                parent2_index_left.pop(0)
        elif len(parent1_index_left) > 0:
            child[parent1_indexes[parent1_index_left[0]]] = parent1[parent1_indexes[parent1_index_left[0]]]
            parent1_index_left.pop(0)
        else:
            child[parent2_indexes[parent2_index_left[0]]] = parent2[parent2_indexes[parent2_index_left[0]]]
            parent2_index_left.pop(0)
        asset_count += 1

    # Normalize the child to ensure it is a valid portfolio
    child = projection_simplex(child)
    return child


def mutation(individual, mutation_rate):
    """
    Perform mutation on an individual.
    
    Parameters:
    - individual: Individual (portfolio) to mutate.
    
    Returns:
    - mutated_individual: Mutated individual (portfolio).
    
    """
    for i in range(len(individual)):
        if individual[i] > 0 and random.random() < mutation_rate:
            #individual[i] *= random.uniform(0.75, 1.25) # Mutation 25%
            individual[i] += random.normalvariate(0, 0.15)  # Add Gaussian noise with mean 0 and std deviation 0.15
    individual = projection_simplex(individual)
    return individual
