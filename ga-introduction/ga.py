    # GA Introduction
    # Copyright (C) 2026  Richard Qin

    # This program is free software: you can redistribute it and/or modify
    # it under the terms of the GNU General Public License as published by
    # the Free Software Foundation, either version 3 of the License, or
    # (at your option) any later version.

    # This program is distributed in the hope that it will be useful,
    # but WITHOUT ANY WARRANTY; without even the implied warranty of
    # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    # GNU General Public License for more details.

    # You should have received a copy of the GNU General Public License
    # along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np

# 1. Hyperparameters
POP_SIZE = 100        # Population size N (each generation has 100 individuals)
DNA_SIZE = 20         # Gene length L (binary encoding length, determines the solution accuracy)
CROSS_RATE = 0.8      # Crossover probability p_c
MUTATION_RATE = 0.01  # Mutation probability p_m
GENERATIONS = 200     # Iteration number (number of steps t of the Markov chain)
X_BOUND = [-1, 2]     # Search space: the domain of the independent variable x

# 2. Define the function and fitness
def F(x):
    """Target function: x * sin(10 * pi * x) + 2.0"""
    return x * np.sin(10 * np.pi * x) + 2.0

def translateDNA(pop):
    """
    Decode the binary sequence to the real interval [-1, 2]
    """
    # Convert binary matrix to decimal integer
    decimal_val = pop.dot(2 ** np.arange(DNA_SIZE)[::-1])
    # Linear scaling to the specified interval
    return decimal_val / float(2**DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]

def get_fitness(pred):
    """
    Fitness function: ensure that the fitness is strictly positive to meet the probability axiom of roulette wheel selection.
    """
    # Subtract the minimum value and add a small value 1e-3 to prevent 0 fitness
    return pred - np.min(pred) + 1e-3

# 3. Define the genetic operators
def select(pop, fitness):
    """Selection operator S: Roulette wheel selection (Proportional Selection)"""
    # Calculate the probability distribution of being selected based on fitness
    prob = fitness / fitness.sum()
    # Perform N times sampling with replacement according to the probability distribution to generate the mating pool
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=prob)
    return pop[idx]

def crossover(parent, pop_copy):
    """Crossover operator C: Single point crossover"""
    if np.random.rand() < CROSS_RATE:
        # Randomly select a mate from the population
        i_ = np.random.randint(0, POP_SIZE, size=1)
        # Generate a random crossover mask (True/False array)
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(bool)
        # Swap genes (corresponding to the pattern destruction and recombination in the pattern theorem)
        parent[cross_points] = pop_copy[i_, cross_points]
    return parent

def mutate(child):
    """Mutation operator M: Basic bit mutation"""
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            # State flip: 1 becomes 0, 0 becomes 1 (corresponding to the probability walk of Hamming distance)
            child[point] = 1 if child[point] == 0 else 0
    return child

# 4. Main program (Markov chain iteration)
def main():
    # Initialize the 0th generation population P(0): generate a 0/1 random matrix of POP_SIZE x DNA_SIZE
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))

    for step in range(GENERATIONS):
        # 1. Calculate the phenotype and fitness of the current population
        x_values = translateDNA(pop)
        fitness = get_fitness(F(x_values))

        # Record and execute elite retention
        best_idx = np.argmax(fitness)
        elite_dna = pop[best_idx].copy()     # Backup the best genes of the current generation
        best_x = x_values[best_idx]
        best_f = F(best_x)

        if step % 40 == 0 or step == GENERATIONS - 1:
            print(f"Generation {step:3d}: Best x = {best_x:.5f}, Max f(x) = {best_f:.5f}")

        # 2. Selection
        pop = select(pop, fitness)

        # 3. Crossover & Mutation
        pop_copy = pop.copy()
        for i in range(POP_SIZE):
            child = crossover(pop[i], pop_copy)
            child = mutate(child)
            pop[i] = child

        new_x_values = translateDNA(pop)
        new_fitness = get_fitness(F(new_x_values))
        worst_idx = np.argmin(new_fitness)
        pop[worst_idx] = elite_dna

    print("\n[Evolution Complete]")
    print(f"Global Optimum Found: x = {best_x:.5f}, f(x) = {best_f:.5f}")

if __name__ == "__main__":
    main()