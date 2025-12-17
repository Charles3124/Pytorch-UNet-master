import numpy as np
import random
import time
from traintest import testFunction

def GA(MaxIter=10, PopSize=10, Bits=24, K=5, CrossoverRate=0.5, MutationRate=0.1):
    # Initialize population
    pop = np.random.randint(0, 2, (PopSize, Bits))
    fitness = testFunction(pop)  # Evaluate fitness

    # Record the best individual and its fitness
    globalBest = {'individual': None, 'fitness': np.inf}

    # Create a file to save the best solution and runtime
    output_file = "GA_results.txt"
    with open(output_file, "w") as file:
        file.write("GA optimization process results:\n")

    start_time = time.time()  # Record start time

    for gen in range(MaxIter):
        # Select the best K individuals as parents
        parents_idx = np.argsort(fitness)[:K]
        parents = pop[parents_idx]

        # Perform crossover to generate offspring
        offspring = []
        for _ in range(PopSize - K):
            # Randomly choose two parents
            p1, p2 = random.sample(list(parents), 2)
            child = np.zeros(Bits, dtype=int)
            for i in range(Bits):
                if random.random() < CrossoverRate:
                    child[i] = p1[i]
                else:
                    child[i] = p2[i]
            offspring.append(child)

        # Perform mutation on offspring
        offspring = np.array(offspring)
        for i in range(offspring.shape[0]):
            for j in range(Bits):
                if random.random() < MutationRate:
                    offspring[i][j] = 1 - offspring[i][j]

        # Combine parents and offspring to form the new population
        pop = np.vstack((parents, offspring))

        # Evaluate fitness of the new population
        fitness = testFunction(pop)

        # Update global best solution
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < globalBest['fitness']:
            globalBest['fitness'] = fitness[best_idx]
            globalBest['individual'] = pop[best_idx].copy()

            # Decode parameters
            parameter = [
                globalBest['SKD'][0:2],  # Blocks_number
                globalBest['SKD'][2:4],  # Filters_number
                # globalBest['SKD'][4:6],  #Filter_size
                globalBest['SKD'][4:6],  # 激活函数ReLU, ELU and LeakyReLU random
                globalBest['SKD'][6],  # 池化层 max mean
                globalBest['SKD'][7:9],  # 优化器  SGD , RMSprop, Adam,and Adamax
                globalBest['SKD'][9:11],  # 批次大小 8，16，32，64
                globalBest['SKD'][11:21],  # 学习率 [10^-6,10^-3]
                globalBest['SKD'][21],
                globalBest['SKD'][22:24]
            ]

            # Save parameters and fitness to file
            with open(output_file, "a") as file:
                file.write(f"Generation {gen + 1}:\n")
                file.write(f"Parameters: {parameter}\n")
                file.write(f"Fitness: {globalBest['fitness']}\n\n")

        print(f"Generation {gen}: Best fitness = {globalBest['fitness']}")

    # Record total runtime
    end_time = time.time()
    total_time = end_time - start_time
    with open(output_file, "a") as file:
        file.write(f"Total runtime: {total_time:.2f} seconds\n")

    # Return the best parameters
    return parameter

if __name__ == '__main__':
    best_params = GA()
    print("Best hyperparameters found:")
    print(best_params)
