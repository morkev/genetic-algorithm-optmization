import random
import sys
import operator
import matplotlib.pyplot as plt

class Knapsack(object):
    
    # Initialize lists and variables
    def __init__(self):
        self.C = 0  # Knapsack capacity
        self.weights = []  # List of weights for items
        self.profits = []  # List of profits for items
        self.opt = []  # Optimal solution (0 or 1) for each item
        self.parents = []  # List of parent solutions
        self.newparents = []  # List of new parent solutions after crossover and mutation
        self.bests = []  # List of (fitness, solution) tuples
        self.best_p = []  # List of best parent solutions
        self.iterated = 1  # Number of generations iterated
        self.population = 0  # Population size
        self.best_fitness_values = []  # Track the best fitness value in each generation

        # Increase max recursion for long stack
        increaseMaxStackSize = 15000
        sys.setrecursionlimit(increaseMaxStackSize)

    # Creates the initial population
    def initialize(self):
        # Generates random parent solutions
        for i in range(self.population):
            parent = []
            for k in range(0, 5):
                k = random.randint(0, 1)
                parent.append(k)
            self.parents.append(parent)

    # Sets the problem-specific details
    def properties(self, weights, profits, opt, C, population):
        self.weights = weights
        self.profits = profits
        self.opt = opt
        self.C = C
        self.population = population
        self.initialize()

    # Calculates the fitness function of each list (sack)
    def fitness(self, item):
        sum_w = 0 # Sum of weights in the knapsack
        sum_p = 0 # Sum of profits in the knapsack

	# Calculates weights and profits of the selected items
        for index, i in enumerate(item):
            if i == 0:
                continue
            else:
                sum_w += self.weights[index]
                sum_p += self.profits[index]
                
	# Checks if the total weight exceeds the knapsack capacity
        # Returns -1 if it does, otherwise it returns the total profit
        if sum_w > self.C:
            return -1
        else: 
            return sum_p

    # Runs generations with the Genetic Algorithm
    def evaluation(self):
        best_pop = self.population // 2  # Number of best parent solutions to keep
        
        # Loops through parents and calculates fitness for each
        for i in range(len(self.parents)):
            parent = self.parents[i]
            ft = self.fitness(parent)
            self.bests.append((ft, parent))

	# Sorts the fitness list by fitness in descending order
        self.bests.sort(key=operator.itemgetter(0), reverse=True)
        self.best_p = self.bests[:best_pop]
        self.best_p = [x[1] for x in self.best_p]
        # Important to keep this line for the matplotlib graph
        self.best_fitness_values.append(self.bests[0][0])  # Store the best fitness value

    # Mutates children after a certain condition
    def mutation(self, ch):
        for i in range(len(ch)):        
            k = random.uniform(0, 1)
            if k > 0.5:
                # If a random float number is greater than 0.5, flip 0 with 1 and vice versa
                if ch[i] == 1:
                    ch[i] = 0
                else: 
                    ch[i] = 1
        return ch

    # Crossover two parents to produce two children by mixing them under random ratio each time
    def crossover(self, ch1, ch2):
        threshold = random.randint(1, len(ch1) - 1)  # Random threshold for crossover point
        tmp1 = ch1[threshold:]  # Second part of ch1 after the threshold
        tmp2 = ch2[threshold:]  # Second part of ch2 after the threshold
        ch1 = ch1[:threshold]  # First part of ch1 until the threshold
        ch2 = ch2[:threshold]  # First part of ch2 until the threshold
        ch1.extend(tmp2)  # Combine first part of ch1 with second part of ch2
        ch2.extend(tmp1)  # Combine first part of ch2 with second part of ch1
        return ch1, ch2

    # Runs the genetic algorithm
    def run(self):
        # Runs the evaluation once
        self.evaluation()
        newparents = []
        pop = len(self.best_p) - 1

	# Creates a list with unique random integers
        sample = random.sample(range(pop), pop)
        for i in range(0, pop):
            # Selects random indices of best children to randomize the process
            if i < pop - 1:
                r1 = self.best_p[i]
                r2 = self.best_p[i + 1]
                nchild1, nchild2 = self.crossover(r1, r2)
                newparents.append(nchild1)
                newparents.append(nchild2)
            else:
                r1 = self.best_p[i]
                r2 = self.best_p[0]
                nchild1, nchild2 = self.crossover(r1, r2)
                newparents.append(nchild1)
                newparents.append(nchild2)

  	# Mutate the new children and potential parents to ensure global optima found
        for i in range(len(newparents)):
            newparents[i] = self.mutation(newparents[i])

	# Check if the optimal solution is found
        if self.opt in newparents:
            print("Optimal solution found in generation: {}".format(self.iterated))
        else:
            self.iterated += 1
            print("Recreating generation. Attempt number: {}".format(self.iterated))
            self.parents = newparents
            self.bests = []
            self.best_p = []
            self.run()

    # Plots the best fitness value in each generation
    def plot_fitness_values(self):
        plt.plot(range(1, len(self.best_fitness_values) + 1), self.best_fitness_values)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness Value')
        plt.title('Knapsack Problem - Genetic Algorithm')
        plt.show()

    # Runs the algorithm and plots the fitness values
    def run_with_plot(self):
        self.run()
        self.plot_fitness_values()


# properties for this particular problem
weights = [12, 7, 11, 8, 9]
profits = [24, 13, 23, 15, 16]
opt = [0, 1, 1, 1, 0]
C = 26
population = 10

k = Knapsack()
k.properties(weights, profits, opt, C, population)
k.run_with_plot()
