import numpy as np 

def fitness_function(x):
    return x**2+4*x+4

population_size = 50
num_generations = 100
mutation_rate = 0.1

population = np.random.uniform(-10, 10, size=(population_size))

for generation in range(num_generations):
    fitness_values = np.array([fitness_function(x) for x in population])
    parents = np.random.choice(population, size=population_size, p=fitness_values/np.sum(fitness_values))

    children = []
    for _ in range(population_size):
        parent1 = np.random.choice(parents)
        parent2 = np.random.choice(parents)
        child = (parent1 + parent2)/2

        if np.random.rand() < mutation_rate:
            child+=np.random.normal(scale=0.5)
        children.append(child)
    population = np.array(children)

best_solution = population[np.argmin([fitness_function(x) for x in population])]
print("best solution", best_solution)
print("Minimum value", fitness_function(best_solution))