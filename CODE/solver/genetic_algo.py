import random as rd
import numpy as np
from .heuristic_greedy import heuristic_greedy


def genetic_algo(
        data,
        iter=250,
        population_size=100,
        mutation_rate=0.05,
        crossover_rate=0.6,
        elitism=True,
        nrep=20,
        nmut=2,
        dynamic_nmut=False,
        sampling='50',
        initial_solution=False,
        silent=True,
        return_res=False):

    POPULATION_SIZE = population_size
    ITERATIONS = iter
    CROSSOVER = crossover_rate
    MUTATION = mutation_rate
    ELITISM = elitism
    NREP = nrep
    NMUT = nmut
    DYNAMIC = dynamic_nmut
    SAMPLING = sampling
    INITIAL_SOLUTION = initial_solution

    class Individual:
        '''
        Class repesenting individual in population
        '''

        def __init__(self, chromosome):
            self.chromosome = chromosome
            self.fitness = self.cal_fitness()

        @classmethod
        def mutated_genes(self):
            '''
            create random genes for mutation
            '''
            int_part = rd.randint(1, data.K)
            frac_part = rd.random()
            return int_part + frac_part

        @classmethod
        def create_gnome(cls):
            '''
            create chromosome or string of genes
            '''
            return cls([cls.mutated_genes() for _ in range(data.N)])

        @classmethod
        def from_routes(cls, routes):
            genes = []
            r = []
            for k in routes:
                int_part = k
                fracs = sorted([rd.random() for _ in routes[k][1:-1]])
                genes += [int_part + frac for frac in fracs]
                r += routes[k][1:-1]

            chromosome = [None for _ in range(data.N)]
            for i, j in enumerate(np.argsort(r)):
                chromosome[i] = genes[j]

            return cls(chromosome)

        def mate(self, other):
            '''
            Perform mating and produce new offspring
            '''
            child_chromosome = []
            for gp1, gp2 in zip(self.chromosome, other.chromosome):

                # random probability
                prob = rd.random()

                if prob < CROSSOVER:
                    child_chromosome.append(gp1)

                else:
                    child_chromosome.append(gp2)

            #  random gene(mutate) for maintaining diversity
            for i in range(len(child_chromosome)):
                if rd.random() < MUTATION:
                    child_chromosome[i] = self.mutated_genes()

            # create new Individual(offspring) using
            # generated chromosome for offspring
            return Individual(child_chromosome)

        def mates(self, other1, other2):
            '''
            Perform mating and produce new offspring
            '''
            child_chromosome = []
            for gp1, gp2, gp3 in zip(self.chromosome, other1.chromosome, other2.chromosome):

                # random probability
                prob = rd.random()

                if prob < CROSSOVER:
                    child_chromosome.append(gp1)

                elif prob < 0.95:
                    child_chromosome.append(gp2)

                else:
                    child_chromosome.append(gp3)

            #  random gene(mutate) for maintaining diversity
            for i in range(len(child_chromosome)):
                if rd.random() < MUTATION:
                    child_chromosome[i] = self.mutated_genes()

            # create new Individual(offspring) using
            # generated chromosome for offspring
            return Individual(child_chromosome)

        def __sub__(self, other):
            return abs(sum(self.chromosome[i] - other.chromosome[i] for i in range(data.N)))

        def to_routes(self):
            '''
            convert individual to routes
            '''
            routes = {k: [0] for k in range(1, data.K+1)}

            for i in np.argsort(self.chromosome):
                routes[int(self.chromosome[i])].append(i+1)

            for k in routes:
                routes[k].append(0)

            return routes

        def cal_fitness(self):
            '''
            calculate fitness of individual
            '''
            routes = self.to_routes()
            working_time = []
            for k in routes:
                fix_time = sum(data.d[e-1] if e != 0 else 0 for e in routes[k])
                travel_time = sum(data.t[i][j]
                                  for i, j in zip(routes[k], routes[k][1:]))
                working_time.append(fix_time + travel_time)

            return max(working_time)

    def create_population(pop_size, old_generation, method):
        offsprings = []
        if method == 'random':
            for _ in range(pop_size):
                parent1, parent2 = rd.sample(old_generation, 2)
                child = parent1.mate(parent2)
                offsprings.append(child)

        elif method == '50':
            for _ in range(pop_size):
                parent1, parent2 = rd.sample(
                    old_generation[:POPULATION_SIZE//2], 2)
                child = parent1.mate(parent2)
                offsprings.append(child)

        elif method == 'tournament':
            for _ in range(pop_size):
                chosen = rd.sample(old_generation, 2)
                parent1 = chosen[0] if chosen[0].fitness < chosen[1].fitness else chosen[1]
                chosen = rd.sample(old_generation, 2)
                parent2 = chosen[0] if chosen[0].fitness < chosen[1].fitness else chosen[1]
                child = parent1.mate(parent2)
                offsprings.append(child)

        elif method == 'roulette':
            fitnesses = [ind.fitness for ind in old_generation]
            total_fitness = sum(fitnesses)
            for _ in range(pop_size):
                chosen = rd.random() * total_fitness
                current = 0
                for i, ind in enumerate(old_generation):
                    current += ind.fitness
                    if current > chosen:
                        parent1 = old_generation[i]
                        break

                chosen = rd.random() * total_fitness
                current = 0
                for i, ind in enumerate(old_generation):
                    current += ind.fitness
                    if current > chosen:
                        parent2 = old_generation[i]
                        break

                child = parent1.mate(parent2)
                offsprings.append(child)

        elif method == 'rank-based':
            pc = 0.3
            weights = [pc * (1 - pc)**i for i in range(len(old_generation)-1)
                       ] + [(1 - pc)**(len(old_generation)-1)]
            for _ in range(pop_size):
                chosen = rd.choices(old_generation, weights=weights, k=2)
                child = chosen[0].mate(chosen[1])
                offsprings.append(child)

        elif method == 'fit-diversity':
            for _ in range(pop_size):
                parent1 = rd.choice(old_generation[:50])
                diversity_weights = [idi - parent1 for idi in old_generation]
                parent2 = rd.choices(old_generation, diversity_weights)[0]
                child = parent1.mate(parent2)
                offsprings.append(child)

        return offsprings

    def main():

        generation = 1
        population = []
        records = {'best_fit': [], 'avg_fit': [], 'worst_fit': []}

        if INITIAL_SOLUTION:
            routes = heuristic_greedy(data, return_routes=True)
            gnome = Individual.from_routes(routes)
            population.append(gnome)
            print(
                f'Initial solution with fitness {gnome.fitness} is inserted into the population')
            for _ in range(POPULATION_SIZE-1):
                gnome = Individual.create_gnome()
                population.append(gnome)
        else:
            # create initial population
            for _ in range(POPULATION_SIZE):
                gnome = Individual.create_gnome()
                population.append(gnome)

        for generation in range(ITERATIONS):

            # sort the population in increasing order of fitness score
            population = sorted(population, key=lambda x: x.fitness)

            avg_fit = sum(ind.fitness for ind in population) / POPULATION_SIZE
            records['avg_fit'].append(avg_fit)
            records['best_fit'].append(population[0].fitness)
            records['worst_fit'].append(population[-1].fitness)

            if not silent:
                print(f'Generation: {generation}')
                print(f'Average fitness: {avg_fit}')
                print(f'Top 3: {[ind.to_routes() for ind in population[:3]]}')

            new_generation = []

            if ELITISM:
                # Perform Elitism, that mean some of fittest population
                # goes to the next generation
                s = NREP
                new_generation.extend(population[:s])
            else:
                s = 0

            # Some mutated gnomes is added in the population
            if DYNAMIC:
                nonlocal NMUT
                if generation % 50 == 0:
                    NMUT *= 2

            for _ in range(NMUT):
                gnome = Individual.create_gnome()
                new_generation.append(gnome)

            # The rest of the new generation is created through mating
            s = POPULATION_SIZE - s - NMUT
            new_generation += create_population(s, population, SAMPLING)

            population = new_generation

        routes = population[0].to_routes()
        if not return_res:
            print(f'\nBest individual is {population[0].fitness}')
            for k in range(data.K):
                route_k = routes[k+1]
                journey = f'Route[{k}] = ' + \
                    ' -> '.join(str(e) for e in route_k)
                fix_time = f'fix time = {sum(data.d[e-1] if e != 0 else 0 for e in route_k)}'
                travel_time = f'travel time = {sum(data.t[i][j] for i, j in zip(route_k, route_k[1:]))}'
                print(f'{journey} | {fix_time} | {travel_time}')
                for i, j in zip(route_k, route_k[1:]):
                    s1 = f'{i} -> {j}'
                    s2 = f'travel time = {data.t[i][j]}'
                    s3 = f'fix time = {data.d[j-1]}' if j != 0 else 'fix time = 0'
                    print(f'{s1} | {s2} | {s3}')

        return records

    return main()
