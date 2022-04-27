import random as rd
import numpy as np
from data import Data

data = Data.generated_with(N=6, K=2, seed=42)
CROSSOVER = 0.4
MUTATION = 0.015


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
        int_part = rd.randint(
            1, data.K)  # Determines the index of the technician
        frac_part = rd.random()  # Determines which customer assigned to the technician
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
            print(prob)

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


def main():
    cnt1 = 0
    cnt2 = 0
    for _ in range(1000000):
        ind1 = Individual.create_gnome()
        ind2 = Individual.create_gnome()
        child = ind1.mate(ind2)
        if child.fitness <= (ind1.fitness + ind2.fitness)/2:
            cnt1 += 1
        if child.fitness <= ind1.fitness or child.fitness <= ind2.fitness:
            cnt2 += 1

    print(f'Better than parents\' average fitness: {cnt1/1000000}')
    print(f'Better than one of parents\' fitness: {cnt2/1000000}')


if __name__ == '__main__':
    main()
