
'''
In this module, I've figured out how to create and individual with my own custom
    function ('create_individual' and 'create_individual_with_max') and how to
    impose a hard limit on the maximum number of ones in a binary vector (which
    is what an individual is)
The hard limit ('max_n_of_ones') is imposed when the individuals are initially
    created (in 'create_individual_with_max' but not in 'create_individual') and
    when fitness is being evaluated (in 'get_fitness_with_hard_max' but not in
    'get_fitness')
I haven't figured out how to pass parameters to these functions (e.g.,
    'create_individual', 'get_fitness_with_hard_max') within the
    'toolbox.register' statements, which forces me to keep variables that are
    shared among functions ('individual_len' and 'max_n_of_ones') as global
    variables
NEXT STEP:  put a 'soft max' limit on the number of ones in the binary vector,
    instead of a 'hard limit'; this could probably be done within DEAP, because
    it allows multi-objective optimization, but it's probably easier to just do
    it in the function that calculates fitness ('get_fitness' or
    'get_fitness_with_hard_max')
'''


import numpy as np
from deap import creator, base, tools, algorithms

individual_len = 20
fit_vector = np.random.normal(size=individual_len)
max_n_of_ones = 4


def create_individual():
    """
    Creates list of binary elements (zeroes and ones) of length 'individual_len'
    The number of ones is probabilistically sampled from a binomial distribution
    The list must have a minimum of one one, i.e., it can not contain only
        zeroes

    :return: binary vector as a list of length 'individual_len'
    """

    #import numpy as np

    #individual_len = 10
    individual = [np.random.binomial(1, 0.2) for i in range(individual_len)]
    if sum(individual) < 1:
        idx = np.random.randint(individual_len)
        individual[idx] = 1
    return individual


def create_individual_with_max():
    """
    Creates list of binary elements (zeroes and ones) of length 'individual_len'
    The number of ones in the list can not exceed 'max_n_of_ones'

    :return: binary vector as a list of length 'individual_len'
    """

    #import numpy as np

    #individual_len = 10
    #max_n_of_ones = 3
    assert (max_n_of_ones < individual_len)

    # allow fewer than 'max_n_of_ones' ones in individual, but with a high
    #   probability that the number of ones will be close to 'max_n_of_ones'
    idx_n = max_n_of_ones + 1
    while (idx_n > max_n_of_ones) or (idx_n == 0):
        idx_n = np.random.poisson(max_n_of_ones)

    ones_idx = np.random.choice(range(individual_len), size=idx_n, replace=False)

    individual = np.zeros(individual_len, dtype='int')
    individual[ones_idx] = 1

    return individual.tolist()


def get_fitness(individual, fit_vector):
    return (np.array(individual).dot(fit_vector),)


def get_fitness_with_hard_max(individual, fit_vector):
    # if number of ones in binary vector exceeds maximum allowed
    #   ('max_n_of_ones'), fitness = 0
    if sum(individual) > max_n_of_ones:
        fitness = 0
    else:
        fitness = np.array(individual).dot(fit_vector)
    return (fitness,)


creator.create('Fitness', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.Fitness)

toolbox = base.Toolbox()
#toolbox.register(
#   'individual', tools.initIterate, creator.Individual, create_individual)
toolbox.register(
    'individual', tools.initIterate, creator.Individual,
    create_individual_with_max)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

#toolbox.register('evaluate', get_fitness, fit_vector=fit_vector)
toolbox.register('evaluate', get_fitness_with_hard_max, fit_vector=fit_vector)
toolbox.register('mate', tools.cxOnePoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
toolbox.register('select', tools.selTournament, tournsize=3)

pop = toolbox.population(n=55)
hof = tools.HallOfFame(5)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register('avg', np.mean)
stats.register('std', np.std)
stats.register('min', np.min)
stats.register('max', np.max)

pop, log = algorithms.eaSimple(
    pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=90, stats=stats,
    halloffame=hof, verbose=True)





