from random import randint
from random import random
from functools import reduce
from operator import add
from operator import mul

def individual(length, min, max):
  """ Creates an individual for a population
  :param length: the number of values in the list
  :param min: the minimum value in the list of values
  :param max: the maximal value in the list of values
  :return: """
  return [ randint(min, max) for x in range(length) ]


def population(count, length, min , max):
  """ Create a number of individuals (i.e., a population).
  :param count: the desired size of the population
  :param length: the number of values per individual
  :param min: the minimum in the individual’s values
  :param max: the maximal in the individual’s values """
  return [individual(length, min, max) for x in range(count) ]

def fitness(individual, target_sum, target_mult):
  """ Determine the fitness of an individual. Lower is better.
  :param individual: the individual to evaluate 
  :param target_sum: the sum that we are aiming for
  :param target_mult: the mult that we ar aiming for
  """
  sum = abs(reduce(add, [i + 1 for i, pipe in enumerate(individual) if pipe == 0], 0) - 36)
  mult = abs(reduce(mul, [i + 1 for i, pipe in enumerate(individual) if pipe == 1], 1) - 360)
  return abs(target_sum - sum) + abs(target_mult - mult)


def grade(population, target_sum, target_mult):
  """ Find average fitness for a population
  :param population: population to evaluate 
  :param target: the value that we are aiming for (X) 
  """
  summed = reduce(add, (fitness(x, target_sum, target_mult) for x in population), 0)
  return summed / len(population)


def evolve(population, target_sum, target_mult, retain=0.2, random_select=0.05, mutate=0.009):
  """ Function for evolving a population, that is, creating offspring (next generation population) from combining (crossover) the fittest individuals of the current population
  :param population: the current population
  :param target_sum: the sum value that we are aiming for
  :param target_mult: the multi value that we are aiming for
  :param retain: the portion of the population that we allow to spawn offspring
  :param random_select: the portion of individuals that are selected at random, not based on their score
  :param mutate: the amount of random change we apply to new offspring
  :return: next generation population """
  graded = [ (fitness(x, target_sum, target_mult), x) for x in population ]
  graded = [ x[1] for x in sorted(graded) ]
  retain_length = int(len(graded) * retain)
  parents = graded[:retain_length]
  # randomly add other individuals to promote genetic # diversity
  for individual in graded[retain_length:]:
    if random_select > random():
      parents.append(individual)
  # crossover parents to create offspring
  desired_length = len(population) - len(parents)
  children = []
  while len(children) < desired_length:
    male = randint(0, len(parents)-1)
    female = randint(0, len(parents)-1)
    if male != female:
      male = parents[male]
      female = parents[female]
      half = int(len(male) / 2)
      child = male[:half] + female[half:]
      children.append(child)
  # mutate some individuals for individual in children:
  for individual in children:
    if mutate > random():
      pos_to_mutate = randint(0, len(individual)-1)
      # this mutation is not ideal, because it
      # restricts the range of possible values,
      # but the function is unaware of the min/max
      # values used to create the individuals
      individual[pos_to_mutate] = randint(min(individual), max(individual))
  parents.extend(children)
  return parents

target_pipe0 = 36 # total sum
target_pipe1 = 360 #total multiplied

p_count = 450 # number of individuals in population
i_length = 10 # N
i_min = 0 # value range for generating individuals
i_max = 1


results = []
for test in range(50):
  p = population(p_count, i_length, i_min, i_max)
  for i in range(100): # we stop after 100 generations
    p = evolve(p, target_pipe0, target_pipe1)
    score = grade(p, target_pipe0, target_pipe1)
    print(score, end=", ")
    if score == 0.0:
      print("yeay gevonden in ", i + 1, " aantal keer ")
      results.append(i + 1)
      break

print(results)
print("gemiddelde ", sum(results) / len(results))



"""
Size of Population

Retain

mutate



"""
