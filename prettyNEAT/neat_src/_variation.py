import sys
#sys.path.append('/content/drive/MyDrive/evojax')

import numpy as np
import itertools
from .ind import Ind
from prettyNEAT.utils import *

import jax.numpy as jnp
import jax


def evolvePop(self, key):
  """ Evolves new population from existing species.
  Wrapper which calls 'recombine' on every species and combines all offspring into a new population. When speciation is not used, the entire population is treated as a single species.
  """
  newpop = []
  for i in range(len(self.species)):
    children, self.innov, key = self.recombine(self.species[i],\
                           self.innov, self.gen, key)
    newpop.append(children)
  self.pop = [indiv for species in newpop for indiv in species] #newpop
  return key
  #self.pop = list(itertools.chain.from_iterable(newPop)) #TODO

def recombine(self, species, innov, gen,key):
  """ Creates next generation of child solutions from a species

  Procedure:
    ) Sort all individuals by rank
    ) Eliminate lower percentage of individuals from breeding pool
    ) Pass upper percentage of individuals to child population unchanged
    ) Select parents by tournament selection
    ) Produce new population through crossover and mutation

  Args:
      species - (Species) -
        .members    - [Ind] - parent population
        .nOffspring - (int) - number of children to produce
      innov   - (np_array)  - innovation record
                [5 X nUniqueGenes]
                [0,:] == Innovation Number
                [1,:] == Source
                [2,:] == Destination
                [3,:] == New Node?
                [4,:] == Generation evolved
      gen     - (int) - current generation

  Returns:
      children - [Ind]      - newly created population
      innov   - (np_array)  - updated innovation record

  """
  p = self.p
  nOffspring = int(species.nOffspring)
  pop = species.members
  children = []

  # Sort by rank
  pop.sort(key=lambda x: x.rank)

  # Cull  - eliminate worst individuals from breeding pool
  numberToCull = int(jnp.floor(p['select_cullRatio'] * len(pop)))
  if numberToCull > 0:
    pop[-numberToCull:] = []

  # Elitism - keep best individuals unchanged
  nElites = int(jnp.floor(len(pop)*p['select_eliteRatio']))
  for i in range(nElites):
    children.append(pop[i])
    nOffspring -= 1

  # Get parent pairs via tournament selection
  # -- As individuals are sorted by fitness, index comparison is
  # enough. In the case of ties the first individual wins


  """
  parentA = np.random.randint(len(pop),size=(nOffspring,p['select_tournSize']))
  parentB = np.random.randint(len(pop),size=(nOffspring,p['select_tournSize']))
  parents = np.vstack( (np.min(parentA,1), np.min(parentB,1) ) )
  parents = np.sort(parents,axis=0) # Higher fitness parent first
  """
  key, subkeyA = jax.random.split(key) #same key will always give the same random numbers
  parentA = jax.random.randint(subkeyA, (nOffspring, p['select_tournSize']), 0, len(pop))
  key, subkeyB = jax.random.split(key)
  parentB = jax.random.randint(subkeyB, (nOffspring, p['select_tournSize']), 0, len(pop))
  min_parentA = jnp.min(parentA, axis=1)
  min_parentB = jnp.min(parentB, axis=1)
  # stack the parents and sort by axis 0 (higher fitness parent first)
  parents = jnp.vstack((min_parentA, min_parentB))
  parents = jnp.sort(parents, axis=0)

  # Breed child population
  for i in range(nOffspring):
    key, subkey = jax.random.split(key)
    if jax.random.uniform(subkey) > p['prob_crossover']:
      # Mutation only: take only highest fit parent
      child, innov,key = pop[parents[0,i]].createChild(p,innov,gen,key=key)
    else:
      # Crossover
      child, innov, key = pop[parents[0,i]].createChild(p,innov,gen,mate=pop[parents[1,i]], key=key)

    child.express()
    children.append(child)

  return children, innov, key
