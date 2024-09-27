import numpy as np
import math
import copy
import json

import sys
#sys.path.append('/content/drive/MyDrive/evojax')

#from prettyNEAT.domain import *  # Task environments, line 230ish (dont need, load in hyperparams manually)
from prettyNEAT.utils import *
from prettyNEAT.neat_src.nsga_sort import nsga_sort
from prettyNEAT.neat_src.ind import * #~line160

import jax.numpy as jnp
import jax

from evojax.algo.base import NEAlgorithm



class NeatAlgo(NEAlgorithm): #need ask,tell, best params, state stuff is optional
  """NEAT main class. Evolves population given fitness values of individuals.
  """
  def __init__(self, hyp):
    """Intialize NEAT algorithm with hyperparameters
    Args:
      hyp - (dict) - algorithm hyperparameters

    Attributes:
      p       - (dict)     - algorithm hyperparameters (see p/hypkey.txt)
      pop     - (Ind)      - Current population
      species - (Species)  - Current species
      innov   - (np_array) - innovation record
                [5 X nUniqueGenes]
                [0,:] == Innovation Number
                [1,:] == Source
                [2,:] == Destination
                [3,:] == New Node?
                [4,:] == Generation evolved
      gen     - (int)      - Current generation
    """
    self.p       = hyp
    self.pop     = []
    self.species = []
    self.innov   = []
    self.gen     = 0
    self.key = jax.random.PRNGKey(42) #needed for randomizing in jax, must pass in every time
    #initialize here, generates seq of rand number (call prngkey once)
    self.best_weights = None
    self.best_activations = None
    self.best_idx = None
    self.pop_size = self.p['popSize'] #used to be initialized to 0, did not work


  ''' Subfunctions '''
  from prettyNEAT.neat_src._variation import evolvePop, recombine
  from prettyNEAT.neat_src._speciate  import Species, speciate, compatDist,\
                          assignSpecies, assignOffspring

  def ask(self):
    """Returns newly evolved population
    """
    if len(self.pop) == 0:
      self.initPop()      # Initialize population
    else:
      self.probMoo()      # Rank population according to objectivess
      self.speciate()     # Divide population into species


      self.key = self.evolvePop(self.key )    # Create child population ... key is updated inside
        # pass in key, return a key

    #jax only deal with arr of equal sz? need pad or reduce arr
      # try padding and slicing to lowest...
    #need to retun weights, not ind obj
    largest_dim = max(len(indiv.wMat) for indiv in self.pop) #find largest indiv, diff sizes cus evolve
    weights_pad = jnp.zeros((len(self.pop), largest_dim, largest_dim))
    activations_pad = jnp.zeros((len(self.pop), largest_dim))
    for i, ind in enumerate(self.pop):
      #preallocate space for og wmat and avec, rest is padded with 0. this way pop is same size
      weights_pad = weights_pad.at[i, : len(ind.wMat), : len(ind.wMat)].set(ind.wMat)
      activations_pad = activations_pad.at[i, : len(ind.aVec)].set(ind.aVec)

    weightnodes = jnp.array([len(ind.wMat) for ind in self.pop])
    return (weightnodes, weights_pad, activations_pad)
    #return self.pop       # Send child population for evaluation

  def tell(self,reward):
    """Assigns fitness to current population

    Args:
      reward - (np_array) - fitness value of each individual
               [nInd X 1]

    """
    for i in range(len(self.pop)):
      self.pop[i].fitness = reward[i]
      self.pop[i].nConn   = self.pop[i].nConn

    self.best_idx = jnp.argmax(reward) #best idx for weights and activation
    self.best_weights = self.pop[self.best_idx].wMat #set best
    self.best_activations = self.pop[self.best_idx].aVec
    # for i in range(np.shape(reward)[0]):
    #   self.pop[i].fitness = reward[i]
    #   self.pop[i].nConn   = self.pop[i].nConn

  def initPop(self):
    """Initialize population with a list of random individuals
    """
    ##  Create base individual
    p = self.p # readability

    # - Create Nodes -
    nodeId = jnp.arange(0, p['ann_nInput'] + p['ann_nOutput'] + 1)
    node = jnp.empty((3, len(nodeId)))
    node = node.at[0, :].set(nodeId)
    # nodeId = np.arange(0,p['ann_nInput']+ p['ann_nOutput']+1,1)
    # node = np.empty((3,len(nodeId)))
    # node[0,:] = nodeId

    # Node types: [1:input, 2:hidden, 3:bias, 4:output]
    node = node.at[1, 0].set(4)
    node = node.at[1, 1 : p["ann_nInput"] + 1].set(1)
    node = node.at[1, (p["ann_nInput"] + 1) :].set(2)
    # node[1,0]             = 4 # Bias
    # node[1,1:p['ann_nInput']+1] = 1 # Input Nodes
    # node[1,(p['ann_nInput']+1):\
    #        (p['ann_nInput']+p['ann_nOutput']+1)]  = 2 # Output Nodes

    # Node Activations
    node = node.at[2, :].set(p['ann_initAct'])
    # node[2,:] = p['ann_initAct']

    # - Create Conns -
    nConn = (p['ann_nInput'] + 1) * p['ann_nOutput']
    ins = jnp.arange(0, p['ann_nInput'] + 1)
    outs = (p['ann_nInput'] + 1) + jnp.arange(0, p['ann_nOutput'])
    # nConn = (p['ann_nInput']+1) * p['ann_nOutput']
    # ins   = np.arange(0,p['ann_nInput']+1,1)            # Input and Bias Ids
    # outs  = (p['ann_nInput']+1) + np.arange(0,p['ann_nOutput']) # Output Ids

    conn = jnp.empty((5, nConn))
    conn = conn.at[0, :].set(jnp.arange(0, nConn))  # Connection Id
    conn = conn.at[1, :].set(jnp.tile(ins, len(outs)))  # Source Nodes
    conn = conn.at[2, :].set(jnp.repeat(outs, len(ins)))  # Destination Nodes
    conn = conn.at[3, :].set(jnp.nan)  # Weight Values
    conn = conn.at[4, :].set(1) # Enabled?
    # conn = np.empty((5,nConn,))
    # conn[0,:] = np.arange(0,nConn,1)      # Connection Id
    # conn[1,:] = np.tile(ins, len(outs))   # Source Nodes
    # conn[2,:] = np.repeat(outs,len(ins) ) # Destination Nodes
    # conn[3,:] = np.nan                    # Weight Values
    # conn[4,:] = 1                         # Enabled?

    # Create population of individuals with varied weights
    pop = []
    for i in range(p['popSize']):
        newInd = Ind(conn, node)

        self.key, subkey = jax.random.split(self.key)
        newInd.conn = newInd.conn.at[3, :].set((2 * (jax.random.uniform(subkey, (nConn,)) - 0.5)) * p["ann_absWCap"])
        self.key, subkey = jax.random.split(self.key)
        newInd.conn = newInd.conn.at[4, :].set(jax.random.uniform(subkey, (nConn,)) < p["prob_initEnable"])
        newInd.express()
        newInd.birth = 0
        pop.append(newInd)
        # newInd.conn[3,:] = (2*(np.random.rand(1,nConn)-0.5))*p['ann_absWCap']
        # newInd.conn[4,:] = np.random.rand(1,nConn) < p['prob_initEnable']
        # newInd.express()
        # newInd.birth = 0
        # pop.append(copy.deepcopy(newInd))

    # - Create Innovation Record -
    innov = jnp.zeros([5, nConn])
    innov = innov.at[0:3, :].set(pop[0].conn[0:3, :])
    innov = innov.at[3, :].set(-1)
    # innov = np.zeros([5,nConn])
    # innov[0:3,:] = pop[0].conn[0:3,:]
    # innov[3,:] = -1

    self.pop = pop
    self.innov = innov
    #key is updated above when directly used

  def probMoo(self):
    """Rank population according to Pareto dominance.
    """
    # Compile objectives
    meanFit = jnp.array([ind.fitness for ind in self.pop])
    nConns = jnp.array([ind.nConn for ind in self.pop])
    nConns = jnp.where(nConns == 0, 1, nConns)
    objVals = jnp.column_stack([meanFit, 1 / nConns])
    # meanFit = np.asarray([ind.fitness for ind in self.pop])
    # nConns  = np.asarray([ind.nConn   for ind in self.pop])
    # nConns[nConns==0] = 1 # No connections is pareto optimal but boring...
    # objVals = np.c_[meanFit,1/nConns] # Maximize

    # Alternate between two objectives and single objective
    self.key, subkey = jax.random.split(self.key)
    rand_val = jax.random.uniform(subkey)

    # rank = jax.lax.cond(
    #     self.p["alg_probMoo"] < rand_val,
    #     lambda _: nsga_sort(objVals[:, [0, 1]]),
    #     lambda _: rankArray(-objVals[:, 0]),
    #     operand=None  # no input is required for the functions
    # ) #no work

    if self.p['alg_probMoo'] < rand_val:
      rank = nsga_sort(objVals[:,[0,1]])
    else: # Single objective
      rank = rankArray(-objVals[:,0])

    # Assign ranks
    for i in range(len(self.pop)):
      self.pop[i].rank = rank[i]


  #load_state and save_state are optional...
  #stuff needed to match evojax ask tell interface... part of neat class
  @property
  def best_params(self) -> jnp.ndarray:
      return self.best_weights, self.best_activations

  @best_params.setter
  def best_params(self, w, a) -> None:
      self.best_weights = w
      self.best_activations = a

#outside of neat class
def loadHyp(pFileName, printHyp=False):
  """Loads hyperparameters from .json file
  Args:
      pFileName - (string) - file name of hyperparameter file
      printHyp  - (bool)   - print contents of hyperparameter file to terminal?

  Note: see p/hypkey.txt for detailed hyperparameter description
  """
  print("THIS SHOULD NOT PRINT")

  with open(pFileName) as data_file: hyp = json.load(data_file)

  # Task hyper parameters
  task = GymTask(games[hyp['task']],paramOnly=True)

  #all this stuff is defined in domain.config
  hyp['ann_nInput']   = task.nInput
  hyp['ann_nOutput']  = task.nOutput
  hyp['ann_initAct']  = task.activations[0] #first one is just 0
  hyp['ann_absWCap']  = task.absWCap
  hyp['ann_mutSigma'] = task.absWCap * 0.2
  #
  # hyp['ann_layers']   = task.layers # if fixed toplogy is used

  if hyp['alg_act'] == 0:
    hyp['ann_actRange'] = task.actRange
  else:
    hyp['ann_actRange']=  task.actRange #all activations
    #hyp['ann_actRange'] = np.full_like(task.actRange,hyp['alg_act'])

  #new hyperparams are done added
  if printHyp is True:
    print(json.dumps(hyp, indent=4, sort_keys=True))
  return hyp

#not used
def updateHyp(hyp,pFileName=None):
  """Overwrites default hyperparameters with those from second .json file
  """
  if pFileName != None:
    print('\t*** Running with hyperparameters: ', pFileName, '\t***')
    with open(pFileName) as data_file: update = json.load(data_file)
    hyp.update(update)

    # Task hyper parameters
    task = GymTask(games[hyp['task']],paramOnly=True)
    hyp['ann_nInput']   = task.nInput
    hyp['ann_nOutput']  = task.nOutput
    hyp['ann_initAct']  = task.activations[0]
    hyp['ann_absWCap']  = task.absWCap
    hyp['ann_mutSigma'] = task.absWCap * 0.1
    hyp['ann_layers']   = task.layers # if fixed toplogy is used


    if hyp['alg_act'] == 0:
      hyp['ann_actRange'] = task.actRange
    else:
      hyp['ann_actRange'] = np.full_like(task.actRange,hyp['alg_act'])

def register(neat_obj):
    flattened = (
        neat_obj.pop,
        neat_obj.species,
        neat_obj.innov,
        neat_obj.gen,
        neat_obj.pop_size,
        neat_obj.ann_nInput,
        neat_obj.ann_nOutput,
        neat_obj._best_wMat,
        neat_obj._best_aVec,
        neat_obj.key,
    )
    aux_data = None
    return flattened, aux_data


def unregister(aux_data, flattened):
    n = NeatAlgo()
    n.pop = flattened[0]
    n.species = flattened[1]
    n.innov = flattened[2]
    n.gen = flattened[3]
    n.pop_size = flattened[4]
    n.ann_nInput = flattened[5]
    n.ann_nOutput = flattened[6]
    n._best_wMat = flattened[7]
    n._best_aVec = flattened[8]
    n.key = flattened[9]
    return n


jax.tree_util.register_pytree_node(NeatAlgo, register, unregister)






