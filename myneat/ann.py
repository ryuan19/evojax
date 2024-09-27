
import numpy as np
import jax
import jax.numpy as jnp
import logging
from typing import Tuple
from evojax.util import * #logging and stuff
#policystate,policynetwork, taskstate from evojax policy
from evojax.task.base import TaskState
from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState


### pOLICY FOR FEEDFWD NEURAL NET FOR NEAT

# -- ANN Ordering -------------------------------------------------------- -- #

def getNodeOrder(nodeG,connG):
  """Builds connection matrix from genome through topological sorting.

  Args:
    nodeG - (np_array) - node genes
            [3 X nUniqueGenes]
            [0,:] == Node Id
            [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
            [2,:] == Activation function (as int)

    connG - (np_array) - connection genes
            [5 X nUniqueGenes]
            [0,:] == Innovation Number (unique Id)
            [1,:] == Source Node Id
            [2,:] == Destination Node Id
            [3,:] == Weight Value
            [4,:] == Enabled?

  Returns:
    Q    - [int]      - sorted node order as indices
    wMat - (np_array) - ordered weight matrix
           [N X N]

    OR

    False, False      - if cycle is found

  Todo:
    * setdiff1d is slow, as all numbers are positive ints is there a
      better way to do with indexing tricks (as in quickINTersect)?
  """
  conn = jnp.copy(connG)
  node = jnp.copy(nodeG)

  nIns = jnp.sum(node[1, :] == 1) + jnp.sum(node[1, :] == 4)
  nOuts = jnp.sum(node[1, :] == 2)
  # nIns = len(node[0,node[1,:] == 1]) + len(node[0,node[1,:] == 4])
  # nOuts = len(node[0,node[1,:] == 2])

  # Create connection and initial weight matrices
  conn = conn.at[3, conn[4, :] == 0].set(jnp.nan)
  src = jnp.asarray(conn[1, :], dtype=jnp.int32)
  dest = jnp.asarray(conn[2, :], dtype=jnp.int32)
  # conn[3,conn[4,:]==0] = np.nan # disabled but still connected
  # src  = conn[1,:].astype(int)
  # dest = conn[2,:].astype(int)


  lookup = node[0, :].astype(jnp.int32) #vectorized version
  index_map = jnp.zeros(jnp.max(lookup) + 1, dtype=int) - 1
  index_map = index_map.at[lookup].set(jnp.arange(len(lookup)))
  src = jnp.where(index_map[src] != -1, index_map[src], src)
  dest = jnp.where(index_map[dest] != -1, index_map[dest], dest)
  # for i in range(len(lookup)): # Can we vectorize this?
  #   src[np.where(src==lookup[i])] = i
  #   dest[np.where(dest==lookup[i])] = i

  wMat = jnp.zeros((node.shape[1], node.shape[1]))
  wMat = wMat.at[src, dest].set(conn[3, :])
  connMat = wMat[nIns+nOuts:, nIns+nOuts:]
  connMat = jnp.where(connMat != 0, 1, 0)
  # wMat = np.zeros((np.shape(node)[1],np.shape(node)[1]))
  # wMat[src,dest] = conn[3,:]
  # connMat = wMat[nIns+nOuts:,nIns+nOuts:]
  # connMat[connMat!=0] = 1

  # Topological Sort of Hidden Nodes
  edge_in = jnp.sum(connMat, axis = 0)
  Q = jnp.where(edge_in == 0)[0]
  for i in range(connMat.shape[0]): #length of arr
      if Q.size == 0 or i >= Q.size:
          Q = jnp.array([]) #relevant?
          return False, False
      edge_out = connMat[Q[i], :]
      edge_in = edge_in - edge_out
      nextNodes = jnp.setdiff1d(jnp.where(edge_in == 0)[0], Q)
      Q = jnp.concatenate([Q, nextNodes])
      if jnp.sum(edge_in) == 0:
            break

  # edge_in = np.sum(connMat,axis=0)
  # Q = np.where(edge_in==0)[0]  # Start with nodes with no incoming connections
  # for i in range(len(connMat)):
  #   if (len(Q) == 0) or (i >= len(Q)):
  #     Q = []
  #     return False, False # Cycle found, can't sort
  #   edge_out = connMat[Q[i],:]
  #   edge_in  = edge_in - edge_out # Remove nodes' conns from total
  #   nextNodes = np.setdiff1d(np.where(edge_in==0)[0], Q)
  #   Q = np.hstack((Q,nextNodes))

  #   if sum(edge_in) == 0:
  #     break

  # Add In and outs back and reorder wMat according to sort
  Q = Q + nIns + nOuts
  Q = jnp.concatenate((lookup[:nIns], Q, lookup[nIns : nIns + nOuts]))
  wMat = wMat[jnp.ix_(Q, Q)]
  # Q += nIns+nOuts
  # Q = np.r_[lookup[:nIns], Q, lookup[nIns:nIns+nOuts]]
  # wMat = wMat[np.ix_(Q,Q)]

  return Q, wMat #no random no key

def getLayer(wMat):
  """Get layer of each node in weight matrix
  Traverse wMat by row, collecting layer of all nodes that connect to you (X).
  Your layer is max(X)+1. Input and output nodes are ignored and assigned layer
  0 and max(X)+1 at the end.

  Args:
    wMat  - (np_array) - ordered weight matrix
           [N X N]

  Returns:
    layer - [int]      - layer # of each node

  Todo:
    * With very large networks this might be a performance sink -- especially,
    given that this happen in the serial part of the algorithm. There is
    probably a more clever way to do this given the adjacency matrix.
  """
  if wMat.size == 0:
        return jnp.array([])

  wMat = jnp.where(jnp.isnan(wMat), 0, wMat)
  wMat = jnp.where(wMat != 0, 1, 0)
  nNode = wMat.shape[0]
  layer = jnp.zeros((nNode,)) #1d tuple
  # wMat[np.isnan(wMat)] = 0
  # wMat[wMat!=0]=1
  # nNode = np.shape(wMat)[0]
  # layer = np.zeros((nNode))

  def loop_until_stable(carry):
        layer, _ = carry
        srcLayer = jnp.max(layer[:, None] * wMat, axis=0) + 1
        return srcLayer, layer
  def condition(carry):
        layer, prevOrder = carry
        return jnp.any(prevOrder != layer)
  #continues until the layers converge (when prevOrder equals layer)
  layer, _ = jax.lax.while_loop(condition, loop_until_stable, (layer, jnp.zeros_like(layer)))
  # while (True):
  #   prevOrder = np.copy(layer)
  #   for curr in range(nNode):
  #     srcLayer=np.zeros((nNode))
  #     for src in range(nNode):
  #       srcLayer[src] = layer[src]*wMat[src,curr]
  #     layer[curr] = np.max(srcLayer)+1
  #   if all(prevOrder==layer):
  #     break
  return layer-1


# -- ANN Activation ------------------------------------------------------ -- #

@jax.jit
def act(nodes, weights, aVec, inPattern): #feedfwd part, used jax.jit bc lots of computations
  """Returns FFANN output given a single input pattern
  If the variable weights is a vector it is turned into a square weight matrix.

  Allows the network to return the result of several samples at once if given a matrix instead of a vector of inputs:
      Dim 0 : individual samples
      Dim 1 : dimensionality of pattern (# of inputs)

  Args:
    weights   - (np_array) - ordered weight matrix or vector
                [N X N] or [N**2]
    aVec      - (np_array) - activation function of each node
                [N X 1]    - stored as ints (see applyAct in ann.py)
    nInput    - (int)      - number of input nodes
    nOutput   - (int)      - number of output nodes
    inPattern - (np_array) - input activation
                [1 X nInput] or [nSamples X nInput]

  Returns:
    output    - (np_array) - output activation
                [1 X nOutput] or [nSamples X nOutput]
  """
  total_nodes = weights.shape[0]

  @jax.jit
  def initialize_nodeAct(inPattern):
      return jnp.concatenate(
          [jnp.array([1.0]), inPattern, jnp.zeros(total_nodes - 13)]
      )

  @jax.jit
  def loop_body(i, nodeAct):
      rawAct = jnp.dot(nodeAct, weights[:, i])
      return nodeAct.at[i].set(applyAct(aVec[i], rawAct))

  nodeAct = initialize_nodeAct(inPattern)
  nodeAct = jax.lax.fori_loop(13, total_nodes, loop_body, nodeAct)

  return jax.lax.dynamic_slice(nodeAct, (nodes - 3,), (3,))
  # Turn weight vector into weight matrix
  # if np.ndim(weights) < 2:
  #     nNodes = int(np.sqrt(np.shape(weights)[0]))
  #     wMat = np.reshape(weights, (nNodes, nNodes))
  # else:
  #     nNodes = np.shape(weights)[0]
  #     wMat = weights
  # wMat[np.isnan(wMat)]=0

  # Vectorize input
  # if np.ndim(inPattern) > 1:
  #     nSamples = np.shape(inPattern)[0]
  # else:
  #     nSamples = 1

  # Run input pattern through ANN
  # nodeAct  = np.zeros((nSamples,nNodes))
  # nodeAct[:,0] = 1 # Bias activation
  # nodeAct[:,1:nInput+1] = inPattern

  # iNode = nInput+1
  # for iNode in range(nInput+1,nNodes):
  #     rawAct = np.dot(nodeAct, wMat[:,iNode]).squeeze()
  #     nodeAct[:,iNode] = applyAct(aVec[iNode], rawAct)
  #     #print(nodeAct)
  #output = nodeAct[:,-nOutput:]
  return output

@jax.jit
def applyAct(actId, x):
  """Returns value after an activation function is applied
  Lookup table to allow activations to be stored in numpy arrays

  case 1  -- Linear
  case 2  -- Unsigned Step Function
  case 3  -- Sin
  case 4  -- Gausian with mean 0 and sigma 1
  case 5  -- Hyperbolic Tangent [tanh] (signed)
  case 6  -- Sigmoid unsigned [1 / (1 + exp(-x))]
  case 7  -- Inverse
  case 8  -- Absolute Value
  case 9  -- Relu
  case 10 -- Cosine
  case 11 -- Squared

  Args:
    actId   - (int)   - key to look up table
    x       - (???)   - value to be input into activation
              [? X ?] - any type or dimensionality

  Returns:
    output  - (float) - value after activation is applied
              [? X ?] - same dimensionality as input
  """
  # print(actId , "hellloo") # Traced<ShapedArray(float32[16])>with<DynamicJaxprTrace(level=6/0)>  ? 
  
  return jax.lax.switch(
      actId.astype(int),  # The key to select which function to apply
      [
          lambda: x,  # Linear
          lambda: jnp.where(x > 0, 1.0, 0.0),  # Unsigned Step Function
          lambda: jnp.sin(jnp.pi * x),  # Sin
          lambda: jnp.exp(-jnp.square(x) / 2.0),  # Gaussian
          lambda: jnp.tanh(x),  # Hyperbolic Tangent (signed)
          lambda: (jnp.tanh(x / 2.0) + 1.0) / 2.0,  # Sigmoid (unsigned)
          lambda: -x,  # Inverse
          lambda: jnp.abs(x),  # Absolute Value
          lambda: jnp.maximum(0, x),  # ReLU
          lambda: jnp.cos(jnp.pi * x),  # Cosine
          lambda: jnp.square(x),  # Squared
      ],
  )
  # if actId == 1:   # Linear
  #   value = x

  # if actId == 2:   # Unsigned Step Function
  #   value = 1.0*(x>0.0)
  #   #value = (np.tanh(50*x/2.0) + 1.0)/2.0

  # elif actId == 3: # Sin
  #   value = np.sin(np.pi*x)

  # elif actId == 4: # Gaussian with mean 0 and sigma 1
  #   value = np.exp(-np.multiply(x, x) / 2.0)

  # elif actId == 5: # Hyperbolic Tangent (signed)
  #   value = np.tanh(x)

  # elif actId == 6: # Sigmoid (unsigned)
  #   value = (np.tanh(x/2.0) + 1.0)/2.0

  # elif actId == 7: # Inverse
  #   value = -x

  # elif actId == 8: # Absolute Value
  #   value = abs(x)

  # elif actId == 9: # Relu
  #   value = np.maximum(0, x)

  # elif actId == 10: # Cosine
  #   value = np.cos(np.pi*x)

  # elif actId == 11: # Squared
  #   value = x**2

  # else:
  #   value = x

  # return value


# -- Action Selection ---------------------------------------------------- -- #

def selectAct(action, actSelect): #softmax only
  """Selects action based on vector of actions

    Single Action:
    - Hard: a single action is chosen based on the highest index
    - Prob: a single action is chosen probablistically with higher values
            more likely to be chosen

    We aren't selecting a single action:
    - Softmax: a softmax normalized distribution of values is returned
    - Default: all actions are returned

  Args:
    action   - (np_array) - vector weighting each possible action
                [N X 1]

  Returns:
    i         - (int) or (np_array)     - chosen index
                         [N X 1]
  """
  if actSelect == 'softmax':
    action = softmax(action)
  else:
    action = action.flatten()
  return action

@jax.jit
def softmax(x):
    """Compute softmax values for each sets of scores in x.
    Assumes: [samples x dims]

    Args:
      x - (np_array) - unnormalized values
          [samples x dims]

    Returns:
      softmax - (np_array) - softmax normalized in dim 1

    Todo: Untangle all the transposes...
    """
    max_x = jnp.max(x, axis=-1, keepdims=True)
    e_x = jnp.exp(x - max_x)
    return e_x / jnp.sum(e_x, axis=-1, keepdims=True)
    # if x.ndim == 1:
    #   e_x = np.exp(x - np.max(x))
    #   return e_x / e_x.sum(axis=0)
    # else:
    #   e_x = np.exp(x.T - np.max(x,axis=1))
    #   return (e_x / e_x.sum(axis=0)).T
@jax.jit
def weightedRandom(weights,key):
  """Returns random index, with each choices chance weighted
  Args:
    weights   - (np_array) - weighting of each choice
                [N X 1]

  Returns:
    i         - (int)      - chosen index
  """
  weights = weights - jnp.min(weights)  # Handle negative values
  cum_weights = jnp.cumsum(weights)

  # Generate a random number
  key, subkey = jax.random.split(key)
  random_value = jax.random.uniform(subkey) * cum_weights[-1]

  # Find the index where the cumulative sum first exceeds the random value
  index = jnp.searchsorted(cum_weights, random_value)

  return index, key
  # minVal = np.min(weights)
  # weights = weights - minVal # handle negative vals
  # cumVal = np.cumsum(weights)
  # pick = np.random.uniform(0, cumVal[-1])
  # for i in range(len(weights)):
  #   if cumVal[i] >= pick:
  #     return i

def selectAct_wr(action, actSelect,key): #weighted random, need key
  """Selects action based on vector of actions

    Single Action:
    - Hard: a single action is chosen based on the highest index
    - Prob: a single action is chosen probablistically with higher values
            more likely to be chosen

    We aren't selecting a single action:
    - Softmax: a softmax normalized distribution of values is returned
    - Default: all actions are returned

  Args:
    action   - (np_array) - vector weighting each possible action
                [N X 1]

  Returns:
    i         - (int) or (np_array)     - chosen index
                         [N X 1]
  """
  if actSelect == 'prob':
    action = weightedRandom(action,key)
  else:
    action = action.flatten()
  return action,key

class NeatPolicy(PolicyNetwork): #feed fwd neural net
   #just has get action in base network

   #use mlp as ref
    def __init__(self, logger: logging.Logger = None):
      if logger is None:
          self._logger = create_logger(name="NeatPolicy")
      else:
          self._logger = logger
    def get_actions(
        self,
        t_states: TaskState,
        nodes: jnp.ndarray,
        weights: jnp.ndarray,
        activations: jnp.ndarray,
        p_states: PolicyState,
    ) -> Tuple[jnp.ndarray, PolicyState]:
        
        def get_single_action(nodes, weights, activations, obs):
            return jax.nn.softmax(act(nodes, weights, activations, obs), axis=-1)

        #get_multiple_action = jax.vmap(get_single_action, in_axes=(None, None, None, 0))
        get_multiple_action = jax.vmap(get_single_action)

        # Handle different cases based on nNodes.shape
        if len(nodes.shape) == 0:  # Single observation
            outputs = get_single_action(nodes, weights, activations, t_states.obs[0])
            outputs = jnp.expand_dims(outputs, axis=0)  # Add batch dimension

        elif len(nodes.shape) == 1:  # Batch of observations
            
            outputs = get_multiple_action(nodes, weights, activations, t_states.obs)

        else:  # More complex/nested batch structure
            outputs = jax.vmap(get_multiple_action)(nodes, weights, activations, t_states.obs)

        return outputs, p_states




# -- File I/O ------------------------------------------------------------ -- #

def exportNet(filename,wMat, aVec):
  indMat = np.c_[wMat,aVec]
  np.savetxt(filename, indMat, delimiter=',',fmt='%1.2e')

def importNet(fileName):
  ind = np.loadtxt(fileName, delimiter=',')
  wMat = ind[:,:-1]     # Weight Matrix
  aVec = ind[:,-1]      # Activation functions

  # Create weight key
  wVec = wMat.flatten()
  wVec[np.isnan(wVec)]=0
  wKey = np.where(wVec!=0)[0]

  return wVec, aVec, wKey


