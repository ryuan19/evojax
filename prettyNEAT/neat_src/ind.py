import numpy as np
import copy

import sys
#sys.path.append('/content/drive/MyDrive/evojax')
#from .ann import getLayer, getNodeOrder
from myneat.ann import * #absolute import, run from parent

import jax.numpy as jnp
import jax


class Ind(): #need to initialize in jax
  """Individual class: genes, network, and fitness
  """
  def __init__(self, conn, node):
    """Intialize individual with given genes
    Args:
      conn - [5 X nUniqueGenes]
             [0,:] == Innovation Number
             [1,:] == Source
             [2,:] == Destination
             [3,:] == Weight
             [4,:] == Enabled?
      node - [3 X nUniqueGenes]
             [0,:] == Node Id
             [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
             [2,:] == Activation function (as int)

    Attributes:
      node    - (np_array) - node genes (see args)
      conn    - (np_array) - conn genes (see args)
      nInput  - (int)      - number of inputs
      nOutput - (int)      - number of outputs
      wMat    - (np_array) - weight matrix, one row and column for each node
                [N X N]    - rows: connection from; cols: connection to
      wVec    - (np_array) - wMat as a flattened vector
                [N**2 X 1]
      aVec    - (np_array) - activation function of each node (as int)
                [N X 1]
      nConn   - (int)      - number of connections
      fitness - (double)   - fitness averaged over all trials (higher better)
      X fitMax  - (double)   - best fitness over all trials (higher better)
      rank    - (int)      - rank in population (lower better)
      birth   - (int)      - generation born
      species - (int)      - ID of species
    """
    self.node = jnp.array(node)
    self.conn = jnp.array(conn)
    self.nInput = jnp.sum(node[1, :] == 1)
    self.nOutput = jnp.sum(node[1, :] == 2)
    self.wMat    = jnp.array([]) #or none if initialize later?
    self.wVec    = jnp.array([])
    self.aVec    = jnp.array([])
    self.nConn   = 0
    self.fitness = 0.0 # Mean fitness over trials
    #self.fitMax  = [] # Best fitness over trials
    self.rank    = 0
    self.birth   = 0
    self.species = 0

  def nConns(self):
    """Returns number of active connections
    """
    return int(jnp.sum(self.conn[4,:]))

  def express(self):
    """Converts genes to weight matrix and activation vector
    """
    order, wMat = getNodeOrder(self.node, self.conn)
    if order is not False:
      self.wMat = wMat
      self.aVec = self.node[2,order]

      wVec = self.wMat.flatten()
      # replace NaNs with 0 using jax.numpy where
      wVec = jnp.where(jnp.isnan(wVec), 0, wVec)
      self.wVec  = wVec
      self.nConn = jnp.sum(wVec!=0)
      return True
    else:
      return False

  def createChild(self, p, innov, gen=0, mate=None, key=None):
    """Create new individual with this individual as a parent

      Args:
        p      - (dict)     - algorithm hyperparameters (see p/hypkey.txt)
        innov  - (np_array) - innovation record
           [5 X nUniqueGenes]
           [0,:] == Innovation Number
           [1,:] == Source
           [2,:] == Destination
           [3,:] == New Node?
           [4,:] == Generation evolved
        gen    - (int)      - (optional) generation (for innovation recording)
        mate   - (Ind)      - (optional) second for individual for crossover


    Returns:
        child  - (Ind)      - newly created individual
        innov  - (np_array) - updated innovation record

    """
    if mate is not None:
      child, key = self.crossover(mate, key) #added key
    else:
      child = Ind(self.conn, self.node)

    child, innov, key = child.mutate(p,innov,gen, key=key)
    return child, innov, key

# -- Canonical NEAT recombination operators ------------------------------ -- #

  def crossover(self,mate, key):
    """Combine genes of two individuals to produce new individual

      Procedure:
      ) Inherit all nodes and connections from most fit parent
      ) Identify matching connection genes in parentA and parentB
      ) Replace weights with parentB weights with some probability

      Args:
        parentA  - (Ind) - Fittest parent
          .conns - (np_array) - connection genes
                   [5 X nUniqueGenes]
                   [0,:] == Innovation Number (unique Id)
                   [1,:] == Source Node Id
                   [2,:] == Destination Node Id
                   [3,:] == Weight Value
                   [4,:] == Enabled?
        parentB - (Ind) - Less fit parent

    Returns:
        child   - (Ind) - newly created individual

    """
    parentA = self
    parentB = mate

    # Inherit all nodes and connections from most fit parent
    child = Ind(parentA.conn, parentA.node)

    # Identify matching connection genes in ParentA and ParentB
    # aConn = np.copy(parentA.conn[0,:])
    # bConn = np.copy(parentB.conn[0,:])
    # matching, IA, IB = np.intersect1d(aConn,bConn,return_indices=True)
    aConn = jnp.copy(parentA.conn[0, :])
    bConn = jnp.copy(parentB.conn[0, :])
    matching, IA, IB = jnp.intersect1d(aConn, bConn, assume_unique=False, return_indices=True)


    # Replace weights with parentB weights with some probability
    bProb = 0.5
    key,subkey = jax.random.split(key)
    bGenes = jax.random.uniform(subkey, shape=(1, len(matching))) < bProb
    child.conn = child.conn.at[3, IA[bGenes[0]]].set(parentB.conn[3, IB[bGenes[0]]]) #jax idxing
    # bGenes = np.random.rand(1,len(matching))<bProb
    # child.conn[3,IA[bGenes[0]]] = parentB.conn[3,IB[bGenes[0]]]

    return child,key

  def mutate(self,p,innov=None,gen=None, key=None):
    """Randomly alter topology and weights of individual

    Args:
      p        - (dict)     - algorithm hyperparameters (see p/hypkey.txt)
      child    - (Ind) - individual to be mutated
        .conns - (np_array) - connection genes
                 [5 X nUniqueGenes]
                 [0,:] == Innovation Number (unique Id)
                 [1,:] == Source Node Id
                 [2,:] == Destination Node Id
                 [3,:] == Weight Value
                 [4,:] == Enabled?
        .nodes - (np_array) - node genes
                 [3 X nUniqueGenes]
                 [0,:] == Node Id
                 [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
                 [2,:] == Activation function (as int)
      innov    - (np_array) - innovation record
                 [5 X nUniqueGenes]
                 [0,:] == Innovation Number
                 [1,:] == Source
                 [2,:] == Destination
                 [3,:] == New Node?
                 [4,:] == Generation evolved

    Returns:
        child   - (Ind)      - newly created individual
        innov   - (np_array) - innovation record

    """
    # Readability
    # Readability (shallow copy?, update)
    # nConn = jnp.shape(self.conn)[1]
    # connG = jnp.copy(self.conn)
    # nodeG = jnp.copy(self.node)
    nConn = self.conn.shape[1]
    connG = self.conn
    nodeG = self.node


    # - Re-enable connections
    disabled = jnp.where(connG[4, :] == 0)[0]
    if len(disabled) > 0: #disable only as needed
      key, subkey = jax.random.split(key)
      reenabled = jax.random.uniform(subkey, shape=(1, len(disabled))) < p['prob_enable']
      connG = connG.at[4, disabled].set(reenabled[0])
    # disabled  = np.where(connG[4,:] == 0)[0]
    # reenabled = np.random.rand(1,len(disabled)) < p['prob_enable']
    # connG[4,disabled] = reenabled

    # - Weight mutation
    # [Canonical NEAT: 10% of weights are fully random...but seriously?]
    key, subkey1 = jax.random.split(key)
    mutatedWeights = jax.random.uniform(subkey1, shape=(1, nConn)) < p['prob_mutConn']
    key, subkey2 = jax.random.split(key)
    weightChange = mutatedWeights * jax.random.normal(subkey2, shape=(1, nConn)) * p['ann_mutSigma']
    connG = connG.at[3, :].add(weightChange[0])
    connG = connG.at[3, :].set(jnp.clip(connG[3, :], -p['ann_absWCap'], p['ann_absWCap']))

    # mutatedWeights = np.random.rand(1,nConn) < p['prob_mutConn'] # Choose weights to mutate
    # weightChange = mutatedWeights * np.random.randn(1,nConn) * p['ann_mutSigma']
    # connG[3,:] += weightChange[0]
    # # Clamp weight strength [ Warning given for nan comparisons ]
    # connG[3, (connG[3,:] >  p['ann_absWCap'])] =  p['ann_absWCap']
    # connG[3, (connG[3,:] < -p['ann_absWCap'])] = -p['ann_absWCap']

    # Split the key for random number generation
    key, subkey3 = jax.random.split(key)

    # Add node mutation with random threshold
    if (jax.random.uniform(subkey3) < p['prob_addNode']) and jnp.any(connG[4, :] == 1):
        connG, nodeG, innov, key = self.mutAddNode(connG, nodeG, innov, gen, p, key)

    # Split the key again for connection mutation
    key, subkey4 = jax.random.split(key)

    # Add connection mutation with random threshold
    if jax.random.uniform(subkey4) < p['prob_addConn']:
        connG, innov, key = self.mutAddConn(connG, nodeG, innov, gen, p, key)

    # if (np.random.rand() < p['prob_addNode']) and np.any(connG[4,:]==1):
    #   connG, nodeG, innov = self.mutAddNode(connG, nodeG, innov, gen, p)

    # if (np.random.rand() < p['prob_addConn']):
    #   connG, innov = self.mutAddConn(connG, nodeG, innov, gen, p)

    child = Ind(connG, nodeG)
    child.birth = gen

    return child, innov, key

  def mutAddNode(self, connG, nodeG, innov, gen, p, key):
    """Add new node to genome

    Args:
      connG    - (np_array) - connection genes
                 [5 X nUniqueGenes]
                 [0,:] == Innovation Number (unique Id)
                 [1,:] == Source Node Id
                 [2,:] == Destination Node Id
                 [3,:] == Weight Value
                 [4,:] == Enabled?
      nodeG    - (np_array) - node genes
                 [3 X nUniqueGenes]
                 [0,:] == Node Id
                 [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
                 [2,:] == Activation function (as int)
      innov    - (np_array) - innovation record
                 [5 X nUniqueGenes]
                 [0,:] == Innovation Number
                 [1,:] == Source
                 [2,:] == Destination
                 [3,:] == New Node?
                 [4,:] == Generation evolved
      gen      - (int) - current generation
      p        - (dict)     - algorithm hyperparameters (see p/hypkey.txt)


    Returns:
      connG    - (np_array) - updated connection genes
      nodeG    - (np_array) - updated node genes
      innov    - (np_array) - updated innovation record

    """
    if innov is None:
      newNodeId = int(jnp.max(nodeG[0,:]+1))
      newConnId = connG[0,-1]+1
    else:
      newNodeId = int(jnp.max(innov[2,:])+1) # next node id is a running counter
      newConnId = innov[0,-1]+1

    # Choose connection to split
    connActive = jnp.where(connG[4,:] == 1)[0]
    if len(connActive) < 1:
      return connG, nodeG, innov # No active connections, nothing to split
    key, subkey = jax.random.split(key)
    connSplit = connActive[jax.random.randint(subkey, minval=0, maxval=len(connActive), shape=())] #scalar
    # connSplit  = connActive[np.random.randint(len(connActive))]

    # Create new node

    key, subkey = jax.random.split(key)
    newActivation = p['ann_actRange'][jax.random.randint(subkey, minval=0, maxval=len(p['ann_actRange']), shape=())]
    newNode = jnp.array([[newNodeId, 3, newActivation]]).T #new node w jax arr

    # newActivation = p['ann_actRange'][np.random.randint(len(p['ann_actRange']))]
    # newNode = np.array([[newNodeId, 3, newActivation]]).T

    # Add connections to and from new node
    # -- Effort is taken to minimize disruption from node addition:
    # The 'weight to' the node is set to 1, the 'weight from' is set to the
    # original  weight. With a near linear activation function the change in performance should be minimal.


    connTo = connG[:, connSplit]
    connFrom = connG[:, connSplit]

    connTo = connTo.at[0].set(newConnId)      #set new connection ID
    connTo = connTo.at[2].set(newNodeId)      # set destination to newNodeId
    connTo = connTo.at[3].set(1)              # set weight to 1


    connFrom = connFrom.at[0].set(newConnId + 1)  # new connection ID + 1
    connFrom = connFrom.at[1].set(newNodeId)      # source to newNodeId
    connFrom = connFrom.at[3].set(connG[3, connSplit])  #  weight to  previous weight

    newConns = jnp.vstack((connTo, connFrom)).T

    # Disable original connection
    connG = connG.at[4, connSplit].set(0)
    #connG[4,connSplit] = 0


    if innov is not None:
        newInnov = jnp.empty((5, 2))
        newInnov = newInnov.at[:, 0].set(jnp.hstack((connTo[0:3], newNodeId, gen)))
        newInnov = newInnov.at[:, 1].set(jnp.hstack((connFrom[0:3], -1, gen)))
        innov = jnp.hstack((innov, newInnov))

    #append newNode and newConns
    nodeG = jnp.hstack((nodeG, newNode))
    connG = jnp.hstack((connG, newConns))

    # # Record innovations
    # if innov is not None:
    #   newInnov = np.empty((5,2))
    #   newInnov[:,0] = np.hstack((connTo[0:3], newNodeId, gen))
    #   newInnov[:,1] = np.hstack((connFrom[0:3], -1, gen))
    #   innov = np.hstack((innov,newInnov))

    # # Add new structures to genome
    # nodeG = np.hstack((nodeG,newNode))
    # connG = np.hstack((connG,newConns))

    return connG, nodeG, innov, key

  def mutAddConn(self, connG, nodeG, innov, gen, p, key=None):
    """Add new connection to genome.
    To avoid creating recurrent connections all nodes are first sorted into
    layers, connections are then only created from nodes to nodes of the same or
    later layers.


    Todo: check for preexisting innovations to avoid duplicates in same gen

    Args:
      connG    - (np_array) - connection genes
                 [5 X nUniqueGenes]
                 [0,:] == Innovation Number (unique Id)
                 [1,:] == Source Node Id
                 [2,:] == Destination Node Id
                 [3,:] == Weight Value
                 [4,:] == Enabled?
      nodeG    - (np_array) - node genes
                 [3 X nUniqueGenes]
                 [0,:] == Node Id
                 [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
                 [2,:] == Activation function (as int)
      innov    - (np_array) - innovation record
                 [5 X nUniqueGenes]
                 [0,:] == Innovation Number
                 [1,:] == Source
                 [2,:] == Destination
                 [3,:] == New Node?
                 [4,:] == Generation evolved
      gen      - (int)      - current generation
      p        - (dict)     - algorithm hyperparameters (see p/hypkey.txt)


    Returns:
      connG    - (np_array) - updated connection genes
      innov    - (np_array) - updated innovation record

    """
    if innov is None:
      newConnId = connG[0,-1]+1
    else:
      newConnId = innov[0,-1]+1

    nIns = jnp.sum(nodeG[1, :] == 1) + jnp.sum(nodeG[1, :] == 4)
    nOuts = jnp.sum(nodeG[1, :] == 2)
    order, wMat = getNodeOrder(nodeG, connG)
    hMat = wMat[nIns:-nOuts, nIns:-nOuts]
    hLay = getLayer(hMat) + 1

    # nIns = len(nodeG[0,nodeG[1,:] == 1]) + len(nodeG[0,nodeG[1,:] == 4])
    # nOuts = len(nodeG[0,nodeG[1,:] == 2])
    # order, wMat = getNodeOrder(nodeG, connG)   # Topological Sort of Network
    # hMat = wMat[nIns:-nOuts,nIns:-nOuts]
    # hLay = getLayer(hMat)+1

    # To avoid recurrent connections nodes are sorted into layers, and connections are only allowed from lower to higher layers
    if jnp.size(hLay) > 0:
        lastLayer = jnp.max(hLay) + 1
    else:
        lastLayer = 1

    # inputs (zeros), hidden layers (hLay), and outputs (lastLayer)
    L = jnp.concatenate([jnp.zeros(nIns), hLay, jnp.full((nOuts,), lastLayer)])
    nodeKey = jnp.column_stack([nodeG[0, order], L])

    # if len(hLay) > 0:
    #   lastLayer = max(hLay)+1
    # else:
    #   lastLayer = 1
    # L = np.r_[np.zeros(nIns), hLay, np.full((nOuts),lastLayer) ]
    # nodeKey = np.c_[nodeG[0,order], L] # Assign Layers

    key, subkey = jax.random.split(key)
    sources = jax.random.permutation(subkey, len(nodeKey))
    #sources = np.random.permutation(len(nodeKey))

    for src in sources:
      srcLayer = nodeKey[src,1]
      dest = jnp.where(nodeKey[:,1] > srcLayer)[0]

      # Finding already existing connections:
      #   ) take all connection genes with this source (connG[1,:])
      #   ) take the destination of those genes (connG[2,:])
      #   ) convert to nodeKey index (Gotta be a better numpy way...)
      srcIndx = jnp.where(connG[1,:]==nodeKey[src,0])[0]
      exist = connG[2,srcIndx]
      existKey = []
      for iExist in exist:
        existKey.append(jnp.where(nodeKey[:,0]==iExist)[0])

      jnpexistKey = jnp.array(existKey)
      dest = jnp.setdiff1d(dest,jnpexistKey) # Remove existing connections

      # Add a random valid connection
      key, subkey = jax.random.split(key)
      dest = jax.random.permutation(subkey, dest) #shuffle and set
      # np.random.shuffle(dest)
      if len(dest)>0:  # (there is a valid connection)

        connNew = jnp.zeros((5, 1))
        connNew = connNew.at[0].set(newConnId)
        connNew = connNew.at[1].set(nodeKey[src, 0])
        connNew = connNew.at[2].set(nodeKey[dest[0], 0])
        key, subkey = jax.random.split(key)
        connNew = connNew.at[3].set((jax.random.uniform(subkey) - 0.5) * 2 * p['ann_absWCap'])
        connNew = connNew.at[4].set(1)
        connG = jnp.concatenate((connG, connNew), axis=1)
        # connNew = np.empty((5,1))
        # connNew[0] = newConnId
        # connNew[1] = nodeKey[src,0]
        # connNew[2] = nodeKey[dest[0],0]
        # connNew[3] = (np.random.rand()-0.5)*2*p['ann_absWCap']
        # connNew[4] = 1
        # connG = np.c_[connG,connNew]

        # Record innovation
        if innov is not None:
          newInnov = jnp.hstack((connNew[0:3].flatten(), -1, gen))
          innov = jnp.hstack((innov,newInnov[:,None]))
        break

    return connG, innov, key

#initialize class for pytree system
def register(ind_obj): #make 1d from Ind obj
    flattened_obj = (
        ind_obj.conn,
        ind_obj.node,
        ind_obj.nInput,
        ind_obj.nOutput,
        ind_obj.wMat,
        ind_obj.wVec,
        ind_obj.aVec,
        ind_obj.nConn,
        ind_obj.fitness,
        ind_obj.rank,
        ind_obj.birth,
        ind_obj.species,
    )
    aux_data = None
    return flattened_obj, aux_data


def unregister(aux_data,  flattened_components): #make from 1d to class
    n = Ind(flattened_components[0], flattened_components[1])
    n.nInput = flattened_components[2]
    n.nOutput = flattened_components[3]
    n.wMat = flattened_components[4]
    n.wVec = flattened_components[5]
    n.aVec = flattened_components[6]
    n.nConn = flattened_components[7]
    n.fitness = flattened_components[8]
    n.rank = flattened_components[9]
    n.birth = flattened_components[10]
    n.species = flattened_components[11]

    return n


jax.tree_util.register_pytree_node(Ind, register, unregister)