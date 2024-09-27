import numpy as np
import jax
import jax.numpy as jnp

def roulette(pArr):
  """Returns random index, with each choices chance weighted
  Args:
    pArr    - (np_array) - vector containing weighting of each choice
              [N X 1]

  Returns:
    choice  - (int)      - chosen index
  """
  spin = np.random.rand()*np.sum(pArr)
  slot = pArr[0]
  choice = len(pArr)
  for i in range(1,len(pArr)):
    if spin < slot:
      choice = i
      break
    else:
      slot += pArr[i]
  return choice

def listXor(b,c):
  """Returns elements in lists b and c they don't share
  """
  A = [a for a in b+c if (a not in b) or (a not in c)]
  return A

def rankArray(X): #made into jax
  """Returns ranking of a list, with ties resolved by first-found first-order
  NOTE: Sorts descending to follow numpy conventions
  """
  tmp = jnp.argsort(X)
  rank = jnp.empty_like(tmp)
  rank = rank.at[tmp].set(jnp.arange(len(X)))
  #rank[tmp] = np.arange(len(X))
  return rank

def tiedRank(X):
  """Returns ranking of a list, with ties recieving and averaged rank
  # Modified from: github.com/cmoscardi/ox_ml_practical/blob/master/util.py
  """
  sorter = jnp.argsort(-X)
  X_sorted = X[sorter]
  # Identify where the values change to detect ties
  diffs = X_sorted[1:] != X_sorted[:-1] #find ties
  group_ids = jnp.cumsum(jnp.concatenate(([0], diffs))) #assign groups
  unique_group_ids, group_starts, group_counts = jnp.unique(
      group_ids, return_index=True, return_counts=True
  )
  group_ends = group_starts + group_counts - 1 # for 0 based indexing

  average_ranks_per_group = (group_starts + group_ends) / 2.0 #calc avg
  average_ranks = average_ranks_per_group[group_ids]
  Rx = jnp.zeros_like(X, dtype=jnp.float32) #move back to og position
  Rx = Rx.at[sorter].set(average_ranks)

  return Rx
  # Z = [(x, i) for i, x in enumerate(X)]
  # Z.sort(reverse=True)
  # n = len(Z)
  # Rx = [0]*n
  # start = 0 # starting mark
  # for i in range(1, n):
  #    if Z[i][0] != Z[i-1][0]:
  #      for j in range(start, i):
  #        Rx[Z[j][1]] = float(start+1+i)/2.0;
  #      start = i
  # for j in range(start, n):
  #   Rx[Z[j][1]] = float(start+1+n)/2.0;

  # return np.asarray(Rx)

def bestIntSplit(ratio, total):
  """Divides a total into integer shares that best reflects ratio
    Args:
      share      - [1 X N ] - Percentage in each pile
      total      - [int   ] - Integer total to split

    Returns:
      intSplit   - [1 x N ] - Number in each pile
  """
  ratio = jnp.asarray(ratio, dtype=jnp.float32)
  total = jnp.asarray(total, dtype=jnp.float32)
  ratio_sum = jnp.sum(ratio)
  ratio = jnp.where(ratio_sum != 1.0, ratio / ratio_sum, ratio)

  float_split = ratio * total
  int_split = jnp.floor(float_split)
  remainder = total - jnp.sum(int_split)
  remainder = remainder.astype(jnp.int32)

  rounding_diff = float_split - int_split
  deserving = jnp.argsort(-rounding_diff)


  def add_remainder(int_split, deserving, remainder):
      indices_to_increment = deserving[:remainder]
      int_split = int_split.at[indices_to_increment].add(1)
      return int_split

  int_split = jax.lax.cond(
      remainder > 0,
      lambda x: add_remainder(x, deserving, remainder),
      lambda x: x,
      int_split
  )

  return int_split.astype(jnp.int32)

  # Handle poorly defined ratio
  # if sum(ratio) is not 1:
  #   ratio = np.asarray(ratio)/sum(ratio)

  # # Get share in real and integer values
  # floatSplit = np.multiply(ratio,total)
  # intSplit   = np.floor(floatSplit)
  # remainder  = int(total - sum(intSplit))

  # # Rank piles by most cheated by rounding
  # deserving = np.argsort(-(floatSplit-intSplit),axis=0)

  # # Distribute remained to most deserving
  # intSplit[deserving[:remainder]] = intSplit[deserving[:remainder]] + 1
  # return intSplit

def quickINTersect(A,B):
  """ Faster set intersect: only valid for vectors of positive integers.
  (useful for matching indices)

    Example:
    A = np.array([0,1,2,3,5],dtype=np.int16)
    B = np.array([0,1,6,5],dtype=np.int16)
    C = np.array([0],dtype=np.int16)
    D = np.array([],dtype=np.int16)

    print(quickINTersect(A,B))
    print(quickINTersect(B,C))
    print(quickINTersect(B,D))
  """
  if (len(A) == 0) or (len(B) == 0):
    return [],[]
  P = np.zeros((1+max(max(A),max(B))),dtype=bool)
  P[A] = True
  IB = P[B]
  P[A] = False # Reset
  P[B] = True
  IA = P[A]

  return IA, IB