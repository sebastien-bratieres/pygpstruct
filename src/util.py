import numpy as np
dtype_for_arrays=np.float32

def merge_dict(d1, d2):
    d1 = d1.copy()
    d1.update(d2)
    return d1

def kron_list(major, minor = [[{}]]):
    """ returns Kronecker products of matrices (major kron minor). 
    both minor and major are expected to be matrices in list-format (row-dominant), 
    that is lists of rows, each row a list of elements.
    
    Usage example:
        a = [{ 'prior' : 'uniform'}, {'prior': 'Gaussian'}]
        b = [{'method' : 'ss'}, {'method' : 'pw'}, {'method' : 'sd'}]
        c = [{'random_seed' : i} for i in range(32)]
        table = kron_list(to_column(c), kron_list(to_row(a), to_row(b)))
    """
    result = []
    for major_row in major:
        for minor_row in minor:
            result_row = []
            for major_elt in major_row:
                for minor_elt in minor_row:
                    result_row.append(merge_dict(major_elt, minor_elt))
            result.append(result_row) 
    return result

# utilities to convert lists to rows resp columns
to_column = lambda l : list(map(lambda x :[x], l))
to_row = lambda l : [l]

'''
# complicated implementation, dismiss
def kron_list(major, minor):
    result = []
    for major_row in major:
        for minor_row in minor:
            result.append(list(
                itertools.chain(
                    *[list(map(lambda minor_elt : merge_dict(major_elt, minor_elt), minor_row)) for major_elt in major_row]
                    )))
    return result
'''

def log_code_hash():
    '''produce log string documenting status/ commit hash of current source code'''
    import subprocess
    import os
    wd = os.path.dirname(os.path.realpath(__file__))
    label = subprocess.check_output(["git", "status", "--porcelain"], cwd=wd).decode('utf-8')
    if label != '':
        label = 'Local changes:\n' + label
    else:
        label = 'Clean working copy. '
    label += 'Hash of current commit: ' + subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=wd).decode('utf-8')
    return label

# will work with dict
# keeps only the last result in memory at any given time (useful for very big results)
# will hash input using str() function (cos it should work on small dicts)
class memoize_once(dict):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        h = str(args)
        if h in self :
            return self[h]
        else:
            self.clear() # will clear dict before storing new key
            result = self[h] = self.func(*args)
            return result
            
def read_randoms(n=-1, type=None, should=None):
    """
    to intialize this function:
    read_randoms.offset=0 #DEBUG
    read_randoms.file = np.loadtxt('/tmp/sb358/ess_randoms.txt') #DEBUG
    
    This function was written for an easy switch between obtaining pseudo-random numbers from
    - a PRNG
    - a file where a sequence of such PRN is stored (to allow reusing the same random sequence between Matlab and Python, in my case)
    """
    if not hasattr(read_randoms, 'file'):
        if type != None:
            if type=='u':
                result=read_randoms.prng.rand(n)
            else:
                result=read_randoms.prng.randn(n)
        else:
            return # type==None, but we're generating random numbers so can't check anything
    else:
        if (n == -1 and should != None):
            n = len(should)
        result = read_randoms.file[read_randoms.offset:read_randoms.offset+n]
        if should != None:
            #print("testing start offset : " + str(read_randoms.offset) + ", length : " + str(n))
            try:
                numpy.testing.assert_almost_equal(should, result)
            except AssertionError as e:
                raise e
            
        read_randoms.offset = read_randoms.offset+n
        if n == 1:
            print("read random number %g" % result)
    return result.astype(dtype_for_arrays)


import time
import os
class stop_check:
    """.evaluate will return True if delay is elapsed OR path exists"""
    def __init__(self, delay = None, path = None):
        self.stop_time = None
        self.path = None
        if (delay != None):
            self.stop_time = delay + time.time()
        if path != None:
            self.path = path
    
    def evaluate(self):
        if self.stop_time != None and self.stop_time < time.time():
            return True
        if self.path != None and os.path.exists(self.path):
            return True
        return False
            
# effective sample size function, from https://code.google.com/p/biopy/source/browse/trunk/biopy/bayesianStats.py
def effective_sample_size(data, stepSize = 1) :
  """ Effective sample size, as computed by BEAST Tracer."""
  samples = len(data)

  assert len(data) > 1,"no stats for short sequences"
  
  maxLag = min(samples//3, 1000)

  gammaStat = [0,]*maxLag
  #varGammaStat = [0,]*maxLag

  varStat = 0.0;

  if type(data) != np.ndarray :
    data = np.array(data)

  normalizedData = data - data.mean()
  
  for lag in range(maxLag) :
    v1 = normalizedData[:samples-lag]
    v2 = normalizedData[lag:]
    v = v1 * v2
    gammaStat[lag] = sum(v) / len(v)
    #varGammaStat[lag] = sum(v*v) / len(v)
    #varGammaStat[lag] -= gammaStat[0] ** 2

    # print lag, gammaStat[lag], varGammaStat[lag]
    
    if lag == 0 :
      varStat = gammaStat[0]
    elif lag % 2 == 0 :
      s = gammaStat[lag-1] + gammaStat[lag]
      if s > 0 :
         varStat += 2.0*s
      else :
        break
      
  # standard error of mean
  # stdErrorOfMean = Math.sqrt(varStat/samples);

  # auto correlation time
  act = stepSize * varStat / gammaStat[0]

  # effective sample size
  ess = (stepSize * samples) / act

  return ess