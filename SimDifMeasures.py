import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.metrics.pairwise import euclidean_distances as ed
from scipy.stats import pearsonr as pc
from scipy.spatial import distance as ds

def cosSim(a, b):
    # https://www.machinelearningplus.com/nlp/cosine-similarity/
    aa = a.reshape(1, len(a))
    bb = b.reshape(1, len(b))
    ans = cs(aa, bb)
    return ans[0]

def cosSimByHand(a, b):
    # https://skipperkongen.dk/2018/09/19/cosine-similarity-in-python/
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    return dot / (norma * normb)

def pearsonCorrCoef(a,b):
    # https://kite.com/python/examples/656/scipy-compute-the-pearson-correlation-coefficient
    ans, pval = pc(a,b)
    return ans

def euclideanDistance(a,b):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html
    aa = a.reshape(1, len(a))
    bb = b.reshape(1, len(b))
    return ed(aa,bb)

def jaccardSimilarity(a,b):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html
    # Jaccard distance = 1- Jaccard similarity
    # Jaccard similarity = 1 - Jaccard distance
    # https://www.statisticshowto.datasciencecentral.com/jaccard-index/
    dis = ds.jaccard(a,b)
    return 1 - dis

def driver(a,b):
    print("Cosine:", cosSim(a,b))
    #print(cosSimByHand(a,b))
    print("Correlation:", pearsonCorrCoef(a,b))
    print("Euclidian:", euclideanDistance(a,b))
    print("Jaccard:", jaccardSimilarity(a,b))

xa = np.array([1, 1, 1, 1])
ya = np.array([2, 2, 2, 2])
xb = np.array([0, 1, 0, 1])
yb = np.array([1, 0, 1, 0])
xc = np.array([0, -1, 0, 1])
yc = np.array([1, 0, -1, 0])
xd = np.array([1, 1, 0, 1, 0, 1])
yd = np.array([1, 1, 1, 0, 0, 1])
xe = np.array([2, -1, 0, 2, 0, -3])
ye = np.array([-1, 1, -1, 0, 0, -1])

print("Problem 7a:"), driver(xa, ya)
print("\nProblem 7b:"), driver(xb, yb)
print("\nProblem 7c:"), driver(xc, yc)
print("\nProblem 7d:"), driver(xd, yd)
print("\nProblem 7e:"), driver(xe, ye)

"""
Output to console:

Problem 7a:
Cosine: [1.]
/home/walters_aj101/.local/lib/python3.6/site-packages/scipy/stats/stats.py:3399:
  PearsonRConstantInputWarning: An input array is constant; the correlation
  coefficent is not defined.
  warnings.warn(PearsonRConstantInputWarning())
Correlation: nan
Euclidian: [[2.]]
Jaccard: 0.0

Problem 7b:
Cosine: [0.]
Correlation: -1.0
Euclidian: [[2.]]
Jaccard: 0.0

Problem 7c:
Cosine: [0.]
Correlation: 0.0
Euclidian: [[2.]]
Jaccard: 0.0

Problem 7d:
Cosine: [0.75]
Correlation: 0.25
Euclidian: [[1.41421356]]
Jaccard: 0.6

Problem 7e:
Cosine: [0.]
Correlation: 5.551115123125783e-17
Euclidian: [[4.69041576]]
Jaccard: 0.0
"""
