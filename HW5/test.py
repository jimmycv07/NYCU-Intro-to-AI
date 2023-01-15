# grid = [[5 for _ in range(8)] for _ in range(9)]
# print(grid)
# g=[ 6 for _ in range(5)]
# print(g)
# v=0
# assert v>0 

# print(int(0.5))

import collections

# def z():
#     return 0
# dic= collections.defaultdict(z)

# dic['o']+=999

# print(dic['o'])
# print(dic['z'])

import util
import random
transProb = util.loadTransProb()
transProbDict = dict()
for (oldTile, newTile) in transProb:
    # print(oldTile, newTile)
    if oldTile not in transProbDict:
        transProbDict[oldTile] = collections.defaultdict(int)
    transProbDict[oldTile][newTile] = transProb[(oldTile, newTile)]
    # if not transProb[(oldTile, newTile)]:
        # print(oldTile, newTile)

particles = collections.defaultdict(int)
potentialParticles = list(transProbDict.keys())
print(potentialParticles)
# print(transProbDict[(8,7)])
print(len(potentialParticles))
# print(transProbDict[(4,10)])
# if transProb==transProbDict:
    # print(66)
# print(transProbDict)
for _ in range(200):
    particleIndex = int(random.random() * len(potentialParticles))
    particles[potentialParticles[particleIndex]] += 1