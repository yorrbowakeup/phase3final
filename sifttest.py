#with open('photozero/pairwise_scores.txt', 'r') as file:
"""
with open('photozero/prepare/list.txt', 'r') as file:
    x = file.read()
    print(x)
"""
from sklearn.cluster import DBSCAN
import numpy as np
import sys

INF = sys.maxsize
g = open('photozero/constraints.txt')
for i in range(1):
    n = int(g.readline())
graph = [[0 for _ in range(n)] for _ in range(n)]
f = open('photozero/pairwise_scores.txt', 'r')
for i in range(n):
    for j in range(n):
        if i != j:
            graph[i][j] = 3

while True:
    line = f.readline()
    if not line:
        break
    a, b, dist = map(float, line.split())
    x, y = int(a), int(b)
    if dist != 0:
        graph[x][y] = 1/dist
        graph[y][x] = 1/dist
print(graph[20][19], graph[19][20])
ds = DBSCAN(eps=2, min_samples=2, metric='precomputed')
graph = np.matrix(graph)
clustering = ds.fit(graph)
print(clustering.labels_)







