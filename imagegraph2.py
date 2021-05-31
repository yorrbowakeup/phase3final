import networkx as nx
import matplotlib.pyplot as plt

#def making_graph(path):
    #G = nx.DiGraph()


def reading(path):
    f = open(path)
    lines = f.readlines()
    G = nx.Graph()
    for line in lines:
        a, b, c = line.split(",")
        a = a.split('/')[-1]
        a = a.split('.')[0]
        b = b.split('/')[-1]
        b = b.split('.')[0]
        G.add_edge(a, b)
        #print(a, b)
    return G


graph = reading('photo_all.txt')
print(list(nx.connected_components(graph)))