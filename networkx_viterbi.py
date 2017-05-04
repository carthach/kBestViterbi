import networkx as nx
import numpy as np
from simple_paths_with_costs import * #Our custom simple_paths
import heapq
import itertools

#Create a NetworkX compatible directed graph that represents a HMM
def createPrunedViterbiGraphWithCosts(a, b, topK, weights=(1.0, 1.0)):
    nObs = len(b)
    nStates = len(a)

    #Number of nodes in the graph
    nNodes = nObs*nStates

    #Create a directed graph
    G = nx.DiGraph()

    #Create all the necessary nodes
    for n in range(0, nNodes):
        G.add_node(n)

    #Create the start and dummy end nodes
    G.add_node(-1)
    G.add_node(nNodes)

    targetCostsTotalWeight = weights[0]
    concatCostsTotalWeight = weights[1]

    from time import time

    t0 = time()



    #Add the weights for the start node
    for i in range(nStates):
        w = b[0, i]
        G.add_edge(-1, i, weight=w)
        G.add_edge((nNodes-1)-i, nNodes, weight=1.0)

    for t in range(0, nNodes-nStates, nStates):
        i_offset = t
        j_offset = t + nStates

        real_t = t / nStates + 1

        for i in range(nStates):
            i_idx = i_offset + i

            h = []

            for j in range(nStates):
                j_idx = j_offset + j

                w = (targetCostsTotalWeight * b[real_t, j]) + (concatCostsTotalWeight * a[i, j])

                heapq.heappush(h, (w, i_idx, j_idx))

            for k in range(5):
                e = h.pop()
                G.add_edge(e[1], e[2], weight=e[0])

    t1 = time()



    #This for the mod operation in the shortest path computation
    G.graph["nStates"] = nStates

    return G

#Create a NetworkX compatible directed graph that represents a HMM
def createViterbiGraphWithCosts(a, b, weights=(1.0, 1.0)):
    nObs = len(b)
    nStates = len(a)

    #Number of nodes in the graph
    nNodes = nObs*nStates

    #Create a directed graph
    G = nx.DiGraph()

    #Create all the necessary nodes
    for n in range(0, nNodes):
        G.add_node(n)

    #Create the start and dummy end nodes
    G.add_node(-1)
    G.add_node(nNodes)

    targetCostsTotalWeight = weights[0]
    concatCostsTotalWeight = weights[1]

    from time import time

    t0 = time()

    #Add the weights for the start node
    for i in range(nStates):
        w = b[0, i]
        G.add_edge(-1, i, weight=w)
        G.add_edge((nNodes-1)-i, nNodes, weight=1.0)

    for t in range(0, nNodes-nStates, nStates):
        i_offset = t
        j_offset = t + nStates

        real_t = t / nStates + 1

        for i in range(nStates):
            i_idx = i_offset + i

            for j in range(nStates):
                j_idx = j_offset + j

                w = (targetCostsTotalWeight * b[real_t, j]) + (concatCostsTotalWeight * a[i, j])

                G.add_edge(i_idx ,j_idx, weight=w)

    t1 = time()

    #This for the mod operation in the shortest path computation
    G.graph["nStates"] = nStates

    return G

#Create a NetworkX compatible directed graph that represents a HMM
def createViterbiGraph(pi, a, b, obs):
    nObs = len(obs)
    nStates = len(a)

    #Number of nodes in the graph
    nNodes = nObs*nStates

    #Create a directed graph
    G = nx.DiGraph()

    #Create all the necessary nodes
    for n in range(0, nNodes):
        G.add_node(n)

    #Create the start and dummy end nodes
    G.add_node(-1)
    G.add_node(nNodes)

    #Add the weights for the start node
    for i in range(nStates):
        w = pi[i] * b[i, obs[0]]
        G.add_edge(-1, i, weight=w)
        G.add_edge((nNodes-1)-i, nNodes, weight=1.0)

    for t in range(0, nNodes-nStates, nStates):
        i_offset = t
        j_offset = t + nStates

        for i in range(nStates):
            i_idx = i_offset + i

            for j in range(nStates):
                j_idx = j_offset + j

                real_t = t / nStates + 1

                w = a[i, j] * b[j, obs[real_t]]

                G.add_edge(i_idx ,j_idx, weight=w)


    #This for the mod operation in the shortest path computation
    G.graph["nStates"] = nStates

    return G

def shortestPaths(G, topK, negativeLogSpace = True):
    if negativeLogSpace:
        for e in G.edges(data=True):
            e[2]["weight"] = np.log(e[2]["weight"])
            e[2]["weight"] = -e[2]["weight"]

    def k_shortest_paths(G, source, target, k, weight=None):
        from itertools import islice
        return list(islice(shortest_simple_paths_with_costs(G, source, target, weight=weight, topK=k), k))

    #Some params
    source = -1
    target = len(G) - 2
    nStates = G.graph["nStates"]

    pathsAndCosts = k_shortest_paths(G, source, target, topK, weight="weight")


    #Antilog and negate to get the correct probabilities
    if negativeLogSpace:
        pathsAndCosts = [(p[0], np.exp(-p[1])) for p in pathsAndCosts]

    # Do a mod by number of states, and remove the dummy nodes
    pathsAndCosts = [(np.mod(p[0][1:-1], nStates), p[1]) for p in pathsAndCosts]

    return pathsAndCosts

#Use Graph techniques for LVA decoding
def kViterbiGraph(pi, a, b, obs, topK):
    G = createViterbiGraph(pi, a, b, obs)
    paths = shortestPaths(G, topK)

    return paths

def kViterbiGraphWithCosts(a, b, topK, weights=(1.0, 1.0)):
    G = createViterbiGraphWithCosts(a, b, weights=weights)
    # G = createPrunedViterbiGraphWithCosts(a, b, topK, weights=weights)

    from time import time

    t0 = time()

    paths = shortestPaths(G, topK, negativeLogSpace=False)

    t1 = time()

    print 'function ITSELF takes %f' % (t1 - t0)

    return paths