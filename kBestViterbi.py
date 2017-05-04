import numpy as np
import pandas as pd
import networkx as nx
import networkx_viterbi as nxv
import heapq
import itertools

#Exhaustive search for verification
def exhaustiveWithCosts(a, b):
    M = len(a)
    S = len(b)

    scores = []

    # track the running best sequence and its score
    best = (None, float('inf'))
    # loop over the cartesian product of |states|^M
    for ss in itertools.product(range(S), repeat=M):
        score = b[0, ss[0]]

        for i in range(1, M):
            score += a[ss[i - 1], ss[i]] + b[i, ss[0]]

        # update the running best
        if score < best[1]:
            best = (ss, score)

        scores.append((score, ss))

        print ss

    return sorted(scores)

#Exhaustive search for verification
def exhaustive(pi, A, O, observations):
    M = len(observations)
    S = pi.shape[0]

    scores = []

    # track the running best sequence and its score
    best = (None, float('-inf'))
    # loop over the cartesian product of |states|^M
    for ss in itertools.product(range(S), repeat=M):
        # score the state sequence
        score = pi[ss[0]] * O[ss[0], observations[0]]
        for i in range(1, M):
            score *= A[ss[i - 1], ss[i]] * O[ss[i], observations[i]]
        # update the running best
        if score > best[1]:
            best = (ss, score)

        scores.append((score, ss))

    return best, scores

#Classic Parallel LVA Decoder using heaps and rankings
def kViterbiParallelWithCosts(a, b, topK, weights=(1.0, 1.0)):
    if topK == 1:
        return ([viterbiWithCosts(a, b, weights)], None, None, None)

    nStates = np.shape(a)[0]
    T = np.shape(b)[0]

    assert (topK <= np.max([np.power(nStates, T), np.inf])), "k < nStates ^ topK"

    # delta --> highest probability of any path that reaches point i
    delta = np.zeros((T, nStates, topK))

    # phi --> argmax
    phi = np.zeros((T, nStates, topK), int)

    #The ranking of multiple paths through a state
    rank = np.zeros((T, nStates, topK), int)

    # for k in range(K):
    for i in range(nStates):
        delta[0, i, 0] = b[0, i]
        phi[0, i, 0] = i

        #Set the other options to 0 initially
        for k in range(1, topK):
            delta[0, i, k] = np.inf
            phi[0, i, k] = i

    targetCostsTotalWeight = weights[0]
    concatCostsTotalWeight = weights[1]

    #Go forward calculating top k scoring paths
    # for each state s1 from previous state s2 at time step t
    for t in range(1, T):
        for s1 in range(nStates):

            h = []

            for s2 in range(nStates):
                # y = np.sort(delta[t-1, s2, :] * a[s2, s1] * b[s1, obs[t]])

                for k in range(topK):
                    targetCost = b[t, s1]
                    concatCost = a[s2, s1]

                    totalCost = delta[t - 1, s2, k] + (targetCostsTotalWeight * targetCost) + (concatCostsTotalWeight * concatCost)

                    state = s2

                    # Push the probability and state that led to it
                    heapq.heappush(h, (totalCost, state))

            #Get the sorted list
            h_sorted = [heapq.heappop(h) for i in range(len(h))]
            # h_sorted.reverse()

            #We need to keep a ranking if a path crosses a state more than once
            rankDict = dict()

            #Retain the top k scoring paths and their phi and rankings
            for k in range(0, topK):
                delta[t, s1, k] = h_sorted[k][0]
                phi[t, s1, k] = h_sorted[k][1]

                state = h_sorted[k][1]

                if state in rankDict:
                    rankDict[state] = rankDict[state] + 1
                else:
                    rankDict[state] = 0

                rank[t, s1, k] = rankDict[state]

    # Put all the last items on the stack
    h = []

    #Get all the topK from all the states
    for s1 in range(nStates):
        for k in range(topK):
            prob = delta[T - 1, s1, k]

            #Sort by the probability, but retain what state it came from and the k
            heapq.heappush(h, (prob, s1, k))

    #Then get sorted by the probability including its state and topK
    h_sorted = [heapq.heappop(h) for i in range(len(h))]
    # h_sorted.reverse()

    # init blank path
    path = np.zeros((topK, T), int)
    path_probs = np.zeros((topK, T), float)

    #Now backtrace for k and each time step
    for k in range(topK):
        #The maximum probability and the state it came from
        max_prob = h_sorted[k][0]
        state = h_sorted[k][1]
        rankK = h_sorted[k][2]

        #Assign to output arrays
        path_probs[k][-1] = max_prob
        path[k][-1] = state

        #Then from t down to 0 store the correct sequence for t+1
        for t in range(T - 2, -1, -1):
            #The next state and its rank
            nextState = path[k][t+1]

            #Get the new state
            p = phi[t+1][nextState][rankK]

            #Pop into output array
            path[k][t] = p

            #Get the correct ranking for the next phi
            rankK = rank[t + 1][nextState][rankK]

    return path, path_probs, delta, phi

#Classic Parallel LVA Decoder using heaps and rankings
def kViterbiParallel(pi, a, b, obs, topK):
    if topK == 1:
        return viterbi(pi, a, b, obs)

    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]

    assert (topK <= np.power(nStates, T)), "k < nStates ^ topK"

    # delta --> highest probability of any path that reaches point i
    delta = np.zeros((T, nStates, topK))

    # phi --> argmax
    phi = np.zeros((T, nStates, topK), int)

    #The ranking of multiple paths through a state
    rank = np.zeros((T, nStates, topK), int)

    # for k in range(K):
    for i in range(nStates):
        delta[0, i, 0] = pi[i] * b[i, obs[0]]
        phi[0, i, 0] = i

        #Set the other options to 0 initially
        for k in range(1, topK):
            delta[0, i, k] = 0.0
            phi[0, i, k] = i

    #Go forward calculating top k scoring paths
    # for each state s1 from previous state s2 at time step t
    for t in range(1, T):
        for s1 in range(nStates):

            h = []

            for s2 in range(nStates):
                # y = np.sort(delta[t-1, s2, :] * a[s2, s1] * b[s1, obs[t]])

                for k in range(topK):
                    prob = delta[t - 1, s2, k] * a[s2, s1] * b[s1, obs[t]]
                    # y_arg = phi[t-1, s2, k]

                    state = s2

                    # Push the probability and state that led to it
                    heapq.heappush(h, (prob, state))

            #Get the sorted list
            h_sorted = [heapq.heappop(h) for i in range(len(h))]
            h_sorted.reverse()

            #We need to keep a ranking if a path crosses a state more than once
            rankDict = dict()

            #Retain the top k scoring paths and their phi and rankings
            for k in range(0, topK):
                delta[t, s1, k] = h_sorted[k][0]
                phi[t, s1, k] = h_sorted[k][1]

                state = h_sorted[k][1]

                if state in rankDict:
                    rankDict[state] = rankDict[state] + 1
                else:
                    rankDict[state] = 0

                rank[t, s1, k] = rankDict[state]

    # Put all the last items on the stack
    h = []

    #Get all the topK from all the states
    for s1 in range(nStates):
        for k in range(topK):
            prob = delta[T - 1, s1, k]

            #Sort by the probability, but retain what state it came from and the k
            heapq.heappush(h, (prob, s1, k))

    #Then get sorted by the probability including its state and topK
    h_sorted = [heapq.heappop(h) for i in range(len(h))]
    h_sorted.reverse()

    # init blank path
    path = np.zeros((topK, T), int)
    path_probs = np.zeros((topK, T), float)

    #Now backtrace for k and each time step
    for k in range(topK):
        #The maximum probability and the state it came from
        max_prob = h_sorted[k][0]
        state = h_sorted[k][1]
        rankK = h_sorted[k][2]

        #Assign to output arrays
        path_probs[k][-1] = max_prob
        path[k][-1] = state

        #Then from t down to 0 store the correct sequence for t+1
        for t in range(T - 2, -1, -1):
            #The next state and its rank
            nextState = path[k][t+1]

            #Get the new state
            p = phi[t+1][nextState][rankK]

            #Pop into output array
            path[k][t] = p

            #Get the correct ranking for the next phi
            rankK = rank[t + 1][nextState][rankK]

    return path, path_probs, delta, phi


# define Viterbi algorithm for shortest path
# code adapted from Stephen Marsland's, Machine Learning An Algorthmic Perspective, Vol. 2
# https://github.com/alexsosn/MarslandMLAlgo/blob/master/Ch16/HMM.py

def viterbiWithCosts(a, b, weights=(1.0, 1.0)):
    nStates = np.shape(a)[0]
    T = np.shape(b)[0]

    # init blank path
    path = np.zeros(T, int)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T), float)
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T), int)

    # init delta and phi
    delta[:, 0] = b[0, :]
    phi[:, 0] = np.arange(nStates)

    targetCostsTotalWeight = weights[0]
    concatCostsTotalWeight = weights[1]

    print('\nStart Walk Forward\n')
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            targetCosts = b[t, s]
            concatCosts = a[:, s]

            totalCosts = (targetCostsTotalWeight * targetCosts) + (concatCostsTotalWeight * concatCosts) + delta[:, t-1]

            delta[s, t] = np.min(totalCosts)
            phi[s, t] = np.argmin(totalCosts)
            print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))

    # find optimal path
    print('-' * 50)
    print('Start Backtrace\n')
    path[T - 1] = np.argmin(delta[:, T - 1])
    # p('init path\n    t={} path[{}-1]={}\n'.format(T-1, T, path[T-1]))
    for t in range(T - 2, -1, -1):
        path[t] = phi[path[t + 1], [t + 1]]
        # p(' '*4 + 't={t}, path[{t}+1]={path}, [{t}+1]={i}'.format(t=t, path=path[t+1], i=[t+1]))
        print('path[{}] = {}'.format(t+1, path[t+1]))

    print('path[{}] = {}'.format(t, path[t]))

    max_prob = np.min(delta[:, T-1])

    return path, delta, phi, max_prob

# define Viterbi algorithm for shortest path
# code adapted from Stephen Marsland's, Machine Learning An Algorthmic Perspective, Vol. 2
# https://github.com/alexsosn/MarslandMLAlgo/blob/master/Ch16/HMM.py

def viterbi(pi, a, b, obs):
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]

    # init blank path
    path = np.zeros(T, int)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T), float)
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T), int)

    # init delta and phi
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    print('\nStart Walk Forward\n')
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t - 1] * a[:, s]) * b[s, obs[t]]
            phi[s, t] = np.argmax(delta[:, t - 1] * a[:, s])
            print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))

    # find optimal path
    print('-' * 50)
    print('Start Backtrace\n')
    path[T - 1] = np.argmax(delta[:, T - 1])
    # p('init path\n    t={} path[{}-1]={}\n'.format(T-1, T, path[T-1]))
    for t in range(T - 2, -1, -1):
        path[t] = phi[path[t + 1], [t + 1]]
        # p(' '*4 + 't={t}, path[{t}+1]={path}, [{t}+1]={i}'.format(t=t, path=path[t+1], i=[t+1]))
        print('path[{}] = {}'.format(t+1, path[t+1]))

    print('path[{}] = {}'.format(t, path[t]))

    max_prob = np.max(delta[:, T-1])

    return path, delta, phi, max_prob

if __name__ == '__main__':
    #some models to try out
    from model_wiki import *
    # from model_tcohn import *

    path, delta, phi, max_prob = viterbi(pi, a, b, obs)
    path, delta, phi, max_prob = kViterbiParallel(pi, a, b, obs, 8)
    paths = nxv.kViterbiGraph(pi, a, b, obs, 8)