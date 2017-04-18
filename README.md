# kBestViterbi
Python code for doing k-Best or List Viterbi Decoding of a HMM

## viterbi(pi, A, O, observations)
A reference implementation of the Viterbi algorithm, robbed from here:-

http://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017

## exhaustive(pi, A, O, observations)
Exhaustively compute all possibilities for the HMM, robbed from here:-
http://people.eng.unimelb.edu.au/tcohn/comp90042/HMM.html

## kViterbiParallel(pi, a, b, obs, k)
Parallel List Viterbi Decoder to retain the top k scoring paths at each state at each time t in the time series.

Adapted from the outline in this paper:-
ieeexplore.ieee.org/iel1/26/12514/00577040.pdf

## kViterbiGraph(pi, a, b, obs, k)
Compute a k length Viterbi list by first converting the HMM into a NetworkX compatible DAG (Directed acyclic graph), 
converting to negative log-space then using Yen's algorithm to return the shortest paths, see this paper below.

https://arxiv.org/pdf/1412.5075.pdf

There are also some models for testing, namely the Wikipedia exam[ple and tcohn's example above.
