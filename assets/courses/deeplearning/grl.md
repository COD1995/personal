---
layout: page
title: Graph Represeantion Learning
description: 
related_publications: false
toc:
    sidebar: left
---

## Reading List

- [A Survey on Graph Representation Learning Methods](https://arxiv.org/abs/2204.01855)

## Overview

Graph representation learning aims to create graph representation vectors that accurately capture the structure and features of large graphs, which is crucial for downstream tasks like *node classification, link prediction, and anomaly detection*. The field has seen significant growth, with methods divided into *traditional graph embedding techniques* and *graph neural network (GNN)-based approaches*. These techniques apply to both *static (fixed) and dynamic (evolving) graphs*. 

## Introduction

Graphs are powerful data structure (social networks, financial transactions and biological networks).
- social networks, people are the nodes and thir friendships constitute the edges.
- financial transactions, the nodes and edges could be people and their money transactions.

How to represent graphs? **Adjacency matrix**, however, dimensionality of the adjacency matrix is often very high, and feature extraction based methods are time consuming and may not represent all the necessary information in the graphs.

*Graph embedding methods* have been very successful in graph representation. These methods project the graph elements (such as nodes, edges and subgraphs) to a lower dimensional space and preserve the properties of graphs.
- traditional graph embedding
  - random walks, factorization embedding methods and non-GNN based deep learning&#8594; static graph and dynamic graph
- GNN based graph embedding methods; node embeddings are obtained by aggregating the embeddings of the node's neighbors.
  - early works based on rnn, later convolutional graph neural nets were developed based on the convolution operation
  - spatial-temporal GNNs and dynamic GNNs leverage the strengths of GNNs in evolving networks.
  
What we will cover? 1. Basic knowledge on Graphs, 2. Traditional node embedding methods for static and dynamic graphs, 3. Static, Spatial-temporal and dynamic GNN, 4. limitation of GNNs.
## Graphs
<div class="definition-box">
<b>DEFINITION 1.</b> Formally, a graph \(G\) is defined as a tuple \(G=(V, E)\) where \(V=\left\{v_{0}, v_{1}, \ldots, v_{n}\right\}\) is the set of \(n\) nodes/vertices
and \(E=\left\{e_{0}, e_{1}, \ldots, e_{m}\right\} \subseteq V \times V\) is the set of \(m\) edges/links of \(G\), where an edge connects two vertices.
</div>

A graph can be directed or undirected. In a directed graph, an edge $$e_{k}=\left(v_{i}, v_{j}\right)$$ has a direction with $$v_{i}$$ being the starting vertex and $$v_{j}$$ the ending vertex. Graphs can be represented by their adjacency, degree and Laplacian matrices, which are defined as follows:

<div class ="definition-box">
<b>DEFINITION 2.</b> The adjacency matrix \(A\) of a graph \(G\) with \(n\) vertices is an \(n \times n\) matrix, where an element \(a_{i j}\) in the
matrix equals to 1 if there is an edge between node pair \(v_{i}\) and \(v_{j}\) and is 0 otherwise. An adjacency matrix can be weighted
in which the value of an element represents the weight (such as importance) of the edge it represents.
</div>

<div class ="definition-box">
<b>DEFINITION 3.</b> The degree matrix \(D\) of a graph \(G\) with \(n\) vertices is an \(n \times n\) diagonal matrix, where an element \(d_{i i}\) is
the degree of node \(v_{i}\) for \(i=\{1, \ldots, n\}\) and all other \(d_{i j}=0\). In undirected graphs, where edges have no direction, the degree
of a node refers to the number of edges attached to that node. For directed graphs, the degree of a node can be the number of
incoming or outgoing edges of that node, resulting in an in-degree or out-degree matrix, respectively.
</div>

<div class ="definition-box">
<b>DEFINITION 4.</b> The Laplacian matrix \(L\) of a graph \(G\) with \(n\) vertices is an \(n \times n\) matrix, defined as \(L=D-A\), where \(D\) and \(A\) are \(G\) 's degree and adjacency matrix, respectively.
</div>

### Graph Embedding

In order to use graphs in downstream machine learning and data mining applications, graphs and their entities such as nodes and edges need to be represented using numerical features. 
  - graph embedding methodds have been proposed, which study the issue of automatically generating representation vectors for the graphs.
  - these methods formulate the graph representation learning as a machine learning task and generate embedding vectors leveraging the structure and properties of the graph as input data.

Graph embedding techniques include node, edge and subgraph embedding techniques, which are defined as follows

<div class ="definition-box">
<b>Definition 5.</b> (Node embedding). Let \(G=(V, E)\) be a graph, where \(V\) and \(E\) are the set of nodes and the set of edges of the graph, respectively. Node embedding learns a mapping function \(f: v_{i} \rightarrow \mathbb{R}^{d}\) that encodes each graph's node \(v_{i}\) into a
low dimensional vector of dimensiond such that \(d<<|V|\) and the similarities between nodes in the graph are preserved in the embedding space.
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/gnn_survey/fig1.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Fig. 1. The graph on the left hand side consists of 6 nodes \(\{a, b, c, d, e, i\}\) and 8 edges. Graph embedding methods map each node of
the graph into an embedding vector with dimension \(d\). For the demonstration purpose, the node \(a\) is embedded into an embedding
vector \(z_{a}\) of dimension 4 with given values.
</div>
<div class ="definition-box">
<b>Definition 6.</b> (Edge embedding). Let \(G=(V, E)\) be a graph, where \(V\) and \(E\) are the set of nodes and the set of edges of
the graph, respectively. Edge embedding converts each edge of \(G\) into a low dimensional vector of dimensiond such that
\(d<<|V|\) and the similarities between edges in the graph are preserved in the embedding space.
</div>
