---
layout: page
title: Graph Represeantion Learning
description: 
related_publications: false
toc:
    sidebar: right
back_link: '/teaching/deeplearning'
back_text: 'Deep Learning'
---

## Reading List
- [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)
  - *Easy to read and understand 10/10*
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
        {% include figure.liquid loading="eager" path="assets/img/gnn_survey/fig1.png" class="img-fluid rounded" %}
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
An embedding vector for the edge $$\left(v_{i}, v_{j}\right)$$ can be obtained by applying a binary operation such as hammard product, mean, weighted-L1 and weighted-L2 on the two node embedding vectors $$z_{i}$$ and $$z_{j}$$.
<div class ="definition-box">
<b>Definition 7.</b> (Subgraph embedding). Let \(G=(V, E)\) be a graph. Subgraph embedding techniques in machine learning convert a subgraph of \(G\) into a low dimensional vector of dimensiond such that \(d<<|V|\) and the similarities between subgraphs are preserved in the embedding space.
</div>
A subgraph embedding vector is usually created by aggregating the embeddings of the nodes in the subgraph using aggregators such as a mean operator. Almost all the graph embedding techniques developed so far are node embedding techniques.

## Graph Embedding Applications
**Node Classification.** Node classification task assigns a label to the nodes in the test dataset. This task has many applications in different domains. For instance, in social networks, a person’s political affiliation can be predicted based on his friends in the network. In node classification, each instance in the training dataset is the node embedding vector and the label of the instance is the node label. Different regular classification methods such as Logistic Regression and Random Forests can be trained on the training dataset and generate the node classification scores for the test data. Similarly, Graph classification can be performed using graph embedding vectors.

**Link Prediction.** Link prediction is one of the important applications of node embedding methods. It predicts the likelihood of an edge formation between two nodes. Examples of this task include recommending friends in social networks and finding biological connections in biological networks. Link prediction can be formulated as a classification task that assigns a label for edges. Edge label 1 means that an edge is likely to be created between two nodes and the label is 0 otherwise. For the training step, a sample training set is generated using positive and negative samples. Positive samples are the edges the exist in the graph. Negative samples are the edges that do not exist and their representation vector can be generated using the node vectors. Similar to node classification, any classification method can be trained on the training set and predict the edge label for test edge instances.

**Anomaly Detection.** Anomaly detection is another application of node embedding methods. The goal of anomaly detection is to detect the nodes, edges, or graphs that are anomalous the time that anomaly occurs. Anomalous nodes and graphs deviate from normal behavior. 
- For instance, in banks’ transaction networks, people who suddenly send or receive large amounts of money or create lots of connections with other people could be potential anomalous nodes.
  
An anomaly detection task can be formulated as a classification task such that each instance in the dataset is the node representation and the instance label is 0 if the node is normal and 1 if the node is anomalous. This formulation needs that we have a dataset with true node labels. One of the issues in anomaly detection is the lack of datasets with true labels.
- An alleviation to this issue in the literature is generating synthetic datasets that model the behaviors of real world datasets.

Another way to formulate the anomaly detection problem, especially in dynamic graphs, is viewing the problem as a change detection task. In order to detect the changes in the graph, one way is to compute the distance between the graph representation vectors at consecutive times. The time points that the value of this difference is far from the previous normal values, a potential anomaly has occurred.

Other graph embedding techniques includes **graph clustering** and **visualization**.

