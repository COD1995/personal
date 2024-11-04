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
- traditional graph 


