---
layout: page
title: CSE 431/531 Algorithm Analysis and Design
description: 
img: assets/img/algorithm.png
year: 2025
category: undergraduate/graduate
related_publications: false
toc:
    sidebar: left
back_link: '/teaching'
back_text: 'Courses Page'
enable_heading_styles: true
---

<div class="course-description-box">
    <p>
        Introduces basic elements of the design and analysis of algorithms. Topics include asymptotic notations and analysis, divide and conquer, greedy algorithms, dynamic programming, fundamental graph algorithms, NP-completeness, and approximation algorithms.
    </p>
    <p>
    For each topic, beside in-depth coverage, we discuss one or more representative problems and their algorithms. In addition to the design and analysis of algorithms, students are expected to gain substantial discrete mathematics problem solving skills essential for computer scientists and engineers.
    </p>
    <p class="course-note">
    <strong>Pre-requisite:</strong> You should have taken CSE250 (data structure) or similar courses before. We expect you to have certain levels of mathematical maturity: You should have basic understanding of calculus (e.g., limit, differentiation, integration) and linear algebra (e.g., matrix, vector space, linear transformation); You should be comfortable to read and write mathematical proofs, understanding common proof strategies (e.g., proof by induction, contradiction). We also expect you to have some programming experience: know what is a computer program, and be able to read and write code.
  </p>
</div>

## Course Logistics
**Course Instructor**: Jue Guo [C]
- *Research Area:* Optimization for machine learning, Adversarial Learning,
Continual Learning and Graph Learning
- Interested in participating in our research? Reach to me by [email](mailto:jueguo@buffalo.edu).
  
**Course Hours:** Session [C]; *Tuesday and Thursday 2:00PM-3:20PM*

**Office Hours:** 3:00pm - 4:00pm on Friday

## Course Outline
Check out the [course material](#lecture-notes) under lecture notes. 

<table class="styled-table">
  <thead>
    <tr>
      <th>Week(s)</th>
      <th>Topics Covered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Week 1 and Week 2</td>
      <td>Math, Machine Learning Review, and Linear Regression</td>
    </tr>
    <tr>
      <td>Week 3 and Week 4</td>
      <td>Review on Linear Regression, Softmax Regression, and MLP</td>
    </tr>
    <tr>
      <td>Week 5 and Week 6</td>
      <td>Optimization, CNN, and Efficient-Net Paper Reading</td>
    </tr>
    <tr>
      <td>Week 7 (One Class)</td>
      <td>Midterm (Coverage on Weeks 1, 2, 3, 4, 5)</td>
    </tr>
    <tr>
      <td>Week 8 and Week 9</td>
      <td>Recurrent Neural Networks and Paper Read on Transformer</td>
    </tr>
    <tr>
      <td>Week 10, Week 11, Week 12, and Week 13</td>
      <td>Graph Neural Network Paper Read</td>
    </tr>
    <tr>
      <td>Week 14 and Week 15</td>
      <td>Catch up Time on the Material if Needed<br>Final and Review</td>
    </tr>
  </tbody>
</table>


## Grading

The following is the outline of the grading: 

### Grading Components

We will have
- **Attendance:** 10 percent (Random Attendance Check)
- **Programming Assignment:** 30 percent (2 PA)
- **Midterm:** 30 percent
- **Final:** 30 percent 

### Grading Rubric

This course is **absolute** grading, meaning no curve, as there is a certain standard we need to uphold for students to have a good knowledge of deep learning.

<table class="styled-table">
  <thead>
    <tr>
      <th>Percentage</th>
      <th>Letter Grade</th>
      <th>Percentage</th>
      <th>Letter Grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>95-100</td>
      <td>A</td>
      <td>70-74</td>
      <td>C+</td>
    </tr>
    <tr>
      <td>90-94</td>
      <td>A-</td>
      <td>65-69</td>
      <td>C</td>
    </tr>
    <tr>
      <td>85-89</td>
      <td>B+</td>
      <td>60-64</td>
      <td>C-</td>
    </tr>
    <tr>
      <td>80-84</td>
      <td>B</td>
      <td>55-59</td>
      <td>D</td>
    </tr>
    <tr>
      <td>75-79</td>
      <td>B-</td>
      <td>0-54</td>
      <td>F</td>
    </tr>
  </tbody>
</table>


### Note on Logistics
- A week-ahead notice for mid-term, based on the pace of the course. 
- The logistic is <span style="color:red;">subject to change</span> based on the overall pace and the performance of the class.


## Lecture Notes
The notes are based on [Dive into Deep Learning](https://d2l.ai/). Throughout my teaching, I have noticed that students sometimes struggle with understanding the derivations in the textbook due to the omission of several steps. To address this, I have expanded the derivations and provided more detailed explanations.

Below are the lecture notes. Please note that these notes are updated regularly, so be sure to check back often for the latest updates.

<table class="styled-table">
  <thead>
    <tr>
      <th>Topic</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Introduction</td>
      <td>
        <ul>
          <li>
          <a href="{{ 'assets/courses/deeplearning/week_1_2/introduction' | relative_url }}">Introduction, Preliminaries & Linear Neural Network</a>
          </li>
          <li>
          <a href="{{ 'assets/courses/deeplearning/week_1_2/classification_convexity' | relative_url }}">Linear Neural Network for Classification & Start of Optimization</a>
          </li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>Optimization</td>
      <td>
        <ul>
          <li>
          <a href="{{ 'assets/courses/deeplearning/optimization/convexity_gd' | relative_url }}">Convexity & Gradient Descent</a>
          </li>
          <li>
          <a href="{{ 'assets/courses/deeplearning/optimization/stochastic_gd' | relative_url }}">Stochastic Gradient Descent</a>
          </li>
          <li>
          <a href="{{ 'assets/courses/deeplearning/optimization/optimization_algorithms' | relative_url }}">Optimization Algorithms</a>
          </li>
        </ul>
      </td>
    </tr>
    <tr>
    <td>DNN and CNN</td>
      <td>
        <ul>
          <li>
            <a href="{{ 'assets/courses/deeplearning/cnn/dnn_cnn' | relative_url }}">DNN and CNNs</a>
          </li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>RNN</td>
      <td>
      <ul>
        <li>
          <a href="{{ 'assets/courses/deeplearning/rnn/markov' | relative_url }}">Markov Models and \(n\)-gram</a>
        </li>
        <li>
          <a href="{{ 'assets/courses/deeplearning/rnn/vinilla_rnn' | relative_url }}">Recurrent Neural Networks</a>
        </li>
        <li>
          <a href="{{ 'assets/courses/deeplearning/rnn/modern_rnn' | relative_url }}">Modern Recurrent Neural Networks</a>
        </li>
      </ul>
      </td>
    </tr>
    <tr>
      <td>GNN</td>
      <td>
        <ul>
            <li>
            <a href="{{ '/assets/courses/deeplearning/gnnpapers/grl' | relative_url }}">Graph Representation Learning</a>
            </li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>