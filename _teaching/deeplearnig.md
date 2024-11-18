---
layout: page
title: Deep Learning
description: 
img: assets/img/Deep-learning.png
year: 2024
category: graduate
related_publications: false
toc:
    sidebar: left
back_link: '/teaching'
back_text: 'Courses Page'
---
<div class="course-description-box">
  <p>
    Deep Learning algorithms learn multi-level representations of data, with each level explaining the data in a hierarchical manner. Such algorithms have been effective at uncovering underlying structure in data, e.g., features to discriminate between classes. They have been successful in many artificial intelligence problems including image classification, speech recognition, and natural language processing.
  </p>
  <p>
    The course, which will be taught through lectures and projects, will cover the underlying theory, the range of applications to which it has been applied, and learning from very large data sets. The course will cover connectionist architectures commonly associated with deep learning, e.g., basic neural networks, convolutional neural networks, and recurrent neural networks. Methods to train and optimize the architectures and methods to perform effective inference with them will be the main focus. Students will be encouraged to use open-source software libraries such as PyTorch.
  </p>
  <p class="course-note">
    <strong>Pre-requisite:</strong> <a href="{{ '/teaching/machinelearning/' | relative_url }}">Introductory Machine Learning (ML)</a>. A course on Probabilistic Graphical Models (PGMs) is helpful but not necessary.
  </p>
</div>


## Course Logistics
**Course Instructor**: Jue Guo [C]
- *Research Area:* Optimization for machine learning, Adversarial Learning,
Continual Learning and Graph Learning
- Interested in participating in our research? Reach to me by [email](mailto:jueguo@buffalo.edu).
  
**Course Hours:** Session [C]; *Tuesday and Thursday 2:00PM-3:20PM*

**Office Hours:** 3:00pm - 4:00pm on Friday

---
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
---
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

<table align="center">
    <tr>
        <th>Percentage</th>
        <th>Letter Grade</th>
        <th>Percentage</th>
        <th>Letter Grade</th>
    </tr>
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
</table>

### Note on Logistics
- A week-ahead notice for mid-term, based on the pace of the course. 
- The logistic is <span style="color:red;">subject to change</span> based on the overall pace and the performance of the class.


---
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
          <a href="{{ 'assets/courses/deeplearning/week_1_2/Intro' | relative_url }}">Introduction, Preliminaries & Linear Neural Network</a>
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


