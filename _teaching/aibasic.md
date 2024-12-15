---
layout: page
title: Basics of Artificial Intelligence
description: 
img: assets/img/aibasic.jpg
year: 2024
category: undergraduate/graduate
related_publications: false
toc:
    sidebar: left
back_link: '/teaching'
back_text: 'Courses Page'
enable_heading_styles: true
---
	
<div class="course-description-box">
  <p>This course is intended for SEAS Engineering graduate students who are interested in understanding the fundamental issues, challenges, and techniques that are associated with recent advances in Artificial Intelligence (AI). The course will discuss the history and properties of basic AI systems, including neural networks, machine learning, and data science, and how to build a basic machine learning and AI project, including data scrapping, data processing, etc. We will discuss the challenges of bias, security, privacy, explainability, ethical issues, and the use of context.</p>
  <p>We will learn about AI's use in applications such as image processing and computer vision, natural language processing, recommendation systems, and gaming. The course is supported by a primer on the use of Python to support homework and projects related to machine learning. The course will be a combination of lectures, discussions, activities, and projects that will prepare students without a computer science background to study and apply artificial intelligence tools and applications in a variety of different domains.</p>
  <p class="course-note">
  <strong>Note:</strong> The course is <strong>NOT</strong> intended for students who have a reasonable background in machine learning, computer science, or Python programming. Undergraduates who wish to take this course and petition for credit need to inquire with the <a href="https://engineering.buffalo.edu/home/academics/grad/contact.html">SEAS graduate office</a>.
</p>

</div>

## Course Logistics
**Course Instructor**: Jue Guo 
- *Research Area:* Optimization for machine learning, Adversarial Learning,
Continual Learning and Graph Learning
  
**Course Hours:** EAS  510LEC - AI1 ; *MoWeFr 2:00PM - 2:50PM, Nsc 205*

**Office Hours:** 3:00pm - 4:00pm on Friday


## Course Outline
Check out the [course material](#lecture-notes) under lecture notes. 

<table class="styled-table">
  <thead>
    <tr>
      <th>Week(s) & Approx. Dates</th>
      <th>Topics Covered</th>
    </tr>
  </thead>
  <tbody>
    <!-- Classes start Wednesday, Jan 22, 2025 -->
    <tr>
      <td>Week 1 and Week 2 (Jan 22 – Feb 4)</td>
      <td>PyTorch Fundamentals, PyTorch Workflow Fundamentals</td>
    </tr>
    <tr>
      <td>Week 3 and Week 4 (Feb 5 – Feb 18)</td>
      <td>PyTorch Neural Network Classification & Computer Vision</td>
    </tr>
    <tr>
      <td>Week 5 and Week 6 (Feb 19 – Mar 3)</td>
      <td>Custom Datasets, Going Modular, and Transfer Learning</td>
    </tr>
    <tr>
      <td>Week 7 (Mar 4 – Mar 10)</td>
      <td>Midterm (Coverage: Weeks 1–5) and Catch Up</td>
    </tr>
    <tr>
      <td>Week 8 (Mar 11 – Mar 16)</td>
      <td>Experiment Tracking & Paper Replicating (start)</td>
    </tr>
    <!-- Spring Recess: Mar 17 – Mar 22, No Classes -->
    <tr>
      <td>Mar 17 – Mar 22</td>
      <td><strong>Spring Recess (No Classes)</strong></td>
    </tr>
    <!-- Classes resume Monday, Mar 24 -->
    <tr>
      <td>Week 9 (Mar 24 – Mar 29)</td>
      <td>Experiment Tracking & Paper Replicating (continued)</td>
    </tr>
    <tr>
      <td>Week 10, 11, 12, and 13 (Mar 30 – Apr 26)</td>
      <td>Model Deployment</td>
    </tr>
    <tr>
      <td>Week 14 and Week 15 (Apr 27 – May 6)</td>
      <td>Catch Up Time on the Material if Needed</td>
    </tr>
  </tbody>
</table>


## Grading

The following is the outline of the grading: 

### Grading Components

As we progress with the course, students are **suggested** to spend weekends to finish the following course on freeCodeCamp: 

- [Scientific Computing with Python](https://www.freecodecamp.org/learn/scientific-computing-with-python/)
- [Data Analysis with Python](https://www.freecodecamp.org/learn/data-analysis-with-python/)
- [College Algebra with Python](https://www.freecodecamp.org/learn/college-algebra-with-python/)

By finishing these courses and getting a certification will allow you to get a **15pts** bonus for this course. These courses are extremely easy and should help you warm up and not be afraid of coding.

We will have
- **Attendance:** 10 percent (Random Pop Quiz)
- **Programming Assignment:** 30 percent (2 PA)
- **Midterm:** 30 percent
- **Final Project:** 30 percent 

### Grading Rubric

The final grade will be determined based on the *overall performance* of the class, taking into consideration all relevant assessments and contributions. 
- The instructor reserves the right to make **final** decisions regarding grades. 
- Please note that excuses for missed work or poor performance, such as personal conflicts or minor inconveniences, will not be considered unless exceptional and documented circumstances arise.
 

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
The lecture notes are based on [PyTorch documentation](https://pytorch.org/) and a variety of other resources related to PyTorch. They aim to provide a comprehensive and accessible explanation of key concepts, offering additional insights and examples to enhance understanding.

<table class="styled-table">
  <thead>
    <tr>
      <th>Week(s)</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Week 1</td>
      <td>
      <ul> 
      <li><a href="{{ 'assets/courses/basicai/pytorch_fundamentals/' | relative_url }}">PyTorch Fundamentals</a></li>
      </ul>
      </td>
    </tr>
  </tbody>
</table>

