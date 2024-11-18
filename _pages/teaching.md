---
layout: page
title: courses
permalink: /teaching/
description: A collection of courses I have taught, complete with resources and materials, covering key AI-focused courses within the Computer Science and Engineering (CSE) department.
nav: true
nav_order: 3
display_categories: [undergraduate/graduate , graduate]
horizontal: false
---

<div class="project">
{% if site.enable_teaching_categories and page.display_categories %}
  <!-- Display categorized courses -->
  {% for category in page.display_categories %}
  <a id="{{ category }}" href=".#{{ category }}">
    <h2 class="teaching-category">{{ category }}</h2>
  </a>
  {% assign categorized_courses = site.teaching | where: "category", category %}
  {% assign sorted_courses = categorized_courses | sort: "year" | reverse %}
  <!-- Generate cards for each course -->
  {% if page.horizontal %}
  <div class="container">
    <div class="row row-cols-1 row-cols-md-2">
    {% for course in sorted_courses %}
      {% include course_horizontal.liquid %}
    {% endfor %}
    </div>
  </div>
  {% else %}
  <div class="row row-cols-1 row-cols-md-3">
    {% for course in sorted_courses %}
      {% include course.liquid %}
    {% endfor %}
  </div>
  {% endif %}
  {% endfor %}

{% else %}

<!-- Display courses without categories -->
{% assign sorted_courses = site.teaching | sort: "year" | reverse %}

<!-- Generate cards for each course -->
{% if page.horizontal %}
  <div class="container">
    <div class="row row-cols-1 row-cols-md-2">
    {% for course in sorted_courses %}
      {% include course_horizontal.liquid %}
    {% endfor %}
    </div>
  </div>
{% else %}
  <div class="row row-cols-1 row-cols-md-3">
    {% for course in sorted_courses %}
      {% include course.liquid %}
    {% endfor %}
  </div>
{% endif %}
{% endif %}
</div>
