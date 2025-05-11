---
layout: about
title: about
permalink: /
subtitle: "<i>\"We choose to go to the moon in this decade and do the other things, not because they are easy, but because they are hard.\"</i> – John F. Kennedy"

profile:
  align: right
  image: prof_pic.jpg
  image_circular: false # crops the image to make it circular
  more_info: >
    <p>301 Davis Hall Desk 12</p>
    <p>12 Capen Hall</p>
    <p>Buffalo, New York 14260-1660</p>
    
selected_papers: false # includes a list of papers marked as "selected={true}"
social: false # includes social icons at the bottom of the page
---

Jue Guo is a Ph.D. candidate in Computer Science at the University at Buffalo, advised by [Prof. A. Erdem Sariyüce](https://sariyuce.com/). His teaching portfolio spans core areas in machine learning, with a focus on deep learning and pattern recognition, where he has guided and inspired the next generation of computer scientists.

His research explores advanced machine learning methodologies across a range of domains, including image classification, natural language processing, continual learning, and adversarial machine learning.

Technically, Jue is proficient in Python, PyTorch, and JavaScript, and adept at bridging theory with practice—developing scalable, research-grade models while maintaining strong algorithmic rigor. His ability to integrate complex ML frameworks with domain-specific challenges positions him as a creative and forward-looking contributor to the field.

<!-- teaching -->
<hr class="divider" />

<div class="courses-container">
  <h2 class="courses-title">Current Courses I’m Teaching</h2>

  <div class="courses-filter">
    <label for="year-select">Filter by Year:</label>
    <select id="year-select">
      <option value="all">All Years</option>
      <option value="2025" selected>2025</option>
      <option value="2024">2024</option>
      <option value="2023">2023</option>
    </select>
  </div>

  <table class="courses-table">
    <tbody>
      <!-- 2025 -->
      <tr data-year="2025">
        <td class="date">Jan 22, 2025</td>
        <td class="course">
          <a href="{{ '/teaching/aibasic' | relative_url }}">
            Start teaching Basics of Artificial Intelligence
          </a>
        </td>
      </tr>
      <!-- 2024 -->
      <tr data-year="2024">
        <td class="date">Aug 26, 2024</td>
        <td class="course">
          <a href="{{ '/teaching/deeplearning' | relative_url }}">
            Start teaching Deep Learning
          </a>
        </td>
      </tr>
      <tr data-year="2024">
        <td class="date">Jun 24, 2024</td>
        <td class="course">
          <a href="{{ '/teaching/pattern' | relative_url }}">
            Start teaching Intro to Pattern Recognition
          </a>
        </td>
      </tr>
      <tr data-year="2024">
        <td class="date">Jan 24, 2024</td>
        <td class="course">
          <a href="{{ '/teaching/machinelearning' | relative_url }}">
            Start teaching Machine Learning
          </a>
        </td>
      </tr>
      <!-- 2023 -->
      <tr data-year="2023">
        <td class="date">Aug 28, 2023</td>
        <td class="course">
          <a href="{{ '/teaching/deeplearning' | relative_url }}">
            Start teaching Deep Learning
          </a>
        </td>
      </tr>
      <tr data-year="2023">
        <td class="date">Jun 26, 2023</td>
        <td class="course">
          <a href="{{ '/teaching/pattern' | relative_url }}">
            Start teaching Intro to Pattern Recognition
          </a>
        </td>
      </tr>
    </tbody>
  </table>
</div>

<style>
  .divider {
    margin: 8rem 0 1rem;
    border: none;
    border-top: 1px solid var(--global-divider-color);
  }
  .courses-container {
    max-width: 700px;
    margin: 0 auto;
    padding: 0 1rem;
  }
  .courses-title {
    font-size: 1.75rem;
    font-weight: 500;
    text-align: center;
    margin-bottom: 1.5rem;
  }
  .courses-filter {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 1rem;
    gap: 0.5rem;
  }
  .courses-filter label {
    font-size: 0.875rem;
    font-weight: 600;
  }
  .courses-filter select {
    font-size: 0.875rem;
    padding: 0.3rem 0.5rem;
    border: 1px solid var(--global-divider-color);
    border-radius: 4px;
    background: var(--global-bg-color);
    color: var(--global-text-color);
    cursor: pointer;
    transition: border-color 0.3s;
  }
  .courses-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 1rem;
    line-height: 1.5;
    background: var(--global-bg-color);
    border: 1px solid var(--global-divider-color);
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
  }
  .courses-table .date {
    font-weight: 600;
    padding: 0.75rem;
    width: 160px;
    color: #000 !important;
  }
  .courses-table .course {
    padding: 0.75rem;
  }
  .courses-table tr:hover .course {
    background-color: #f7f7f7;
  }
  .courses-table a {
    color: var(--global-theme-color);
    text-decoration: none;
  }
  .courses-table a:hover {
    text-decoration: underline;
  }
  #year-select:focus {
    border-color: var(--global-theme-color);
  }
</style>

<script>
  const yearSelect = document.getElementById('year-select');
  const rows = document.querySelectorAll('.courses-table tbody tr');

  function filterByYear() {
    const sel = yearSelect.value;
    rows.forEach(r => {
      r.style.display = (sel === 'all' || r.dataset.year === sel) ? '' : 'none';
    });
  }

  yearSelect.addEventListener('change', filterByYear);
  // initial filter to most recent year
  filterByYear();
</script>

--- 







