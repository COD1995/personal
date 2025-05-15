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

<section class="courses-section">
  <h2 class="courses-title">Teaching</h2>

  <div class="courses-filter">
    <label for="year-select">Filter by Year:</label>
    <select id="year-select">
      <option value="all">All Years</option>
      <option value="2025" selected>2025</option>
      <option value="2024">2024</option>
      <option value="2023">2023</option>
    </select>
  </div>

  <div class="course-cards">
    <!-- 2025 -->
    <article class="course-card" data-year="2025">
      <time class="course-date">May 27, 2025</time>
      <h3 class="course-name">
        <a href="{{ '/teaching/algo' | relative_url }}">
          Algorithm Analysis and Design
        </a>
      </h3>
    </article>
    <article class="course-card" data-year="2025">
      <time class="course-date">Jan 22, 2025</time>
      <h3 class="course-name">
        <a href="{{ '/teaching/aibasic' | relative_url }}">
          Basics of Artificial Intelligence
        </a>
      </h3>
    </article>

    <!-- 2024 -->
    <article class="course-card" data-year="2024">
      <time class="course-date">Aug 26, 2024</time>
      <h3 class="course-name">
        <a href="{{ '/teaching/deeplearning' | relative_url }}">
          Deep Learning
        </a>
      </h3>
    </article>
    <article class="course-card" data-year="2024">
      <time class="course-date">Jun 24, 2024</time>
      <h3 class="course-name">
        <a href="{{ '/teaching/pattern' | relative_url }}">
          Intro to Pattern Recognition
        </a>
      </h3>
    </article>
    <article class="course-card" data-year="2024">
      <time class="course-date">Jan 24, 2024</time>
      <h3 class="course-name">
        <a href="{{ '/teaching/machinelearning' | relative_url }}">
          Machine Learning
        </a>
      </h3>
    </article>

    <!-- 2023 -->
    <article class="course-card" data-year="2023">
      <time class="course-date">Aug 28, 2023</time>
      <h3 class="course-name">
        <a href="{{ '/teaching/deeplearning' | relative_url }}">
          Deep Learning
        </a>
      </h3>
    </article>
    <article class="course-card" data-year="2023">
      <time class="course-date">Jun 26, 2023</time>
      <h3 class="course-name">
        <a href="{{ '/teaching/pattern' | relative_url }}">
          Intro to Pattern Recognition
        </a>
      </h3>
    </article>
  </div>
</section>

<style>
  /* divider */
  .divider {
    margin: 6rem 0 2rem;
    border: none;
    border-top: 2px solid var(--global-divider-color);
  }

  /* section container & spacing */
  .courses-section {
    padding-top: 10rem;       /* ensure no overlap */
    max-width: 900px;
    /* margin: 0 auto; */
    /* padding-left: 1rem;
    padding-right: 1rem; */
  }

  .courses-title {
    font-size: 2.5rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--global-text-color);
    margin-bottom: 1.5rem;
    
    /* Thick colored underline */
    text-decoration-line: underline;
    text-decoration-color: rgba(156, 39, 176, 0.3);
    text-decoration-thickness: 0.3em;
    text-underline-offset: 0.2em;
  }


  /* filter */
  .courses-filter {
    display: flex;
    align-items: center;      /* <-- add this */
    justify-content: flex-end;
    margin-bottom: 1.5rem;
    gap: 0.5rem;
  }
  .courses-filter label {
    font-size: 0.9rem;
    font-weight: 500;
  }
  .courses-filter select {
    font-size: 0.9rem;
    padding: 0.4rem 0.6rem;
    border: 1px solid var(--global-divider-color);
    border-radius: 4px;
    background: var(--global-bg-color);
    color: var(--global-text-color);
    transition: border-color 0.2s;
    margin-top: -4px; 
  }
  .courses-filter select:focus {
    border-color: var(--global-theme-color);
    outline: none;
  }

  /* cards grid */
  .course-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, auto));
    justify-content: center;
    gap: 1.5rem;
  }

  .course-card {
    background: var(--global-bg-color);
    border: 1px solid var(--global-divider-color);
    border-radius: 8px;
    padding: 1.25rem;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.04);
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .course-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 6px 14px rgba(0, 0, 0, 0.08);
  }

  /* date & title */
  .course-date {
    display: block;
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--global-text-color);
    margin-bottom: 0.5rem;
  }
  .course-name {
    margin: 0;
    font-size: 1.125rem;
    font-weight: 600;
  }
  .course-name a {
    color: var(--global-theme-color);
    text-decoration: none;
  }
  .course-name a:hover {
    text-decoration: underline;
  }
</style>

<script>
  const yearSelect = document.getElementById('year-select');
  const cards = document.querySelectorAll('.course-card');

  function filterByYear() {
    const year = yearSelect.value;
    cards.forEach(card => {
      card.style.display =
        year === 'all' || card.dataset.year === year
          ? 'block'
          : 'none';
    });
  }

  yearSelect.addEventListener('change', filterByYear);
  filterByYear();  // initial filter
</script>

--- 







