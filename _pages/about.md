---
layout: about
title: Jue Guo
subtitle: Ph.D. candidate · Computer Science · University at Buffalo
permalink: /

profile:
  image: prof_pic.png
  more_info: >
    <p>301 Davis Hall, Desk 12</p>
    <p>12 Capen Hall</p>
    <p>Buffalo, NY 14260</p>
---

I am a Ph.D. candidate in Computer Science at the University at Buffalo, advised by [Prof. A. Erdem Sariyüce](https://sariyuce.com/). My research is in machine learning — image classification, NLP, continual learning, and adversarial ML.

On the teaching side, I have led graduate and undergraduate courses across deep learning and pattern recognition. I work mostly in Python and PyTorch, and I care about bridging research-grade rigor with code that ships.

> *"We choose to go to the moon in this decade and do the other things, not because they are easy, but because they are hard."* — John F. Kennedy

<figure class="home-banner">
  <img src="{{ '/assets/img/class_photo.jpeg' | relative_url }}" alt="Class photo" loading="lazy">
</figure>

<section class="courses-section">
  <h2 class="courses-title">Teaching</h2>

  <div class="courses-filter">
    <label for="year-select">Year</label>
    <select id="year-select">
      <option value="all">All</option>
      <option value="2025" selected>2025</option>
      <option value="2024">2024</option>
      <option value="2023">2023</option>
    </select>
  </div>

  <div class="course-cards">
    <article class="course-card" data-year="2025">
      <time class="course-date">Summer 2025</time>
      <h3 class="course-name"><a href="{{ '/teaching/algo/' | relative_url }}">Algorithm Analysis &amp; Design</a></h3>
    </article>
    <article class="course-card" data-year="2025">
      <time class="course-date">Spring 2025</time>
      <h3 class="course-name"><a href="{{ '/teaching/aibasic/' | relative_url }}">Basics of Artificial Intelligence</a></h3>
    </article>
    <article class="course-card" data-year="2024">
      <time class="course-date">Fall 2024</time>
      <h3 class="course-name"><a href="{{ '/teaching/deeplearning/' | relative_url }}">Deep Learning</a></h3>
    </article>
    <article class="course-card" data-year="2024">
      <time class="course-date">Summer 2024</time>
      <h3 class="course-name"><a href="{{ '/teaching/pattern/' | relative_url }}">Intro to Pattern Recognition</a></h3>
    </article>
    <article class="course-card" data-year="2024">
      <time class="course-date">Spring 2024</time>
      <h3 class="course-name"><a href="{{ '/teaching/introml/' | relative_url }}">Intro to Machine Learning</a></h3>
    </article>
    <article class="course-card" data-year="2023">
      <time class="course-date">Fall 2023</time>
      <h3 class="course-name"><a href="{{ '/teaching/deeplearning/' | relative_url }}">Deep Learning</a></h3>
    </article>
    <article class="course-card" data-year="2023">
      <time class="course-date">Summer 2023</time>
      <h3 class="course-name"><a href="{{ '/teaching/pattern/' | relative_url }}">Intro to Pattern Recognition</a></h3>
    </article>
  </div>
</section>

<script>
  (function () {
    var sel = document.getElementById('year-select');
    var cards = document.querySelectorAll('.course-card');
    function apply() {
      var y = sel.value;
      cards.forEach(function (c) {
        c.style.display = (y === 'all' || c.dataset.year === y) ? '' : 'none';
      });
    }
    sel.addEventListener('change', apply);
    apply();
  })();
</script>
