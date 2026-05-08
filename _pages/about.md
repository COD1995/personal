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
      <option value="all" selected>All</option>
      <option value="2025">2025</option>
      <option value="2024">2024</option>
      <option value="2023">2023</option>
    </select>
  </div>

  <div class="course-cards">
    <article class="course-card" data-mode="all">
      <time class="course-date">Summer 2025</time>
      <h3 class="course-name"><a href="{{ '/teaching/algo/' | relative_url }}">Algorithm Analysis &amp; Design</a></h3>
    </article>
    <article class="course-card" data-mode="all">
      <time class="course-date">Spring 2025</time>
      <h3 class="course-name"><a href="{{ '/teaching/aibasic/' | relative_url }}">Basics of Artificial Intelligence</a></h3>
    </article>
    <article class="course-card" data-mode="all">
      <time class="course-date">Fall 2024 · Fall 2023</time>
      <h3 class="course-name"><a href="{{ '/teaching/deeplearning/' | relative_url }}">Deep Learning</a></h3>
    </article>
    <article class="course-card" data-mode="all">
      <time class="course-date">Summer 2024 · Summer 2023</time>
      <h3 class="course-name"><a href="{{ '/teaching/pattern/' | relative_url }}">Intro to Pattern Recognition</a></h3>
    </article>
    <article class="course-card" data-mode="all">
      <time class="course-date">Spring 2024</time>
      <h3 class="course-name"><a href="{{ '/teaching/introml/' | relative_url }}">Intro to Machine Learning</a></h3>
    </article>

    <article class="course-card" data-mode="year" data-year="2025">
      <time class="course-date">Summer 2025</time>
      <h3 class="course-name"><a href="{{ '/teaching/algo/' | relative_url }}">Algorithm Analysis &amp; Design</a></h3>
    </article>
    <article class="course-card" data-mode="year" data-year="2025">
      <time class="course-date">Spring 2025</time>
      <h3 class="course-name"><a href="{{ '/teaching/aibasic/' | relative_url }}">Basics of Artificial Intelligence</a></h3>
    </article>
    <article class="course-card" data-mode="year" data-year="2024">
      <time class="course-date">Fall 2024</time>
      <h3 class="course-name"><a href="{{ '/teaching/deeplearning/' | relative_url }}">Deep Learning</a></h3>
    </article>
    <article class="course-card" data-mode="year" data-year="2024">
      <time class="course-date">Summer 2024</time>
      <h3 class="course-name"><a href="{{ '/teaching/pattern/' | relative_url }}">Intro to Pattern Recognition</a></h3>
    </article>
    <article class="course-card" data-mode="year" data-year="2024">
      <time class="course-date">Spring 2024</time>
      <h3 class="course-name"><a href="{{ '/teaching/introml/' | relative_url }}">Intro to Machine Learning</a></h3>
    </article>
    <article class="course-card" data-mode="year" data-year="2023">
      <time class="course-date">Fall 2023</time>
      <h3 class="course-name"><a href="{{ '/teaching/deeplearning/' | relative_url }}">Deep Learning</a></h3>
    </article>
    <article class="course-card" data-mode="year" data-year="2023">
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
        var show = (y === 'all')
          ? c.dataset.mode === 'all'
          : c.dataset.mode === 'year' && c.dataset.year === y;
        c.style.display = show ? '' : 'none';
      });
    }
    sel.addEventListener('change', apply);
    apply();
  })();
</script>

<section class="resources-section">
  <h2 class="resources-title">Self-learning resources</h2>
  <p class="resources-intro">Free books I keep pointing students to when they want to dig into machine learning or deep learning on their own.</p>
  <ul class="resources-list">
    <li class="resource">
      <img class="resource-cover" src="{{ '/assets/img/books/d2l.jpg' | relative_url }}" alt="Cover of Dive into Deep Learning" loading="lazy">
      <div class="resource-text">
        <a class="resource-title" href="https://d2l.ai" target="_blank" rel="noopener">Dive into Deep Learning</a>
        <span class="resource-byline">Zhang, Lipton, Li &amp; Smola</span>
        <p class="resource-desc">Interactive textbook with full implementations in PyTorch, MXNet, and JAX. The reference I lean on most in CSE&nbsp;676.</p>
      </div>
    </li>
    <li class="resource">
      <img class="resource-cover" src="{{ '/assets/img/books/deep-learning-with-python.jpg' | relative_url }}" alt="Cover of Deep Learning with Python" loading="lazy">
      <div class="resource-text">
        <a class="resource-title" href="https://deeplearningwithpython.io" target="_blank" rel="noopener">Deep Learning with Python</a>
        <span class="resource-byline">François Chollet</span>
        <p class="resource-desc">Practical, Keras-first introduction to neural networks. Excellent for building intuition before going under the hood.</p>
      </div>
    </li>
    <li class="resource">
      <img class="resource-cover" src="{{ '/assets/img/books/bishop-deep-learning.jpg' | relative_url }}" alt="Cover of Deep Learning: Foundations and Concepts" loading="lazy">
      <div class="resource-text">
        <a class="resource-title" href="https://www.bishopbook.com" target="_blank" rel="noopener">Deep Learning: Foundations and Concepts</a>
        <span class="resource-byline">Christopher M. Bishop &amp; Hugh Bishop</span>
        <p class="resource-desc">A modern volume bridging classical ML to deep learning — the spiritual successor to PRML.</p>
      </div>
    </li>
    <li class="resource">
      <img class="resource-cover" src="{{ '/assets/img/books/bishop-prml.jpg' | relative_url }}" alt="Cover of Pattern Recognition and Machine Learning" loading="lazy">
      <div class="resource-text">
        <span class="resource-title">Pattern Recognition and Machine Learning</span>
        <span class="resource-byline">Christopher M. Bishop</span>
        <p class="resource-desc">The classic graduate ML text (2006). Rigorous foundation in Bayesian methods, kernels, and graphical models — freely available online as a PDF.</p>
      </div>
    </li>
  </ul>
</section>

