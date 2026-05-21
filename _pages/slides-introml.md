---
layout: page
title: Slides
permalink: /teaching/introml/slides/
back_link: '/teaching/introml/'
back_text: 'Intro to Machine Learning'
---

<div class="slide-viewer" data-baseurl="{{ site.baseurl }}">
  <h2 class="slide-title" id="slide-title">Loading…</h2>

  <div class="slide-stage" id="slide-stage">
    <canvas class="slide-canvas" id="slide-canvas" aria-label="slide page"></canvas>
    <button class="slide-hit slide-hit-prev" id="slide-hit-prev" type="button" aria-label="previous page"></button>
    <button class="slide-hit slide-hit-next" id="slide-hit-next" type="button" aria-label="next page"></button>
    <div class="slide-loading" id="slide-loading">Loading slides…</div>
  </div>

  <div class="slide-controls">
    <button id="slide-prev" type="button">&larr; Prev</button>
    <span class="slide-counter"><span id="slide-page-num">–</span> / <span id="slide-page-count">–</span></span>
    <button id="slide-next" type="button">Next &rarr;</button>
    <a id="slide-download" class="slide-download" href="#" download>Download PDF</a>
  </div>

  <p style="margin-top: 0.75rem; font-size: 0.85rem; color: var(--muted); font-family: var(--sans);">
    Click the left or right side of the slide, use the buttons above, or press ← / → to navigate.
  </p>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js" defer></script>
<script>
window.addEventListener('DOMContentLoaded', function () {
  var titles = {
    'ch1/01a_curve_fitting':         'Ch 1.1 — Curve Fitting',
    'ch1/01b_probability':           'Ch 1.2 — Probability Theory',
    'ch1/01c_model_selection_curse': 'Ch 1.3–1.4 — Model Selection & The Curse of Dimensionality',
    'ch1/01d_decision_theory':       'Ch 1.5 — Decision Theory',
    'ch1/01e_information_theory':    'Ch 1.6 — Information Theory'
  };

  var viewer       = document.querySelector('.slide-viewer');
  var baseurl      = viewer.getAttribute('data-baseurl') || '';
  var titleEl      = document.getElementById('slide-title');
  var stage        = document.getElementById('slide-stage');
  var canvas       = document.getElementById('slide-canvas');
  var ctx          = canvas.getContext('2d');
  var pageNumEl    = document.getElementById('slide-page-num');
  var pageCountEl  = document.getElementById('slide-page-count');
  var prevBtn      = document.getElementById('slide-prev');
  var nextBtn      = document.getElementById('slide-next');
  var hitPrev      = document.getElementById('slide-hit-prev');
  var hitNext      = document.getElementById('slide-hit-next');
  var downloadEl   = document.getElementById('slide-download');
  var loadingEl    = document.getElementById('slide-loading');

  var params = new URLSearchParams(window.location.search);
  var deck = params.get('deck');

  if (!deck || !/^[a-z0-9_\-\/]+$/i.test(deck) || deck.indexOf('..') !== -1) {
    showError('No deck specified. Use ?deck=ch1/01a_curve_fitting');
    return;
  }

  titleEl.textContent = titles[deck] || deck;
  document.title = (titles[deck] || deck) + ' — Slides';

  var pdfUrl = baseurl + '/assets/slides/introml/' + deck + '.pdf';
  downloadEl.href = pdfUrl;
  downloadEl.setAttribute('download', deck.split('/').pop() + '.pdf');

  if (typeof pdfjsLib === 'undefined') {
    showError('PDF.js failed to load.');
    return;
  }
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

  var pdfDoc = null;
  var currentPage = 1;
  var rendering = false;
  var pendingPage = null;

  function showError(msg) {
    titleEl.textContent = msg;
    if (loadingEl) loadingEl.remove();
    var err = document.createElement('div');
    err.className = 'slide-error';
    err.textContent = msg;
    stage.appendChild(err);
    prevBtn.disabled = true;
    nextBtn.disabled = true;
  }

  function updateButtons() {
    if (!pdfDoc) return;
    prevBtn.disabled = currentPage <= 1;
    nextBtn.disabled = currentPage >= pdfDoc.numPages;
  }

  function render(num) {
    if (!pdfDoc) return;
    rendering = true;
    pdfDoc.getPage(num).then(function (page) {
      var stageWidth = stage.clientWidth;
      var stageHeight = stage.clientHeight;
      var dpr = window.devicePixelRatio || 1;
      var unscaled = page.getViewport({ scale: 1 });
      var scale = Math.min(stageWidth / unscaled.width, stageHeight / unscaled.height);
      if (!isFinite(scale) || scale <= 0) scale = 1;
      var viewport = page.getViewport({ scale: scale * dpr });

      canvas.width = Math.floor(viewport.width);
      canvas.height = Math.floor(viewport.height);
      canvas.style.width = (viewport.width / dpr) + 'px';
      canvas.style.height = (viewport.height / dpr) + 'px';

      var task = page.render({ canvasContext: ctx, viewport: viewport });
      task.promise.then(function () {
        rendering = false;
        if (loadingEl && loadingEl.parentNode) loadingEl.remove();
        if (pendingPage !== null) {
          var next = pendingPage; pendingPage = null;
          render(next);
        }
      }).catch(function (err) {
        rendering = false;
        if (err && err.name !== 'RenderingCancelledException') console.error(err);
      });
    });
    pageNumEl.textContent = num;
    updateButtons();
  }

  function queueRender(num) {
    if (rendering) pendingPage = num;
    else render(num);
  }

  function go(delta) {
    if (!pdfDoc) return;
    var next = currentPage + delta;
    if (next < 1 || next > pdfDoc.numPages) return;
    currentPage = next;
    queueRender(currentPage);
  }

  prevBtn.addEventListener('click', function () { go(-1); });
  nextBtn.addEventListener('click', function () { go(1);  });
  hitPrev.addEventListener('click', function () { go(-1); });
  hitNext.addEventListener('click', function () { go(1);  });

  document.addEventListener('keydown', function (e) {
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
    if (e.key === 'ArrowLeft')  { go(-1); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { go(1);  e.preventDefault(); }
    else if (e.key === 'Home')       { if (pdfDoc) { currentPage = 1; queueRender(1); } e.preventDefault(); }
    else if (e.key === 'End')        { if (pdfDoc) { currentPage = pdfDoc.numPages; queueRender(currentPage); } e.preventDefault(); }
  });

  var resizeTimer;
  window.addEventListener('resize', function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () { queueRender(currentPage); }, 150);
  });

  pdfjsLib.getDocument(pdfUrl).promise.then(function (doc) {
    pdfDoc = doc;
    pageCountEl.textContent = doc.numPages;
    render(1);
  }).catch(function (err) {
    console.error(err);
    showError('Failed to load: ' + deck + '.pdf');
  });
});
</script>
