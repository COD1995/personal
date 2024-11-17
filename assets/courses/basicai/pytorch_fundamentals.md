---
layout: page
title: Pytorch Fundamentals
description: 
related_publications: false
---
{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/00_pytorch_fundamentals.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/00_pytorch_fundamentals.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
{% jupyter_notebook jupyter_path %}
{% else %}

<p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}
