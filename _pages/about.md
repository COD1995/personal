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

news: false # includes a list of news items
selected_papers: false # includes a list of papers marked as "selected={true}"
social: false # includes social icons at the bottom of the page
---

Jue Guo is a dedicated Ph.D. candidate in Computer Science with a strong academic and professional background. His teaching experience spans crucial areas of the field, notably in deep learning and pattern recognition, where he has effectively imparted knowledge and inspired future computer scientists. Jue’s research has been particularly focused on pioneering machine learning methodologies, encompassing diverse applications such as image classification, natural language processing, continual learning, and medical imaging.

Jue’s technical proficiency is demonstrated by his expertise in python, pytorch and JavaScript, which he seamlessly integrates into his research and development work. His adept use of various machine learning frameworks allows him to approach and solve complex computational challenges with precision and innovation. This combination of teaching acumen, research excellence, and versatile programming skills positions Jue as an accomplished and forward-thinking contributor to the field of computer science.

<!-- teaching -->
<hr style="margin: 8rem 0 1.5rem 0; border: none; border-top: 1px solid var(--global-divider-color);" />

<div class="courses-container" style="max-width: 700px; margin: 0 auto; padding: 0 1rem;">
  <h2 style="
    font-size: 1.75rem;
    font-weight: 500;
    margin-bottom: 1rem;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    color: var(--global-text-color);
    text-align: center;">
    Current Courses I’m Teaching
  </h2>
  <div class="year-filter" style="
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
  ">
    <label for="year-select" style="
      font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;
      font-size:0.875rem;
      color:var(--global-text-color);
      font-weight:600;
      margin:0;
    ">
      Filter by Year:
    </label>
    <select id="year-select" style="
      font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;
      font-size:0.875rem;
      padding:0.3rem 0.5rem;
      border:1px solid var(--global-divider-color);
      border-radius:4px;
      background: var(--global-bg-color);
      color: var(--global-text-color);
      cursor: pointer;
      outline: none;
      transition: border-color 0.3s;
    ">
      <option value="all">All Years</option>
      <option value="2025">2025</option>
      <option value="2024">2024</option>
      <option value="2023">2023</option>
    </select>
  </div>
  <table class="courses-table" style="
    width:100%; 
    border-collapse:collapse; 
    font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;
    color:var(--global-text-color); 
    font-size:1rem; 
    line-height:1.5;
    background: var(--global-bg-color);
    border: 1px solid var(--global-divider-color);
    border-radius: 6px;
    overflow: hidden; 
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
  ">
    <tbody>
      <tr data-year="2025">
        <td style="font-weight:600; padding:0.75rem;">Jan 22, 2025</td>
        <td style="padding:0.75rem;">
          <a href="{{ '/teaching/aibasic' | relative_url }}" style="color:var(--global-theme-color); text-decoration:none;">
            start teaching basics of artificial intelligence
          </a>
        </td>
      </tr>
      <!-- 2024 Courses -->
      <tr data-year="2024">
        <td style="font-weight:600; padding:0.75rem; width:150px;">Aug 26, 2024</td>
        <td style="padding:0.75rem;">
          <a href="{{ '/teaching/deeplearnig' | relative_url }}" style="color:var(--global-theme-color); text-decoration:none;">start teaching deep learning</a>
        </td>
      </tr>
      <tr data-year="2024">
        <td style="font-weight:600; padding:0.75rem;">Jun 24, 2024</td>
        <td style="padding:0.75rem;">
          <a href="{{ '/teaching/pattern' | relative_url }}" style="color:var(--global-theme-color); text-decoration:none;">start teaching intro to pattern recognition</a>
        </td>
      </tr>
      <tr data-year="2024">
        <td style="font-weight:600; padding:0.75rem;">Jan 24, 2024</td>
        <td style="padding:0.75rem;">
          <a href="{{ '/teaching/machinelearning' | relative_url }}" style="color:var(--global-theme-color); text-decoration:none;">start teaching machine learning</a>
        </td>
      </tr>
      <!-- 2023 Courses -->
      <tr data-year="2023">
        <td style="font-weight:600; padding:0.75rem;">Aug 28, 2023</td>
        <td style="padding:0.75rem;">
          <a href="{{ '/teaching/deeplearnig' | relative_url }}" style="color:var(--global-theme-color); text-decoration:none;">start teaching deep learning</a>
        </td>
      </tr>
      <tr data-year="2023">
        <td style="font-weight:600; padding:0.75rem;">Jun 26, 2023</td>
        <td style="padding:0.75rem;">
          <a href="{{ '/teaching/pattern' | relative_url }}" style="color:var(--global-theme-color); text-decoration:none;">start teaching intro to pattern recognition</a>
        </td>
      </tr>
    </tbody>
  </table>
</div>

<style>
  .courses-table tbody tr {
    transition: background-color 0.3s, color 0.3s;
  }

  .courses-table tbody tr:hover td {
    background-color: #f7f7f7;
    color: var(--global-text-color);
  }

  .courses-table tbody tr a:hover {
    text-decoration: underline;
  }

  #year-select:hover,
  #year-select:focus {
    border-color: var(--global-theme-color);
  }
</style>

<script>
  const yearSelect = document.getElementById('year-select');
  const rows = document.querySelectorAll('.courses-table tbody tr');

  yearSelect.addEventListener('change', function() {
    const selectedYear = this.value;
    rows.forEach(row => {
      const rowYear = row.getAttribute('data-year');
      if (selectedYear === 'all' || rowYear === selectedYear) {
        row.style.display = '';
      } else {
        row.style.display = 'none';
      }
    });
  });
</script>

--- 







