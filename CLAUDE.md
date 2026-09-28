# CLAUDE.md

Personal academic website for Jue Guo, hand-rolled minimal Jekyll site (replaces a heavily customized al-folio install — see `git log` if you need archeology).

## What this site is

- **Stack:** Jekyll 4.x → static HTML → GitHub Pages.
- **Deploy:** Push to `master` → GitHub Actions builds → GitHub Pages at the custom domain **https://jue-guo.com** (registered at GoDaddy; DNS: four A records to GitHub Pages IPs + `www` CNAME → cod1995.github.io; custom domain set in repo Settings → Pages). The old `cod1995.github.io/personal/` address redirects there. `url` is `https://jue-guo.com`, `baseurl` is empty.
- **Design intent:** Low-key, classic, professional — ivory/navy, Source Sans 3, hairline rules, small caps; long-form course notes stay in a readable serif column. No Bootstrap, no MDB, no font-awesome, no Tabler icons, no JS frameworks. One stylesheet.

## Layout of the source

```
_config.yml              site metadata + collections
Gemfile                  jekyll + jekyll-feed + jekyll-sitemap + webrick (that's it)

_layouts/
  default.liquid         <html> shell — head, header, main, footer
  about.liquid           home page — hero, stats, about, teaching, office hours, programs, books (contact lives in the footer)
  page.liquid            everything else (teaching index, CV)
  course.liquid          course pages (catalog-entry style, see "Course pages" below)
  lecture.liquid         lecture notes for a course module (EAS 510 Basics of AI) — header, objectives, body, prev/next

_includes/
  head.liquid            <head> contents — meta, canonical, CSS, Google Fonts
  header.liquid          sticky top nav (About / Teaching / Community / CV / Contact) — "Community" links to the AI Office Hours section; don't label it "Office Hours" (students would mistake it for course office hours); also the reading-progress hairline (.scroll-progress + small inline script)
  footer.liquid          two-column "Correspondence" footer: letterhead-style details (email, office, links) + "Get in touch" form; base row with © and back-to-top
  icon.liquid            inline SVG icon set
  course-grid.liquid     course list from _data/courses.yml (+ guest-lecture note); links to /teaching/<slug>/
  figure.liquid          minimal <figure> wrapper used by course markdown
  slide.liquid           walks site.static_files in a folder, renders <img> stack
  semester-year-toggle.liquid
                         <select> + inline JS to toggle [data-semester-year] blocks
  teaching/<course>/<sem>.liquid
                         per-semester course schedule fragments

_pages/
  about.md               home page (permalink: /) — hero front matter + bio
  teaching.md            /teaching/ index — list of courses

_teaching/               Jekyll collection (output: true, /teaching/:path/)
  algo.liquid            CSE 431/531
  aibasic.liquid         Basics of AI (EAS 510) — outline links to the module notes
  aibasic/NN-<slug>.md   EAS 510 lecture notes, modules 00–09 (layout: lecture) → /teaching/aibasic/<slug>/
  deeplearning.liquid    CSE 676
  pattern.liquid         Intro to Pattern Recognition

assets/
  css/main.scss          THE stylesheet — has front matter so Jekyll compiles to main.css
  img/                   prof_pic.jpg (web-optimized; prof_pic.png is the original), class_photo2_plate.jpg (16:9 crop of class_photo2.jpeg for the home page), course banners (algorithm.png, Deep-learning.png, etc.)
  pdf/cv.pdf             compiled from _cv/cv.tex (pdflatex; needs fontawesome5 + sourcesanspro) — rebuild and copy here after editing the LaTeX
  img/courses/aibasic/   SVG figures for the EAS 510 lecture notes (NN-*.svg, editable)
```

## Build / run

```bash
bundle install
bundle exec jekyll serve --host 127.0.0.1 --port 4001
# → http://127.0.0.1:4001/
```

`bundle exec jekyll build` for one-shot. Output goes to `_site/` (gitignored).

## Design system (in `assets/css/main.scss`)

Restyled Sep 27 2026 to a quiet, classic look ("old money" academic): ivory paper, navy ink, Source Sans 3 type, hairline rules, small caps. No gradients, glows, drop shadows, pills, or icon badges — keep it that way.
CSS custom properties at the top — change there before touching rules:

- Light: `--bg #f6f2e9` (ivory), `--surface`, `--surface-2` (alternate section band), `--text #1e1c19`, `--muted`, `--faint`, `--rule`, `--rule-strong`
- `--accent #1f2e4a` (navy: links, primary button), `--brass #8a6a3a` (eyebrows, small ornaments)
- Light theme only — dark mode was removed at Jue's request (Sep 27 2026); don't reintroduce it.
- Fonts: **Source Sans 3 everywhere** (Jue's choice, Sep 27 2026 — a clear humanist sans, free cousin of the Myriad-style figure font he likes). `--display`, `--serif`, `--prose`, `--sans` all point to it (the variable names are historical); `--mono` JetBrains Mono for code. Loaded from Google Fonts in `head.liquid`.
- `%smallcaps` placeholder = uppercase letter-spaced serif label; used by nav, buttons, eyebrows.
- Widths: `--max 760px`; `--max-wide 1120px` (`.wrap.wide`); course pages `.page.wrap` 880px.

Components: `.btn` (`.btn-primary` navy fill / `.btn-ghost` outline, square corners), `.link-quiet`, `.text-links` (italic links separated by middots), `.eyebrow`, `.rule-short`, `.section` (`.section-alt`), `.split`, `.course-grid` / `.course` (two-column list with hairlines; each entry is one `.course-link` with a "Syllabus & materials →" cue and a hover underline on the title), `.plate` (one matted photograph with italic caption; `.plate-wide` on /teaching/), `.teaching-lead` (home teaching intro + photo), `.programs` (roman-numbered), `.books`, CV `.cv-section`.

Icons: `_includes/icon.liquid` holds inline SVGs; nothing in the current design uses them.

## Content data (`_data/`)

Home page and /teaching/ are data-driven — edit YAML, not HTML:

- `courses.yml` — course cards (slug → `/teaching/<slug>/`, code, title, terms, blurb, icon)
- `stats.yml` — the four numbers under the hero (+ the three on /teaching/ are in `_pages/teaching.md`)
- `roles.yml` — "Currently" list
- `programs.yml` — "AI programs through CIFF" (roman-numbered) (executive programs, technical training, custom curriculum, applied research)
- `community.yml` — AI Office Hours platform links (label, url, icon, one-line note).
- `meetup_events.json` — **generated**: upcoming AI Office Hours events from the public Meetup iCal feed, written by `scripts/fetch_meetup.py`. Not automated (by choice): after adding or changing a Meetup event, run `python3 scripts/fetch_meetup.py` locally, then commit the JSON and push. Ended events hide themselves client-side, so a stale file only means new events are missing. If Meetup is unreachable the script leaves the file untouched.
- `sessions.yml` — AI Office Hours decks (files in `assets/slides/ai-office-hours/`)
- `resources.yml` — recommended books (covers in `assets/img/books/`)
- `aibasic.yml` — EAS 510 module list (num, slug, title, weeks, summary, learnpytorch.io reading link). Drives the lecture layout's eyebrow, companion-reading link, and previous/next links. Keep it in sync with `_teaching/aibasic/` and the outline in `_teaching/aibasic.liquid`.

Hero copy (eyebrow, headline, lede, interests, portrait) lives in `_pages/about.md` front matter; the bio is its markdown body.
Page front matter extras for `layout: page`: `eyebrow`, `wide: true`, `prose: false` (course-notes body sizing off).

## Adding content

- **New course:** drop `_teaching/<slug>.liquid` with front matter `layout: page`, `title`, `description`, optional `back_link: '/teaching/'`. Add an entry to `_data/courses.yml` (shows on both home and /teaching/).
- **New semester for an existing course:** create `_includes/teaching/<course>/<sem>.liquid` and reference it from the parent course file inside a `<div data-semester-year="...">` block; add it to the `semesters:` front-matter list.
- **Lecture notes (long markdown):** for EAS 510, add `_teaching/aibasic/NN-<slug>.md` (layout: lecture) and a matching entry in `_data/aibasic.yml`, then link it from the outline in `_teaching/aibasic.liquid`. For another course, copy that pattern (a data file + the lecture layout, which currently reads `site.data.aibasic`).

## Course pages (`_teaching/*.liquid`)

Redone Sep 28 2026 in a "catalog entry" style: `_layouts/course.liquid` (layout: course). Left column = course facts, right = title, description, and "What the course covers". Everything is front matter:
`title`, `code`, `level`, `taught`, `prereq`, `text`, `grading`, `description` (meta), `lede` (first paragraph), `about` (HTML, more paragraphs), and `semesters:` — each `{value, label, selected, intro, assessment: [[name, weight], ...], outline: [{when, topic, guide} | {group}]}`. The semester select appears automatically when there is more than one semester; it toggles both the outline and the assessment weights. The page body holds optional extra sections (introml: course policies; pattern: final project).
**EAS 510 Basics of AI has full, published lecture notes** (Sep 28 2026): `_teaching/aibasic/00-…09-*.md`, one per learnpytorch.io chapter, rewritten in Jue's teaching voice for engineering students with no CS background (own prose; code path follows Daniel Bourke's MIT-licensed *Learn PyTorch for Deep Learning*, credited on every page). See "Lecture notes (EAS 510)" below.
For the other courses, lecture notes, slides, reading notes, and project briefs are **not published** — they were half-finished. They live in `_private-course-notes/` (git-ignored and excluded from the build). To publish one again, move it back to its original path and link it from the course page. The old per-semester includes are in `_private-course-notes/_old-course-includes/`.

## Lecture notes (EAS 510)

- Front matter: `layout: lecture`, `module: "NN"`, `title`, `description`, `math: true` (loads MathJax; use `$$…$$` for inline and display math), `objectives:` (list, shown as "By the end of this module you can").
- Body pattern: `* Contents` + `{:toc}` (kramdown TOC), intro, `##` sections, Summary, Exercises (`{: .exercises}` on the line before the ordered list), Going further.
- ```` ```python ```` = code; a ```` ```text ```` block directly after a python block is its printed output (styled as a dashed "Output" box by CSS — `div.language-python + div.language-text`). Other fences (`console`, `shell`) are plain code.
- Callouts: a blockquote followed by `{: .callout}` (navy rule) or `{: .callout-warn}` (brass rule); first word bold ("**Note.**", "**Watch out.**").
- Figures: hand-written SVGs in `assets/img/courses/aibasic/NN-*.svg` (site palette, editable; no titles baked in), included with `<figure class="figure figure-wide figure-plain">` (`figure-wide` stretches diagrams to the column; leave it off for matplotlib plots).
- Links between modules: `{{ '/teaching/aibasic/<slug>/' | relative_url }}`.
- Every output in the notes is real: the notes were executed cell by cell (CPU, PyTorch 2.14) and the output blocks were written by the runner. Pretrained torchvision weights couldn't be downloaded in that sandbox, so modules 06–09 deliberately show no training/accuracy output for pretrained models (only shapes, parameter counts, file sizes, CPU timings); accuracy figures quoted there come from the learnpytorch.io chapters. The runner, preview script, style guide, and the notes with their `<!-- runner: … -->` directives are in `Claude outputs/aibasic-notes-build/` (not in git) for re-running after edits.
- Mobile: tables become horizontally scrollable under 700px; long inline code wraps.
- Heading tracker: a small script at the bottom of `_layouts/lecture.liquid` builds an "On this page" list from the note's `h2`/`h3` headings and highlights the section being read. At 1200px and wider it is a sticky right-hand sidebar (h3s expand under the active h2; the inline `{:toc}` list is hidden); below that it is a sticky "Contents" drop-down under the site header showing the current section. It also sets `scroll-padding-top` so linked headings clear the header. Sections with subheadings get an arrow button (`.lec-toc-toggle`) that shows/hides them; the section being read opens automatically, sections the reader opens stay open, and a section the reader collapses stays collapsed until they scroll out of it. Styles: `.lec-toc*` in main.scss.

## Conventions / gotchas

- `permalink: pretty` in `_config.yml` — every output is a directory with `index.html`.
- Don't reintroduce Bootstrap classes (`row`, `col-*`) in new content — they won't style. Use plain HTML or the existing classes (`course-grid`, `styled-table`, `course-description-box`, `course-semester-info`).
- `figure.liquid` and `slide.liquid` are intentionally minimal — they exist only so the inherited `assets/courses/**` markdown still builds. Don't expand them with carousels or lazy-loading shims unless asked.
- The semester-toggle JS lives inside `_includes/semester-year-toggle.liquid` (one DOMContentLoaded handler, vanilla JS). The email-copy JS is at the bottom of `_layouts/default.liquid`.
- Header "active" state is `class="active"`, set in `_includes/header.liquid` based on `page.url contains '/teaching/'` etc.

## What was deleted (don't bring back without asking)

al-folio's bibliography flow, blog/posts, projects, repositories, profiles, CV-from-yaml layout, archive layouts, Distill layout, MathJax/TikZJax/Mermaid/Vega/ECharts/Leaflet/Chart.js loaders, Bootstrap, MDB, Tabler icons, font-awesome, jekyll-scholar, jekyll-archives, jekyll-paginate-v2, jekyll-tabs, jekyll-toc, jekyll-imagemagick, the Ruby plugins under `_plugins/`, Docker setup, lighthouse_results, the Einstein placeholder content. See first commits after the rewrite for the diff.

- Office: 335 Jacobs Management Center, School of Management (not Davis Hall — Jue is no longer based in CSE). Don't name the CSE department on the site; course codes like "CSE 676" are fine.
- Contact form: in `_includes/footer.liquid`, posts to Web3Forms (`web3forms_key` in `_config.yml`; free tier 250 messages/month, delivered to guoj1995@gmail.com). Inline JS shows the sent/error message; `botcheck` is the spam honeypot. Remove the key to fall back to a plain email link.
- Contact form anti-spam (client side, in the footer script): blocks sends under 3 s after page load, messages under 20 characters, and link-heavy messages (3+ links, or links with under 20 characters of other text). Each shows a polite prompt. Thresholds are the `MIN_SECONDS` / `MIN_CHARS` constants.
- Contact form has a "Regarding" dropdown (topics listed in `_includes/footer.liquid`); the chosen topic is added to the email subject ("Website message — <topic>") so messages are easy to sort in Gmail.
