# CLAUDE.md

Personal academic website for Jue Guo, hand-rolled minimal Jekyll site (replaces a heavily customized al-folio install — see `git log` if you need archeology).

## What this site is

- **Stack:** Jekyll 4.x → static HTML → GitHub Pages.
- **Deploy:** Pushed branch builds at `https://cod1995.github.io/personal/` (`url` + `baseurl` set in `_config.yml`).
- **Design intent:** Minimal, serif body / sans nav, narrow column. No Bootstrap, no MDB, no font-awesome, no Tabler icons, no JS frameworks. One stylesheet.

## Layout of the source

```
_config.yml              site metadata + collections
Gemfile                  jekyll + jekyll-feed + jekyll-sitemap + webrick (that's it)

_layouts/
  default.liquid         <html> shell — head, header, main, footer
  about.liquid           home page (hero w/ photo + bio + content)
  page.liquid            everything else (course pages, teaching index)

_includes/
  head.liquid            <head> contents — meta, canonical, CSS, Google Fonts
  header.liquid          sticky top nav (About / Teaching / CV / Email / GitHub)
  footer.liquid          one-line footer
  figure.liquid          minimal <figure> wrapper used by course markdown
  slide.liquid           walks site.static_files in a folder, renders <img> stack
  semester-year-toggle.liquid
                         <select> + inline JS to toggle [data-semester-year] blocks
  teaching/<course>/<sem>.liquid
                         per-semester course schedule fragments

_pages/
  about.md               home page (permalink: /) — bio + filterable teaching cards
  teaching.md            /teaching/ index — list of courses

_teaching/               Jekyll collection (output: true, /teaching/:path/)
  algo.liquid            CSE 431/531
  aibasic.liquid         Basics of AI
  deeplearning.liquid    CSE 676
  pattern.liquid         Intro to Pattern Recognition

assets/
  css/main.scss          THE stylesheet — has front matter so Jekyll compiles to main.css
  img/                   prof_pic.jpg, course banners (algorithm.png, Deep-learning.png, etc.)
  pdf/cv.pdf             linked from nav
  courses/               long-form lecture notes as .md files (basicai/, deeplearning/)
                         these are reachable as pages and use {% include figure.liquid %} / slide.liquid
```

## Build / run

```bash
bundle install
bundle exec jekyll serve --host 127.0.0.1 --port 4001
# → http://127.0.0.1:4001/personal/
```

`bundle exec jekyll build` for one-shot. Output goes to `_site/` (gitignored).

## Design system (in `assets/css/main.scss`)

CSS custom properties at the top, change there before touching rules:

- `--bg #fbfaf7` (cream), `--text #1d1d1f`, `--muted #5f5f63`
- `--accent #8b3a3a` (muted red — used for links, hover underlines, accent borders)
- `--rule #e5e2dc` (hairline borders)
- `--max 760px` (home/article width), `--page.wrap 880px` (course pages, set in rule)
- `--serif "Source Serif 4"`, `--sans "Inter"`, `--mono "JetBrains Mono"` (loaded from Google Fonts in `head.liquid`)

Layout system: every page wrapper uses `.wrap`. Home uses `.home.wrap`, course/page uses `.page.wrap`. The header uses its own `.nav-wrap` widened to `--max-wide`.

## Adding content

- **New course:** drop `_teaching/<slug>.liquid` with front matter `layout: page`, `title`, `description`, optional `back_link: '/teaching/'`. Add a card in `_pages/teaching.md` and (optionally) `_pages/about.md`.
- **New semester for an existing course:** create `_includes/teaching/<course>/<sem>.liquid` and reference it from the parent course file inside a `<div data-semester-year="...">` block; add it to the `semesters:` front-matter list.
- **Lecture notes (long markdown):** drop `.md` under `assets/courses/...` with front matter `layout: page`, `title`, optional `back_link`. They auto-render as pages.

## Conventions / gotchas

- `permalink: pretty` in `_config.yml` — every output is a directory with `index.html`.
- Don't reintroduce Bootstrap classes (`row`, `col-*`) in new content — they won't style. Use plain HTML or the existing classes (`course-cards`, `course-card`, `styled-table`, `course-description-box`, `course-semester-info`).
- `figure.liquid` and `slide.liquid` are intentionally minimal — they exist only so the inherited `assets/courses/**` markdown still builds. Don't expand them with carousels or lazy-loading shims unless asked.
- The semester-toggle JS lives inside `_includes/semester-year-toggle.liquid` (one DOMContentLoaded handler, vanilla JS). The home page has its own inline year-filter script.
- Header "active" state is `class="active"`, set in `_includes/header.liquid` based on `page.url contains '/teaching/'` etc.

## What was deleted (don't bring back without asking)

al-folio's bibliography flow, blog/posts, projects, repositories, profiles, CV-from-yaml layout, archive layouts, Distill layout, MathJax/TikZJax/Mermaid/Vega/ECharts/Leaflet/Chart.js loaders, Bootstrap, MDB, Tabler icons, font-awesome, jekyll-scholar, jekyll-archives, jekyll-paginate-v2, jekyll-tabs, jekyll-toc, jekyll-imagemagick, the Ruby plugins under `_plugins/`, Docker setup, lighthouse_results, the Einstein placeholder content. See first commits after the rewrite for the diff.
