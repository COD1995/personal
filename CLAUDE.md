# CLAUDE.md

Personal academic website for Jue Guo, hand-rolled minimal Jekyll site (replaces a heavily customized al-folio install — see `git log` if you need archeology).

## What this site is

- **Stack:** Jekyll 4.x → static HTML → GitHub Pages.
- **Deploy:** Push to `master` → GitHub Actions builds → GitHub Pages at the custom domain **https://jue-guo.com** (registered at GoDaddy; DNS: four A records to GitHub Pages IPs + `www` CNAME → cod1995.github.io; custom domain set in repo Settings → Pages). The old `cod1995.github.io/personal/` address redirects there. `url` is `https://jue-guo.com`, `baseurl` is empty.
- **Design intent:** Low-key, classic, professional — ivory/navy, Garamond, hairline rules, small caps; long-form course notes stay in a readable serif column. No Bootstrap, no MDB, no font-awesome, no Tabler icons, no JS frameworks. One stylesheet.

## Layout of the source

```
_config.yml              site metadata + collections
Gemfile                  jekyll + jekyll-feed + jekyll-sitemap + webrick (that's it)

_layouts/
  default.liquid         <html> shell — head, header, main, footer
  about.liquid           home page — hero, stats, about, teaching, office hours, programs, books (contact lives in the footer)
  page.liquid            everything else (course pages, teaching index)

_includes/
  head.liquid            <head> contents — meta, canonical, CSS, Google Fonts
  header.liquid          sticky top nav (About / Teaching / Community / CV / Contact) — "Community" links to the AI Office Hours section; don't label it "Office Hours" (students would mistake it for course office hours)
  footer.liquid          two-column "Correspondence" footer: letterhead-style details (email, office, links) + "Send a note" form; base row with © and back-to-top
  icon.liquid            inline SVG icon set
  course-grid.liquid     course list from _data/courses.yml (+ guest-lecture note)
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
  aibasic.liquid         Basics of AI
  deeplearning.liquid    CSE 676
  pattern.liquid         Intro to Pattern Recognition

assets/
  css/main.scss          THE stylesheet — has front matter so Jekyll compiles to main.css
  img/                   prof_pic.jpg (web-optimized; prof_pic.png is the original), class_photo2_plate.jpg (16:9 crop of class_photo2.jpeg for the home page), course banners (algorithm.png, Deep-learning.png, etc.)
  pdf/cv.pdf             linked from nav
  courses/               long-form lecture notes as .md files (basicai/, deeplearning/)
                         these are reachable as pages and use {% include figure.liquid %} / slide.liquid
```

## Build / run

```bash
bundle install
bundle exec jekyll serve --host 127.0.0.1 --port 4001
# → http://127.0.0.1:4001/
```

`bundle exec jekyll build` for one-shot. Output goes to `_site/` (gitignored).

## Design system (in `assets/css/main.scss`)

Restyled Sep 27 2026 to a quiet, classic look ("old money" academic): ivory paper, navy ink, Garamond type, hairline rules, small caps. No gradients, glows, drop shadows, pills, or icon badges — keep it that way.
CSS custom properties at the top — change there before touching rules:

- Light: `--bg #f6f2e9` (ivory), `--surface`, `--surface-2` (alternate section band), `--text #1e1c19`, `--muted`, `--faint`, `--rule`, `--rule-strong`
- `--accent #1f2e4a` (navy: links, primary button), `--brass #8a6a3a` (eyebrows, small ornaments)
- Light theme only — dark mode was removed at Jue's request (Sep 27 2026); don't reintroduce it.
- Fonts: `--display` Cormorant Garamond (headings, name, numbers), `--serif` EB Garamond (body + small caps UI), `--prose` Source Serif 4 (long-form course notes — `.page:not(.page--plain) .page-body`), `--sans` Inter (only tables/UI inside course notes), `--mono` JetBrains Mono. Loaded from Google Fonts in `head.liquid`.
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
- `community.yml` — AI Office Hours links (Meetup, Discord, YouTube, LinkedIn Group, Substack) and the `next:` live session. The next-session card hides itself client-side once `ends` has passed; update or delete `next:` after each session.
- `sessions.yml` — AI Office Hours decks (files in `assets/slides/ai-office-hours/`)
- `resources.yml` — recommended books (covers in `assets/img/books/`)

Hero copy (eyebrow, headline, lede, interests, portrait) lives in `_pages/about.md` front matter; the bio is its markdown body.
Page front matter extras for `layout: page`: `eyebrow`, `wide: true`, `prose: false` (Garamond body instead of the Source Serif course-notes body).

## Adding content

- **New course:** drop `_teaching/<slug>.liquid` with front matter `layout: page`, `title`, `description`, optional `back_link: '/teaching/'`. Add an entry to `_data/courses.yml` (shows on both home and /teaching/).
- **New semester for an existing course:** create `_includes/teaching/<course>/<sem>.liquid` and reference it from the parent course file inside a `<div data-semester-year="...">` block; add it to the `semesters:` front-matter list.
- **Lecture notes (long markdown):** drop `.md` under `assets/courses/...` with front matter `layout: page`, `title`, optional `back_link`. They auto-render as pages.

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
