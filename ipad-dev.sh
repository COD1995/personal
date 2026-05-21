#!/usr/bin/env bash
# ipad-dev.sh — start Jekyll bound to localhost so the VS Code tunnel can forward it to iPad.
# Run on the Mac from the vscode.dev terminal when working from iPad.
# When sitting at the Mac, just use `bundle exec jekyll serve` directly — no script needed.
#
# Usage:
#   ./ipad-dev.sh [port]      default port 4000
#
# After it starts, in vscode.dev on the iPad:
#   1. Cmd+Shift+P → "Ports: Focus on Ports View"
#   2. Forward the printed port (Private)
#   3. Open <forwarded-url>/personal/ in Safari   (baseurl from _config.yml)

set -euo pipefail
cd "$(dirname "$0")"

PORT="${1:-4000}"

command -v bundle >/dev/null \
  || { echo "error: 'bundle' not on PATH — install Bundler (gem install bundler) or rbenv/asdf shim" >&2; exit 1; }
[ -f Gemfile ] \
  || { echo "error: no Gemfile in $(pwd) — is this a Jekyll project?" >&2; exit 1; }

cat <<EOF

Starting Jekyll on 127.0.0.1:$PORT.
Site has baseurl '/personal' from _config.yml — the live URL will end in /personal/.

iPad steps in vscode.dev:
  1. Cmd+Shift+P → "Ports: Focus on Ports View"
  2. Forward port $PORT (visibility: Private)
  3. Open <forwarded-url>/personal/ in Safari

Edit any .md / .html / _config.yml → save → Jekyll rebuilds (incremental) → pull-to-refresh in Safari.
Ctrl+C to stop.

EOF

exec bundle exec jekyll serve \
  --host 127.0.0.1 \
  --port "$PORT" \
  --incremental \
  --watch
