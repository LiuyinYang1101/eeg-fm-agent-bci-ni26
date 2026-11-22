#!/usr/bin/env bash
# Build docs/index.html (a standalone page for GitHub Pages) from site/_page.html (the page fragment).
set -euo pipefail
cd "$(dirname "$0")/.."
DESC="I supervised Claude agents through an EEG decoding competition for 48 days. We finished 13th of 302 (0.844; best entry 0.872). The code, the decision log and the mistakes are public."
{
  printf '<!doctype html>\n<html lang="en">\n<head>\n'
  sed -n '1,/^<\/style>$/p' site/_page.html
  printf '<meta name="description" content="%s">\n' "$DESC"
  printf '<meta property="og:title" content="I let AI agents run my EEG competition entry for 48 days">\n'
  printf '<meta property="og:description" content="%s">\n' "$DESC"
  printf '<meta property="og:type" content="article">\n'
  printf '</head>\n<body>\n'
  sed '1,/^<\/style>$/d' site/_page.html
  printf '</body>\n</html>\n'
} > docs/index.html
echo "built docs/index.html ($(wc -c < docs/index.html) bytes)"
