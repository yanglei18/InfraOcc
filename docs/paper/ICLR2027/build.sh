#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TLROOT="${TLROOT:-$HOME/.local/texlive-apt}"

if [[ -x "$TLROOT/usr/bin/latexmk" ]]; then
  export PATH="$TLROOT/usr/bin:$PATH"
  export TEXMFCNF="$TLROOT/usr/share/texlive/texmf-dist/web2c:$TLROOT/usr/share/texmf/web2c:"
  export TEXMFROOT="$TLROOT/usr/share/texlive"
  export TEXMFDIST="$TLROOT/usr/share/texlive/texmf-dist"
  export TEXMFDEBIAN="$TLROOT/usr/share/texmf"
  export TEXMFLOCAL="$TLROOT/usr/local/share/texmf"
  export TEXMFSYSVAR="$TLROOT/var/lib/texmf"
  export TEXMFSYSCONFIG="$TLROOT/etc/texmf"
  export TEXMFVAR="$TLROOT/texmf-var"
  export TEXMFCONFIG="$TLROOT/texmf-config"
  export TEXMF="{$TEXMFCONFIG,$TEXMFVAR,$HOME/texmf,$TEXMFLOCAL,$TEXMFSYSCONFIG,$TEXMFSYSVAR,$TEXMFDEBIAN,$TEXMFDIST}"
  export VARTEXFONTS="$TLROOT/var/cache/fonts"
fi

cd "$HERE"
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
