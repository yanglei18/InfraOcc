# InfraOcc TPAMI 2026 Paper Source

## Entry Point

- `main.tex`: paper entry point.
- `main.pdf`: compiled preview, regenerated from `main.tex`.
- `egbib.bib`: bibliography database.
- `IEEEtran.cls`: IEEE Computer Society journal class.

## Source Layout

- `latex/1-intro.tex` to `latex/5-conclusion.tex`: section text.
- `latex/table_macros.tex`: shared table macros.
- `latex/authors.tex`: reserved author metadata placeholder.
- `latex/fig/figXX-*.tex`: figure wrappers, numbered by first appearance.
- `latex/tab/tabXX-*.tex`: table wrappers, numbered by first appearance.
- `figures/figXX-*`: figure assets used by the wrappers.
- `resources/`: supporting source artifacts not directly compiled.

## Figure Format Policy

- Use PNG for raster images, rendered occupancy maps, and composed qualitative figures.
- Use PDF for vector plots such as robustness curves.
- Keep only the format referenced by the corresponding `latex/fig/figXX-*.tex` file.

## Build

Run from this directory:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
