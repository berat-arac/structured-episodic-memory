# Structured Episodic Memory (SEM) paper source

This folder is Overleaf / standard LaTeX compatible.

## Layout
- Page 1: title page
- Page 2: contents
- Page 3: abbreviations and notation
- Main paper: two-column academic layout with abstract, numbered sections, figures, tables, references, consistent paragraph indentation, and an implementation-details appendix

## Files
- `paper.tex` - paper source
- `references.bib` - bibliography source kept with the release
- `figures/*.pdf` - vector figures generated from experiment outputs
- `Structured_Episodic_Memory_SEM.pdf` - compiled 13-page preview

## Compile
Run:

```bash
latexmk -pdf paper.tex
```

or upload the entire folder to Overleaf.

## Before public release
At the end of `paper.tex`, replace:

`https://PLACEHOLDER-SEM-PROJECT-LINK`

with the final GitHub, project page, or Zenodo record.

The paper title is intentionally exactly `Structured Episodic Memory (SEM)`.
