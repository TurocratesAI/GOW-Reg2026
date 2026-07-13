# REG2026 challenge paper (MICCAI 2026)

LaTeX source for the Grounded Ontology Walker paper. Springer LNCS format, anonymized for double-blind review.

## Build

The paper needs the Springer LNCS class files `llncs.cls` and `splncs04.bst`.

- Overleaf (recommended): create a new project from the "Springer LNCS" template, then upload the contents of
  this folder (`main.tex`, `references.bib`, `tables/`, `figures/`). The template provides the class files.
- Local: place `llncs.cls` and `splncs04.bst` in this folder, then run
  `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## Layout

- `main.tex` the paper.
- `references.bib` bibliography (verify each entry against the canonical citation before camera-ready).
- `tables/*.tex` result tables, regenerated from the eval CSV by `../gow/eval/make_tables.py` (do not hand-edit).
- `figures/*` figures (pipeline schematic, a reasoning-chain example, QC/grounding crops, the OOD separation
  plot). Generate the data plots with `../gow/eval/plot_ood.py`.

## Rules (enforced)

- Double-blind: no author names, affiliations, repo URLs, grant IDs, or self-identifying acknowledgments in the
  review PDF. Add the author block only at camera-ready.
- LNCS template is not modified (no margin, spacing, or font changes). Limit: 8 pages of content plus 2 pages of
  references.
- No em or en dashes anywhere, including LaTeX `--` and `---`. Use single hyphens or rephrase.
- Every reported number traces to a committed eval artifact; tables are generated, not hand-typed.
