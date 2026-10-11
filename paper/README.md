# Manuscript

*How Many Bits Does a Training Example Cost? Per-Example Capacity versus Fidelity via
Linearized Laplace Unlearning* — [capacity_fidelity.pdf](capacity_fidelity.pdf)

Build: `pdflatex capacity_fidelity.tex` (run twice). Set `\anonymoustrue` near the top of the
`.tex` file for double-blind review; this hides the author names and the repository link.

Regenerate the figures (about 12 minutes on one CPU core) from the repository root:

    python examples/paper/reproduce_paper.py --out paper/figures
