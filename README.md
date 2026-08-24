# Mathematical Modeling with Technology

This repository contains the lecture notebooks and the Quarto book that publishes them as a GitHub Pages site.

## Build locally

Install [Quarto](https://quarto.org/) and a Python/Jupyter environment, then run from the repository root:

```bash
quarto preview
```

To create the static site in `_site/`:

```bash
quarto render
```

The public book uses the saved notebook outputs and does not execute Python during rendering. Run a notebook locally when you want to regenerate its outputs.

## Module 0 time-series notebooks

The original full-day notebooks in `Lectures/Module0_TimeSeries/day*_student.ipynb` and
`day*_instructor.ipynb` are retained as source material. The shorter class-session
notebooks are generated with:

```bash
python scripts/split_time_series_notebooks.py
```

Public notebooks use numbered filenames without an audience suffix. Matching solution
notebooks use the same stem plus `_instructor` and are intentionally omitted from the
Quarto chapter list.

To add or refresh the **Open in Colab** badge in every public notebook listed in
`_quarto.yml`, run:

```bash
python scripts/add_colab_badges.py
```

The Quarto chapter list is the source of truth, so this command does not modify
instructor notebooks or unlisted source notebooks.

## GitHub Pages

The workflow in `.github/workflows/quarto-publish.yml` renders the book and publishes it to the `gh-pages` branch whenever `main` changes. In the repository settings, set **Pages → Build and deployment → Source** to **Deploy from a branch**, choose `gh-pages`, and use the `/ (root)` folder.

The expected project URL is:

<https://mbanuelos.github.io/grad_math_modeling/>

The book structure and visual conventions are defined in [`_quarto.yml`](_quarto.yml), [`index.qmd`](index.qmd), [`preface.qmd`](preface.qmd), and [`styles.css`](styles.css).
