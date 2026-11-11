# NBA moneyline

Pre-game logistic model for whether the home team wins a regular-season game. Feature definitions are in [docs/feature-selection.md](docs/feature-selection.md). The Streamlit app has a daily slate (model probability and the margin versus the DraftKings line) and a profit tab for the holdout closes.

## Setup

```bash
conda env create -f environment.yml
conda activate nba_moneyline
```

Run commands from the project root.

## Pipeline

```bash
PYTHONPATH=src python -m nba_moneyline acquire
PYTHONPATH=src python -m nba_moneyline player-boxes
PYTHONPATH=src python -m nba_moneyline features
PYTHONPATH=src python -m nba_moneyline train
PYTHONPATH=src python -m nba_moneyline closing-odds --season 2025
```

`closing-odds` stores DraftKings closes for eligible games in that season (`2025` is the 2025–26 holdout, `2026` is the current season).

## App

```bash
PYTHONPATH=src streamlit run app.py
```

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
