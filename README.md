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
PYTHONPATH=src python -m nba_moneyline opening-odds
```

`closing-odds` stores the Action Network consensus close (book 15, the same book as the hockey model) from 2023–24 on. `opening-odds` stores the Action Network Open line (book 30) from 2024–25 on. The daily slate still uses the current DraftKings price.

## App

```bash
PYTHONPATH=src streamlit run app.py
```

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
