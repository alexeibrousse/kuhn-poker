# Kuhn Poker

A small playground for experimenting with agents that play the simplified card game **Kuhn poker**.  The project
includes a game engine, several rule-based opponents, and two different implementations of a neural network policy (NumPy and
PyTorch) with the training and analysis tools that come with it.

## Installation

```bash
pip install -r requirements.txt
```

## Train a policy network

Pick either of the training scripts:

### NumPy version

```bash
python scripts/numpy_train.py
```

Trains a NumPy-based policy network and writes logs, plots, and a summary to `data/numpy-nn/<timestamp>/`. When training
finishes it automatically runs `numpy_analysis.py` to generate graphs.

### PyTorch version

```bash
python scripts/torch_train.py
```

Trains a PyTorch network and saves results to `data/pytorch-nn/<timestamp>/`. Afterward `torch_analysis.py` is invoked to
produce plots and a summary file.

## Analyze an existing run

To recreate graphs for a past run, supply the run directory to the appropriate analysis script:

```bash
python scripts/numpy_analysis.py path/to/run
# or
python scripts/torch_analysis.py path/to/run
```

## Play agents against each other

```bash
python scripts/bot_vs_bot.py --agent1 nash --agent2 random --episodes 1000
```

The match history and summary statistics are written to `data/bot_vs_bot/<timestamp>/`.

## Repository structure

- `subpoker/` – core game engine, agent implementations and neural network helpers
- `scripts/` – training, analysis and simulation utilities
- `Images/` – diagrams of the game and agent strategies
- `data/` – output from training or simulations (created when scripts run)
