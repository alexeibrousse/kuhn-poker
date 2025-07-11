# kuhn-poker

This project implements several agents for Kuhn poker, including a simple neural network learner.

## Training the neural network

Run `scripts/train.py` to train the NumPy-based policy network. The script logs statistics 1000 times per run and saves them to the `Data` directory:

```
python scripts/train.py
```

Files in `Data/numpy-nn/<timestamped-run>/` include:
- `config.json` – hyperparameters and run configuration
- `full_episode_history.csv` – per-episode logs of all moves and rewards
- `learning_curve.pdf` – plot of average reward over time
- `strategy_metrics.pdf` – plot of bluff/call/fold trends
- `training_history.csv` – statistics every N episodes
- `training_summary.json` – summary metrics from the run

## Analysing the training history

To generate additional graphs showing bluff and call rates over time, run:

```
python scripts/analyze_training.py
```

This reads `training_history.csv` and saves `strategy_metrics.png` into the same `Data` folder.
