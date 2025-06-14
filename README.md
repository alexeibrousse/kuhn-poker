# kuhn-poker

This project implements several agents for Kuhn poker, including a simple neural network learner.

## Training the neural network

Run `scripts/train.py` to train the NumPy-based policy network. The script logs statistics every 1000 episodes and saves them to the `Data` directory:

```
python scripts/train.py
```

Artifacts written to `Data/`:

- `network_config.json` – hyperparameters of the network used during training
- `training_history.csv` – snapshot of action counts and rewards every 1000 episodes
- `learning_curve.png` – plot of average reward over time

## Analysing the training history

To generate additional graphs showing bluff and call rates over time, run:

```
python scripts/analyze_training.py
```

This reads `training_history.csv` and saves `strategy_metrics.png` into the same `Data` folder.
