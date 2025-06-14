import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data")
history_file = os.path.join(DATA_DIR, "training_history.csv")

if not os.path.exists(history_file):
    raise FileNotFoundError(f"{history_file} not found")

history = pd.read_csv(history_file)

# Compute metrics
history["bluff_rate"] = (
    history["1_bet"] + history["2_bet"]
) / (
    history[[f"{h}_{a}" for h in (1, 2) for a in ("check", "call", "bet", "fold")]].sum(axis=1)
)
history["value_bet_rate"] = history["3_bet"] / (
    history[[f"3_{a}" for a in ("check", "call", "bet", "fold")]].sum(axis=1)
)
history["call_rate"] = (
    history["1_call"] + history["2_call"] + history["3_call"]
) / (
    history[[f"{h}_{a}" for h in (1, 2, 3) for a in ("call", "fold")]].sum(axis=1)
)

plt.figure()
plt.plot(history["episode"], history["bluff_rate"], label="Bluff rate")
plt.plot(history["episode"], history["value_bet_rate"], label="Value bet rate")
plt.plot(history["episode"], history["call_rate"], label="Call rate")
plt.xlabel("Episode")
plt.ylabel("Rate")
plt.title("Strategy Metrics Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "strategy_metrics.png"))

