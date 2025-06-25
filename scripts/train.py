import os
import sys
import json
from datetime import datetime
import subprocess
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the parent directory is in the path for module imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from subpoker.engine import KuhnPokerEnv
from subpoker.agents import NashAgent, RuleBasedAgent
from subpoker.neural_net import NeuralNet


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data")
RUNS_BASE = os.path.join(DATA_DIR, "numpy-nn")
os.makedirs(RUNS_BASE, exist_ok=True)
run_name = datetime.now().strftime("%d-%m-%y_%H-%M")
RUN_DIR = os.path.join(RUNS_BASE, run_name)
os.makedirs(RUN_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description="Train the neural network agent")
parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
args = parser.parse_args()

used_seed = args.seed if args.seed is not None else random.randint(0, 2**32 -1)
used_seed = 762523726


random.seed(used_seed)
np.random.seed(used_seed)

env = KuhnPokerEnv(used_seed)
state = env.reset()


n_epochs = 2000000
log_interval = n_epochs // 100
nn = NeuralNet(input_size=19, hidden_size=70, output_size=3, learning_rate=5e-6)
agent = RuleBasedAgent()
player_number = 0

# Record metadata about the network configuration
metadata = {
    "implementation": "numpy",
    "input_size": nn.input_size,
    "hidden_sizes": [nn.hidden_size],
    "activation": "ReLU",
    "learning_rate": nn.lr,
    "output_size": nn.output_size,
    "n_epochs": n_epochs,
    "log_interval": log_interval,
    "agent": agent.name,
    "player": player_number,
    "seed": used_seed,
}
with open(os.path.join(RUN_DIR, "config.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)


def encode_state(state: dict) -> np.ndarray:
    """
    Converts the environment state into a neural network input vector.
    """
    hand = state["hand"]
    player = state["player"]
    history = state["history"]

    # One-hot encoding the card 
    card_vec = [0, 0, 0]
    card_vec[hand -1] = 1

    action_index = {"check": 0, "call": 1, "bet": 2, "fold": 3, "none": 4}
    history_mat = np.zeros((3,5), dtype= int)

    for i in range(3): # 3 slots as 3 actions max per round.
        action = history[i] if i < len(history) else "none"
        history_mat[i, action_index[action]] = 1     
    return np.concatenate([card_vec, [player], history_mat.ravel()]) #History of ongoing round


def nnbot(state: dict) -> tuple: # Playing the round for the neural network
    X = encode_state(state)

    #Putting check and call together
    legal = env.legal_actions() # Returns the legal actions available
    action_map = ["check/call", "bet", "fold"]
    legal_indices = [] 
    if "check" in legal or "call" in legal: # Append the legal indices
        legal_indices.append(0)
    if "bet" in legal:
        legal_indices.append(1)
    if "fold" in legal:
        legal_indices.append(2)
    
    chosen_actions = [action_map[i] for i in legal_indices] # Keep the legal actions
    chosen_indices = legal_indices # For clarity

    probs = nn.forward(X)
    filtered_probs = probs[chosen_indices] # Removing the probabilities of illegal actions
    
    total = np.sum(filtered_probs)
    if total == 0:
        # Uniform probability over legal actions if all logits are zero
        filtered_probs = np.ones_like(filtered_probs) / len(filtered_probs)
    else:
        filtered_probs /= total # Normalizing these probabilities

    action = np.random.choice(chosen_actions, p=filtered_probs) # Choosing one action
    action_index = chosen_indices[chosen_actions.index(action)]

    # Convert unified label back into an actual valid action
    if action == "check/call":
        if "check" in env.legal_actions():
            real_action = "check"
        elif "call" in env.legal_actions():
            real_action = "call"
        else:
            raise ValueError("Neither 'check' nor 'call' is legal — invalid state")
    else:
        real_action = action

    return real_action, X, probs, action_index



# Data gathering
action_log = pd.DataFrame(
    0, index=[1, 2, 3], columns=["check", "call", "bet", "fold"], dtype=int
)

win_loss_log = {
    "wins": 0,
    "losses": 0,
    "reward_won": 0,
    "reward_lost": 0,
}

# Additional counters for bluff and fold analysis
bluff_wins = 0
bluff_losses = 0
bets_by_us = 0
folds_after_our_bet = 0


episode_rewards = []  # Rewards after one round
average_rewards = []
baseline = 0.0
history_records = [] # Summary for analysis
episode_history = [] # Full history of episodes

# Beginning of the training

for e in range(n_epochs):
    state = env.reset()
    done = False
    trajectory = []
    reward = 0
    bet_bluff = False
    last_was_bet_by_us = False


    while not done:
        if state["player"] == player_number:
            action, X, probs, action_index = nnbot(state)
            trajectory.append((X, action_index, probs))
            hand = state["hand"]
            action_log.loc[hand, action] += 1
            if action == "bet":
                bets_by_us += 1
                last_was_bet_by_us = True
                if hand in (1, 2):
                    bet_bluff = True
            else:
                last_was_bet_by_us = False
        else:
            legal = env.legal_actions()
            action = agent.act(state, legal)

        state, step_rewards, done, _ = env.step(action)
        reward = step_rewards[0]
    
    if done:
        if reward > 0:
            win_loss_log["wins"] += 1
            win_loss_log["reward_won"] += reward
        else:
            win_loss_log["losses"] += 1
            win_loss_log["reward_lost"] += -reward
        
        if bet_bluff:
            if reward > 0:
                bluff_wins += 1
            elif reward < 0:
                bluff_losses += 1


    baseline = 0.90 * baseline + 0.10 * reward # Update the baseline to reduce variance for backpropagation.

    if trajectory:
        dW1 = np.zeros_like(nn.W1) # Sum of all gradients in one episode.
        db1 = np.zeros_like(nn.b1)
        dW2 = np.zeros_like(nn.W2)
        db2 = np.zeros_like(nn.b2)

        advantage = reward - baseline
    
        for X, action_index, probs in trajectory:
            gW1, gb1, gW2, gb2 = nn.backward(X, action_index, advantage, probs) # Gradients for single step.
            dW1 += gW1 
            db1 += gb1
            dW2 += gW2
            db2 += gb2
        
        dW1 /= len(trajectory) # Average gradient per episode
        db1 /= len(trajectory)
        dW2 /= len(trajectory)
        db2 /= len(trajectory)
        nn.update(dW1, db1, dW2, db2)

    episode_rewards.append(reward)

    episode_history.append({"episode": e + 1, "hand": env.hands[player_number], 
                            "opp_hand": env.hands[1 - player_number], 
                            "history": "-".join(env.history), 
                            "reward": reward})


    if (e + 1) % log_interval == 0:
        avg = np.mean(episode_rewards[-log_interval:])
        average_rewards.append(avg)
        record = {
            "episode": e + 1,
            "avg_reward": avg,
            "wins": win_loss_log["wins"],
            "losses": win_loss_log["losses"],
        }
        for hand in (1, 2, 3):
            for act in ("check", "call", "bet", "fold"):
                record[f"{hand}_{act}"] = int(action_log.loc[hand, act]) # type: ignore
        history_records.append(record)
        action_log.loc[:, :] = 0
        win_loss_log = {"wins": 0, "losses": 0, "reward_won": 0, "reward_lost": 0}


# Creating full csv file
df_full = pd.DataFrame(episode_history)
df_full.to_csv(os.path.join(RUN_DIR, "full_episode_history.csv"), index=False)

# Creating summary csv file
df_history = pd.DataFrame(history_records)
df_history.to_csv(os.path.join(RUN_DIR, "training_history.csv"), index=False)



# Calculating statistics
total_reward = sum(episode_rewards)
total_wins = sum(r > 0 for r in episode_rewards)
total_losses = sum(r < 0 for r in episode_rewards)
win_rate = total_wins / (total_wins + total_losses)
last_10pct_index = int(len(episode_rewards) * 0.9)
avg_reward_last_10pct = float(np.mean(episode_rewards[last_10pct_index:]))

# Bluff rates
if not df_history.empty:
    recent_rows = max(1, int(len(df_history) * 0.1))
    recent = df_history.tail(recent_rows)
    bluff_bets = (recent["1_bet"] + recent["2_bet"]).sum()
    action_cols = [f"{h}_{a}" for h in (1, 2) for a in ("check", "call", "bet", "fold")]
    bluff_denominator = recent[action_cols].sum().sum()
    bluff_rate_last_10pct = float(bluff_bets / bluff_denominator) if bluff_denominator else 0.0
else:
    bluff_rate_last_10pct = 0.0
bluff_win_rate = bluff_wins / (bluff_wins + bluff_losses) if (bluff_wins + bluff_losses) > 0 else 0.0

fold_frequency = folds_after_our_bet / bets_by_us if bets_by_us else 0.0



summary = {
    "total_reward": total_reward,
    "total_wins": total_wins,
    "total_losses": total_losses,
    "win_rate": win_rate,
    "avg_reward_last_10pct": avg_reward_last_10pct,
    "bluff_rate_last_10pct": bluff_rate_last_10pct,
    "bluff_wins": bluff_wins,
    "bluff_losses": bluff_losses,
    "bluff_win_rate": bluff_win_rate,
    "fold_frequency_after_bet": fold_frequency,
}

for key, value in summary.items():
    if isinstance(value, float):
        summary[key] = round(value, 3)

with open(os.path.join(RUN_DIR, "training_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)



# Plotting learning curve
plt.plot(range(log_interval, n_epochs + 1, log_interval), average_rewards)
plt.xlabel("Episode")
plt.ylabel("Average Reward (last %d)" % log_interval)
plt.title("Neural Network Learning Progress")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, "learning_curve.png"))
plt.close()


script_path = os.path.join(os.path.dirname(__file__), "analyze_training.py")
subprocess.run([sys.executable, script_path, RUN_DIR], check=True)
