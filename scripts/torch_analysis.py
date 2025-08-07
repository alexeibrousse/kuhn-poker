"""
Analyzes logged data from a training run of the NumNet agent and generates graphs.
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

from utils import parse_episode


def analyze(df: pd.DataFrame, n_epochs: int, run_dir: str) -> None:
    """
    Computes various graphs and statistics from the training data and generates plots.
    """
    # Split data into chunks for smoothing
    interval = max(1, n_epochs // 100)
    episodes = []
    avg_rewards = []
    baseline_means = []
    grad_norm_means = []
    entropy_means = []
    lr_means = []
    p_check_means = []
    p_bet_means = []
    p_call_means = []
    p_fold_means = []
    call_rates = []
    bluff_rates = []
    value_bet_rates = []

    for start in range(0, len(df), interval):
        chunk = df.iloc[start:start + interval]
        if chunk.empty:
            continue
        episodes.append(int(chunk["episode"].iloc[-1]))
        avg_rewards.append(chunk["reward"].mean())
        baseline_means.append(chunk["baseline"].mean())
        grad_norm_means.append(chunk["grad_norm"].mean())
        entropy_means.append(chunk["entropy"].mean())
        lr_means.append(chunk["learning_rate"].mean())
        p_check_means.append(chunk["p_check"].mean())
        p_bet_means.append(chunk["p_bet"].mean())
        p_call_means.append(chunk["p_call"].mean())
        p_fold_means.append(chunk["p_fold"].mean())
    
        bluff = 0
        value_bet = 0
        call = 0
        responded = 0
        hands12 = 0
        king_count = 0
        for _, row in chunk.iterrows():
            b, vb, c, _, r, j, q, k = parse_episode(row)
            bluff += int(b)
            value_bet += int(vb)
            call += int(c)
            responded += int(r)
            if j or q:
                hands12 += 1
            if k:
                king_count += 1
        bluff_rates.append(bluff / hands12 if hands12 else 0.0)
        value_bet_rates.append(value_bet / king_count if king_count else 0.0)
        call_rates.append(call / responded if responded else 0.0)


    # Summary for last 10% of episodes
    recent_count = max(1, int(len(df) * 0.1))
    recent = df.tail(recent_count)
    avg_reward_last = float(recent["reward"].mean())
    win_rate_last = float((recent["reward"] > 0).mean())


    summary = {
        "average_reward": round(avg_reward_last, 4),
        "win_rate": round(win_rate_last, 4)
    }
    with open(os.path.join(run_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 1. Average reward over time
    plt.figure()
    plt.plot(episodes, avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Average Reward Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "avg_reward.pdf"))
    plt.close()

    # 2. Baseline vs Average Reward
    plt.figure()
    plt.plot(episodes, baseline_means, label="Baseline")
    plt.plot(episodes, avg_rewards, label="Average Reward")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("Baseline vs Average Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "baseline_vs_reward.pdf"))
    plt.close()

    # 3. Gradient norm
    plt.figure()
    plt.plot(episodes, grad_norm_means)
    plt.xlabel("Episode")
    plt.ylabel("Gradient Norm")
    plt.title("Average Gradient Norm")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "grad_norm.pdf"))
    plt.close()

    # 4. Entropy coefficient
    plt.figure()
    plt.plot(episodes, entropy_means)
    plt.xlabel("Episode")
    plt.ylabel("Entropy Coefficient")
    plt.title("Entropy Coefficient Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "entropy_coeff.pdf"))
    plt.close()

    # 5. Learning rate
    plt.figure()
    plt.plot(episodes, lr_means)
    plt.xlabel("Episode")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "learning_rate.pdf"))
    plt.close()

    # 6. First-move action probabilities
    plt.figure()
    plt.plot(episodes, p_check_means, label="p_check")
    plt.plot(episodes, p_bet_means, label="p_bet")
    plt.plot(episodes, p_call_means, label="p_call")
    plt.plot(episodes, p_fold_means, label="p_fold")
    plt.xlabel("Episode")
    plt.ylabel("Probability")
    plt.title("First-Move Action Probabilities")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "first_move_probs.pdf"))
    plt.close()

    # 7. Strategic action rates
    plt.figure()
    plt.plot(episodes, bluff_rates, label="Bluff rate")
    plt.plot(episodes, value_bet_rates, label="Value bet rate")
    plt.plot(episodes, call_rates, label="Call rate")
    plt.xlabel("Episode")
    plt.ylabel("Rate")
    plt.title("Strategic Action Rates")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "strategic_rates.pdf"))
    plt.close()



def main() -> None:
    """
    Main function, loads the run directory and analyze the training data.
    """
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    data_file = os.path.join(run_dir, "full_training_data.csv")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"{data_file} not found")
    df = pd.read_csv(data_file)

    analyze(df, len(df), run_dir)




if __name__ == "__main__":
    main()

