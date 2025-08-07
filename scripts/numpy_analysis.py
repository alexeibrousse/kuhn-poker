"""
Analyzes logged data from a training run of the NumNet agent and generates graphs.
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_run_dir, parse_episode



def analyze(df: pd.DataFrame, n_epochs: int, run_dir: str) -> None:
    """
    Computes various graphs and statistics from the training data and generates plots.
    """
    interval = max(1, n_epochs // 100)
    episodes = []
    avg_rewards = []
    bluff_rates = []
    call_rates = []
    value_bet_rates = []
    jack_bluff_rates = []
    queen_bluff_rates = []
    baseline_means = []
    reward_means = []
    grad_norm_means = []

    for start in range(0, len(df), interval):
        chunk = df.iloc[start:start + interval]
        if chunk.empty:
            continue
        episodes.append(int(chunk["episode"].iloc[-1]))
        reward_means.append(chunk["reward"].mean())
        baseline_means.append(chunk["baseline"].astype(float).mean())
        grad_norm_means.append(chunk["gradient norm"].astype(float).mean())

        bluff = 0
        value_bet = 0
        call = 0
        responded = 0
        jack_bluff = 0
        queen_bluff = 0
        jack_count = 0
        queen_count = 0
        king_count = 0
        hands12 = 0

        for _, row in chunk.iterrows():
            b, vb, c, f, r, j, q, k = parse_episode(row)
            bluff += int(b)
            value_bet += int(vb)
            call += int(c)
            responded += int(r)
            if j:
                jack_count += 1
                if b:
                    jack_bluff += 1
            if q:
                queen_count += 1
                if b:
                    queen_bluff += 1
            if k:
                king_count += 1
            if j or q:
                hands12 += 1

        bluff_rates.append(bluff / hands12 if hands12 else 0.0)
        value_bet_rates.append(value_bet / king_count if king_count else 0.0)
        call_rates.append(call / responded if responded else 0.0)
        jack_bluff_rates.append(jack_bluff / jack_count if jack_count else 0.0)
        queen_bluff_rates.append(queen_bluff / queen_count if queen_count else 0.0)
        avg_rewards.append(chunk["reward"].mean())


    
    # Summary for last 10% of episodes
    recent_count = max(1, int(len(df) * 0.1))
    recent = df.tail(recent_count)

    avg_reward_last = float(recent["reward"].mean())
    win_rate_last = float((recent["reward"] > 0).mean())

    bluff_total = bluff_success = 0
    call_total = call_success = 0
    faced_bet = fold_total = 0
    for _, row in recent.iterrows():
        bluff, _, call, fold, responded, *_ = parse_episode(row)
        if bluff:
            bluff_total += 1
            if row.get("reward", 0) > 0:
                bluff_success += 1
        if call:
            call_total += 1
            if row.get("reward", 0) > 0:
                call_success += 1
        if responded:
            faced_bet += 1
            if fold:
                fold_total += 1

    summary = {
        "average_reward": round(avg_reward_last, 4),
        "win_rate": round(win_rate_last, 4),
        "bluff_success_rate": round(bluff_success / bluff_total, 4) if bluff_total else 0.0,
        "call_success_rate": round(call_success / call_total, 4) if call_total else 0.0,
        "fold_frequency_after_bet": round(fold_total / faced_bet, 4) if faced_bet else 0.0,
    }

    with open(os.path.join(run_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)




    # 1. Average reward curve
    plt.figure()
    plt.plot(episodes, avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Average Reward Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "avg_reward.pdf"))
    plt.close()



    # 2. Strategy metrics
    plt.figure()
    plt.plot(episodes, bluff_rates, label="Bluff rate")
    plt.plot(episodes, value_bet_rates, label="Value bet rate")
    plt.plot(episodes, call_rates, label="Call rate")
    plt.xlabel("Episode")
    plt.ylabel("Rate")
    plt.title("Strategy Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "strategy_metrics.pdf"))
    plt.close()



    # 3. Jack bluff rate
    plt.figure()
    plt.plot(episodes, jack_bluff_rates, label="Jack bluff rate")
    plt.xlabel("Episode")
    plt.ylabel("Rate")
    plt.title("Jack Bluff Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "jack_bluff_rate.pdf"))
    plt.close()



    # 4. Queen bluff rate
    plt.figure()
    plt.plot(episodes, queen_bluff_rates, label="Queen bluff rate")
    plt.xlabel("Episode")
    plt.ylabel("Rate")
    plt.title("Queen Bluff Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "queen_bluff_rate.pdf"))
    plt.close()



    # 5. Baseline vs reward
    plt.figure()
    plt.plot(episodes, baseline_means, label="Baseline")
    plt.plot(episodes, reward_means, label="Reward")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("Baseline vs Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "baseline_vs_reward.pdf"))
    plt.close()



    # 6. Gradient norm
    plt.figure()
    plt.plot(episodes, grad_norm_means, label="Gradient norm")
    plt.xlabel("Episode")
    plt.ylabel("Norm")
    plt.title("Average Gradient Norm")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "grad_norm.pdf"))
    plt.close()





def main() -> None:
    """
    Main function, loads the run directory and analyze the training data.
    """
    run_dir = load_run_dir(sys.argv[1] if len(sys.argv) > 1 else None)
    data_file = os.path.join(run_dir, "full_training_data.csv")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"{data_file} not found")
    df = pd.read_csv(data_file)
    n_epochs = len(df)

    analyze(df, n_epochs, run_dir)



if __name__ == "__main__":
    main()

