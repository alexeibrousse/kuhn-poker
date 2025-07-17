import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "numpy-nn")


def load_run_dir(path: str | None) -> str:
    """Return path to run directory. If ``path`` is None, use latest."""
    if path:
        return path
    if not os.path.exists(BASE_DIR):
        raise FileNotFoundError(f"{BASE_DIR} does not exist")
    runs = sorted(d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d)))
    if not runs:
        raise FileNotFoundError(f"No runs found in {BASE_DIR}")
    return os.path.join(BASE_DIR, runs[-1])


def parse_episode(row: pd.Series) -> tuple[bool, bool, bool, bool, int, int, int]:
    """Extract episode metrics from a row.

    Returns a tuple ``(bluff, value_bet, call, responded_to_bet, jack_hand, queen_hand, king_hand)``.
    ``bluff`` is True if we bet with J or Q and opponent responded.
    ``value_bet`` is True if we bet with K and opponent responded.
    ``call`` is True if our last action was a call.
    ``responded_to_bet`` indicates we faced a bet and either called or folded.
    Also returns indicators for having each hand.
    """
    history = row.get("history", "")
    actions = history.split("-") if isinstance(history, str) and history else []
    first_to_act = int(row.get("first_to_act", 0))
    hand = int(row.get("hand", 0))

    bluff = False
    value_bet = False
    call = False
    responded = False

    if actions:
        last_idx = len(actions) - 1
        actor_last = (first_to_act + last_idx) % 2
        last_action = actions[-1]
        if actor_last == 0 and last_action in ("call", "fold"):
            responded = True
            if last_action == "call":
                call = True

        if len(actions) >= 2:
            bet_idx = len(actions) - 2
            actor_bet = (first_to_act + bet_idx) % 2
            if actor_bet == 0 and actions[bet_idx] == "bet":
                if hand in (1, 2):
                    bluff = True
                elif hand == 3:
                    value_bet = True

    return bluff, value_bet, call, responded, hand == 1, hand == 2, hand == 3


def analyze(df: pd.DataFrame, n_epochs: int, run_dir: str) -> None:
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
            b, vb, c, r, j, q, k = parse_episode(row)
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

    # 3. Bluff rate per card
    plt.figure()
    plt.plot(episodes, jack_bluff_rates, label="Jack bluff rate")
    plt.plot(episodes, queen_bluff_rates, label="Queen bluff rate")
    plt.xlabel("Episode")
    plt.ylabel("Rate")
    plt.title("Bluff Rate by Card")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "bluff_rate_cards.pdf"))
    plt.close()

    # 4. Baseline vs reward
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

    # 5. Gradient norm
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
    run_dir = load_run_dir(sys.argv[1] if len(sys.argv) > 1 else None)
    data_file = os.path.join(run_dir, "full_training_data.csv")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"{data_file} not found")
    df = pd.read_csv(data_file)

    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        n_epochs = int(cfg.get("number_epochs", len(df)))
    else:
        n_epochs = len(df)

    analyze(df, n_epochs, run_dir)


if __name__ == "__main__":
    main()