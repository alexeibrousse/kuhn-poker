"""
This program loads a PyNet model from a timestamped directory and is evaluated against a NashAgent with alpha=0.333 in inference mode.
Fairness to the testing is ensured by using a 12-cycle of the 6 possible hands with seat mirroring, 
being repeated 1000 times per random seed, with 6 different fixed seeds.
The file receives one argument, the run directory containing the ``model.pth`` and ``config.json``files.
"""

import os
import sys
import json

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
import torch
from torch.distributions import Categorical

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from subpoker.engine import KuhnPokerEnv
from subpoker.pytorch_nn import PyNet
from subpoker.agents import NashAgent


# —————— Utils —————— #

def fix_seed(seed) -> None:
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)

if len(sys.argv) < 2:
        raise ValueError("Please provide the run directory containing model.pth and config.json.")

RUN_DIR = sys.argv[1]
TRAINING_DIR = os.path.join(RUN_DIR, "training")
TESTING_DIR = os.path.join(RUN_DIR, "testing")
os.makedirs(TESTING_DIR, exist_ok=True)



# —————— Config & Constants —————— #

SEEDS = [1337, 271828, 314159, 8675309, 29461070]
EPISODES_CYCLES = 10000
DEALS = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)]
FIRST_PLAYERS = [0, 1]

ACTION_MAP = ["check", "bet", "call", "fold"]
ACTION_INDICES = {a: i for i, a in enumerate(ACTION_MAP)}

ALL_ACTION_MASKS = {}

def add_mask(actions):
    mask = torch.zeros(len(ACTION_MAP), dtype=torch.float32)
    for a in actions:
        mask[ACTION_INDICES[a]] = 1.0
    ALL_ACTION_MASKS[frozenset(actions)] = mask

add_mask(["check", "bet"])
add_mask(["call", "fold"])


VALID_HISTORIES = {
    (): 0,
    ("check",): 1,
    ("bet",): 2,
    ("check", "bet"): 3,
}



# —————— Helper functions —————— #  

def load_model(cfg,random_seed: int) -> PyNet:
    model = PyNet(cfg["input_size"], cfg["hidden_size"], cfg["output_size"], cfg["initial_learning_rate"], random_seed)
    state_dict = torch.load(os.path.join(TRAINING_DIR, "model.pth"), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model



def encode_state(state: dict) -> torch.Tensor:
    """Encode the game *state* into a tensor representation."""
    hand = state["hand"]
    hand_vec = [1.0 if (i + 1) == hand else 0 for i in range(3)]
    history = tuple(state["history"])
    history_index = VALID_HISTORIES[history]
    history_vec = [1.0 if i == history_index else 0.0 for i in range(4)]
    return torch.tensor(hand_vec + history_vec, dtype=torch.float32)



def sample_action(probs: torch.Tensor, legal_actions: list[str]) -> str:
    """Sample an action index from *probs* restricted to *legal_actions*."""
    mask = ALL_ACTION_MASKS[frozenset(legal_actions)]
    masked = probs * mask
    total = masked.sum()
    if total.item() == 0:
        masked = mask / mask.sum()
    else:
        masked /= total
    dist = Categorical(masked)
    index = dist.sample().item()
    return ACTION_MAP[index]  # type: ignore


# —————— Main —————— #

def main():
    with open(os.path.join(RUN_DIR, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    rewards_list = []
    logs = []

    bet = 0
    check = 0
    call = 0
    fold = 0

    with torch.inference_mode():
        for seed in SEEDS:
            env = KuhnPokerEnv(seed)
            nn = load_model(cfg, seed)
            agent = NashAgent(alpha=0.333, random_seed=seed)
            torch.manual_seed(seed)

            for _ in trange(EPISODES_CYCLES, desc="Testing Epochs"):
                for fp in FIRST_PLAYERS:
                    for h0, h1 in DEALS:
                        env.reset()
                        env.hands = [h0, h1]
                        env.first_player = fp
                        done = False

                        while not done:
                            state = env.get_state()
                            legal = env.legal_actions()

                            if state["player"] == 0:
                                x = encode_state(state)
                                probs = nn.forward(x)
                                action = sample_action(probs, legal)
                            
                            else:
                                action = agent.act(state, legal)
                            
                            state, rewards, done, _ = env.step(action)
                        
                        reward = rewards[0]  # type: ignore
                        rewards_list.append(reward)
                        history = env.history
                        logs.append({
                        "hand": env.hands[0],
                        "opp_hand": env.hands[1],
                        "first_to_act": fp,
                        "history": "-".join(history),
                        "reward": reward
                    })
                    
                        if fp == 0:
                            first_action = history[0]
                        else:
                            first_action = history[1] if len(history) > 1 else None
                        if first_action == "bet":
                            bet += 1
                        elif first_action == "check":
                            check += 1
                        
                        respond = None
                        if fp == 0 and len(history) >= 3 and history[0] == "check" and history[1] == "bet":
                            respond = history[2]
                        elif fp == 1 and len(history) >= 2 and history[0] == "bet":
                            respond = history[1]
                        if respond == "call":
                            call += 1
                        elif respond == "fold":
                            fold += 1
                        
    total_open = bet + check
    total_respond = call + fold
    bet_pct = bet / total_open if total_open else 0.0
    check_pct = check / total_open if total_open else 0.0
    call_pct = call / total_respond if total_respond else 0.0
    fold_pct = fold / total_respond if total_respond else 0.0

    summary = pd.DataFrame({
        "action": ["bet", "check", "call", "fold"],
        "percentage": [bet_pct * 100, check_pct * 100, call_pct * 100, fold_pct * 100],
    })
    summary.to_csv(os.path.join(TESTING_DIR, "summary.csv"), index=False)

    avg_series = pd.Series(rewards_list).expanding(200).mean()
    plt.figure()
    plt.plot(avg_series)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Average Reward Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(TESTING_DIR, "avg_reward.pdf"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].bar(["bet", "check"], [bet_pct * 100, check_pct * 100])
    axes[0].set_title("Opening Action %")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Percentage")
    axes[1].bar(["call", "fold"], [call_pct * 100, fold_pct * 100])
    axes[1].set_title("Response Action %")
    axes[1].set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(os.path.join(TESTING_DIR, "strategy_metrics.pdf"))
    plt.close(fig)

    pd.DataFrame(logs).to_csv(os.path.join(TESTING_DIR, "full_testing_data.csv"), index=False)



if __name__ == "__main__":
    main()

