"""Train a simple policy network for Kuhn poker using PyNet, a PyTorch-based neural network implementation."""


import os
import sys
import subprocess

import pandas as pd
import random

import torch
from torch.distributions import Categorical

from tqdm import trange

from utils import create_run_dir, save_metadata, steps_to_threshold

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from subpoker.engine import KuhnPokerEnv
from subpoker.pytorch_nn import PyNet
from subpoker.agents import RuleBasedAgent, NashAgent



# —————— Utils —————— #

RUN_DIR = create_run_dir("pytorch-nn")

ACTION_MAP = ["check", "bet", "call", "fold"]

valid_histories = {
    (): 0,
    ("check",): 1,
    ("bet",): 2,
    ("check", "check"): 3,
    ("check", "bet"): 4,
    ("bet", "call"): 5,
    ("bet", "fold"): 6,
    ("check", "bet", "call"): 7,
    ("check", "bet", "fold"): 8,
}



# —————— Hyperparameters —————— #

EPOCHS = 300000
HIDDEN_SIZE = 70
LEARNING_RATE = 1e-4

LR_DECAY_RATE = 0.999
ENTROPY_COEFF = 1e-2
ENTROPY_DECAY = 0.999
GRADIENT_CLIP = 10.0
BASELINE_MOMENTUM = 0.10
BASELINE_BOUND = 10
BASELINE_DECAY = 0.999

RANDOM_SEED = 1



# —————— Feature Toggles —————— #

USE_LR_DECAY                = True 
USE_ENTROPY                 = False 
USE_ENTROPY_DECAY           = False 
USE_BASELINE_BOUND          = False 
USE_BASELINE_DECAY          = False 
USE_GRADIENT_CLIPPING       = False
RANDOM_REPRODUCIBILITY      = False



# —————— Environment and Reproducibility —————— #

if not RANDOM_REPRODUCIBILITY:
    RANDOM_SEED = random.randint(0, 2**32 - 1)

nn = PyNet(input_size=12, hidden_size=HIDDEN_SIZE, output_size=4, learning_rate=LEARNING_RATE, random_seed=RANDOM_SEED)
env = KuhnPokerEnv(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

agent = NashAgent()

PLAYER_NUMBER = 0



# —————— Metadata —————— #

metadata = {
    "implementation": "pytorch",
    "number_epochs": EPOCHS,
    "agent": agent.name,
    "input_size": nn.input_size,
    "hidden_size": nn.hidden_size,
    "output_size": nn.output_size,
    "initial_learning_rate": LEARNING_RATE,
    "activation": "ReLU",
    "lr_decay_rate": f"YES - {LR_DECAY_RATE}" if USE_LR_DECAY else "NO",
    "entropy_coeff": f"YES - {ENTROPY_COEFF}" if USE_ENTROPY else "NO",
    "entropy_decay_rate": f"YES - {ENTROPY_DECAY}" if USE_ENTROPY_DECAY else "NO",
    "baseline_momentum": BASELINE_MOMENTUM,
    "baseline_bound": BASELINE_BOUND if USE_BASELINE_BOUND else "NO",
    "baseline_decay": BASELINE_DECAY if USE_BASELINE_DECAY else "NO",
    "gradient_clip": GRADIENT_CLIP if USE_GRADIENT_CLIPPING else "NO",
    "random_seed": f"{RANDOM_SEED} - chosen" if RANDOM_REPRODUCIBILITY else f"{RANDOM_SEED} - random",
}   



# —————— Training —————— #

def encode_state(state: dict) -> torch.Tensor:
    """Encode the game *state* into a tensor representation."""
    hand = state["hand"]
    hand_vec = [1.0 if (i + 1) == hand else 0 for i in range(3)]
    history = tuple(state["history"])
    action_index = valid_histories[history]
    history_vec = [1.0 if i == action_index else 0.0 for i in range(9)]

    return torch.tensor(hand_vec + history_vec, dtype=torch.float32)


def sample_action(probs: torch.Tensor, legal_actions: list):
    """Sample an action from *probs* restricted to *legal_actions*."""
    mask = torch.zeros_like(probs)
    indices = [ACTION_MAP.index(action) for action in legal_actions]
    mask[indices] = 1.0

    masked_probs = probs * mask

    if masked_probs.sum().item() == 0:
        masked_probs = mask / mask.sum()
    else:
        masked_probs /= masked_probs.sum()
    
    dist = Categorical(masked_probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    entropy = dist.entropy()

    return action.item(), log_prob, entropy


def train():
    """Main training loop for PyNet."""
    baseline = 0.0

    save_metadata(metadata, RUN_DIR)
    logs = []  # Collect logs for analysis

    for e in trange(1, EPOCHS + 1, desc="Training Epochs"):
        state = env.reset()
        done = False
        saved_log_probs = []
        saved_entropies = []
        first_probs = None

        while not done:
            if state["player"] == PLAYER_NUMBER:
                x = encode_state(state)
                probs = nn.forward(x)
                legal_actions = env.legal_actions()

                if first_probs is None:
                    first_probs = probs.detach().cpu().numpy()

                action_index, log_prob, entropy = sample_action(probs, legal_actions)
                action = ACTION_MAP[action_index]  # type: ignore

                saved_log_probs.append(log_prob)
                saved_entropies.append(entropy)

                state, rewards, done, _ = env.step(action)
        
            else:
                with torch.no_grad():
                    legal = env.legal_actions()
                    action = agent.act(state, legal)
                    state, rewards, done, _ = env.step(action)
        
        reward = rewards[PLAYER_NUMBER]  # type: ignore
        
        if USE_BASELINE_DECAY:
            advantage = reward - baseline
            baseline = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * reward
        else:
            advantage = reward - baseline
            baseline = BASELINE_MOMENTUM * baseline + (1 - BASELINE_MOMENTUM) * reward
        
        if USE_BASELINE_BOUND:
            baseline = max(min(baseline, BASELINE_BOUND), -BASELINE_BOUND)
        
        policy_loss = -torch.stack(saved_log_probs).sum() * advantage

        if USE_ENTROPY:
            global ENTROPY_COEFF
            entropy_loss = -torch.stack(saved_entropies).sum()
            loss = policy_loss + entropy_loss * ENTROPY_COEFF
        else:
            loss = policy_loss
        
        nn.optimizer.zero_grad()
        loss.backward()

        if USE_GRADIENT_CLIPPING:
            grad_norm = torch.nn.utils.clip_grad_norm_(nn.parameters(), GRADIENT_CLIP).item()
        else:
            total_norm = 0.0
            for p in nn.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5

        nn.optimizer.step()
        

        if USE_LR_DECAY and e % (EPOCHS // steps_to_threshold(LEARNING_RATE, 1e-8, LR_DECAY_RATE)) == 0:
            nn.optimizer.param_groups[0]["lr"] *= LR_DECAY_RATE 

        if USE_ENTROPY_DECAY:
            ENTROPY_COEFF *= ENTROPY_DECAY
        

        lr = nn.optimizer.param_groups[0]["lr"]
        p_check, p_bet, p_call, p_fold = first_probs  # type: ignore
        
        logs.append(
            {
                "episode": e,
                "hand": env.hands[PLAYER_NUMBER],
                "opp_hand": env.hands[1 - PLAYER_NUMBER],
                "first_to_act": env.first_to_start(),
                "history": "-".join(env.history),
                "history_length": len(env.history),
                "reward": reward,
                "baseline": baseline,
                "grad_norm": grad_norm,
                "entropy_coeff": ENTROPY_COEFF,
                "learning_rate": lr,
                "p_check": p_check,
                "p_bet": p_bet,
                "p_call": p_call,
                "p_fold": p_fold,
            }
        )


    df = pd.DataFrame(logs)
    df.to_csv(os.path.join(RUN_DIR, "full_training_data.csv"), index=False)
    analysis_script = os.path.join(os.path.dirname(__file__), "torch_analysis.py")
    subprocess.run([sys.executable, analysis_script, RUN_DIR], check=True)



if __name__ == "__main__":
    train()

