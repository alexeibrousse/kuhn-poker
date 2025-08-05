import os
import sys
import subprocess

import numpy as np
import pandas as pd
import torch
from tqdm import trange

from utils import create_run_dir, save_metadata

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from subpoker.engine import KuhnPokerEnv
from subpoker.agents import RuleBasedAgent
from subpoker.pytorch_nn import PyNet



# ————— Environment and reproducibility ————— #

random_seed = None
random_seed = 1906220402
if random_seed is not None:
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    env = KuhnPokerEnv(random_seed)
player_number = 0

"""
Seeds for reproducibility:
1. 525518843
2. 2342489760
3. 2097210685
4. 1906220402
"""



# ————— Hyperparameters ————— #

n_epochs = 1000000
nn = PyNet(input_size=18, hidden_size=20, output_size=4, learning_rate=5e-5)
agent = RuleBasedAgent()
initial_lr = nn.learning_rate
decay_rate = 0.99
baseline_momentum = 0.10
baseline_bound = 15
entropy_coeff = 0.01
entropy_schedule = 0.99
gradient_clip = 10.0



# ————— Metadata ————— #

metadata = {
    "implementation": "pytorch",
    "agent": agent.name,
    "input_size": nn.input_size,
    "hidden_size": nn.hidden_size,
    "output_size": nn.output_size,
    "initial_learning_rate": initial_lr,
    "decay_rate": decay_rate,
    "activation": "ReLU",
    "number_epochs": n_epochs,
    "baseline_momentum": baseline_momentum,
    "baseline_bound": baseline_bound,
    "entropy_coeff": entropy_coeff,
    "entropy_schedule": entropy_schedule,
    "random_seed": random_seed,
    "gradient_clip": gradient_clip,
}   





# ————— Training helper functions ————— #

def encode_state(state: dict) -> torch.Tensor:
    """
    Encoding the game state into a 18-dimensional vector.
    The first three dimensions represent the player's card as one-hot:
        [0, 0, 1] is King
        [0, 1, 0] is Queen
        [1, 0, 0] is Jack
    The remaining 15 dimensions encode the round history as a 3x5 matrix
    (three steps max, five actions including "none"). "none" is a placeholder for actions that have not occurred.
    """
    hand = state["hand"]
    history = state["history"]

    # One-hot encoding of the player's hand
    card_vec = [0, 0, 0]
    card_vec[hand - 1] = 1

    action_index = {"check": 0, "call": 1, "bet": 2, "fold": 3, "none": 4}
    
    history_mat = np.zeros((3, 5), dtype=int)

    for i in range(3):
        action = history[i] if i < len(history) else "none"
        history_mat[i, action_index[action]] = 1  
    
    vector = np.concatenate([card_vec, history_mat.ravel()]) #History of ongoing round
    return torch.tensor(vector, dtype=torch.float32)



def action_probs(state: dict) -> tuple:
    """ Calculates the action probabilities for the given state. """
    X = encode_state(state)
    probs = nn(X)

    legal_actions = env.legal_actions()
    action_to_index = {"check": 0, "call": 1, "bet": 2, "fold": 3}
    legal_indices = [action_to_index[i] for i in legal_actions]
    
    filtered_probs = probs[legal_indices]
    filtered_sum = torch.sum(filtered_probs).item()

    if filtered_sum > 0:
        filtered_probs /= filtered_sum
    else:
        filtered_probs = torch.ones_like(filtered_probs) / len(filtered_probs)

    action = np.random.choice(legal_actions, p=filtered_probs.detach().numpy())
    action_index = action_to_index[action]

    return action, X, probs.detach(), action_index



def update_baseline(baseline: float, reward: float) -> float:
    """
    Updates the baseline to reduce variance, bounded in [-bound, bound]. 
    """
    momentum, bound = baseline_momentum, baseline_bound

    if not (0 <= momentum < 1): # If momentum is 1, the baseline is the reward.
        raise ValueError("Momentum must be in ]0, 1[.")
    if bound <= 0:
        raise ValueError("Bound must be positive.")

    baseline = baseline * momentum + reward * (1 - momentum)

    return max(-bound, min(bound, baseline))  # Ensure baseline is within bounds



def update_advantage(baseline: float, reward: float) -> float:
    """
    Computes the advantage. Can't explain better.
    """
    return reward - baseline



def learning_rate_decay(episode: int) -> float:
    if initial_lr <= 0:
        raise ValueError("Initial learning rate must be positive")
    if not (0 < decay_rate <= 1):
        raise ValueError("Decay rate must be in ]0, 1]")
    return initial_lr * decay_rate * (1 - episode / n_epochs)



def entropy_coeff_schedule(episode: int) -> float:
    """
    Linearly decays the 'entropy_coefficient' to 0.0 over training.
    """
    return entropy_coeff * entropy_schedule * (1 - (episode / n_epochs))



def entropy_loss(probs: torch.Tensor) -> float:
    """
    Computes the entropy loss for the given probabilities.
    The addition of 1e-10 is to avoid log(0)
    """  
    return -torch.sum(probs * torch.log(probs + 1e-10)).item()




def clip_gradients(max_norm: float = gradient_clip) -> float:
    """
    Clips the gradients to prevent exploding gradients.
    """
    return float(torch.nn.utils.clip_grad_norm_(nn.parameters(), max_norm))



def step(state: dict) -> tuple:
    """
    Plays a round (episode) of the game, alternating between the agent and the neural network.
    """
    done = False
    trajectory: list[tuple[np.ndarray, int, np.ndarray]] = [] # Stores the trajectory of the episode
    reward: int = 0

    while not done:
        if state["player"] != player_number: # Agent's turn to play
            legal = env.legal_actions()
            action = agent.act(state, legal)
    
        else: # Neural network's turn to play
                action, X, probs, action_index = action_probs(state)
                trajectory.append((X, action_index, probs)) # Store the trajectory

        state, step_rewards, done, _ = env.step(action)
        reward = step_rewards[player_number]

    # Round has finished.

    return state, reward, done, trajectory


def update_nn(trajectory: list[tuple[torch.Tensor, int, torch.Tensor]], advantage: float) -> float:
    """
    This updates the neural network weights and biases based on the trajectory of this episode.
    Returns the norm of the weight gradients if gradient are clipped, otherwise None.
    """
    grad_norm = 0.0

    for X, action_index, probs in trajectory:
        entropy = entropy_loss(probs)
        step_advantage = advantage + entropy_coeff * entropy 
        
        nn.optimizer.zero_grad()
        out = nn.forward(X)
        log_prob = torch.log(out[action_index] + 1e-10)
        loss = -log_prob * step_advantage
        loss.backward()
        grad_norm = clip_gradients()
        nn.optimizer.step()

    return grad_norm





# ————— Utils and data logging ————— #

def data_log(store: list[dict], episode: int, reward: int, baseline: float,
             grad_norm: float) -> None:
    store.append({
        "episode": episode,
        "first_to_act": env.first_to_start(),
        "hand": env.hands[player_number],
        "opp_hand": env.hands[1 - player_number],
        "history": "-".join(env.history),
        "history_length": len(env.history),
        "reward": reward,
        "baseline": f"{baseline:,.3f}",
        "result": "win" if reward > 0 else "loss",
        "learning_rate": f"{nn.optimizer.param_groups[0]['lr']:.3e}",
        "gradient norm": f"{grad_norm:,.3f}",
    })





# ————— Main training loop ————— #

def main() -> None:
    """
    Main training loop for the neural network.
    """
    save_metadata(metadata, RUN_DIR)
    baseline = 0.0
    state = env.reset()
    episode_data: list[dict] = []

    for e in trange(1, n_epochs + 1, desc="Training"):
        state, reward, done, trajectory = step(state)
        advantage = update_advantage(baseline, reward)
        fixed_baseline = baseline
        baseline = update_baseline(baseline, reward)

        lr = learning_rate_decay(e)
        nn.optimizer.param_groups[0]["lr"] = lr

        global entropy_coeff
        entropy_coeff = entropy_coeff_schedule(e)

        grad_norm = update_nn(trajectory, advantage)
        if done:
            data_log(episode_data, e, reward, fixed_baseline, grad_norm)
            state = env.reset()
    
    df = pd.DataFrame(episode_data)
    df.to_csv(os.path.join(RUN_DIR, "full_training_data.csv"), index=False)





if __name__ == "__main__":
    RUN_DIR = create_run_dir()
    main()
    print("1/2 - Training completed.")
    analysis_script = os.path.join(os.path.dirname(__file__), "nn_analysis.py")
    subprocess.run([sys.executable, analysis_script, RUN_DIR], check=True)
    print("2/2 - Analysis completed.")

