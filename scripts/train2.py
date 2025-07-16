import random
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import sys


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from subpoker.engine import KuhnPokerEnv
from subpoker.agents import RuleBasedAgent
from subpoker.numpy_nn import NeuralNet


# Environment and reproductibility
random_seed = random.randint(0, 2**32 - 1)
np.random.seed(random_seed)
env = KuhnPokerEnv(random_seed)
player_number = 0


# Hyperparameters
n_epochs = 50000
nn = NeuralNet(input_size=18, hidden_size=70, output_size=4, learning_rate=1e-5)
agent = RuleBasedAgent()
initial_lr = nn.lr
decay_rate = 0.99
baseline_momentum = 0.30
baseline_bound = 2
entropy_coeff = 0.01


# Metadata for logging
metadata = {
    "implementation": "numpy",
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
    "random_seed": random_seed,
}   




# Training helper functions
def encode_state(state: dict) -> np.ndarray:
    """
    Enconding the state of the game into a 18-dimension vector/
    3 first dimension are the encoded card:
        [0, 0, 1] is King
        [0, 1, 0] is Queen
        [1, 0, 0] is Jack
    Last 15 dimensions are the history of the ongoing round, represented as a 3x5 matrix (3 steps max, 5 actions including "none")
    "none" is a placeholder which can be filled  with the action taken at that step.
    """
    hand = state["hand"]
    history = state["history"]

    # One-hot enconding of the player's hand
    card_vec = [0, 0, 0]
    card_vec[hand - 1] = 1

    action_index = {"check": 0, "call": 1, "bet": 2, "fold": 3, "none": 4}
    
    history_mat = np.zeros((3,5), dtype= int)

    for i in range(3):
        action = history[i] if i < len(history) else "none"
        history_mat[i, action_index[action]] = 1     

    return np.concatenate([card_vec, history_mat.ravel()]) #History of ongoing round



def legal_mask() -> tuple[list[int], list[str]]:
    """
    Returns a list of the legal actions available as indices.
    """
    legal = env.legal_actions()

    action_map = ["check", "call", "bet", "fold"]
    legal_indices = []

    if "check" in legal:
        legal_indices.append(0)
    if "call" in legal:
        legal_indices.append(1)
    if "bet" in legal:
        legal_indices.append(2)
    if "fold" in legal:
        legal_indices.append(3)
    
    legal_actions = [action_map[i] for i in legal_indices] # Strings of the legal indices

    return legal_indices, legal_actions



def action_probs(state: dict) -> tuple[str, np.ndarray, np.ndarray, int]:
    """
    Calculates the action probabilities for the given state.
    """
    X = encode_state(state)
    legal_indices, legal_actions = legal_mask()

    probs = nn.forward(X)
    filtered_probs = probs[legal_indices]

    filtered_sum = np.sum(filtered_probs)

    if filtered_sum > 0:
        filtered_probs /= filtered_sum
    else:
        # Uniform probability over legal actions if all logits are zero
        filtered_probs = np.ones_like(filtered_probs) / len(filtered_probs)
        
    action = np.random.choice(legal_actions, p=filtered_probs)
    action_index = legal_indices[legal_actions.index(action)]

    return action, X, probs, action_index



def update_baseline(baseline: float, reward: float) -> float:
    """
    Update the baseline to reduce variance, bounded in [-bound, bound]. 
    """
    momentum, bound = baseline_momentum, baseline_bound

    if momentum <= 0 or momentum >= 1: # If momentum is 0, there is no baseline. If momentum is 1, the baseline is the reward.
        raise ValueError("Momentum must be in ]0, 1[.")
    if bound <= 0:
        raise ValueError("Bound must be positive.")

    baseline = baseline * momentum + reward * (1 - momentum)

    return max(-bound, min(bound, baseline))  # Ensure baseline is within bounds



def update_advantage( baseline: float, reward: float) -> float:
    """
    Computes the advantage.
    """
    return reward - baseline



def learning_rate_decay(episode: int) -> float:
    """
    Exponentially decays the learning rate based on the episode number.
    decay_rate is in ]0, 1].
    """
    if nn.lr <= 0: # Initial learning rate
        raise ValueError("Initial learning rate must be positive.")
    if not (0 < decay_rate <= 1): # If decay_rate is 0, no learning occurs. If decay_rate is 1, the learning rate is constant.
        raise ValueError("Decay rate must be in ]0, 1].") 
    
    return initial_lr * (decay_rate ** (episode / n_epochs))



def entropy_loss(probs: np.ndarray) -> float:
    """
    Computes the entropy loss for the given probabilities.
    This encourages exploration by penalizing certainty.
    The addition of 1e-10 is to avoid log(0)
    """
    return -np.sum(probs * np.log(probs + 1e-10))



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



def update_nn(trajectory: list[tuple[np.ndarray, int, np.ndarray]], advantage: float) -> None:
    """
    This updates the neural network weights and biases based on the trajectory of this episode.
    """
    dW1 = np.zeros_like(nn.W1)
    db1 = np.zeros_like(nn.b1)
    dW2 = np.zeros_like(nn.W2)
    db2 = np.zeros_like(nn.b2)

    for X, action_index, probs in trajectory:
        entropy = entropy_loss(probs)
        step_advantage = advantage + entropy_coeff * entropy 

        # Gradients for single step
        gW1, gb1, gW2, gb2 = nn.backward(X, action_index, step_advantage, probs)
        dW1 += gW1
        db1 += gb1
        dW2 += gW2
        db2 += gb2
    
    dW1 /= len(trajectory) # Average gradient per episode
    db1 /= len(trajectory)
    dW2 /= len(trajectory)
    db2 /= len(trajectory)
    nn.update(dW1, db1, dW2, db2)





# Utils and data logging

def create_run_dir() -> str:
    """
    Creates a timestamped run directory inside 'data/numpy-nn'.
    Returns the path to the created directory.
    """
    base_dir = os.path.join(os.getcwd(), "data", "numpy-nn")
    os.makedirs(base_dir, exist_ok=True)
    run_name = datetime.now().strftime("%d-%m-%y_%H-%M")
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_metadata() -> None:
    """
    Saves the metadata dictionary to a config.json file inside the run directory.
    """
    with open(os.path.join(RUN_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def data_log(episode_data: list[dict], episode: int, reward: int) -> None:
    """
    Stores the data of the current episode into a list.
    """
    episode_data.append({
        "episode": episode,
        "hand": env.hands[player_number],
        "opp_hand": env.hands[1 - player_number],
        "history": "-".join(env.history),
        "reward": reward
    })



def main() -> None:
    """
    Main training loop for the neural network.
    """
    baseline = 0.0 # Initial baseline
    state = env.reset() # Initial state of the game
    episode_data: list[dict] = [] # Stores data for each episode, to be analyzed by data_analysis.py

    for e in range(n_epochs):
        state, reward, done, trajectory = step(state)
        advantage = update_advantage(baseline, reward)
        baseline = update_baseline(baseline, reward)
        nn.lr = learning_rate_decay(e)
        update_nn(trajectory, advantage)
        if done:
            data_log(episode_data, e, reward)
            state = env.reset()
    
    df = pd.DataFrame(episode_data)
    df.to_csv(os.path.join(RUN_DIR, "full_training_data.csv"), index=False)


if __name__ == "__main__":
    RUN_DIR = create_run_dir()
    save_metadata()
    main()
