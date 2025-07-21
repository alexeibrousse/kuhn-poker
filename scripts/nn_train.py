import random
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import sys
import subprocess
from tqdm import trange

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from subpoker.engine import KuhnPokerEnv
from subpoker.agents import RuleBasedAgent
from subpoker.numpy_nn import NeuralNet


# ————— Environment and reproducibility ————— #

random_seed = random.randint(0, 2**32 - 1)
# random_seed = 1906220402
np.random.seed(random_seed)
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
nn = NeuralNet(input_size=18, hidden_size=20, output_size=4, learning_rate=5e-5)
agent = RuleBasedAgent()
initial_lr = nn.lr
decay_rate = 0.99
baseline_momentum = 0.10
baseline_bound = 15
entropy_coeff = 0.01
entropy_schedule = 0.99
gradient_clip = 10.0



# ————— Metadata ————— #

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
    "entropy_schedule": entropy_schedule,
    "random_seed": random_seed,
    "gradient_clip": gradient_clip,
}   





# ————— Training helper functions ————— #

def encode_state(state: dict) -> np.ndarray:
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

    return np.concatenate([card_vec, history_mat.ravel()]) #History of ongoing round



def action_probs(state: dict) -> tuple[str, np.ndarray, np.ndarray, int]:
    """ Calculates the action probabilities for the given state. """
    X = encode_state(state)
    probs = nn.forward(X)
    
    legal_actions = env.legal_actions()
    action_to_index = {"check": 0, "call": 1, "bet": 2, "fold": 3}
    legal_indices = [action_to_index[i] for i in legal_actions]

    filtered_probs = probs[legal_indices]
    filtered_sum = np.sum(filtered_probs)

    if filtered_sum > 0:
        filtered_probs /= filtered_sum
    else:
        # Uniform probability over legal actions if all logits are zero
        filtered_probs = np.ones_like(filtered_probs) / len(filtered_probs)
        
    action = np.random.choice(legal_actions, p=filtered_probs)
    action_index = action_to_index[action]

    return action, X, probs, action_index



def update_baseline(baseline: float, reward: float) -> float:
    """
    Updates the baseline to reduce variance, bounded in [-bound, bound]. 
    """
    momentum, bound = baseline_momentum, baseline_bound

    if momentum <= 0 or momentum >= 1: # If momentum is 0, there is no baseline. If momentum is 1, the baseline is the reward.
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
    """
    Linearly decays the learning rate based on the episode number.
    decay_rate is in ]0, 1].
    If decay_rate is 0, no learning occurs. If decay_rate is 1, the learning rate is constant.
    """
    if nn.lr <= 0: # Initial learning rate
        raise ValueError("Initial learning rate must be positive.")
    if not (0 < decay_rate <= 1):
        raise ValueError("Decay rate must be in ]0, 1].") 
    
    return initial_lr * decay_rate * (1 - episode / n_epochs)



def entropy_coeff_schedule(episode: int) -> float:
    """
    Linearly decays the 'entropy_coefficient' to 0.0 over training.
    """
    
    return entropy_coeff * entropy_schedule * (1 - (episode / n_epochs))



def entropy_loss(probs: np.ndarray) -> float:
    """
    Computes the entropy loss for the given probabilities.
    The addition of 1e-10 is to avoid log(0)
    """  

    return -np.sum(probs * np.log(probs + 1e-10))



def clip_gradients(dW1: np.ndarray, db1: np.ndarray, dW2: np.ndarray, db2: np.ndarray, max_norm: float = gradient_clip) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Scales gradients so their global norm does not exceed 'max_norm'.
    """
    total_norm = np.sqrt(np.sum(dW1**2) + np.sum(db1**2) + np.sum(dW2**2) + np.sum(db2**2))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        dW1 *= scale
        db1 *= scale
        dW2 *= scale
        db2 *= scale
    return dW1, db1, dW2, db2



def step(state: dict, collect_probs: bool = False) -> tuple:
    """
    Plays a round (episode) of the game, alternating between the agent and the neural network.
    """
    done = False
    trajectory: list[tuple[np.ndarray, int, np.ndarray]] = [] # Stores the trajectory of the episode
    reward: int = 0
    all_probs = []

    while not done:
        if state["player"] != player_number: # Agent's turn to play
            legal = env.legal_actions()
            action = agent.act(state, legal)
        
        else: # Neural network's turn to play
            action, X, probs, action_index = action_probs(state)
            trajectory.append((X, action_index, probs)) # Store the trajectory
            if collect_probs:
                all_probs.append(np.round(probs,3).tolist())
        
        state, step_rewards, done, _ = env.step(action)
        reward = step_rewards[player_number]

    # Round has finished.

    if collect_probs:
        return state, reward, done, trajectory, all_probs
    else:
        return state, reward, done, trajectory



def update_nn(trajectory: list[tuple[np.ndarray, int, np.ndarray]], advantage: float) -> float:
    """
    This updates the neural network weights and biases based on the trajectory of this episode.
    Returns the norm of the weight gradients.
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

    dW1, db1, dW2, db2 = clip_gradients(dW1, db1, dW2, db2) 

    grad_norm = gradient_norm(dW1, dW2)
    nn.update(dW1, db1, dW2, db2)

    return grad_norm





# ————— Utils and data logging ————— #

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



def gradient_norm(dW1: np.ndarray, dW2: np.ndarray) -> float:
    """
    Computes the norm of the weight gradients.
    """
    return np.sqrt(np.sum(dW1**2) + np.sum(dW2**2))



def data_log(episode_data: list[dict], episode: int, reward: int, baseline: float, grad_norm: float, all_probs: list[list]) -> None:
    """
    Stores the data of the current episode into a list.
    """
    episode_data.append({
        "episode": episode,
        "hand": env.hands[player_number],
        "opp_hand": env.hands[1 - player_number],
        "first_to_act": env.first_to_start(),
        "history": "-".join(env.history),
        "history_length": len(env.history),
        "reward": reward,
        "baseline": f"{baseline:,.3f}",
        "result": "win" if reward > 0 else "loss",
        "learning_rate": f"{nn.lr:,.3e}",
        "gradient norm": f"{grad_norm:,.3f}",
        "all_probs": all_probs,
    })





# ————— Main training loop ————— #

def main() -> None:
    """
    Main training loop for the neural network.
    """
    save_metadata()
    baseline = 0.0 # Initial baseline
    state = env.reset() # Initial state of the game
    episode_data: list[dict] = [] # Stores data for each episode, to be analyzed by data_analysis.py

    for e in trange(1, n_epochs + 1, desc="Training"):
        all_probs = []  # Collect all probs for this episode

        state, reward, done, trajectory, all_probs = step(state, collect_probs=True)
        advantage = update_advantage(baseline, reward)
        fixed_baseline = baseline
        baseline = update_baseline(baseline, reward)
        nn.lr = learning_rate_decay(e)
        global entropy_coeff
        entropy_coeff = entropy_coeff_schedule(e)
        grad_norm = update_nn(trajectory, advantage)
        if done:
            data_log(episode_data, e, reward, fixed_baseline, grad_norm, all_probs)
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
