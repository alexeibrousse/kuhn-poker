import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from subpoker.engine import KuhnPokerEnv 
from subpoker.agents import BluffAgent, RuleBasedAgent, RandomAgent
from subpoker.neural_net import NeuralNet

env = KuhnPokerEnv()
state = env.reset()

n_epochs = 100000
nn = NeuralNet(input_size=19, hidden_size=50, output_size=3, learning_rate=1e-4)
agent = RandomAgent()


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


    probs  = nn.forward(X)
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



# Data Gathering

action_log = {
    1:{"check": 0, "call":0, "bet": 0, "fold": 0},
    2:{"check": 0, "call":0, "bet": 0, "fold": 0},
    3:{"check": 0, "call":0, "bet": 0, "fold": 0},
}

win_loss_log = {
    "wins": 0,
    "losses": 0,
    "reward_won": 0,
    "reward_lost": 0,
}



episode_rewards = [] # Rewards after one round
average_rewards = []
baseline= 0.0

# Beginning of the training

for e in range(n_epochs):
    state = env.reset()
    done = False
    trajectory = []
    reward = 0

    while not done:
        if state["player"] == 0:
            action, X, probs, action_index = nnbot(state)
            trajectory.append((X, action_index, probs))
            hand = state["hand"]
            action_log[hand][action] += 1
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

    baseline = 0.90 * baseline + 0.10 * reward # Update the baseline to reduce variance for backpropagation.

    if trajectory:
        dW1 = np.zeros_like(nn.W1) # Sum of all gradients in one episode.
        db1 = np.zeros_like(nn.b1)
        dW2 = np.zeros_like(nn.W2)
        db2 = np.zeros_like(nn.b2)

        advantage = reward - baseline
    
        for X, action_index, probs in trajectory:
            gW1, gb1, gW2, gb2 = nn.backward(X, action_index, advantage, probs, e, n_epochs) # Gradients for single step.
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

    

    if e % 500 == 0 and episode_rewards:
        avg = np.mean(episode_rewards[-500:])
        average_rewards.append(avg)


plt.plot(range(0, n_epochs, 500), average_rewards)
plt.xlabel("Epoch")
plt.ylabel("Average Reward (last 500)")
plt.title("Neural Network Learning Progress")
plt.grid(True)
plt.show()


print(np.mean(episode_rewards[-10000:]))

df = pd.DataFrame(action_log)
df.index.name = "Hand"
print(df)
print()


# Display win/loss stats

won_rewards = win_loss_log["reward_won"]
lost_rewards = win_loss_log["reward_lost"]
total_games = win_loss_log["wins"] + win_loss_log["losses"]
win_rate = win_loss_log["wins"] / total_games if total_games > 0 else 0


print(f"Lost rewards: {lost_rewards}")
print(f"Total games: {total_games}")
print(f"Win rate: {win_rate:.2%}")
print(f"Rewards won: {won_rewards - lost_rewards}")