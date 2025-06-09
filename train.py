import numpy as np
import matplotlib.pyplot as plt
from subpoker.engine import KuhnPokerEnv 
from subpoker.agents import RuleBasedAgent
from subpoker.neural_net import NeuralNet

env = KuhnPokerEnv()
state = env.reset()

n_epochs = 50000
# input_size: 3 card bits + 1 player bit + 3*5 history encoding
nn = NeuralNet(input_size=19, hidden_size=200, output_size=3, learning_rate=0.0001)
rbb = RuleBasedAgent()


def encode_state(state):
    """Encode the environment state into a neural network input vector."""

    hand = state["hand"]
    player = state["player"]
    history = state["history"]

    # one-hot encode the private card
    card_vec = [0, 0, 0]
    card_vec[hand - 1] = 1

    # Encode up to the first three actions in history. Each position is one-hot
    # over [check, bet, call, fold, none]
    actions = ["check", "bet", "call", "fold", "none"]
    hist_vec = []
    for i in range(3):
        if i < len(history):
            act = history[i]
        else:
            act = "none"
        hist_vec.extend([1 if act == a else 0 for a in actions])

    return np.array(card_vec + [player] + hist_vec)


def nnbot(state): # Playing the round for the neural network
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


    probs, z1, a1, z2 = nn.forward(X)

    filtered_probs = probs[chosen_indices] # Removing the probabilities of illegal actions
    total = np.sum(filtered_probs)
    if total == 0:
        # Uniform probability over legal actions if all logits are ~zero
        filtered_probs = np.ones_like(filtered_probs) / len(filtered_probs)
    else:
        filtered_probs /= total # Normalizing these probabilites

    action = np.random.choice(chosen_actions, p=filtered_probs) # Choosing one action
    action_index = action_index = chosen_indices[chosen_actions.index(action)]

    # Convert unified label back into an actual engine-valid action
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


episode_rewards = []
average_rewards = []
big_ave = []
baseline = 0.0

for e in range(n_epochs):
    state = env.reset()
    done = False
    trajectory = []
    reward = 0

    while not done:
        if state["player"] == 0:
            action, X, probs, action_index = nnbot(state)
            trajectory.append((X, action_index, probs))
        else:
            legal = env.legal_actions()
            action = rbb.act(state, legal)

        state, step_rewards, done, _ = env.step(action)
        reward = step_rewards[0]

    baseline = 0.99 * baseline + 0.01 * reward

    if trajectory:
        dW1 = np.zeros_like(nn.W1)
        db1 = np.zeros_like(nn.b1)
        dW2 = np.zeros_like(nn.W2)
        db2 = np.zeros_like(nn.b2)
        advantage = reward - baseline
        for X, action_index, probs in trajectory:
            gW1, gb1, gW2, gb2 = nn.backward(X, action_index, advantage, probs)
            dW1 += gW1
            db1 += gb1
            dW2 += gW2
            db2 += gb2

        dW1 /= len(trajectory)
        db1 /= len(trajectory)
        dW2 /= len(trajectory)
        db2 /= len(trajectory)
        nn.update(dW1, db1, dW2, db2)

    episode_rewards.append(reward)

    if e % 1000 == 0 and episode_rewards:
        avg = np.mean(episode_rewards[-1000:])
        average_rewards.append(avg)

    if e % 5000 == 0 and episode_rewards:
        aver = np.mean(episode_rewards[-5000:])
        big_ave.append(aver)


plt.plot(range(0, n_epochs, 1000), average_rewards)
plt.plot(range(0, n_epochs, 5000), big_ave)
plt.xlabel("Epoch")
plt.ylabel("Average Reward (last 10)")
plt.title("Neural Network Learning Progress")
plt.grid(True)
plt.show()
