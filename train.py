import numpy as np
import matplotlib.pyplot as plt
from subpoker.engine import KuhnPokerEnv 
from subpoker.agents import RuleBasedAgent
from subpoker.neural_net import NeuralNet

env = KuhnPokerEnv()
state = env.reset()

n_epochs = 50000
nn = NeuralNet(input_size=4, hidden_size=200, output_size=3, learning_rate=0.0001)
rbb = RuleBasedAgent()


def encode_state(state):
    """
    Converts a player's hand into a one-hot vector.
    Jack -> [1, 0, 0]
    Queen -> [1, 1, 0]
    King -> [1, 0, 1]
    Position is either 0 or 1

    """
    hand = state["hand"]
    player = state["player"]

    one_hot_card = [0, 0, 0]
    one_hot_card[hand -1] = 1

    return np.array(one_hot_card + [player])


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


rewards = []
average_rewards = []
big_ave = []

for e in range(n_epochs):
    state = env.reset()
    done = False
    training_data = None
    reward = 0

    while not done:
        if state["player"] == 0:
            action, X, probs, action_index = nnbot(state)
            training_data = (X, action_index, probs)
        else:
            legal = env.legal_actions()
            action = rbb.act(state, legal)

        state, rewards, done, _ = env.step(action)
        reward = rewards[0]

    if training_data is not None:
        X, action_index, probs = training_data
        grads = nn.backward(X, action_index, reward, probs)
        dW1, db1, dW2, db2 = grads
        nn.update(dW1, db1, dW2, db2)
        rewards.append(reward)

    if e % 1000 == 0 and rewards:
        avg = np.mean(rewards[-1000:])
        average_rewards.append(avg)
    
    if e % 5000 == 0 and rewards:
        aver = np.mean(rewards[-5000:])
        big_ave.append(aver)


plt.plot(range(0, n_epochs, 1000), average_rewards)
plt.plot(range(0, n_epochs, 5000), big_ave)
plt.xlabel("Epoch")
plt.ylabel("Average Reward (last 10)")
plt.title("Neural Network Learning Progress")
plt.grid(True)
plt.show()