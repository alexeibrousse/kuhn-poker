import argparse
import os
import sys

# Ensure the parent directory is in the path for module imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from subpoker.engine import KuhnPokerEnv
from subpoker.agents import (
    RuleBasedAgent,
    BluffAgent,
    RandomAgent,
    AlwaysLieAgent,
    NashAgent,
)

AGENT_MAP = {
    "rule": RuleBasedAgent,
    "bluff": BluffAgent,
    "random": RandomAgent,
    "lie": AlwaysLieAgent,
    "nash": NashAgent,
    "nashagent": NashAgent,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Play Kuhn poker bots against each other")
    parser.add_argument("--agent1", default="rule", help="Agent for player 0")
    parser.add_argument("--agent2", default="nash", help="Agent for player 1")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to play")
    parser.add_argument("--output", default="bot_vs_bot_results.csv", help="CSV file to save results")
    return parser.parse_args()

def make_agent(name: str):
    cls = AGENT_MAP.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown agent type: {name}")
    return cls()

def main():
    args = parse_args()
    agent1 = make_agent(args.agent1)
    agent2 = make_agent(args.agent2)
    agents = [agent1, agent2]

    env = KuhnPokerEnv()
    results = []

    for episode in range(args.episodes):
        state = env.reset()
        done = False
        while not done:
            player = env.current_player
            legal = env.legal_actions()
            action = agents[player].act(state, legal)
            state, rewards, done, _ = env.step(action)

        result = {
            "episode": episode,
            "winner": env.winner,
            "reward_p0": rewards[0], # type: ignore
            "reward_p1": rewards[1], # type: ignore
            "history": "-".join(env.history),
        }
        results.append(result)

    df = pd.DataFrame(results)

    win_counts = df["winner"].value_counts().to_dict()
    avg_reward_p0 = df["reward_p0"].mean()
    avg_reward_p1 = df["reward_p1"].mean()

    print(f"Player 0 ({args.agent1}) wins: {win_counts.get(0, 0)}")
    print(f"Player 1 ({args.agent2}) wins: {win_counts.get(1, 0)}")
    print(f"Average reward P0: {avg_reward_p0:.3f}")
    print(f"Average reward P1: {avg_reward_p1:.3f}")

if __name__ == "__main__":
    main()
