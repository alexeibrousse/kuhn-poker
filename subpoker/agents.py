import random

class Agent():
    def act(self, state: dict, legal_actions: list) -> str:
        """ Base class for all agents. """
        raise NotImplementedError
    
class RuleBasedAgent(Agent):
    """
    This agent follows strict, most logic rules.
    King (3) -> Bet
    Jack (1) -> Check or Fold
    Queen (2) -> Check or Call
    """
    def act(self, state: dict, legal_actions: list) -> str:
        hand = state["hand"]
        
        if hand == 3 and "bet" in legal_actions:
            return "bet"
        
        if hand == 1 and "fold" in legal_actions:
            return "fold"
        return legal_actions[0]

class RandomAgent(Agent):
    """
    This agent returns a random action.
    """
    def act(self, state: dict, legal_actions: list) -> str:
        return random.choice(legal_actions)
