import random

class Agent():
    def act(self, state: dict, legal_actions: list) -> str:
        """ Base class for all agents. """
        raise NotImplementedError
    

class RuleBasedAgent(Agent):
    def act(self, state: dict, legal_actions: list) -> str:
        """
        Logical rule-based agent:
        - King (3): bet if possible; otherwise call
        - Queen (2): check or call; never bet
        - Jack (1): check; fold if faced with a bet
        """
        hand = state["hand"]

        # Opening moves
        if "bet" in legal_actions:
            if hand == 3:  # King bets
                return "bet"
            else:  # Queen and Jack never bet
                return "check" if "check" in legal_actions else legal_actions[0]

        # Responding to a bet
        if "call" in legal_actions:
            if hand == 2 or hand == 3:  # King and Queen call
                return "call"
            else:  # Jack folds
                return "fold"

        # Default to check
        if "check" in legal_actions:
            return "check"

        return legal_actions[0]  # Fallback
    


class RandomAgent(Agent):
    """
    This agent returns a random action.
    """
    def act(self, state: dict, legal_actions: list) -> str:
        return random.choice(legal_actions)
