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



class BluffAgent(Agent):
    """Agent that bluffs with a configurable probability."""

    def __init__(self, bluff_prob: float = 0.5):
        self.bluff_prob = bluff_prob

    def act(self, state: dict, legal_actions: list) -> str:
        hand = state["hand"]

        if "bet" in legal_actions:
            if hand == 3:
                return "bet"
            if random.random() < self.bluff_prob:
                return "bet"
            return "check" if "check" in legal_actions else legal_actions[0]

        if "call" in legal_actions:
            if hand == 3:
                return "call"
            if random.random() < self.bluff_prob:
                return "call"
            return "fold"

        if "check" in legal_actions:
            return "check"

        return legal_actions[0]



class AlwaysLieAgent(Agent):
    """Agent that inverts normal strength tells and always misrepresents its hand."""

    def act(self, state: dict, legal_actions: list) -> str:
        hand = state["hand"]

        if "bet" in legal_actions:
            if hand == 3:
                return "check" if "check" in legal_actions else legal_actions[0]
            return "bet"

        if "call" in legal_actions:
            if hand == 3:
                return "fold"
            return "call"

        if "check" in legal_actions:
            return "check"

        return legal_actions[0]


class NashAgent(Agent):
    """Kuhn poker Nash equilibrium strategy."""

    def act(self, state: dict, legal_actions: list) -> str:
        hand = state["hand"]
        history = state["history"]

        # Player 1 actions
        if history == []:
            if hand == 3:
                return "bet"
            if hand == 2:
                return "check"
            return "bet" if random.random() < 1/3 else "check"

        # Player 2 after check
        if history == ["check"]:
            if hand == 3:
                return "bet"
            if hand == 2:
                return "bet" if random.random() < 1/3 else "check"
            return "check"

        # Player 2 facing bet
        if history == ["bet"]:
            if hand == 3:
                return "call"
            if hand == 2:
                return "call" if random.random() < 1/3 else "fold"
            return "fold"

        # Player 1 facing bet after checking
        if history == ["check", "bet"]:
            if hand == 3:
                return "call"
            if hand == 2:
                return "call" if random.random() < 1/3 else "fold"
            return "fold"

        if "check" in legal_actions:
            return "check"

        return random.choice(legal_actions)
