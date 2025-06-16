import random

class Agent():
    def act(self, state: dict, legal_actions: list) -> str:
        """ Base class for all agents. """
        raise NotImplementedError
    

class RuleBasedAgent(Agent):
    def __init__(self):
        self.name = "RuleBasedAgent"

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
    def __init__(self):
        self.name = "RandomAgent"
    """
    This agent returns a random action.
    """
    def act(self, state: dict, legal_actions: list) -> str:
        return random.choice(legal_actions)



class BluffAgent(Agent):
    """Agent that bluffs with a configurable probability."""

    def __init__(self, bluff_prob: float = 0.5):
        self.bluff_prob = bluff_prob
        self.name = "BluffAgent"

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

    def __init__(self):
        self.name = "AlwaysLieAgent"
    
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
    """Kuhn poker Nash equilibrium strategy with a parameter ``alpha``.

    The parameter ``alpha`` must be in the interval [0, 1/3].  Actions are
    chosen according to the following pattern:

    Player 1 (first to act)
        Jack  -> bet with probability ``alpha``
        Queen -> always check.  If player 2 subsequently bets, call with
                  probability ``1/3 + alpha``
        King  -> bet with probability ``3 * alpha``

    Player 2 (facing a bet)
        Jack  -> fold (or check if player 1 checked)
        Queen -> call with probability ``1/3``
        King  -> always call

    When player 1 checks and player 2 is to act, the agent follows the common
    equilibrium continuation where player 2 bets with a King, bets with a Queen
    with probability ``1 - 3 * alpha`` and checks with a Jack.
    """

    def __init__(self, alpha: float = 1/3):
        if not 0 <= alpha <= 1/3:
            raise ValueError("alpha must be between 0 and 1/3")
        self.alpha = alpha
        self.name = f"NashAgent(alpha={alpha})"
    
    def act(self, state: dict, legal_actions: list) -> str:
        hand = state["hand"]
        history = state["history"]

        a = self.alpha

        # Player 1 actions (opening move)
        if history == []:
            if hand == 1:  # Jack
                return "bet" if random.random() < a else "check"
            if hand == 2:  # Queen
                return "check"
            if hand == 3:  # King
                return "bet" if random.random() < 3 * a else "check"

        # Player 2 after a check from player 1
        if history == ["check"]:
            if hand == 1:  # Jack
                return "check"
            if hand == 2:  # Queen
                return "bet" if random.random() < max(0, 1 - 3 * a) else "check"
            if hand == 3:  # King
                return "bet"

        # Player 2 facing a bet from player 1
        if history == ["bet"]:
            if hand == 1:  # Jack
                return "fold"
            if hand == 2:  # Queen
                return "call" if random.random() < 1/3 else "fold"
            if hand == 3:  # King
                return "call"

        # Player 1 responding to a bet after checking
        if history == ["check", "bet"]:
            if hand == 1:  # Jack
                return "fold"
            if hand == 2:  # Queen
                return "call" if random.random() < (1/3 + a) else "fold"
            if hand == 3:  # King
                return "call"

        # Default fall back if no specific rule applies
        if "check" in legal_actions:
            return "check"

        return random.choice(legal_actions)
