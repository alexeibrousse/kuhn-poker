import os
import json
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Base directory used by training and analysis scripts
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "numpy-nn")


def create_run_dir(base_dir: Optional[str] = None) -> str:
    """Create and return a timestamped run directory."""
    base = base_dir or BASE_DIR
    os.makedirs(base, exist_ok=True)
    run_name = datetime.now().strftime("%d-%m-%y_%H-%M")
    run_dir = os.path.join(base, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_metadata(metadata: dict, run_dir: str) -> None:
    """Save run metadata as JSON in *run_dir*."""
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def load_run_dir(path: Optional[str] = None, base_dir: Optional[str] = None) -> str:
    """Return path to a run directory. Uses latest run when *path* is None."""
    base = base_dir or BASE_DIR
    if path:
        return path
    if not os.path.exists(base):
        raise FileNotFoundError(f"{base} does not exist")
    runs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    if not runs:
        raise FileNotFoundError(f"No runs found in {base}")
    return os.path.join(base, runs[-1])


def gradient_norm(*grads: np.ndarray) -> float:
    """Return the global L2 norm of the provided gradients."""
    return float(np.sqrt(sum(np.sum(g ** 2) for g in grads)))


def parse_episode(row: pd.Series) -> Tuple[bool, bool, bool, bool, bool, int, int, int]:
    """Extract common metrics from a logged episode row."""
    history = row.get("history", "")
    actions = history.split("-") if isinstance(history, str) and history else []
    first = int(row.get("first_to_act", 0))
    hand = int(row.get("hand", 0))

    bluff = False
    value_bet = False
    call = False
    fold = False
    responded = False

    if actions:
        last_idx = len(actions) - 1
        actor_last = (first + last_idx) % 2
        last_action = actions[-1]
        if actor_last == 0 and last_action in ("call", "fold"):
            responded = True
            if last_action == "call":
                call = True
            elif last_action == "fold":
                fold = True

        if len(actions) >= 2:
            bet_idx = len(actions) - 2
            actor_bet = (first + bet_idx) % 2
            if actor_bet == 0 and actions[bet_idx] == "bet":
                if hand in (1, 2):
                    bluff = True
                elif hand == 3:
                    value_bet = True

    return bluff, value_bet, call, fold, responded, hand == 1, hand == 2, hand == 3
