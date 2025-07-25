import sys
import os
from tqdm import trange
import numpy as np

from utils import create_run_dir, save_metadata

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from subpoker.engine import KuhnPokerEnv
from subpoker.agents import RuleBasedAgent
from subpoker.pytorch_nn import PyNet



# ————— Environment and reproducibility ————— #

random_seed = 1906220402
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
nn = PyNet(input_size=18, hidden_size=20, output_size=4, learning_rate=5e-5)
agent = RuleBasedAgent()