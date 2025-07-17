import os
import pandas as pd

def load_data(run_dir):
    df = pd.read_csv(os.path.join(run_dir, "full_training_data.csv"))
    return df
