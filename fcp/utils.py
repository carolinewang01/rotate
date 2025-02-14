import os
from datetime import datetime
import pickle


def save_train_run(config, out):
    curr_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(config["RESULTS_PATH"], curr_datetime) 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    savepath = f"{save_dir}/train_run.pkl"
    with open(savepath, "wb") as f:
        pickle.dump(out, f)
    return savepath

def load_checkpoints(path):
    with open(path, "rb") as f:
        out = pickle.load(f)
        checkpoints = out["checkpoints"]
    return checkpoints
