import os
import pickle


def save_train_run(out, savedir, savename):
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)
        
    savepath = f"{savedir}/{savename}.pkl"
    with open(savepath, "wb") as f:
        pickle.dump(out, f)
    return savepath

def load_checkpoints(path):
    with open(path, "rb") as f:
        out = pickle.load(f)
        checkpoints = out["checkpoints"]
    return checkpoints
