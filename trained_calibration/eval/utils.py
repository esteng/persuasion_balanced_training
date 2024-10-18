from pathlib import Path
import json 
import pdb

def make_global_path(path):
    curr_path = Path(__file__).absolute()
    root_dir = curr_path.parent.parent.parent
    return str(root_dir / path)

def identify_best_checkpoint(dir):
    path = Path(dir)

    training_states = [x for x in path.glob("checkpoint-*/trainer_state.json")]
    last_state = training_states[-1]
    with open(last_state) as f1:
        data = json.load(f1)

    best_checkpoint = data['best_model_checkpoint']
    best_checkpoint = make_global_path(best_checkpoint)

    return best_checkpoint