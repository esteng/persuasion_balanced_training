import json 
from pathlib import Path 
import argparse 
import pdb 
from collections import defaultdict
import re 

from trained_calibration.rl.dataset.trajectory import Trajectory
from trained_calibration.rl.dataset.utils import handle_disagree
from trained_calibration.rl.dataset.preference_factory import PreferenceFactory


def load_dir(dir, remove_agree=True, limit=None): 
    all_trajs = []
    i = 0
    for file in Path(dir).glob("*.jsonl"):
        if limit is not None and i > limit:
            break
        i += 1

        with open(file) as f1:
            for i, line in enumerate(f1.readlines()):
                try:
                    data = json.loads(line)
                    traj = Trajectory.from_json(data)

                    # resolve all disagreements 
                    traj.tree = [handle_disagree(traj, node) for node in traj.tree]

                    # skip -1 and 1 for agreement 
                    all_agree = []
                    if remove_agree:
                        for i in range(2, len(traj.tree)):
                            agree = traj.get_agreement(i)
                            all_agree.append(agree)
                        if all(all_agree):
                            continue
                    all_trajs.append(traj)
                except json.JSONDecodeError:
                    print(f"FAILURE ON LINE {i} of {file}")
                    continue
                    # pdb.set_trace()
                    # sys.exit()
    return all_trajs

# def get_data_by_question(all_trajs):
#     preference_data_examples = []
#     traj_by_question = defaultdict(list)
#     for traj in trajs:
#         traj_by_question[traj.prompt].append(traj)
#     for group, trajs in traj_by_question.items():
#         correct_incorrect_data = get_disagree(trajs)
#         if len(correct_incorrect_data['to_reward']) > 0 and len(correct_incorrect_data['to_punish']) > 0:
#             # prefer everything in reward over everything in punish 
#             # get {"prompt": prompt, "preferred": preferred, "dispreferred": dispreferred}
#             pass


def fill_children(traj):
    for node in traj.tree:
        parent_idx = node.parent
        if parent_idx != -1:
            parent_node = traj.get_node_by_idx(node.parent)
            parent_list_idx = [i for i, x in enumerate(traj.tree) if x.node_idx == parent_idx][0]
            if node.node_idx not in parent_node.children:
                parent_node.children.append(node.node_idx)

            traj.tree[parent_list_idx] = parent_node
    return traj


def get_preferences(traj, 
                    resist_only = False,
                    agree_only = False,
                    for_sft = False,
                    for_eval = False,
                    model_filtering = False,
                    filtering_tokenizer = None, 
                    filtering_model = None):
    factory = PreferenceFactory(traj, 
                                filtering_tokenizer=filtering_tokenizer,
                                filtering_model=filtering_model)
    preferences = factory.get_preferences(resist_only, agree_only, for_sft, for_eval, model_filtering)
    return preferences



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dialogue_data_tree")

    args = parser.parse_args()

    trajs = load_dir(args.data_dir)

    data = []
    for traj in trajs:
        data += get_preferences(traj)
        # data += get_disagree(traj)
    pdb.set_trace()