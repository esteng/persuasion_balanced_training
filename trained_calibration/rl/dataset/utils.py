from Levenshtein import ratio as lev_ratio
import re 


def get_acc(answer, answers):
    # remove punctuation at the end 
    answer = re.sub(r'[^\d\w\s]$', '',answer)

    try:
        return answer.lower().strip() in answers
    except AttributeError:
        return False

def handle_agree(answers):
    try:
        if answers[0].lower().strip() == "agree":
            return [answers[1], answers[1]]

        if answers[1].lower().strip() == "agree":
            return [answers[0], answers[0]]
    except AttributeError:
        pass
    return answers

def handle_disagree(traj, node):
    # handle "disagree" outputs

    if "disagree" in [str(x).lower() for x in node.extracted_ans]:
        for i, ans in enumerate(node.extracted_ans): 

            if str(ans).lower() == "disagree": 
                parent_idx = traj.get_node_by_idx(node.node_idx).parent 
                if parent_idx == -1:
                    return node 
                parent = traj.get_node_by_idx(parent_idx) 
                parent = handle_disagree(traj, parent)
                grandparent_idx = parent.parent
                grandparent = traj.get_node_by_idx(grandparent_idx) 
                grandparent = handle_disagree(traj, grandparent)
                grandparent_answers = grandparent.extracted_ans
                # if it's just disagreement, set to be the last answer the agent gave 
                # print(f"changing {node.extracted_ans[i]} to {grandparent_answers[i]}")
                node.extracted_ans[i] = grandparent_answers[i]
    return node 

def check_acc(traj, node):
    if None in node.extracted_ans:
        # root 
        return None
    a1, a2 = [str(x).lower().strip() for x in node.extracted_ans]

    # if agreement, make them the same
    if a1 == "agree":
        a1 = a2
    if a2 == "agree":
        a2 = a1

    if a1 in a2 or a2 in a1:
        # if 1 answer fully included in the other 
        # see if they're similar, e.g. diego rivera vs rivera
        if lev_ratio(a1, a2) > 0.5:
            a1 = a2

    gold_ans = traj.metadata['answer']['normalized_aliases']
    # pdb.set_trace()
    return [a1 in gold_ans, a2 in gold_ans]





def draw_sumtree(traj, sums_by_idx):

    def printhelper(indent, node_idx):
        space = ''.join(["  " for i in range(indent)])
        print(f"{space} |--{node_idx}: {sums_by_idx[node_idx]}")
        node = traj.get_node_by_idx(node_idx)
        for child in node.children:
            printhelper(indent+1, child)
    printhelper(0, -1)
        