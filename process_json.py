import json
from itertools import combinations
from collections import OrderedDict
import numpy as np

x = 1
def remove_duplicate_outputs(models_output, final_ranking, score_reasons):
    """
    Remove duplicate model outputs, keeping only the first occurrence.
    """
    global x
    unique_dict = OrderedDict()
    if len(final_ranking) > 5:
        x += 1
    for output, rank, reason in zip(models_output, final_ranking, score_reasons):
        if output not in unique_dict:
            unique_dict[output] = (rank, reason)
    unique_models_output = list(unique_dict.keys())
    unique_final_ranking = [v[0] for v in unique_dict.values()]
    unique_score_reasons = [v[1] for v in unique_dict.values()]
    return unique_models_output, unique_final_ranking, unique_score_reasons

def rank_score(rank_diff):
    """
    Compute a rank-based score.
    """
    if rank_diff == 1:
        return 0.3
    elif rank_diff == 2:
        return 0.6
    else:  # rank_diff >= 3
        return 1.0

def sigmoid(x, k=0.5, x0=0):
    # return 0.5 + 0.5 * 1 / (1 + np.exp(-k * (x - x0)))
    return 1 - np.exp(-k * x)

def reward_score(critic_rewards, chosen_idx, rejected_idx):
    """
    Compute a reward-based score between 0 and 1 based on the critic_rewards difference.
    """
    reward_diff = critic_rewards[chosen_idx] - critic_rewards[rejected_idx]
    return max(0.0, sigmoid(reward_diff))  # Normalize to 0-1 range

def rank_reward_score(rank_diff, reward_diff):
    """
    Combine rank and reward scores into a single ls_factor, with rank as the primary driver.
    """
    base_rank_score = rank_score(rank_diff)
    return base_rank_score * (0.5 + 0.5 * reward_diff)  # Weighted adjustment with reward_diff

def process_all_pairs(item, score_type):
    """
    Process an item to generate all ranked model output pairs, removing duplicate outputs.
    """
    final_ranking = item.get('final_ranking', [])
    models_output = item.get('outputs', [])
    score_reasons = item.get('score_reasons', [])
    critic_rewards = item.get('critic_rewards', [])
    id_ = item.get('id')
    image = item.get('image')
    video = item.get('video')
    if image:
        key = 'image'
        value = image
    else:
        key = 'video'
        value = video
    question = item.get('prompt')

    # Remove duplicate model outputs
    unique_models_output, unique_final_ranking, unique_score_reasons = remove_duplicate_outputs(
        models_output, final_ranking, score_reasons
    )

    pairs = []
    num_models = len(unique_final_ranking)

    # Generate all possible model pairs
    for i, j in combinations(range(num_models), 2):
        rank_i = unique_final_ranking[i]
        rank_j = unique_final_ranking[j]
        question = item.get('prompt')

        # Skip pairs with the same rank
        if rank_i == rank_j:
            continue

        # Determine chosen (higher rank) and rejected (lower rank)
        if rank_i < rank_j:
            chosen_idx, rejected_idx = i, j
        else:
            chosen_idx, rejected_idx = j, i

        # Compute scores based on the selected type
        rank_diff = abs(rank_i - rank_j)
        if score_type == "rank":
            ls_factor = rank_score(rank_diff)
        elif score_type == "reward":
            reward_diff = reward_score(critic_rewards, chosen_idx, rejected_idx)
            ls_factor = reward_diff
        elif score_type == "rank_reward":
            # if critic_rewards[chosen_idx] >= critic_rewards[rejected_idx]:
            reward_diff = reward_score(critic_rewards, chosen_idx, rejected_idx)
            ls_factor = rank_reward_score(rank_diff, reward_diff)
            # else:
            #     ls_factor = rank_score(rank_diff)
        else:
            raise ValueError(f"Unknown score type: {score_type}")

        # Create pair JSON object
        pair = {
            'id': id_,
            key: value,
            'prompt': question,
            'chosen': unique_models_output[chosen_idx],
            'rejected': unique_models_output[rejected_idx],
            'chosen_reason': unique_score_reasons[chosen_idx],
            'rejected_reason': unique_score_reasons[rejected_idx],
            'ls_factor': ls_factor,  # Add ls_factor to the pair
            'rewards_pos_neg': [critic_rewards[chosen_idx], critic_rewards[rejected_idx]],
            'rank_pos_neg': [unique_final_ranking[chosen_idx], unique_final_ranking[rejected_idx]]
        }
        pairs.append(pair)

    return pairs

# Example of reading the JSONL file and processing each item
def process_file(input_file, output_file):
    with open(input_file, 'r') as infile, \
         open(output_file, 'w') as outfile:

        data = json.load(infile)
        for item in data:
            item["cosine_similarity"] = item["cosine_similarity"][0]

        json.dump(data, outfile, ensure_ascii=False, indent=2)


# File paths
input_file = "/home/yilin/dataset-construct/output/yilin_pref_data.json"
output_file = "/home/yilin/dataset-construct/output/yilin_pref_data—1.json"

# Process the input file and save the outputs
process_file(input_file, output_file)