import numpy as np

def get_random_matching_pairs(matches, n_pairs=1):
  pairs = []
  # Store as a copy, to protect the original object
  matches = list(matches)

  import random
  for i in range(0, n_pairs):
    group_of_matches = random.choice(matches)
    random.shuffle(group_of_matches)
    pair = [0,0,True]
    pair[0] = group_of_matches[0]
    pair[1] = group_of_matches[1]
    pairs.append(pair)

  return pairs

def get_random_non_matching_pairs(matches, all_ids=[], n_pairs=1):
  pairs = []
  # Store as a copy, to protect the original object
  matches = list(matches)

  # if ids are given, find the lonely ids
  # then add them as lone ids to the matches list
  if len(all_ids) > 0:
    all_matched_ids = [seg_id for group in matches for seg_id in group]
    all_non_matched_ids = [seg_id for seg_id in all_ids if seg_id not in all_matched_ids]
    matches += [[seg_id] for seg_id in all_non_matched_ids]

  import random
  for i in range(0, n_pairs):
    random.shuffle(matches)
    pair = [0,0,False]
    pair[0] = random.choice(matches[0])
    pair[1] = random.choice(matches[1])
    pairs.append(pair)

  return pairs

def generate_samples(ids, features, matches, n_samples=100, ratio_of_matches_to_non_matches=0.5):
  X = []
  true_Y = []

  # Get pairs of matching segment ids
  n_samples_which_are_matches = int(n_samples * ratio_of_matches_to_non_matches)
  matching_pairs = get_random_matching_pairs(matches, n_pairs=n_samples_which_are_matches)
  # Get pairs of non-matching segment ids
  n_samples_which_are_not_matches = n_samples - n_samples_which_are_matches
  non_matching_pairs = get_random_non_matching_pairs(matches, all_ids=ids,  n_pairs=n_samples_which_are_not_matches)

  # Mix together matching and non-matching pairs
  all_pairs = matching_pairs + non_matching_pairs
  import random
  random.shuffle(all_pairs)

  # Generate sample for each pair
  for pair in all_pairs:
    features1 = list(features[ids.index(pair[0])])
    features2 = list(features[ids.index(pair[1])])
    ids_are_a_match = float(pair[2])
    X.append(features1+features2);
    true_Y.append(ids_are_a_match);

  return X, true_Y
