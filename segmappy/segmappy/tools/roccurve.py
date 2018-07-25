from __future__ import print_function
import numpy as np
import random
from sklearn import metrics
import pickle


def get_roc_pairs(
    segments,
    classes,
    duplicate_classes,
    USE_LAST_SAMPLE_ONLY=False,
    ALWAYS_AGAINST_LAST=False,
    MIN_DISTANCE_NEGATIVES=20.0,
):
    pair_ids = []
    pair_labels = []

    # calculate centroids
    centroids = [np.mean(segment, 0) for segment in segments]

    # positive samples
    for cls in np.unique(classes):
        class_ids = np.where(classes == cls)[0]

        sequences = duplicate_classes[class_ids]
        unique_sequences = np.unique(sequences)

        if unique_sequences.size > 1:
            for i, sequence_1 in enumerate(unique_sequences):
                for sequence_2 in unique_sequences[i + 1 :]:
                    ids_1 = class_ids[np.where(sequences == sequence_1)[0]]
                    ids_2 = class_ids[np.where(sequences == sequence_2)[0]]

                    if USE_LAST_SAMPLE_ONLY:
                        pair_ids.append(ids_1[-1], ids_2[-1])
                        pair_labels.append(1)
                    elif ALWAYS_AGAINST_LAST:
                        for id_1 in ids_1:
                            pair_ids.append([id_1, ids_2[-1]])
                            pair_labels.append(1)

                        for id_2 in ids_2:
                            pair_ids.append([ids_1[-1], id_2])
                            pair_labels.append(1)
                    else:
                        for id_1 in ids_1:
                            for id_2 in ids_2:
                                pair_ids.append([id_1, id_2])
                                pair_labels.append(1)

    n_positives = len(pair_ids)

    # negative samples
    random.seed(54321)
    ids = range(len(segments))

    last_ids = []
    for sequence in np.unique(duplicate_classes):
        last_ids.append(np.where(duplicate_classes == sequence)[0][-1])

    n_negatives = 0
    while n_negatives < n_positives:
        id_1 = random.sample(ids, 1)[0]
        id_2 = random.sample(ids, 1)[0]

        if ALWAYS_AGAINST_LAST:
            id_2 = random.sample(last_ids, 1)[0]

        dist = np.linalg.norm(centroids[id_1] - centroids[id_2])
        if dist >= MIN_DISTANCE_NEGATIVES:
            pair_ids.append([id_1, id_2])
            pair_labels.append(0)
            n_negatives += 1

    pair_ids = np.array(pair_ids)
    pair_labels = np.array(pair_labels)

    print("Positive pairs: %d" % n_positives)
    print("Negative pairs: %d" % n_negatives)

    return pair_ids, pair_labels


def get_roc_curve(features, pair_ids, pair_labels):
    y_pred = []
    for i in range(pair_ids.shape[0]):
        y_pred.append(
            1.0
            / (
                np.linalg.norm(features[pair_ids[i, 0]] - features[pair_ids[i, 1]])
                + 1e-12
            )
        )

    fpr, tpr, thresholds = metrics.roc_curve(pair_labels, y_pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    return fpr, tpr, roc_auc
