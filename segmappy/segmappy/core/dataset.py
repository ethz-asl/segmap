from __future__ import print_function
import numpy as np
import os

from .config import get_default_dataset_dir

class Dataset(object):
    # load config values
    def __init__(
        self,
        folder="dataset",
        base_dir=get_default_dataset_dir(),
        require_change=0.0,
        use_merges=True,
        keep_match_thresh=0.0,
        use_matches=True,
        min_class_size=1,
        require_relevance=0.0,
        require_diff_points=0,
        normalize_classes=True,
    ):
        abs_folder = os.path.abspath(os.path.join(base_dir, folder))
        try:
            assert os.path.isdir(abs_folder)
        except AssertionError:
            raise IOError("Dataset folder {} not found.".format(abs_folder))

        self.folder = abs_folder
        self.require_change = require_change
        self.use_merges = use_merges
        self.keep_match_thresh = keep_match_thresh
        self.use_matches = use_matches
        self.min_class_size = min_class_size
        self.require_relevance = require_relevance
        self.require_diff_points = require_diff_points
        self.normalize_classes = normalize_classes

    # load the segment dataset
    def load(self, preprocessor=None):
        from ..tools.import_export import load_segments, load_positions, load_features

        # load all the csv files
        self.segments, sids, duplicate_sids = load_segments(folder=self.folder)
        self.positions, pids, duplicate_pids = load_positions(folder=self.folder)
        self.features, self.feature_names, fids, duplicate_fids = load_features(
            folder=self.folder
        )

        self.classes = np.array(sids)
        self.duplicate_classes = self.classes.copy()
        self.positions = np.array(self.positions)
        self.features = np.array(self.features)
        self.duplicate_ids = np.array(duplicate_sids)

        # load labels
        from ..tools.import_export import load_labels

        self.labels, self.lids = load_labels(folder=self.folder)
        self.labels = np.array(self.labels)
        self.labels_dict = dict(zip(self.lids, self.labels))

        # load matches
        from ..tools.import_export import load_matches

        self.matches = load_matches(folder=self.folder)

        if self.require_change > 0.0:
            self._remove_unchanged()

        # combine sequences that are part of a merger
        if self.use_merges:
            from ..tools.import_export import load_merges

            merges, _ = load_merges(folder=self.folder)
            self._combine_sequences(merges)
            self.duplicate_classes = self.classes.copy()

        # remove small irrelevant segments
        if self.require_relevance > 0:
            self._remove_irrelevant()

        # only use segments that are different enough
        if self.require_diff_points > 0:
            assert preprocessor is not None
            self._remove_similar(preprocessor)

        # combine classes based on matches
        if self.use_matches:
            self._combine_classes()

        # normalize ids and remove small classes
        self._normalize_classes()

        print(
            "  Found",
            self.n_classes,
            "valid classes with",
            len(self.segments),
            "segments",
        )

        self._sort_ids()

        return (
            self.segments,
            self.positions,
            self.classes,
            self.n_classes,
            self.features,
            self.matches,
            self.labels_dict,
        )

    def _remove_unchanged(self):
        keep = np.ones(self.classes.size).astype(np.bool)
        for cls in np.unique(self.classes):
            class_ids = np.where(self.classes == cls)[0]

            prev_size = self.segments[class_ids[0]].shape[0]
            for class_id in class_ids[1:]:
                size = self.segments[class_id].shape[0]
                if size < prev_size * (1.0 + self.require_change):
                    keep[class_id] = False
                else:
                    prev_size = size

        self._trim_data(keep)

        print("  Found %d segments that changed enough" % len(self.segments))

    # list of sequence pairs to merge and correct from the matches table
    def _combine_sequences(self, merges):
        # calculate the size of each sequence based on the last element
        last_sizes = {}
        subclasses = {}
        for cls in np.unique(self.classes):
            class_ids = np.where(self.classes == cls)[0]
            last_id = class_ids[np.argmax(self.duplicate_ids[class_ids])]
            last_sizes[cls] = len(self.segments[last_id])
            subclasses[cls] = []

        # make merges and keep a list of the merged sequences for each class
        for merge in merges:
            merge_sequence, target_sequence = merge

            merge_ids = np.where(self.classes == merge_sequence)[0]
            target_ids = np.where(self.classes == target_sequence)[0]

            self.classes[merge_ids] = target_sequence
            self.duplicate_ids[target_ids] += merge_ids.size

            subclasses[target_sequence].append(merge_sequence)
            subclasses[target_sequence] += subclasses[merge_sequence]
            del subclasses[merge_sequence]

        # calculate how relevant the merges are based on size
        relevant = {}
        new_class = {}
        for main_class in subclasses:
            relevant[main_class] = True
            new_class[main_class] = main_class

            main_size = last_sizes[main_class]
            for sub_class in subclasses[main_class]:
                new_class[sub_class] = main_class
                sub_size = last_sizes[sub_class]
                if float(sub_size) / main_size < self.keep_match_thresh:
                    relevant[sub_class] = False
                else:
                    relevant[sub_class] = True

        # ignore non-relevant merges and for the relevant merges replace
        # the merged class with the new class name
        new_matches = []
        for match in self.matches:
            new_match = []
            for cls in match:
                if relevant[cls]:
                    new_match.append(new_class[cls])

            if len(new_match) > 1:
                new_matches.append(new_match)

        print("  Found %d matches that are relevant after merges" % len(new_matches))

        self.matches = new_matches

    # combine the classes in a 1d vector of labeled classes based on a 2d
    # listing of segments that should share the same class
    def _combine_classes(self):
        # filtered out non-unique matches
        unique_matches = set()
        for match in self.matches:
            unique_match = []
            for cls in match:
                if cls not in unique_match:
                    unique_match.append(cls)

            if len(unique_match) > 1:
                unique_match = tuple(sorted(unique_match))
                if unique_match not in unique_matches:
                    unique_matches.add(unique_match)

        unique_matches = [list(match) for match in unique_matches]
        print("  Found %d matches that are unique" % len(unique_matches))

        # combine matches with classes in common
        groups = {}
        class_group = {}

        for i, cls in enumerate(np.unique(unique_matches)):
            groups[i] = [cls]
            class_group[cls] = i

        for match in unique_matches:
            main_group = class_group[match[0]]

            for cls in match:
                other_group = class_group[cls]
                if other_group != main_group:
                    for other_class in groups[other_group]:
                        if other_class not in groups[main_group]:
                            groups[main_group].append(other_class)
                            class_group[other_class] = main_group

                    del groups[other_group]

        self.matches = [groups[i] for i in groups]
        print("  Found %d matches after grouping" % len(self.matches))

        # combine the sequences into the same class
        for match in self.matches:
            assert len(match) > 1
            for other_class in match[1:]:
                self.classes[self.classes == other_class] = match[0]

    # make class ids sequential and remove classes that are too small
    def _normalize_classes(self):
        # mask of segments to keep
        keep = np.ones(self.classes.size).astype(np.bool)

        # number of classes and current class counter
        self.n_classes = 0
        for i in np.unique(self.classes):
            # find the elements in the class
            idx = self.classes == i
            if np.sum(idx) >= self.min_class_size:
                # if class is large enough keep and relabel
                if self.normalize_classes:
                    self.classes[idx] = self.n_classes

                # found one more class
                self.n_classes = self.n_classes + 1
            else:
                # mark class for removal and delete label information
                keep = np.logical_and(keep, np.logical_not(idx))

        # remove data on the removed classes
        self._trim_data(keep)

    # remove segments that are too small compared to the last
    # element in the sequence
    def _remove_irrelevant(self):
        keep = np.ones(self.classes.size).astype(np.bool)
        for cls in np.unique(self.classes):
            class_ids = np.where(self.classes == cls)[0]
            last_id = class_ids[np.argmax(self.duplicate_ids[class_ids])]
            last_size = len(self.segments[last_id])

            for class_id in class_ids:
                segment_size = len(self.segments[class_id])
                if float(segment_size) / last_size < self.require_relevance:
                    keep[class_id] = False

        self._trim_data(keep)

        print("  Found %d segments that are relevant" % len(self.segments))

    # remove segments that are too similar based on hamming distance
    def _remove_similar(self, preprocessor):
        keep = np.ones(self.classes.size).astype(np.bool)
        for c in np.unique(self.classes):
            class_ids = np.where(self.classes == c)[0]

            # sort duplicates in chronological order
            class_ids = class_ids[np.argsort(self.duplicate_ids[class_ids])]

            segments_class = [self.segments[i] for i in class_ids]
            segments_class = preprocessor._rescale_coordinates(segments_class)
            segments_class = preprocessor._voxelize(segments_class)

            for i, segment_1 in enumerate(segments_class):
                for segment_2 in segments_class[i + 1 :]:
                    diff = np.sum(np.abs(segment_1 - segment_2))

                    if diff < self.require_diff_points:
                        keep[class_ids[i]] = False
                        break

        self._trim_data(keep)

        print("  Found %d segments that are dissimilar" % len(self.segments))

    def _sort_ids(self):
        ordered_ids = []
        for cls in np.unique(self.classes):
            class_ids = np.where(self.classes == cls)[0]
            class_sequences = self.duplicate_classes[class_ids]
            unique_sequences = np.unique(class_sequences)

            for unique_sequence in unique_sequences:
                sequence_ids = np.where(class_sequences == unique_sequence)[0]
                sequence_ids = class_ids[sequence_ids]
                sequence_frame_ids = self.duplicate_ids[sequence_ids]

                # order chronologically according to frame id
                sequence_ids = sequence_ids[np.argsort(sequence_frame_ids)]

                ordered_ids += sequence_ids.tolist()

        ordered_ids = np.array(ordered_ids)

        self.segments = [self.segments[i] for i in ordered_ids]
        self.classes = self.classes[ordered_ids]

        if self.positions.size > 0:
            self.positions = self.positions[ordered_ids]
        if self.features.size > 0:
            self.features = self.features[ordered_ids]

        self.duplicate_ids = self.duplicate_ids[ordered_ids]
        self.duplicate_classes = self.duplicate_classes[ordered_ids]

    # keep only segments and corresponding data where the keep parameter is true
    def _trim_data(self, keep):
        self.segments = [segment for (k, segment) in zip(keep, self.segments) if k]
        self.classes = self.classes[keep]

        if self.positions.size > 0:
            self.positions = self.positions[keep]
        if self.features.size > 0:
            self.features = self.features[keep]

        self.duplicate_ids = self.duplicate_ids[keep]
        self.duplicate_classes = self.duplicate_classes[keep]
