from __future__ import print_function
import numpy as np

from ..tools.classifiertools import to_onehot


class Generator(object):
    def __init__(
        self,
        preprocessor,
        segment_ids,
        n_classes,
        train=True,
        batch_size=16,
        shuffle=False,
        pointnet=False
    ):
        self.preprocessor = preprocessor
        self.segment_ids = segment_ids
        self.n_classes = n_classes
        self.train = train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pointnet = pointnet

        self.n_segments = len(self.segment_ids)
        self.n_batches = int(np.ceil(float(self.n_segments) / batch_size))

        self._i = 0

    def __iter__(self):
        return self

    def next(self):
        if self.shuffle and self._i == 0:
            np.random.shuffle(self.segment_ids)

        self.batch_ids = self.segment_ids[self._i : self._i + self.batch_size]

        self._i = self._i + self.batch_size
        if self._i >= self.n_segments:
            self._i = 0

        batch_segments, batch_classes = self.preprocessor.get_processed(
            self.batch_ids, train=self.train, pointnet=self.pointnet)

        if self.n_classes is None:
            batch_classes = np.array([])
        else:
            batch_classes = to_onehot(batch_classes, self.n_classes)

        return batch_segments, batch_classes

class GeneratorTriplet(object):
    def __init__(
        self,
        preprocessor,
        segment_ids,
        margin=0.1,
        train=True,
        batch_size=16,
        pointnet=False
    ):
        self.preprocessor = preprocessor
        self.segment_ids = segment_ids
        self.train = train
        self.batch_size = batch_size
        self.margin = margin
        self.pointnet = pointnet

        self.n_segments = len(self.segment_ids)
        self.n_batches = int(np.ceil(float(self.n_segments) / batch_size) / 10)
        self.segment_classes = self.preprocessor.classes[self.segment_ids]

    def __iter__(self):
        return self

    def init_model(self, sess, cnn_input, cnn_scales, descriptor):
        self.tf_sess = sess
        self.tf_cnn_input = cnn_input
        self.tf_cnn_scales = cnn_scales
        self.tf_descriptor = descriptor

    def _get_descriptors(self, get_ids):
        segments, _ = self.preprocessor.get_processed(
            get_ids, train=self.train, pointnet=self.pointnet)

        descriptors = self.tf_sess.run(
            self.tf_descriptor, feed_dict={self.tf_cnn_input: segments,
            self.tf_cnn_scales: self.preprocessor.last_scales},
        )

        scales = self.preprocessor.last_scales

        return descriptors, segments, scales

    def next(self, soft_margin, neg_subset=80, neg_pick=5):
        batch_segments = []
        batch_segments_positive = []
        batch_segments_negative = []

        self.cnn_scales = []
        self.positive_scales = []
        self.negative_scales = []

        # pick set of randoms for negatives
        random_negative_ids = np.random.choice(self.segment_ids, neg_subset)
        random_negative_classes = self.preprocessor.classes[random_negative_ids]
        random_negative_descriptors, random_negative_segments, \
            random_negative_scales = self._get_descriptors(random_negative_ids)

        while True:
            # pick random anchor
            random_id = np.random.choice(self.segment_ids)
            random_class = self.preprocessor.classes[random_id]

            # find positives
            same_class_ids = self.segment_ids[np.where(
                self.segment_classes == random_class)[0]]
            if same_class_ids.size <= 1:
                continue

            np.random.shuffle(same_class_ids)
            same_class_ids = same_class_ids[:8]

            idx = np.where(same_class_ids == random_id)[0]
            if idx.size != 0:
                same_class_ids[idx[0]] = same_class_ids[-1]
            same_class_ids[-1] = random_id

            random_positive_descriptors, random_positive_segments, \
                random_positive_scales = self._get_descriptors(same_class_ids)

            anchor_descriptor = random_positive_descriptors[-1]
            anchor_segment = random_positive_segments[-1]
            anchor_scale = random_positive_scales[-1]

            # Form triplets
            for i in range(same_class_ids.size - 1):
                positive_id = same_class_ids[i]
                positive_descriptor = random_positive_descriptors[i]
                positive_segment = random_positive_segments[i]
                positive_scale = random_positive_scales[i]

                dp = np.sum(np.square(anchor_descriptor - positive_descriptor))
                dn = np.sum(np.square(anchor_descriptor - random_negative_descriptors),
                    axis=1)
                losses = self.margin + dp - dn

                idx = np.argsort(losses)[::-1]
                idx = np.random.choice(idx[:neg_pick])
                negative_id = random_negative_ids[idx]
                negative_class = random_negative_classes[idx]

                if losses[idx] <= 0 or negative_class == random_class:
                    continue

                if losses[idx] >= self.margin:
                    continue

                negative_segment = random_negative_segments[idx]
                negative_scale = random_negative_scales[idx]

                batch_segments.append(anchor_segment)
                batch_segments_positive.append(positive_segment)
                batch_segments_negative.append(negative_segment)

                self.cnn_scales.append(anchor_scale)
                self.positive_scales.append(positive_scale)
                self.negative_scales.append(negative_scale)

                if len(batch_segments) == self.batch_size:
                    break

            if len(batch_segments) == self.batch_size:
                break

        batch_segments = np.array(batch_segments)
        batch_segments_positive = np.array(batch_segments_positive)
        batch_segments_negative = np.array(batch_segments_negative)

        self.cnn_scales = np.array(self.cnn_scales)
        self.positive_scales = np.array(self.positive_scales)
        self.negative_scales = np.array(self.negative_scales)

        return batch_segments, batch_segments_positive, batch_segments_negative

class GeneratorFeatures(object):
    def __init__(self, features, classes, n_classes=2, batch_size=16, shuffle=True):
        self.features = features
        self.classes = np.asarray(classes)
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = features.shape[0]
        self.n_batches = int(np.ceil(float(self.n_samples) / batch_size))
        self._i = 0

        self.sample_ids = list(range(self.n_samples))
        if shuffle:
            np.random.shuffle(self.sample_ids)

    def next(self):
        batch_ids = self.sample_ids[self._i : self._i + self.batch_size]

        self._i = self._i + self.batch_size
        if self._i >= self.n_samples:
            self._i = 0

        batch_features = self.features[batch_ids, :]
        batch_classes = self.classes[batch_ids]
        batch_classes = to_onehot(batch_classes, self.n_classes)

        return batch_features, batch_classes
