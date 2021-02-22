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
        gen = Generator(
            self.preprocessor,
            get_ids,
            None,
            train=True,
            batch_size=self.batch_size,
            shuffle=False,
        )

        descriptors = []
        for batch in range(gen.n_batches):
            batch_segments, _ = gen.next()

            batch_descriptors = self.tf_sess.run(
                self.tf_descriptor, feed_dict={self.tf_cnn_input: batch_segments,
                self.tf_cnn_scales: self.preprocessor.last_scales},
            )

            for batch_descriptor in batch_descriptors:
                descriptors.append(batch_descriptor)

        return np.array(descriptors)

    def next(self):
        batch_ids = []
        batch_ids_positive = []
        batch_ids_negative = []

        # pick set of randoms for negatives
        num_negative_batches = 8
        random_negative_ids = np.random.choice(
            self.segment_ids, self.batch_size * num_negative_batches)
        random_negative_classes = self.preprocessor.classes[random_negative_ids]
        random_negative_descriptors = self._get_descriptors(random_negative_ids)

        losses = []
        while True:
            # pick random anchor
            random_id = np.random.choice(self.segment_ids)
            random_class = self.preprocessor.classes[random_id]

            # find positives
            same_class_ids = self.segment_ids[np.where(
                self.segment_classes == random_class)[0]]
            if same_class_ids.size <= 1:
                continue

            random_positive_descriptors = self._get_descriptors(same_class_ids)
            anchor_descriptor = random_positive_descriptors[
                np.where(same_class_ids == random_id)[0][0]]

            # Form triplets
            for i in range(same_class_ids.size):
                positive_id = same_class_ids[i]
                if positive_id == random_id:
                    continue
                positive_descriptor = random_positive_descriptors[i]

                while True:
                    index = np.random.choice(len(random_negative_ids))
                    negative_id = random_negative_ids[index]
                    negative_descriptor = random_negative_descriptors[index]
                    if self.preprocessor.classes[negative_id] != random_class:
                        break

                batch_ids.append(random_id)
                batch_ids_positive.append(positive_id)
                batch_ids_negative.append(negative_id)

                dp = np.sum(np.square(anchor_descriptor - positive_descriptor))
                dn = np.sum(np.square(anchor_descriptor - negative_descriptor))
                l = max(self.margin + dp - dn, 0)
                losses.append(l)

                if len(batch_ids) == self.batch_size:
                    break

            if len(batch_ids) == self.batch_size:
                break

        #print("Estimated loss: ", np.mean(losses))

        batch_segments, _ = self.preprocessor.get_processed(
            batch_ids, train=self.train, pointnet=self.pointnet)
        self.cnn_scales = self.preprocessor.last_scales

        batch_segments_positive, _ = self.preprocessor.get_processed(
            batch_ids_positive, train=self.train, pointnet=self.pointnet)
        self.positive_scales = self.preprocessor.last_scales

        batch_segments_negative, _ = self.preprocessor.get_processed(
            batch_ids_negative, train=self.train, pointnet=self.pointnet)
        self.negative_scales = self.preprocessor.last_scales

        self.cnn_scales[:] = 0
        self.positive_scales[:] = 0
        self.negative_scales[:] = 0

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
