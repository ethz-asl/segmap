from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Preprocessor(object):
    def __init__(
        self,
        augment_angle=0.0,
        augment_remove_random_min=0.0,
        augment_remove_random_max=0.0,
        augment_remove_plane_min=0.0,
        augment_remove_plane_max=0.0,
        augment_jitter=0.0,
        scale=(1, 1, 1),
        voxels=(1, 1, 1),
        batch_size=16
    ):
        self.augment_remove_random_min = augment_remove_random_min
        self.augment_remove_random_max = augment_remove_random_max
        self.augment_remove_plane_min = augment_remove_plane_min
        self.augment_remove_plane_max = augment_remove_plane_max
        self.augment_angle = augment_angle
        self.augment_jitter = augment_jitter

        self.scale = np.array(scale)
        self.voxels = np.array(voxels)
        self.batch_size = batch_size

        self.color = False
        self.n_semcls = 35

        min_voxel_side_length_m = 0.1
        self.min_scale = self.voxels * min_voxel_side_length_m

        self.last_scales = []

    def init_segments(
        self, segments, segments_color, segments_class, classes
    ):
        self.segments = segments
        self.segments_color = segments_color
        self.segments_class = segments_class
        self.classes = np.array(classes)

    def get_processed(self, segment_ids, train=True):
        batch_segments = []
        batch_segments_color = []
        batch_segments_class = []
        for i in segment_ids:
            batch_segments.append(self.segments[i])
            batch_segments_color.append(self.segments_color[i])
            batch_segments_class.append(self.segments_class[i])

        batch_segments = self.process(batch_segments, batch_segments_color, batch_segments_class, train)
        batch_classes = self.classes[segment_ids]

        return batch_segments, batch_classes

    def process(self, segments, segments_color, segments_class, train=True):
        if train:
            if self.augment_remove_random_max > 0:
                segments, segments_color, segments_class = self._augment_remove_random(
                    segments, segments_color, segments_class)

            if self.augment_remove_plane_max > 0:
                segments, segments_color, segments_class = self._augment_remove_plane(
                    segments, segments_color, segments_class)

            if self.augment_angle > 0:
                segments = self._augment_rotation(segments)

        # rescale coordinates and center
        segments = self._rescale_coordinates(segments)

        # randomly displace the segment
        #if train and self.augment_jitter > 0:
        #    segments = self._augment_jitter(segments)

        # insert into voxel grid
        segments = self._voxelize(segments, segments_color, segments_class)

        return segments

    def get_n_batches(self, train=True):
        if train:
            return self.n_batches_train
        else:
            return self.n_batches_test

    # create rotation matrix that rotates point around
    # the origin by an angle theta, expressed in radians
    def _get_rotation_matrix_z(self, theta):
        R_z = [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]

        return np.array(R_z)

    # augment with multiple rotation of the same segment
    def _augment_rotation(self, segments):
        angle_rad = self.augment_angle * np.pi / 180

        for i in range(len(segments)):
            rotation = np.random.uniform(-angle_rad, angle_rad)
            rotation = self._get_rotation_matrix_z(rotation)
            segments[i] = np.matmul(segments[i], rotation.T)

        return segments

    def _augment_remove_random(self, segments, segments_color, segments_class):
        for i in range(len(segments)):
            # percentage of points to remove
            remove = (np.random.random() * (self.augment_remove_random_max
                - self.augment_remove_random_min) + self.augment_remove_random_min)
            n_keep = int(segments[i].shape[0] * (1 - remove))

            # get indices to keep
            idx = np.random.choice(segments[i].shape[0], n_keep, replace=False)

            segments[i] = segments[i][idx]
            segments_color[i] = segments_color[i][idx]
            segments_class[i] = segments_class[i][idx]

        return segments, segments_color, segments_class

    def _augment_remove_plane(self, segments, segments_color, segments_class):
        for i, segment in enumerate(segments):
            # center segment
            center = np.mean(segment, axis=0)
            segment = segment - center

            # slice off a section of the segment
            while True:
                # generate random plane
                plane_norm = np.random.random(3) - 0.5
                plane_norm = plane_norm / np.sqrt(np.sum(plane_norm ** 2))

                # on which side of the plane each point is
                sign = np.dot(segment, plane_norm)

                # find an offset that removes a desired amount of points
                found = False
                plane_offsets = np.linspace(
                    -np.max(self.scale), np.max(self.scale), 100
                )
                np.random.shuffle(plane_offsets)
                for plane_offset in plane_offsets:
                    keep = sign + plane_offset > 0
                    remove_percentage = 1 - (np.sum(keep) / float(keep.size))

                    if (
                        remove_percentage > self.augment_remove_plane_min
                        and remove_percentage < self.augment_remove_plane_max
                    ):
                        segments[i] = segments[i][keep]
                        segments_color[i] = segments_color[i][keep]
                        segments_class[i] = segments_class[i][keep]
                        found = True
                        break

                if found:
                    break

        return segments, segments_color, segments_class

    def _augment_jitter(self, segments):
        # TODO(smauq): Implement
        return segments

    def _rescale_coordinates(self, segments):
        # center corner to origin
        centered_segments = []
        for segment in segments:
            segment = segment - np.min(segment, axis=0)
            centered_segments.append(segment)
        segments = centered_segments

        # store the last scaling factors that were used
        self.last_scales = []

        rescaled_segments = []
        for segment in segments:
            # rescale coordinates to fit inside voxel matrix
            scale = np.max(segment, axis=0)
            thresholded_scale = np.maximum(scale, self.min_scale)
            segment = segment / thresholded_scale * (self.voxels - 1)

            # center segment inside voxel grid
            center = np.mean(segment, axis=0)
            segment = segment + (self.voxels - 1) / 2.0 - center

            self.last_scales.append(scale)
            rescaled_segments.append(segment)

        return rescaled_segments

    def _voxelize(self, segments, segments_color, segments_class):
        if self.color:
            voxelized_segments = np.zeros((len(segments),) + tuple(self.voxels) + (3 + self.n_semcls,))
        else:
            voxelized_segments = np.zeros((len(segments),) + tuple(self.voxels) + (1,))

        for i, segment in enumerate(segments):
            # remove out of bounds points
            keep = np.logical_and(np.all(segment < self.voxels, axis=1), np.all(segment >= 0, axis=1))
            segment = segment[keep]

            # round coordinates
            segment = segment.astype(np.int)

            # fill voxel grid
            if self.color:
                voxelized_segments[i, segment[:, 0], segment[:, 1], segment[:, 2], :3] = segments_color[i][keep]
                voxelized_segments[i, segment[:, 0], segment[:, 1], segment[:, 2], segments_class[i][keep] + 3] = 1
            else:
                voxelized_segments[i, segment[:, 0], segment[:, 1], segment[:, 2]] = 1

        return voxelized_segments
