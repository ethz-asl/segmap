from __future__ import print_function

import numpy as np

def voxelize( segments, size ):
  voxelized_segments = []

  xyz_scales = []
  for segment in segments:
    ## Scale the segment into a [0,1]^3 cube
    xyz = np.copy(segment)
    xyz -= xyz.min(0)
    scale = xyz.max(0) - xyz.min(0)
    scale[np.where(scale == 0)] = 0.00001
    xyz /= scale
    xyz[:,np.where(scale == 0.0001)] = 0.5
    # xyz coordinates to ijk coordinates
    ind = np.floor(xyz*size).astype(int) 
    # if x, y, or z is on the edge, move it inside
    ind[np.where(ind == size)] = size-1
    # transform into tuple
    ind = ( ind.T[0] , ind.T[1] , ind.T[2] )
    # Voxels with a point inside set to 1
    occupancy = np.zeros((size,size,size))
    occupancy[ind]=1

    xyz_scales.append(scale[:])
    voxelized_segments.append( np.copy(occupancy) )

  return voxelized_segments, xyz_scales

def create_rotations(segments, n_angles=10,
                     offset_by_fraction_of_single_angle=0,
                     classes=[], silent=False):
  offset = offset_by_fraction_of_single_angle * (2*np.pi/n_angles)
  angles = np.linspace(2*np.pi/n_angles, 2*np.pi, n_angles) - offset
  rotated_segments = []
  rotated_classes = []
  for th in angles:
    x2x_y2y = np.array([ np.cos(th), -np.cos(th), 1 ])
    x2y_y2x = np.array([ np.sin(th),  np.sin(th), 0 ])
    rotated_segments = rotated_segments + [(segment*x2x_y2y+segment[:,[1,0,2]]*x2y_y2x)
                                           for segment in segments]
    if classes:
      rotated_classes = rotated_classes + classes

  if not silent:
    print("  Created " + str(len(rotated_segments)) + " rotated segments", end="")
    print(" (" + str(len(segments)) + " segments * " + str(n_angles) + " rotation angles)")
  if classes:
    if not silent: print("  Listed classes for rotated segments.")
    return rotated_segments, rotated_classes
  return rotated_segments

def unvoxelize(vox):
  result = np.array(np.where(vox > 0)).T.astype(float)
  if len(result) == 0:
    result = [np.array([0,0,0])]
  # Center points around 0, and scale down
  result = result - (np.array(vox.shape) / 2)
  result = result / np.array(vox.shape)
  return np.array(result)

def recenter_segment(segment):
  return segment - np.mean(segment, axis=0)

def create_twins(segments):
  angles = np.random.random((len(segments),2))
  twins_a = []; twins_b = []
  for offsets, segment in zip(angles, segments):
    twin_a = create_rotations([segment], n_angles=1, offset_by_fraction_of_single_angle=offsets[0], silent=True)[0]
    twin_b = create_rotations([segment], n_angles=1, offset_by_fraction_of_single_angle=offsets[1], silent=True)[0]
    twins_a.append(twin_a)
    twins_b.append(twin_b)
  return twins_a, twins_b

def random_rotated(segments, silent=True):
  angles = np.random.random((len(segments)))
  rotated_segments = []
  for offset, segment in zip(angles, segments):
    rotated_segment = create_rotations([segment], n_angles=1, offset_by_fraction_of_single_angle=offset, silent=silent)[0]
    rotated_segments.append(rotated_segment)
  return rotated_segments

def align(segments, precision=12):
    """ segments is a list of segment, each segment being a list of 3d points
        precision corresponds to the precision of alignment, measured in fractions of a quarter rotation"""
    aligned_segments = []
    longest_directions = []
    for segment in segments:
        if len(segment) < 2: 
            aligned_segments.append(segment)
            longest_directions.append(0)
            continue
        ## find the rotation of the segment which leads to the bounding box with largest aspect ratio
        best_aspect_ratio = 0
        angles = np.linspace(0, np.pi/2., precision)
        for angle in angles:
            rotated = create_rotations([segment], 1, angle, silent=True)[0]
            x = rotated[:,0]
            dx = max(x) - min(x)
            y = rotated[:,1]
            dy = max(y) - min(y)
            assert dx > 0 and dy > 0
            horizontal = dx > dy
            lrg = max((dx,dy))
            sml = min((dx,dy))
            if lrg/sml > best_aspect_ratio:
                best_angle = angle if horizontal else angle + np.pi/2
        ## Decide between the two possible angles, by choosing side with most points.
        rotated = create_rotations([segment], 1, best_angle, silent=True)[0]
        x = rotated[:,0]
        y = rotated[:,1]
        x_bias = 1.*np.sum(x > (min(x)/2. + max(x)/2.))/len(x) - 0.5
        y_bias = 1.*np.sum(y > (min(y)/2. + max(y)/2.))/len(y) - 0.5
        if abs(x_bias) > abs(y_bias):
            best_angle = best_angle if x_bias > 0 else np.mod(best_angle + np.pi, np.pi * 2.)
        else:
            best_angle = best_angle if y_bias < 0 else np.mod(best_angle + np.pi, np.pi * 2.)

        aligned_segments.append(create_rotations([segment], 1, best_angle, silent=True)[0])
        longest_directions.append(best_angle)
    return aligned_segments, longest_directions

