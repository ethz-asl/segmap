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
                     classes=[]):
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

  print("  Created " + str(len(rotated_segments)) + " rotated segments", end="")
  print(" (" + str(len(segments)) + " segments * " + str(n_angles) + " rotation angles)")
  if classes:
    print("  Listed classes for rotated segments.")
    return rotated_segments, rotated_classes
  return rotated_segments

def unvoxelize(vox):
  result = np.array(np.where(vox > 0)).T.astype(float)
  if len(result) == 0:
    result = [np.array([0,0,0])]
  return np.array(result)

def recenter_segment(segment):
  return segment - np.mean(segment, axis=0)
