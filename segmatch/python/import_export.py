from __future__ import print_function

import numpy as np

database_folder = "./database/"

def load_segments(folder=database_folder, filename="segments_database.csv"):
  # container
  segments = []

  # extract and store point data
  from pandas import read_csv
  extracted_data = read_csv(folder+filename, delimiter=' ').as_matrix()
  point_ids = extracted_data[:,0].astype(int)
  points = extracted_data[:,1:]
  id_changes = np.where(np.abs(np.diff(point_ids)) > 0)[0] + 1
  segments = np.split(points, id_changes)
  ids = [seg[0] for seg in np.split(point_ids, id_changes)]
  if len(set(ids)) != len(ids):
    raise ValueError("Id collision when importing segments. Two segments with same id exist in file.")

  print("  Found " + str(len(segments)) + " segments")
  return segments, ids


def load_features(folder=database_folder, filename="features_database.csv"):
  # containers
  ids = []
  features = []
  feature_names = []

  with open(folder+filename) as inputfile:
    for line in inputfile:
      split_line = line.strip().split(' ')
      # feature names
      if len(feature_names) == 0:
        feature_names = split_line[1::2]
      # id
      segment_id = split_line[0]
      ids.append(int(segment_id))
      # feature values
      features.append( np.array([float(i) for i in split_line[2::2]]) )

  print("  Found features for " + str(len(features)) + " segments", end="")
  if 'autoencoder_feature1' in feature_names: print("(incl. autoencoder features)", end="")
  print(" ")
  return features, feature_names, ids



def load_matches(folder=database_folder, filename="matches_database.csv"):
  # containers
  matches = []

  with open(folder+filename) as inputfile:
    for line in inputfile:
      split_line = line.strip().split(' ')
      matches.append( [int(i) for i in split_line if i != ''] )

  print("  Found " + str(len(matches)) + " groups of matches")
  return np.array(matches)

def load_classes(folder=database_folder, filename="classes_database.csv"):
  ids_and_classes = load_list_of_lists(folder+filename)
  classes = [item[1] for item in ids_and_classes]
  ids = [item[0] for item in ids_and_classes]
  return classes, ids


def write_features(ids, features, fnames, folder=database_folder, filename="features_database.csv"):
  assert len(ids) == len(features)
  assert len(fnames) == len(features[0])

  # mix together the ids, features and fnames
  ids_features_and_names = [[ids[i]]+
                        [item for sublist in zip(fnames,f) for item in sublist]
                        for i, f in enumerate(features)]

  # Write everything to file
  write_list_of_lists(ids_features_and_names, folder+filename)

  print("  " + str(len(ids_features_and_names)) + " features written ", end="")
  print("to " + folder+filename)



def write_matches(matches, folder=database_folder, filename="predicted_matches.csv"):
  # Write everything to file
  write_list_of_lists(matches, folder+filename)

  print("  " + str(len(matches)) + " matches written ", end="")
  print("to " + folder+filename)

def write_segments(ids, segments, folder=database_folder, filename="segments_database.csv"):
  path = folder+filename
  with open(path, 'w') as outputfile:
      for id_, segment in zip(ids, segments):
          for point in segment:
              outputfile.write(str(id_))
              for value in point:
                  outputfile.write(" " + str(value))
              outputfile.write('\n')

def write_classes(ids, classes, folder=database_folder, filename="classes_database.csv"):
  assert len(ids) == len(classes)

  # mix together the ids and classes
  ids_and_classes =    [[id_] + [class_]
                        for id_, class_ in zip(ids, classes)]

  # Write everything to file
  write_list_of_lists(ids_and_classes, folder+filename)

  print("  " + str(len(ids_and_classes)) + " segment classes written ", end="")
  print("to " + folder+filename)

def write_list_of_lists(list_of_lists, path):
  with open(path, 'w') as outputfile:
      for line in list_of_lists:
          for value in line:
              outputfile.write(str(value) + " ")
          outputfile.write('\n')

def load_list_of_lists(path):
  with open(path) as infile:
      list_of_lists = [line.strip().split(' ') for line in infile]
  return convert_strings_to_floats_in_list_of_lists(list_of_lists)


def convert_strings_to_floats_in_list_of_lists(list_of_lists):
  result = []
  for list_ in list_of_lists:
    result_line = []
    for item in list_:
      try:
        num = float(item)
        if num.is_integer():
            num = int(num)
        result_line.append(num)
      except:
        result_line.append(item)
    result.append(result_line)
  return result

def load_playground_segments():
  folder = "../point clouds/segments/"
  # points
  a = []
  b = []
  # difference of normals
  a_don = []
  b_don = []

  # list files
  import os
  load_playground_segments.filenames = os.listdir(folder)

  # find the segment files
  def filename_is_segment_filename(filename):
    assert type(filename) is str
    # these rules define the pattern for a segment file name
    # [a|b][number][.pcd]
    return (
          filename[0] in [ 'a' , 'b' ] \
      and filename[1:-4].isdigit()     \
      and filename[-4:] == ".pcd"      \
      )
  segment_filenames = [i for i in load_playground_segments.filenames if filename_is_segment_filename(i)]
  n_segment_files = len(segment_filenames)
  if n_segment_files == 0:
    print("  error: no segment files found")

  # Find the DoN files
  def filename_is_don_filename(filename):
    assert type(filename) is str
    # these rules define the pattern for a DoN file name
    # [DON_][a|b][number][.pcd]
    return (
          filename[0:4] == "DON_"       \
      and filename[4] in [ 'a' , 'b' ] \
      and filename[5:-4].isdigit()     \
      and filename[-4:] == ".pcd"      \
      )
  don_filenames = [i for i in load_playground_segments.filenames if filename_is_don_filename(i)]
  n_don_files = len(don_filenames)
  if n_don_files != n_segment_files:
    print("  warning: amount of DoN files does not match amount of segment files")

  n_segments = n_segment_files/2
  print("  Found " + str(n_segments) + " segments")
  for i in range(1,n_segments+1):
    n = str(i)
    # extract and store point data
    an_data = np.loadtxt(folder+"a"+n+".pcd", delimiter=" ", skiprows=11, unpack=False)
    bn_data = np.loadtxt(folder+"b"+n+".pcd", delimiter=" ", skiprows=11, unpack=False)
    a.append( an_data[:,0:3] )
    b.append( bn_data[:,0:3] )
    # extract and store DoN data
    if i <= ( n_don_files/2 ):
      an_don_data = np.loadtxt(folder+"DON_a"+n+".pcd", delimiter=" ", unpack=False)
      bn_don_data = np.loadtxt(folder+"DON_b"+n+".pcd", delimiter=" ", unpack=False)
      a_don.append( an_don_data )
      b_don.append( bn_don_data )

  return a, b, a_don, b_don
