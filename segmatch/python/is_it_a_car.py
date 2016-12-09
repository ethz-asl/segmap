import numpy as np
from scipy.misc import imread
import os

import matplotlib.pyplot as plt
plt.close("all")
plt.ion()
plt.figure()

import sys
script_path = sys.argv[0]
output_file = "./database/classes.csv"
directory = "/tmp/online_matcher/visuals/"
filenames = os.listdir(directory)

segment_filenames = [filename for filename in filenames if "segment" in filename]
segment_ids = np.array([int(segment_filename.rsplit('.',1)[0].rsplit('segment',1)[1]) for segment_filename in segment_filenames])
sort_indx = segment_ids.argsort()
segment_filenames = np.array(segment_filenames)[sort_indx]
segment_ids = np.array(segment_ids)[sort_indx]

classes = [[id_, "unknown"] for id_ in segment_ids]

i = 0

# Load already classified values
with open(output_file) as file_:
    for line in file_:
        split_line = line.strip().split(' ')
        segment_id = int(split_line[0])
        class_ = split_line[1]

        if class_ != "unknown":
            i = list(segment_ids).index(segment_id)
            print("Segment of id "+str(segment_id)+" is already classified")
            assert classes[i] == [segment_id, "unknown"]
            classes[i] = [segment_id, class_]

while True:
    id_ = segment_ids[i]
    segment_filename = segment_filenames[i]
    image = imread(directory+segment_filename)
    plt.figure(1)
    plt.cla()
    plt.imshow(image)
    plt.title(str(id_))

    keys = input("Is it a car? (c=car, Default=no) ")
    class_ = None
    if keys == 'c':
        class_ = "car"
    if keys == 'w':
        class_ = "wall"
    if keys == 'b':
        i = i - 2
    if keys == 'q':
        break

    if class_ is not None:
        classes[i] = [id_, class_]

    i = i + 1
    if i >= len(segment_ids):
        break

from import_export import write_list_of_lists
write_list_of_lists(classes, output_file)

plt.close("all")
