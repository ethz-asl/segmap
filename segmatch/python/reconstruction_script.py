from __future__ import print_function
import numpy as np

## COMMAND LINE ARGUMENTS ##
############################
import sys
if len(sys.argv) != 5:
  print("Arguments provided:")
  print(str(sys.argv))
  print("Prescribed:")
  print("$ python [this_script] [model_path] [segments_path] [features_output_path] [latent_space_dimension]")
  exit()
script_path = sys.argv[0]
model_path = sys.argv[1]
segments_fifo_path = sys.argv[2]
features_fifo_path = sys.argv[3]
latent_space_dim = int(sys.argv[4])
RC_CONFIDENCE = 0.2
voxel_side = 24

import os
try:
  os.unlink(segments_fifo_path)
  os.unlink(features_fifo_path)
  print("Unlinked FIFO")
except:
  print("")


## INIT AUTOENCODER ##
######################
print("Restoring Model...")
import pickle
try:
  MP = pickle.load(open(model_path+"model_params.pckl", 'rb'))
except:
  raise ValueError('Could not find model params')
from autoencoder import model
vae = model.Autoencoder(MP)
# Restore from previous run
try:
  vae.saver.restore(vae.sess, model_path+"model.checkpoint")
  print("Model restored.")
except:
  raise ValueError("Could not load model.")

print("__INIT_COMPLETE__")


os.mkfifo(segments_fifo_path)
os.mkfifo(features_fifo_path)

try:
  if True:
    ## LOAD DATA ##
    ###############
    print("Waiting for features...")
    from import_export import load_features
    features, feature_names, ids = load_features(folder="", filename=features_fifo_path)
    features = np.array(features)
    ae_features = features[:,:-3]
    sc_features = features[:,-3:]

    ## PROFILING ##
    ###############
    PROFILING = True
    if PROFILING:
        from timeit import default_timer as timer
        total_start = timer()
        reconstr_start = timer()

    ## RECONSTRUCT SEGMENTS ##
    ##########################
    segments_vox = vae.batch_decode(ae_features)
    segments_vox = [np.reshape(vox, [voxel_side, voxel_side, voxel_side]) for vox in segments_vox]
    from voxelize import unvoxelize
    segments = [unvoxelize(vox > RC_CONFIDENCE) for vox in segments_vox]
    segments = [segment*scale for (segment, scale) in zip(segments, sc_features)]

    if PROFILING:
        reconstr_end = timer()
        overhead_out_start = timer()

    print("__RCST_COMPLETE__")

    ## OUTPUT DATA TO FILES ##
    ##########################
    print("Writing segments")
    from import_export import write_segments
    write_segments(ids, segments, folder="", filename=segments_fifo_path)

    if PROFILING:
        overhead_out_end = timer()
        total_end = timer()
        print("Timings:")
        print("  Total - ", end='')
        print(total_end - total_start)
        print("  Reconstruction - ", end='')
        print(reconstr_end - reconstr_start)
        print("  Overhead out - ", end='')
        print(overhead_out_end - overhead_out_start)
except KeyboardInterrupt:
  print("Autoencoder descriptor script interrupted")
finally:
  os.unlink(segments_fifo_path)
  os.unlink(features_fifo_path)

