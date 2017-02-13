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
  while True:
    ## LOAD DATA ##
    ###############
    print("Waiting for segments...")
    from import_export import load_segments
    segments, ids = load_segments(folder="", filename=segments_fifo_path)

    ## PROFILING ##
    ###############
    PROFILING = True
    if PROFILING:
        from timeit import default_timer as timer
        total_start = timer()
        vox_start = timer()

    # Voxelize
    voxel_side = 24
    voxel_size = voxel_side * voxel_side * voxel_side
    from voxelize import voxelize
    segments_vox, xyz_scale_features = voxelize(segments,voxel_side)

    if PROFILING:
        vox_end = timer()
        predict_start = timer()

    ## COMPUTE FEATURES ##
    ######################
    ae_features, ae_log_sigma_sq = vae.batch_encode([np.reshape(vox, MP.INPUT_SHAPE) for vox in segments_vox])
    ae_features[:,:vae.MP.COERCED_LATENT_DIMS] = 0
    ae_fnames = ['autoencoder_'+str(i) for i in range(latent_space_dim)]
    sc_features = [sorted(xyz_scale) + list(xyz_scale) for xyz_scale in xyz_scale_features]
    sc_fnames = ['scale_sml', 'scale_med', 'scale_lrg', 'scale_x', 'scale_y', 'scale_z']
    features = [list(ae) + list(sc) for ae, sc in zip(ae_features, sc_features)]
    fnames = ae_fnames + sc_fnames

    if PROFILING:
        predict_end = timer()
        overhead_out_start = timer()

    print("__DESC_COMPLETE__")

    ## OUTPUT DATA TO FILES ##
    ##########################
    print("Writing features")
    from import_export import write_features
    write_features(ids, features, fnames, folder="", filename=features_fifo_path)

    if PROFILING:
        overhead_out_end = timer()
        total_end = timer()
        print("Timings:")
        print("  Total - ", end='')
        print(total_end - total_start)
        print("  Voxelization - ", end='')
        print(vox_end - vox_start)
        print("  Predict - ", end='')
        print(predict_end - predict_start)
        print("  Overhead out - ", end='')
        print(overhead_out_end - overhead_out_start)
except KeyboardInterrupt:
  print("Autoencoder descriptor script interrupted")
finally:
  os.unlink(segments_fifo_path)
  os.unlink(features_fifo_path)

