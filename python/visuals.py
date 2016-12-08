import numpy as np
from matplotlib import pyplot as plt

def points2d(x, y, c,
             small_fonts=True, no_axis=False, cmap_name='plasma', s=50):
    out = plt.scatter(x, y, c=c, alpha=0.8, marker='.', lw = 0, cmap=plt.get_cmap(cmap_name), s=s)
    if small_fonts: # Tiny fonts for axis tick numbers
      plt.setp(plt.gca().get_xticklabels(), fontsize=0)
      plt.setp(plt.gca().get_yticklabels(), fontsize=0)
    if no_axis: # No axes or bounding boxes at all
      plt.axis('off')
    return out

def visuals_of_segments(segments, ids, features=None,
                        directory="/tmp/online_matcher/visuals/", black_and_white=False):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig = None
    for id_, segment in zip(ids, segments):
        seg_features = None if features is None else features[ids.index(id_)]
        img_path = directory+"segment"+str(id_)+".png"
        single_segment_as_figure(segment, seg_features, black_and_white, fig).savefig(img_path)
        plt.close(plt.gcf())

def visuals_of_matches(matches, segments, ids, features=None,
                       directory="/tmp/online_matcher/visuals/", black_and_white=False, oneview=False):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig = None
    for i, match in enumerate(matches):
        all_ids_in_match_are_known = True
        for j, id_ in enumerate(match):
            if id_ not in ids:
                all_ids_in_match_are_known = False
                print("Unknown id ("+str(id_)+") in match["+str(j)+"].")
        if all_ids_in_match_are_known:
            img_paths = []
            for id_ in match:
                segment = segments[ids.index(id_)]
                seg_features = None if features is None else features[ids.index(id_)]
                img_path = directory+"match"+str(i)+"seg"+str(id_)+".png"
                img_paths.append(img_path)
                single_segment_as_figure(segment, seg_features, black_and_white, fig, oneview).savefig(img_path)
                plt.clf()
            # Concatenate images
            import sys
            from PIL import Image
            images = [Image.open(img_path) for img_path in img_paths]
            widths, heights = zip(*(i.size for i in images))
            total_width = sum(widths)
            max_height = max(heights)
            new_im = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for im in images:
                new_im.paste(im, (x_offset,0))
                x_offset += im.size[0]
            new_im.save(directory+"match"+str(i)+".png")
            for img_path in img_paths:
                os.remove(img_path)

def single_segment_as_figure(segment, seg_features=None, black_and_white=False, fig=None, oneview=False):
    if fig == None:
      fig = plt.figure('visuals')

    X = segment[:,0]
    Y = segment[:,1]
    Z = segment[:,2]
    th = -np.pi/4
    XP = X*np.cos(th) + Y*np.sin(th)
    ZP = Z*np.cos(th) - (-X*np.sin(th)+Y*np.cos(th))*np.sin(th)
    YP = Z*np.sin(th) + (-X*np.sin(th)+Y*np.cos(th))*np.cos(th)

    color= 'k' if black_and_white else YP

    if oneview:
      nvp = 2 if seg_features != None else 1
      plt.subplot(nvp, 1, 1)
      points2d(XP, ZP, color)
      plt.axis('equal')
    else:
      nvp = 4 if seg_features != None else 3
      plt.subplot(nvp, 3, 1)
      points2d(X, Z, color)
      plt.axis('equal')
      plt.subplot(nvp, 3, 3)
      points2d(-Y, Z, color)
      plt.axis('equal')
      plt.subplot(nvp, 3, 7)
      points2d(X, Y, color)
      plt.axis('equal')

      plt.subplot(nvp, 3, 2)
      points2d(X*np.cos(th) - Y*np.sin(th), Z, color)
      plt.axis('equal')
      plt.subplot(nvp, 3, 4)
      points2d(X, Z*np.cos(th) - Y*np.sin(th), color)
      plt.axis('equal')

      plt.subplot(nvp, 3, 5)
      points2d(XP, ZP, color)
      plt.axis('equal')

    if seg_features is not None:
        plt.subplot(nvp, 1, nvp)
        plt.bar(range(len(seg_features)), seg_features)


    plt.tight_layout()

    return fig

def single_segment_as_gif(segment,
                          directory="/tmp/online_matcher/visuals/animated/", frames=60, black_and_white=False):
  import os
  if not os.path.exists(directory):
      os.makedirs(directory)

  import voxelize
  rotations = voxelize.create_rotations([voxelize.recenter_segment(segment)], n_angles=frames)
  segments_as_gif(rotations, filename='segment',
                  directory=directory, black_and_white=black_and_white)

def single_segment_reconstruction_as_gif(segment, vae, confidence=0.3,
                                         directory="/tmp/online_matcher/visuals/animated/",
                                         frames=60, black_and_white=False):
  import os
  if not os.path.exists(directory):
      os.makedirs(directory)

  import voxelize
  import autoencoder.model
  VOXEL_SIDE = vae.MP.INPUT_SHAPE[0]
  segments_vox, features_voxel_scale = voxelize.voxelize([segment], VOXEL_SIDE)
  reconstruction_vox = vae.batch_encode_decode([np.reshape(sample, vae.MP.INPUT_SHAPE) for sample in segments_vox])
  reconstruction_vox = [np.reshape(vox, [VOXEL_SIDE, VOXEL_SIDE, VOXEL_SIDE]) for vox in reconstruction_vox]
  from voxelize import unvoxelize
  reconstruction = [unvoxelize(vox > confidence) for vox in reconstruction_vox]
  reconstruction = [voxelize.recenter_segment(segment*scale) for (segment, scale) in zip(reconstruction, features_voxel_scale)]
  rotations = voxelize.create_rotations(reconstruction, n_angles=frames)

  segments_as_gif(rotations, rotate_YP=(2*np.pi/frames), filename='reconstruction',
                  directory=directory, black_and_white=black_and_white)

def single_segment_rotations_reconstruction_as_gif(segment, vae, confidence=0.3,
                                                   directory="/tmp/online_matcher/visuals/animated/",
                                                   frames=120, black_and_white=False):
  import os
  if not os.path.exists(directory):
      os.makedirs(directory)

  import voxelize
  rotations = voxelize.create_rotations([segment], n_angles=frames)
  import autoencoder.model
  VOXEL_SIDE = vae.MP.INPUT_SHAPE[0]
  rotations_vox, features_voxel_scale = voxelize.voxelize(rotations, VOXEL_SIDE)
  reconstruction_vox = vae.batch_encode_decode([np.reshape(sample, vae.MP.INPUT_SHAPE) for sample in rotations_vox], batch_size=120)
  reconstruction_vox = [np.reshape(vox, [VOXEL_SIDE, VOXEL_SIDE, VOXEL_SIDE]) for vox in reconstruction_vox]
  from voxelize import unvoxelize
  reconstruction = [unvoxelize(vox > confidence) for vox in reconstruction_vox]
  reconstruction = [voxelize.recenter_segment(segment*scale) for (segment, scale) in zip(reconstruction, features_voxel_scale)]

  segments_as_gif(reconstruction, rotate_YP=(2*np.pi/frames), filename='reconstruction_rot',
                  directory=directory, black_and_white=black_and_white)

def single_segment_degeneration_as_gif(segment, vae, confidence=0.3,
                                       directory="/tmp/online_matcher/visuals/animated/",
                                       frames=60, black_and_white=False):
  import os
  if not os.path.exists(directory):
      os.makedirs(directory)

  import voxelize
  import autoencoder.model
  VOXEL_SIDE = vae.MP.INPUT_SHAPE[0]
  segment_vox, features_voxel_scale = voxelize.voxelize([segment], VOXEL_SIDE)
  segment_vox = [np.reshape(sample, vae.MP.INPUT_SHAPE) for sample in segment_vox]
  for i in range(frames):
    reconstruction_vox = vae.batch_encode_decode(reconstruction_vox) if i > 0 else segment_vox
    degen_vox = degen_vox + list(reconstruction_vox) if i > 0 else list(reconstruction_vox)
  degen_vox = [np.reshape(vox, [VOXEL_SIDE, VOXEL_SIDE, VOXEL_SIDE]) for vox in degen_vox]
  from voxelize import unvoxelize
  reconstruction = [unvoxelize(vox > confidence) for vox in degen_vox]
  reconstruction = [voxelize.recenter_segment(segment*features_voxel_scale[0]) for segment in reconstruction]
  print(len(reconstruction))

  segments_as_gif(reconstruction, rotate_YP=0, filename='degeneration',
                  directory=directory, black_and_white=black_and_white)

def single_segment_confidence_as_gif(segment, vae,
                                     directory="/tmp/online_matcher/visuals/animated/",
                                     frames=60, black_and_white=False):
  import os
  if not os.path.exists(directory):
      os.makedirs(directory)

  import voxelize
  import autoencoder.model
  VOXEL_SIDE = vae.MP.INPUT_SHAPE[0]
  segment_vox, features_voxel_scale = voxelize.voxelize([segment], VOXEL_SIDE)
  segment_vox = [np.reshape(sample, vae.MP.INPUT_SHAPE) for sample in segment_vox]
  reconstruction_vox = vae.batch_encode_decode(segment_vox)
  reconstruction_vox = [np.reshape(vox, [VOXEL_SIDE, VOXEL_SIDE, VOXEL_SIDE]) for vox in reconstruction_vox]
  from voxelize import unvoxelize
  cmin=0.1; cmax=np.amax(reconstruction_vox);
  confidences = list(np.linspace(cmin,cmax,frames/2))+list(np.linspace(cmax,cmin,frames/2))
  reconstruction = [unvoxelize(reconstruction_vox[0] > confidence) for confidence in confidences]
  reconstruction = [segment*features_voxel_scale[0] for segment in reconstruction]

  segments_as_gif(reconstruction, rotate_YP=0, filename='confidence',
                  directory=directory, black_and_white=black_and_white)

def segments_as_gif(segments, filename='segment', rotate_YP=None,
                    directory="/tmp/online_matcher/visuals/animated/", black_and_white=False, 
                    framerate=30):
  for i, segment in enumerate(segments):
    X = segment[:,0]
    Y = segment[:,1]
    Z = segment[:,2]
    th = -np.pi/4
    XP = X*np.cos(th) + Y*np.sin(th)
    ZP = Z*np.cos(th) - (-X*np.sin(th)+Y*np.cos(th))*np.sin(th)
    zmin = min(min(ZP), zmin) if i > 0 else min(ZP)
    zmax = max(max(ZP), zmax) if i > 0 else max(ZP)
    xmin = min(min(XP), xmin) if i > 0 else min(XP)
    xmax = max(max(XP), xmax) if i > 0 else max(XP)
    if rotate_YP == None:
      YP = Z*np.sin(th) + (-X*np.sin(th)+Y*np.cos(th))*np.cos(th)

  for i, segment in enumerate(segments):
    X = segment[:,0]
    Y = segment[:,1]
    Z = segment[:,2]
    th = -np.pi/4
    XP = X*np.cos(th) + Y*np.sin(th)
    ZP = Z*np.cos(th) - (-X*np.sin(th)+Y*np.cos(th))*np.sin(th)
    if rotate_YP != None:
      # keep color consistent between rotated reconstructions
      phi = rotate_YP*i
      x2x_y2y = np.array([ np.cos(phi), -np.cos(phi), 1 ])
      x2y_y2x = np.array([ np.sin(phi),  np.sin(phi), 0 ])
      unrotated = segment*x2x_y2y+segment[:,[1,0,2]]*x2y_y2x
      XuR = unrotated[:,0]
      YuR = unrotated[:,1]
      YP = Z*np.sin(th) + (-XuR*np.sin(th)+YuR*np.cos(th))*np.cos(th)

    fig = plt.figure('visuals')
    color= 'k' if black_and_white else -YP
    points2d(XP, ZP, color, no_axis=True, s=130)
    plt.ylim([zmin, zmax])
    plt.xlim([xmin, xmax])
    plt.gca().set_aspect('equal', adjustable='box')

    plt.tight_layout()
    img_path = directory+"frame"+str(i).zfill(3)+".png"
    plt.gcf().savefig(img_path)
    saved_fig_paths = saved_fig_paths + [img_path] if i > 0 else [img_path]
    plt.clf()

  import subprocess
  subprocess.call(['ffmpeg', '-framerate', '30', '-i', directory+'frame%03d.png', '-r', '30', '-y', directory+'output.mp4'])
  subprocess.call(['ffmpeg', '-i', directory+'output.mp4', '-y', directory+filename+'.gif'])

  import os
  for path in saved_fig_paths:
    os.remove(path)

def compare_features(features_a, features_b, feature_labels=None, N = 30, d = 1.5, title = ""):
  feature_labels = ['unknown']*len(features_a[0]) if feature_labels is None else feature_labels
  import matplotlib.pyplot as plt
  plt.ion()

  plt.figure(figsize=(26,8))
  # a segments features
  ax = plt.subplot(2, 1, 1)
  plt.title("Comparison of Features"+title)
  colors = ['b','g','r','y','w'] 
  n_features = min([len(features_a[0]),5])
  for i in range(0,N):
    f = features_a[i]
    for j in range(0, n_features ):
      ax.bar((i*d)-0.2+j*0.2, f[j],width=0.2,color=colors[j],align='center')
  plt.xlim([-d, N*d+d])
  plt.ylabel("Feature value in A")
  # legends
  import matplotlib.patches as mptch
  patches = [mptch.Patch(color=colors[i], label=feature_labels[i]) for i in range(0,n_features)]
  plt.legend(handles=patches)
  # b segments features
  ax = plt.subplot(2, 1, 2)
  for i in range(0,N):
    f = features_b[i]
    for j in range(0, n_features ):
      ax.bar((i*d)-0.2+j*0.2, f[j],width=0.2,color=colors[j],align='center')
  plt.xlim([-d, N*d+d])
  plt.ylabel("Feature value in B")
  plt.xlabel("Object #")
  plt.tight_layout()

  return

def cycle_color(i):
  import matplotlib
  colors=list(matplotlib.colors.cnames.keys())
  return colors[i%len(colors)]

def visualize_matches_for_two_features(x_axis_feature_index, y_axis_feature_index,
                                       ids, features, feature_names, matches,
				       hide_segments_with_zero_matches=True):
  X = []
  Y = []
  C = []
  for i, group in enumerate(matches):
    group_color = cycle_color(i)
    if hide_segments_with_zero_matches and len(group)==1:
      continue
    for seg_id in group:
      point_x = features[ids.index(seg_id)][x_axis_feature_index]
      point_y = features[ids.index(seg_id)][y_axis_feature_index]
      X.append(point_x)
      Y.append(point_y)
      C.append(group_color)
  import matplotlib.pyplot as plt
  import matplotlib.cm as cm
  plt.scatter(X,Y,color=C, lw = 0)
  plt.xlabel(feature_names[x_axis_feature_index])
  plt.ylabel(feature_names[y_axis_feature_index])

def tSNE(ids, features):
  raise NotImplementedError
  return 0
