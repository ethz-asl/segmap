from __future__ import print_function
from builtins import input
import numpy as np
import sys

# sequentially view a set of segments
def visualize(segments, extra_info=None, show_all=False, no_ticks=False):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # scale the axes to match for all the segments
    axes_min = np.array(np.min(segments[0], axis=0))
    axes_max = np.array(np.max(segments[0], axis=0))

    for seg in segments[1:]:
        axes_min = np.minimum(axes_min, np.min(seg, axis=0))
        axes_max = np.maximum(axes_max, np.max(seg, axis=0))

    # display segments
    fig_id = 1
    plt.ion()
    for i, seg in enumerate(segments):
        if show_all:
            fig_id = i + 1

        fig = plt.figure(fig_id)
        plt.clf()
        ax = fig.add_subplot(111, projection="3d")

        ax.set_xlim(axes_min[0], axes_max[0])
        ax.set_ylim(axes_min[1], axes_max[1])
        ax.set_zlim(axes_min[2], axes_max[2])

        if no_ticks:
            tick_count = 3
            ax.set_xticks(np.linspace(axes_min[0], axes_max[0], tick_count + 2)[1:-1])
            ax.set_yticks(np.linspace(axes_min[1], axes_max[1], tick_count + 2)[1:-1])
            ax.set_zticks(np.linspace(axes_min[2], axes_max[2], tick_count + 2)[1:-1])

            plt.setp(ax.get_xmajorticklabels(), visible=False)
            plt.setp(ax.get_ymajorticklabels(), visible=False)
            plt.setp(ax.get_zmajorticklabels(), visible=False)

        ax.scatter(seg[:, 0], seg[:, 1], seg[:, 2])

        info = "Segment " + str(i)
        if extra_info is not None:
            info = info + " " + str(extra_info[i])
        sys.stdout.write(info)

        fig.canvas.flush_events()

        if not show_all:
            key = input()
            if key == "q":
                break
        else:
            sys.stdout.write("\n")

    if show_all:
        input()

    plt.ioff()
    plt.close("all")


def to_onehot(y, n_classes):
    y_onehot = np.zeros((len(y), n_classes))
    for i, cls in enumerate(y):
        y_onehot[i, cls] = 1

    return y_onehot


# sequentially view a set of segments
def visualize_side_by_side(segments, extra_info=None, show_all=False):

    import matplotlib.cm as cm

    n_views = 6.0
    if len(segments) < n_views:
        return

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # scale the axes to match for all the segments
    axes_min = np.array(np.min(segments[0], axis=0))
    axes_max = np.array(np.max(segments[0], axis=0))
    max_range = 0
    for seg in segments[1:]:
        axes_min = np.minimum(axes_min, np.min(seg, axis=0))
        axes_max = np.maximum(axes_max, np.max(seg, axis=0))
        X = seg[:, 0]
        Y = seg[:, 1]
        Z = seg[:, 2]
        new_max_range = np.array([X.max() - X.min(), Y.max() - Y.min()]).max() / 2.0
        if new_max_range > max_range:
            max_range = new_max_range
    fig = plt.figure(1, frameon=False)
    plt.clf()
    cmap = plt.cm.jet
    # fig, axs = plt.subplots(1,len(segments), projection='3d', facecolor='w', edgecolor='w') #figsize=(15, 6)
    fig.subplots_adjust(hspace=.5, wspace=.001)

    views_ids = [0]
    segments_temp = []
    for i in range(int(n_views - 1)):
        idx = i * len(segments) / n_views
        print(idx)
        views_ids = views_ids + [int(idx)]
        segments_temp.append(segments[int(idx)])
    segments_temp.append(segments[len(segments) - 1])
    segments = segments_temp

    print(max_range)

    for i, seg in enumerate(segments):
        ax = fig.add_subplot(1, len(segments), i + 1, projection="3d")
        ax.set_xlim(axes_min[0], axes_max[0])
        ax.set_ylim(axes_min[1], axes_max[1])
        ax.set_zlim(axes_min[2], axes_max[2])

        mid_x = (seg[:, 0].max() + seg[:, 0].min()) * 0.5
        mid_y = (seg[:, 1].max() + seg[:, 1].min()) * 0.5
        mid_z = (seg[:, 2].max() + seg[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_aspect(1)

        plt.setp(ax.get_xmajorticklabels(), visible=False)
        plt.setp(ax.get_ymajorticklabels(), visible=False)
        plt.setp(ax.get_zmajorticklabels(), visible=False)

        tick_count = 3
        ax.set_xticks(np.linspace(axes_min[0], axes_max[0], tick_count + 2)[1:-1])
        ax.set_yticks(np.linspace(axes_min[1], axes_max[1], tick_count + 2)[1:-1])
        ax.set_zticks(np.linspace(axes_min[2], axes_max[2], tick_count + 2)[1:-1])

        ax.set_xticklabels([1, 2, 3, 4])
        # fig.patch.set_visible(False)
        # ax.axis('off')
        ax.scatter(
            seg[:, 0],
            seg[:, 1],
            seg[:, 2],
            s=1,
            c=seg[:, 2],
            marker="o",
            lw=0,
            depthshade=False,
            cmap="jet_r",
        )
        ax.grid(b=False)
        ax.patch.set_facecolor("white")
        ax.set_axis_off()
    plt.draw()
    plt.pause(0.001)

    key = input()


def get_default_dataset(config, folder):
    from ..core.dataset import Dataset

    dataset = Dataset(
        folder=folder,
        base_dir=config.base_dir,
        use_merges=config.use_merges,
        use_matches=config.use_matches,
        min_class_size=config.min_class_size,
        require_diff_points=config.require_diff_points,
        keep_match_thresh=config.keep_match_thresh,
        require_relevance=config.require_relevance,
    )

    return dataset


def get_default_preprocessor(config):
    from ..core.preprocessor import Preprocessor

    preprocessor = Preprocessor(
        augment_angle=config.augment_angle,
        augment_remove_random_min=config.augment_remove_random_min,
        augment_remove_random_max=config.augment_remove_random_max,
        augment_remove_plane_min=config.augment_remove_plane_min,
        augment_remove_plane_max=config.augment_remove_plane_max,
        augment_jitter=config.augment_jitter,
        align=config.align,
        scale_method=config.scale_method,
        scale=config.scale,
        center_method=config.center_method,
        voxels=config.voxels,
        remove_mean=config.remove_mean,
        remove_std=config.remove_std,
        batch_size=config.batch_size,
    )

    return preprocessor
