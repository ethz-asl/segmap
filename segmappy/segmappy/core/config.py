import configparser
import os

def get_segmap_home_dir():
    segmap_home = os.path.abspath(
        os.path.join(os.path.expanduser("~"), ".segmap/")
    )

    # If home directory doesn't exist create
    if not os.path.isdir(segmap_home):
        try:
            os.mkdir(segmap_home)
        except OSError as e:
            print('Error: Could not create SegMap home directory')
            raise

        # Copy config into the new home directory
        config_src = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '../config/default_training.ini')
        )

        import shutil
        shutil.copy(config_src, segmap_home)

    return segmap_home

def get_config_dir():
    # Returns the package-wide config directory.
    return get_segmap_home_dir()

def get_default_model_dir():
    # Returns the package-wide default trained model directory.
    return os.path.join(get_segmap_home_dir(), "trained_models/")

def get_default_dataset_dir():
    # Returns the package-wide default datasets directory.
    return os.path.join(get_segmap_home_dir(), "training_datasets/")

class Config(object):
    def __init__(self, name="train.ini"):
        path = os.path.join(get_config_dir(), name)
        if not os.path.isfile(path):
            raise IOError("Config file '{}' not found.".format(path))

        config = configparser.ConfigParser()
        config.read(path)

        # general
        try:
            self.base_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), config.get("general", "base_dir")
                )
            )
        except:
            self.base_dir = get_default_dataset_dir()
            print(
                "No dataset base directory provided, defaulting to {}.".format(
                    self.base_dir
                )
            )

        self.cnn_train_folders = config.get("general", "cnn_train_folders")
        self.cnn_test_folder = config.get("general", "cnn_test_folder")
        self.semantics_train_folder = config.get("general", "semantics_train_folder")
        self.use_merges = config.getboolean("general", "use_merges")
        self.keep_match_thresh = config.getfloat("general", "keep_match_thresh")
        self.use_matches = config.getboolean("general", "use_matches")
        self.min_class_size = config.getint("general", "min_class_size")
        self.require_relevance = config.getfloat("general", "require_relevance")
        self.require_diff_points = config.getint("general", "require_diff_points")

        # augment
        self.augment_angle = config.getfloat("augment", "augment_angle")
        self.augment_remove_random_min = config.getfloat(
            "augment", "augment_remove_random_min"
        )
        self.augment_remove_random_max = config.getfloat(
            "augment", "augment_remove_random_max"
        )
        assert self.augment_remove_random_max >= self.augment_remove_random_min
        self.augment_remove_plane_min = config.getfloat(
            "augment", "augment_remove_plane_min"
        )
        self.augment_remove_plane_max = config.getfloat(
            "augment", "augment_remove_plane_max"
        )
        assert self.augment_remove_plane_max >= self.augment_remove_plane_min
        self.augment_jitter = config.getfloat("augment", "augment_jitter")

        # normalize
        self.align = config.get("normalize", "align")
        assert self.align in ("none", "eigen", "robot")
        self.scale_method = config.get("normalize", "scale_method")
        assert self.scale_method in ("fixed", "aspect", "fit")
        self.center_method = config.get("normalize", "center_method")
        assert self.center_method in ("mean", "min_max", "none")
        self.scale = tuple(
            config.getint("normalize", "scale_" + axis) for axis in ("x", "y", "z")
        )
        self.voxels = tuple(
            config.getint("normalize", "voxels_" + axis) for axis in ("x", "y", "z")
        )
        self.remove_mean = config.getboolean("normalize", "remove_mean")
        self.remove_std = config.getboolean("normalize", "remove_std")

        # train
        try:
            self.model_base_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), config.get("train", "model_base_dir"))
            )
        except:
            self.model_base_dir = get_default_model_dir()
            print(
                "No model base directory provided, defaulting to {}.".format(
                    self.model_base_dir
                )
            )
        self.cnn_model_folder = os.path.abspath(
            os.path.join(self.model_base_dir, config.get("train", "cnn_model_folder"))
        )
        try:
            self.semantics_folder_name = config.get("train", "semantics_model_folder")
        except:
            self.semantics_folder_name = "semantics_nn"
        self.semantics_model_folder = os.path.abspath(
            os.path.join(self.model_base_dir, self.semantics_folder_name)
        )
        self.test_size = config.getfloat("train", "test_size")
        self.n_epochs = config.getint("train", "n_epochs")
        self.batch_size = config.getint("train", "batch_size")
        self.log_path = config.get("train", "log_path")
        self.debug_path = config.get("train", "debug_path")
