from setuptools import setup, find_packages

# check some version of tensorflow has been installed
try:
    import tensorflow
    if not tensorflow.__version__.startswith('1.8'):
        print("Warning: You are running tensorflow version {}. segmappy was tested with version 1.8.0 . If you encounter issues, please report them to the segmap developers.".format(tensorflow.__version__))
except ImportError:
    print("Error: segmappy requires some version of Tensorflow (with/without GPU).")
    raise

# python package setup
setup(
    name="segmappy",
    version="0.1",
    description="Segmap Python Tools",
    url="http://github.com/ethz-asl/segmap",
    author="Andrei Cramariuc",
    author_email="andrei.cramariuc@gmail.com",
    license="MIT",
    packages=find_packages(),
    scripts=["bin/ensure_segmappy_is_installed.py",
             "bin/segmappy_train_cnn",
             "bin/segmappy_train_semantics",
             "bin/segmappy_plot_roc_from_matches",
             "bin/segmappy_plot_acc_versus_size",
             "bin/segmappy_download_datasets"],
    package_data = {'segmappy': ['config/*.ini']},
    install_requires = [
    "scikit-learn>=0.19.1",
    "scipy>=0.19.1",
    "configparser>=3.5.0",
    "future>=0.16.0",
    "matplotlib>=2.2.2",
    "numpy>=1.14.3",
    "pandas>=0.22.0"
    ],
    zip_safe=False,
)
