## *SegMap* 

SegMap is a map representation based on 3D segments allowing for robot localization, environment reconstruction, and semantics extraction. The SegMap code is open-source (BSD License) and has been tested under Ubuntu 14.04, 16.04 and ROS Indigo, Kinetic. Please remember that this is on-going research code which is subject to changes in the future.

### Related Publications

R. Dubé, A. Cramariuc, D. Dugas, J. Nieto, R. Siegwart, and C. Cadena. **"SegMap: 3D Segment Mapping using Data-Driven Descriptors."** *Robotics: Science and Systems (RSS), 2018.* [pdf](http://www.roboticsproceedings.org/rss14/p03.pdf) - [video](https://youtu.be/CMk4w4eRobg)

R. Dubé, MG. Gollub, H. Sommer, I. Gilitschenski, R. Siegwart, C. Cadena and , J. Nieto. **"Incremental Segment-Based Localization in 3D Point Clouds."** *IEEE Robotics and Automation Letters, 2018.* [pdf](http://n.ethz.ch/~cesarc/files/RAL2018_rdube.pdf)

R. Dubé, D. Dugas, E. Stumm, J. Nieto, R. Siegwart, and C. Cadena. **"SegMatch: Segment Based Place Recognition in 3D Point Clouds."** *IEEE International Conference on Robotics and Automation, 2017.* [pdf](https://arxiv.org/pdf/1609.07720.pdf) - [video](https://youtu.be/iddCgYbgpjE)

## Features

- 3D CNN encoder – decoder
- ICP based LiDAR odometry
- Dynamic voxel grid
- Single and multi-robot SLAM back-end
- Generic incremental region-growing segmentation
- Incremental geometric verification


## Installation

The SegMap repository contains the following modules:
- segmap: The C++ library for 3D segment mapping.
- segmap_ros: ROS interface for segmap.
- segmapper: Example application using segmap and ros.
- laser_slam: Backend for the example application, based on LiDAR sensor data.
- segmappy: Python library for training and evaluating the neural network models.

This section provides a step by step guide to installing the full flavor of SegMap.
Advanced: it is also possible to use the SegMap C++ library standalone in a C++ project, or integrate the C++ library and ROS interface in a ROS project.

### Dependencies

First install the required system packages:
```
$ sudo apt-get install python-wstool doxygen python3-pip python3-dev python-virtualenv dh-autoreconf
```
Set up the workspace configuration:
```
$ mkdir -p ~/segmap_ws/src
$ cd ~/segmap_ws
$ catkin init
$ catkin config --merge-devel
$ catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
```
Then use wstool for fetching catkin dependencies:
```
$ cd src
$ git clone https://github.com/ethz-asl/segmap.git
$ wstool init
$ wstool merge segmap/dependencies.rosinstall
$ wstool update
```

#### Tensorflow

| If you do not plan on using the deep learned descriptors in SegMap you should at least install the CPU version of Tensorflow to compile the whole package. Otherwise we recommend the GPU version. |
| --- |

| The precompiled version of Tensorflow can only be used with Ubuntu 14.04, for further explanations see [here](https://github.com/tradr-project/tensorflow_ros_cpp#c-abi-difference-problems) (requirements for the GPU version [here](https://www.tensorflow.org/install/install_sources#tested_source_configurations)). For Ubuntu 16.04 and 18.04 you must compile tensorflow from source. After compiling tensorflow, generate the pip package and continue with these instructions, installing the custom generated package instead of the precompiled one. |
| --- |

SegMap relies on the package [tensorflow_ros_cpp](https://github.com/tradr-project/tensorflow_ros_cpp) for linking to the tensorflow C++ API. All our code has been tested with Tensorflow 1.8.

```
$ virtualenv ~/segmappyenv
$ source ~/segmappyenv/bin/activate
(segmappyenv)$ pip install --upgrade pip
(segmappyenv)$ pip install catkin_pkg empy pyyaml
(segmappyenv)$ pip install tensorflow-gpu==1.8.0
```

##### Build tensorflow_ros_cpp
For more details on the optional compile flags or in case of issues compiling see the [FAQ](https://github.com/ethz-asl/segmap/wiki/FAQ#q-issues-compiling-tensorflow_ros_cpp).

```
$ cd ~/segmap_ws
$ catkin build tensorflow_ros_cpp
```

### Build SegMap

Finally, build the *segmapper* package which will compile all dependencies and SegMap modules:
```
$ cd ~/segmap_ws
$ catkin build segmapper
```

#### (Optional) Install SegmapPy python package

Installing segmappy allows you to train data-driven models yourself to use with SegMap.
```
$ cd src/segmap/segmappy/
$ source ~/segmappyenv/bin/activate
(segmappyenv)$ pip install .
```

## Running segmapper Examples

Make sure to source the SegMap workspace before running the segmapper demonstrations:
```
$ source ~/segmap_ws/devel/setup.bash
```
To train new models see intructions [here](https://github.com/ethz-asl/segmap/wiki/Training-new-models).

#### Download demonstration files

To download all necessary files, copy the content of the [segmap_data](http://robotics.ethz.ch/~asl-datasets/segmap/segmap_data/) into ```~/.segmap/```. If you installed the segmappy python package you can run the automated download script. **Note: These models have been trained using Tensorflow 1.8 and are only guaranteed to work for that version.**

#### Run online SLAM example

An online SLAM example can be run with
```
$ roslaunch segmapper kitti_loop_closure.launch
```

#### Run localization against known map example 

A localization example can be run with
```
$ roslaunch segmapper kitti_localization.launch
```


#### Run online SLAM with CNN example

An online SLAM example with data-driven descriptor can be run with
```
$ roslaunch segmapper cnn_kitti_loop_closure.launch
```
You can now visualize the reconstructed target map in rviz by subscribing to `/segmatch/target_reconstruction`.

More details on the demonstrations can be found [here](https://github.com/ethz-asl/segmap/blob/master/wiki/demonstrations.md).


## License

SegMap is released under [BSD 3-Clause License](https://github.com/ethz-asl/segmap/blob/master/LICENSE)

Thank you for citing the related publication if you use SegMap in academic work:
```
@inproceedings{segmap2018,
  title={{SegMap}: 3D Segment Mapping using Data-Driven Descriptors},
  author={Dub{\'e}, Renaud and Cramariuc, Andrei and Dugas, Daniel and Nieto, Juan and Siegwart, Roland and Cadena, Cesar},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2018}
}
```

```
@inproceedings{segmatch2017,
  title={SegMatch: Segment based place recognition in 3d point clouds},
  author={Dub{\'e}, Renaud and Dugas, Daniel and Stumm, Elena and Nieto, Juan and Siegwart, Roland and Cadena, Cesar},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  pages={5266--5272},
  year={2017},
  organization={IEEE}
}
```

If you use the incremental version for the segmentation or geometric verification algorithms (used in the provided configuration files), please consider citing the describing paper:

```
@article{incremental2018,
  title={Incremental Segment-Based Localization in {3D} Point Clouds},
  author={Dub{\'e}, Renaud and Gollub, Mattia G and Sommer, Hannes and Gilitschenski, Igor and Siegwart, Roland and Cadena, Cesar and Nieto, Juan},
  journal={IEEE Robotics and Automation Letters},
  volume={3},
  number={3},
  pages={1832--1839},
  year={2018},
  publisher={IEEE}
}
```

## Contributing to *SegMap*

We would be very grateful if you would contribute to the code base by reporting bugs, leaving comments and proposing new features through issues and pull requests. Please see the dedicated [wiki page](https://github.com/ethz-asl/segmap/wiki/Contributing-to-SegMap) on this topic and feel free to get in touch at rdube(at)ethz(dot)ch, dugasd(at)ethz(dot)ch and crandrei(at)ethz(dot)ch. Thank you!
