## *SegMatch*

*SegMatch* is a reliable loop-closure detection algorithm based on the matching of 3D segments. The *SegMatch* code is open-source (BSD License) and has been tested under Ubuntu 14.04 and ROS Indigo. Please remember that this is on-going research code which is subject to changes in the future. Several new features and demonstrations will be added soon.

Please consult our [paper](https://arxiv.org/pdf/1609.07720v1.pdf), [video](https://www.youtube.com/watch?v=iddCgYbgpjE) and [wiki](https://github.com/ethz-asl/segmatch/wiki) for the algorithm description and for instructions on running demonstrations.

### Compiling *SegMatch*
The following configuration was tested under Ubuntu 14.04 and ROS indigo. Please see the final note if you want to compile under ROS Kinetic.

First install the required system packages:
```
$ sudo apt-get install libopencv-dev python-wstool doxygen
```
Then use wstool for installing catkin dependencies:
```
$ cd ~/catkin_ws/src
$ wstool init
$ wstool merge segmatch/dependencies.rosinstall
$ wstool update
```
Finally build the *laser_mapper* package which will compile all *SegMatch* modules:
```
$ cd ~/catkin_ws
$ catkin build -DCMAKE_BUILD_TYPE=Release laser_mapper
```
Building dependencies will require some time according to which new package need to be built (eg. Building the `pcl_catkin` package can take up to two hours). See this link for installing [catkin_tools](http://catkin-tools.readthedocs.io/en/latest/installing.html). 

Consult the [wiki](https://github.com/ethz-asl/segmatch/wiki) for instructions on running the demonstrations.

Note: If you are using ROS Kinetic, you might want to run the following command in your catkin workspace prior to building the packages:
```
$ catkin config --merge-devel
```

### Contributing to *SegMatch*

We would be very grateful if you would contribute to the code base by reporting bugs, leaving comments and proposing new features through issues and pull requests. Please see the dedicated [wiki page](https://github.com/ethz-asl/segmatch/wiki/Contributing-to-SegMatch) on this topic and feel free to get in touch at rdube(at)ethz.ch. Thank you!

### Citing *SegMatch*

Thank you for citing our *SegMatch* paper if you use any of this code: 
```
@article{dube2016segmatch,
  title={SegMatch: Segment based loop-closure for 3D point clouds},
  author={Dub{\'e}, Renaud and Dugas, Daniel and Stumm, Elena and Nieto, Juan and Siegwart, Roland and Cadena, Cesar},
  journal={arXiv preprint arXiv:1609.07720},
  year={2016}
}
```
