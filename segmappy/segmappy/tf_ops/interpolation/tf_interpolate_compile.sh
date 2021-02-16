#!/bin/bash

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so \
    -shared -fPIC -I$TF_INC -I$TF_INC/external/nsync/public \
    -I/usr/local/cuda-10.0/include \
    -lcudart -L /usr/local/cuda-10.0/lib64/ \
    -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
