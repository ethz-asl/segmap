#!/usr/bin/python

import rosbag
from tf.msg import tfMessage
import rospy
import os
import sys
import argparse
import yaml

def remove_tf(inbagfile, outbagfile, frame):
    print '   Processing input bagfile: ', inbagfile
    print '  Writing to output bagfile: ', outbagfile
    print '             Removing frame: ', frame
           
    with rosbag.Bag(outbagfile, 'w') as outbag:
        for topic, msg, t in rosbag.Bag(inbagfile).read_messages():
            if topic == "/tf" and msg.transforms:
                filteredTransforms = [];
                for m in msg.transforms:
                    if m.header.frame_id != frame and m.child_frame_id != frame:
                        filteredTransforms.append(m)
                if len(filteredTransforms)>0:
                    msg.transforms = filteredTransforms
                    outbag.write(topic, msg, t)
            else:
                outbag.write(topic, msg, t)
    print 'Closing output bagfile and exit...'
    outbag.close();
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Removes all transforms from the /tf topic that contains the given frame_id in the header as parent or child.')
    parser.add_argument('-i', metavar='INPUT_BAGFILE', required=True, help='input bagfile')
    parser.add_argument('-o', metavar='OUTPUT_BAGFILE', required=True, help='output bagfile')
    parser.add_argument('-f', metavar='FRAME', required=True, help='frame to remove')
    args = parser.parse_args()
    try:
        remove_tf(args.i,args.o,args.f)
    except Exception, e:
        import traceback
        traceback.print_exc()
