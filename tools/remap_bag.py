#!/usr/bin/python

import rospy
import rosbag
import os
import sys
import argparse
import yaml

def remove_tf(inbag,outbag,prefix):
  print '   Processing input bagfile: ', inbag
  print '  Writing to output bagfile: ', outbag
  print '              Adding prefix: ', prefix

  outbag = rosbag.Bag(outbag,'w')
  for topic, msg, t in rosbag.Bag(inbag,'r').read_messages():
      if topic == "/tf":
          new_transforms = []
          for transform in msg.transforms:
	    
	    if transform.header.frame_id[0] == '/':
	      transform.header.frame_id = prefix + transform.header.frame_id
	    else:
              transform.header.frame_id = prefix + '/' + transform.header.frame_id
	    
	    if transform.child_frame_id[0] == '/':
	      transform.child_frame_id = prefix + transform.child_frame_id
	    else:
              transform.child_frame_id = prefix + '/' + transform.child_frame_id
	    
            new_transforms.append(transform)
          msg.transforms = new_transforms
      else:
	try: 
	  if msg.header.frame_id[0] == '/':
	    msg.header.frame_id = prefix + msg.header.frame_id
	  else:
	    msg.header.frame_id = prefix + '/' + msg.header.frame_id
	except:
	  pass
	
	if topic[0] == '/':
	  topic = prefix + topic
	else:
	  topic = prefix + '/' + topic
      outbag.write(topic, msg, t)
  print 'Closing output bagfile and exit...'
  outbag.close();

if __name__ == "__main__":

  parser = argparse.ArgumentParser(
      description='removes all transforms from the /tf topic that contain one of the given frame_ids in the header as parent or child.')
  parser.add_argument('-i', metavar='INPUT_BAGFILE', required=True, help='input bagfile')
  parser.add_argument('-o', metavar='OUTPUT_BAGFILE', required=True, help='output bagfile')
  parser.add_argument('-p', metavar='PREFIX', required=True, help='prefix to add to the frame ids')
  args = parser.parse_args()

  try:
    remove_tf(args.i,args.o,args.p)
  except Exception, e:
    import traceback
traceback.print_exc()