#!/usr/bin/python
"""
Copyright (c) 2012,
Systems, Robotics and Vision Group
University of the Balearican Islands
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Systems, Robotics and Vision Group, University of
      the Balearican Islands nor the names of its contributors may be used to
      endorse or promote products derived from this software without specific
      prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

PKG = 'bag_tools' # this package name

import roslib; roslib.load_manifest(PKG)
import rospy
import rosbag
import os
import sys
import argparse

def cut(inbag, outbagfile, start, duration):
  start_time = rospy.Time.from_sec(999999999999)

  rospy.loginfo('   Looking for smallest time in: %s', inbag)
  for topic, msg, t in rosbag.Bag(inbag,'r').read_messages():
    if t < start_time:
      start_time = t
    break
  rospy.loginfo('   Bagfile starts at %s', start_time)
  start_time = start_time + rospy.Duration.from_sec(start)
  end_time = start_time + rospy.Duration.from_sec(duration)
  rospy.loginfo('   Cutting out from %s to %s',start_time, end_time)
  outbag = rosbag.Bag(outbagfile, 'w')
  num_messages = 0
  #rospy.loginfo('   Extracting messages from:', inbag)
  for topic, msg, t in rosbag.Bag(inbag,'r').read_messages(start_time=start_time, end_time=end_time):
    outbag.write(topic, msg, t)
    num_messages = num_messages + 1
  outbag.close()
  rospy.loginfo('     New output bagfile has %s messages', num_messages)


if __name__ == "__main__":
  rospy.init_node('cut')
  parser = argparse.ArgumentParser(
      description='Cuts out a section from an input bagfile and writes it to an output bagfile')
  parser.add_argument('--inbag', help='input bagfile(s)', required=True)
  parser.add_argument('--outbag', help='output bagfile', required=True)
  parser.add_argument('--start', help='start time', type=float, required=True)
  parser.add_argument('--duration', help='duration of the resulting part', type=float, required=True)
  args = parser.parse_args()
  try:
    cut(args.inbag, args.outbag, args.start, args.duration)
  except Exception, e:
    import traceback
traceback.print_exc()