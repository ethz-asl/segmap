from __future__ import print_function

import numpy as np

def asarray(x):
  if x is None: return None
  y = np.array(x)
  if len(y.shape) == 0: y = np.array([x])
  return y

class Average:
  def __init__(self, values=None, value_names=None):
    self.values = asarray(values)
    self.names = value_names
    self.name_suffix = ': '
    self.count = 0.
  def add(self, latest_values):
    latest = asarray(latest_values)
    self.count += 1.
    if self.values is None:
      self.values = latest
      if self.names is None:
        self.names = ['']*len(self.values)
        self.name_suffix = ''
      if len(self.names) != len(self.values): raise ValueError
    else:
      if latest.shape != self.values.shape: raise ValueError
      self.values = (self.values * (self.count-1.) + latest) / self.count
  def __str__(self):
    if self.count == 0.: return "Not initialized"
    return ', '.join([name+self.name_suffix+str(val) for name, val in zip(self.names, self.values)])
