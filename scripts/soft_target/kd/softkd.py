# !/usr/bin/python
# -*- coding: utf-8 -*-
import os, math, sys, inspect, argparse, shutil, lmdb, math
import numpy as np
from util import *

class SoftKnowledgeDistiller:
  def __init__(self, teacher_proto, teacher_model):
    # init member
    self.teacher_proto = teacher_proto
    self.teacher_model = teacher_model
    self.topk = 0
    self.nsorter = 3
    # import caffe
    caffe_dir = os.path.abspath("../../..");
    self.caffe = import_caffe(caffe_dir)
    # intermediary

  def setParam(self, *args, **kwds):
    for (k,v) in kwds.items():
      setattr(self, k, v)
    if hasattr(self, "gpu_id"):
      self.caffe.set_device(self.gpu_id)

  def distill(self, supervise_layer_name, lmdb):
    if os.path.exists(lmdb):
      shutil.rmtree(lmdb)    
    print "extract soft target from teacher ..."
    feature_to_lmdb(self.caffe, self.teacher_proto, self.teacher_model, \
                    supervise_layer_name, lmdb, \
                    self.nsorter, self.topk)
