# !/usr/bin/python
# -*- coding: utf-8 -*-
import os, sys, inspect, argparse, lmdb
pfolder = os.path.realpath(
  os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if pfolder not in sys.path:
  sys.path.insert(0, pfolder)
reload(sys)

import numpy as np
from kd.softkd import SoftKnowledgeDistiller

if __name__=="__main__":
  parser = argparse.ArgumentParser(description='merge mutiple lmdbs')
  parser.add_argument("-teacher_proto", help="teacher proto", required=True)
  parser.add_argument("-teacher_model", help="teacher caffemodel", required=True)
  parser.add_argument("-topk", help="if <=0, dense target else sparse target", \
                      type=int, required=True)
  parser.add_argument("-device_id", help="GPU ID", type=int, required=True)
  parser.add_argument("-softmax_name", help="name of softmax layer", required=True)
  parser.add_argument("-lmdb", help="path to lmdb", required=True)
  args = parser.parse_args()

  kder = SoftKnowledgeDistiller(args.teacher_proto, args.teacher_model)
  kder.setParam(topk=args.topk, gpu_id=args.device_id, nsorter=3)
  kder.distill(args.softmax_name, args.lmdb)

