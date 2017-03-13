# !/usr/bin/python
# -*- coding: utf-8 -*-
import os, sys, inspect, argparse, lmdb
import numpy as np

def merge_lmdb(srcs, dst):
  lmdb_wd = lmdb.open(dst, map_size=1e12)
  lmdb_wtxn = lmdb_wd.begin(write=True)
  lmdb_wcursor = lmdb_wtxn.cursor()
  commit_size = 50000
  total_num = 0
  for src in srcs:
    print "merging %s, " % src,
    lmdb_rd = lmdb.open(src, map_size=1e12)
    lmdb_rtxn = lmdb_rd.begin(write=False)
    lmdb_rcursor = lmdb_rtxn.cursor()
    offset = 0
    for (k, v) in lmdb_rcursor:
      lmdb_wcursor.put("{:0>10d}".format(total_num), v)
      total_num += 1
      if (offset + 1) % commit_size == 0:
        lmdb_wtxn.commit()
        lmdb_wtxn = lmdb_wd.begin(write=True)
        lmdb_wcursor = lmdb_wtxn.cursor()
      offset += 1
    if offset % commit_size != 0:
      lmdb_wtxn.commit()
      lmdb_wtxn = lmdb_wd.begin(write=True)
      lmdb_wcursor = lmdb_wtxn.cursor()
    print "%d data instances" % offset
  print "merged %d data instances" % total_num

if __name__=="__main__":
  parser = argparse.ArgumentParser(description='merge mutiple lmdbs')
  parser.add_argument("-srcs", help="pathes to src lmdb, quote them", required=True)
  parser.add_argument("-dst", help="path to dst lmdb", required=True)

  args = parser.parse_args()
  srcs = [src.strip() for src in args.srcs.split(' ')]
  dst = args.dst
  merge_lmdb(srcs, dst)
