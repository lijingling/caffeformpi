# !/usr/bin/python
# -*- coding: utf-8 -*-
import os, sys, cv2, lmdb, math, time, threading
from Queue import Queue
from google.protobuf import text_format
import numpy as np
global TRAIN
global TEST
global LMDB
TRAIN = 0
TEST = 1
LMDB = 1
CAFFE = 1

def enum_t(**enums):
    return type('Enum', (), enums)

def import_caffe(caffe_root):
   caffe_dir = os.path.join(caffe_root, 'python')
   if caffe_dir not in sys.path:
       sys.path.insert(0, caffe_dir)
   # import caffe
   import caffe
   caffe.set_mode_gpu()
   return caffe

def parse_net_proto(caffe, proto_fn):
   fd = open(proto_fn, "r")
   net_proto = caffe.proto.caffe_pb2.NetParameter()
   text_format.Merge(str(fd.read()), net_proto)
   fd.close()
   return net_proto

def parse_datalayer_proto(net_proto, phase):
  if phase == "TRAIN":
    phase = 0
  elif phase == "TEST":
    phase = 1
  else:
    raise Exception("invalid phase [%s]" % phase)
  datalayer = None
  for i in xrange(len(net_proto.layer)):
    for top in net_proto.layer[i].top:
      if top == 'data':
        for include_attr in net_proto.layer[i].include:
          if include_attr.phase == phase:
            datalayer = net_proto.layer[i]
            return datalayer
  return datalayer

def parse_dataparam_proto(datalayer_proto):
  LMDB = 1
  dataparam_proto = None
  t = datalayer_proto.type.lower()
  if t == 'data' and datalayer_proto.data_param.backend == LMDB:
    dataparam_proto = datalayer_proto.data_param
  elif t == 'imagedata':
    dataparam_proto = datalayer_proto.image_data_param
  else:
    raise Exception("unsupported data type [%s]" % t)
  return (t, dataparam_proto)

def get_lmdb_data_num(caffe, source):
  lmdb_env = lmdb.open(source)
  lmdb_txn = lmdb_env.begin()
  lmdb_cursor = lmdb_txn.cursor()

  data_num = 0
  for key, value in lmdb_cursor:
    data_num += 1
  return data_num

def get_image_data_num(source):
  data_num = 0
  for line in open(source, 'r'):
    data_num += 1
  return data_num

def get_data_num(caffe, datalayer_type, datasource):
  if datalayer_type == 'data':
    return get_lmdb_data_num(caffe, datasource)
  elif datalayer_type == 'imagedata':
    return get_image_data_num(datasource)
  else:
    raise Exception("unsupported data type [%s]" % datalayer_type)

def batch_to_lmdb(caffe, batch, lmdb_txn, offset, sorter, topk):
  argsort = None
  if topk > 0:
    sorter.sort(batch, topk)
    argsort = sorter.get_result(batch.shape[0])
  for idx in range(batch.shape[0]):
    if argsort is not None:
      sparse_data = np.zeros((2, 1, topk), dtype=np.float)
      local_data = batch[idx, :]
      sparse_data[0, :, :] = argsort[idx]
      sparse_data[1, :, :] = local_data[(sparse_data[0, :, :]).astype(np.int)].\
        reshape((1, topk))
      datum = caffe.io.array_to_datum(sparse_data)
    else:
      datum = caffe.io.array_to_datum(batch[np.newaxis, np.newaxis, idx])
    lmdb_txn.put("{:0>10d}".format(offset + idx), datum.SerializeToString())

def feature_to_lmdb(caffe, proto_fn, model_fn, \
    supervise_layer_name, lmdb_fn, \
    nworker=3, topk=0):
  proto = parse_net_proto(caffe, proto_fn)
  datalayer_proto = parse_datalayer_proto(proto, "TEST")
  (datalayer_type, dataparam_proto) = parse_dataparam_proto(datalayer_proto)
  datasource = dataparam_proto.source
  batch_size = dataparam_proto.batch_size
  data_num =  get_data_num(caffe, datalayer_type, datasource)

  net = caffe.Net(proto_fn, model_fn, caffe.TEST)
  iter_num = int(math.ceil(1.0 * data_num / batch_size))
  sorter = TopKSorter(nworker) if topk > 0 else None
  lmdb_fd = lmdb.open(lmdb_fn, map_size=1e12)
  commit_size = 1000
  offset = 0
  lmdb_txn = lmdb_fd.begin(write=True)
  for i in xrange(iter_num):
    t1 = time.time()
    net.forward()
    batch_num = min(batch_size, data_num - i * batch_size)
    batch = net.blobs[supervise_layer_name].data.astype('float')[:batch_num]
    batch = batch.reshape(batch.shape[0], batch.shape[1])
    batch_to_lmdb(caffe, batch, lmdb_txn, offset, sorter, topk)
    offset += batch.shape[0]
    t2 = time.time()
    print "\rprogress %.2f%%(%d/%d), %.fms/batch. remain %.2fhours" % \
     (100.0 * offset / data_num, offset, data_num, (t2 - t1) * 1000.0,
     (t2 - t1) * (iter_num - i - 1) / 60 / 60),
    sys.stdout.flush()
    if (i + 1) % commit_size == 0:
      lmdb_txn.commit()
      lmdb_txn = lmdb_fd.begin(write=True)
  if iter_num % batch_size != 0:
    lmdb_txn.commit()
  print
  if sorter is not None:
    sorter.finish()

class TopKSorter:
  class InnerSortter(threading.Thread):
    def __init__(self, batch_q, result_q):
      threading.Thread.__init__(self, name="inner_sorter")
      self.batch_q = batch_q
      self.result_q = result_q
      self._stop = threading.Event()

    def run(self):
      while True:
        (idx, vec, topk) = self.batch_q.get()
        argsort = np.argpartition(vec, -topk)[-topk:]
        self.result_q.put((idx, argsort))

    def stop(self):
      self._stop.set()

  def __init__(self, nworker):
    self.batch_q = Queue()
    self.result_q = Queue()
    self.sorters = [None] * nworker
    for i in xrange(nworker):
      sorter = TopKSorter.InnerSortter(self.batch_q, self.result_q)
      self.sorters[i] = sorter
      sorter.start()

  def sort(self, batch, topk):
    for idx in xrange(batch.shape[0]):
      self.batch_q.put((idx, batch[idx, :], topk))

  def get_result(self, batch_size):
    results = [None] * batch_size
    for i in xrange(batch_size):
      (idx, argsort) = self.result_q.get()
      results[idx] = argsort
    return results

  def finish(self):
    for sorter in self.sorters:
      sorter.stop()
