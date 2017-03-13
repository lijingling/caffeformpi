# !/usr/bin/python
# -*- coding: utf-8 -*-
import os, sys, inspect, argparse, re
import subprocess, time, datetime, signal
from threading import Thread

kWaitBeforeLauch = 10 # sec
kMaxSinceLastModify = 5 * 60 # sec
kMonitorCicle = 10 #sec

class TError:
  DONE = 1
  NO_LOG = 2
  HANG = 3
  DEAD = 4

def shell_command(cmd_str, log):
  with open(log, 'w') as log_fd:
    proc = subprocess.Popen([cmd_str], shell=True, \
         stdout=log_fd, stderr=log_fd)
    return proc

def terminate_command(proc):
  if proc.poll() is None:
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM) 

def get_last_snapshot(log_data):
  r = re.compile(r"^.*Snapshotting solver state to binary proto file .*$", re.MULTILINE)
  matches = r.findall(log_data)
  if len(matches) > 0:
    return matches[-1].split()[-1]
  return None

def optimaize_done(log_data):
  r = re.compile(r"^.*Optimization Done.$", re.MULTILINE)  
  matches = r.findall(log_data)
  return len(matches) == 2
 
def log_changed(proc, log, timeout, errcode):
  if not os.path.isfile(log):
    errcode[0] = TError.NO_LOG
    return
  while proc.poll() is None:
    if not os.path.isfile(log):
      errcode[0] = TError.NO_LOG
      return
    last_modify = os.stat(log).st_mtime
    sec_since_last_modify = time.time() - last_modify
    if time.time() - last_modify > timeout:
      errcode[0] = TError.HANG
      return
    time.sleep(kMonitorCicle)
  errcode[0] = TError.DONE if optimaize_done(open(log).read()) else TError.DEAD

def caffe_train(caffe, n, solver, log, weights, snapshot, timeout):
  if os.path.isfile(log):
    os.remove(log) 
 
  cmd_str = ""
  mca_opts = "--mca mpi_paffinity_alone 1"
  if n > 1:   
    cmd_str += "mpirun %s -np %d " % (mca_opts, n)
  cmd_str += caffe + " train"
  cmd_str += " --solver=" + solver
  if weights is not None:
    cmd_str += " --weights=" + weights
  if snapshot is not None:
    cmd_str += " --snapshot=" + snapshot
  print cmd_str 
  sys.stdout.flush()

  proc = shell_command(cmd_str, log)
  time.sleep(kWaitBeforeLauch)
  errcode = [-1]
  monitor_thread = Thread(target=log_changed, args=(proc, log, timeout, errcode,))
  monitor_thread.start()
  monitor_thread.join()
  return (errcode[0], proc)
    
if __name__=="__main__":
  parser = argparse.ArgumentParser(description='Caffe train script. Let Caffe survive from hang/dead automatically')
  parser.add_argument("-caffe", help="path to caffe binary", required=True)
  parser.add_argument("-n", type=int, help="number of parallel", required=True)
  parser.add_argument("-solver", help="path to solver", required=True)
  parser.add_argument("-log_prefix", help="prefix of log file", required=True)
  parser.add_argument("-weights", help="path to weights", required=False)
  parser.add_argument("-snapshot", help="path to snapshot", required=False)
  parser.add_argument("-survive_from_hang", help="whether to survive from hang", \
                      type=bool, required=False, default=True)
  parser.add_argument("-survive_from_dead", help="whether to survive from dead", \
                      type=bool, required=False, default=True)
  parser.add_argument("-timeout", help="timeout for hang, in seconds", \
                      type=int, required=False, default=kMaxSinceLastModify)
  args = parser.parse_args()
  
  log_prefix = args.log_prefix
  caffe = args.caffe
  if not os.path.isfile(caffe):
    print "caffe[%s] binary not found" % caffe
    exit(0)
  n = args.n 
  if n <= 0:
    print "number of parallel[%d] must be larger than 0" % n 
  solver = args.solver
  if not os.path.isfile(solver):
    print "sovler[%s] not found" % solver
    exit(0)
  weights = args.weights
  if weights is not None and not os.path.isfile(weigths):
    print "weights[%s] not found" % weights
    exit(0)
  snapshot = args.snapshot
  if snapshot is not None and not os.path.isfile(snapshot):
    print "snapshot[%s] not found" % snapshot
    exit(0)
  survive_from_hang = args.survive_from_hang
  survive_from_dead = args.survive_from_dead
  timeout = args.timeout
  
  os.setpgrp()
  print "PID = %d" % os.getpid()
  log_idx = 0
  while True:
    print "-----------%d-------------" % (log_idx + 1)
    sys.stdout.flush()
    log = "%s%d" % (log_prefix, log_idx)
    (errcode, proc) = \
      caffe_train(caffe, n, solver, log, weights, snapshot, timeout)  
    if errcode == TError.DONE:
      print "Optimize done."
      exit(0)
    elif errcode == TError.NO_LOG:
      print "Odds enough! There is no log from your app."
      exit(-1) 
    elif errcode == TError.HANG:
      print "Your app has hang at least %d seconds." % kMaxSinceLastModify
      if survive_from_hang:
        print "Kill app"
        terminate_command(proc)
      else:
        exit(0)
    elif errcode == TError.DEAD:
      print "Your app has dead."
      if not survive_from_dead:
        exit(0)
    else:
      print "Unknown error."
      exit(0)

    snapshot = get_last_snapshot(open(log).read())
    if snapshot is None:
      print "Snapshot not found. It unable to keep going."
      exit(0)
    weights = None 
    log_idx += 1
    print "Ready to Keep going with snapshot[%s]" % snapshot
    sys.stdout.flush()
