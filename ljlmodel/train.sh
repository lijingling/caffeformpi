#!/usr/bin/env sh

SOLVER=/ssd/lijingling/resnet_MSRA/caffe-mpi/MSRA/solver.prototxt
STAGE=/ssd/lijingling/resnet_MSRA/caffe-mpi/MSRA/snapshot_iter_1250000.solverstate
mpirun --mca mpi_paffinity_alone 1 -np 2 ../build/tools/caffe train \
	--solver=$SOLVER \
	2>&1 | tee log/train_wgs.log
