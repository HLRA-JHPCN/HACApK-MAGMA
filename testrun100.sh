#!/bin/sh

#PBS -q h-regular
#PBS -l select=1:mpiprocs=1
#PBS -W group_list=gi75
#PBS -l walltime=2:00:00

cd ${PBS_O_WORKDIR}
module purge
module load intel/16.0.4.258 openmpi-gdr/2.1.1/intel cuda
ulimit -s 1000000
export KMP_AFFINITY=granularity=fine,compact,verbose
export OMP_NUM_THREADS=18
export MKL_NUM_THREADS=1
env
date
echo "======== ======== ======== ======== ======== ======== ======== ========"
mpirun -np 1 numactl --localalloc --cpunodebind=0 ./bem-bb-SCM.out ./input_100ts.pbf 2>&1 | tee log_100ts.txt
echo "======== ======== ======== ======== ======== ======== ======== ========"
date
