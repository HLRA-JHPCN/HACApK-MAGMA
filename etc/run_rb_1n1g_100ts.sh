#!/bin/sh

#PBS -q h-short
#PBS -l select=1:mpiprocs=1
#PBS -W group_list=gi75
#PBS -l walltime=10:00

cd ${PBS_O_WORKDIR}
. /etc/profile.d/modules.sh
module load intel/17.0.2.174 openmpi-gdr/2.1.1/intel cuda/8.0.44
ulimit -s 1000000
export OMP_NUM_THREADS=1
env
date
echo "======== ======== ======== ======== ======== ======== ======== ========"
mpirun -np 1 ./bem-bb-SCM.out ./input_100ts.pbf 2>&1 | tee log_100ts.txt
echo "======== ======== ======== ======== ======== ======== ======== ========"
date
