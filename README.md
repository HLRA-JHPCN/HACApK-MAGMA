# HACApK-FPGA

1. Integrating MAGMA's batched GEMV into HACApK's matrix-vector multiply

To compile with MAGMA, uncomment and modify the MAGMA part of Makefile
> make sure to link to MAGMA (with atomic support) available in HACApK-FPGA/magma repo.

To test it, from command line, run, for example, 'mpirun -np 1 ./bem-bb-SCM.out ../bem_bb_inputs/input_100ts.pbf'
> a few inputs are available in HACApK-FPGA/bem_bb_inputs' repo.

> output shows time spent by calling batched GPU kernel
>>   time_batch: 1.14e-02 seconds
   
> output also shows time spent by calling BLAS's GEMV
>>   time_cpu: 2.94e-01 seconds

> it can also call MAGMA BLAS (not batched) by defining 'GPU' instead of 'CPU' in HACApK_FPGA.c 
 
 
2. Translating HACApK matrix into BLR storage and calling ACA/rSVD based LU factorization over PaRSEC

To compile with this option, modify the PaRSEC version of Makefile
> (some source files have not been commited, yet :).
