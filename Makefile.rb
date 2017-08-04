# -*- Makefile -*-
# Makefile for Reedbush-H
OPTFLAGS = -qopenmp -O3 -xCORE-AVX2
CC=mpicc
F90=mpif90
CCFLAGS = $(OPTFLAGS)
F90FLAGS = $(OPTFLAGS) -fpp -align array64byte
LDFLAGS = -mkl -cxxlib

LINK=$(F90)

OBJS = HACApK_MAGMA.o HACApK_MGPU.o HACApK_BATCH.o HACApK_BATCH_SORT.o HACApK_SOLVER.o \
       HACApK_lib.o m_ppohBEM_user_func.o m_ppohBEM_matrix_element_ij.o m_HACApK_calc_entry_ij.o \
       m_HACApK_base.o m_HACApK_solve.o m_HACApK_use.o m_ppohBEM_bembb2hacapk.o bem-bb-fw-HACApK-0.4.2.o


# MAGMA
CUDA_DIR  = /lustre/app/acc/cuda/8.0.44
MAGMA_DIR = /lustre/gc26f/c26002/work/magma-2.2.0_intel17.0.4.196
#MPI_DIR   = /usr/apps.sp3/isv/intel/ParallelStudioXE/ClusterEdition/2016-Update3/compilers_and_libraries_2016.3.210/linux/mpi/intel64

INCS+= -I$(CUDA_DIR)/include -I$(MAGMA_DIR)/include -I$(MPI_DIR)/include

LIBS+= -L$(CUDA_DIR)/lib64 -lcublas -lcusparse -lcudart $(MAGMA_DIR)/lib/libmagma.a

#CCFLAGS += -DISO_C_BINDING
#F90FLAGS+= -DISO_C_BINDING

CCFLAGS += -DHAVE_MAGMA -DADD_ -DMAGMA_WITH_MKL
F90FLAGS+= -DHAVE_MAGMA -DADD_ -DMAGMA_WITH_MKL

CCFLAGS += -DHAVE_MAGMA_BATCH
F90FLAGS+= -DHAVE_MAGMA_BATCH

CCFLAGS += -DPROF_MAGMA_BATCH
F90FLAGS+= -DPROF_MAGMA_BATCH

CCFLAGS += -DBICG_MAGMA_BATCH
F90FLAGS+= -DBICG_MAGMA_BATCH

CCFLAGS += -mkl=sequential
F90FLAGS+= -mkl=sequential

# runs only MGPU version
#CCFLAGS += -DBICG_MAGMA_MGPU
#F90FLAGS+= -DBICG_MAGMA_MGPU

OPTFLAGS += -O3
#CCFLAGS += -g
#F90FLAGS+= -g
#LDFLAGS += -g
#CCFLAGS += -Wall -Wremarks -Wcheck


TARGET=bem-bb-SCM.out

.SUFFIXES: .o .c .f90

$(TARGET): $(OBJS)
			$(LINK) -o $@ $(OBJS) $(LDFLAGS) $(LIBS)

.c.o: *.c
			$(CC) -c $(CCFLAGS) $(INCS) $<
.f90.o: *.f90
			$(F90) -c $< $(F90FLAGS) $(INCS)
clean:
	rm -f *.o *.mod $(TARGET)

rmod:
	rm -f m_*.o *.mod

