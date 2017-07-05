#SYSTEM = FX10
#SYSTEM = INTEL
#SYSTEM = XC30
SYSTEM = Tsubame

#FX10
ifeq ($(SYSTEM),FX10)
OPTFLAGS = -fs
CC=mpifccpx
F90=mpifrtpx -Kfast,openmp
#F90=mpifrtpx -Kopenmp
CCFLAGS = $(OPTFLAGS)
F90FLAGS = $(OPTFLAGS) -Cfpp
LDFLAGS = -SSL2
endif

#intel
ifeq ($(SYSTEM),INTEL)
#OPTFLAGS = -O3 -traceback -ip -heap-arrays -qopenmp
OPTFLAGS = -qopenmp -O3 -ip
CC=mpiicc
F90=mpiifort
CCFLAGS = $(OPTFLAGS)
#F90FLAGS = $(OPTFLAGS) -fpp -assume nounderscore -names uppercase
F90FLAGS = $(OPTFLAGS) -fpp
#F90FLAGS = $(OPTFLAGS) -fpp -check all
#F90FLAGS = -fpe0 -traceback -g -CB -assume nounderscore -names lowercase -fpp -check all
#LDFLAGS = -mkl -trace
LDFLAGS = -mkl
endif

#XC30
ifeq ($(SYSTEM),XC30)
OPTFLAGS = -O2 -homp
CC=cc
F90=ftn
CCFLAGS = $(OPTFLAGS)
F90FLAGS = $(OPTFLAGS)
endif

#Tsubame
SYSTEM = TSUBAME
ifeq ($(SYSTEM),TSUBAME)
#OPTFLAGS = -O3 -traceback -ip -heap-arrays -qopenmp
#OPTFLAGS = -openmp -O3
OPTFLAGS = -qopenmp
#CC=mpicc
#F90=mpif90
CC=mpiicc
F90=mpiifort
CCFLAGS = $(OPTFLAGS)
F90FLAGS = $(OPTFLAGS) -fpp
#F90FLAGS = $(OPTFLAGS) -fpp1 -check all
LDFLAGS = -mkl
#INCS+= -I/usr/apps.sp3/isv/intel/ParallelStudioXE/ClusterEdition/2016-Update3/compilers_and_libraries_2016.3.210/linux/compiler/include
endif

LINK=$(F90)

OBJS = HACApK_MAGMA.o HACApK_MGPU.o HACApK_BATCH.o HACApK_BATCH_SORT.o HACApK_SOLVER.o \
       HACApK_lib.o m_ppohBEM_user_func.o m_ppohBEM_matrix_element_ij.o m_HACApK_calc_entry_ij.o \
       m_HACApK_base.o m_HACApK_solve.o m_HACApK_use.o m_ppohBEM_bembb2hacapk.o bem-bb-fw-HACApK-0.4.2.o \

# PaRSEC
#CPP = mpiicpc

# forcing the link with OpenMPI
#LINK = ifort
#MPI_DIR = /opt/ompi/2.0.1
#INCS = -I$(MPI_DIR)/include
#LIBS = -cxxlib
#LIBS+= $(MPI_DIR)/lib/libmpi_mpifh.so

#PARSEC_DIR = /home/yamazaki/parsec-bitbucket/dplasma
#SOLVER_DIR = /home/yamazaki/pulsar/ierus/main/parsec
#IERUS_DIR  = /home/yamazaki/pulsar/ierus/ierus_source/parsec

#OBJS+= HACApK_PaRSEC.o 
#LIBS+= $(IERUS_DIR)/ACA.o $(IERUS_DIR)/DataFile.o $(IERUS_DIR)/rsvd.o $(IERUS_DIR)/symbolic.o \
#       $(SOLVER_DIR)/lib/*.o $(SOLVER_DIR)/lib/ierus/*.o \
#       $(PARSEC_DIR)/libdague.a $(PARSEC_DIR)/data_dist/matrix/libdague_distribution_matrix.a \
#       -lpthread -lm -lhwloc

#INCS+= -I$(IERUS_DIR) -I$(SOLVER_DIR)/include  \
#       -I$(PARSEC_DIR) -I$(PARSEC_DIR)/include

#CCFLAGS += -DHAVE_PaRSEC
#F90FLAGS+= -DHAVE_PaRSEC


# MAGMA
CUDA_DIR  = /usr/apps.sp3/cuda/7.5
MAGMA_DIR = /home/usr2/16IH0462/magma/release/magma-2.2.0
MPI_DIR   = /usr/apps.sp3/isv/intel/ParallelStudioXE/ClusterEdition/2016-Update3/compilers_and_libraries_2016.3.210/linux/mpi/intel64

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
CCFLAGS += -DBICG_MAGMA_MGPU
F90FLAGS+= -DBICG_MAGMA_MGPU

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

