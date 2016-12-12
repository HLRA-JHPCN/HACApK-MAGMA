#SYSTEM = FX10
SYSTEM = INTEL
#SYSTEM = XC30

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
#OPTFLAGS = -O3 -traceback -ip -heap-arrays -openmp
OPTFLAGS = -openmp -O3 -ip
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

LINK=$(F90)

#CC  = icc
#F90 = ifort
LINK = icpc
LINK = ifort
CPP  = mpiicpc
#LINK = mpiifort

OBJS = HACApK_FPGA.o \
       m_bem-bb-fw-coordinate.o HACApK_lib.o bem-bb-template-SCM-0.4.1.o m_HACApK_calc_entry_ij.o \
       m_HACApK_base.o m_HACApK_solve.o m_HACApK_use.o m_ppohBEM_bembb2hacapk.o bem-bb-fw-HACApK-0.4.1.o \

INCS =
LIBS = -cxxlib 
#LIBS+= -lifcore

#MPI
MPI_DIR    = /opt/ompi/2.0.1

INCS+= -I$(MPI_DIR)/include 
LIBS+= $(MPI_DIR)/lib/libmpi_mpifh.so

# PaRSEC
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
CUDA_DIR  = /opt/cuda/7.5
MAGMA_DIR = /home/yamazaki/magma/bitbuckets/magma

INCS+= -I$(CUDA_DIR)/include -I$(MAGMA_DIR)/include

LIBS+= -L$(CUDA_DIR)/lib64 -lcublas -lcusparse -lcudart $(MAGMA_DIR)/lib/libmagma.a

CCFLAGS += -DHAVE_MAGMA 
F90FLAGS+= -DHAVE_MAGMA

CCFLAGS += -DHAVE_MAGMA_BATCH
F90FLAGS+= -DHAVE_MAGMA_BATCH


TARGET=bem-bb-SCM.out
.SUFFIXES: .o .c .f90

$(TARGET): $(OBJS)
			$(LINK) -o $@ $(OBJS) $(LDFLAGS) $(LIBS)

.c.o: *.c
			$(CC) -g -c $(CCFLAGS) $(INCS) $<
.cpp.o: *.c
			$(CPP) -g -c $(CCFLAGS) $(INCS) $<
.f90.o: *.f90
#			echo 'f90 complile'
			$(F90) -g -c $< $(F90FLAGS) $(INCS)
clean:
	rm -f *.o *.mod $(TARGET)

rmod:
	rm -f m_*.o *.mod

