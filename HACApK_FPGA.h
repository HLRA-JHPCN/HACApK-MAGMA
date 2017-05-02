typedef struct stc_HACApK_leafmtx {
  int ltmtx;
  int kt;
  int nstrtl, ndl;
  int nstrtt, ndt;
  int a1size; //
  double *a1;
  double *a2;
} stc_HACApK_leafmtx;

#if defined(HAVE_MAGMA) | defined(HAVE_MAGMA_BATCH)
#include "magma_v2.h"
#endif

typedef struct stc_HACApK_leafmtxp {
  int nd;
  int nlf;
  int nlfkt;
  int ktmax;
  int st_lf_stride; //
#if defined(HAVE_MAGMA) | defined(HAVE_MAGMA_BATCH)
  // GPU memory
  int m;         // matrix dimension
  int n;         // matrix dimension
  int gn;        // matrix dimension (global)
  int max_block; // max block size
  magmaDouble_ptr *mtx1_gpu;
  magmaDouble_ptr *mtx2_gpu;
  magmaDouble_ptr zu_gpu;
  magmaDouble_ptr *zau_gpu;
  magmaDouble_ptr *zau_pin;
  magmaDouble_ptr *zbu_gpu; 

  // for batch
  int num_batch;    // number of batch
  int total_size_y; // 
  int transA;
  double **d_A_array;
  double **d_X_array;
  double **d_Y_array;
  magma_int_t *d_M, *d_N;
  magma_int_t *d_lda;
  magma_int_t *d_inc;
  // 
  magma_int_t *batch_order;
  double **h_A_array;
  double **h_X_array;
  double **h_Y_array;
  magma_int_t *h_type;
  magma_int_t *h_I, *h_J;
  magma_int_t *h_M, *h_N;
  magma_int_t *h_lda;
  magma_int_t *max_M, *max_N;
  // streamed GEMV
  magma_int_t num_streamed;
  magma_int_t num_streamed_t;
  double **h_A_array_streamed;
  double **h_X_array_streamed;
  double **h_Y_array_streamed;
  magma_int_t *h_M_streamed, *h_N_streamed;
  magma_int_t *h_lda_streamed;
  // MPI info
  //MPI_Comm mpi_comm;
  int      mpi_rank;
#endif
  //
  stc_HACApK_leafmtx *st_lf;
} stc_HACApK_leafmtxp;
