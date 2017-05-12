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

typedef struct stc_HACApK_lcontrol {
  int lf_umpi;
  int lpmd_offset;
  int lod_offset;
  int lsp_offset;
  int lnp_offset; 
  int lthr_offset;
  double *param;       // 100
  //magma_int_t *lpmd; // 30
  //magma_int_t *lod;  // nd
  //magma_int_t *lsp;  // lpmd[1]; //nrank
  //magma_int_t *lnp;  // lpmd[1]; //nrank
  //magma_int_t *lthr;
} stc_HACApK_lcontrol;


void c_hacapk_adot_body_lfcpy_batch_sorted_(int *nd, stc_HACApK_leafmtxp *st_leafmtxp);
void c_hacapk_adot_body_lfmtx_batch_queue(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu,
                                          double *time_batch, double *time_set, double *time_copy,
                                          int on_gpu, magma_queue_t queue);
void c_hacapk_adot_body_lfmtx_batch_(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu,
                                     double *time_batch, double *time_set, double *time_copy);
