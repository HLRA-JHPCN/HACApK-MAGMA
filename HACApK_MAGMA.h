typedef struct stc_HACApK_leafmtx {
  int ltmtx;
  int kt;
  int nstrtl, ndl;
  int nstrtt, ndt;
  long long int a1size; //
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
  // multi-GPU support
  double ***d_A_mgpu;
  double ***d_X_mgpu;
  double ***d_Y_mgpu;
  magma_int_t **d_M_mgpu, **d_N_mgpu;
  magma_int_t **d_lda_mgpu;
  magma_int_t **d_inc_mgpu;
  magmaDouble_ptr *zu_mgpu;
  magmaDouble_ptr *zau_mgpu;
  magmaDouble_ptr *zbu_mgpu; 
  double ***h_A_mgpu;
  double ***h_X_mgpu;
  double ***h_Y_mgpu;
  magma_int_t **h_type_mgpu;
  magma_int_t **h_I_mgpu, **h_J_mgpu;
  magma_int_t **h_M_mgpu, **h_N_mgpu;
  magma_int_t **h_lda_mgpu;
  magma_int_t **max_M_mgpu, **max_N_mgpu;
  magma_int_t *nlf_mgpu;
  magma_int_t *num_batch_mgpu;
  magma_int_t *total_size_y_mgpu;
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


#if defined(HAVE_MAGMA) | defined(HAVE_MAGMA_BATCH)
#define num_streams 1
#define max(a,b) (((a) > (b) ? (a) : (b)))
#define min(a,b) (((a) < (b) ? (a) : (b)))

#define batch_count 5000
#define batch_pad 32
#define MAGMA_BATCH_DGEMV_ATOMIC
#define BATCH_IN_PLACE_Y // this is needed with c_hacapk_adot_body_lfcpy_batch_sorted_
#define SORT_BATCH_BY_SIZES
#define USE_QSORT
#define batch_max_blocksize 10000000 
//#define batch_max_blocksize 1000 

// sort blocks for batched kernel to utilize GPU better
#define sort_array_size 4
#define sort_group_size 1

// On Tsubame 2
//#define procs_per_node 3 // number processes per node (used to figure out which process uses which gpu)
//#define gpus_per_proc 3  // number of gpus per process (used for multi-GPU/proc support)

// On Tsubame 4
#define procs_per_node 4 // number processes per node (used to figure out which process uses which gpu)
#define gpus_per_proc 4  // number of gpus per process (used for multi-GPU/proc support)

// On Reedbush
//#define procs_per_node 2 // number processes per node (used to figure out which process uses which gpu)
//#define gpus_per_proc 2  // number of gpus per process (used for multi-GPU/proc support)


void c_hacapk_adot_body_lfcpy_batch_sorted_(int *nd, stc_HACApK_leafmtxp *st_leafmtxp);
void c_hacapk_adot_body_lfmtx_batch_queue(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu,
                                          double *time_batch, double *time_set, double *time_copy,
                                          int on_gpu, magma_queue_t queue);
void c_hacapk_adot_body_lfmtx_batch_(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu,
                                     double *time_batch, double *time_set, double *time_copy);

void c_hacapk_adot_body_lfcpy_batch_sorted_mgpu_(int *nd, stc_HACApK_leafmtxp *st_leafmtxp,
                                                 magma_queue_t *queue);
void c_hacapk_adot_body_lfmtx_batch_mgpu(int flag, double *zau, 
                                         stc_HACApK_leafmtxp *st_leafmtxp, stc_HACApK_lcontrol *st_ctl, 
                                         double *zu, double *zbu,
                                         double *zau_cpu, double *zu_cpu,
                                         double *time_batch, double *time_set, double *time_copy,
                                         double *time_set1, double *time_set2, double *time_set3,
                                         int on_gpu, magma_queue_t *queue);
void c_hacapk_adot_body_lfmtx_batch_mgpu2(int flag, double *zau, 
                                          stc_HACApK_leafmtxp *st_leafmtxp, stc_HACApK_lcontrol *st_ctl, 
                                          double **zu_mgpu, double *zbu,
                                          double *zau_cpu, double *zu_cpu,
                                          double *time_batch, double *time_set, double *time_copy,
                                          double *time_set1, double *time_set2, double *time_set3,
                                          int on_gpu, magma_queue_t *queue);

int hacapk_size_sorter(const void* arg1,const void* arg2);
int hacapk_size_sorter_trans(const void* arg1,const void* arg2);
void hacapk_sort(int n, int *sizes);

static int get_device_id(stc_HACApK_leafmtxp *st_leafmtxp) {
    return (st_leafmtxp->mpi_rank)%procs_per_node;
}
#endif
