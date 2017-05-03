#include	<stdio.h>
#include	<stdlib.h>
#include	<time.h>
#include	"omp.h"
#include	"mpi.h"
#include	"HACApK_FPGA.h"

//!***c_HACApK_adot_body_lfmtx
void  c_hacapk_adot_body_lfmtx_(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu) {
 register int ip,il,it;
 int nlf,ndl,ndt,nstrtl,nstrtt,kt,itl,itt,ill;
 int st_lf_stride = st_leafmtxp->st_lf_stride;
 
 nlf=st_leafmtxp->nlf;
 for (ip = 0; ip < nlf; ip++) {
   /**/
   stc_HACApK_leafmtx *sttmp;
   sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
   /**/

   ndl   =sttmp->ndl; // m: number of rows
   ndt   =sttmp->ndt; // n: number of columns
   nstrtl=sttmp->nstrtl; // i: index of first row (base-1)
   nstrtt=sttmp->nstrtt; // j: index of first column (base-1)
   if (sttmp->ltmtx == 1) { // compressed
     /**/
     double *a2tmp = (double *)((void*)(sttmp->a1)+sttmp->a1size);
     /**/
     kt = sttmp->kt; // rank
     
     for(il=0; il<kt; il++){
       zbu[il]=0.0;
       for(it=0; it<ndt; it++){ // zbu := V'*zu
         itt=it+nstrtt-1;
         itl=it+il*ndt; 
         zbu[il] += sttmp->a1[itl]*zu[itt];
       }
     }
     for(il=0; il<kt; il++){ // zau := U*zbu
       for(it=0; it<ndl; it++){
         ill=it+nstrtl-1;
         itl=it+il*ndl; 
         zau[ill] += a2tmp[itl]*zbu[il];
       }
     }
   } else if(sttmp->ltmtx==2){ // full
     for(il=0; il<ndl; il++){
       ill=il+nstrtl-1; 
       for(it=0; it<ndt; it++){
         itt=it+nstrtt-1; 
         itl=it+il*ndt;
         zau[ill] += sttmp->a1[itl]*zu[itt];
       }
     }
   }
 }
}

// /////////////////////////////////////////////////////////////////////////
// DGEMV using MAGMA
// > using CuBLAS DGEMV (out-of-place/in-place)
// > using MAGMA batched DGEMV (out-of-place/in-pace)
// > using MAGMA batched DGEMV, sorted (in-place)
// /////////////////////////////////////////////////////////////////////////
#if defined(HAVE_MAGMA) | defined(HAVE_MAGMA_BATCH)
#define num_streams 1
#define max(a,b) (((a) > (b) ? (a) : (b)))
#define min(a,b) (((a) < (b) ? (a) : (b)))

/////////////////////////////////////////////////////
// MatVec on GPU
void  c_hacapk_adot_body_lfmtx_gpu_(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu) {
    // constants
    int ione = 1;
    double one = 1.0;
    double zero = 0.0;

    int ip;
    int nlf, ndl, ndt, nstrtl, nstrtt, kt;
    int st_lf_stride = st_leafmtxp->st_lf_stride;
 
    //#define GPU
    #define CPU
    #if defined(GPU)
    // allocate queue
    magma_device_t cdev;
    magma_queue_t queue[num_streams];
    magma_getdevice( &cdev );
    for (ip = 0; ip < num_streams; ip++) {
        magma_queue_create( cdev, &queue[ip] );
    }

    // copy the input vector to GPU
    magma_dsetvector( st_leafmtxp->gn,  zu, 1, st_leafmtxp->zu_gpu,  1, queue[0] );
    magma_dsetvector( st_leafmtxp->m,  zau, 1, st_leafmtxp->zau_gpu[0], 1, queue[0] );
    for (ip = 1; ip < num_streams; ip++) {
        magmablas_dlaset( MagmaFull, st_leafmtxp->m, 1, zero, zero, 
                          st_leafmtxp->zau_gpu[ip], st_leafmtxp->m, queue[ip] );
    }
    #endif

    // parse all the blocks
    #ifdef PROF_MAGMA_BATCH
    double tic = MPI_Wtime();
    #endif
    nlf=st_leafmtxp->nlf;
    for (ip = 0; ip < nlf; ip++) {
        /**/
        stc_HACApK_leafmtx *sttmp;
        sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
        /**/

        ndl    = sttmp->ndl; // m: number of rows
        ndt    = sttmp->ndt; // n: number of columns
        nstrtl = sttmp->nstrtl; // i: index of first row (base-1)
        nstrtt = sttmp->nstrtt; // j: index of first column (base-1)
        #if defined(GPU)
        int stream_id = ip%num_streams;
        #endif
        if (sttmp->ltmtx == 1) { // compressed
            /**/
            double *a2tmp = (double *)((void*)(sttmp->a1)+sttmp->a1size);
            /**/
            kt=sttmp->kt; // rank
            // zbu := V'*zu
            #if defined(GPU)
            magmablas_dgemv(MagmaTrans, ndt, kt, 
                            one,  st_leafmtxp->mtx1_gpu[ip], ndt, 
                                 &(st_leafmtxp->zu_gpu[nstrtt-1]), ione,
                            zero, st_leafmtxp->zbu_gpu[stream_id], ione,
                            queue[stream_id] );
            #elif defined(CPU)
            dgemv_("T", &ndt, &kt, 
                   &one, sttmp->a1, &ndt, 
                         &zu[nstrtt-1], &ione,
                   &zero, zbu, &ione );
            #else
            for (il = 0; il < kt; il++) {
                zbu[il]=0.0;
                for ( it = 0; it < ndt; it++) { 
                    itt=it+nstrtt-1;
                    itl=it+il*ndt; 
                    zbu[il] += sttmp->a1[itl]*zu[itt];
                }
            }
            #endif

            // zau :+= U*zbu
            #if defined(GPU)
            magmablas_dgemv(MagmaNoTrans, ndl, kt, 
                            one,   st_leafmtxp->mtx2_gpu[ip], ndl, 
                                   st_leafmtxp->zbu_gpu[stream_id], ione,
                            one, &(st_leafmtxp->zau_gpu[stream_id][nstrtl-1]), ione,
                            queue[stream_id] );
            #elif defined(CPU)
            dgemv_("N", &ndl, &kt, 
                   &one, a2tmp, &ndl, 
                         zbu, &ione,
                   &one, &zau[nstrtl-1], &ione );
            #else
            for(il = 0; il < kt; il++) {
                for(it = 0; it < ndl; it++){
                    ill=it+nstrtl-1;
                    itl=it+il*ndl; 
                    zau[ill] += a2tmp[itl]*zbu[il];
                }
            }
            #endif
        } else if(sttmp->ltmtx == 2) { // full
            #if defined(GPU)
            magmablas_dgemv(MagmaTrans, ndt, ndl, 
                            one,   st_leafmtxp->mtx1_gpu[ip], ndt, 
                                 &(st_leafmtxp->zu_gpu[nstrtt-1]), ione,
                            one, &(st_leafmtxp->zau_gpu[stream_id][nstrtl-1]), ione,
                            queue[stream_id] );
            #elif defined(CPU)
            dgemv_("T", &ndt, &ndl, 
                   &one, sttmp->a1, &ndt, 
                         &zu[nstrtt-1], &ione,
                   &one, &zau[nstrtl-1], &ione );
            #else
            for (il = 0; il < ndl; il++){
                ill=il+nstrtl-1; 
                for (it = 0; it < ndt; it++) {
                    itt=it+nstrtt-1; 
                    itl=it+il*ndt;
                    zau[ill] += sttmp->a1[itl]*zu[itt];
                }
            }
            #endif
        }
    }
    #if defined(GPU)
    for (ip = 1; ip < num_streams; ip++) {
        magma_queue_sync( queue[ip] );
        magma_queue_destroy( queue[ip] );
        magma_daxpy( st_leafmtxp->m, one, st_leafmtxp->zau_gpu[ip], 1,
                                          st_leafmtxp->zau_gpu[0],  1,
                     queue[0] );
    }
    // synch to get time
    MPI_Barrier(MPI_COMM_WORLD);
    magma_queue_sync( queue[0] );
    #ifdef PROF_MAGMA_BATCH
    if (st_leafmtxp->mpi_rank == 0) {
        printf( " time_gpu: %.2e seconds\n\n",MPI_Wtime()-tic );
    }
    #endif
    // copy back
    magma_dgetvector( st_leafmtxp->m, st_leafmtxp->zau_gpu[0], 1, zau, 1, queue[0] );
    magma_queue_destroy( queue[0] );
    #else
    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef PROF_MAGMA_BATCH
    if (st_leafmtxp->mpi_rank == 0) {
        printf( " time_cpu: %.2e seconds\n\n",MPI_Wtime()-tic );
    }
    #endif
    #endif
}

/////////////////////////////////////////////////////
// copy blocks to GPU
void  c_hacapk_adot_body_lfcpy_gpu_(int *nd, stc_HACApK_leafmtxp *st_leafmtxp) {
    #define GPU
    #if defined(GPU)
    // local variables
    register int ip;
    int nlf, ndl, ndt, nstrtl, nstrtt, kt;
    int st_lf_stride = st_leafmtxp->st_lf_stride;

    // let me initialize here for now..
    magma_init();
    //st_leafmtxp->mpi_comm = MPI_COMM_WORLD; // comm world for now
    MPI_Comm_rank(MPI_COMM_WORLD, &(st_leafmtxp->mpi_rank));
    if (st_leafmtxp->mpi_rank == 0) magma_print_environment();

    // allocate queue
    magma_device_t cdev;
    magma_queue_t queue = NULL;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    // number of blocks
    nlf = st_leafmtxp->nlf; 

    // initialize data structure
    st_leafmtxp->gn = *nd;
    st_leafmtxp->m = 0;
    st_leafmtxp->n = 0;
    st_leafmtxp->max_block = 0;
    st_leafmtxp->mtx1_gpu = (magmaDouble_ptr*)malloc(nlf * sizeof(magmaDouble_ptr));
    st_leafmtxp->mtx2_gpu = (magmaDouble_ptr*)malloc(nlf * sizeof(magmaDouble_ptr));

    // parse all the blocks
    for (ip = 0; ip < nlf; ip++) {
        /**/
        stc_HACApK_leafmtx *sttmp;
        sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
        ndl    = sttmp->ndl; // m: number of rows
        ndt    = sttmp->ndt; // n: number of columns
        nstrtl = sttmp->nstrtl; // i: index of first row (base-1)
        nstrtt = sttmp->nstrtt; // j: index of first column (base-1)

        // local matrix size
        if (nstrtl == nstrtt) {
            st_leafmtxp->m += ndl;
        }
        if (st_leafmtxp->max_block < max(ndl, ndt)) {
            st_leafmtxp->max_block = max(ndl, ndt);
        }
        /**/
        if (sttmp->ltmtx == 1) { // compressed
            /**/
            double *a2tmp = (double *)((void*)(sttmp->a1)+sttmp->a1size);
            /**/
            kt = sttmp->kt; // rank
            // copy V
            st_leafmtxp->mtx1_gpu[ip] = NULL;
            int retval = magma_malloc( (void**) &(st_leafmtxp->mtx1_gpu[ip]), (ndt*kt)*sizeof(double) );
            if ( MAGMA_SUCCESS != retval ) {
                fprintf( stderr, "!!!! magma_malloc failed for mtx1_gpu[0][%d]\n", ip);
                exit(0);
            }
            magma_dsetmatrix( ndt, kt, sttmp->a1, ndt, st_leafmtxp->mtx1_gpu[ip], ndt, queue );

            // copy U
            st_leafmtxp->mtx2_gpu[ip] = NULL;
            retval = magma_malloc( (void**) &(st_leafmtxp->mtx2_gpu[ip]), (ndl*kt)*sizeof(double) );
            if ( MAGMA_SUCCESS != retval ) {
                fprintf( stderr, "!!!! magma_malloc failed for mtx2_gpu[1][%d]\n", ip);
                exit(0);
            }
            magma_dsetmatrix( ndl, kt, a2tmp, ndl, st_leafmtxp->mtx2_gpu[ip], ndl, queue );
        } else if (sttmp->ltmtx == 2) { // full
            st_leafmtxp->mtx1_gpu[ip] = NULL;
            int retval = magma_malloc( (void**) &(st_leafmtxp->mtx1_gpu[ip]), (ndt*ndl)*sizeof(double) );
            if ( MAGMA_SUCCESS != retval ) {
                fprintf( stderr, "!!!! magma_malloc failed for mtx1_gpu[0][%d]\n", ip);
                exit(0);
            }
            magma_dsetmatrix( ndt, ndl, sttmp->a1, ndt, st_leafmtxp->mtx1_gpu[ip], ndt, queue );
            st_leafmtxp->mtx2_gpu[ip] = NULL;
        }
    }
    // let's just use global size.
    st_leafmtxp->m = st_leafmtxp->gn;
    st_leafmtxp->n = st_leafmtxp->gn;
    // workspace for GEMV on GPU
    st_leafmtxp->zu_gpu = NULL; 
    st_leafmtxp->zau_gpu = NULL;
    st_leafmtxp->zbu_gpu = NULL;
    if (st_leafmtxp->max_block > 0) {
        st_leafmtxp->zbu_gpu = (double**)malloc( num_streams * sizeof(double*) );
        for (ip = 0; ip < num_streams; ip++) {
            int retval = magma_malloc( (void**) &st_leafmtxp->zbu_gpu[ip], (st_leafmtxp->max_block)*sizeof(double) );
            if ( MAGMA_SUCCESS != retval ) {
                fprintf( stderr, "!!!! magma_malloc failed for zbu_gpu\n");
                exit(0);
            }
        }
    }
    if (st_leafmtxp->m > 0) {
        st_leafmtxp->zau_gpu = (double**)malloc( num_streams * sizeof(double*) );
        for (ip = 0; ip < num_streams; ip++) {
            int retval = magma_malloc( (void**) &st_leafmtxp->zau_gpu[ip], (st_leafmtxp->m)*sizeof(double) );
            if ( MAGMA_SUCCESS != retval ) {
                fprintf( stderr, "!!!! magma_malloc failed for zau_gpu\n");
                exit(0);
            }
        }
    }
    if (st_leafmtxp->n > 0) {
        int retval = magma_malloc( (void**) &st_leafmtxp->zu_gpu, (st_leafmtxp->gn)*sizeof(double) );
        if ( MAGMA_SUCCESS != retval ) {
            fprintf( stderr, "!!!! magma_malloc failed for zu_gpu\n");
            exit(0);
        }
    }
    if (st_leafmtxp->mpi_rank == 0) {
        printf( " %d-by-%d matrix (# blocks=%d)\n",st_leafmtxp->m,st_leafmtxp->n,nlf );
    }
    magma_queue_destroy( queue );
    #endif
}

/////////////////////////////////////////////////////
// delete GPU memory
void  c_hacapk_adot_body_lfdel_gpu_(stc_HACApK_leafmtxp *st_leafmtxp) {
    #if defined(GPU)
    int ip; 
    for(ip = 0; ip < st_leafmtxp->nlf; ip++) {
        if(st_leafmtxp->mtx1_gpu[ip] != NULL) {
            magma_free(st_leafmtxp->mtx1_gpu[ip]);
            st_leafmtxp->mtx1_gpu[ip] = NULL;
        }
        if(st_leafmtxp->mtx2_gpu[ip] != NULL) {
            magma_free(st_leafmtxp->mtx2_gpu[ip]);
            st_leafmtxp->mtx2_gpu[ip] = NULL;
        }
    }
    free(st_leafmtxp->mtx1_gpu);
    free(st_leafmtxp->mtx2_gpu);

    if (st_leafmtxp->zu_gpu != NULL) {
        magma_free(st_leafmtxp->zu_gpu);
        st_leafmtxp->zu_gpu = NULL;
    } 
    if (st_leafmtxp->zbu_gpu != NULL) {
        for (ip = 0; ip < num_streams; ip++) {
            magma_free(st_leafmtxp->zbu_gpu[ip]);
        }
        free(st_leafmtxp->zbu_gpu);
        st_leafmtxp->zbu_gpu = NULL;
    } 
    if (st_leafmtxp->zau_gpu != NULL) {
        for (ip = 0; ip < num_streams; ip++) {
            magma_free(st_leafmtxp->zau_gpu[ip]); 
        }
        free(st_leafmtxp->zau_gpu); 
        st_leafmtxp->zau_gpu = NULL;
    }

    // let me finalize it here for now
    magma_finalize();
    #endif
}
#endif

#if defined(HAVE_MAGMA_BATCH)
//#define batch_count 10000
#define batch_count 5000
//#define batch_count 1
#define batch_pad 32
#define MAGMA_BATCH_DGEMV_ATOMIC
#define BATCH_IN_PLACE_Y // this is needed with c_hacapk_adot_body_lfcpy_batch_sorted_
#define SORT_BATCH_BY_SIZES
#define USE_QSORT
#define gpus_per_proc 3
#define batch_max_blocksize 10000000 
//#define batch_max_blocksize 1000 

void magma_iprint( magma_int_t m, magma_int_t n,
                   magma_int_t *A, magma_int_t lda ) {
    int i,j;
    for (i=0; i<m; i++) {
        for (j=0; j<n; j++) printf( "%d ",A[i+j*lda] );
        printf( "\n" );
    }
}

void magma_iprint_gpu( magma_int_t m, magma_int_t n,
                       magma_int_t *dA, magma_int_t ldda,
                       magma_queue_t queue ) {
  
    magma_int_t *A;

    magma_imalloc_cpu( &A, m*n );
    magma_igetmatrix( m, n, dA, ldda, A, m, queue );
    magma_iprint( m, n, A, m );
    magma_free_cpu( A );
}


static int get_device_id(stc_HACApK_leafmtxp *st_leafmtxp) {
    return (st_leafmtxp->mpi_rank)%gpus_per_proc;
}

/////////////////////////////////////////////////////
// MatVec with batched GEMV
int c_hacapk_adot_body_lfmtx_batch_dgemv(stc_HACApK_leafmtxp *st_leafmtxp, int *saved_ip[2],
                                         int *ip_start, int num_batch, int count,
                                         int *batchCount, int *num_saved,
                                         double * zau_batch, magma_queue_t queue);
int c_hacapk_adot_body_lfmtx_batch_daxpy(stc_HACApK_leafmtxp *st_leafmtxp, int *saved_ip[2],
                                         int num_batch, int count, int k_start, int ip_start,
                                         double* zau_batch, double* zau, magma_queue_t queue);

void c_hacapk_adot_body_lfmtx_batch_(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu,
                                     double *time_batch, double *time_set, double *time_copy) {
    // constants
    double zero = 0.0;

    int ip;
    int nlf = st_leafmtxp->nlf;
    int *saved_ip[2]; 
    if (st_leafmtxp->batch_order == NULL) {
        saved_ip[0] = (int*)malloc( nlf * sizeof(int) ); 
        saved_ip[1] = (int*)malloc( nlf * sizeof(int) ); 
    }

    // allocate queue
    magma_device_t cdev;
    magma_queue_t queue;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    // copy the input vector to GPU
    double * zau_batch = NULL;
    //MPI_Barrier(MPI_COMM_WORLD);
    #ifdef PROF_MAGMA_BATCH
    //double time_copy = 0.0;
    //double time_batch = 0.0;
    #endif
    int ii, dgemv_count = 1;
    for (ii=0; ii<dgemv_count; ii++) {
        #ifdef PROF_MAGMA_BATCH
        double tic = MPI_Wtime();
        #endif
        int num_saved = 0, count = 0;
        #if defined(ACCUME_ON_CPU)
        magma_dmalloc_pinned( &zau_batch,  batch_count*(st_leafmtxp->max_block) );
        #else
        magma_dsetvector( st_leafmtxp->m, zau, 1, st_leafmtxp->zau_gpu[0], 1, queue );
        #endif
        magma_dsetvector( st_leafmtxp->gn,  zu, 1, st_leafmtxp->zu_gpu,  1, queue );

        // start timer
        #ifdef PROF_MAGMA_BATCH
        *time_copy += MPI_Wtime()-tic;
        tic = MPI_Wtime();
        #endif

        // parse all the blocks
        int num_batch = 0;
        #if defined(BATCH_IN_PLACE_Y)
        // first part of low-rank, zbu := V'*zu
        magmablas_dlaset( MagmaFull, st_leafmtxp->total_size_y, 1, zero, zero, 
                          st_leafmtxp->zbu_gpu[0], st_leafmtxp->total_size_y, queue );
        magma_queue_sync( queue );
        *time_set += MPI_Wtime()-tic;
        tic = MPI_Wtime();
        #endif
        fflush(stdout);
        for (ip = 0; ip < max(st_leafmtxp->num_batch, nlf) || num_saved > 0;) {
            /**/
            int ip_start = ip;
            int num_start = num_batch - (ip < nlf ? 0 : st_leafmtxp->num_streamed);
            int batchCount = 0;

            // call batched GEMV and non-blocking copy to CPU
            //#define PROF_MAGMA_BATCH_COUNT_2
            #ifdef PROF_MAGMA_BATCH_COUNT_2
            magma_queue_sync( queue );
            double time_start = MPI_Wtime();
            #endif
            c_hacapk_adot_body_lfmtx_batch_dgemv(st_leafmtxp, saved_ip,
                                                 &ip, num_start, count,
                                                 &batchCount, &num_saved,
                                                 zau_batch, queue);
            #ifdef PROF_MAGMA_BATCH_COUNT_2
            magma_queue_sync( queue );
            if (st_leafmtxp->mpi_rank == 0) {
                printf( " %d: time_dgemv_once : %.2e seconds\n", count,MPI_Wtime()-time_start );
            }
            #endif
            #if defined(ACCUME_ON_CPU)
            magma_queue_sync( queue );
            #endif

            // accumulate the results of GEMVs
            #if !defined(BATCH_IN_PLACE_Y)
            c_hacapk_adot_body_lfmtx_batch_daxpy(st_leafmtxp, saved_ip,
                                                 num_batch, count, k_start, ip_start,
                                                 zau_batch, zau, queue);
            #endif

            //#define PROF_MAGMA_BATCH_COUNT_1
            #ifdef PROF_MAGMA_BATCH_COUNT_1
            if (st_leafmtxp->mpi_rank == 0) printf( " batchCount = %d, ip=%d:%d\n",batchCount,ip_start,ip_start+batchCount-1 );
            #endif
            num_batch +=(1+ batchCount);
            count ++;
            // standard GEMV for large full/V-matrix
            if (num_batch == (nlf+count)-st_leafmtxp->num_streamed) {
                // streamed GEMV
                int jj;
                double one  = 1.0;
                for (jj=0; jj<st_leafmtxp->num_streamed; jj++) {
                    magmablas_dgemv( MagmaNoTrans, 
                                     st_leafmtxp->h_M_streamed[jj], st_leafmtxp->h_N_streamed[jj],
                                     one, st_leafmtxp->h_A_array_streamed[jj], st_leafmtxp->h_lda_streamed[jj],
                                          st_leafmtxp->h_X_array_streamed[jj], 1,
                                     one, st_leafmtxp->h_Y_array_streamed[jj], 1, queue );
                }
                num_batch += st_leafmtxp->num_streamed;
                ip += st_leafmtxp->num_streamed;
            }
            // standard GEMV for large U-matrix
            if (num_batch == (st_leafmtxp->num_batch+count)-st_leafmtxp->num_streamed_t) {
                int jj;
                double one  = 1.0;
                for (jj=st_leafmtxp->num_streamed; jj<st_leafmtxp->num_streamed+st_leafmtxp->num_streamed_t; jj++) {
                    magmablas_dgemv( MagmaNoTrans, 
                                     st_leafmtxp->h_M_streamed[jj], st_leafmtxp->h_N_streamed[jj],
                                     one, st_leafmtxp->h_A_array_streamed[jj], st_leafmtxp->h_lda_streamed[jj],
                                          st_leafmtxp->h_X_array_streamed[jj], 1,
                                     one, st_leafmtxp->h_Y_array_streamed[jj], 1, queue );
                }
                num_batch += st_leafmtxp->num_streamed_t;
                ip += st_leafmtxp->num_streamed_t;
            }
        }
        // stop timer
        magma_queue_sync( queue );
        #ifdef PROF_MAGMA_BATCH
        *time_batch += MPI_Wtime()-tic;
        tic = MPI_Wtime();
        #endif
        #if defined(ACCUME_ON_CPU)
        magma_free_pinned(zau_batch);
        #else
        magma_dgetvector( st_leafmtxp->m, st_leafmtxp->zau_gpu[0], 1, zau, 1, queue );
        #endif
        #ifdef PROF_MAGMA_BATCH
        *time_copy += MPI_Wtime()-tic;
        #endif
    }
    #define PROF_MAGMA_BATCH_COUNT
    #ifdef PROF_MAGMA_BATCH_COUNT
    if (st_leafmtxp->mpi_rank == 0) {
        printf( " time_copy : %.2e seconds\n",  *time_copy /dgemv_count );
        printf( " time_set  : %.2e seconds\n",  *time_set  /dgemv_count );
        printf( " time_batch: %.2e seconds\n\n",*time_batch/dgemv_count );
    }
    fflush(stdout);
    #endif

    if (st_leafmtxp->batch_order == NULL) {
        free(saved_ip[0]);
        free(saved_ip[1]);
    }
    magma_queue_sync( queue );
    magma_queue_destroy( queue );
}

// batched GEMV
int c_hacapk_adot_body_lfmtx_batch_dgemv(stc_HACApK_leafmtxp *st_leafmtxp, int *saved_ip[2],
                                         int *ip_start, int num_batch, int count,
                                         int *batchCount, int *num_saved,
                                         double * zau_batch, magma_queue_t queue) {
    double one = 1.0;

    int *d_M = st_leafmtxp->d_M;
    int *d_N = st_leafmtxp->d_N;
    int *d_inc = st_leafmtxp->d_inc;
    int *d_lda = st_leafmtxp->d_lda;
    double **d_A_array = st_leafmtxp->d_A_array;
    double **d_X_array = st_leafmtxp->d_X_array;
    double **d_Y_array = st_leafmtxp->d_Y_array;

    int k, ip;
    int k_start;
    int nlf = st_leafmtxp->nlf;

    if (st_leafmtxp->batch_order != NULL) {
        *num_saved = 0;
        if (*ip_start < nlf) {
            *batchCount = min(batch_count, (nlf-st_leafmtxp->num_streamed)-(*ip_start));
        } else {
            *batchCount = min(batch_count, (st_leafmtxp->num_batch-st_leafmtxp->num_streamed_t)-(*ip_start));
        }
        ip = (*ip_start) + (*batchCount);
    } else {
        k_start = *num_saved;
        *batchCount = k_start;
        *num_saved = 0;

        int st_lf_stride = st_leafmtxp->st_lf_stride;
        for (k = k_start, ip = *ip_start; k < batch_count && ip < nlf; ip++, k++) {
            /**/
            stc_HACApK_leafmtx *sttmp;
            sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
            /**/
            if (sttmp->ltmtx == 1) { // compressed
                /**/
                int batch_id = count%2;
                saved_ip[batch_id][*num_saved] = ip;
                // zbu := V'*zu
                (*num_saved) ++;
                (*batchCount) ++;
            } else if(sttmp->ltmtx == 2) { // full
                (*batchCount) ++;
            }
        }
    }

    #if defined(MAGMA_BATCH_DGEMV_ATOMIC)
    #if defined(BATCH_IN_PLACE_Y)
      #if defined(SORT_BATCH_BY_SIZES)
      // passing max M and N
      magmablas_dgemv_vbatched_max_nocheck_atomic(
                                      st_leafmtxp->transA, &d_M[num_batch], &d_N[num_batch],
                                      one, &d_A_array[num_batch], &d_lda[num_batch],
                                           &d_X_array[num_batch], &d_inc[num_batch],
                                           &d_Y_array[num_batch], &d_inc[num_batch],
                                      *batchCount, st_leafmtxp->max_M[count], st_leafmtxp->max_N[count],
                                      queue);
      /*magmablas_dgemv_vbatched_max_nocheck(
                                      st_leafmtxp->transA, &d_M[num_batch], &d_N[num_batch],
                                      one, &d_A_array[num_batch], &d_lda[num_batch],
                                           &d_X_array[num_batch], &d_inc[num_batch],
                                      one, &d_Y_array[num_batch], &d_inc[num_batch],
                                      *batchCount, st_leafmtxp->max_M[count], st_leafmtxp->max_N[count],
                                      queue);*/
      #else
      // beta is one
      //magmablas_dgemv_vbatched_atomic
      magmablas_dgemv_vbatched_nocheck_atomic(
                                      st_leafmtxp->transA, &d_M[num_batch], &d_N[num_batch],
                                      one, &d_A_array[num_batch], &d_lda[num_batch],
                                           &d_X_array[num_batch], &d_inc[num_batch],
                                           &d_Y_array[num_batch], &d_inc[num_batch],
                                      *batchCount, queue);
      #endif
    #else
    magmablas_dgemv_vbatched_atomic(st_leafmtxp->transA, &d_M[num_batch], &d_N[num_batch],
                                    one,  &d_A_array[num_batch], &d_lda[num_batch],
                                          &d_X_array[num_batch], &d_inc[num_batch],
                                    zero, &d_Y_array[num_batch], &d_inc[num_batch],
                                    *batchCount, queue);
    #endif
    #else
    #if defined(BATCH_IN_PLACE_Y)
    magmablas_dgemv_vbatched(st_leafmtxp->transA, &d_M[num_batch], &d_N[num_batch],
                             one, &d_A_array[num_batch], &d_lda[num_batch],
                                  &d_X_array[num_batch], &d_inc[num_batch],
                             one, &d_Y_array[num_batch], &d_inc[num_batch],
                             *batchCount, queue);
    #else
    magmablas_dgemv_vbatched(st_leafmtxp->transA, &d_M[num_batch], &d_N[num_batch],
                             one,  &d_A_array[num_batch], &d_lda[num_batch],
                                   &d_X_array[num_batch], &d_inc[num_batch],
                             zero, &d_Y_array[num_batch], &d_inc[num_batch],
                             *batchCount, queue);
    #endif
    #endif
    
    #if defined(ACCUME_ON_CPU)
    /* get results */
    int size_y = 0;
    for (k = 0; k < k_start; k++) {
        int batch_id = (count-1)%2;
        int iip = saved_ip[batch_id][k];
        int st_lf_stride = st_leafmtxp->st_lf_stride;

        /**/
        stc_HACApK_leafmtx *sttmp;
        sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * iip;
        /**/

        int ndl = sttmp->ndl; // m: number of rows
        magma_dgetvector_async( ndl, h_Y_array[num_batch+k], 1, &zau_batch[size_y], 1, queue );
        size_y += ndl;
    }
    for (k = k_start, ip = *ip_start; k < batch_count && ip < nlf; ip++, k++) {
        int st_lf_stride = st_leafmtxp->st_lf_stride;
        /**/
        stc_HACApK_leafmtx *sttmp;
        sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
        /**/

        if(sttmp->ltmtx == 2) { // full
            int ndl = sttmp->ndl; // m: number of rows
            magma_dgetvector_async( ndl, h_Y_array[num_batch+k], 1, &zau_batch[size_y], 1, queue );
            size_y += ndl;
        }
    }
    #endif
    *ip_start = ip;
    return 0;
}

// accumlate results of GEMVs
int c_hacapk_adot_body_lfmtx_batch_daxpy(stc_HACApK_leafmtxp *st_leafmtxp, int *saved_ip[2],
                                         int num_batch, int count, int k_start, int ip_start,
                                         double* zau_batch, double* zau, magma_queue_t queue) {
    // constants
    #if defined(ACCUME_ON_CPU)
    int ione = 1;
    #endif
    double one = 1.0;

    int k, ip;
    int nlf = st_leafmtxp->nlf;
    int st_lf_stride = st_leafmtxp->st_lf_stride;

    int size_y = 0;
    for (k = 0; k < k_start; k++) {
        int batch_id = (count-1)%2;
        int iip = saved_ip[batch_id][k];
        /**/
        stc_HACApK_leafmtx *sttmp;
        sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * iip;
        /**/

        int ndl    = sttmp->ndl;    // m: number of rows
        int nstrtl = sttmp->nstrtl; // i: index of first row
        #if defined(ACCUME_ON_CPU)
        daxpy_(&ndl,
               &one, &zau_batch[size_y], &ione,
                     &zau[nstrtl-1], &ione );
        #else
        magma_daxpy(ndl, one, st_leafmtxp->h_Y_array[num_batch+k], 1,
                             &st_leafmtxp->zau_gpu[0][nstrtl-1],  1,
                    queue );
        #endif
        size_y += ndl;
    }
    for (k = k_start, ip = ip_start; k < batch_count && ip < nlf; ip++, k++) {
        /**/
        stc_HACApK_leafmtx *sttmp;
        sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
        /**/

        int ndl    = sttmp->ndl;    // m: number of rows
        int nstrtl = sttmp->nstrtl; // i: index of first row

        if(sttmp->ltmtx == 2) { // full
            #if defined(ACCUME_ON_CPU)
            daxpy_(&ndl,
                   &one, &zau_batch[size_y], &ione,
                         &zau[nstrtl-1], &ione );
            #else
            magma_daxpy(ndl, one, st_leafmtxp->h_Y_array[num_batch+k], 1,
                                 &st_leafmtxp->zau_gpu[0][nstrtl-1],  1,
                        queue );
            #endif
            size_y += ndl;
        }
    }
    return 0;
}


/////////////////////////////////////////////////////
// copy blocks to GPU
void  c_hacapk_adot_body_lfcpy_batch_(stc_HACApK_leafmtxp *st_leafmtxp) {

    // local variables
    int ip, i, j, k;
    int nlf, ndl, ndt, nstrtl, nstrtt, kt, lda;
    int st_lf_stride = st_leafmtxp->st_lf_stride;

    // let me initialize here for now..
    magma_init();
    //st_leafmtxp->mpi_comm = MPI_COMM_WORLD; // comm world for now
    MPI_Comm_rank(MPI_COMM_WORLD, &(st_leafmtxp->mpi_rank));
    if (st_leafmtxp->mpi_rank == 0) magma_print_environment();

    // allocate queue
    magma_device_t cdev;
    magma_queue_t queue = NULL;
    magma_setdevice( get_device_id(st_leafmtxp) );
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );
    printf( " processor %d uses %d GPU\n",st_leafmtxp->mpi_rank,(st_leafmtxp->mpi_rank)%gpus_per_proc);

    // number of blocks
    nlf = st_leafmtxp->nlf; 
    int *saved_ip[2]; 
    int *saved_bt[2]; 
    int *saved_sz[2]; 
    saved_ip[0] = (int*)malloc( nlf * sizeof(int) ); 
    saved_ip[1] = (int*)malloc( nlf * sizeof(int) ); 
    saved_bt[0] = (int*)malloc( nlf * sizeof(int) ); 
    saved_bt[1] = (int*)malloc( nlf * sizeof(int) ); 
    saved_sz[0] = (int*)malloc( nlf * sizeof(int) ); 
    saved_sz[1] = (int*)malloc( nlf * sizeof(int) ); 

    // initialize data structure
    st_leafmtxp->m = 0;
    st_leafmtxp->n = 0;
    st_leafmtxp->max_block = 0;
    // do GEMV(NoTrans)
    st_leafmtxp->transA = MagmaTrans;

    int num_batch = 0;
    int num_saved = 0;
    int total_size_a = 0;
    int total_size_y = 0;
    for (ip = 0; ip < nlf; ip++) {
        /**/
        stc_HACApK_leafmtx *sttmp;
        sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
        /**/

        kt     = sttmp->kt;  // rank
        ndl    = sttmp->ndl; // m: number of rows
        ndt    = sttmp->ndt; // n: number of columns
        nstrtl = sttmp->nstrtl; // i: index of first row (base-1)
        nstrtt = sttmp->nstrtt; // j: index of first column (base-1)

        // local matrix size
        if (nstrtl == nstrtt) {
            st_leafmtxp->m += ndl;
        }
        if (st_leafmtxp->max_block < max(ndl, ndt)) {
            st_leafmtxp->max_block = max(ndl, ndt);
        }

        if (sttmp->ltmtx == 1) { // compressed
            lda = magma_roundup( ndt, batch_pad );
            total_size_a += lda*kt;
            lda = magma_roundup( kt, batch_pad );
            total_size_a += lda*ndl;

            total_size_y += kt;
            #if !defined(BATCH_IN_PLACE_Y)
            total_size_y += ndl;
            #endif
            num_batch += 2;
        } else {                 // full
            lda = magma_roundup( ndt, batch_pad );
            total_size_a += lda*ndl;
            #if !defined(BATCH_IN_PLACE_Y)
            total_size_y += ndl;
            #endif
            num_batch += 1;
        }
    }
    // let's just use global size.
    st_leafmtxp->m = st_leafmtxp->gn;
    st_leafmtxp->n = st_leafmtxp->gn;
    fflush(stdout);
    if (st_leafmtxp->mpi_rank == 0) {
        printf( " %d-by-%d matrix (# blocks=%d)\n",st_leafmtxp->m,st_leafmtxp->n,nlf );
        printf( "  total_size_y=%d, total_size_a=%d\n",total_size_y,total_size_a );
    }
    fflush(stdout);

    // workspace for GEMV on GPU
    st_leafmtxp->zu_gpu = NULL; 
    st_leafmtxp->zau_gpu = NULL;
    st_leafmtxp->zbu_gpu = NULL;
    if (st_leafmtxp->m > 0) {
        st_leafmtxp->zau_gpu = (double**)malloc( sizeof(double*) );
        int retval = magma_malloc( (void**) &st_leafmtxp->zau_gpu[0], (st_leafmtxp->m)*sizeof(double) );
        if ( MAGMA_SUCCESS != retval ) {
            fprintf( stderr, "!!!! magma_malloc failed for zau_gpu\n");
            exit(0);
        }
    }
    if (st_leafmtxp->n > 0) {
        int retval = magma_malloc( (void**) &st_leafmtxp->zu_gpu, (st_leafmtxp->n)*sizeof(double) );
        if ( MAGMA_SUCCESS != retval ) {
            fprintf( stderr, "!!!! magma_malloc failed for zu_gpu\n");
            exit(0);
        }
    }
    if (total_size_y > 0) {
        st_leafmtxp->zbu_gpu = (double**)malloc( sizeof(double*) );
        int retval = magma_malloc( (void**) &st_leafmtxp->zbu_gpu[0], total_size_y*sizeof(double) );
        if ( MAGMA_SUCCESS != retval ) {
            fprintf( stderr, "!!!! magma_malloc failed for zbu_gpu\n");
            exit(0);
        }
        st_leafmtxp->total_size_y = total_size_y;
    }
    double *dA = NULL;
    if (total_size_a > 0) {
        int retval = magma_malloc( (void**) &dA, total_size_a*sizeof(double) );
        if ( MAGMA_SUCCESS != retval ) {
            fprintf( stderr, "!!!! magma_malloc failed for dA(%d)\n",total_size_a);
            exit(0);
        }
    }

    // extra space for M and N with batch
    num_batch += 1+(num_batch+batch_count-1)/batch_count;

    double **h_A_array, **h_X_array, **h_Y_array;
    magma_malloc_cpu((void**)&(h_A_array), num_batch*sizeof(double*));
    magma_malloc_cpu((void**)&(h_X_array), num_batch*sizeof(double*));
    magma_malloc_cpu((void**)&(h_Y_array), num_batch*sizeof(double*));

    magma_int_t *h_M, *h_N, *h_lda, *h_inc;
    magma_imalloc_cpu(&h_M, num_batch);
    magma_imalloc_cpu(&h_N, num_batch);
    magma_imalloc_cpu(&h_lda, num_batch);
    magma_imalloc_cpu(&h_inc, num_batch);

    // parse all the blocks
    total_size_y = 0;
    total_size_a = 0;
    num_batch = 0;
    int count = 0;
    //#define OUTPUT_SIZES
    #ifdef OUTPUT_SIZES
    FILE *fp;
    char filename[100];
    sprintf(filename,"sizes_%d.dat",st_leafmtxp->mpi_rank);
    fp = fopen(filename,"w");
    #endif
    for (ip = 0; ip < nlf || num_saved > 0;) {
        /**/
        for (k = 0; k<num_saved; k++) {
            int batch_id = (count-1)%2;
            int iip = saved_ip[batch_id][k];

            /**/
            stc_HACApK_leafmtx *sttmp;
            sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * iip;
            /**/

            kt     = sttmp->kt;  // rank
            ndl    = sttmp->ndl; // m: number of rows
            ndt    = sttmp->ndt; // n: number of columns
            nstrtl = sttmp->nstrtl; // i: index of first row (base-1)
            nstrtt = sttmp->nstrtt; // j: index of first column (base-1)
            lda = magma_roundup( kt, batch_pad );

            // copy U^T
            double *a2tmp = (double *)((void*)(sttmp->a1)+sttmp->a1size);
            double *tmp = (double*)malloc(ndl*kt * sizeof(double));
            for (i=0; i<ndl; i++) {
                for (j=0; j<kt; j++) tmp[j + i*kt] = a2tmp[i + j*ndl];
            }

            h_A_array[num_batch] = &dA[total_size_a];
            magma_dsetmatrix( kt, ndl, tmp, kt, h_A_array[num_batch], lda, queue );
            #ifdef OUTPUT_SIZES
            fprintf( fp,"2 %d %d\n",kt,ndl );
            #endif
            total_size_a += lda*ndl;
            free(tmp);

            // pointer to input, zu
            int size_y = saved_sz[batch_id][k];
            h_X_array[num_batch] = &st_leafmtxp->zbu_gpu[0][size_y];

            // pointer to output, y
            #if defined(BATCH_IN_PLACE_Y) //& defined(MAGMA_BATCH_DGEMV_ATOMIC)
            h_Y_array[num_batch] = &st_leafmtxp->zau_gpu[0][nstrtl-1];
            #else
            h_Y_array[num_batch] = &st_leafmtxp->zbu_gpu[0][total_size_y];
            total_size_y += ndl;
            #endif

            // dimmension
            h_M[num_batch] = kt;
            h_N[num_batch] = ndl;
            h_lda[num_batch] = lda;
            h_inc[num_batch] = 1;

            num_batch ++;
        }

        /**/
        int k_start = num_saved;
        num_saved = 0;
        for (k = k_start; k < batch_count && ip < nlf; ip++, k++) {
            /**/
            stc_HACApK_leafmtx *sttmp;
            sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
            /**/

            ndl    = sttmp->ndl; // m: number of rows
            ndt    = sttmp->ndt; // n: number of columns
            nstrtl = sttmp->nstrtl; // i: index of first row (base-1)
            nstrtt = sttmp->nstrtt; // j: index of first column (base-1)

            if (sttmp->ltmtx == 1) { // compressed
                /**/
                int batch_id = count%2;
                /**/
                kt = sttmp->kt; // rank
                lda = magma_roundup( ndt, batch_pad );
                // copy V
                h_A_array[num_batch] = &dA[total_size_a];
                magma_dsetmatrix( ndt, kt, sttmp->a1, ndt, h_A_array[num_batch], lda, queue );
                #ifdef OUTPUT_SIZES
                fprintf( fp,"1 %d %d\n",ndt,kt );
                #endif
                total_size_a += lda*kt;

                // pointer to input, zu
                h_X_array[num_batch] = &st_leafmtxp->zu_gpu[nstrtt-1];

                // pointer to output, y
                h_Y_array[num_batch] = &st_leafmtxp->zbu_gpu[0][total_size_y];
                saved_sz[batch_id][num_saved] = total_size_y;
                total_size_y += kt;

                // dimension
                h_M[num_batch] = ndt;
                h_N[num_batch] = kt;
                h_lda[num_batch] = lda;
                h_inc[num_batch] = 1;
                num_batch ++;

                saved_ip[batch_id][num_saved] = ip;
                saved_bt[batch_id][num_saved] = k;

                num_saved ++;
            } else if(sttmp->ltmtx == 2) { // full
                // copy matrix
                lda = magma_roundup( ndt, batch_pad );
                h_A_array[num_batch] = &dA[total_size_a];
                #ifdef OUTPUT_SIZES
                fprintf( fp,"3 %d %d\n",ndt,ndl );
                #endif
                magma_dsetmatrix( ndt, ndl, sttmp->a1, ndt, h_A_array[num_batch], lda, queue );
                total_size_a += lda*ndl;

                // pointer to input, zu
                h_X_array[num_batch] = &st_leafmtxp->zu_gpu[nstrtt-1];

                // pointer to output, y
                #if defined(BATCH_IN_PLACE_Y) //& defined(MAGMA_BATCH_DGEMV_ATOMIC)
                h_Y_array[num_batch] = &st_leafmtxp->zau_gpu[0][nstrtl-1];
                #else
                h_Y_array[num_batch] = &st_leafmtxp->zbu_gpu[0][total_size_y];
                total_size_y += ndl;
                #endif

                // dimension
                h_M[num_batch] = ndt;
                h_N[num_batch] = ndl;
                h_lda[num_batch] = lda;
                h_inc[num_batch] = 1;
                num_batch ++;
            }
        }
        // extra space for M and N with batched
        h_A_array[num_batch] = NULL;
        num_batch ++;
        count ++;
    }
    #ifdef OUTPUT_SIZES
    fclose(fp);
    #endif
    magma_malloc((void**)&(st_leafmtxp->d_A_array), num_batch*sizeof(double*));
    magma_malloc((void**)&(st_leafmtxp->d_X_array), num_batch*sizeof(double*));
    magma_malloc((void**)&(st_leafmtxp->d_Y_array), num_batch*sizeof(double*));
    magma_setvector(num_batch, sizeof(double*), h_A_array, 1, st_leafmtxp->d_A_array, 1, queue );
    magma_setvector(num_batch, sizeof(double*), h_X_array, 1, st_leafmtxp->d_X_array, 1, queue );
    magma_setvector(num_batch, sizeof(double*), h_Y_array, 1, st_leafmtxp->d_Y_array, 1, queue );

    magma_imalloc(&st_leafmtxp->d_M, num_batch+1);
    magma_imalloc(&st_leafmtxp->d_N, num_batch+1);
    magma_imalloc(&st_leafmtxp->d_lda, num_batch+1);
    magma_imalloc(&st_leafmtxp->d_inc, num_batch+1);
    magma_setvector(num_batch, sizeof(magma_int_t), h_M, 1, st_leafmtxp->d_M, 1, queue );
    magma_setvector(num_batch, sizeof(magma_int_t), h_N, 1, st_leafmtxp->d_N, 1, queue );
    magma_setvector(num_batch, sizeof(magma_int_t), h_lda, 1, st_leafmtxp->d_lda, 1, queue );
    magma_setvector(num_batch, sizeof(magma_int_t), h_inc, 1, st_leafmtxp->d_inc, 1, queue );

    magma_queue_destroy( queue );

    st_leafmtxp->h_M = h_M;
    st_leafmtxp->h_N = h_N;
    st_leafmtxp->h_lda = h_lda;
    st_leafmtxp->h_A_array = h_A_array;
    st_leafmtxp->h_X_array = h_X_array;
    st_leafmtxp->h_Y_array = h_Y_array;

    magma_free_cpu(h_inc);
    free(saved_ip[0]);
    free(saved_ip[1]);
    free(saved_bt[0]);
    free(saved_bt[1]);
    free(saved_sz[0]);
    free(saved_sz[1]);

    // no maping to sort
    st_leafmtxp->num_batch = 0;
    st_leafmtxp->batch_order = NULL;
}


/////////////////////////////////////////////////////
// delete GPU memory
void  c_hacapk_adot_body_lfdel_batch_(stc_HACApK_leafmtxp *st_leafmtxp) {
    if (st_leafmtxp->max_block > 0) {
        magma_free(st_leafmtxp->zbu_gpu[0]);
        free(st_leafmtxp->zbu_gpu);
    }
    if (st_leafmtxp->m > 0) {
        magma_free(st_leafmtxp->zau_gpu[0]);
        free(st_leafmtxp->zau_gpu);
    }
    if (st_leafmtxp->n > 0) {
        magma_free(st_leafmtxp->zu_gpu);
    }
    magma_free(st_leafmtxp->h_A_array[0]);

    magma_free(st_leafmtxp->d_A_array);
    magma_free(st_leafmtxp->d_X_array);
    magma_free(st_leafmtxp->d_Y_array);
    magma_free(st_leafmtxp->d_M);
    magma_free(st_leafmtxp->d_N);
    magma_free(st_leafmtxp->d_lda);
    magma_free(st_leafmtxp->d_inc);

    magma_free_cpu(st_leafmtxp->h_M);
    magma_free_cpu(st_leafmtxp->h_N);
    magma_free_cpu(st_leafmtxp->h_lda);
    magma_free_cpu(st_leafmtxp->h_A_array);
    magma_free_cpu(st_leafmtxp->h_X_array);
    magma_free_cpu(st_leafmtxp->h_Y_array);
    if (st_leafmtxp->num_streamed > 0) {
        magma_free_cpu(st_leafmtxp->h_M_streamed);
        magma_free_cpu(st_leafmtxp->h_N_streamed);
        magma_free_cpu(st_leafmtxp->h_lda_streamed);
        magma_free_cpu(st_leafmtxp->h_A_array_streamed);
        magma_free_cpu(st_leafmtxp->h_X_array_streamed);
        magma_free_cpu(st_leafmtxp->h_Y_array_streamed);
    }
    // let me finalize it here for now
    magma_finalize();
}

// sort blocks for batched kernel to utilize GPU better
#define sort_array_size 4
#define sort_group_size 8

#ifdef SORT_BATCH_BY_SIZES
int hacapk_size_sorter(const void* arg1,const void* arg2) {
  const int *val1 = (const int*)arg1;
  const int *val2 = (const int*)arg2;

  #define BY_GROUP
  #if defined(BY_GROUP)
  // sort by n "group", whithin group, sort by m
  return (val2[3] == val1[3] ? (val2[1] < val1[1]) : val2[3] < val1[3]);
  #elif defined(BY_N)
  // sort by n
  return (val2[2] < val1[2]);
  #else
  // sort by m
  return (val2[1] < val1[1]);
  #endif
}

int hacapk_size_sorter_trans(const void* arg1,const void* arg2) {
  const int *val1 = (const int*)arg1;
  const int *val2 = (const int*)arg2;

  #if defined(BY_N)
    #if defined(BY_GROUP)
    // sort by "group", whithin group, sort by m
    const int id1 = (val1[1]-1)/sort_group_size;
    const int id2 = (val2[1]-1)/sort_group_size;
    return (id1 == id2 ? (val2[1] < val1[1]) : id2 < id1);
    #else
    // sort by m
    return (val2[1] < val1[1]);
    #endif
  #else
    #if defined(BY_GROUP)
    // sort by "group", whithin group, sort by m
    const int id1 = (val1[2]-1)/sort_group_size;
    const int id2 = (val2[2]-1)/sort_group_size;
    return (id1 == id2 ? (val2[2] < val1[2]) : id2 < id1);
    #else
    // sort by n
    return (val2[2] < val1[2]);
    #endif
  #endif
}

static void hacapk_sort(int n, int *sizes) {
  int igap, i, j, k;
  int temp;
  igap = n / 2;
  while (igap > 0) {
    for (i = igap; i < n; i++) {
        j = i - igap;
        while (j >= 0) {
            if (sizes[j*sort_array_size] > sizes[(j + igap)*sort_array_size]) {
                for (k=0; k<sort_array_size; k++) {
                    temp = sizes[j*sort_array_size + k];
                    sizes[j*sort_array_size + k] = sizes[(j+igap)*sort_array_size + k];
                    sizes[(j+igap)*sort_array_size + k] = temp;
                }
                j = j - igap;
            } else {
                break;
            }
        }
    }
    igap = igap / 2;
  }
}
#endif

void c_hacapk_adot_body_lfcpy_batch_sorted_(int *nd, stc_HACApK_leafmtxp *st_leafmtxp) {

    // local variables
    int ip, i, j, k;
    int nlf, ndl, ndt, nstrtl, nstrtt, kt, lda;
    int st_lf_stride = st_leafmtxp->st_lf_stride;

    // let me initialize here for now..
    magma_init();
    //st_leafmtxp->mpi_comm = MPI_COMM_WORLD; // comm world for now
    MPI_Comm_rank(MPI_COMM_WORLD, &(st_leafmtxp->mpi_rank));
    if (st_leafmtxp->mpi_rank == 0) magma_print_environment();

    // allocate queue
    magma_device_t cdev;
    magma_queue_t queue = NULL;
    magma_setdevice( get_device_id(st_leafmtxp) );
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );
    int name_len;
    char proc_name[300];
    MPI_Get_processor_name( proc_name, &name_len );
    printf( " processor %d uses %d GPU on %s\n",st_leafmtxp->mpi_rank,(st_leafmtxp->mpi_rank)%gpus_per_proc,proc_name);

    // number of blocks
    nlf = st_leafmtxp->nlf; 
    int *saved_sz; 
    saved_sz = (int*)malloc( nlf * sizeof(int) ); 

    // initialize data structure
    st_leafmtxp->gn = *nd;
    st_leafmtxp->m = 0;
    st_leafmtxp->n = 0;
    st_leafmtxp->max_block = 0;
    // do GEMV(NoTrans/Trans)
    st_leafmtxp->transA = MagmaNoTrans;

    int num_batch = 0;
    int total_size_a = 0;
    int total_size_y = 0;
    for (ip = 0; ip < nlf; ip++) {
        /**/
        stc_HACApK_leafmtx *sttmp;
        sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
        /**/

        kt     = sttmp->kt;  // rank
        ndl    = sttmp->ndl; // m: number of rows
        ndt    = sttmp->ndt; // n: number of columns
        nstrtl = sttmp->nstrtl; // i: index of first row (base-1)
        nstrtt = sttmp->nstrtt; // j: index of first column (base-1)

        // local matrix size
        if (nstrtl == nstrtt) {
            st_leafmtxp->m += ndl;
        }
        if (st_leafmtxp->max_block < max(ndl, ndt)) {
            st_leafmtxp->max_block = max(ndl, ndt);
        }

        if (sttmp->ltmtx == 1) { // compressed
            if (st_leafmtxp->transA == MagmaTrans) {
                lda = magma_roundup( ndt, batch_pad );
                total_size_a += lda*kt;
                lda = magma_roundup( kt, batch_pad );
                total_size_a += lda*ndl;
            } else {
                lda = magma_roundup( kt, batch_pad );
                total_size_a += lda*ndt;
                lda = magma_roundup( ndl, batch_pad );
                total_size_a += lda*kt;
            }

            total_size_y += kt;
            num_batch += 2;
        } else { // full
            if (st_leafmtxp->transA == MagmaTrans) {
                lda = magma_roundup( ndt, batch_pad );
                total_size_a += lda*ndl;
            } else {
                lda = magma_roundup( ndl, batch_pad );
                total_size_a += lda*ndt;
            }
            num_batch += 1;
        }
    }
    // let's just use global size.
    st_leafmtxp->m = st_leafmtxp->gn;
    st_leafmtxp->n = st_leafmtxp->gn;
    fflush(stdout);
    if (st_leafmtxp->mpi_rank == 0) {
        printf( " %d-by-%d matrix (# blocks=%d)\n",st_leafmtxp->m,st_leafmtxp->n,nlf );
        printf( "  total_size_y=%d, total_size_a=%d\n",total_size_y,total_size_a );
        if (st_leafmtxp->transA == MagmaTrans) {
            printf( "  > batched GEMV with Transposes\n" );
        } else {
            printf( "  > batched GEMV with Non Transposes\n" );
        }
    }
    fflush(stdout);

    // workspace for GEMV on GPU
    st_leafmtxp->zu_gpu = NULL; 
    st_leafmtxp->zau_gpu = NULL;
    st_leafmtxp->zbu_gpu = NULL;
    if (st_leafmtxp->m > 0) {
        st_leafmtxp->zau_gpu = (double**)malloc( sizeof(double*) );
        int retval = magma_malloc( (void**) &st_leafmtxp->zau_gpu[0], (st_leafmtxp->m)*sizeof(double) );
        if ( MAGMA_SUCCESS != retval ) {
            fprintf( stderr, "!!!! magma_malloc failed for zau_gpu (m=%d)\n",st_leafmtxp->m);
            exit(0);
        }
    }
    if (st_leafmtxp->n > 0) {
        int retval = magma_malloc( (void**) &st_leafmtxp->zu_gpu, (st_leafmtxp->gn)*sizeof(double) );
        if ( MAGMA_SUCCESS != retval ) {
            fprintf( stderr, "!!!! magma_malloc failed for zu_gpu\n");
            exit(0);
        }
    }
    if (total_size_y > 0) {
        st_leafmtxp->zbu_gpu = (double**)malloc( sizeof(double*) );
        int retval = magma_malloc( (void**) &st_leafmtxp->zbu_gpu[0], total_size_y*sizeof(double) );
        if ( MAGMA_SUCCESS != retval ) {
            fprintf( stderr, "!!!! magma_malloc failed for zbu_gpu\n");
            exit(0);
        }
        st_leafmtxp->total_size_y = total_size_y;
    }
    double *dA = NULL;
    if (total_size_a > 0) {
        int retval = magma_malloc( (void**) &dA, total_size_a*sizeof(double) );
        if ( MAGMA_SUCCESS != retval ) {
            fprintf( stderr, "!!!! magma_malloc failed for dA(%d)\n",total_size_a);
            exit(0);
        }
    }

    // extra space for M and N with batch
    int count = (nlf+batch_count-1)/batch_count;
    count += ((num_batch-nlf)+batch_count-1)/batch_count;

    st_leafmtxp->num_batch = num_batch;
    num_batch += 2+count;

    double **h_A_array, **h_X_array, **h_Y_array;
    magma_malloc_cpu((void**)&(h_A_array), num_batch*sizeof(double*));
    magma_malloc_cpu((void**)&(h_X_array), num_batch*sizeof(double*));
    magma_malloc_cpu((void**)&(h_Y_array), num_batch*sizeof(double*));

    magma_int_t *h_type, *h_M, *h_N, *h_I, *h_J, *h_lda, *h_inc;
    magma_imalloc_cpu(&h_M, num_batch);
    magma_imalloc_cpu(&h_N, num_batch);
    magma_imalloc_cpu(&h_I, num_batch);
    magma_imalloc_cpu(&h_J, num_batch);
    magma_imalloc_cpu(&h_lda, num_batch);
    magma_imalloc_cpu(&h_inc, num_batch);
    magma_imalloc_cpu(&h_type, num_batch);

    magma_int_t *max_M, *max_N;
    magma_imalloc_cpu(&max_M, count);
    magma_imalloc_cpu(&max_N, count);

    // sort batch
    int num_streamed = 0;
    int num_streamed_t = 0;
    int lwork = 0;
    int *sizes = (int*)malloc( sort_array_size*num_batch * sizeof(int) );
    for (ip = 0; ip < nlf; ip++) {
        /**/
        stc_HACApK_leafmtx *sttmp;
        sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
        ndl = sttmp->ndl; // m: number of rows
        ndt = sttmp->ndt; // n: number of columns
        kt  = sttmp->kt;  // rank
        /**/

        if (sttmp->ltmtx == 1) { // compressed
            // dimension
            sizes[sort_array_size*ip + 0] = ip;
            sizes[sort_array_size*ip + 1] = ndt;
            sizes[sort_array_size*ip + 2] = kt;
            #if defined(BY_N)
            sizes[sort_array_size*ip + 3] = (kt-1) / sort_group_size;
            #else
            sizes[sort_array_size*ip + 3] = (ndt-1) / sort_group_size;
            #endif
            lwork = max(lwork, ndt*kt);
            if (max(ndt, kt) > batch_max_blocksize) {
                num_streamed ++;
            }
        } else if(sttmp->ltmtx == 2) { // full
            // dimension
            sizes[sort_array_size*ip + 0] = ip;
            sizes[sort_array_size*ip + 1] = ndt;
            sizes[sort_array_size*ip + 2] = ndl;
            #if defined(BY_N)
            sizes[sort_array_size*ip + 3] = (ndl-1) / sort_group_size;
            #else
            sizes[sort_array_size*ip + 3] = (ndt-1) / sort_group_size;
            #endif
            lwork = max(lwork, ndt*ndl);
            if (max(ndt, ndl) > batch_max_blocksize) {
                num_streamed ++;
            }
        }
    }
    num_batch = nlf;
    for (ip = 0; ip < nlf; ip++) {
        /**/
        stc_HACApK_leafmtx *sttmp;
        sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
        ndl = sttmp->ndl; // m: number of rows
        ndt = sttmp->ndt; // n: number of columns
        kt  = sttmp->kt;  // rank
        /**/

        if (sttmp->ltmtx == 1) { // compressed
            // dimension
            sizes[sort_array_size*num_batch + 0] = ip;
            sizes[sort_array_size*num_batch + 1] = kt;
            sizes[sort_array_size*num_batch + 2] = ndl;
            #if defined(BY_N)
            sizes[sort_array_size*num_batch + 3] = (ndl-1) / sort_group_size;
            #else
            sizes[sort_array_size*num_batch + 3] = (kt-1) / sort_group_size;
            #endif
            num_batch ++;
            lwork = max(lwork, kt*ndl);
            if (max(kt, ndl) > batch_max_blocksize) {
                num_streamed_t ++;
                num_streamed ++;
            }
        }
    }
    if (st_leafmtxp->mpi_rank == 0) {
        printf( "\n\n ++ num_batch=%d (this include num_streamed), num_streamed=%d,%d ++\n\n",num_batch,num_streamed,num_streamed_t );
    }

    #if defined(SORT_BATCH_BY_SIZES)
    #if defined(USE_QSORT)
    qsort( sizes, nlf, sort_array_size*sizeof(int), hacapk_size_sorter );
    qsort( &sizes[sort_array_size*nlf], num_batch-nlf, sort_array_size*sizeof(int), hacapk_size_sorter_trans );
    //qsort( &sizes[sort_array_size*nlf], num_batch-nlf, sort_array_size*sizeof(int), hacapk_size_sorter );
    #else
    hacapk_sort(nlf, sizes);
    hacapk_sort(num_batch-nlf, &sizes[nlf]);
    #endif
    #endif
    st_leafmtxp->batch_order = (int*)malloc(num_batch * sizeof(int));
    #ifdef OUTPUT_SIZES
    FILE *fp;
    char filename[100];
    sprintf(filename,"sizes_sorted_%d.dat",st_leafmtxp->mpi_rank);
    fp = fopen(filename,"w");
    fprintf(fp, "%d\n",num_batch);
    #endif
    for (ip = 0; ip < num_batch; ip++) {
        st_leafmtxp->batch_order[ip] = sizes[sort_array_size*ip + 0];
        #ifdef OUTPUT_SIZES
        fprintf(fp, "%d %d %d\n",sizes[sort_array_size*ip + 0],sizes[sort_array_size*ip + 1],sizes[sort_array_size*ip + 2]);
        #endif
    }
    #ifdef OUTPUT_SIZES
    fclose(fp);
    #endif
    free(sizes);

    // space for streamed GEMV
    double **h_A_array_streamed, **h_X_array_streamed, **h_Y_array_streamed;
    magma_int_t *h_M_streamed, *h_N_streamed, *h_lda_streamed;
    if (num_streamed > 0) {
        magma_malloc_cpu((void**)&(h_A_array_streamed), num_streamed*sizeof(double*));
        magma_malloc_cpu((void**)&(h_X_array_streamed), num_streamed*sizeof(double*));
        magma_malloc_cpu((void**)&(h_Y_array_streamed), num_streamed*sizeof(double*));

        magma_imalloc_cpu(&h_M_streamed, num_streamed);
        magma_imalloc_cpu(&h_N_streamed, num_streamed);
        magma_imalloc_cpu(&h_lda_streamed, num_streamed);
    }
    num_streamed = 0;

    int tp;
    //#define PAD_FOR_FIXED_SIZE_BATCH
    #ifdef PAD_FOR_FIXED_SIZE_BATCH
    // find maxes for fixed-size batch
    int Max_M = 0, Max_N = 0;
    int max_count = count, count_ = 0, count_comp = 0;
    total_size_a = 0;
    total_size_y = 0;
    count = 0; max_M[0] = 0; max_N[0] = 0;
    for (tp = 0; tp < nlf; tp++) {
        ip = st_leafmtxp->batch_order[tp];
        /**/
        stc_HACApK_leafmtx *sttmp;
        sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
        ndl = sttmp->ndl; // m: number of rows
        ndt = sttmp->ndt; // n: number of columns
        kt  = sttmp->kt;  // rank
        /**/

        if (sttmp->ltmtx == 1) { // compressed
            if (max(kt, ndt) > batch_max_blocksize) {
                Max_N = max(Max_N, max_N[count]);

                max_M[count] = max(max_M[count], kt);
                max_N[count] = max(max_N[count], ndt);
                count_comp ++;
                count_ ++;
            }
        } else if(sttmp->ltmtx == 2) { // full
            if (max(ndl, ndt) > batch_max_blocksize) {
                Max_M = max(Max_M, max_M[count]);
                Max_N = max(Max_N, max_N[count]);

                max_M[count] = max(max_M[count], ndl);
                max_N[count] = max(max_N[count], ndt);
                count_ ++;
            }
        }
        if (count_ == batch_count && count+1 < max_count) {
            total_size_a += count_ * max_M[count]*max_N[count];
            total_size_y += count_comp * max_M[count];

            count_ = 0;
            count_comp = 0;
            count ++;
            max_M[count] = 0;
            max_N[count] = 0;
        }
    }
    if (count_ > 0) {
        total_size_a += count_ * max_M[count]*max_N[count];
        total_size_y += count_comp * max_M[count];

        count_ = 0;
        count_comp = 0;
        count ++;
        max_M[count] = 0;
        max_N[count] = 0;
    }
    for (tp = nlf; tp < num_batch; tp++) {
        ip = st_leafmtxp->batch_order[tp];
        /**/
        stc_HACApK_leafmtx *sttmp;
        sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
        ndl = sttmp->ndl; // m: number of rows
        ndt = sttmp->ndt; // n: number of columns
        kt  = sttmp->kt;  // rank
        /**/

        if (sttmp->ltmtx == 1) { // compressed
            if (max(ndl, kt) > batch_max_blocksize) {
                Max_M = max(Max_M, max_M[count]);

                max_M[count] = max(max_M[count], ndl);
                max_N[count] = max(max_N[count], kt);
                count_ ++;
            }
        }
        if (count_ == batch_count && count+1 < max_count) {
            total_size_a += count_ * max_M[count]*max_N[count];

            count_ = 0;
            count ++;
            max_M[count] = 0;
            max_N[count] = 0;
        }
    }
    if (count_ > 0) {
        total_size_a += count_ * max_M[count]*max_N[count];
    }
    // reallocate
    if (total_size_a > 0) {
        magma_free(dA);
        int retval = magma_malloc( (void**) &dA, total_size_a*sizeof(double) );
        if ( MAGMA_SUCCESS != retval ) {
            fprintf( stderr, "!!!! magma_malloc failed for dA(%d)\n",total_size_a);
            exit(0);
        }
    }
    if (total_size_y > 0) {
        magma_free(st_leafmtxp->zbu_gpu);
        st_leafmtxp->zbu_gpu = (double**)malloc( sizeof(double*) );
        int retval = magma_malloc( (void**) &st_leafmtxp->zbu_gpu[0], total_size_y*sizeof(double) );
        if ( MAGMA_SUCCESS != retval ) {
            fprintf( stderr, "!!!! magma_malloc failed for zbu_gpu\n");
            exit(0);
        }
        st_leafmtxp->total_size_y = total_size_y;
    }
    double zero = 0.0;
    magmablas_dlaset( MagmaFull, total_size_y, 1, zero, zero, 
                      st_leafmtxp->zbu_gpu[0], total_size_y, queue );
    if (st_leafmtxp->m > 0) {
        magma_free(st_leafmtxp->zau_gpu);
        int retval = magma_malloc( (void**) &st_leafmtxp->zau_gpu[0], (st_leafmtxp->m+Max_M)*sizeof(double) );
        if ( MAGMA_SUCCESS != retval ) {
            fprintf( stderr, "!!!! magma_malloc failed for zau_gpu\n");
            exit(0);
        }
    }
    magmablas_dlaset( MagmaFull, st_leafmtxp->m+Max_M, 1, zero, zero, 
                      st_leafmtxp->zau_gpu[0], total_size_y, queue );
    if (st_leafmtxp->n > 0) {
        magma_free(st_leafmtxp->zu_gpu);
        int retval = magma_malloc( (void**) &st_leafmtxp->zu_gpu, (st_leafmtxp->gn+Max_N)*sizeof(double) );
        if ( MAGMA_SUCCESS != retval ) {
            fprintf( stderr, "!!!! magma_malloc failed for zu_gpu\n");
            exit(0);
        }
    }
    magmablas_dlaset( MagmaFull, st_leafmtxp->gn+Max_N, 1, zero, zero, 
                      st_leafmtxp->zu_gpu[0], total_size_y, queue );
    #endif

    // parse all the blocks
    double *work = (double*)malloc(lwork * sizeof(double));
    total_size_y = 0;
    total_size_a = 0;
    num_batch = 0;
    count = 0;
    for (tp = 0; tp < st_leafmtxp->num_batch;) {
        /**/
        int tp_start = tp;
        int max_m = 0;
        int max_n = 0;
        for (k = 0; k < batch_count && tp < (tp_start < nlf ? nlf : st_leafmtxp->num_batch); tp++, k++) {
            ip = st_leafmtxp->batch_order[tp];
            /**/
            stc_HACApK_leafmtx *sttmp;
            sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
            /**/

            ndl    = sttmp->ndl; // m: number of rows
            ndt    = sttmp->ndt; // n: number of columns
            nstrtl = sttmp->nstrtl; // i: index of first row
            nstrtt = sttmp->nstrtt; // j: index of first column

            if (sttmp->ltmtx == 1) { // compressed
                kt = sttmp->kt; // rank

                if (tp < nlf) {
                    // dimension
                    h_type[num_batch] = 1;
                    h_I[num_batch] = nstrtl;
                    h_J[num_batch] = nstrtt;
                    if (st_leafmtxp->transA == MagmaTrans) {
                        #ifdef PAD_FOR_FIXED_SIZE_BATCH
                        h_M[num_batch] = max_M[count];
                        h_N[num_batch] = max_N[count];
                        #else
                        h_M[num_batch] = ndt;
                        h_N[num_batch] = kt;
                        #endif

                        max_m = max(max_m, ndt);
                        max_n = max(max_n, kt);
                    } else {
                        #ifdef PAD_FOR_FIXED_SIZE_BATCH
                        h_M[num_batch] = max_M[count];
                        h_N[num_batch] = max_N[count];
                        #else
                        h_M[num_batch] = kt;
                        h_N[num_batch] = ndt;
                        #endif

                        max_m = max(max_m, kt);
                        max_n = max(max_n, ndt);
                    }
                    lda = magma_roundup( h_M[num_batch], batch_pad );
                    #ifdef PAD_FOR_FIXED_SIZE_BATCH
                    double zero = 0.0;
                    magmablas_dlaset( MagmaFull, h_M[num_batch], h_N[num_batch], zero, zero, 
                                      &dA[total_size_a], lda, queue );
                    #endif

                    if (st_leafmtxp->transA == MagmaTrans) {
                        // copy V
                        magma_dsetmatrix( ndt, kt, sttmp->a1, ndt, &dA[total_size_a], lda, queue );
                    } else {
                        // copy V^T
                        for (i=0; i<ndt; i++) {
                            for (j=0; j<kt; j++) work[j + i*kt] = (sttmp->a1)[i + j*ndt];
                        }
                        magma_dsetmatrix( kt, ndt, work, kt, &dA[total_size_a], lda, queue );
                    }

                    if (max(ndt, kt) > batch_max_blocksize) {
                        // dimension,
                        h_M_streamed[num_streamed] = h_M[num_batch];
                        h_N_streamed[num_streamed] = h_N[num_batch];
                        //printf( " %d: compress-1(%d): %dx%d at (%d,%d)\n",ip,num_streamed,h_M_streamed[num_streamed],h_N_streamed[num_streamed],nstrtl,nstrtt);

                        // pointer to input, A
                        h_A_array_streamed[num_streamed] = &dA[total_size_a];
                        total_size_a += lda*h_N_streamed[num_streamed];

                        // pointer to input, zu
                        h_X_array_streamed[num_streamed] = &st_leafmtxp->zu_gpu[nstrtt-1];

                        // pointer to output, y
                        h_Y_array_streamed[num_streamed] = &st_leafmtxp->zbu_gpu[0][total_size_y];
                        saved_sz[ip] = total_size_y;
                        total_size_y += kt;

                        // ld and inc
                        h_lda_streamed[num_streamed] = lda;
                        num_streamed ++;
                    } else {
                        // pointer to input, A
                        h_A_array[num_batch] = &dA[total_size_a];
                        total_size_a += lda*h_N[num_batch];

                        // pointer to input, zu
                        h_X_array[num_batch] = &st_leafmtxp->zu_gpu[nstrtt-1];

                        // pointer to output, y
                        h_Y_array[num_batch] = &st_leafmtxp->zbu_gpu[0][total_size_y];
                        saved_sz[ip] = total_size_y;
                        total_size_y += kt;

                        // ld and inc
                        h_lda[num_batch] = lda;
                        h_inc[num_batch] = 1;
                        num_batch ++;
                    }
                } else {
                    /**/
                    double *a2tmp = (double *)((void*)(sttmp->a1)+sttmp->a1size);
                    /**/
                    // dimmension
                    h_type[num_batch] = 2;
                    h_I[num_batch] = nstrtl;
                    h_J[num_batch] = nstrtt;
                    if (st_leafmtxp->transA == MagmaTrans) {
                        #ifdef PAD_FOR_FIXED_SIZE_BATCHx
                        h_M[num_batch] = max_M[count];
                        h_N[num_batch] = max_N[count];
                        #else
                        h_M[num_batch] = kt;
                        h_N[num_batch] = ndl;
                        #endif

                        max_m = max(max_m, kt);
                        max_n = max(max_n, ndl);
                    } else {
                        #ifdef PAD_FOR_FIXED_SIZE_BATCHx
                        h_M[num_batch] = max_M[count];
                        h_N[num_batch] = max_N[count];
                        #else
                        h_M[num_batch] = ndl;
                        h_N[num_batch] = kt;
                        #endif

                        max_m = max(max_m, ndl);
                        max_n = max(max_n, kt);
                    }
                    lda = magma_roundup( h_M[num_batch], batch_pad );
                    #ifdef PAD_FOR_FIXED_SIZE_BATCHx
                    double zero = 0.0;
                    magmablas_dlaset( MagmaFull, h_M[num_batch], h_N[num_batch], zero, zero, 
                                      &dA[total_size_a], lda, queue );
                    #endif

                    if (st_leafmtxp->transA == MagmaTrans) {
                        // copy U^T
                        for (i=0; i<ndl; i++) {
                            for (j=0; j<kt; j++) work[j + i*kt] = a2tmp[i + j*ndl];
                        }
                        magma_dsetmatrix( kt, ndl, work, kt, &dA[total_size_a], lda, queue );
                    } else {
                        // copy U
                        magma_dsetmatrix( ndl, kt, a2tmp, ndl, &dA[total_size_a], lda, queue );
                    }

                    if (max(ndl, kt) > batch_max_blocksize) {
                        // dimension,
                        h_M_streamed[num_streamed] = h_M[num_batch];
                        h_N_streamed[num_streamed] = h_N[num_batch];
                        //printf( " %d: compress-2(%d): %dx%d at (%d,%d)\n",ip,num_streamed,h_M_streamed[num_streamed],h_N_streamed[num_streamed],nstrtl,nstrtt);

                        // pointer to input, A
                        h_A_array_streamed[num_streamed] = &dA[total_size_a];
                        total_size_a += lda*h_N_streamed[num_streamed];

                        // pointer to input, zu
                        int size_y = saved_sz[ip];
                        h_X_array_streamed[num_streamed] = &st_leafmtxp->zbu_gpu[0][size_y];

                        // pointer to output, y
                        h_Y_array_streamed[num_streamed] = &st_leafmtxp->zau_gpu[0][nstrtl-1];

                        // ld and inc
                        h_lda_streamed[num_streamed] = lda;
                        num_streamed ++;
                    } else {
                        // pointer to input, A
                        h_A_array[num_batch] = &dA[total_size_a];
                        total_size_a += lda*h_N[num_batch];

                        // pointer to input, zu
                        int size_y = saved_sz[ip];
                        h_X_array[num_batch] = &st_leafmtxp->zbu_gpu[0][size_y];

                        // pointer to output, y
                        h_Y_array[num_batch] = &st_leafmtxp->zau_gpu[0][nstrtl-1];

                        // ld and inc
                        h_lda[num_batch] = lda;
                        h_inc[num_batch] = 1;
                        num_batch ++;
                    }
                }
            } else if(sttmp->ltmtx == 2) { // full
                // dimension
                h_type[num_batch] = 0;
                h_I[num_batch] = nstrtl;
                h_J[num_batch] = nstrtt;
                if (st_leafmtxp->transA == MagmaTrans) {
                    h_M[num_batch] = ndt;
                    h_N[num_batch] = ndl;

                    max_m = max(max_m, ndt);
                    max_n = max(max_n, ndl);
                } else {
                    h_M[num_batch] = ndl;
                    h_N[num_batch] = ndt;

                    max_m = max(max_m, ndl);
                    max_n = max(max_n, ndt);
                }
                lda = magma_roundup( h_M[num_batch], batch_pad );

                // copy matrix
                if (st_leafmtxp->transA == MagmaTrans) {
                    magma_dsetmatrix( ndt, ndl, sttmp->a1, ndt, &dA[total_size_a], lda, queue );
                } else {
                    for (i=0; i<ndt; i++) {
                        for (j=0; j<ndl; j++) work[j + i*ndl] = (sttmp->a1)[i + j*ndt];
                    }
                    magma_dsetmatrix( ndl, ndt, work, ndl, &dA[total_size_a], lda, queue );
                }

                if (max(ndt, ndl) > batch_max_blocksize) {
                    // dimension,
                    h_M_streamed[num_streamed] = h_M[num_batch];
                    h_N_streamed[num_streamed] = h_N[num_batch];
                    //printf( " full-stream(%d): %dx%d\n",num_streamed,h_M_streamed[num_streamed],h_N_streamed[num_streamed]);

                    // pointer to input, A
                    h_A_array_streamed[num_streamed] = &dA[total_size_a];
                    total_size_a += lda*h_N_streamed[num_streamed];

                    // pointer to input, zu
                    h_X_array_streamed[num_streamed] = &st_leafmtxp->zu_gpu[nstrtt-1];

                    // pointer to output, y
                    h_Y_array_streamed[num_streamed] = &st_leafmtxp->zau_gpu[0][nstrtl-1];

                    // ld and inc
                    h_lda_streamed[num_streamed] = lda;
                    num_streamed ++;
                } else {
                    // pointer to input, A
                    h_A_array[num_batch] = &dA[total_size_a];
                    total_size_a += lda*h_N[num_batch];

                    // pointer to input, zu
                    h_X_array[num_batch] = &st_leafmtxp->zu_gpu[nstrtt-1];

                    // pointer to output, y
                    h_Y_array[num_batch] = &st_leafmtxp->zau_gpu[0][nstrtl-1];

                    // ld and inc
                    h_lda[num_batch] = lda;
                    h_inc[num_batch] = 1;
                    num_batch ++;
                }
            }
        }
        #ifdef PAD_FOR_FIXED_SIZE_BATCH
        if (max_M[count] != max_m) printf( " max_M[%d]=%d, max_m=%d\n",count,max_M[count],max_m );
        if (max_N[count] != max_n) printf( " max_N[%d]=%d, max_n=%d\n",count,max_N[count],max_n );
        #endif
        max_M[count] = h_M[num_batch] = max_m;
        max_N[count] = h_N[num_batch] = max_n;
        count ++;

        // extra space for M and N with batched
        h_A_array[num_batch] = NULL;
        num_batch ++;
    }
    free(work);

    magma_malloc((void**)&(st_leafmtxp->d_A_array), num_batch*sizeof(double*));
    magma_malloc((void**)&(st_leafmtxp->d_X_array), num_batch*sizeof(double*));
    magma_malloc((void**)&(st_leafmtxp->d_Y_array), num_batch*sizeof(double*));
    magma_setvector(num_batch, sizeof(double*), h_A_array, 1, st_leafmtxp->d_A_array, 1, queue );
    magma_setvector(num_batch, sizeof(double*), h_X_array, 1, st_leafmtxp->d_X_array, 1, queue );
    magma_setvector(num_batch, sizeof(double*), h_Y_array, 1, st_leafmtxp->d_Y_array, 1, queue );

    magma_imalloc(&st_leafmtxp->d_M, num_batch+1);
    magma_imalloc(&st_leafmtxp->d_N, num_batch+1);
    magma_imalloc(&st_leafmtxp->d_lda, num_batch+1);
    magma_imalloc(&st_leafmtxp->d_inc, num_batch+1);
    magma_setvector(num_batch, sizeof(magma_int_t), h_M, 1, st_leafmtxp->d_M, 1, queue );
    magma_setvector(num_batch, sizeof(magma_int_t), h_N, 1, st_leafmtxp->d_N, 1, queue );
    magma_setvector(num_batch, sizeof(magma_int_t), h_lda, 1, st_leafmtxp->d_lda, 1, queue );
    magma_setvector(num_batch, sizeof(magma_int_t), h_inc, 1, st_leafmtxp->d_inc, 1, queue );

    magma_queue_sync( queue );
    magma_queue_destroy( queue );

    st_leafmtxp->h_type = h_type;
    st_leafmtxp->h_I = h_I;
    st_leafmtxp->h_J = h_J;
    st_leafmtxp->h_M = h_M;
    st_leafmtxp->h_N = h_N;
    st_leafmtxp->h_lda = h_lda;
    st_leafmtxp->h_A_array = h_A_array;
    st_leafmtxp->h_X_array = h_X_array;
    st_leafmtxp->h_Y_array = h_Y_array;

    st_leafmtxp->max_M = max_M;
    st_leafmtxp->max_N = max_N;

    // streamed GEMV
    st_leafmtxp->num_streamed = num_streamed-num_streamed_t;
    st_leafmtxp->num_streamed_t = num_streamed_t;
    st_leafmtxp->h_M_streamed = h_M_streamed;
    st_leafmtxp->h_N_streamed = h_N_streamed;
    st_leafmtxp->h_lda_streamed = h_lda_streamed;
    st_leafmtxp->h_A_array_streamed = h_A_array_streamed;
    st_leafmtxp->h_X_array_streamed = h_X_array_streamed;
    st_leafmtxp->h_Y_array_streamed = h_Y_array_streamed;

    magma_free_cpu(h_inc);
    free(saved_sz);
}

// !! BiCG in C !!
double c_hacapk_dotp_d(int nd, double *b, double *a) {
    double norm = 0.0;
    int ii;
    for (ii=0; ii<nd; ii++) norm += b[ii]*a[ii];
    return norm;
}

void c_hacapk_adot_cax_lfmtx_comm(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, stc_HACApK_lcontrol *st_ctl,
                                  double *wws, double *wwr, int *isct, int *irct, int nd, double *time_mpi) {
   int ione = 1;
   double one = 1.0;

   double tic;
   int *lpmd = st_ctl->lpmd; 
   int *lnp = st_ctl->lnp;
   int *lsp = st_ctl->lsp;
   int mpinr = lpmd[2]; 
   int nrank = lpmd[1]; 
   MPI_Comm icomm = MPI_COMM_WORLD;
   if (nrank > 1) {
       int ncdp = (mpinr+1)%nrank;
       int ncsp = (mpinr+nrank-1)%nrank;
       isct[0] = lnp[mpinr];
       isct[1] = lsp[mpinr];

       dlacpy_( "F", &lnp[mpinr], &ione, &zau[lsp[mpinr]], &lnp[mpinr], wws, &lnp[mpinr] );

       int ic;
       for (ic=1; ic<nrank; ic++) {
           MPI_Status stat;

           tic = MPI_Wtime();
           MPI_Sendrecv(isct, 2, MPI_INT, ncdp, 1,
                        irct, 2, MPI_INT, ncsp, 1, icomm, &stat);
           MPI_Sendrecv(wws, isct[0], MPI_DOUBLE, ncdp, 1,
                        wwr, irct[0], MPI_DOUBLE, ncsp, 1, icomm, &stat);
           *time_mpi += (MPI_Wtime()-tic);
           daxpy_( &irct[0], &one, wwr, &ione, &zau[irct[1]], &ione );
           dlacpy_( "F", &irct[0], &ione, wwr, &irct[0], wws, &irct[0] );
           isct[0] = irct[0];
           isct[1] = irct[1];
       }
    }
}
void c_hacapk_bicgstab_cax_lfmtx_flat_(stc_HACApK_leafmtxp *st_leafmtxp, stc_HACApK_lcontrol *st_ctl,
                                       double *u, double *b, double*param, int nd, int nstp, int lrtrn) {
    // local constants
    int ione = 1;
    double zero =  0.0;
    double one  =  1.0;
    double mone = -1.0;
    // local arrays
    double *zr, *zshdw, *zp, *zt, *zkp, *zakp, *zkt, *zakt;
    double *wws, *wwr;
    int *lpmd;
    int isct[2], irct[2];
    // local variables
    double eps, alpha, beta, zeta, zz, zden, znorm, znormold, bnorm, zrnorm;
    double en_measure_time, st_measure_time, time;
    int step, mstep;
    int mpinr, nrank, icomm, ierr;
    double time_spmv, time_mpi, time_batch, time_set, time_copy, tic;
 
    lpmd = st_ctl->lpmd;
    mpinr = lpmd[2]; 
    nrank = lpmd[1]; 
    icomm = MPI_COMM_WORLD; //lpmd[0];
    MPI_Barrier( icomm );
    mstep = param[82];
    eps = param[90];
    wws = (double*)malloc(nd * sizeof(double));
    wwr = (double*)malloc(nd * sizeof(double));

    zt = (double*)malloc(nd * sizeof(double));
    zr = (double*)malloc(nd * sizeof(double));
    zp = (double*)malloc(nd * sizeof(double));
    zkp = (double*)malloc(nd * sizeof(double));
    zakp = (double*)malloc(nd * sizeof(double));
    zkt = (double*)malloc(nd * sizeof(double));
    zakt= (double*)malloc(nd * sizeof(double));
    zshdw = (double*)malloc(nd * sizeof(double));
    alpha = 0.0; beta = 0.0;  zeta = 0.0;
    zz=c_hacapk_dotp_d(nd, b, b); 
    bnorm=sqrt(zz);
    dlaset_( "F", &nd, &ione, &zero, &zero, zp, &nd );
    dlaset_( "F", &nd, &ione, &zero, &zero, zp, &nd );
    dlaset_( "F", &nd, &ione, &zero, &zero, zakp, &nd );
    dlacpy_( "F", &nd, &ione, b, &nd, zr, &nd );
    tic = MPI_Wtime();
    dlacpy_( "F", &nd, &ione, &zero, &zero, zshdw, &nd );
    c_hacapk_adot_body_lfmtx_batch_(zshdw,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy);
    time_spmv += (MPI_Wtime()-tic);
    c_hacapk_adot_cax_lfmtx_comm(zshdw, st_leafmtxp, st_ctl, wws, wwr, isct, irct, nd, &time_mpi);
    daxpy_( &nd, &mone, zshdw, &ione, zr, &ione );
    dlacpy_( "F", &nd, &ione, zr, &nd, zshdw, &nd );
    zrnorm = c_hacapk_dotp_d(nd, zr, zr); 
    zrnorm = sqrt(zrnorm);
    if (mpinr == 0) {
        printf( "\n ** BICG with MAGMA batched **\n" );
        printf( "\nOriginal relative residual norm = %.2e\n",zrnorm/bnorm );
        printf( "HACApK_bicgstab_lfmtx_flat start\n" );
    }
    // copy matrix to GPU
    c_hacapk_adot_body_lfcpy_batch_sorted_(&nd, st_leafmtxp);
    time_spmv = 0.0;
    time_mpi = 0.0;
    time_batch = 0.0;
    time_set = 0.0;
    time_copy = 0.0;
    MPI_Barrier( icomm );
    st_measure_time = MPI_Wtime();
    for ( step=1; step<=mstep; step++ ) {
        if (zrnorm/bnorm < eps) break;
        // zp(:nd) = zr(:nd) + beta*(zp(:nd) - zeta*zakp(:nd))
        zeta = -zeta;
        daxpy_( &nd, &zeta, zakp, &ione, zp, &ione );
        dlascl_( "F", &ione, &ione, &beta, &one, &nd, &ione, zp, &nd );
        daxpy_( &nd, &one, zr, &ione, zp, &ione );
        //
        dlacpy_( "F", &nd, &ione, zp, &nd, zkp, &nd );
        //  .. SpMV ..
        dlaset_( "F", &nd, &ione, &zero, &zero, zakp, &nd );
        tic = MPI_Wtime();
        c_hacapk_adot_body_lfmtx_batch_(zakp,st_leafmtxp,zkp,wws, &time_batch,&time_set,&time_copy);
        time_spmv += (MPI_Wtime()-tic);
        c_hacapk_adot_cax_lfmtx_comm(zakp,st_leafmtxp,st_ctl,wws,wwr,isct,irct,nd, &time_mpi);
        //
        znorm = c_hacapk_dotp_d(nd, zshdw, zr); 
        zden = c_hacapk_dotp_d(nd, zshdw, zakp);
        alpha = -znorm/zden;
        znormold = znorm;
        dlacpy_( "F", &nd, &ione, zr, &nd, zt, &nd );
        daxpy_( &nd, &one, zakp, &ione, zt, &ione );
        alpha = -alpha;
        dlacpy_( "F", &nd, &ione, zt, &nd, zkt, &nd );
        //  .. SpMV ..
        dlaset_( "F", &nd, &ione, &zero, &zero, zakt, &nd );
        tic = MPI_Wtime();
        c_hacapk_adot_body_lfmtx_batch_(zakt,st_leafmtxp,zkt,wws, &time_batch,&time_set,&time_copy);
        time_spmv += (MPI_Wtime()-tic);
        c_hacapk_adot_cax_lfmtx_comm(zakt,st_leafmtxp,st_ctl,wws,wwr,isct,irct,nd, &time_mpi);
        //
        znorm = c_hacapk_dotp_d(nd, zakt, zt); 
        zden = c_hacapk_dotp_d( nd, zakt, zakt);
        zeta = znorm/zden;
        // u(:nd) = u(:nd) + alpha*zkp(:nd) + zeta*zkt(:nd)
        daxpy_( &nd, &alpha, zkp, &ione, u, &ione );
        daxpy_( &nd, &zeta,  zkt, &ione, u, &ione );
        //
        zeta = -zeta;
        dlacpy_( "F", &nd, &ione, zr, &nd, zt, &nd );
        daxpy_( &nd, &zeta, zakt, &ione, zt, &ione );
        beta = c_hacapk_dotp_d(nd, zshdw, zr);
        beta = alpha/zeta * beta/znormold;
        zrnorm = c_hacapk_dotp_d(nd, zr, zr);
        zrnorm = sqrt(zrnorm);
        nstp = step;
        en_measure_time = MPI_Wtime();
        time = en_measure_time - st_measure_time;
        if (st_ctl->param[0] > 0 && mpinr==0) {
            printf( " %d: time=%.2e log10(zrnorm/bnorm)=%.2e\n",step,time,log10(zrnorm/bnorm) );
        }
    }
    MPI_Barrier( icomm );
    en_measure_time = MPI_Wtime();
    time = en_measure_time - st_measure_time;
    // delete matrix
    c_hacapk_adot_body_lfdel_batch_(st_leafmtxp);
    if (st_ctl->param[0]>0) {
        printf( " End: %d, %.2e\n",mpinr,time );
        if (mpinr==0) {
            printf( "       BiCG        = %.2e\n", time );
            printf( "        time_mpi   = %.2e\n", time_mpi );
            printf( "        time_spmv  = %.2e\n", time_spmv );
            printf( "        > time_copy  = %.2e\n", time_copy );
            printf( "        > time_set   = %.2e\n", time_set );
            printf( "        > time_batch = %.2e\n", time_batch );
        }
    }
    free(wws);
    free(wwr);

    free(zt);
    free(zr);
    free(zp);
    free(zkp);
    free(zakp);
    free(zkt);
    free(zakt);
    free(zshdw);
}
/////////////////////////////////////////////////////
//
#endif
