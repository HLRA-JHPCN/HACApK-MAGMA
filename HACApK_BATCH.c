
#if defined(HAVE_MAGMA) | defined(HAVE_MAGMA_BATCH)

#include "HACApK_MAGMA.h"

// /////////////////////////////////////////////////////////////////////////
// DGEMV using MAGMA
// > using CuBLAS DGEMV (out-of-place/in-place)
// > using MAGMA batched DGEMV (out-of-place/in-pace)
// > using MAGMA batched DGEMV, sorted (in-place)
// /////////////////////////////////////////////////////////////////////////
#if defined(HAVE_MAGMA_BATCH)

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


/////////////////////////////////////////////////////
// MatVec with batched GEMV

int c_hacapk_adot_body_lfmtx_batch_dgemv(stc_HACApK_leafmtxp *st_leafmtxp, int *saved_ip[2],
                                         int *ip_start, int num_batch, int count,
                                         int *batchCount, int *num_saved,
                                         double * zau_batch, magma_queue_t queue);
int c_hacapk_adot_body_lfmtx_batch_daxpy(stc_HACApK_leafmtxp *st_leafmtxp, int *saved_ip[2],
                                         int num_batch, int count, int k_start, int ip_start,
                                         double* zau_batch, double* zau, magma_queue_t queue);

void c_hacapk_adot_body_lfmtx_batch_queue(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu,
                                          double *time_batch, double *time_set, double *time_copy, int on_gpu,
                                          magma_queue_t queue) {
    // constants
    double zero = 0.0;

    int ip;
    int nlf = st_leafmtxp->nlf;
    int *saved_ip[2]; 
    if (st_leafmtxp->batch_order == NULL) {
        saved_ip[0] = (int*)malloc( nlf * sizeof(int) ); 
        saved_ip[1] = (int*)malloc( nlf * sizeof(int) ); 
    }

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
        double tic;
        #endif
        int num_saved = 0, count = 0;
        //if ( magma_is_devptr( zu ) == 1)
        if (on_gpu == 1) {
            // vectors are on GPU
            #ifdef PROF_MAGMA_BATCH
            tic = MPI_Wtime();
            #endif
            // input
            magmablas_dlacpy( MagmaFull, st_leafmtxp->gn, 1, zu, st_leafmtxp->gn, 
                              st_leafmtxp->zu_gpu, st_leafmtxp->gn, queue );
            // output
            //magmablas_dlacpy( MagmaFull, st_leafmtxp->m, 1, zau, st_leafmtxp->m, 
            //                  st_leafmtxp->zau_gpu[0], st_leafmtxp->m, queue );
            magmablas_dlaset( MagmaFull, st_leafmtxp->m, 1, zero, zero, 
                              st_leafmtxp->zau_gpu[0], st_leafmtxp->m, queue );
            #ifdef PROF_MAGMA_BATCH
            magma_queue_sync( queue );
            *time_set += (MPI_Wtime()-tic);
            #endif
        } else {
            // vectors are on CPU
            #ifdef PROF_MAGMA_BATCH
            tic = MPI_Wtime();
            #endif
            magma_dsetvector( st_leafmtxp->gn,  zu, 1, st_leafmtxp->zu_gpu,  1, queue );
            #if defined(ACCUME_ON_CPU)
            magma_dmalloc_pinned( &zau_batch,  batch_count*(st_leafmtxp->max_block) );
            #else
            magma_dsetvector( st_leafmtxp->m, zau, 1, st_leafmtxp->zau_gpu[0], 1, queue );
            #endif
            #ifdef PROF_MAGMA_BATCH
            *time_copy += (MPI_Wtime()-tic);
            #endif
        }

        // parse all the blocks
        int num_batch = 0;
        #if defined(BATCH_IN_PLACE_Y)
        #ifdef PROF_MAGMA_BATCH
        tic = MPI_Wtime();
        #endif
        // first part of low-rank, zbu := V'*zu
        magmablas_dlaset( MagmaFull, st_leafmtxp->total_size_y, 1, zero, zero, 
                          st_leafmtxp->zbu_gpu[0], st_leafmtxp->total_size_y, queue );
        #ifdef PROF_MAGMA_BATCH
        magma_queue_sync( queue );
        *time_set += (MPI_Wtime()-tic);
        tic = MPI_Wtime();
        #endif
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
        #ifdef PROF_MAGMA_BATCH
        magma_queue_sync( queue );
        *time_batch += (MPI_Wtime()-tic);
        #endif
        //if ( magma_is_devptr( zu ) == 1)
        if (on_gpu == 1) {
            // vectors are on GPU
            #ifdef PROF_MAGMA_BATCH
            tic = MPI_Wtime();
            #endif
            magmablas_dlacpy( MagmaFull, st_leafmtxp->m, 1, st_leafmtxp->zau_gpu[0], st_leafmtxp->m, 
                              zau, st_leafmtxp->m, queue );
            #ifdef PROF_MAGMA_BATCH
            magma_queue_sync( queue );
            *time_set += MPI_Wtime()-tic;
            #endif
        } else {
            #if defined(ACCUME_ON_CPU)
             magma_free_pinned(zau_batch);
            #else
            #ifdef PROF_MAGMA_BATCH
            tic = MPI_Wtime();
            #endif
            magma_dgetvector( st_leafmtxp->m, st_leafmtxp->zau_gpu[0], 1, zau, 1, queue );
            #ifdef PROF_MAGMA_BATCH
            *time_copy += MPI_Wtime()-tic;
            #endif
            #endif
        }
    }
    //#define PROF_MAGMA_BATCH_COUNT
    #ifdef PROF_MAGMA_BATCH_COUNT
    if (st_leafmtxp->mpi_rank == 0) {
        printf( " time_copy : %.2e seconds\n",  *time_copy /dgemv_count );
        printf( " time_set  : %.2e seconds\n",  *time_set  /dgemv_count );
        printf( " time_batch: %.2e seconds\n",  *time_batch/dgemv_count );
        printf( " total     : %.2e seconds\n\n",(*time_copy+*time_set+*time_batch)/dgemv_count );
    }
    fflush(stdout);
    #endif

    if (st_leafmtxp->batch_order == NULL) {
        free(saved_ip[0]);
        free(saved_ip[1]);
    }
}

void c_hacapk_adot_body_lfmtx_batch_(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu,
                                     double *time_batch, double *time_set, double *time_copy) {
    // allocate queue
    magma_device_t cdev;
    magma_queue_t queue;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    int on_gpu = 0;
    c_hacapk_adot_body_lfmtx_batch_queue(zau, st_leafmtxp, zu, zbu,
                                         time_batch, time_set, time_copy,
                                         on_gpu, queue);
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
#ifdef MAGMA_INIT_PER
    magma_init();
#endif
    //st_leafmtxp->mpi_comm = MPI_COMM_WORLD; // comm world for now
    MPI_Comm_rank(MPI_COMM_WORLD, &(st_leafmtxp->mpi_rank));
    if (st_leafmtxp->mpi_rank == 0) magma_print_environment();

    // allocate queue
    magma_device_t cdev;
    magma_queue_t queue = NULL;
    magma_setdevice( get_device_id(st_leafmtxp) );
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );
    printf( " processor %d uses %d GPU\n",st_leafmtxp->mpi_rank,(st_leafmtxp->mpi_rank)%procs_per_node);

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

    magma_free_cpu(st_leafmtxp->h_I);
    magma_free_cpu(st_leafmtxp->h_J);
    magma_free_cpu(st_leafmtxp->h_M);
    magma_free_cpu(st_leafmtxp->h_N);
    magma_free_cpu(st_leafmtxp->h_lda);
    magma_free_cpu(st_leafmtxp->h_type);
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

    magma_free_cpu(st_leafmtxp->max_N);
    magma_free_cpu(st_leafmtxp->max_M);
    free(st_leafmtxp->batch_order);
    // let me finalize it here for now
#ifdef MAGMA_INIT_PER
    magma_finalize();
#endif
}

#endif

#endif
