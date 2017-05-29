#include	<stdio.h>
#include	<stdlib.h>
#include	<time.h>
#include	"omp.h"
#include	"mpi.h"
#include	"HACApK_MAGMA.h"

// /////////////////////////////////////////////////////////////////////////
// DGEMV using MAGMA
// > using CuBLAS DGEMV (out-of-place/in-place)
// > using MAGMA batched DGEMV (out-of-place/in-pace)
// > using MAGMA batched DGEMV, sorted (in-place)
// /////////////////////////////////////////////////////////////////////////
#if defined(HAVE_MAGMA_BATCH)

// multi-GPU version 
void c_hacapk_adot_body_lfcpy_batch_sorted_mgpu_(int *nd, stc_HACApK_leafmtxp *st_leafmtxp,
                                                 magma_queue_t *queues) {

    // local variables
    int ip, i, j, k;
    int nlf, ndl, ndt, nstrtl, nstrtt, kt, lda;
    int st_lf_stride = st_leafmtxp->st_lf_stride;

    // let me initialize here for now..
    magma_init();
    //st_leafmtxp->mpi_comm = MPI_COMM_WORLD; // comm world for now
    MPI_Comm_rank(MPI_COMM_WORLD, &(st_leafmtxp->mpi_rank));
    if (st_leafmtxp->mpi_rank == 0) magma_print_environment();

    int name_len;
    char proc_name[300];
    MPI_Get_processor_name( proc_name, &name_len );
    printf( " processor %d uses %d GPU on %s\n",st_leafmtxp->mpi_rank,(st_leafmtxp->mpi_rank)%procs_per_node,proc_name);

    // number of blocks
    nlf = st_leafmtxp->nlf; 

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
    int d;
    st_leafmtxp->zu_mgpu  = (double**)malloc(gpus_per_proc * sizeof(double*)); 
    st_leafmtxp->zau_mgpu = (double**)malloc(gpus_per_proc * sizeof(double*));
    st_leafmtxp->zbu_mgpu = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **dA = (double**)malloc(gpus_per_proc * sizeof(double*));
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice( (d+get_device_id(st_leafmtxp))%procs_per_node );

        if (st_leafmtxp->m > 0) {
            int retval = magma_malloc( (void**) &st_leafmtxp->zau_mgpu[d], (st_leafmtxp->m)*sizeof(double) );
            if ( MAGMA_SUCCESS != retval ) {
                fprintf( stderr, "!!!! magma_malloc failed for zau_gpu (m=%d)\n",st_leafmtxp->m);
                exit(0);
            }
        }
        if (total_size_y > 0) {
            int retval = magma_malloc( (void**) &st_leafmtxp->zbu_mgpu[d], total_size_y*sizeof(double) );
            if ( MAGMA_SUCCESS != retval ) {
                fprintf( stderr, "!!!! magma_malloc failed for zbu_gpu\n");
                exit(0);
            }
            st_leafmtxp->total_size_y = total_size_y;
        }
        if (total_size_a > 0) {
            int retval = magma_malloc( (void**) &dA[d], total_size_a*sizeof(double) );
            if ( MAGMA_SUCCESS != retval ) {
                fprintf( stderr, "!!!! magma_malloc failed for dA(%d)\n",total_size_a);
                exit(0);
            }
        }
        if (st_leafmtxp->n > 0) {
            int retval = magma_malloc( (void**) &st_leafmtxp->zu_mgpu[d], (st_leafmtxp->gn)*sizeof(double) );
            if ( MAGMA_SUCCESS != retval ) {
                fprintf( stderr, "!!!! magma_malloc failed for zu_gpu\n");
                exit(0);
            }
        }
    }

    // extra space for M and N with batch
    int count_tot = (nlf+batch_count-1)/batch_count;
    count_tot += ((num_batch-nlf)+batch_count-1)/batch_count;

    st_leafmtxp->num_batch = num_batch;
    num_batch += 2+count_tot;

    double ***h_A_array = (double***)malloc(gpus_per_proc * sizeof(double**));
    double ***h_X_array = (double***)malloc(gpus_per_proc * sizeof(double**));
    double ***h_Y_array = (double***)malloc(gpus_per_proc * sizeof(double**));
    magma_int_t **h_M = (magma_int_t**)malloc(gpus_per_proc * sizeof(magma_int_t*));
    magma_int_t **h_N = (magma_int_t**)malloc(gpus_per_proc * sizeof(magma_int_t*));
    magma_int_t **h_I = (magma_int_t**)malloc(gpus_per_proc * sizeof(magma_int_t*));
    magma_int_t **h_J = (magma_int_t**)malloc(gpus_per_proc * sizeof(magma_int_t*));
    magma_int_t **h_lda = (magma_int_t**)malloc(gpus_per_proc * sizeof(magma_int_t*));
    magma_int_t **h_inc = (magma_int_t**)malloc(gpus_per_proc * sizeof(magma_int_t*));
    magma_int_t **max_M = (magma_int_t**)malloc(gpus_per_proc * sizeof(magma_int_t*));
    magma_int_t **max_N = (magma_int_t**)malloc(gpus_per_proc * sizeof(magma_int_t*));
    magma_int_t **h_type = (magma_int_t**)malloc(gpus_per_proc * sizeof(magma_int_t*));
    for (d=0; d<gpus_per_proc; d++) {
        magma_malloc_cpu((void**)&(h_A_array[d]), num_batch*sizeof(double*));
        magma_malloc_cpu((void**)&(h_X_array[d]), num_batch*sizeof(double*));
        magma_malloc_cpu((void**)&(h_Y_array[d]), num_batch*sizeof(double*));

        magma_imalloc_cpu(&(h_M[d]), num_batch);
        magma_imalloc_cpu(&(h_N[d]), num_batch);
        magma_imalloc_cpu(&(h_I[d]), num_batch);
        magma_imalloc_cpu(&(h_J[d]), num_batch);
        magma_imalloc_cpu(&(h_lda[d]), num_batch);
        magma_imalloc_cpu(&(h_inc[d]), num_batch);
        magma_imalloc_cpu(&(h_type[d]), num_batch);

        magma_imalloc_cpu(&(max_M[d]), count_tot);
        magma_imalloc_cpu(&(max_N[d]), count_tot);
    }

    // sort batch
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
        }
    }
    if (st_leafmtxp->mpi_rank == 0) {
        printf( "\n\n ++ num_batch=%d (nlf=%d) ++\n\n",num_batch,nlf );
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

    int tp;
    // parse all the blocks
    double *work = (double*)malloc(lwork * sizeof(double));
    int *count = (int*)malloc(gpus_per_proc * sizeof(int));
    int *max_m = (int*)malloc(gpus_per_proc * sizeof(int));
    int *max_n = (int*)malloc(gpus_per_proc * sizeof(int));
    int *owner = (int*)malloc(nlf * sizeof(int));
    int *nlf_mgpu       = (int*)malloc(gpus_per_proc * sizeof(int));
    int *num_batch_mgpu = (int*)malloc(gpus_per_proc * sizeof(int));
    int *total_size_y_mgpu = (int*)malloc(gpus_per_proc * sizeof(int));
    int *total_size_a_mgpu = (int*)malloc(gpus_per_proc * sizeof(int));
    int **saved_sz_mgpu = (int**)malloc(gpus_per_proc * sizeof(int*)); 
    for (d=0; d<gpus_per_proc; d++) {
        count[d] = 0;
        max_m[d] = max_n[d] = 0;
        num_batch_mgpu[d] = 0;
        total_size_y_mgpu[d] = 0;
        total_size_a_mgpu[d] = 0;
        saved_sz_mgpu[d] = (int*)malloc( nlf * sizeof(int) ); 
    }
    for (tp = 0; tp < st_leafmtxp->num_batch;) {
        /**/
        int tp_start = tp;
        int tp_end = (tp_start < nlf ? nlf : st_leafmtxp->num_batch);
        int tp_inc = min(gpus_per_proc*batch_count, tp_end-tp_start);

        for (k = 0; k < gpus_per_proc*batch_count && tp < tp_end; tp++, k++) {
            ip = st_leafmtxp->batch_order[tp];
            int d = (tp < nlf ? tp%gpus_per_proc : owner[ip]);
            magma_setdevice( (d+get_device_id(st_leafmtxp))%procs_per_node );
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
                    // which gpu owns this block?
                    owner[ip] = d;
                    // dimension
                    h_type[d][num_batch_mgpu[d]] = 1;
                    h_I[d][num_batch_mgpu[d]] = nstrtl;
                    h_J[d][num_batch_mgpu[d]] = nstrtt;
                    if (st_leafmtxp->transA == MagmaTrans) {
                        h_M[d][num_batch_mgpu[d]] = ndt;
                        h_N[d][num_batch_mgpu[d]] = kt;

                        max_m[d] = max(max_m[d], ndt);
                        max_n[d] = max(max_n[d], kt);
                    } else {
                        h_M[d][num_batch_mgpu[d]] = kt;
                        h_N[d][num_batch_mgpu[d]] = ndt;

                        max_m[d] = max(max_m[d], kt);
                        max_n[d] = max(max_n[d], ndt);
                    }
                    lda = magma_roundup( h_M[d][num_batch_mgpu[d]], batch_pad );

                    if (st_leafmtxp->transA == MagmaTrans) {
                        // copy V
                        magma_dsetmatrix( ndt, kt, sttmp->a1, ndt, &dA[d][total_size_a_mgpu[d]], lda, queues[d] );
                    } else {
                        // copy V^T
                        for (i=0; i<ndt; i++) {
                            for (j=0; j<kt; j++) work[j + i*kt] = (sttmp->a1)[i + j*ndt];
                        }
                        magma_dsetmatrix( kt, ndt, work, kt, &dA[d][total_size_a_mgpu[d]], lda, queues[d] );
                    }

                    // pointer to input, A
                    h_A_array[d][num_batch_mgpu[d]] = &dA[d][total_size_a_mgpu[d]];
                    total_size_a_mgpu[d] += lda*h_N[d][num_batch_mgpu[d]];

                    // pointer to input, zu
                    h_X_array[d][num_batch_mgpu[d]] = &st_leafmtxp->zu_mgpu[d][nstrtt-1];

                    // pointer to output, y
                    h_Y_array[d][num_batch_mgpu[d]] = &st_leafmtxp->zbu_mgpu[d][total_size_y_mgpu[d]];
                    saved_sz_mgpu[d][ip] = total_size_y_mgpu[d];
                    total_size_y_mgpu[d] += kt;

                    // ld and inc
                    h_lda[d][num_batch_mgpu[d]] = lda;
                    h_inc[d][num_batch_mgpu[d]] = 1;
//#define CHECKI
#ifdef CHECKI
if (st_leafmtxp->mpi_rank == 0)
printf( " compress-1(tp=%d, ip=%d, d=%d, %dx%d,%d:max=%d,%d)\n",tp,ip,d, h_M[d][num_batch_mgpu[d]],h_N[d][num_batch_mgpu[d]],h_lda[d][num_batch_mgpu[d]],max_m[d],max_n[d] );
#endif
                } else {
                    /**/
                    double *a2tmp = (double *)((void*)(sttmp->a1)+sttmp->a1size);
                    /**/
                    // dimmension
                    h_type[d][num_batch_mgpu[d]] = 2;
                    h_I[d][num_batch_mgpu[d]] = nstrtl;
                    h_J[d][num_batch_mgpu[d]] = nstrtt;
                    if (st_leafmtxp->transA == MagmaTrans) {
                        h_M[d][num_batch_mgpu[d]] = kt;
                        h_N[d][num_batch_mgpu[d]] = ndl;

                        max_m[d] = max(max_m[d], kt);
                        max_n[d] = max(max_n[d], ndl);
                    } else {
                        h_M[d][num_batch_mgpu[d]] = ndl;
                        h_N[d][num_batch_mgpu[d]] = kt;

                        max_m[d] = max(max_m[d], ndl);
                        max_n[d] = max(max_n[d], kt);
                    }
                    lda = magma_roundup( h_M[d][num_batch_mgpu[d]], batch_pad );

                    if (st_leafmtxp->transA == MagmaTrans) {
                        // copy U^T
                        for (i=0; i<ndl; i++) {
                            for (j=0; j<kt; j++) work[j + i*kt] = a2tmp[i + j*ndl];
                        }
                        magma_dsetmatrix( kt, ndl, work, kt, &dA[d][total_size_a_mgpu[d]], lda, queues[d] );
                    } else {
                        // copy U
                        magma_dsetmatrix( ndl, kt, a2tmp, ndl, &dA[d][total_size_a_mgpu[d]], lda, queues[d] );
                    }

                    // pointer to input, A
                    h_A_array[d][num_batch_mgpu[d]] = &dA[d][total_size_a_mgpu[d]];
                    total_size_a_mgpu[d] += lda*h_N[d][num_batch_mgpu[d]];

                    // pointer to input, zu
                    int size_y = saved_sz_mgpu[d][ip];
                    h_X_array[d][num_batch_mgpu[d]] = &st_leafmtxp->zbu_mgpu[d][size_y];

                    // pointer to output, y
                    h_Y_array[d][num_batch_mgpu[d]] = &st_leafmtxp->zau_mgpu[d][nstrtl-1];

                    // ld and inc
                    h_lda[d][num_batch_mgpu[d]] = lda;
                    h_inc[d][num_batch_mgpu[d]] = 1;
#ifdef CHECKI
if (st_leafmtxp->mpi_rank == 0)
printf( " compress-2(tp=%d, ip=%d, d=%d, %dx%d,%d:max=%d,%d)\n",tp,ip,d, h_M[d][num_batch_mgpu[d]],h_N[d][num_batch_mgpu[d]],h_lda[d][num_batch_mgpu[d]],max_m[d],max_n[d] );
#endif
                }
            } else if(sttmp->ltmtx == 2) { // full
                // dimension
                h_type[d][num_batch_mgpu[d]] = 0;
                h_I[d][num_batch_mgpu[d]] = nstrtl;
                h_J[d][num_batch_mgpu[d]] = nstrtt;
                if (st_leafmtxp->transA == MagmaTrans) {
                    h_M[d][num_batch_mgpu[d]] = ndt;
                    h_N[d][num_batch_mgpu[d]] = ndl;

                    max_m[d] = max(max_m[d], ndt);
                    max_n[d] = max(max_n[d], ndl);
                } else {
                    h_M[d][num_batch_mgpu[d]] = ndl;
                    h_N[d][num_batch_mgpu[d]] = ndt;

                    max_m[d] = max(max_m[d], ndl);
                    max_n[d] = max(max_n[d], ndt);
                }
                lda = magma_roundup( h_M[d][num_batch_mgpu[d]], batch_pad );

                // copy matrix
                if (st_leafmtxp->transA == MagmaTrans) {
                    magma_dsetmatrix( ndt, ndl, sttmp->a1, ndt, &dA[d][total_size_a_mgpu[d]], lda, queues[d] );
                } else {
                    for (i=0; i<ndt; i++) {
                        for (j=0; j<ndl; j++) work[j + i*ndl] = (sttmp->a1)[i + j*ndt];
                    }
                    magma_dsetmatrix( ndl, ndt, work, ndl, &dA[d][total_size_a_mgpu[d]], lda, queues[d] );
                }

                // pointer to input, A
                h_A_array[d][num_batch_mgpu[d]] = &dA[d][total_size_a_mgpu[d]];
                total_size_a_mgpu[d] += lda*h_N[d][num_batch_mgpu[d]];

                // pointer to input, zu
                h_X_array[d][num_batch_mgpu[d]] = &st_leafmtxp->zu_mgpu[d][nstrtt-1];

                // pointer to output, y
                h_Y_array[d][num_batch_mgpu[d]] = &st_leafmtxp->zau_mgpu[d][nstrtl-1];

                // ld and inc
                h_lda[d][num_batch_mgpu[d]] = lda;
                h_inc[d][num_batch_mgpu[d]] = 1;
#ifdef CHECKI
if (st_leafmtxp->mpi_rank == 0)
printf( " dense(tp=%d, ip=%d, d=%d, %dx%d,%d)\n",tp,ip,d, h_M[d][num_batch_mgpu[d]],h_N[d][num_batch_mgpu[d]],h_lda[d][num_batch_mgpu[d]] );
#endif
            }
            num_batch_mgpu[d] ++;

            int offset = (tp < nlf ? 0 : nlf_mgpu[d])-count[d];
            if ((num_batch_mgpu[d]-offset)%batch_count == 0 && 
#if 0
                tp != nlf-1 
                && tp != st_leafmtxp->num_batch-1
#else
                (tp_start+tp_inc <= nlf
                 || (tp_start+tp_inc > nlf && tp_start+tp_inc <= st_leafmtxp->num_batch))
#endif
               ) 
            {
                max_M[d][count[d]] = h_M[d][num_batch_mgpu[d]] = max_m[d];
                max_N[d][count[d]] = h_N[d][num_batch_mgpu[d]] = max_n[d];
                // extra space for M and N with batched
                h_A_array[d][num_batch] = NULL;
                count[d] ++;
#ifdef CHECKI
if (st_leafmtxp->mpi_rank == 0)
printf( " >> count[%d]=%d, max=%d,%d (num_batch=%d == %d)\n",d,count[d],max_m[d],max_n[d],num_batch_mgpu[d],batch_count );
#endif
                num_batch_mgpu[d] ++;

                max_m[d] = max_n[d] = 0;
            }
        }
        if (tp == nlf || tp == st_leafmtxp->num_batch) {
            // left over
            for (d=0; d<gpus_per_proc; d++) {
                if (max_m[d] > 0 || max_n[d] > 0) {
                    max_M[d][count[d]] = h_M[d][num_batch_mgpu[d]] = max_m[d];
                    max_N[d][count[d]] = h_N[d][num_batch_mgpu[d]] = max_n[d];
                    // extra space for M and N with batched
                    h_A_array[d][num_batch] = NULL;
                    count[d] ++;
#ifdef CHECKI
if (st_leafmtxp->mpi_rank == 0)
printf( " >> left-over << count[%d]=%d, max=%d,%d\n",d,count[d],max_m[d],max_n[d] );
#endif
                    num_batch_mgpu[d] ++;

                    max_m[d] = max_n[d] = 0;
                }
                if (tp == nlf) nlf_mgpu[d] = num_batch_mgpu[d]-count[d];
            }
        }
    }
    free(owner);
    free(work);

    st_leafmtxp->d_A_mgpu = (double***)malloc(gpus_per_proc*sizeof(double**));
    st_leafmtxp->d_X_mgpu = (double***)malloc(gpus_per_proc*sizeof(double**));
    st_leafmtxp->d_Y_mgpu = (double***)malloc(gpus_per_proc*sizeof(double**));
    st_leafmtxp->d_M_mgpu = (int**)malloc(gpus_per_proc * sizeof(int*));
    st_leafmtxp->d_N_mgpu = (int**)malloc(gpus_per_proc * sizeof(int*));
    st_leafmtxp->d_lda_mgpu = (int**)malloc(gpus_per_proc * sizeof(int*));
    st_leafmtxp->d_inc_mgpu = (int**)malloc(gpus_per_proc * sizeof(int*));
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice( (d+get_device_id(st_leafmtxp))%procs_per_node );

        magma_malloc((void**)&(st_leafmtxp->d_A_mgpu[d]), num_batch*sizeof(double*));
        magma_malloc((void**)&(st_leafmtxp->d_X_mgpu[d]), num_batch*sizeof(double*));
        magma_malloc((void**)&(st_leafmtxp->d_Y_mgpu[d]), num_batch*sizeof(double*));
        magma_setvector(num_batch, sizeof(double*), h_A_array[d], 1, st_leafmtxp->d_A_mgpu[d], 1, queues[d] );
        magma_setvector(num_batch, sizeof(double*), h_X_array[d], 1, st_leafmtxp->d_X_mgpu[d], 1, queues[d] );
        magma_setvector(num_batch, sizeof(double*), h_Y_array[d], 1, st_leafmtxp->d_Y_mgpu[d], 1, queues[d] );

        magma_imalloc(&st_leafmtxp->d_M_mgpu[d], num_batch+1);
        magma_imalloc(&st_leafmtxp->d_N_mgpu[d], num_batch+1);
        magma_imalloc(&st_leafmtxp->d_lda_mgpu[d], num_batch+1);
        magma_imalloc(&st_leafmtxp->d_inc_mgpu[d], num_batch+1);
        magma_setvector(num_batch, sizeof(magma_int_t), h_M[d], 1, st_leafmtxp->d_M_mgpu[d], 1, queues[d] );
        magma_setvector(num_batch, sizeof(magma_int_t), h_N[d], 1, st_leafmtxp->d_N_mgpu[d], 1, queues[d] );
        magma_setvector(num_batch, sizeof(magma_int_t), h_lda[d], 1, st_leafmtxp->d_lda_mgpu[d], 1, queues[d] );
        magma_setvector(num_batch, sizeof(magma_int_t), h_inc[d], 1, st_leafmtxp->d_inc_mgpu[d], 1, queues[d] );

        num_batch_mgpu[d] -= count[d]; // remove the extras at the end of each batch
        magma_queue_sync(queues[d]);
    }
    // main GPU
    magma_setdevice(get_device_id(st_leafmtxp));

    st_leafmtxp->h_type_mgpu = h_type;
    st_leafmtxp->h_I_mgpu = h_I;
    st_leafmtxp->h_J_mgpu = h_J;
    st_leafmtxp->h_M_mgpu = h_M;
    st_leafmtxp->h_N_mgpu = h_N;
    st_leafmtxp->h_lda_mgpu = h_lda;
    st_leafmtxp->h_A_mgpu = h_A_array;
    st_leafmtxp->h_X_mgpu = h_X_array;
    st_leafmtxp->h_Y_mgpu = h_Y_array;

    st_leafmtxp->max_M_mgpu = max_M;
    st_leafmtxp->max_N_mgpu = max_N;

    st_leafmtxp->nlf_mgpu = nlf_mgpu;
    st_leafmtxp->num_batch_mgpu = num_batch_mgpu;

    magma_free_cpu(h_inc);
    for (d=0; d<gpus_per_proc; d++) {
        free(saved_sz_mgpu[d]);
    }
    free(saved_sz_mgpu);
}

// batched GEMV
int c_hacapk_adot_body_lfmtx_mgpu_dgemv(int d, int ip,
                                        stc_HACApK_leafmtxp *st_leafmtxp, int *saved_ip[2],
                                        int *ip_start, int num_batch, int count,
                                        int *batchCount, int *num_saved,
                                        magma_queue_t queue) {
    double one = 1.0;

    int *d_M = st_leafmtxp->d_M_mgpu[d];
    int *d_N = st_leafmtxp->d_N_mgpu[d];
    int *d_inc = st_leafmtxp->d_inc_mgpu[d];
    int *d_lda = st_leafmtxp->d_lda_mgpu[d];
    int *max_M = st_leafmtxp->max_M_mgpu[d];
    int *max_N = st_leafmtxp->max_N_mgpu[d];
    double **d_A_array = st_leafmtxp->d_A_mgpu[d];
    double **d_X_array = st_leafmtxp->d_X_mgpu[d];
    double **d_Y_array = st_leafmtxp->d_Y_mgpu[d];

    int k, ip_end;
    int k_start;
    int nlf = st_leafmtxp->nlf_mgpu[d];
    int batch_count_per_gpu = (batch_count + gpus_per_proc-1)/gpus_per_proc;

    *num_saved = 0;
#ifdef CHECKI
if (st_leafmtxp->mpi_rank == 0)
printf( " >> num_batch=%d, nlf=%d",num_batch,nlf );
#endif
    if (num_batch-count >= nlf) {
        nlf = st_leafmtxp->num_batch_mgpu[d];
#ifdef CHECKI
if (st_leafmtxp->mpi_rank == 0)
printf( "->%d",nlf );
#endif
    }
    int batch_left = nlf-(num_batch-count);
#ifdef CHECKI
if (st_leafmtxp->mpi_rank == 0) {
printf( "\n" );
printf( " %d: batch_left=%d-%d=%d (ip=%d,nlf=%d)\n",d,nlf,num_batch-count,batch_left,ip,nlf);
}
#endif
    if (batch_left < batch_count) {
        *batchCount = batch_left;
    } else {
        *batchCount = batch_count;
    }
    ip_end = (*ip_start) + (*batchCount);

    // passing max M and N
    #if 1
    magmablas_dgemv_vbatched_max_nocheck_atomic(
                                    st_leafmtxp->transA, &d_M[num_batch], &d_N[num_batch],
                                    one, &d_A_array[num_batch], &d_lda[num_batch],
                                         &d_X_array[num_batch], &d_inc[num_batch],
                                         &d_Y_array[num_batch], &d_inc[num_batch],
                                    *batchCount, max_M[count], max_N[count],
                                    queue);
    #else
    int b;
    for (b=0; b<*batchCount; b++) {
#ifdef CHECKI
if (st_leafmtxp->mpi_rank == 0)
printf( "dgemv(%dx%d, %d) (%s)\n",st_leafmtxp->h_M_mgpu[d][num_batch+b], st_leafmtxp->h_N_mgpu[d][num_batch+b],st_leafmtxp->h_lda_mgpu[d][num_batch+b],
       (st_leafmtxp->h_M_mgpu[d][num_batch+b] > st_leafmtxp->h_lda_mgpu[d][num_batch+b] ? "fail" : "ok") );
#endif
        magmablas_dgemv( st_leafmtxp->transA,
                         st_leafmtxp->h_M_mgpu[d][num_batch+b], st_leafmtxp->h_N_mgpu[d][num_batch+b],
                         one, st_leafmtxp->h_A_mgpu[d][num_batch+b], st_leafmtxp->h_lda_mgpu[d][num_batch+b],
                              st_leafmtxp->h_X_mgpu[d][num_batch+b], 1,
                         one, st_leafmtxp->h_Y_mgpu[d][num_batch+b], 1, queue );
    }
    #endif
    *ip_start = ip_end;
    return 0;
}

void c_hacapk_adot_body_lfmtx_batch_mgpu(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu,
                                         double *zau_cpu, double *zu_cpu,
                                         double *time_batch, double *time_set, double *time_copy, int on_gpu,
                                         magma_queue_t *queue) {
    // constants
    double zero = 0.0;

    int ip, d;
    int nlf = st_leafmtxp->nlf;
    int *saved_ip[2];

    // copy the input vector to GPU
    int *ip_d = (int*)malloc(gpus_per_proc * sizeof(int));
    int *num_batch = (int*)malloc(gpus_per_proc * sizeof(int));
    int num_saved = 0, count = 0;
    // vectors are on GPU
    #ifdef PROF_MAGMA_BATCH
    double tic = MPI_Wtime();
    #endif
    if (gpus_per_proc > 1) {
        magma_setdevice( get_device_id(st_leafmtxp) );
        magma_dgetvector( st_leafmtxp->gn, zu, 1, zu_cpu,  1, queue[0] );
        //magma_dgetvector( st_leafmtxp->m, zau, 1, zau_cpu, 1, queue[0] );
    }
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice( (d+get_device_id(st_leafmtxp))%procs_per_node );
        if (d ==0) {
            magmablas_dlacpy( MagmaFull, st_leafmtxp->gn, 1, zu, st_leafmtxp->gn,
                              st_leafmtxp->zu_mgpu[d], st_leafmtxp->gn, queue[d]);
            magmablas_dlacpy( MagmaFull, st_leafmtxp->m, 1, zau, st_leafmtxp->m,
                              st_leafmtxp->zau_mgpu[d], st_leafmtxp->m, queue[d] );
        } else {
            magma_dsetvector( st_leafmtxp->gn,
                              zu_cpu, 1, st_leafmtxp->zu_mgpu[d], 1, queue[d] );
            //magma_dsetvector( st_leafmtxp->m,
            //                  zau_cpu, 1, st_leafmtxp->zau_mgpu[d], 1, queue[d] );
            magmablas_dlaset( MagmaFull, st_leafmtxp->m, 1, zero, zero,
                              st_leafmtxp->zau_mgpu[d], st_leafmtxp->m, queue[d] );
        }
        // first part of low-rank, zbu := V'*zu
        magmablas_dlaset( MagmaFull, st_leafmtxp->total_size_y, 1, zero, zero,
                          st_leafmtxp->zbu_mgpu[d], st_leafmtxp->total_size_y, queue[d] );
        ip_d[d] = 0;
        num_batch[d] = 0;
    }
    #ifdef PROF_MAGMA_BATCH
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice( (d+get_device_id(st_leafmtxp))%procs_per_node );
        magma_queue_sync( queue[d] );
    }
    *time_set += (MPI_Wtime()-tic);
    tic = MPI_Wtime();
    #endif
    fflush(stdout);

    // parse all the blocks
    for (ip = 0; ip < max(st_leafmtxp->num_batch, nlf) || num_saved > 0;) {
        /**/
        int ip_start = ip;
        for (d=0; d<gpus_per_proc; d++) {
            int num_start = num_batch[d];
            int batchCount = 0;
            magma_setdevice( (d+get_device_id(st_leafmtxp))%procs_per_node );

            // call batched GEMV and non-blocking copy to CPU
            c_hacapk_adot_body_lfmtx_mgpu_dgemv(d, ip_start, 
                                                st_leafmtxp, saved_ip,
                                                &ip_d[d], num_start, count,
                                                &batchCount, &num_saved,
                                                queue[d]);

            num_batch[d] += (1+ batchCount);
#ifdef CHECKI
printf( " %d: count=%d, ip=%d, batchCount=%d (%d, %d)\n",d,count,ip,batchCount,st_leafmtxp->num_batch, nlf );
#endif
            ip += batchCount;
        }
        count ++;
    }
    // stop timer
    #ifdef PROF_MAGMA_BATCH
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice( (d+get_device_id(st_leafmtxp))%procs_per_node );
        magma_queue_sync( queue[d] );
    }
    *time_batch += (MPI_Wtime()-tic);
    tic = MPI_Wtime();
    #endif
    // vectors are on GPU, accumulate on the main GPU
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice( (d+get_device_id(st_leafmtxp))%procs_per_node );
        if (d == 0) {
            magmablas_dlacpy( MagmaFull, st_leafmtxp->m, 1, st_leafmtxp->zau_mgpu[d], st_leafmtxp->m,
                              zau, st_leafmtxp->m, queue[d] );
        } else {
            magma_dgetvector( st_leafmtxp->m, 
                              st_leafmtxp->zau_mgpu[d], 1, zau_cpu, 1, queue[d] );

            magma_setdevice( get_device_id(st_leafmtxp) );
            magma_dsetvector( st_leafmtxp->m,
                              zau_cpu, 1, st_leafmtxp->zau_mgpu[0], 1, queue[0] );
            magma_daxpy( st_leafmtxp->m, 1.0, st_leafmtxp->zau_mgpu[0], 1, zau, 1, queue[0] );
        }
    }
    // set back to main GPU
    #ifdef PROF_MAGMA_BATCH
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice( (d+get_device_id(st_leafmtxp))%procs_per_node );
        magma_queue_sync( queue[d] );
    }
    *time_set += MPI_Wtime()-tic;
    #endif
    magma_setdevice( get_device_id(st_leafmtxp) );

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
}

#endif
