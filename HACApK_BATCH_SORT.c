
#if defined(HAVE_MAGMA) | defined(HAVE_MAGMA_BATCH)

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

/////////////////////////////////////////////////////
// Sorter for MatVec with batched GEMV

#ifdef SORT_BATCH_BY_SIZES
int hacapk_size_sorter(const void* arg1,const void* arg2) {
  const int *val1 = (const int*)arg1;
  const int *val2 = (const int*)arg2;

  //#define BY_GROUP
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
    return (id1 == id2 ? (val2[2] < val1[2]) : id2 < id1);
    #else
    // sort by m
    return (val2[1] < val1[1]);
    #endif
  #else
    #if defined(BY_GROUP)
    // sort by "group", whithin group, sort by m
    const int id1 = (val1[2]-1)/sort_group_size;
    const int id2 = (val2[2]-1)/sort_group_size;
    return (id1 == id2 ? (val2[1] < val1[1]) : id2 < id1);
    #else
    // sort by n
    return (val2[2] < val1[2]);
    #endif
  #endif
}

void hacapk_sort(int n, int *sizes) {
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
    int name_len;
    char proc_name[300];
    MPI_Get_processor_name( proc_name, &name_len );
    printf( " processor %d uses %d GPU on %s\n",st_leafmtxp->mpi_rank,(st_leafmtxp->mpi_rank)%procs_per_node,proc_name);

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
        int retval = magma_dmalloc( &st_leafmtxp->zau_gpu[0], st_leafmtxp->m );
        if ( MAGMA_SUCCESS != retval ) {
            fprintf( stderr, "!!!! magma_dmalloc failed for zau_gpu (m=%d)\n",st_leafmtxp->m);
            exit(0);
        }
    }
    if (st_leafmtxp->n > 0) {
        int retval = magma_dmalloc( &st_leafmtxp->zu_gpu, st_leafmtxp->gn );
        if ( MAGMA_SUCCESS != retval ) {
            fprintf( stderr, "!!!! magma_dmalloc failed for zu_gpu\n");
            exit(0);
        }
    }
    if (total_size_y > 0) {
        st_leafmtxp->zbu_gpu = (double**)malloc( sizeof(double*) );
        int retval = magma_dmalloc( &st_leafmtxp->zbu_gpu[0], total_size_y );
        if ( MAGMA_SUCCESS != retval ) {
            fprintf( stderr, "!!!! magma_dmalloc failed for zbu_gpu\n");
            exit(0);
        }
        st_leafmtxp->total_size_y = total_size_y;
    }
    double *dA = NULL;
    if (total_size_a > 0) {
        int retval = magma_dmalloc( &dA, total_size_a );
        if ( MAGMA_SUCCESS != retval ) {
            fprintf( stderr, "!!!! magma_dmalloc failed for dA(%d)\n",total_size_a);
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
            sizes[sort_array_size*ip + 1] = ndt; // # of columns
            sizes[sort_array_size*ip + 2] = kt;  // # of rows
            #if defined(BY_N)
            sizes[sort_array_size*ip + 3] = (ndt-1) / sort_group_size;
            #else
            sizes[sort_array_size*ip + 3] = (kt-1) / sort_group_size;
            #endif
            lwork = max(lwork, ndt*kt);
            if (max(ndt, kt) > batch_max_blocksize) {
                num_streamed ++;
            }
        } else if(sttmp->ltmtx == 2) { // full
            // dimension
            sizes[sort_array_size*ip + 0] = ip;
            sizes[sort_array_size*ip + 1] = ndt; // # of columns
            sizes[sort_array_size*ip + 2] = ndl; // # of rows
            #if defined(BY_N)
            sizes[sort_array_size*ip + 3] = (ndt-1) / sort_group_size;
            #else
            sizes[sort_array_size*ip + 3] = (ndl-1) / sort_group_size;
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
            sizes[sort_array_size*num_batch + 1] = kt;  // # of columns
            sizes[sort_array_size*num_batch + 2] = ndl; // # of rows
            #if defined(BY_N)
            sizes[sort_array_size*num_batch + 3] = (kt-1) / sort_group_size;
            #else
            sizes[sort_array_size*num_batch + 3] = (ndl-1) / sort_group_size;
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
    #define OUTPUT_SIZES
    #ifdef OUTPUT_SIZES
    FILE *fp;
    char filename[100];
    sprintf(filename,"sizes_%d.dat",st_leafmtxp->mpi_rank);
    fp = fopen(filename,"w");
    fprintf(fp, "%d\n",num_batch);
    for (ip = 0; ip < num_batch; ip++) {
        fprintf(fp, "%d %d %d\n",sizes[sort_array_size*ip + 0],sizes[sort_array_size*ip + 1],sizes[sort_array_size*ip + 2]);
    }
    fclose(fp);
    #endif

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
    //#define OUTPUT_SIZES
    #ifdef OUTPUT_SIZES
    sprintf(filename,"sizes_sorted_%d.dat",st_leafmtxp->mpi_rank);
    fp = fopen(filename,"w");
    fprintf(fp, "%d %d\n",num_batch,nlf);
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
        int retval = magma_dmalloc( &dA, total_size_a );
        if ( MAGMA_SUCCESS != retval ) {
            fprintf( stderr, "!!!! magma_dmalloc failed for dA(%d)\n",total_size_a);
            exit(0);
        }
    }
    if (total_size_y > 0) {
        magma_free(st_leafmtxp->zbu_gpu);
        st_leafmtxp->zbu_gpu = (double**)malloc( sizeof(double*) );
        int retval = magma_dmalloc( &st_leafmtxp->zbu_gpu[0], total_size_y );
        if ( MAGMA_SUCCESS != retval ) {
            fprintf( stderr, "!!!! magma_dmalloc failed for zbu_gpu\n");
            exit(0);
        }
        st_leafmtxp->total_size_y = total_size_y;
    }
    double zero = 0.0;
    magmablas_dlaset( MagmaFull, total_size_y, 1, zero, zero, 
                      st_leafmtxp->zbu_gpu[0], total_size_y, queue );
    if (st_leafmtxp->m > 0) {
        magma_free(st_leafmtxp->zau_gpu);
        int retval = magma_dmalloc( &st_leafmtxp->zau_gpu[0], st_leafmtxp->m+Max_M );
        if ( MAGMA_SUCCESS != retval ) {
            fprintf( stderr, "!!!! magma_dmalloc failed for zau_gpu\n");
            exit(0);
        }
    }
    magmablas_dlaset( MagmaFull, st_leafmtxp->m+Max_M, 1, zero, zero, 
                      st_leafmtxp->zau_gpu[0], total_size_y, queue );
    if (st_leafmtxp->n > 0) {
        magma_free(st_leafmtxp->zu_gpu);
        int retval = magma_dmalloc( &st_leafmtxp->zu_gpu, st_leafmtxp->gn+Max_N );
        if ( MAGMA_SUCCESS != retval ) {
            fprintf( stderr, "!!!! magma_dmalloc failed for zu_gpu\n");
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

#endif

#endif
