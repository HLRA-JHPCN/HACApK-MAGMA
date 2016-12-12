#include	<stdio.h>
#include	<stdlib.h>
#include	<time.h>
#include	"omp.h"
#include	"mpi.h"
#include	"HACApK_FPGA.h"
#include        <ISO_Fortran_binding.h>

//!***c_HACApK_adot_body_lfmtx
void  c_hacapk_adot_body_lfmtx_(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu) {
 register int ip,il,it;
 int nlf,ndl,ndt,nstrtl,nstrtt,kt,itl,itt,ill;
 int st_lf_stride = st_leafmtxp->st_lf_stride;
 int a1size;
 
 nlf=st_leafmtxp->nlf;
  //fprintf(stderr,"nlf=%d \n",nlf);

 for (ip = 0; ip < nlf; ip++) {
   /**/
   stc_HACApK_leafmtx *sttmp;
   sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
   //fprintf(stderr, "%d: %p\n", ip, sttmp);
   /**/

   ndl   =sttmp->ndl; // m: number of rows
   ndt   =sttmp->ndt; // n: number of columns
   nstrtl=sttmp->nstrtl; // i: index of first row
   nstrtt=sttmp->nstrtt; // j: index of first column
   //fprintf(stderr,"ip=%d, ndl=%d, ndt=%d, nstrtl=%d, nstrtt=%d \n",ip,ndl,ndt,nstrtl,nstrtt);
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
#if defined(HAVE_MAGMA) | defined(HAVE_MAGMA_BATCH)
#define num_streams 5
#define max(a,b) (((a) > (b) ? (a) : (b)))

/////////////////////////////////////////////////////
// MatVec on GPU
void  c_hacapk_adot_body_lfmtx_gpu_(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu) {
    // constants
    int ione = 1;
    double one = 1.0;
    double zero = 0.0;

    int ip, il, it;
    int nlf, ndl, ndt, nstrtl, nstrtt, kt, itl, itt, ill;
    int st_lf_stride = st_leafmtxp->st_lf_stride;
    int a1size;
 
    //#define GPU
    #if defined(GPU)
    // allocate queue
    magma_device_t cdev;
    magma_queue_t queue[num_streams];
    magma_getdevice( &cdev );
    for (ip = 0; ip < num_streams; ip++) {
        magma_queue_create( cdev, &queue[ip] );
    }

    // copy the input vector to GPU
    magma_dsetvector( st_leafmtxp->n,  zu, 1, st_leafmtxp->zu_gpu,  1, queue[0] );
    magma_dsetvector( st_leafmtxp->m, zau, 1, st_leafmtxp->zau_gpu[0], 1, queue[0] );
    for (ip = 1; ip < num_streams; ip++) {
        magmablas_dlaset( MagmaFull, st_leafmtxp->m, 1, zero, zero, 
                          st_leafmtxp->zau_gpu[ip], st_leafmtxp->m, queue[ip] );
    }
    #endif

    // parse all the blocks
    double tic = MPI_Wtime();
    nlf=st_leafmtxp->nlf;
    for (ip = 0; ip < nlf; ip++) {
        /**/
        stc_HACApK_leafmtx *sttmp;
        sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
        /**/

        ndl    = sttmp->ndl; // m: number of rows
        ndt    = sttmp->ndt; // n: number of columns
        nstrtl = sttmp->nstrtl; // i: index of first row
        nstrtt = sttmp->nstrtt; // j: index of first column
        #if defined(GPU)
        int stream_id = ip%num_streams;
        #endif
        if (sttmp->ltmtx == 1) { // compressed
            /**/
            double *a2tmp = (double *)((void*)(sttmp->a1)+sttmp->a1size);
            /**/
            kt=sttmp->kt; // rank
            // zbu := V'*zu
            #define CPU
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
    magma_queue_sync( queue[0] );
    printf( " time_gpu: %.2e seconds\n",MPI_Wtime()-tic );
    // copy back
    magma_dgetvector( st_leafmtxp->m, st_leafmtxp->zau_gpu[0], 1, zau, 1, queue[0] );
    magma_queue_destroy( queue[0] );
    #else
    printf( " time_cpu: %.2e seconds\n",MPI_Wtime()-tic );
    #endif
}

/////////////////////////////////////////////////////
// copy blocks to GPU
void  c_hacapk_adot_body_lfcpy_gpu_(stc_HACApK_leafmtxp *st_leafmtxp) {
    // constants
    int ione = 1;
    double one = 1.0;
    double zero = 0.0;

    // local variables
    register int ip, il, it;
    int nlf, ndl, ndt, nstrtl, nstrtt, kt, itl;
    int st_lf_stride = st_leafmtxp->st_lf_stride;
    int a1size;

    // let me initialize here for now..
    magma_init();
    magma_print_environment();

    // allocate queue
    magma_device_t cdev;
    magma_queue_t queue = NULL;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    // number of blocks
    nlf = st_leafmtxp->nlf; 

    // initialize data structure
    st_leafmtxp->m = 0;
    st_leafmtxp->n = 0;
    st_leafmtxp->max_block = 0;
    st_leafmtxp->mtx1_gpu = (magmaDouble_ptr*)malloc(nlf * sizeof(magmaDouble_ptr));
    st_leafmtxp->mtx2_gpu = (magmaDouble_ptr*)malloc(nlf * sizeof(magmaDouble_ptr));

    // parse all the blocks
    FILE *fp = fopen("sizes.dat","w");
    for (ip = 0; ip < nlf; ip++) {
        /**/
        stc_HACApK_leafmtx *sttmp;
        sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
        ndl    = sttmp->ndl; // m: number of rows
        ndt    = sttmp->ndt; // n: number of columns
        nstrtl = sttmp->nstrtl; // i: index of first row
        nstrtt = sttmp->nstrtt; // j: index of first column
        if (nstrtl == nstrtt) {
            // diagonal block
            st_leafmtxp->m += ndl;
            st_leafmtxp->n += ndt;
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
                fprintf( stderr, "!!!! magma_malloc failed for leafmtxp[0][%d]\n", ip);
                exit(0);
            }
            magma_dsetmatrix( ndt, kt, sttmp->a1, ndt, st_leafmtxp->mtx1_gpu[ip], ndt, queue );
            fprintf( fp,"1 %d %d\n",kt,ndt );

            // copy U
            st_leafmtxp->mtx2_gpu[ip] = NULL;
            retval = magma_malloc( (void**) &(st_leafmtxp->mtx2_gpu[ip]), (ndl*kt)*sizeof(double) );
            if ( MAGMA_SUCCESS != retval ) {
                fprintf( stderr, "!!!! magma_malloc failed for leafmtxp[1][%d]\n", ip);
                exit(0);
            }
            magma_dsetmatrix( ndl, kt, a2tmp, ndl, st_leafmtxp->mtx2_gpu[ip], ndl, queue );
            fprintf( fp,"2 %d %d\n",ndl,kt );
        } else if (sttmp->ltmtx == 2) { // full
            st_leafmtxp->mtx1_gpu[ip] = NULL;
            int retval = magma_malloc( (void**) &(st_leafmtxp->mtx1_gpu[ip]), (ndt*ndl)*sizeof(double) );
            if ( MAGMA_SUCCESS != retval ) {
                fprintf( stderr, "!!!! magma_malloc failed for leafmtxp[0][%d]\n", ip);
                exit(0);
            }
            magma_dsetmatrix( ndt, ndl, sttmp->a1, ndt, st_leafmtxp->mtx1_gpu[ip], ndt, queue );
            fprintf( fp,"3 %d %d\n",ndl,ndt );
            st_leafmtxp->mtx2_gpu[ip] = NULL;
        }
    }
    fclose(fp);
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
        int retval = magma_malloc( (void**) &st_leafmtxp->zu_gpu, (st_leafmtxp->n)*sizeof(double) );
        if ( MAGMA_SUCCESS != retval ) {
            fprintf( stderr, "!!!! magma_malloc failed for zu_gpu\n");
            exit(0);
        }
    }
    printf( " %d-by-%d matrix\n",st_leafmtxp->m,st_leafmtxp->n );

    magma_queue_destroy( queue );
}

/////////////////////////////////////////////////////
// delete GPU memory
void  c_hacapk_adot_body_lfdel_gpu_(stc_HACApK_leafmtxp *st_leafmtxp) {
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
}
#endif

#define batch_count 10000

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
// MatVec with OpenMP, version 1
void c_hacapk_adot_body_lfmtx_batch_v1_(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu) {
    // constants
    int ione = 1;
    double one = 1.0;
    double zero = 0.0;

    int ip, iip, k;
    int ndl, ndt, nstrtl, nstrtt, kt;
    int st_lf_stride = st_leafmtxp->st_lf_stride;
    int a1size;

    int num_saved = 0, count = 0;
    int nlf=st_leafmtxp->nlf;
    int *saved_ip[2]; 
    int *saved_bt[2]; 
    saved_ip[0] = (int*)malloc( nlf * sizeof(int) ); 
    saved_ip[1] = (int*)malloc( nlf * sizeof(int) ); 
    saved_bt[0] = (int*)malloc( nlf * sizeof(int) ); 
    saved_bt[1] = (int*)malloc( nlf * sizeof(int) ); 

    double ** zau_batch = (double**)malloc(batch_count * sizeof(double*));
    double ** zbu_batch = (double**)malloc(batch_count * sizeof(double*));
    for (ip = 0; ip < batch_count; ip++) {
        zau_batch[ip] = (double*)malloc(st_leafmtxp->max_block * sizeof(double));
        zbu_batch[ip] = (double*)malloc(st_leafmtxp->max_block * sizeof(double));
    }
    #define GPU
    #if defined(GPU)
    // allocate queue
    magma_device_t cdev;
    magma_queue_t queue;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    // copy the input vector to GPU
    magma_dsetvector( st_leafmtxp->n,  zu, 1, st_leafmtxp->zu_gpu,  1, queue );
    magma_dsetvector( st_leafmtxp->m, zau, 1, st_leafmtxp->zau_gpu[0], 1, queue );
    #endif

    // parse all the blocks
    int num_batch = 0;
    int *d_M = st_leafmtxp->d_M;
    int *d_N = st_leafmtxp->d_N;
    int *d_inc = st_leafmtxp->d_inc;
    double **d_A_array = st_leafmtxp->d_A_array;
    double **d_X_array = st_leafmtxp->d_X_array;
    double **d_Y_array = st_leafmtxp->d_Y_array;
    double **h_Y_array = st_leafmtxp->h_Y_array;

    double tic = MPI_Wtime();
    for (ip = 0; ip < nlf || num_saved > 0;) {

        /**/
        int batchCount = 0;
        #if defined(GPU)
        batchCount = num_saved;
        #else
        for (k = 0; k<num_saved; k++) {
            int batch_id = (count-1)%2;
            int iip = saved_ip[batch_id][k];
            int ibt = saved_bt[batch_id][k];

            /**/
            stc_HACApK_leafmtx *sttmp;
            sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * iip;
            /**/

            kt     = sttmp->kt;  // rank
            ndl    = sttmp->ndl; // m: number of rows
            ndt    = sttmp->ndt; // n: number of columns
            nstrtl = sttmp->nstrtl; // i: index of first row
            nstrtt = sttmp->nstrtt; // j: index of first column

            // zau :+= U*zbu
            double *a2tmp = (double *)((void*)(sttmp->a1)+sttmp->a1size);
            //printf( " saved dgemv(ip=%d, k=%d), %dx%d\n",iip,ibt,ndl,kt );
            dgemv_("N", &ndl, &kt, 
                   &one,  a2tmp, &ndl, 
                          zbu_batch[ibt], &ione,
                   &zero, zau_batch[k], &ione );
            batchCount ++;
        }
        #endif

        /**/
        int k_start = num_saved;
        int ip_start = ip;
        num_saved = 0;
        for (k = k_start; k < batch_count && ip < nlf; ip++, k++) {
            /**/
            stc_HACApK_leafmtx *sttmp;
            sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
            /**/

            ndl    = sttmp->ndl; // m: number of rows
            ndt    = sttmp->ndt; // n: number of columns
            nstrtl = sttmp->nstrtl; // i: index of first row
            nstrtt = sttmp->nstrtt; // j: index of first column
            if (sttmp->ltmtx == 1) { // compressed
                /**/
                int batch_id = count%2;
                #if !defined(GPU)
                double *a2tmp = (double *)((void*)(sttmp->a1)+sttmp->a1size);
                /**/
                kt = sttmp->kt; // rank
                // zbu := V'*zu
                //printf( " compressed dgemv(ip=%d, k=%d), %dx%d\n",ip,k, ndl,kt );
                dgemv_("T", &ndt, &kt, 
                       &one,  sttmp->a1, &ndt, 
                             &zu[nstrtt-1], &ione,
                       &zero, zbu_batch[k], &ione );
                #endif

                saved_ip[batch_id][num_saved] = ip;
                saved_bt[batch_id][num_saved] = k;
                num_saved ++;
                batchCount ++;
            } else if(sttmp->ltmtx == 2) { // full
                //printf( " full dgemv(ip=%d, k=%d), %dx%d, %.2e,%.2e\n",ip,k, ndt,ndl,one,zero );
                #if !defined(GPU)
                dgemv_("T", &ndt, &ndl, 
                       &one,  sttmp->a1, &ndt, 
                             &zu[nstrtt-1], &ione,
                       &zero, zau_batch[k], &ione );
                #endif
                batchCount ++;
            }
        }
        #if defined(GPU)
        magmablas_dgemv_vbatched(MagmaTrans, &d_M[num_batch], &d_N[num_batch],
                                 one,  &d_A_array[num_batch], &d_M[num_batch],
                                       &d_X_array[num_batch], &d_inc[num_batch],
                                 zero, &d_Y_array[num_batch], &d_inc[num_batch],
                                 batchCount, queue);
        #endif

        /**/
        for (k = 0; k < k_start; k++) {
            int batch_id = (count-1)%2;
            int iip = saved_ip[batch_id][k];
            int ibt = saved_bt[batch_id][k];

            /**/
            stc_HACApK_leafmtx *sttmp;
            sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * iip;
            /**/

            ndl    = sttmp->ndl; // m: number of rows
            ndt    = sttmp->ndt; // n: number of columns
            nstrtl = sttmp->nstrtl; // i: index of first row
            nstrtt = sttmp->nstrtt; // j: index of first column
            //printf( " compress: daxpy(ip=%d, k=%d->%d, ndl=%d)\n",iip,ibt,k,ndl );
            #if defined(GPU)
            magma_dgetvector( ndl, h_Y_array[num_batch+k], 1, zau_batch[k], 1, queue );
            #endif
            daxpy_(&ndl,
                   &one, zau_batch[k], &ione,
                        &zau[nstrtl-1], &ione );
        }

        for (k = k_start, ip = ip_start; k < batch_count && ip < nlf; ip++, k++) {
            /**/
            stc_HACApK_leafmtx *sttmp;
            sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
            /**/

            ndl    = sttmp->ndl; // m: number of rows
            ndt    = sttmp->ndt; // n: number of columns
            nstrtl = sttmp->nstrtl; // i: index of first row
            nstrtt = sttmp->nstrtt; // j: index of first column

            if(sttmp->ltmtx == 2) { // full
                //printf( " full: daxpy(k=%d, ndl=%d), one=%.2e\n",k,ndl,one );
                #if defined(GPU)
                magma_dgetvector( ndl, h_Y_array[num_batch+k], 1, zau_batch[k], 1, queue );
                #endif
                daxpy_(&ndl,
                       &one, zau_batch[k], &ione,
                            &zau[nstrtl-1], &ione );
            }
        }
        num_batch +=(1+ batchCount);
        count ++;
    }
    printf( " time_batch: %.2e seconds\n",MPI_Wtime()-tic );
    for (ip = 0; ip < batch_count; ip++) {
        free(zau_batch[ip]);
        free(zbu_batch[ip]);
    }
    free(zau_batch);
    free(zbu_batch);

    free(saved_ip[0]);
    free(saved_ip[1]);
    free(saved_bt[0]);
    free(saved_bt[1]);

    #if defined(GPU)
    magma_queue_destroy( queue );
    #endif
}

/////////////////////////////////////////////////////
// MatVec with OpenMP
int c_hacapk_adot_body_lfmtx_batch_dgemv(stc_HACApK_leafmtxp *st_leafmtxp, int *saved_ip[2],
                                         int *ip_start, int num_batch, int count,
                                         int *batchCount, int *num_saved,
                                         double * zau_batch, magma_queue_t queue);
int c_hacapk_adot_body_lfmtx_batch_daxpy(stc_HACApK_leafmtxp *st_leafmtxp, int *saved_ip[2],
                                         int num_batch, int count, int k_start, int ip_start,
                                         double* zau_batch, double* zau, magma_queue_t queue);

void c_hacapk_adot_body_lfmtx_batch_(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu) {
    // constants
    int ione = 1;
    double one = 1.0;
    double zero = 0.0;

    int ip, iip, k;
    int ndl, ndt, nstrtl, nstrtt, kt;
    int st_lf_stride = st_leafmtxp->st_lf_stride;

    int num_saved = 0, count = 0;
    int nlf = st_leafmtxp->nlf;
    int *saved_ip[2]; 
    saved_ip[0] = (int*)malloc( nlf * sizeof(int) ); 
    saved_ip[1] = (int*)malloc( nlf * sizeof(int) ); 

    double * zau_batch;
    magma_dmalloc_pinned( &zau_batch,  batch_count*(st_leafmtxp->max_block) );
    // allocate queue
    magma_device_t cdev;
    magma_queue_t queue;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    // copy the input vector to GPU
    magma_dsetvector( st_leafmtxp->n,  zu, 1, st_leafmtxp->zu_gpu,  1, queue );
    magma_dsetvector( st_leafmtxp->m, zau, 1, st_leafmtxp->zau_gpu[0], 1, queue );

    // parse all the blocks
    int num_batch = 0, size_y;
    int *d_M = st_leafmtxp->d_M;
    int *d_N = st_leafmtxp->d_N;
    int *d_inc = st_leafmtxp->d_inc;
    double **d_A_array = st_leafmtxp->d_A_array;
    double **d_X_array = st_leafmtxp->d_X_array;
    double **d_Y_array = st_leafmtxp->d_Y_array;
    double **h_Y_array = st_leafmtxp->h_Y_array;

    double tic = MPI_Wtime();
    for (ip = 0; ip < nlf || num_saved > 0;) {

        /**/
        int batchCount = 0;
        int k_start = num_saved;
        int ip_start = ip;

        // call batched GEMV and non-blocking copy to CPU
        c_hacapk_adot_body_lfmtx_batch_dgemv(st_leafmtxp, saved_ip,
                                             &ip, num_batch, count,
                                             &batchCount, &num_saved,
                                             zau_batch, queue);
        magma_queue_sync( queue );

        /* accumulate on CPU */
        c_hacapk_adot_body_lfmtx_batch_daxpy(st_leafmtxp, saved_ip,
                                             num_batch, count, k_start, ip_start,
                                             zau_batch, zau, queue);

        printf( " batchCount = %d\n",batchCount );
        num_batch +=(1+ batchCount);
        count ++;
    }
    #if !defined(ACCUME_ON_CPU)
    magma_dgetvector( st_leafmtxp->m, st_leafmtxp->zau_gpu[0], 1, zau, 1, queue );
    #endif
    printf( " time_batch: %.2e seconds\n",MPI_Wtime()-tic );
    magma_free_pinned(zau_batch);

    free(saved_ip[0]);
    free(saved_ip[1]);

    magma_queue_destroy( queue );
}

//
int c_hacapk_adot_body_lfmtx_batch_dgemv(stc_HACApK_leafmtxp *st_leafmtxp, int *saved_ip[2],
                                         int *ip_start, int num_batch, int count,
                                         int *batchCount, int *num_saved,
                                         double * zau_batch, magma_queue_t queue) {
    double one = 1.0;
    double zero = 0.0;

    int *d_M = st_leafmtxp->d_M;
    int *d_N = st_leafmtxp->d_N;
    int *d_inc = st_leafmtxp->d_inc;
    double **d_A_array = st_leafmtxp->d_A_array;
    double **d_X_array = st_leafmtxp->d_X_array;
    double **d_Y_array = st_leafmtxp->d_Y_array;
    double **h_Y_array = st_leafmtxp->h_Y_array;

    int k, ip;
    int k_start = *num_saved;
    int nlf = st_leafmtxp->nlf;
    int st_lf_stride = st_leafmtxp->st_lf_stride;

    *batchCount = k_start;
    *num_saved = 0;
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
    magmablas_dgemv_vbatched(MagmaTrans, &d_M[num_batch], &d_N[num_batch],
                             one,  &d_A_array[num_batch], &d_M[num_batch],
                                   &d_X_array[num_batch], &d_inc[num_batch],
                             zero, &d_Y_array[num_batch], &d_inc[num_batch],
                             *batchCount, queue);

    #if defined(ACCUME_ON_CPU)
    /* get results */
    int size_y = 0;
    for (k = 0; k < k_start; k++) {
        int batch_id = (count-1)%2;
        int iip = saved_ip[batch_id][k];

        /**/
        stc_HACApK_leafmtx *sttmp;
        sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * iip;
        /**/

        int ndl    = sttmp->ndl; // m: number of rows
        magma_dgetvector_async( ndl, h_Y_array[num_batch+k], 1, &zau_batch[size_y], 1, queue );
        size_y += ndl;
    }
    for (k = k_start, ip = *ip_start; k < batch_count && ip < nlf; ip++, k++) {
        /**/
        stc_HACApK_leafmtx *sttmp;
        sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
        /**/

        if(sttmp->ltmtx == 2) { // full
            int ndl    = sttmp->ndl; // m: number of rows
            magma_dgetvector_async( ndl, h_Y_array[num_batch+k], 1, &zau_batch[size_y], 1, queue );
            size_y += ndl;
        }
    }
    #endif
    *ip_start = ip;
}

//
int c_hacapk_adot_body_lfmtx_batch_daxpy(stc_HACApK_leafmtxp *st_leafmtxp, int *saved_ip[2],
                                         int num_batch, int count, int k_start, int ip_start,
                                         double* zau_batch, double* zau, magma_queue_t queue) {

    int ione = 1;
    double one = 1.0;
    double zero = 0.0;

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

        int ndl    = sttmp->ndl; // m: number of rows
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

        int ndl    = sttmp->ndl; // m: number of rows
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
}

/////////////////////////////////////////////////////
// MatVec with OpenMP
void  c_hacapk_adot_body_lfmtx_omp_(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu) {
    // constants
    int ione = 1;
    double one = 1.0;
    double zero = 0.0;

    int ip, il, it;
    int nlf, ndl, ndt, nstrtl, nstrtt, kt, itl, itt;
    int st_lf_stride = st_leafmtxp->st_lf_stride;
    int a1size;
 
    // parse all the blocks
    double tic = MPI_Wtime();
    nlf=st_leafmtxp->nlf;
    #pragma omp parallel
    #pragma omp master
    {
        int tid  = omp_get_thread_num();
        if (tid == 0) {
            int nthreads = omp_get_max_threads();
            printf( " num_threads=%d\n",nthreads );
        }
        for (ip = 0; ip < nlf; ip++) {
            /**/
            stc_HACApK_leafmtx *sttmp;
            sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
            /**/

            ndl    = sttmp->ndl; // m: number of rows
            ndt    = sttmp->ndt; // n: number of columns
            nstrtl = sttmp->nstrtl; // i: index of first row
            nstrtt = sttmp->nstrtt; // j: index of first column
            if (sttmp->ltmtx == 1) { // compressed
                /**/
                double *a2tmp = (double *)((void*)(sttmp->a1)+sttmp->a1size);
                /**/
                kt = sttmp->kt; // rank
                // zbu := V'*zu
                #pragma omp task 
                {
                    dgemv_("T", &ndt, &kt, 
                           &one, sttmp->a1, &ndt, 
                                 &zu[nstrtt-1], &ione,
                           &zero, zbu, &ione );
                }
                // zau :+= U*zbu
                #pragma omp task 
                {
                    dgemv_("N", &ndl, &kt, 
                           &one, a2tmp, &ndl, 
                                 zbu, &ione,
                           &one, &zau[nstrtl-1], &ione );
                }
            } else if(sttmp->ltmtx == 2) { // full
                #pragma omp task 
                {
                    dgemv_("T", &ndt, &ndl, 
                           &one, sttmp->a1, &ndt, 
                                 &zu[nstrtt-1], &ione,
                           &one, &zau[nstrtl-1], &ione );
                }
            }
        }
    }
    printf( " time_batch: %.2e seconds\n",MPI_Wtime()-tic );
}

/////////////////////////////////////////////////////
// copy blocks to GPU
void  c_hacapk_adot_body_lfcpy_batch_(stc_HACApK_leafmtxp *st_leafmtxp) {
    // constants
    int ione = 1;
    double one = 1.0;
    double zero = 0.0;

    // local variables
    int i, j, k;
    int ip, il, it;
    int nlf, ndl, ndt, nstrtl, nstrtt, kt, itl;
    int st_lf_stride = st_leafmtxp->st_lf_stride;
    int a1size;

    // let me initialize here for now..
    magma_init();
    magma_print_environment();

    // allocate queue
    magma_device_t cdev;
    magma_queue_t queue = NULL;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

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

    int retval;
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
        nstrtl = sttmp->nstrtl; // i: index of first row
        nstrtt = sttmp->nstrtt; // j: index of first column

        if (nstrtl == nstrtt) {
            // diagonal block
            st_leafmtxp->m += ndl;
            st_leafmtxp->n += ndt;
        }
        if (st_leafmtxp->max_block < max(ndl, ndt)) {
            st_leafmtxp->max_block = max(ndl, ndt);
        }

        if (sttmp->ltmtx == 1) { // compressed
            total_size_a += kt*(ndl+ndt);
            total_size_y += ndl+kt;
            num_batch += 2;
        } else {                 // full
            total_size_a += ndl*ndt;
            total_size_y += ndl;
            num_batch += 1;
        }
    }

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
    }
    double *dA = NULL;
    if (total_size_a > 0) {
        int retval = magma_malloc( (void**) &dA, total_size_a*sizeof(double) );
        if ( MAGMA_SUCCESS != retval ) {
            fprintf( stderr, "!!!! magma_malloc failed for dA\n");
            exit(0);
        }
    }

    // extra space for M and N with batch
    num_batch += 1+(num_batch+batch_count-1)/batch_count;
    st_leafmtxp->num_batch = num_batch;

    double **h_A_array, **h_X_array, **h_Y_array;
    magma_malloc_cpu((void**)&(h_A_array), num_batch*sizeof(double*));
    magma_malloc_cpu((void**)&(h_X_array), num_batch*sizeof(double*));
    magma_malloc_cpu((void**)&(h_Y_array), num_batch*sizeof(double*));

    magma_int_t *h_M, *h_N, *h_inc;
    magma_imalloc_cpu(&h_M, num_batch);
    magma_imalloc_cpu(&h_N, num_batch);
    magma_imalloc_cpu(&h_inc, num_batch);

    // parse all the blocks
    int count = 0;
    total_size_y = 0;
    total_size_a = 0;
    num_batch = 0;
    for (ip = 0; ip < nlf || num_saved > 0;) {
        /**/
        for (k = 0; k<num_saved; k++) {
            int batch_id = (count-1)%2;
            int iip = saved_ip[batch_id][k];
            int ibt = saved_bt[batch_id][k];

            /**/
            stc_HACApK_leafmtx *sttmp;
            sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * iip;
            /**/

            kt     = sttmp->kt;  // rank
            ndl    = sttmp->ndl; // m: number of rows
            ndt    = sttmp->ndt; // n: number of columns
            nstrtl = sttmp->nstrtl; // i: index of first row
            nstrtt = sttmp->nstrtt; // j: index of first column

            // copy U^T
            double *a2tmp = (double *)((void*)(sttmp->a1)+sttmp->a1size);
            double *tmp = (double*)malloc(ndl*kt * sizeof(double));
            for (i=0; i<ndl; i++) {
                for (j=0; j<kt; j++) tmp[j + i*kt] = a2tmp[i + j*ndl];
            }

            h_A_array[num_batch] = &dA[total_size_a];
            magma_dsetmatrix( kt, ndl, tmp, kt, h_A_array[num_batch], kt, queue );
            total_size_a += kt*ndl;
            free(tmp);

            // pointer to input, zu
            int size_y = saved_sz[batch_id][k];
            h_X_array[num_batch] = &st_leafmtxp->zbu_gpu[0][size_y];

            // pointer to output, y
            h_Y_array[num_batch] = &st_leafmtxp->zbu_gpu[0][total_size_y];
            total_size_y += ndl;

            // dimmension
            h_M[num_batch] = kt;
            h_N[num_batch] = ndl;
            h_inc[num_batch] = 1;

            num_batch ++;
        }

        /**/
        int k_start = num_saved;
        int ip_start = ip;
        num_saved = 0;
        for (k = k_start; k < batch_count && ip < nlf; ip++, k++) {
            /**/
            stc_HACApK_leafmtx *sttmp;
            sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
            /**/

            ndl    = sttmp->ndl; // m: number of rows
            ndt    = sttmp->ndt; // n: number of columns
            nstrtl = sttmp->nstrtl; // i: index of first row
            nstrtt = sttmp->nstrtt; // j: index of first column

            if (sttmp->ltmtx == 1) { // compressed
                /**/
                int batch_id = count%2;
                double *a2tmp = (double *)((void*)(sttmp->a1)+sttmp->a1size);
                /**/
                kt = sttmp->kt; // rank
                // copy V
                h_A_array[num_batch] = &dA[total_size_a];
                magma_dsetmatrix( ndt, kt, sttmp->a1, ndt, h_A_array[num_batch], ndt, queue );
                total_size_a += ndt*kt;

                // pointer to input, zu
                h_X_array[num_batch] = &st_leafmtxp->zu_gpu[nstrtt-1];

                // pointer to output, y
                h_Y_array[num_batch] = &st_leafmtxp->zbu_gpu[0][total_size_y];
                saved_sz[batch_id][num_saved] = total_size_y;
                total_size_y += kt;

                // dimension
                h_M[num_batch] = ndt;
                h_N[num_batch] = kt;
                h_inc[num_batch] = 1;
                num_batch ++;

                saved_ip[batch_id][num_saved] = ip;
                saved_bt[batch_id][num_saved] = k;

                num_saved ++;
            } else if(sttmp->ltmtx == 2) { // full
                // copy matrix
                h_A_array[num_batch] = &dA[total_size_a];
                magma_dsetmatrix( ndt, ndl, sttmp->a1, ndt, h_A_array[num_batch], ndt, queue );
                total_size_a += ndt*ndl;

                // pointer to input, zu
                h_X_array[num_batch] = &st_leafmtxp->zu_gpu[nstrtt-1];

                // pointer to output, y
                h_Y_array[num_batch] = &st_leafmtxp->zbu_gpu[0][total_size_y];
                total_size_y += ndl;

                // dimension
                h_M[num_batch] = ndt;
                h_N[num_batch] = ndl;
                h_inc[num_batch] = 1;
                num_batch ++;
            }
        }
        // extra space for M and N with batched
        h_A_array[num_batch] = NULL;
        num_batch ++;
        count ++;
    }
    magma_malloc((void**)&(st_leafmtxp->d_A_array), num_batch*sizeof(double*));
    magma_malloc((void**)&(st_leafmtxp->d_X_array), num_batch*sizeof(double*));
    magma_malloc((void**)&(st_leafmtxp->d_Y_array), num_batch*sizeof(double*));
    magma_setvector(num_batch, sizeof(double*), h_A_array, 1, st_leafmtxp->d_A_array, 1, queue );
    magma_setvector(num_batch, sizeof(double*), h_X_array, 1, st_leafmtxp->d_X_array, 1, queue );
    magma_setvector(num_batch, sizeof(double*), h_Y_array, 1, st_leafmtxp->d_Y_array, 1, queue );

    magma_imalloc(&st_leafmtxp->d_M, num_batch+1);
    magma_imalloc(&st_leafmtxp->d_N, num_batch+1);
    magma_imalloc(&st_leafmtxp->d_inc, num_batch+1);
    magma_setvector(num_batch, sizeof(magma_int_t), h_M, 1, st_leafmtxp->d_M, 1, queue );
    magma_setvector(num_batch, sizeof(magma_int_t), h_N, 1, st_leafmtxp->d_N, 1, queue );
    magma_setvector(num_batch, sizeof(magma_int_t), h_inc, 1, st_leafmtxp->d_inc, 1, queue );

    magma_queue_destroy( queue );

    st_leafmtxp->h_M = h_M;
    st_leafmtxp->h_N = h_N;
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
}

/////////////////////////////////////////////////////
// delete GPU memory
void  c_hacapk_adot_body_lfdel_batch_(stc_HACApK_leafmtxp *st_leafmtxp) {
    int ip; 
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

    magma_free(st_leafmtxp->d_X_array);
    magma_free(st_leafmtxp->d_Y_array);
    magma_free(st_leafmtxp->d_M);
    magma_free(st_leafmtxp->d_N);
    magma_free(st_leafmtxp->d_inc);
    magma_free(st_leafmtxp->d_A_array);

    magma_free_cpu(st_leafmtxp->h_M);
    magma_free_cpu(st_leafmtxp->h_N);
    magma_free_cpu(st_leafmtxp->h_A_array);
    magma_free_cpu(st_leafmtxp->h_X_array);
    magma_free_cpu(st_leafmtxp->h_Y_array);
}

