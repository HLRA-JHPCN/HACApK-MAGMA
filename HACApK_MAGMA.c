#include	<stdio.h>
#include	<stdlib.h>
#include	<time.h>
#include	"omp.h"
#include	"mpi.h"
#include	"HACApK_MAGMA.h"

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
#if defined(HAVE_MAGMA)

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
