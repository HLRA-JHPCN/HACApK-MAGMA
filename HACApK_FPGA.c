#include	<stdio.h>
#include	<stdlib.h>
#include	<time.h>
#include	"mpi.h"
#include	"HACApK_FPGA.h"
#include        <ISO_Fortran_binding.h>

//!***c_HACApK_adot_body_lfmtx
 void  c_hacapk_adot_body_lfmtx_
 (double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu){
 register int ip,il,it;
 int nlf,ndl,ndt,nstrtl,nstrtt,kt,itl,itt,ill;
 int st_lf_stride = st_leafmtxp->st_lf_stride;
 int a1size;
 
 nlf=st_leafmtxp->nlf;
  //fprintf(stderr,"nlf=%d \n",nlf);

 for(ip=0; ip<nlf; ip++){
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
   if(sttmp->ltmtx==1){ // compressed
     /**/
     double *a2tmp = (double *)((void*)(sttmp->a1)+sttmp->a1size);
     /**/
     kt=sttmp->kt; // rank
     
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
#ifdef HAVE_MAGMA
#define num_streams 5
#define max(a,b) (((a) > (b) ? (a) : (b)))

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
 
    #define GPU
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
    FILE *fp = fopen("sizes.dat","w");
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
            fprintf( fp, "%d %d %d\n",1,kt,ndt );
            #if defined(GPU)
            magmablas_dgemv(MagmaTrans, ndt, kt, 
                            one,  st_leafmtxp->mtx1_gpu[ip], ndt, 
                                 &(st_leafmtxp->zu_gpu[nstrtt-1]), ione,
                            zero, st_leafmtxp->zbu_gpu[stream_id], ione,
                            queue[stream_id] );
            dgemv_("T", &ndt, &kt, 
                   &one, sttmp->a1, &ndt, 
                         &zu[nstrtt-1], &ione,
                   &zero, zbu, &ione );
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
            fprintf( fp, "%d %d %d\n",1,ndl,kt );
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
            fprintf( fp, "%d %d %d\n",1,ndl,ndt );
            #if defined(GPU)
            magmablas_dgemv(MagmaTrans, ndt, ndl, 
                            one,  st_leafmtxp->mtx1_gpu[ip], ndt, 
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
        magma_daxpy( st_leafmtxp->m, one, st_leafmtxp->zau_gpu[ip], 1, st_leafmtxp->zau_gpu[0], 1, queue[0] );
    }
    // synch to get time
    magma_queue_sync( queue[0] );
    printf( " time: %.2e seconds\n",MPI_Wtime()-tic );
    // copy back
    magma_dgetvector( st_leafmtxp->m, st_leafmtxp->zau_gpu[0], 1, zau, 1, queue[0] );
    magma_queue_destroy( queue[0] );
    #else
    printf( " time: %.2e seconds\n",MPI_Wtime()-tic );
    #endif
    fclose(fp);
}

// copy blocks to GPU
void  c_hacapk_adot_body_lfcpy_gpu_(stc_HACApK_leafmtxp *st_leafmtxp) {
    // constants
    int ione = 1;
    double one = 1.0;
    double zero = 0.0;

    // local variables
    register int ip, il, it;
    int nlf, ndl, ndt, nstrtl, nstrtt, kt, itl, itt, ill;
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

            // copy U
            st_leafmtxp->mtx2_gpu[ip] = NULL;
            retval = magma_malloc( (void**) &(st_leafmtxp->mtx2_gpu[ip]), (ndl*kt)*sizeof(double) );
            if ( MAGMA_SUCCESS != retval ) {
                fprintf( stderr, "!!!! magma_malloc failed for leafmtxp[1][%d]\n", ip);
                exit(0);
            }
            magma_dsetmatrix( ndl, kt, a2tmp, ndl, st_leafmtxp->mtx2_gpu[ip], ndl, queue );
        } else if (sttmp->ltmtx == 2) { // full
            st_leafmtxp->mtx1_gpu[ip] = NULL;
            int retval = magma_malloc( (void**) &(st_leafmtxp->mtx1_gpu[ip]), (ndt*ndl)*sizeof(double) );
            if ( MAGMA_SUCCESS != retval ) {
                fprintf( stderr, "!!!! magma_malloc failed for leafmtxp[0][%d]\n", ip);
                exit(0);
            }
            magma_dsetmatrix( ndt, ndl, sttmp->a1, ndt, st_leafmtxp->mtx1_gpu[ip], ndt, queue );
            st_leafmtxp->mtx2_gpu[ip] = NULL;
        }
    }
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

    magma_queue_destroy( queue );
}

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
