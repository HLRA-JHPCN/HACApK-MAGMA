
#if defined(HAVE_MAGMA) | defined(HAVE_MAGMA_BATCH)

#include "HACApK_MAGMA.h"

void c_hacapk_adot_cax_lfmtx_comm_getvector(double *zau_gpu, double *zau,
                                            stc_HACApK_lcontrol *st_ctl, double *buffer, int *disps,
                                            double *wws, double *wwr, int *isct, int *irct, int nd,
                                            double *time_copy, double *time_mpi, magma_queue_t queue) {
    double tic;
    int *lpmd = (int*)((void*)st_ctl->param + st_ctl->lpmd_offset);
    int mpinr = lpmd[2];
    int nrank = lpmd[1];

    if (nrank > 1) {
        int *lsp = (int*)((void*)st_ctl->param + st_ctl->lsp_offset);
        int *lnp = (int*)((void*)st_ctl->param + st_ctl->lnp_offset);

        magma_queue_sync( queue );
        #if defined(PROF_MAGMA_BATCH)
        tic = MPI_Wtime();
        #endif
        int ione = 1;
        double zero = 0.0;
        lapackf77_dlaset( "F", &nd, &ione, &zero, &zero, zau, &nd );
        //#define USE_NONBLOCKINC_GATHER
        #if defined(USE_NONBLOCKINC_GATHER)
        magma_dgetvector( lnp[mpinr], &zau_gpu[lsp[mpinr]-1], 1, &zau[lsp[mpinr]-1], 1, queue );
        #else
        magma_dgetvector_async( lnp[mpinr], &zau_gpu[lsp[mpinr]-1], 1, &zau[lsp[mpinr]-1], 1, queue );
        #endif
        #if defined(PROF_MAGMA_BATCH)
        *time_copy += MPI_Wtime()-tic;
        #endif
    }
}


void c_hacapk_adot_cax_lfmtx_comm_mpi(double *zau_gpu, double *zau,
                                      stc_HACApK_lcontrol *st_ctl, double *buffer, int *disps,
                                      double *wws, double *wwr, int *isct, int *irct, int nd,
                                      double *time_copy, double *time_mpi, magma_queue_t queue,
                                      MPI_Request *request) {
    double tic;
    int *lpmd = (int*)((void*)st_ctl->param + st_ctl->lpmd_offset);
    int mpinr = lpmd[2];
    int nrank = lpmd[1];

    if (nrank > 1) {
        int *lsp = (int*)((void*)st_ctl->param + st_ctl->lsp_offset);
        int *lnp = (int*)((void*)st_ctl->param + st_ctl->lnp_offset);
        MPI_Comm icomm = MPI_COMM_WORLD;

        #ifdef USE_NONBLOCKINC_GATHER
        MPI_Iallgatherv(&zau[lsp[mpinr]-1], lnp[mpinr], MPI_DOUBLE, buffer, lnp, disps, MPI_DOUBLE, MPI_COMM_WORLD, request);
        #else
        int ic;
        int ncdp = (mpinr+1)%nrank;       // my destination neighbor
        int ncsp = (mpinr+nrank-1)%nrank; // my source neighbor
        tic = MPI_Wtime();
        MPI_Allgatherv(&zau[lsp[mpinr]-1], lnp[mpinr], MPI_DOUBLE, buffer, lnp, disps, MPI_DOUBLE, MPI_COMM_WORLD);

        int ione = 1;
        double one = 1.0;
        #if defined(REPRODUCIBLE_SUM) // !! make sure "reproduciblity" with some extra flops !!
        double zero = 0.0;
        lapackf77_dlaset( "F", &lnp[mpinr], &ione, &zero, &zero, &zau[lsp[mpinr]-1], &nd );
        for (ic=0; ic<nrank; ic++) {
           irct[0] = lnp[ic];
           irct[1] = lsp[ic];
           blasf77_daxpy( &irct[0], &one, &buffer[disps[ic]], &ione, &zau[irct[1]-1], &ione );
        }
        #else
        for (ic=1; ic<nrank; ic++) {
           int nctp = (ncsp-ic+nrank+1)%nrank; // where it came from
           irct[0] = lnp[nctp];
           irct[1] = lsp[nctp];
           blasf77_daxpy( &irct[0], &one, &buffer[disps[nctp]], &ione, &zau[irct[1]-1], &ione );
        }
        #endif
        *time_mpi += (MPI_Wtime()-tic);
        #endif
    }
}


void c_hacapk_adot_cax_lfmtx_comm_setvector(double *zau_gpu, double *zau,
                                            stc_HACApK_lcontrol *st_ctl, double *buffer, int *disps,
                                            double *wws, double *wwr, int *isct, int *irct, int nd,
                                            double *time_copy, double *time_mpi, magma_queue_t queue,
                                            MPI_Request *request) {
    double tic;
    int *lpmd = (int*)((void*)st_ctl->param + st_ctl->lpmd_offset);
    int mpinr = lpmd[2];
    int nrank = lpmd[1];

    if (nrank > 1) {
        #if defined(PROF_MAGMA_BATCH)
        tic = MPI_Wtime();
        #endif
        #ifdef USE_NONBLOCKINC_GATHER
        int ic;
        int ncdp = (mpinr+1)%nrank;       // my destination neighbor
        int ncsp = (mpinr+nrank-1)%nrank; // my source neighbor
        int *lsp = (int*)((void*)st_ctl->param + st_ctl->lsp_offset);
        int *lnp = (int*)((void*)st_ctl->param + st_ctl->lnp_offset);

        MPI_Status status;
        MPI_Wait(request, &status);

        int ione = 1;
        double one = 1.0;
        #if defined(REPRODUCIBLE_SUM) // !! make sure "reproduciblity" with some extra flops !!
        double zero = 0.0;
        lapackf77_dlaset( "F", &lnp[mpinr], &ione, &zero, &zero, &zau[lsp[mpinr]-1], &nd );
        for (ic=0; ic<nrank; ic++) {
           irct[0] = lnp[ic];
           irct[1] = lsp[ic];
           blasf77_daxpy( &irct[0], &one, &buffer[disps[ic]], &ione, &zau[irct[1]-1], &ione );
        }
        #else
        for (ic=1; ic<nrank; ic++) {
           int nctp = (ncsp-ic+nrank+1)%nrank; // where it came from
           irct[0] = lnp[nctp];
           irct[1] = lsp[nctp];
           blasf77_daxpy( &irct[0], &one, &buffer[disps[nctp]], &ione, &zau[irct[1]-1], &ione );
        }
        #endif
        #endif
        magma_dsetvector_async( nd, zau, 1, zau_gpu, 1, queue );
        //magma_dsetvector( nd, zau, 1, zau_gpu, 1, queue );
        #if defined(PROF_MAGMA_BATCH)
        *time_copy += MPI_Wtime()-tic;
        #endif
    }
}

///////////////////////////////////////////////////////////////////////////
// pipelined version
// 
//#define hacapck_magma_dgemv magmablas_dgemv
#define hacapck_magma_dgemv magma_dgemv
// on one GPU / proc (original)
void c_hacapk_bicgstab_cax_lfmtx_pipe_(stc_HACApK_leafmtxp *st_leafmtxp, stc_HACApK_lcontrol *st_ctl,
                                       double *u_cpu, double *b_cpu, double*param, int *nd, int *nstp, int *lrtrn) {
    // local constants
    int ione  = 1;
    int izero = 0;
    double zero =  0.0;
    double one  =  1.0;
    double mone = -1.0;
    // local arrays
    double *zz, *zw, *zr0;
    double *zr, *zv, *zp, *zt, *zx, *zy, *zs, *zb, *zq;
    double *b, *u, *wws, *wwr;
    double *dnorm, *hnorm;
    double *zau_cpu, *wws_cpu, *wwr_cpu;
    int *lpmd = (int*)((void*)st_ctl->param + st_ctl->lpmd_offset);
    int isct[2], irct[2];
    // local variables
    double zwnorm, zsnorm, zznorm;
    double eps, alpha, beta, zeta, zden, znorm, znormold, bnorm, zrnorm;
    double en_measure_time, st_measure_time, time;
    int info, step, mstep;
    int mpinr, nrank, ierr;
    double time_spmv, time_mpi, time_batch, time_set, time_copy, tic;
 
    MPI_Comm icomm = MPI_COMM_WORLD; //lpmd[0];
    mstep = param[82];
    eps = param[90];
    mpinr = lpmd[2]; 
    nrank = lpmd[1]; 
    MPI_Barrier( icomm );

    int on_gpu = 1;
    magma_device_t cdev;
    magma_queue_t queue;

    magma_setdevice( get_device_id(st_leafmtxp) );
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    if (MAGMA_SUCCESS != magma_dmalloc(&u, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&b, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zq,(*nd)*2) ||
        MAGMA_SUCCESS != magma_dmalloc(&zt, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zr,(*nd)*4) ||
        MAGMA_SUCCESS != magma_dmalloc(&zp, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zb, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zx, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zv, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&wws, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&wwr, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zr0, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&dnorm, 5)) {
      printf( " failed to allocate vectors (nd=%d)\n",*nd );
    }
    zy = &zq[*nd];
    zw = &zr[*nd];
    zs = &zw[*nd];
    zz = &zs[*nd];
    // use pinned memory for buffer
    //wws_cpu = (double*)malloc((*nd) * sizeof(double));
    //wwr_cpu = (double*)malloc((*nd) * sizeof(double));
    if (MAGMA_SUCCESS != magma_dmalloc_pinned(&wws_cpu, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc_pinned(&wwr_cpu, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc_pinned(&zau_cpu, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc_pinned(&hnorm, 4)) {
        printf( " failed to allocate pinned vectors (nd=%d)\n",*nd );
    }
    // MPI buffer
    double *buffer = NULL;
    int *disps = NULL;
    c_hacapk_adot_cax_lfmtx_comm_setup(st_ctl, &buffer, &disps);

    MPI_Barrier( icomm );
    #ifdef WARMUP_MPI
    c_hacapk_adot_cax_lfmtx_warmup(st_ctl, zau_cpu, wws_cpu, wwr_cpu, *nd);
    #endif
    // copy matrix to GPU
    c_hacapk_adot_body_lfcpy_batch_sorted_(nd, st_leafmtxp);
    //#define DIAGONAL_SCALE
    #ifdef DIAGONAL_SCALE
    // create handle
    cublasHandle_t handle = magma_queue_get_cublas_handle( queue );

    int nlf=st_leafmtxp->nlf;
    int st_lf_stride = st_leafmtxp->st_lf_stride;
    double *D  = (double*)malloc((*nd) * sizeof(double));
    double *Dr = (double*)malloc((*nd) * sizeof(double));

    double *d;
    double *t;
    int ip, jj;
    for (jj=0; jj<(*nd); jj++) {
        D[jj] = 0.0;
    }
    for(ip=0; ip<nlf; ip++){
        stc_HACApK_leafmtx *sttmp;
        sttmp = (stc_HACApK_leafmtx *)((void *)(st_leafmtxp->st_lf) + st_lf_stride * ip);
        int nstrtl = sttmp->nstrtl-1; // i: index of first row (1-base)
        int nstrtt = sttmp->nstrtt-1; // j: index of first column (1-base)
        if (nstrtl == nstrtt) {
            int ndl = sttmp->ndl; // m: number of rows
            for (jj=0; jj<ndl; jj++) {
                D[nstrtl + jj] = sttmp->a1[jj + jj*ndl];
            }
        }
    }
    MPI_Allreduce(D, Dr, *nd, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (jj=0; jj<(*nd); jj++) {
        Dr[jj] = 1.0/sqrt(fabs(Dr[jj]));
    }
    magma_dmalloc(&d, *nd);
    magma_dmalloc(&t, *nd);
    magma_dsetvector( *nd, Dr, 1, d, 1, queue );
    free(D); free(Dr);
    #endif
    //
    time_spmv = 0.0;
    time_mpi = 0.0;
    time_batch = 0.0;
    time_set = 0.0;
    time_copy = 0.0;
    MPI_Barrier( icomm );
    st_measure_time = MPI_Wtime();
    // copy the input vector to GPU
    magma_dsetvector_async( *nd, b_cpu, 1, b, 1, queue );
    magma_dsetvector_async( *nd, u_cpu, 1, u, 1, queue );
    // init
    alpha = 0.0; beta = 0.0; zeta = 0.0;
    #if defined(DDOT_BY_DGEMV)
    hacapck_magma_dgemv( MagmaTrans, *nd, 1,
                         one,  b, *nd,
                               b, ione,
                         zero, dnorm, ione,
                         queue );
    magma_dgetvector( 1, dnorm, 1, hnorm, 1, queue );
    bnorm = hnorm[0];
    #else
    bnorm = magma_ddot(*nd, b, ione, b, ione, queue); 
    #endif
    bnorm=sqrt(bnorm);
    //  .. SpMV: zv=A*u ..
    magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zv, *nd, queue );
    magma_queue_sync( queue );
    tic = MPI_Wtime();
    #ifdef DIAGONAL_SCALE
    // t = D*u, and zv=A*t
    cublasDsbmv(handle, CUBLAS_FILL_MODE_LOWER, *nd, izero, &one, d, ione, u, ione, &zero, t, ione);
    c_hacapk_adot_body_lfmtx_batch_queue(zv,st_leafmtxp,t,wws, &time_batch,&time_set,&time_copy, 
                                         on_gpu, queue, NULL, NULL);
    #else
    c_hacapk_adot_body_lfmtx_batch_queue(zv,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy, 
                                         on_gpu, queue, NULL, NULL);
    #endif
    magma_queue_sync( queue );
    time_spmv += (MPI_Wtime()-tic);
    c_hacapk_adot_cax_lfmtx_comm_gpu(1, zv, zau_cpu, st_ctl,buffer,disps,
                                     wws_cpu, wwr_cpu, isct, irct, *nd, 
                                     &time_copy,&time_mpi, queue);
    // zr = zr - zv
    magmablas_dlacpy( MagmaFull, *nd, ione, b, *nd, zr, *nd, queue );
    magma_daxpy( *nd, mone, zv, ione, zr, ione, queue );
    magmablas_dlacpy( MagmaFull, *nd, ione, zr, *nd, zr0, *nd, queue );
    //  .. SpMV: zw = A*zr ..
    magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zw, *nd, queue );
    magma_queue_sync( queue );
    tic = MPI_Wtime();
    #ifdef DIAGONAL_SCALE
    // t= D*zr, and zw=A*t
    cublasDsbmv(handle, CUBLAS_FILL_MODE_LOWER, *nd, izero, &one, d, ione, zr, ione, &zero, t, ione);
    c_hacapk_adot_body_lfmtx_batch_queue(zw,st_leafmtxp,t,wws, &time_batch,&time_set,&time_copy, 
                                         on_gpu, queue, NULL, NULL);
    #else
    c_hacapk_adot_body_lfmtx_batch_queue(zw,st_leafmtxp,zr,wws, &time_batch,&time_set,&time_copy, 
                                         on_gpu, queue, NULL, NULL);
    #endif
    magma_queue_sync( queue );
    time_spmv += (MPI_Wtime()-tic);
    c_hacapk_adot_cax_lfmtx_comm_gpu(1, zw, zau_cpu, st_ctl,buffer,disps,
                                     wws_cpu, wwr_cpu, isct, irct, *nd, 
                                     &time_copy,&time_mpi, queue);
    /*{
        double *zw_local = (double*)malloc(*nd * sizeof(double));
        double *zw_min   = (double*)malloc(*nd * sizeof(double));
        double *zw_max   = (double*)malloc(*nd * sizeof(double));
        magma_dgetvector( *nd, zw, 1, zw_local, 1, queue );
        MPI_Allreduce(zw_local, zw_max, *nd, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
        MPI_Allreduce(zw_local, zw_min, *nd, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );
        FILE *fp;
        char filename[100];
        sprintf(filename,"zw_%d.dat",mpinr);
        fp = fopen(filename,"w");
        int ii;
        for (ii=disps[mpinr]; ii<disps[mpinr+1]; ii++) {
            if (zw_max[ii] != zw_min[ii]) fprintf(fp, " %d: %.2e - %.2e = %.2e, %.2e, %.2e\n", ii,
                                                   zw_max[ii],zw_min[ii], zw_max[ii]-zw_min[ii],
                                                   zw_max[ii]-zw_local[ii],zw_min[ii]-zw_local[ii]);
        }
        fclose(fp);
        //MPI_Barrier(MPI_COMM_WORLD); MPI_Finalize(); exit(0);
    }*/
    //  .. SpMV: zt = A*zw ..
    magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zt, *nd, queue );
    magma_queue_sync( queue );
    tic = MPI_Wtime();
    #ifdef DIAGONAL_SCALE
    // t=D*zw, and zt=A*t
    cublasDsbmv(handle, CUBLAS_FILL_MODE_LOWER, *nd, izero, &one, d, ione, zw, ione, &zero, t, ione);
    c_hacapk_adot_body_lfmtx_batch_queue(zt,st_leafmtxp,t,wws, &time_batch,&time_set,&time_copy, 
                                         on_gpu, queue, NULL, NULL);
    #else
    c_hacapk_adot_body_lfmtx_batch_queue(zt,st_leafmtxp,zw,wws, &time_batch,&time_set,&time_copy, 
                                         on_gpu, queue, NULL, NULL);
    #endif
    magma_queue_sync( queue );
    time_spmv += (MPI_Wtime()-tic);
    c_hacapk_adot_cax_lfmtx_comm_gpu(1, zt, zau_cpu, st_ctl,buffer,disps,
                                     wws_cpu, wwr_cpu, isct, irct, *nd, 
                                     &time_copy,&time_mpi, queue);
    //
    #if defined(DDOT_BY_DGEMV)
    hacapck_magma_dgemv( MagmaTrans, *nd, 2,
                         one,  zr, *nd,
                               zr, ione,
                         zero, dnorm, ione,
                         queue );
    magma_dgetvector( 2, dnorm, 1, hnorm, 1, queue );
    znorm  = hnorm[0];
    zrnorm = hnorm[1];
    #else
    znorm  = magma_ddot(*nd, zr, ione, zr, ione, queue); 
    zrnorm = magma_ddot(*nd, zw, ione, zr, ione, queue); 
    #endif
    alpha = znorm/zrnorm;
    /*{
        double znorm_max, znorm_min;
        MPI_Allreduce(&znorm, &znorm_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
        MPI_Allreduce(&znorm, &znorm_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );
        double zrnorm_max, zrnorm_min;
        MPI_Allreduce(&zrnorm, &zrnorm_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
        MPI_Allreduce(&zrnorm, &zrnorm_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );
        if (mpinr == 0) {
            znorm_max -= znorm_min;
            zrnorm_max -= zrnorm_min;
            printf( " alpha=%.2e/%.2e=%.2e (%.2e,%.2e)\n",znorm,zrnorm, alpha, znorm_max,zrnorm_max );
        }
        //if (mpinr == 0) magma_dprint_gpu(*nd,1, zt,*nd,queue);
    }*/
    zrnorm = sqrt(znorm);
    if (mpinr == 0) {
        printf( "\n ** pipelined BICG (version 1) with MAGMA batched on GPU **\n" );
        printf( "\nOriginal relative residual norm = %.2e/%.2e = %.2e\n",zrnorm,bnorm,zrnorm/bnorm );
        printf( "HACApK_bicgstab_lfmtx_pipe start\n" );
    }
    for ( step=1; step<=mstep; step++ ) {
        if (zrnorm/bnorm < eps) break;
        if (step > 1) { 
            // zp(:nd) = zr(:nd) + beta*(zp(:nd) - zeta*zs(:nd))
            magma_daxpy( *nd, -zeta, zs, ione, zp, ione, queue );
            magmablas_dlascl( MagmaFull, ione, ione, one, beta, *nd, ione, zp, *nd, queue, &info );
            magma_daxpy( *nd, one, zr, ione, zp, ione, queue );
            // zs(:nd) = zw(:nd) + beta*(zs(:nd) - zeta*zz(:nd))
            magma_daxpy( *nd, -zeta, zz, ione, zs, ione, queue );
            magmablas_dlascl( MagmaFull, ione, ione, one, beta, *nd, ione, zs, *nd, queue, &info );
            magma_daxpy( *nd, one, zw, ione, zs, ione, queue );
            // zz(:nd) = zt(:nd) + beta*(z(:nd) - zeta*zv(:nd))
            magma_daxpy( *nd, -zeta, zv, ione, zz, ione, queue );
            magmablas_dlascl( MagmaFull, ione, ione, one, beta, *nd, ione, zz, *nd, queue, &info );
            magma_daxpy( *nd, one, zt, ione, zz, ione, queue );
        } else {
            magmablas_dlacpy( MagmaFull, *nd, ione, zr, *nd, zp, *nd, queue );
            magmablas_dlacpy( MagmaFull, *nd, ione, zw, *nd, zs, *nd, queue );
            magmablas_dlacpy( MagmaFull, *nd, ione, zt, *nd, zz, *nd, queue );
        }
        // zq(:nd) = zr(:nd) - alpha*zs(:nd)
        magmablas_dlacpy( MagmaFull, *nd, ione, zr, *nd, zq, *nd, queue );
        magma_daxpy( *nd, -alpha, zs, ione, zq, ione, queue );
        // zy(:nd) = zw(:nd) - alpha*zz(:nd)
        magmablas_dlacpy( MagmaFull, *nd, ione, zw, *nd, zy, *nd, queue );
        magma_daxpy( *nd, -alpha, zz, ione, zy, ione, queue );
        // [znorm, zden] = zy'*[zq, zy]
        #if defined(DDOT_BY_DGEMV)
        hacapck_magma_dgemv( MagmaTrans, *nd, 2,
                             one,  zq, *nd,
                                   zy, ione,
                             zero, dnorm, ione,
                             queue );
        magma_dgetvector( 2, dnorm, 1, hnorm, 1, queue );
        zrnorm = hnorm[0];
        zden   = hnorm[1];
        #else
        zrnorm = magma_ddot(*nd, zy, ione, zq, ione, queue); 
        zden   = magma_ddot(*nd, zy, ione, zy, ione, queue);
        #endif
        zeta = zrnorm/zden;
        /*{
            double zrnorm_max, zrnorm_min;
            MPI_Allreduce(&zrnorm, &zrnorm_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
            MPI_Allreduce(&zrnorm, &zrnorm_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );
            double zden_max, zden_min;
            MPI_Allreduce(&zden, &zden_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
            MPI_Allreduce(&zden, &zden_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );

            double *zy_local = (double*)malloc(*nd * sizeof(double));
            double *zy_min   = (double*)malloc(*nd * sizeof(double));
            double *zy_max   = (double*)malloc(*nd * sizeof(double));
            magma_dgetvector( *nd, zy, 1, zy_local, 1, queue );
            MPI_Allreduce(zy_local, zy_max, *nd, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
            MPI_Allreduce(zy_local, zy_min, *nd, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );
            if (mpinr == 0) {
                int ii;
                for (ii=0; ii<*nd; ii++) if (zy_max[ii] != zy_min[ii]) printf( " %d: %.2e - %.2e = %.2e\n",ii,zy_max[ii],zy_min[ii],zy_max[ii]-zy_min[ii] );
                zrnorm_max -= zrnorm_min;
                zden_max -= zden_min;
                printf( " %d: zeta=%.2e/%.2e=%.2e, alpha=%.2e (%.2e,%.2e)\n",step,zrnorm,zden,zeta, alpha, zrnorm_max,zden_max );
            }
            free(zy_local); free(zy_min); free(zy_max);
        }*/
        //  .. SpMV: zv = A*zz ..
        magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zv, *nd, queue );
        magma_queue_sync( queue );
        tic = MPI_Wtime();
        #ifdef DIAGONAL_SCALE
        // t=D*zz, and zv=A*t
        cublasDsbmv(handle, CUBLAS_FILL_MODE_LOWER, *nd, izero, &one, d, ione, zz, ione, &zero, t, ione);
        c_hacapk_adot_body_lfmtx_batch_queue(zv,st_leafmtxp,t,wws, &time_batch,&time_set,&time_copy,
                                             on_gpu, queue, NULL, NULL);
        #else
        magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zv, *nd, queue );
        magma_queue_sync( queue );
        tic = MPI_Wtime();
        c_hacapk_adot_body_lfmtx_batch_queue(zv,st_leafmtxp,zz,wws, &time_batch,&time_set,&time_copy,
                                             on_gpu, queue, NULL, NULL);
        #endif
        magma_queue_sync( queue );
        time_spmv += (MPI_Wtime()-tic);
        c_hacapk_adot_cax_lfmtx_comm_gpu(1, zv, zau_cpu, st_ctl,buffer,disps,
                                         wws_cpu,wwr_cpu, isct,irct,*nd, 
                                         &time_copy,&time_mpi, queue);
        // zx(:nd) = zx(:nd) + alpha*zp(:nd) + zeta*zq(:nd)
        magma_daxpy( *nd, alpha, zp, ione, zx, ione, queue );
        magma_daxpy( *nd,  zeta, zq, ione, zx, ione, queue );
        // zr(:nd) = zq(:nd) - zeta*zy(:nd)
        magmablas_dlacpy( MagmaFull, *nd, ione, zq, *nd, zr, *nd, queue );
        magma_daxpy( *nd, -zeta, zy, ione, zr, ione, queue );
        // zw(:nd) = zy(:nd) + zeta*(zt(:nd) - alpha*zv(:nd))
        magmablas_dlacpy( MagmaFull, *nd, ione, zt, *nd, zw, *nd, queue );
        magma_daxpy( *nd, -alpha, zv, ione, zw, ione, queue );
        magmablas_dlascl( MagmaFull, ione, ione, one, -zeta, *nd, ione, zw, *nd, queue, &info );
        magma_daxpy( *nd, one, zy, ione, zw, ione, queue );
        // all-reduces
        // > znorm = zr'*zr0
        znormold = znorm;
        #if defined(DDOT_BY_DGEMV)
        hacapck_magma_dgemv( MagmaTrans, *nd, 4,
                             one,  zr, *nd,
                                   zr0, ione,
                             zero, dnorm, ione,
                             queue );
        // > znorm = zr'*zr
        hacapck_magma_dgemv( MagmaTrans, *nd, 1,
                             one,  zr, *nd,
                                   zr, ione,
                             zero, &dnorm[4], ione,
                             queue );
        magma_dgetvector( 5, dnorm, 1, hnorm, 1, queue );
        znorm  = hnorm[0];
        zwnorm = hnorm[1];
        zsnorm = hnorm[2];
        zznorm = hnorm[3];
        zrnorm = hnorm[4];
        #else
        znorm  = magma_ddot(*nd, zr, ione, zr0, ione, queue); 
        zwnorm = magma_ddot(*nd, zw, ione, zr0, ione, queue); 
        zsnorm = magma_ddot(*nd, zs, ione, zr0, ione, queue); 
        zznorm = magma_ddot(*nd, zz, ione, zr0, ione, queue); 
        // > znorm = zr'*zr
        zrnorm = magma_ddot(*nd, zr, ione, zr, ione, queue); 
        #endif
        /*{
            double znorm_max, znorm_min;
            MPI_Allreduce(&znorm, &znorm_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
            MPI_Allreduce(&znorm, &znorm_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );
            double zwnorm_max, zwnorm_min;
            MPI_Allreduce(&zwnorm, &zwnorm_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
            MPI_Allreduce(&zwnorm, &zwnorm_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );
            double zsnorm_max, zsnorm_min;
            MPI_Allreduce(&zsnorm, &zsnorm_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
            MPI_Allreduce(&zsnorm, &zsnorm_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );
            double zznorm_max, zznorm_min;
            MPI_Allreduce(&zznorm, &zznorm_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
            MPI_Allreduce(&zznorm, &zznorm_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );
            if (mpinr == 0) {
                znorm_max -= znorm_min;
                zwnorm_max -= zwnorm_min;
                zsnorm_max -= zsnorm_min;
                zznorm_max -= zznorm_min;
                printf( "   : %.2e, %.2e, %.2e, %.2e\n",znorm_max,zwnorm_max,zsnorm_max,zznorm_max );
                printf( " %d: znorm=%.2e zwnorm=%.2e zsnorm=%.2e nnzorm=%.2e\n",step,znorm,zwnorm,zsnorm,zznorm );
            }
        }*/
        // beta
        beta = (alpha/zeta)*(znorm/znormold);
        alpha = znorm/(zwnorm+beta*zsnorm-beta*zeta*zznorm);
        /*{
            double zrnorm_max, zrnorm_min;
            MPI_Allreduce(&zrnorm, &zrnorm_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
            MPI_Allreduce(&zrnorm, &zrnorm_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );
            zrnorm_max -= zrnorm_min;
            if (mpinr == 0) {
                printf( "    : %.2e\n",zrnorm_max );
            }
        }*/
        zrnorm = sqrt(zrnorm);
        *nstp = step;
        en_measure_time = MPI_Wtime();
        time = en_measure_time - st_measure_time;
        /*{
            double bnorm_max, bnorm_min;
            MPI_Allreduce(&bnorm, &bnorm_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
            MPI_Allreduce(&bnorm, &bnorm_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );
            double zrnorm_max, zrnorm_min;
            MPI_Allreduce(&zrnorm, &zrnorm_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
            MPI_Allreduce(&zrnorm, &zrnorm_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );
            zrnorm_max -= zrnorm_min;
            bnorm_max -= bnorm_min;
            if (mpinr == 0) {
                printf( "   : %.2e, %.2e\n",zrnorm_max,bnorm_max );
            }
        }*/
        if (st_ctl->param[0] > 0 && mpinr == 0) {
            printf( " %d: time=%.2e log10(zrnorm/bnorm)=log10(%.2e/%.2e)=%.2e\n",step,time,zrnorm,bnorm,log10(zrnorm/bnorm) );
        }
        //  .. SpMV: zt = A*zw ..
        magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zt, *nd, queue );
        magma_queue_sync( queue );
        tic = MPI_Wtime();
        #ifdef DIAGONAL_SCALE
        // t=D*zw, and zt=A*t
        cublasDsbmv(handle, CUBLAS_FILL_MODE_LOWER, *nd, izero, &one, d, ione, zw, ione, &zero, t, ione);
        c_hacapk_adot_body_lfmtx_batch_queue(zt,st_leafmtxp,t,wws, &time_batch,&time_set,&time_copy,
                                             on_gpu, queue, NULL, NULL);
        #else
        c_hacapk_adot_body_lfmtx_batch_queue(zt,st_leafmtxp,zw,wws, &time_batch,&time_set,&time_copy,
                                             on_gpu, queue, NULL, NULL);
        #endif
        magma_queue_sync( queue );
        time_spmv += (MPI_Wtime()-tic);
        c_hacapk_adot_cax_lfmtx_comm_gpu(1, zt,zau_cpu, st_ctl,buffer,disps,
                                         wws_cpu,wwr_cpu, isct,irct,*nd, 
                                         &time_copy,&time_mpi, queue);
    }
    magma_dgetvector( *nd, u, 1, u_cpu, 1, queue );
    MPI_Barrier( icomm );
    en_measure_time = MPI_Wtime();
    time = en_measure_time - st_measure_time;
    if (st_ctl->param[0] > 0) {
        //printf( " End: %d, %.2e\n",mpinr,time );
        if (mpinr == 0) {
            printf( "       BiCG        = %.5e\n", time );
            printf( "        time_mpi   = %.5e\n", time_mpi );
            printf( "        time_copy  = %.5e\n", time_copy );
            printf( "        time_spmv  = %.5e\n", time_spmv );
            printf( "        > time_batch = %.5e\n", time_batch );
            printf( "        > time_set   = %.5e\n", time_set );
        }
    }
    magma_queue_destroy( queue );
    // delete matrix
    c_hacapk_adot_body_lfdel_batch_(st_leafmtxp);

    // free cpu memory
    if (buffer != NULL) free(buffer);
    free(disps);
    magma_free_pinned(wws_cpu);
    magma_free_pinned(wwr_cpu);
    magma_free_pinned(zau_cpu);

    // free gpu memory
    magma_free(u);
    magma_free(b);

    magma_free(wws);
    magma_free(wwr);

    magma_free(zr0);

    magma_free(zt);
    magma_free(zr);
    magma_free(zp);
    magma_free(zq);
    magma_free(zb);
    magma_free(zx);
    magma_free(zv);
}

// on one GPU / proc (to overlap allgather with ddots)
void c_hacapk_bicgstab_cax_lfmtx_pipe2_(stc_HACApK_leafmtxp *st_leafmtxp, stc_HACApK_lcontrol *st_ctl,
                                        double *u_cpu, double *b_cpu, double*param, int *nd, int *nstp, int *lrtrn) {
    // local constants
    int ione  = 1;
    int izero = 0;
    double zero =  0.0;
    double one  =  1.0;
    double mone = -1.0;
    // local arrays
    double *zz, *zw, *zr0;
    double *zr, *zv, *zp, *zt, *zx, *zy, *zs, *zb, *zq;
    double *b, *u, *wws, *wwr;
    double *dnorm, *hnorm;
    double *zau_cpu, *wws_cpu, *wwr_cpu;
    int *lpmd = (int*)((void*)st_ctl->param + st_ctl->lpmd_offset);
    int isct[2], irct[2];
    // local variables
    double zwnorm, zsnorm, zznorm;
    double eps, alpha, beta, zeta, zden, znorm, znormold, bnorm, zrnorm;
    double en_measure_time, st_measure_time, time;
    int info, step, mstep;
    int mpinr, nrank, ierr;
    double time_spmv, time_mpi, time_batch, time_set, time_copy, time_dot, tic;
 
    MPI_Comm icomm = MPI_COMM_WORLD; //lpmd[0];
    mstep = param[82];
    eps = param[90];
    mpinr = lpmd[2]; 
    nrank = lpmd[1]; 
    MPI_Barrier( icomm );

    int on_gpu = 1;
    magma_device_t cdev;
    magma_event_t event;
    magma_queue_t queue, queue_comm;

    magma_setdevice( get_device_id(st_leafmtxp) );
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );
    magma_queue_create( cdev, &queue_comm );
    magma_event_create( &event );

    if (MAGMA_SUCCESS != magma_dmalloc(&u, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&b, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zq,(*nd)*2) ||
        MAGMA_SUCCESS != magma_dmalloc(&zt, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zr,(*nd)*4) ||
        MAGMA_SUCCESS != magma_dmalloc(&zp, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zb, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zx, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zv, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&wws, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&wwr, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zr0, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&dnorm, 5)) {
      printf( " failed to allocate vectors (nd=%d)\n",*nd );
    }
    zy = &zq[*nd];
    zw = &zr[*nd];
    zs = &zw[*nd];
    zz = &zs[*nd];
    // use pinned memory for buffer
    //wws_cpu = (double*)malloc((*nd) * sizeof(double));
    //wwr_cpu = (double*)malloc((*nd) * sizeof(double));
    if (MAGMA_SUCCESS != magma_dmalloc_pinned(&wws_cpu, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc_pinned(&wwr_cpu, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc_pinned(&zau_cpu, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc_pinned(&hnorm, 5)) {
        printf( " failed to allocate pinned vectors (nd=%d)\n",*nd );
    }
    // MPI buffer
    double *buffer = NULL;
    int *disps = NULL;
    c_hacapk_adot_cax_lfmtx_comm_setup(st_ctl, &buffer, &disps);

    MPI_Barrier( icomm );
    #ifdef WARMUP_MPI
    c_hacapk_adot_cax_lfmtx_warmup(st_ctl, zau_cpu, wws_cpu, wwr_cpu, *nd);
    #endif
    // copy matrix to GPU
    c_hacapk_adot_body_lfcpy_batch_sorted_(nd, st_leafmtxp);
    //#define DIAGONAL_SCALE
    #ifdef DIAGONAL_SCALE
    // create handle
    cublasHandle_t handle = magma_queue_get_cublas_handle( queue );

    int nlf=st_leafmtxp->nlf;
    int st_lf_stride = st_leafmtxp->st_lf_stride;
    double *D  = (double*)malloc((*nd) * sizeof(double));
    double *Dr = (double*)malloc((*nd) * sizeof(double));

    double *d;
    double *t;
    int ip, jj;
    for (jj=0; jj<(*nd); jj++) {
        D[jj] = 0.0;
    }
    for(ip=0; ip<nlf; ip++){
        stc_HACApK_leafmtx *sttmp;
        sttmp = (stc_HACApK_leafmtx *)((void *)(st_leafmtxp->st_lf) + st_lf_stride * ip);
        int nstrtl = sttmp->nstrtl-1; // i: index of first row (1-base)
        int nstrtt = sttmp->nstrtt-1; // j: index of first column (1-base)
        if (nstrtl == nstrtt) {
            int ndl = sttmp->ndl; // m: number of rows
            for (jj=0; jj<ndl; jj++) {
                D[nstrtl + jj] = sttmp->a1[jj + jj*ndl];
            }
        }
    }
    MPI_Allreduce(D, Dr, *nd, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (jj=0; jj<(*nd); jj++) {
        Dr[jj] = 1.0/sqrt(fabs(Dr[jj]));
    }
    magma_dmalloc(&d, *nd);
    magma_dmalloc(&t, *nd);
    magma_dsetvector( *nd, Dr, 1, d, 1, queue );
    free(D); free(Dr);
    #endif
    //
    time_spmv = 0.0;
    time_mpi = 0.0;
    time_batch = 0.0;
    time_set = 0.0;
    time_copy = 0.0;
    time_dot = 0.0;
    MPI_Barrier( icomm );
    st_measure_time = MPI_Wtime();
    // copy the input vector to GPU
    magma_dsetvector_async( *nd, b_cpu, 1, b, 1, queue );
    magma_dsetvector_async( *nd, u_cpu, 1, u, 1, queue );
    // init
    alpha = 0.0; beta = 0.0; zeta = 0.0;
    #if defined(PROF_MAGMA_BATCH)
    magma_queue_sync( queue );
    tic = MPI_Wtime();
    #endif
    #if defined(DDOT_BY_DGEMV)
    magma_dgemv( MagmaTrans, *nd, 1,
                 one,  b, *nd,
                       b, ione,
                 zero, dnorm, ione,
                 queue );
    magma_dgetvector( 1, dnorm, 1, hnorm, 1, queue );
    bnorm = hnorm[0];
    #else
    bnorm = magma_ddot(*nd, b, ione, b, ione, queue); 
    #endif
    #if defined(PROF_MAGMA_BATCH)
    magma_queue_sync( queue );
    time_dot += (MPI_Wtime()-tic);
    #endif
    bnorm=sqrt(bnorm);
    //  .. SpMV: zv=A*u ..
    //magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zv, *nd, queue );
    #if defined(PROF_MAGMA_BATCH)
    magma_queue_sync( queue );
    tic = MPI_Wtime();
    #endif
    #ifdef DIAGONAL_SCALE
    // t = D*u, and zv=A*t
    cublasDsbmv(handle, CUBLAS_FILL_MODE_LOWER, *nd, izero, &one, d, ione, u, ione, &zero, t, ione);
    c_hacapk_adot_body_lfmtx_batch_queue(zv,st_leafmtxp,t,wws, &time_batch,&time_set,&time_copy, 
                                         on_gpu, queue, NULL, NULL);
    #else
    c_hacapk_adot_body_lfmtx_batch_queue(zv,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy, 
                                         on_gpu, queue, NULL, NULL);
    #endif
    #if defined(PROF_MAGMA_BATCH)
    magma_queue_sync( queue );
    time_spmv += (MPI_Wtime()-tic);
    #endif
    c_hacapk_adot_cax_lfmtx_comm_gpu(1, zv, zau_cpu, st_ctl,buffer,disps,
                                     wws_cpu, wwr_cpu, isct, irct, *nd, 
                                     &time_copy,&time_mpi, queue);
    // zr = zr - zv
    magmablas_dlacpy( MagmaFull, *nd, ione, b, *nd, zr, *nd, queue );
    magma_daxpy( *nd, mone, zv, ione, zr, ione, queue );
    magmablas_dlacpy( MagmaFull, *nd, ione, zr, *nd, zr0, *nd, queue );
    //  .. SpMV: zw = A*zr ..
    //magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zw, *nd, queue );
    #if defined(PROF_MAGMA_BATCH)
    magma_queue_sync( queue );
    tic = MPI_Wtime();
    #endif
    #ifdef DIAGONAL_SCALE
    // t= D*zr, and zw=A*t
    cublasDsbmv(handle, CUBLAS_FILL_MODE_LOWER, *nd, izero, &one, d, ione, zr, ione, &zero, t, ione);
    c_hacapk_adot_body_lfmtx_batch_queue(zw,st_leafmtxp,t,wws, &time_batch,&time_set,&time_copy, 
                                         on_gpu, queue, NULL, NULL);
    #else
    c_hacapk_adot_body_lfmtx_batch_queue(zw,st_leafmtxp,zr,wws, &time_batch,&time_set,&time_copy, 
                                         on_gpu, queue, NULL, NULL);
    #endif
    #if defined(PROF_MAGMA_BATCH)
    magma_queue_sync( queue );
    time_spmv += (MPI_Wtime()-tic);
    #endif
    c_hacapk_adot_cax_lfmtx_comm_gpu(1, zw, zau_cpu, st_ctl,buffer,disps,
                                     wws_cpu, wwr_cpu, isct, irct, *nd, 
                                     &time_copy,&time_mpi, queue);
    //  .. SpMV: zt = A*zw ..
    //magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zt, *nd, queue );
    #if defined(PROF_MAGMA_BATCH)
    magma_queue_sync( queue );
    tic = MPI_Wtime();
    #endif
    #ifdef DIAGONAL_SCALE
    // t=D*zw, and zt=A*t
    cublasDsbmv(handle, CUBLAS_FILL_MODE_LOWER, *nd, izero, &one, d, ione, zw, ione, &zero, t, ione);
    c_hacapk_adot_body_lfmtx_batch_queue(zt,st_leafmtxp,t,wws, &time_batch,&time_set,&time_copy, 
                                         on_gpu, queue, NULL, NULL);
    #else
    c_hacapk_adot_body_lfmtx_batch_queue(zt,st_leafmtxp,zw,wws, &time_batch,&time_set,&time_copy, 
                                         on_gpu, queue, NULL, NULL);
    #endif
    #if defined(PROF_MAGMA_BATCH)
    magma_queue_sync( queue );
    time_spmv += (MPI_Wtime()-tic);
    #endif
    c_hacapk_adot_cax_lfmtx_comm_gpu(1, zt, zau_cpu, st_ctl,buffer,disps,
                                     wws_cpu, wwr_cpu, isct, irct, *nd, 
                                     &time_copy,&time_mpi, queue);
    //
    #if defined(PROF_MAGMA_BATCH)
    magma_queue_sync( queue );
    tic = MPI_Wtime();
    #endif
    #if defined(DDOT_BY_DGEMV)
    magma_dgemv( MagmaTrans, *nd, 2,
                 one,  zr, *nd,
                       zr, ione,
                 zero, dnorm, ione,
                 queue );
    magma_dgetvector( 2, dnorm, 1, hnorm, 1, queue );
    znorm  = hnorm[0];
    zrnorm = hnorm[1];
    #else
    znorm  = magma_ddot(*nd, zr, ione, zr, ione, queue); 
    zrnorm = magma_ddot(*nd, zw, ione, zr, ione, queue); 
    #endif
    #if defined(PROF_MAGMA_BATCH)
    magma_queue_sync( queue );
    time_dot += (MPI_Wtime()-tic);
    #endif
    alpha = znorm/zrnorm;
    //if (mpinr == 0) printf( " alpha=%.2e/%.2e=%.2e\n",znorm,zrnorm, alpha );
    zrnorm = sqrt(znorm);
    if (mpinr == 0) {
        printf( "\n ** pipelined BICG (version 2) with MAGMA batched on GPU **\n" );
        printf( "\nOriginal relative residual norm = %.2e/%.2e = %.2e\n",zrnorm,bnorm,zrnorm/bnorm );
        printf( "HACApK_bicgstab_lfmtx_pipe start\n" );
    }
    for ( step=1; step<=mstep; step++ ) {
        if (zrnorm/bnorm < eps) break;
        if (step > 1) { 
            // zp(:nd) = zr(:nd) + beta*(zp(:nd) - zeta*zs(:nd))
            magma_daxpy( *nd, -zeta, zs, ione, zp, ione, queue );
            magmablas_dlascl( MagmaFull, ione, ione, one, beta, *nd, ione, zp, *nd, queue, &info );
            magma_daxpy( *nd, one, zr, ione, zp, ione, queue );
            // zs(:nd) = zw(:nd) + beta*(zs(:nd) - zeta*zz(:nd))
            magma_daxpy( *nd, -zeta, zz, ione, zs, ione, queue );
            magmablas_dlascl( MagmaFull, ione, ione, one, beta, *nd, ione, zs, *nd, queue, &info );
            magma_daxpy( *nd, one, zw, ione, zs, ione, queue );
            // zz(:nd) = zt(:nd) + beta*(z(:nd) - zeta*zv(:nd))
            magma_daxpy( *nd, -zeta, zv, ione, zz, ione, queue );
            magmablas_dlascl( MagmaFull, ione, ione, one, beta, *nd, ione, zz, *nd, queue, &info );
            magma_daxpy( *nd, one, zt, ione, zz, ione, queue );
        } else {
            magmablas_dlacpy( MagmaFull, *nd, ione, zr, *nd, zp, *nd, queue );
            magmablas_dlacpy( MagmaFull, *nd, ione, zw, *nd, zs, *nd, queue );
            magmablas_dlacpy( MagmaFull, *nd, ione, zt, *nd, zz, *nd, queue );
        }
        // zq(:nd) = zr(:nd) - alpha*zs(:nd)
        magmablas_dlacpy( MagmaFull, *nd, ione, zr, *nd, zq, *nd, queue );
        magma_daxpy( *nd, -alpha, zs, ione, zq, ione, queue );
        // zy(:nd) = zw(:nd) - alpha*zz(:nd)
        magmablas_dlacpy( MagmaFull, *nd, ione, zw, *nd, zy, *nd, queue );
        magma_daxpy( *nd, -alpha, zz, ione, zy, ione, queue );
        //if (mpinr == 0) printf( " %d: zeta=%.2e/%.2e=%.2e, alpha=%.2e\n",step,zrnorm,zden,zeta, alpha );
        //  .. SpMV: zv = A*zz ..
        //magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zv, *nd, queue );
        #if defined(PROF_MAGMA_BATCH)
        magma_queue_sync( queue );
        tic = MPI_Wtime();
        #endif
        #ifdef DIAGONAL_SCALE
        // t=D*zz, and zv=A*t
        cublasDsbmv(handle, CUBLAS_FILL_MODE_LOWER, *nd, izero, &one, d, ione, zz, ione, &zero, t, ione);
        c_hacapk_adot_body_lfmtx_batch_queue(zv,st_leafmtxp,t,wws, &time_batch,&time_set,&time_copy,
                                             on_gpu, queue, NULL, NULL);
        #else
        //magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zv, *nd, queue );
        #if defined(PROF_MAGMA_BATCH)
        magma_queue_sync( queue );
        tic = MPI_Wtime();
        #endif
        c_hacapk_adot_body_lfmtx_batch_queue(zv,st_leafmtxp,zz,wws, &time_batch,&time_set,&time_copy,
                                             on_gpu, queue, NULL, NULL);
        #endif
        #if defined(PROF_MAGMA_BATCH)
        magma_queue_sync( queue );
        time_spmv += (MPI_Wtime()-tic);
        #endif
        /////////////////////////////////////////////////////////
        // start communication
        MPI_Request request;
        #ifdef USE_NONBLOCKINC_GATHER
        c_hacapk_adot_cax_lfmtx_comm_getvector(zv, zau_cpu, st_ctl,buffer,disps,
                                               wws_cpu,wwr_cpu, isct,irct,*nd, 
                                               &time_copy,&time_mpi, queue);
        c_hacapk_adot_cax_lfmtx_comm_mpi(zv, zau_cpu, st_ctl,buffer,disps,
                                         wws_cpu,wwr_cpu, isct,irct,*nd, 
                                         &time_copy,&time_mpi, queue, &request);
        #else
        magma_event_record( event, queue );
        magma_queue_wait_event( queue_comm, event );
        c_hacapk_adot_cax_lfmtx_comm_getvector(zv, zau_cpu, st_ctl,buffer,disps,
                                               wws_cpu,wwr_cpu, isct,irct,*nd, 
                                               &time_copy,&time_mpi, queue_comm);
        #endif
        /////////////////////////////////////////////////////////
        // [znorm, zden] = zy'*[zq, zy]
        #if defined(PROF_MAGMA_BATCH)
        magma_queue_sync( queue );
        tic = MPI_Wtime();
        #endif
        #if defined(DDOT_BY_DGEMV)
        magma_dgemv( MagmaTrans, *nd, 2,
                     one,  zq, *nd,
                           zy, ione,
                     zero, dnorm, ione,
                     queue );
        magma_dgetvector( 2, dnorm, 1, hnorm, 1, queue );
        zrnorm = hnorm[0];
        zden   = hnorm[1];
        #else
        zrnorm = magma_ddot(*nd, zy, ione, zq, ione, queue); 
        zden   = magma_ddot(*nd, zy, ione, zy, ione, queue);
        #endif
        #if defined(PROF_MAGMA_BATCH)
        magma_queue_sync( queue );
        time_dot += (MPI_Wtime()-tic);
        #endif
        zeta = zrnorm/zden;
        /////////////////////////////////////////////////////////
        // finish communication
        #if !defined(USE_NONBLOCKINC_GATHER)
        magma_queue_sync( queue_comm );
        c_hacapk_adot_cax_lfmtx_comm_mpi(zv, zau_cpu, st_ctl,buffer,disps,
                                         wws_cpu,wwr_cpu, isct,irct,*nd, 
                                         &time_copy,&time_mpi, queue, &request);
        #endif
        c_hacapk_adot_cax_lfmtx_comm_setvector(zv, zau_cpu, st_ctl,buffer,disps,
                                               wws_cpu,wwr_cpu, isct,irct,*nd, 
                                               &time_copy,&time_mpi, queue, &request);
        /////////////////////////////////////////////////////////
        // zx(:nd) = zx(:nd) + alpha*zp(:nd) + zeta*zq(:nd)
        magma_daxpy( *nd, alpha, zp, ione, zx, ione, queue );
        magma_daxpy( *nd,  zeta, zq, ione, zx, ione, queue );
        // zr(:nd) = zq(:nd) - zeta*zy(:nd)
        magmablas_dlacpy( MagmaFull, *nd, ione, zq, *nd, zr, *nd, queue );
        magma_daxpy( *nd, -zeta, zy, ione, zr, ione, queue );
        // zw(:nd) = zy(:nd) + zeta*(zt(:nd) - alpha*zv(:nd))
        magmablas_dlacpy( MagmaFull, *nd, ione, zt, *nd, zw, *nd, queue );
        magma_daxpy( *nd, -alpha, zv, ione, zw, ione, queue );
        magmablas_dlascl( MagmaFull, ione, ione, one, -zeta, *nd, ione, zw, *nd, queue, &info );
        magma_daxpy( *nd, one, zy, ione, zw, ione, queue );
        //  .. SpMV: zt = A*zw ..
        //magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zt, *nd, queue );
        #if defined(PROF_MAGMA_BATCH)
        magma_queue_sync( queue );
        tic = MPI_Wtime();
        #endif
        #ifdef DIAGONAL_SCALE
        // t=D*zw, and zt=A*t
        cublasDsbmv(handle, CUBLAS_FILL_MODE_LOWER, *nd, izero, &one, d, ione, zw, ione, &zero, t, ione);
        c_hacapk_adot_body_lfmtx_batch_queue(zt,st_leafmtxp,t,wws, &time_batch,&time_set,&time_copy,
                                             on_gpu, queue, NULL, NULL);
        #else
        c_hacapk_adot_body_lfmtx_batch_queue(zt,st_leafmtxp,zw,wws, &time_batch,&time_set,&time_copy,
                                             on_gpu, queue, NULL, NULL);
        #endif
        #if defined(PROF_MAGMA_BATCH)
        magma_queue_sync( queue );
        time_spmv += (MPI_Wtime()-tic);
        #endif
        ///////////////////////////////////////////////////////
        // start communication
        #ifdef USE_NONBLOCKINC_GATHER
        c_hacapk_adot_cax_lfmtx_comm_getvector(zt,zau_cpu, st_ctl,buffer,disps,
                                               wws_cpu,wwr_cpu, isct,irct,*nd, 
                                               &time_copy,&time_mpi, queue);
        c_hacapk_adot_cax_lfmtx_comm_mpi(zt,zau_cpu, st_ctl,buffer,disps,
                                         wws_cpu,wwr_cpu, isct,irct,*nd, 
                                         &time_copy,&time_mpi, queue, &request);
        #else
        magma_event_record( event, queue );
        magma_queue_wait_event( queue_comm, event );
        c_hacapk_adot_cax_lfmtx_comm_getvector(zt,zau_cpu, st_ctl,buffer,disps,
                                               wws_cpu,wwr_cpu, isct,irct,*nd, 
                                               &time_copy,&time_mpi, queue);
        #endif
        ///////////////////////////////////////////////////////
        // all-reduces
        // > znorm = zr'*zr0
        znormold = znorm;
        #if defined(PROF_MAGMA_BATCH)
        magma_queue_sync( queue );
        tic = MPI_Wtime();
        #endif
        #if defined(DDOT_BY_DGEMV)
        magma_dgemv( MagmaTrans, *nd, 1,
                     one,  zr, *nd,
                           zr, ione,
                     zero, &dnorm[4], ione,
                     queue );
        magma_dgemv( MagmaTrans, *nd, 4,
                     one,  zr, *nd,
                           zr0, ione,
                     zero, dnorm, ione,
                     queue );
        magma_dgetvector( 5, dnorm, 1, hnorm, 1, queue );
        znorm  = hnorm[0];
        zwnorm = hnorm[1];
        zsnorm = hnorm[2];
        zznorm = hnorm[3];
        zrnorm = hnorm[4]; //magma_ddot(*nd, zr, ione, zr, ione, queue); 
        #else
        znorm  = magma_ddot(*nd, zr, ione, zr0, ione, queue); 
        zwnorm = magma_ddot(*nd, zw, ione, zr0, ione, queue); 
        zsnorm = magma_ddot(*nd, zs, ione, zr0, ione, queue); 
        zznorm = magma_ddot(*nd, zz, ione, zr0, ione, queue); 
        zrnorm = magma_ddot(*nd, zr, ione, zr, ione, queue); 
        #endif
        #if defined(PROF_MAGMA_BATCH)
        magma_queue_sync( queue );
        time_dot += (MPI_Wtime()-tic);
        #endif
        ///////////////////////////////////////////////////////
        // finish communication
        #if !defined(USE_NONBLOCKINC_GATHER)
        magma_queue_sync( queue_comm );
        c_hacapk_adot_cax_lfmtx_comm_mpi(zt,zau_cpu, st_ctl,buffer,disps,
                                         wws_cpu,wwr_cpu, isct,irct,*nd, 
                                         &time_copy,&time_mpi, queue, &request);
        #endif
        c_hacapk_adot_cax_lfmtx_comm_setvector(zt,zau_cpu, st_ctl,buffer,disps,
                                               wws_cpu,wwr_cpu, isct,irct,*nd, 
                                               &time_copy,&time_mpi, queue, &request);
        ///////////////////////////////////////////////////////
        //if (mpinr == 0) printf( " %d: znorm=%.2e zwnorm=%.2e zsnorm=%.2e nnzorm=%.2e\n",step,znorm,zwnorm,zsnorm,zznorm );
        // beta
        beta = (alpha/zeta)*(znorm/znormold);
        alpha = znorm/(zwnorm+beta*zsnorm-beta*zeta*zznorm);
        // > znorm = zr'*zr
        zrnorm = sqrt(zrnorm);
        *nstp = step;
        en_measure_time = MPI_Wtime();
        time = en_measure_time - st_measure_time;
        if (st_ctl->param[0] > 0 && mpinr == 0) {
            printf( " %d: time=%.2e log10(zrnorm/bnorm)=log10(%.2e/%.2e)=%.2e\n",step,time,zrnorm,bnorm,log10(zrnorm/bnorm) );
        }
    }
    magma_dgetvector( *nd, u, 1, u_cpu, 1, queue );
    MPI_Barrier( icomm );
    en_measure_time = MPI_Wtime();
    time = en_measure_time - st_measure_time;
    if (st_ctl->param[0] > 0) {
        //printf( " End: %d, %.2e\n",mpinr,time );
        if (mpinr == 0) {
            printf( "       BiCG        = %.5e\n", time );
            printf( "        time_mpi   = %.5e\n", time_mpi );
            printf( "        time_copy  = %.5e\n", time_copy );
            printf( "        time_spmv  = %.5e\n", time_spmv );
            printf( "        > time_batch = %.5e\n", time_batch );
            printf( "        > time_set   = %.5e\n", time_set );
            printf( "        time_dot   = %.5e\n", time_dot );
        }
    }
    magma_queue_destroy( queue );
    // delete matrix
    c_hacapk_adot_body_lfdel_batch_(st_leafmtxp);

    // free cpu memory
    if (buffer != NULL) free(buffer);
    free(disps);
    magma_free_pinned(wws_cpu);
    magma_free_pinned(wwr_cpu);
    magma_free_pinned(zau_cpu);

    // free gpu memory
    magma_free(u);
    magma_free(b);

    magma_free(wws);
    magma_free(wwr);

    magma_free(zr0);

    magma_free(zt);
    magma_free(zr);
    magma_free(zp);
    magma_free(zq);
    magma_free(zb);
    magma_free(zx);
    magma_free(zv);
}

#endif
