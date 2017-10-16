
#if defined(HAVE_MAGMA) | defined(HAVE_MAGMA_BATCH)

#include "HACApK_MAGMA.h"

// !! BiCG in C !!
double c_hacapk_dotp_d(int nd, double *b, double *a) {
    double norm = 0.0;
    int ii;
    for (ii=0; ii<nd; ii++) {
        norm += b[ii]*a[ii];
    }
    return norm;
}

#define PINNED_BUFFER
//#define PORTABLE_BUFFER
#if defined(PORTABLE_BUFFER)
magma_int_t
magma_dmalloc_pinned_portable( double** ptrPtr, size_t size )
{
    if (size == 0) size = 1;
    unsigned int flags = cudaHostAllocWriteCombined | cudaHostAllocPortable;
    if ( cudaSuccess != cudaHostAlloc( (void**)ptrPtr, size * sizeof(double), flags )) {
        return MAGMA_ERR_HOST_ALLOC;
    }
    return MAGMA_SUCCESS;
}
#endif

//#define WARMUP_MPI
void c_hacapk_adot_cax_lfmtx_warmup(stc_HACApK_lcontrol *st_ctl,
                                    double*zau, double *wws, double *wwr, int nd) {
    double tic;
    int *lpmd = (int*)((void*)st_ctl->param + st_ctl->lpmd_offset); 
    int mpinr = lpmd[2]; 
    int nrank = lpmd[1]; 
   
    if (nrank > 1) {
        int *lnp = (int*)((void*)st_ctl->param + st_ctl->lnp_offset);
        MPI_Comm icomm = MPI_COMM_WORLD;

        int ncdp = (mpinr+1)%nrank;       // my destination neighbor
        int ncsp = (mpinr+nrank-1)%nrank; // my source neighbor

        int ione = 1;
        double one = 1.0;
        lapackf77_dlaset( "F", &nd, &ione, &one, &one, zau, &nd );
        lapackf77_dlaset( "F", &nd, &ione, &one, &one, wws, &nd );
        lapackf77_dlaset( "F", &nd, &ione, &one, &one, wwr, &nd );
        {
           int nctp = (ncsp+nrank)%nrank; // where it came from

           double zero = 0.0;
           int send_count = nd; //lnp[nctp];
           int recv_count = nd; //lnp[mpinr];
           MPI_Status stats[2];
           MPI_Request reqs[2];
           //printf( " %d: send(%d/%d to %d), recv(%d/%d from %d)\n",mpinr,send_count,nd,ncdp,recv_count,nd,ncsp );
           MPI_Isend(wws, send_count, MPI_DOUBLE, ncdp, nrank+1, MPI_COMM_WORLD, &reqs[0]);
           MPI_Irecv(wwr, recv_count, MPI_DOUBLE, ncsp, nrank+1, MPI_COMM_WORLD, &reqs[1]);
           MPI_Waitall(2, reqs, stats);
           /*if (ncdp == 0) {
               printf( " TEST: wws(%d):",mpinr );
               magma_dprint(10,1,wws,send_count);
           }
           if (mpinr == 0) {
               printf( " TEST: wwr(%d):",mpinr );
               magma_dprint(10,1,wwr,recv_count);
           }*/
        }
    }
}

void c_hacapk_adot_cax_lfmtx_comm_setup(stc_HACApK_lcontrol *st_ctl,
                                        double **buffer, int **p_disps) {
    int *lpmd = (int*)((void*)st_ctl->param + st_ctl->lpmd_offset);
    int mpinr = lpmd[2];
    int nrank = lpmd[1];

    int buffer_size = 0;
    int *disps  = (int*)malloc((1+nrank) * sizeof(int));
    disps[0] = 0;
    if (nrank > 1) {
        int *lnp = (int*)((void*)st_ctl->param + st_ctl->lnp_offset);
        int ic;

        for (ic=0; ic<nrank; ic++) {
           disps[ic+1] = disps[ic]+lnp[ic];
        }
        buffer_size = disps[nrank];
    }
    *p_disps = disps;
    if (buffer_size > 0) {
        *buffer = (double*)malloc(buffer_size * sizeof(double));
    } else {
        *buffer = NULL;
    }
}

void c_hacapk_adot_cax_lfmtx_comm(double *zau, stc_HACApK_lcontrol *st_ctl, double* buffer, int *disps,
                                  double *wws, double *wwr, int *isct, int *irct, int nd, double *time_mpi) {
    int ione = 1;
    double one = 1.0;

    double tic;
    int *lpmd = (int*)((void*)st_ctl->param + st_ctl->lpmd_offset); 
    int mpinr = lpmd[2]; 
    int nrank = lpmd[1]; 
   
    if (nrank > 1) {
        int *lsp = (int*)((void*)st_ctl->param + st_ctl->lsp_offset);
        int *lnp = (int*)((void*)st_ctl->param + st_ctl->lnp_offset);
        MPI_Comm icomm = MPI_COMM_WORLD;

        int ic;
        int ncdp = (mpinr+1)%nrank;       // my destination neighbor
        int ncsp = (mpinr+nrank-1)%nrank; // my source neighbor
        #define COMM_BY_ALLGATHERV
        #if defined(COMM_BY_ALLGATHERV)
        tic = MPI_Wtime();
        MPI_Allgatherv(&zau[lsp[mpinr]-1], lnp[mpinr], MPI_DOUBLE, buffer, lnp, disps, MPI_DOUBLE, MPI_COMM_WORLD);
        *time_mpi += (MPI_Wtime()-tic);
        for (ic=1; ic<nrank; ic++) {
           int nctp = (ncsp-ic+nrank+1)%nrank; // where it came from
           irct[0] = lnp[nctp];
           irct[1] = lsp[nctp];
           blasf77_daxpy( &irct[0], &one, &buffer[disps[nctp]], &ione, &zau[irct[1]-1], &ione );
        }
        #else
        isct[0] = lnp[mpinr];
        isct[1] = lsp[mpinr];

        // copy local vector to send buffer
        dlacpy_( "F", &lnp[mpinr], &ione, &zau[lsp[mpinr]-1], &lnp[mpinr], wws, &lnp[mpinr] );
        for (ic=1; ic<nrank; ic++) {
           MPI_Status stat;
           tic = MPI_Wtime();
           #if 0
           MPI_Sendrecv(isct, 2, MPI_INT, ncdp, 2*(ic-1),
                        irct, 2, MPI_INT, ncsp, 2*(ic-1), 
                        icomm, &stat);
           #else // read offset/size from structure
           int nctp = (ncsp-ic+nrank+1)%nrank; // where it came from
           irct[0] = lnp[nctp];
           irct[1] = lsp[nctp];
           #endif

           //#define COMM_BY_ISEND_IRECV
           #if defined(COMM_BY_ISEND_IRECV)
           MPI_Status stats[2];
           MPI_Request reqs[2];
           if (MPI_SUCCESS != MPI_Isend(wws, isct[0], MPI_DOUBLE, ncdp, nrank+ic, MPI_COMM_WORLD, &reqs[0])) 
               printf( "MPI_Isend failed\n" );
           if (MPI_SUCCESS != MPI_Irecv(wwr, irct[0], MPI_DOUBLE, ncsp, nrank+ic, MPI_COMM_WORLD, &reqs[1]))
               printf( "MPI_Irecv failed\n" );
           if (MPI_SUCCESS != MPI_Waitall(2, reqs, stats))
               printf( "MPI_Waitall failed\n" );
           #else
           int info = 
           MPI_Sendrecv(wws, isct[0], MPI_DOUBLE, ncdp, 2*(ic-1)+1,
                        wwr, irct[0], MPI_DOUBLE, ncsp, 2*(ic-1)+1,
                        icomm, &stat);
           if (info != MPI_SUCCESS) printf( " MPI_Sendrecv failed with info=%d\n",info );
           #endif
           *time_mpi += (MPI_Wtime()-tic);
           blasf77_daxpy( &irct[0], &one, wwr, &ione, &zau[irct[1]-1], &ione );

           dlacpy_( "F", &irct[0], &ione, wwr, &irct[0], wws, &irct[0] );
           isct[0] = irct[0];
           isct[1] = irct[1];
        }
        #endif
    }
}

void c_hacapk_adot_cax_lfmtx_comm_gpu(int flag, double *zau_gpu, double *zau,
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
        if (flag == 1) {
            int ione = 1;
            double zero = 0.0;
            lapackf77_dlaset( "F", &nd, &ione, &zero, &zero, zau, &nd );
            magma_dgetvector( lnp[mpinr], &zau_gpu[lsp[mpinr]-1], 1, &zau[lsp[mpinr]-1], 1, queue );
        }
        #if defined(PROF_MAGMA_BATCH)
        *time_copy += MPI_Wtime()-tic;
        #endif

        c_hacapk_adot_cax_lfmtx_comm(zau, st_ctl,buffer,disps, wws,wwr, isct,irct, nd,time_mpi);

        #if defined(PROF_MAGMA_BATCH)
        magma_queue_sync( queue );
        tic = MPI_Wtime();
        #endif
        magma_dsetvector_async( nd, zau, 1, zau_gpu, 1, queue );
        //magma_dsetvector( nd, zau, 1, zau_gpu, 1, queue );
        #if defined(PROF_MAGMA_BATCH)
        *time_copy += MPI_Wtime()-tic;
        #endif
    }
}


void c_hacapk_bicgstab_cax_lfmtx_flat_(stc_HACApK_leafmtxp *st_leafmtxp, stc_HACApK_lcontrol *st_ctl,
                                       double *u, double *b, double*param, int *nd, int *nstp, int *lrtrn) {
    // local constants
    int ione = 1;
    double zero =  0.0;
    double one  =  1.0;
    double mone = -1.0;
    // local arrays
    double *zr, *zshdw, *zp, *zt, *zkp, *zakp, *zkt, *zakt;
    double *wws, *wwr;
    int *lpmd = (int*)((void*)st_ctl->param + st_ctl->lpmd_offset);
    int isct[2], irct[2];
    // local variables
    double eps, alpha, beta, zeta, zz, zden, znorm, znormold, bnorm, zrnorm;
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

    wws = (double*)malloc((*nd) * sizeof(double));
    wwr = (double*)malloc((*nd) * sizeof(double));

    zt = (double*)malloc((*nd) * sizeof(double));
    zr = (double*)malloc((*nd) * sizeof(double));
    zp = (double*)malloc((*nd) * sizeof(double));
    zkp = (double*)malloc((*nd) * sizeof(double));
    zakp = (double*)malloc((*nd) * sizeof(double));
    zkt = (double*)malloc((*nd) * sizeof(double));
    zakt= (double*)malloc((*nd) * sizeof(double));
    zshdw = (double*)malloc((*nd) * sizeof(double));

    // MPI buffer
    double * buffer = NULL;
    int *disps = NULL;
    c_hacapk_adot_cax_lfmtx_comm_setup(st_ctl, &buffer, &disps);

    // copy matrix to GPU
    c_hacapk_adot_body_lfcpy_batch_sorted_(nd, st_leafmtxp);
    //
    time_spmv = 0.0;
    time_mpi = 0.0;
    time_batch = 0.0;
    time_set = 0.0;
    time_copy = 0.0;
    MPI_Barrier( icomm );
    st_measure_time = MPI_Wtime();
    // init
    alpha = 0.0; beta = 0.0; zeta = 0.0;
    zz = c_hacapk_dotp_d(*nd, b, b ); 
    bnorm=sqrt(zz);
    lapackf77_dlacpy( "F", nd, &ione, b, nd, zr, nd );
    //  .. SpMV ..
    tic = MPI_Wtime();
    lapackf77_dlaset( "F", nd, &ione, &zero, &zero, zshdw, nd );
    c_hacapk_adot_body_lfmtx_batch_(zshdw,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy);
    time_spmv += (MPI_Wtime()-tic);
    c_hacapk_adot_cax_lfmtx_comm(zshdw, st_ctl,buffer,disps, wws, wwr, isct, irct, *nd, &time_mpi);
    //
    blasf77_daxpy( nd, &mone, zshdw, &ione, zr, &ione );
    lapackf77_dlacpy( "F", nd, &ione, zr, nd, zshdw, nd );
    zrnorm = c_hacapk_dotp_d(*nd, zr, zr ); 
    zrnorm = sqrt(zrnorm);
    if (mpinr == 0) {
        printf( "\n ** BICG (c version, flat) **\n" );
        printf( "\nOriginal relative residual norm = %.2e/%.2e = %.2e\n",zrnorm,bnorm,zrnorm/bnorm );
        printf( "HACApK_bicgstab_lfmtx_flat start\n" );
    }
    for ( step=1; step<=mstep; step++ ) {
        if (zrnorm/bnorm < eps) break;
        // zp(:nd) = zr(:nd) + beta*(zp(:nd) - zeta*zakp(:nd))
        if (beta == zero) {
            lapackf77_dlacpy( "F", nd, &ione, zr, nd, zp, nd );
        } else {
            blasf77_daxpy( nd, &zeta, zakp, &ione, zp, &ione );
            lapackf77_dlascl( "G", &ione, &ione, &one, &beta, nd, &ione, zp, nd, &info );
            blasf77_daxpy( nd, &one, zr, &ione, zp, &ione );
        }
        // zkp(:nd) = zp(:nd)
        lapackf77_dlacpy( "F", nd, &ione, zp, nd, zkp, nd );
        //  .. SpMV ..
        lapackf77_dlaset( "F", nd, &ione, &zero, &zero, zakp, nd );
        tic = MPI_Wtime();
        c_hacapk_adot_body_lfmtx_batch_(zakp,st_leafmtxp,zkp,wws, &time_batch,&time_set,&time_copy);
        time_spmv += (MPI_Wtime()-tic);
        c_hacapk_adot_cax_lfmtx_comm(zakp, st_ctl,buffer,disps, wws,wwr,isct,irct,*nd, &time_mpi);
        //
        znorm = c_hacapk_dotp_d(*nd, zshdw, zr ); 
        zden = c_hacapk_dotp_d(*nd, zshdw, zakp );
        alpha = -znorm/zden;
        znormold = znorm;
        // zt(:nd) = zr(:nd) - alpha*zakp(:nd)
        lapackf77_dlacpy( "F", nd, &ione, zr, nd, zt, nd );
        blasf77_daxpy( nd, &alpha, zakp, &ione, zt, &ione );
        alpha = -alpha;
        // zkt(:nd) = zt(:nd)
        lapackf77_dlacpy( "F", nd, &ione, zt, nd, zkt, nd );
        //  .. SpMV ..
        lapackf77_dlaset( "F", nd, &ione, &zero, &zero, zakt, nd );
        tic = MPI_Wtime();
        c_hacapk_adot_body_lfmtx_batch_(zakt,st_leafmtxp,zkt,wws, &time_batch,&time_set,&time_copy);
        time_spmv += (MPI_Wtime()-tic);
        c_hacapk_adot_cax_lfmtx_comm(zakt, st_ctl,buffer,disps, wws,wwr,isct,irct,*nd, &time_mpi);
        //
        znorm = c_hacapk_dotp_d(*nd, zakt, zt ); 
        zden = c_hacapk_dotp_d( *nd, zakt, zakt );
        zeta = znorm/zden;
        // u(:nd) = u(:nd) + alpha*zkp(:nd) + zeta*zkt(:nd)
        blasf77_daxpy( nd, &alpha, zkp, &ione, u, &ione );
        blasf77_daxpy( nd, &zeta,  zkt, &ione, u, &ione );
        // zr(:nd) = zt(:nd) - zeta*zakt(:nd)
        zeta = -zeta;
        lapackf77_dlacpy( "F", nd, &ione, zt, nd, zr, nd );
        blasf77_daxpy( nd, &zeta, zakt, &ione, zr, &ione );
        // beta = alpha/zeta * HACApK_dotp_d(nd,zshdw,zr)/znormold;
        beta = c_hacapk_dotp_d(*nd, zshdw, zr);
        beta = -alpha/zeta * beta/znormold;
        zrnorm = c_hacapk_dotp_d(*nd, zr, zr); 
        zrnorm = sqrt(zrnorm);
        *nstp = step;
        en_measure_time = MPI_Wtime();
        time = en_measure_time - st_measure_time;
        if (st_ctl->param[0] > 0 && mpinr == 0) {
            printf( " %d: time=%.2e log10(zrnorm/bnorm)=log10(%.2e/%.2e)=%.2e\n",step,time,zrnorm,bnorm,log10(zrnorm/bnorm) );
        }
    }
    MPI_Barrier( icomm );
    en_measure_time = MPI_Wtime();
    time = en_measure_time - st_measure_time;
    if (st_ctl->param[0] > 0) {
        //printf( " End: %d, %.2e\n",mpinr,time );
        if (mpinr == 0) {
            printf( "       BiCG        = %.5e\n", time );
            printf( "        time_mpi   = %.5e\n", time_mpi );
            printf( "        time_spmv  = %.5e\n", time_spmv );
            printf( "        > time_copy  = %.5e\n", time_copy );
            printf( "        > time_set   = %.5e\n", time_set );
            printf( "        > time_batch = %.5e\n", time_batch );
        }
    }
    // delete matrix
    c_hacapk_adot_body_lfdel_batch_(st_leafmtxp);

    // free cpu memory
    if (buffer != NULL) free(buffer);
    free(disps);
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

#include "magma_v2.h"

// BATCH on GPU
void c_hacapk_bicgstab_cax_lfmtx_gpu_(stc_HACApK_leafmtxp *st_leafmtxp, stc_HACApK_lcontrol *st_ctl,
                                      double *u_cpu, double *b_cpu, double*param, int *nd, int *nstp, int *lrtrn) {
    // local constants
    int ione = 1;
    double zero =  0.0;
    double one  =  1.0;
    double mone = -1.0;
    // local arrays
    double *zr, *zshdw, *zp, *zt, *zkp, *zakp, *zkt, *zakt;
    double *b, *u, *wws, *wwr;
    double *wws_cpu, *wwr_cpu, *zau_cpu;
    int *lpmd = (int*)((void*)st_ctl->param + st_ctl->lpmd_offset);
    int isct[2], irct[2];
    // local variables
    double eps, alpha, beta, zeta, zz, zden, znorm, znormold, bnorm, zrnorm;
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
    #ifdef GPU_AWARE
    if (mpinr == 0) {
        printf( "Compile time check : ");
        #if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
        printf( "This MPI library has CUDA-aware support.\n" );
        #elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
        printf( "This MPI library does not have CUDA-aware support.\n" );
        #else
        printf( "This MPI library cannot determine if there is CUDA-aware support.\n" );
        #endif /* MPIX_CUDA_AWARE_SUPPORT */
  
        printf( "Run time check     : " );
        #if defined(MPIX_CUDA_AWARE_SUPPORT)
        if (1 == MPIX_Query_cuda_support()) {
            printf( "This MPI library has CUDA-aware support.\n" );
        } else {
            printf( "This MPI library does not have CUDA-aware support.\n" );
        }
        #else /* !defined(MPIX_CUDA_AWARE_SUPPORT) */
        printf( "This MPI library cannot determine if there is CUDA-aware support.\n" );
        #endif /* MPIX_CUDA_AWARE_SUPPORT */
    }
    #endif

    int on_gpu = 1;
    magma_device_t cdev;
    magma_queue_t queue;

    magma_setdevice( get_device_id(st_leafmtxp) );
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    magma_queue_t *queue_hcmv = NULL;
    magma_event_t *event_hcmv = NULL;
    if (num_queues > 1) {
        int i;
        queue_hcmv = (magma_queue_t*)malloc(num_queues * sizeof(magma_queue_t));
        event_hcmv = (magma_event_t*)malloc(num_queues * sizeof(magma_event_t));
        for (i=0; i<num_queues; i++) {
            magma_queue_create( cdev, &queue_hcmv[i] );
            magma_event_create( &event_hcmv[i] );
        }
    }
    if (MAGMA_SUCCESS != magma_dmalloc(&u, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&b, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zt, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zr, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zp, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&wws, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&wwr, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zkp, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zkt, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zakp, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zakt, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zshdw, *nd)) {
      printf( " failed to allocate u or b (nd=%d)\n",*nd );
    }
    // use pinned memory for buffer
    //wws_cpu = (double*)malloc((*nd) * sizeof(double));
    //wwr_cpu = (double*)malloc((*nd) * sizeof(double));
    if (MAGMA_SUCCESS != magma_dmalloc_pinned(&wws_cpu, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc_pinned(&wwr_cpu, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc_pinned(&zau_cpu, *nd)) {
      printf( " failed to allocate pinned memory (nd=%d)\n",*nd );
    }

    // MPI buffer
    double *buffer = NULL;
    int *disps = NULL;
    c_hacapk_adot_cax_lfmtx_comm_setup(st_ctl, &buffer, &disps);

    // copy matrix to GPU
    c_hacapk_adot_body_lfcpy_batch_sorted_(nd, st_leafmtxp);
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
    zz = magma_ddot(*nd, b, ione, b, ione, queue); 
    bnorm=sqrt(zz);
    //  .. SpMV ..
    //magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zshdw, *nd, queue );
    #if defined(PROF_MAGMA_BATCH)
    magma_queue_sync( queue );
    tic = MPI_Wtime();
    #endif
    c_hacapk_adot_body_lfmtx_batch_queue(zshdw,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy, 
                                         on_gpu, queue, queue_hcmv, event_hcmv);
    #if defined(PROF_MAGMA_BATCH)
    magma_queue_sync( queue );
    time_spmv += (MPI_Wtime()-tic);
    #endif
    c_hacapk_adot_cax_lfmtx_comm_gpu(1, zshdw, zau_cpu, st_ctl,buffer,disps, wws_cpu, wwr_cpu, isct, irct, *nd, 
                                     &time_copy,&time_mpi, queue);
    //
    magmablas_dlacpy( MagmaFull, *nd, ione, b, *nd, zr, *nd, queue );
    magma_daxpy( *nd, mone, zshdw, ione, zr, ione, queue );
    magmablas_dlacpy( MagmaFull, *nd, ione, zr, *nd, zshdw, *nd, queue );
    zrnorm = magma_ddot(*nd, zr, ione, zr, ione, queue); 
    zrnorm = sqrt(zrnorm);
    if (mpinr == 0) {
        printf( "\n ** BICG with MAGMA batched on GPU **\n" );
        printf( "\nOriginal relative residual norm = %.2e/%.2e = %.2e\n",zrnorm,bnorm,zrnorm/bnorm );
        printf( " first time_mpi=%.2e\n",time_mpi );
        printf( "HACApK_bicgstab_lfmtx_gpu start\n" );
        if (num_queues > 1) printf( " ** num_queues = %d **\n",num_queues );
    }
    for ( step=1; step<=mstep; step++ ) {
        if (zrnorm/bnorm < eps) break;
        // zp(:nd) = zr(:nd) + beta*(zp(:nd) - zeta*zakp(:nd))
        if (beta == zero) {
            magmablas_dlacpy( MagmaFull, *nd, ione, zr, *nd, zp, *nd, queue );
        } else {
            magma_daxpy( *nd, zeta, zakp, ione, zp, ione, queue );
            magmablas_dlascl( MagmaFull, ione, ione, one, beta, *nd, ione, zp, *nd, queue, &info );
            magma_daxpy( *nd, one, zr, ione, zp, ione, queue );
        }
        // zkp(:nd) = zp(:nd)
        magmablas_dlacpy( MagmaFull, *nd, ione, zp, *nd, zkp, *nd, queue );
        //  .. SpMV ..
        //magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zakp, *nd, queue );
        #if defined(PROF_MAGMA_BATCH)
        magma_queue_sync( queue );
        tic = MPI_Wtime();
        #endif
        c_hacapk_adot_body_lfmtx_batch_queue(zakp,st_leafmtxp,zkp,wws, &time_batch,&time_set,&time_copy,
                                             on_gpu, queue, queue_hcmv, event_hcmv);
        #if defined(PROF_MAGMA_BATCH)
        magma_queue_sync( queue );
        time_spmv += (MPI_Wtime()-tic);
        #endif
        c_hacapk_adot_cax_lfmtx_comm_gpu(1, zakp, zau_cpu, st_ctl,buffer,disps, wws_cpu,wwr_cpu, isct,irct,*nd, 
                                         &time_copy,&time_mpi, queue);
        //
        znorm = magma_ddot(*nd, zshdw, ione, zr, ione, queue); 
        zden = magma_ddot(*nd, zshdw, ione, zakp, ione, queue);
        alpha = -znorm/zden;
        znormold = znorm;
        // zt(:nd) = zr(:nd) - alpha*zakp(:nd)
        magmablas_dlacpy( MagmaFull, *nd, ione, zr, *nd, zt, *nd, queue );
        magma_daxpy( *nd, alpha, zakp, ione, zt, ione, queue );
        alpha = -alpha;
        // zkt(:nd) = zt(:nd)
        magmablas_dlacpy( MagmaFull, *nd, ione, zt, *nd, zkt, *nd, queue );
        //  .. SpMV ..
        //magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zakt, *nd, queue );
        #if defined(PROF_MAGMA_BATCH)
        magma_queue_sync( queue );
        tic = MPI_Wtime();
        #endif
        c_hacapk_adot_body_lfmtx_batch_queue(zakt,st_leafmtxp,zkt,wws, &time_batch,&time_set,&time_copy,
                                             on_gpu, queue, queue_hcmv, event_hcmv);
        #if defined(PROF_MAGMA_BATCH)
        magma_queue_sync( queue );
        time_spmv += (MPI_Wtime()-tic);
        #endif
        c_hacapk_adot_cax_lfmtx_comm_gpu(1, zakt,zau_cpu, st_ctl,buffer,disps, wws_cpu,wwr_cpu, isct,irct,*nd, 
                                         &time_copy,&time_mpi, queue);
        //
        znorm = magma_ddot(*nd, zakt, ione, zt, ione, queue); 
        zden = magma_ddot( *nd, zakt, ione, zakt, ione, queue);
        zeta = znorm/zden;
        // u(:nd) = u(:nd) + alpha*zkp(:nd) + zeta*zkt(:nd)
        magma_daxpy( *nd, alpha, zkp, ione, u, ione, queue );
        magma_daxpy( *nd, zeta,  zkt, ione, u, ione, queue );
        // zr(:nd) = zt(:nd) - zeta*zakt(:nd)
        zeta = -zeta;
        magmablas_dlacpy( MagmaFull, *nd, ione, zt, *nd, zr, *nd, queue );
        magma_daxpy( *nd, zeta, zakt, ione, zr, ione, queue );
        // beta = alpha/zeta * HACApK_dotp_d(nd,zshdw,zr)/znormold;
        beta = magma_ddot(*nd, zshdw, ione, zr, ione, queue);
        beta = -alpha/zeta * beta/znormold;
        zrnorm = magma_ddot(*nd, zr, ione, zr, ione, queue);
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
#if 1
    c_hacapk_adot_body_lfmtx_batch_queue(zshdw,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy, 
                                         on_gpu, queue, NULL, NULL);
    c_hacapk_adot_cax_lfmtx_comm_gpu(1, zshdw, zau_cpu, st_ctl,buffer,disps, wws_cpu, wwr_cpu, isct, irct, *nd, 
                                     &time_copy,&time_mpi, queue);
    magmablas_dlascl( MagmaFull, ione, ione, one, mone, *nd, ione, zshdw, *nd, queue, &info );
    magma_daxpy( *nd, one, b, ione, zshdw, ione, queue );
    zrnorm = sqrt(magma_ddot(*nd, zshdw, ione, zshdw, ione, queue));
    printf( " resnorm=%.2e\n",zrnorm );
#endif
    if (st_ctl->param[0] > 0) {
        st_leafmtxp->gflops *= (1.0+2.0*(double)step);
        //printf( " End: %d, %.2e\n",mpinr,time );
        if (mpinr == 0) {
            printf( "       BiCG        = %.5e\n", time );
            printf( "        time_mpi   = %.5e\n", time_mpi );
            printf( "        time_copy  = %.5e\n", time_copy );
            printf( "        time_spmv  = %.5e\n", time_spmv );
            printf( "        > time_batch = %.5e (%.2fGflop/s)\n", time_batch,
                    st_leafmtxp->gflops/time_batch );
            printf( "        > time_set   = %.5e\n", time_set );
        }
        FILE *fp;
        char filename[100];
        sprintf(filename,"summary_%d.dat",mpinr);
        fp = fopen(filename,"w");
        fprintf( fp,"       BiCG        = %.5e\n", time );
        fprintf( fp,"        time_mpi   = %.5e\n", time_mpi );
        fprintf( fp,"        time_copy  = %.5e\n", time_copy );
        fprintf( fp,"        time_spmv  = %.5e\n", time_spmv );
        fprintf( fp,"        > time_batch = %.5e (%.2fGflop, %.2fGflop/s)\n", time_batch,
                 st_leafmtxp->gflops,st_leafmtxp->gflops/time_batch );
        fprintf( fp,"        > time_set   = %.5e\n", time_set );
        fclose(fp);
    }
    magma_queue_sync( queue );
    magma_queue_destroy( queue );

    // delete matrix
    c_hacapk_adot_body_lfdel_batch_(st_leafmtxp);

    // free gpu memory
    magma_free(u);
    magma_free(b);

    magma_free(wws);
    magma_free(wwr);

    magma_free(zt);
    magma_free(zr);
    magma_free(zp);
    magma_free(zkp);
    magma_free(zakp);
    magma_free(zkt);
    magma_free(zakt);
    magma_free(zshdw);

    // free cpu memory
    if (buffer != NULL) free(buffer);
    free(disps);
    //free(wws_cpu);
    //free(wwr_cpu);
    magma_free_pinned(wws_cpu);
    magma_free_pinned(wwr_cpu);
    magma_free_pinned(zau_cpu);
}


// BICG on multiple GPUs
void c_hacapk_bicgstab_cax_lfmtx_mgpu_(stc_HACApK_leafmtxp *st_leafmtxp, stc_HACApK_lcontrol *st_ctl,
                                       double *u_cpu, double *b_cpu, double*param, int *nd, int *nstp, int *lrtrn) {
    // local constants
    int ione = 1;
    double zero =  0.0;
    double one  =  1.0;
    double mone = -1.0;
    // local arrays
    int *lpmd = (int*)((void*)st_ctl->param + st_ctl->lpmd_offset);
    int isct[2], irct[2];
    // local variables
    double eps, alpha, beta, zeta, zz, zden, znorm, znormold, bnorm, zrnorm;
    double en_measure_time, st_measure_time, time;
    int info, step, mstep;
    int mpinr, nrank, ierr;
    double time_spmv, time_mpi, time_batch, time_set, time_copy, tic;
    double time_set1, time_set2, time_set3; 

    MPI_Comm icomm = MPI_COMM_WORLD; //lpmd[0];
    mstep = param[82];
    eps = param[90];
    mpinr = lpmd[2]; 
    nrank = lpmd[1]; 
    MPI_Barrier( icomm );

    #define ACCUM_ON_CPU
    #if defined(ACCUM_ON_CPU)
    int flag = 0;
    #else
    int flag = 1;
    #endif
    int on_gpu = 1, d, gpu_id = get_device_id(st_leafmtxp);
    magma_device_t cdev;
    magma_queue_t *queue = (magma_queue_t *)malloc(2*gpus_per_proc * sizeof(magma_queue_t));
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice((gpu_id+d)%procs_per_node);
        magma_getdevice( &cdev );
        magma_queue_create( cdev, &queue[d] );
        magma_queue_create( cdev, &queue[d+gpus_per_proc] );
    }
    // main GPU
    magma_setdevice(gpu_id);

    // use pinned memory for CPU buffer
    double *zau_cpu, *wws_cpu, *wwr_cpu;
#if defined(PINNED_BUFFER)
    magma_dmalloc_pinned(&zau_cpu,  *nd);
    magma_dmalloc_pinned(&wws_cpu,  *nd);
    magma_dmalloc_pinned(&wwr_cpu, (*nd)*gpus_per_proc);
#elif defined(PORTABLE_BUFFER)
    if (MAGMA_SUCCESS != magma_dmalloc_pinned_portable(&zau_cpu,  *nd) ||
        MAGMA_SUCCESS != magma_dmalloc_pinned_portable(&wws_cpu,  *nd) ||
        MAGMA_SUCCESS != magma_dmalloc_pinned_portable(&wwr_cpu, (*nd)*gpus_per_proc)) {
      printf( " failed to allocate pinned memory (nd=%d)\n",*nd );
    }
#else
    magma_dmalloc_cpu(&zau_cpu,  *nd);
    magma_dmalloc_cpu(&wws_cpu,  *nd);
    magma_dmalloc_cpu(&wwr_cpu, (*nd)*gpus_per_proc);
#endif

    double *zr, *zshdw, *zp, *zt, *zkp, *zakp, *zkt, *zakt;
    double *b = NULL, *u = NULL, *wws, *wwr; 
    if (MAGMA_SUCCESS != magma_dmalloc(&u, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&b, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zt, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zr, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zp, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&wws, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&wwr, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zkp, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zkt, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zakp, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zakt, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zshdw, *nd)) {
      printf( " failed to allocate vectors (nd=%d)\n",*nd );
    }
    // MPI buffer
    double *buffer = NULL;
    int *disps = NULL;
    c_hacapk_adot_cax_lfmtx_comm_setup(st_ctl, &buffer, &disps);

    // copy matrix to GPU
    c_hacapk_adot_body_lfcpy_batch_sorted_mgpu_(nd, st_leafmtxp, queue);
    #ifdef WARMUP_MPI
    c_hacapk_adot_cax_lfmtx_warmup(st_ctl, zau_cpu, wws_cpu, wwr_cpu, *nd);
    #endif
    //
    time_spmv = 0.0;
    time_mpi = 0.0;
    time_batch = 0.0;
    time_set = 0.0;
    time_set1 = 0.0;
    time_set2 = 0.0;
    time_set3 = 0.0;
    time_copy = 0.0;
    MPI_Barrier( icomm );
    st_measure_time = MPI_Wtime();
    // copy the input vector to GPU
    magma_dsetvector_async( *nd, b_cpu, 1, b, 1, queue[0] );
    magma_dsetvector_async( *nd, u_cpu, 1, u, 1, queue[0] );
    // init
    alpha = zero; beta = zero; zeta = zero;
    zz = magma_ddot(*nd, b, ione, b, ione, queue[0]); 
    bnorm=sqrt(zz);
    // .. SpMV ..
    int flag_set = 1;
    //magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zshdw, *nd, queue[0] );
    magma_queue_sync( queue[0] );
    tic = MPI_Wtime();
    c_hacapk_adot_body_lfmtx_batch_mgpu(flag_set, zshdw,st_leafmtxp,st_ctl,u,wws, zau_cpu,wwr_cpu,
                                        &time_batch,&time_set,&time_copy,
                                        &time_set1, &time_set2, &time_set3,
                                        on_gpu, queue);
    magma_queue_sync( queue[0] );
    time_spmv += (MPI_Wtime()-tic);
    c_hacapk_adot_cax_lfmtx_comm_gpu(flag, zshdw, zau_cpu,
                                     st_ctl,buffer,disps, wws_cpu, wwr_cpu, isct, irct, *nd, 
                                     &time_copy,&time_mpi, queue[0]);
    //
    magmablas_dlacpy( MagmaFull, *nd, ione, b, *nd, zr, *nd, queue[0] );
    magma_daxpy( *nd, mone, zshdw, ione, zr, ione, queue[0] );
    magmablas_dlacpy( MagmaFull, *nd, ione, zr, *nd, zshdw, *nd, queue[0] );
    zrnorm = magma_ddot(*nd, zr, ione, zr, ione, queue[0]); 
    zrnorm = sqrt(zrnorm);
    if (mpinr == 0) {
        printf( "\n ** BICG with MAGMA batched on multiple GPUs (%d GPUs) **\n",gpus_per_proc );
        printf( "\nOriginal relative residual norm = %.2e/%.2e = %.2e\n",zrnorm,bnorm,zrnorm/bnorm );
        printf( " first time_mpi=%.2e\n",time_mpi );
        printf( "HACApK_bicgstab_lfmtx_mgpu start\n" );
    }
    for ( step=1; step<=mstep; step++ ) {
        if (zrnorm/bnorm < eps) break;
        // zp(:nd) = zr(:nd) + beta*(zp(:nd) - zeta*zakp(:nd))
        if (beta == zero) {
            magmablas_dlacpy( MagmaFull, *nd, ione, zr, *nd, zp, *nd, queue[0] );
        } else {
            magma_daxpy( *nd, zeta, zakp, ione, zp, ione, queue[0] );
            magmablas_dlascl( MagmaFull, ione, ione, one, beta, *nd, ione, zp, *nd, queue[0], &info );
            magma_daxpy( *nd, one, zr, ione, zp, ione, queue[0] );
        }
        // zkp(:nd) = zp(:nd)
        magmablas_dlacpy( MagmaFull, *nd, ione, zp, *nd, zkp, *nd, queue[0] );
        //  .. SpMV ..
        //magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zakp, *nd, queue[0] );
        magma_queue_sync( queue[0] );
        tic = MPI_Wtime();
        c_hacapk_adot_body_lfmtx_batch_mgpu(flag_set, zakp,st_leafmtxp,st_ctl, zkp,wws, zau_cpu,wwr_cpu,
                                            &time_batch,&time_set,&time_copy, 
                                            &time_set1, &time_set2, &time_set3,
                                            on_gpu, queue);
        magma_queue_sync( queue[0] );
        time_spmv += (MPI_Wtime()-tic);
        c_hacapk_adot_cax_lfmtx_comm_gpu(flag, zakp, zau_cpu,
                                         st_ctl,buffer,disps, wws_cpu,wwr_cpu, isct,irct,*nd, 
                                         &time_copy,&time_mpi, queue[0]);
        //
        znorm = magma_ddot( *nd, zshdw, ione, zr, ione, queue[0] ); 
        zden = magma_ddot( *nd, zshdw, ione, zakp, ione, queue[0] );
        alpha = -znorm/zden;
        znormold = znorm;
//if (mpinr == 0) printf( " alpha=%.2e znorm=%.2e, zden=%.2e, time_mpi=%.2e\n",alpha,znorm,zden,time_mpi);
        // zt(:nd) = zr(:nd) - alpha*zakp(:nd)
        magmablas_dlacpy( MagmaFull, *nd, ione, zr, *nd, zt, *nd, queue[0] );
        magma_daxpy( *nd, alpha, zakp, ione, zt, ione, queue[0] );
        alpha = -alpha;
        // zkt(:nd) = zt(:nd)
        magmablas_dlacpy( MagmaFull, *nd, ione, zt, *nd, zkt, *nd, queue[0] );
        //  .. SpMV ..
        //magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zakt, *nd, queue[0] );
        magma_queue_sync( queue[0] );
        tic = MPI_Wtime();
        c_hacapk_adot_body_lfmtx_batch_mgpu(flag_set, zakt,st_leafmtxp,st_ctl, zkt,wws, zau_cpu,wwr_cpu, 
                                            &time_batch,&time_set,&time_copy,
                                            &time_set1, &time_set2, &time_set3,
                                            on_gpu, queue);
        magma_queue_sync( queue[0] );
        time_spmv += (MPI_Wtime()-tic);
        c_hacapk_adot_cax_lfmtx_comm_gpu(flag, zakt,zau_cpu,
                                         st_ctl,buffer,disps, wws_cpu,wwr_cpu, isct,irct,*nd, 
                                         &time_copy,&time_mpi, queue[0]);
        //
        znorm = magma_ddot( *nd, zakt, ione, zt, ione, queue[0] ); 
        zden = magma_ddot( *nd, zakt, ione, zakt, ione, queue[0] );
        zeta = znorm/zden;
//if (mpinr == 0) printf( " zeta=%.2e znorm=%.2e, zden=%.2e, time_mpi=%.2e\n",zeta,znorm,zden,time_mpi );
        // u(:nd) = u(:nd) + alpha*zkp(:nd) + zeta*zkt(:nd)
        magma_daxpy( *nd, alpha, zkp, ione, u, ione, queue[0] );
        magma_daxpy( *nd, zeta,  zkt, ione, u, ione, queue[0] );
        // zr(:nd) = zt(:nd) - zeta*zakt(:nd)
        zeta = -zeta;
        magmablas_dlacpy( MagmaFull, *nd, ione, zt, *nd, zr, *nd, queue[0] );
        magma_daxpy( *nd, zeta, zakt, ione, zr, ione, queue[0] );
        // beta = alpha/zeta * HACApK_dotp_d(nd,zshdw,zr)/znormold;
        beta = magma_ddot( *nd, zshdw, ione, zr, ione, queue[0]);
        beta = -alpha/zeta * beta/znormold;
        zrnorm = magma_ddot( *nd, zr, ione, zr, ione, queue[0] );
        zrnorm = sqrt(zrnorm);
        *nstp = step;
        en_measure_time = MPI_Wtime();
        time = en_measure_time - st_measure_time;
        if (st_ctl->param[0] > 0 && mpinr == 0) {
            printf( " %d: time=%.2e log10(zrnorm/bnorm)=log10(%.2e/%.2e)=%.2e\n",step,time,zrnorm,bnorm,log10(zrnorm/bnorm) );
        }
    }
    magma_dgetvector( *nd, u, 1, u_cpu, 1, queue[0] );
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
            printf( "          + time_set1 = %.5e\n", time_set1 );
            printf( "          + time_set2 = %.5e\n", time_set2 );
            printf( "          + time_set3 = %.5e\n", time_set3 );
        }
    }
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice((gpu_id+d)%procs_per_node);
        magma_queue_sync( queue[d] );
        magma_queue_destroy( queue[d] );
        magma_queue_destroy( queue[d+gpus_per_proc] );
    }
    free(queue);
    // delete matrix
    c_hacapk_adot_body_lfdel_mgpu_(st_leafmtxp);

    magma_setdevice(gpu_id);
    // free gpu memory
    magma_free(u);
    magma_free(b);

    magma_free(wws);
    magma_free(wwr);

    magma_free(zt);
    magma_free(zr);
    magma_free(zp);
    magma_free(zkp);
    magma_free(zakp);
    magma_free(zkt);
    magma_free(zakt);
    magma_free(zshdw);

    // free cpu memory
    if (buffer != NULL) free(buffer);
    free(disps);
#if defined(PINNED_BUFFER) | defined(PORTABLE_BUFFER)
    magma_free_pinned(zau_cpu);
    magma_free_pinned(wws_cpu);
    magma_free_pinned(wwr_cpu);
#else
    magma_free_cpu(zau_cpu);
    magma_free_cpu(wws_cpu);
    magma_free_cpu(wwr_cpu);
#endif
}

// BICG on multiple GPUs (redudant vector operation on each GPU)
void c_hacapk_bicgstab_cax_lfmtx_mgpu2_(stc_HACApK_leafmtxp *st_leafmtxp, stc_HACApK_lcontrol *st_ctl,
                                        double *u_cpu, double *b_cpu, double*param, int *nd, int *nstp, int *lrtrn) {
    // local constants
    int ione = 1;
    double zero =  0.0;
    double one  =  1.0;
    double mone = -1.0;
    // local arrays
    int *lpmd = (int*)((void*)st_ctl->param + st_ctl->lpmd_offset);
    int isct[2], irct[2];
    // local variables
    double eps, alpha, beta, zeta, zz, zden, znorm, znormold, bnorm, zrnorm;
    double en_measure_time, st_measure_time, time;
    int info, step, mstep;
    int mpinr, nrank, ierr;
    double time_spmv, time_mpi, time_batch, time_set, time_copy, time_dot, tic, toc;
    double time_set1, time_set2, time_set3, time_set4; 

    MPI_Comm icomm = MPI_COMM_WORLD; //lpmd[0];
    mstep = param[82];
    eps = param[90];
    mpinr = lpmd[2]; 
    nrank = lpmd[1]; 
    MPI_Barrier( icomm );
    #ifdef GPU_AWARE
    if (mpinr == 0) {
        printf( "Compile time check : ");
        #if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
        printf( "This MPI library has CUDA-aware support.\n" );
        #elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
        printf( "This MPI library does not have CUDA-aware support.\n" );
        #else
        printf( "This MPI library cannot determine if there is CUDA-aware support.\n" );
        #endif /* MPIX_CUDA_AWARE_SUPPORT */
  
        printf( "Run time check     : " );
        #if defined(MPIX_CUDA_AWARE_SUPPORT)
        if (1 == MPIX_Query_cuda_support()) {
            printf( "This MPI library has CUDA-aware support.\n" );
        } else {
            printf( "This MPI library does not have CUDA-aware support.\n" );
        }
        #else /* !defined(MPIX_CUDA_AWARE_SUPPORT) */
        printf( "This MPI library cannot determine if there is CUDA-aware support.\n" );
        #endif /* MPIX_CUDA_AWARE_SUPPORT */
    }
    #endif


    //int gpus_per_node = 4;
    //magma_device_t devices[ MagmaMaxGPUs ];
    //magma_getdevices( devices, MagmaMaxGPUs, &gpus_per_node );

    #if defined(ACCUM_ON_CPU)
    int flag = 0;
    #else
    int flag = 1;
    #endif
    int on_gpu = 1, d, gpu_id = (gpus_per_proc*get_device_id(st_leafmtxp))%gpus_per_node;
    magma_device_t cdev;
    magma_queue_t *queue = (magma_queue_t *)malloc(2*gpus_per_proc * sizeof(magma_queue_t));
    magma_event_t *event = (magma_event_t *)malloc(  gpus_per_proc * sizeof(magma_event_t));

    magma_queue_t **queue_hcmv = (magma_queue_t**)malloc(gpus_per_proc * sizeof(magma_queue_t**));
    magma_event_t **event_hcmv = (magma_event_t**)malloc(gpus_per_proc * sizeof(magma_event_t**));

    printf( " process %d on GPU(%d:%d)\n",mpinr,gpu_id,gpu_id+gpus_per_proc-1);
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice(gpu_id+d);
        magma_getdevice( &cdev );
        magma_queue_create( cdev, &queue[d] );
        magma_queue_create( cdev, &queue[d+gpus_per_proc] );
        magma_event_create( &event[d] );
        if (num_queues > 1) {
            int i;
            queue_hcmv[d] = (magma_queue_t*)malloc(num_queues * sizeof(magma_queue_t));
            event_hcmv[d] = (magma_event_t*)malloc(num_queues * sizeof(magma_event_t));
            for (i=0; i<num_queues; i++) {
                magma_queue_create( cdev, &queue_hcmv[d][i] );
                magma_event_create( &event_hcmv[d][i] );
            }
        } else {
            queue_hcmv[d] = NULL;
            event_hcmv[d] = NULL;
        }
    }
    // main GPU
    magma_setdevice(gpu_id);
    st_leafmtxp->iwork = (int*)malloc(2*gpus_per_proc * sizeof(int));

    // use pinned memory for CPU buffer
    double *zau_cpu1, *zau_cpu2, *wws_cpu, *wwr_cpu;
    magma_dmalloc_pinned(&zau_cpu1, *nd);
    magma_dmalloc_pinned(&zau_cpu2, *nd);
    magma_dmalloc_pinned(&wws_cpu,  *nd);
    magma_dmalloc_pinned(&wwr_cpu, (*nd)*gpus_per_proc);

    // allocate GPU vectors
    double *wws, *wwr;
    if (MAGMA_SUCCESS != magma_dmalloc(&wws, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&wwr, *nd)) {
      printf( " failed to allocate vectors (nd=%d)\n",*nd );
    }

    double **b = (double**)malloc(gpus_per_proc * sizeof(double*));
#define redundant_u
#if !defined(redundant_u)
    double **u = (double**)malloc(gpus_per_proc * sizeof(double*));
#endif
    double **zr = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **zp = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **zt = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **zkp = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **zkt = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **zakp =(double**)malloc(gpus_per_proc * sizeof(double*));
    double **zshdw = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **zakt = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **dBuffer = (double**)malloc(gpus_per_proc * sizeof(double*));
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice(gpu_id+d);
        if (MAGMA_SUCCESS != magma_dmalloc(&b[d], *nd) ||
#if !defined(redundant_u)
            MAGMA_SUCCESS != magma_dmalloc(&u[d], *nd) ||
#endif
            MAGMA_SUCCESS != magma_dmalloc(&zt[d], *nd) ||
            MAGMA_SUCCESS != magma_dmalloc(&zr[d], *nd) ||
            MAGMA_SUCCESS != magma_dmalloc(&zp[d], *nd) ||
            MAGMA_SUCCESS != magma_dmalloc(&zkp[d], *nd) ||
            MAGMA_SUCCESS != magma_dmalloc(&zkt[d], *nd) ||
            MAGMA_SUCCESS != magma_dmalloc(&zakp[d], *nd) ||
            MAGMA_SUCCESS != magma_dmalloc(&zakt[d], *nd) ||
            MAGMA_SUCCESS != magma_dmalloc(&zshdw[d], *nd)) {
          printf( " failed to allocate vectors (nd=%d)\n",*nd );
        }
    }
    magma_setdevice(gpu_id);
    for (d=0; d<gpus_per_proc; d++) {
        magma_dmalloc(&dBuffer[d], *nd);
    }
    // setup peers
    for (d=1; d<gpus_per_proc; d++) {
        int canAccessPeer;
        magma_setdevice(gpu_id+d);
        cudaDeviceCanAccessPeer(&canAccessPeer, gpu_id+d, gpu_id);
        if (canAccessPeer == 1) {
            if (cudaDeviceEnablePeerAccess( gpu_id, 0 ) != cudaSuccess) {
                printf( " %d:%d: cudaDeviceEnablePeerAccess( %d ) failed\n",mpinr,gpu_id+d,gpu_id );
            }
        } else {
            printf( "cudaDeviceCanAccessPeer( %d, %d ) failed\n",gpu_id+d,gpu_id );
        }
    }
    // MPI buffer
    double *buffer = NULL;
    int *disps = NULL;
    c_hacapk_adot_cax_lfmtx_comm_setup(st_ctl, &buffer, &disps);

    #ifdef WARMUP_MPI
    c_hacapk_adot_cax_lfmtx_warmup(st_ctl, zau_cpu, wws_cpu, wwr_cpu, *nd);
    #endif
    // copy matrix to GPU
    c_hacapk_adot_body_lfcpy_batch_sorted_mgpu_(nd, st_leafmtxp, queue);
#if defined(redundant_u)
    double **u = st_leafmtxp->zu_mgpu;
#endif
    //
    time_spmv = 0.0;
    time_mpi = 0.0;
    time_batch = 0.0;
    time_set = 0.0;
    time_set1 = 0.0;
    time_set2 = 0.0;
    time_set3 = 0.0;
    time_set4 = 0.0;
    time_copy = 0.0;
    time_dot = 0.0;
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice(gpu_id+d);
        magma_queue_sync( queue[d] );
    }
    MPI_Barrier( icomm );
    st_measure_time = MPI_Wtime();
    // copy the input vector to GPU
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice(gpu_id+d);
        magma_dsetvector_async( *nd, b_cpu, 1, b[d], 1, queue[d] );
        magma_dsetvector_async( *nd, u_cpu, 1, u[d], 1, queue[d] );
    }
    // init
    alpha = 0.0; beta = 0.0; zeta = 0.0;
    magma_setdevice(gpu_id);
    zz = magma_ddot(*nd, b[0], ione, b[0], ione, queue[0]); 
    bnorm=sqrt(zz);
    // .. SpMV ..
    int flag_set = 1;
    #if defined(PROF_MAGMA_BATCH)
    magma_queue_sync( queue[0] );
    tic = MPI_Wtime();
    #endif
    c_hacapk_adot_body_lfmtx_batch_mgpu2(0, zshdw[0],st_leafmtxp,st_ctl,u,wws, zau_cpu1,wwr_cpu,
                                         dBuffer, event, &time_batch,&time_set,&time_copy,
                                         &time_set1, &time_set2, &time_set3,
                                         on_gpu, queue, queue_hcmv, event_hcmv);
    #if defined(PROF_MAGMA_BATCH)
    magma_queue_sync( queue[0] );
    time_spmv += (MPI_Wtime()-tic);
    #endif
    c_hacapk_adot_cax_lfmtx_comm_gpu(flag, zshdw[0], zau_cpu1,
                                     st_ctl,buffer,disps, wws_cpu, wwr_cpu, isct, irct, *nd, 
                                     &time_copy,&time_mpi, queue[0]);
    #if defined(PROF_MAGMA_BATCH)
    tic = MPI_Wtime();
    #endif
    #if 0 
    if (nrank == 1) {
        magma_setdevice(gpu_id);
        magma_dgetvector( *nd, zshdw[0], 1, zau_cpu1, 1, queue[0] );
    }
    for (d=1; d<gpus_per_proc; d++) {
        magma_setdevice(gpu_id+d);
        magma_dsetvector_async( *nd, zau_cpu1, 1, zshdw[d], 1, queue[d] );
    }
    #else
    magma_event_record( event[0], queue[0] );
    for (d=1; d<gpus_per_proc; d++) {
        magma_setdevice(gpu_id+d);
        magma_queue_wait_event( queue[d], event[0] );
        cudaMemcpyPeerAsync(zshdw[d], gpu_id+d, zshdw[0], gpu_id, (*nd)*sizeof(double),
                            magma_queue_get_cuda_stream( queue[d] ) );
        magma_event_record( event[d], queue[d] );
        magma_queue_wait_event( queue[0], event[d] );
    }
    #endif
    #if defined(PROF_MAGMA_BATCH)
    for (d=1; d<gpus_per_proc; d++) {
        magma_queue_sync( queue[d] );
    }
    toc = (MPI_Wtime()-tic);
    time_set  += toc;
    time_set4 += toc;
    #endif
    //
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice(gpu_id+d);
        magmablas_dlacpy( MagmaFull, *nd, ione, b[d], *nd, zr[d], *nd, queue[d] );
        magma_daxpy( *nd, mone, zshdw[d], ione, zr[d], ione, queue[d] );
        magmablas_dlacpy( MagmaFull, *nd, ione, zr[d], *nd, zshdw[d], *nd, queue[d] );
    }
    magma_setdevice(gpu_id);
    zrnorm = magma_ddot(*nd, zr[0], ione, zr[0], ione, queue[0]); 
    zrnorm = sqrt(zrnorm);
    if (mpinr == 0) {
        printf( "\n ** BICG with MAGMA batched on multiple GPUs (%d GPUs) **\n",gpus_per_proc );
        printf( "\nOriginal relative residual norm = %.2e/%.2e = %.2e\n",zrnorm,bnorm,zrnorm/bnorm );
        printf( " first time_mpi=%.2e\n",time_mpi );
        printf( "HACApK_bicgstab_lfmtx_mgpu2 start\n" );
    }
    for ( step=1; step<=mstep; step++ ) {
        if (zrnorm/bnorm < eps) break;
        // zp(:nd) = zr(:nd) + beta*(zp(:nd) - zeta*zakp(:nd))
        for (d=0; d<gpus_per_proc; d++) {
            magma_setdevice(gpu_id+d);
            if (beta == zero) {
                magmablas_dlacpy( MagmaFull, *nd, ione, zr[d], *nd, zp[d], *nd, queue[d] );
            } else {
                magma_daxpy( *nd, zeta, zakp[d], ione, zp[d], ione, queue[d] );
                magmablas_dlascl( MagmaFull, ione, ione, one, beta, *nd, ione, zp[d], *nd, queue[d], &info );
                magma_daxpy( *nd, one, zr[d], ione, zp[d], ione, queue[d] );
            }
        }
        // zkp(:nd) = zp(:nd)
        for (d=0; d<gpus_per_proc; d++) {
            magma_setdevice(gpu_id+d);
            magmablas_dlacpy( MagmaFull, *nd, ione, zp[d], *nd, zkp[d], *nd, queue[d] );
        }
        //  .. SpMV ..
        magma_setdevice(gpu_id);
        #if defined(PROF_MAGMA_BATCH)
        magma_queue_sync( queue[0] );
        tic = MPI_Wtime();
        #endif
        c_hacapk_adot_body_lfmtx_batch_mgpu2(flag_set, zakp[0],st_leafmtxp,st_ctl, zkp,wws, zau_cpu2,wwr_cpu,
                                             dBuffer,event, &time_batch,&time_set,&time_copy, 
                                             &time_set1, &time_set2, &time_set3,
                                             on_gpu, queue, queue_hcmv, event_hcmv);
        #if defined(PROF_MAGMA_BATCH)
        magma_queue_sync( queue[0] );
        time_spmv += (MPI_Wtime()-tic);
        #endif
        c_hacapk_adot_cax_lfmtx_comm_gpu(flag, zakp[0], zau_cpu2,
                                         st_ctl,buffer,disps, wws_cpu,wwr_cpu, isct,irct,*nd, 
                                         &time_copy,&time_mpi, queue[0]);
        #if defined(PROF_MAGMA_BATCH)
        tic = MPI_Wtime();
        #endif
        #if 0
        if (nrank == 1) {
            magma_setdevice(gpu_id);
            magma_dgetvector( *nd, zakp[0], 1, zau_cpu2, 1, queue[0] );
        }
        for (d=1; d<gpus_per_proc; d++) {
            magma_setdevice(gpu_id+d);
            magma_dsetvector_async( *nd, zau_cpu2, 1, zakp[d], 1, queue[d] );
        }
        #else
        magma_event_record( event[0], queue[0] );
        for (d=1; d<gpus_per_proc; d++) {
            magma_setdevice(gpu_id+d);
            magma_queue_wait_event( queue[d], event[0] );
            cudaMemcpyPeerAsync(zakp[d], gpu_id+d, zakp[0], gpu_id, (*nd)*sizeof(double),
                                 magma_queue_get_cuda_stream( queue[d] ) );
            magma_event_record( event[d], queue[d] );
            magma_queue_wait_event( queue[0], event[d] );
        }
        #endif
        #if defined(PROF_MAGMA_BATCH)
        for (d=1; d<gpus_per_proc; d++) {
            magma_queue_sync( queue[d] );
        }
        toc = (MPI_Wtime()-tic);
        time_set  += toc;
        time_set4 += toc;

        tic = MPI_Wtime();
        #endif
        //
        magma_setdevice(gpu_id);
        znorm = magma_ddot( *nd, zshdw[0], ione, zr[0], ione, queue[0] ); 
        zden = magma_ddot( *nd, zshdw[0], ione, zakp[0], ione, queue[0] );
        alpha = -znorm/zden;
        znormold = znorm;
        #if defined(PROF_MAGMA_BATCH)
        for (d=1; d<gpus_per_proc; d++) {
            magma_queue_sync( queue[d] );
        }
        time_dot += (MPI_Wtime()-tic);
        #endif
        // zt(:nd) = zr(:nd) - alpha*zakp(:nd)
        for (d=0; d<gpus_per_proc; d++) {
            magma_setdevice(gpu_id+d);
            magmablas_dlacpy( MagmaFull, *nd, ione, zr[d], *nd, zt[d], *nd, queue[d] );
            magma_daxpy( *nd, alpha, zakp[d], ione, zt[d], ione, queue[d] );
        }
        alpha = -alpha;
        // zkt(:nd) = zt(:nd)
        for (d=0; d<gpus_per_proc; d++) {
            magma_setdevice(gpu_id+d);
            magmablas_dlacpy( MagmaFull, *nd, ione, zt[d], *nd, zkt[d], *nd, queue[d] );
        }
        //  .. SpMV ..
        magma_setdevice(gpu_id);
        #if defined(PROF_MAGMA_BATCH)
        magma_queue_sync( queue[0] );
        tic = MPI_Wtime();
        #endif
        c_hacapk_adot_body_lfmtx_batch_mgpu2(flag_set, zakt[0],st_leafmtxp,st_ctl, zkt,wws, zau_cpu1,wwr_cpu, 
                                             dBuffer,event, &time_batch,&time_set,&time_copy,
                                             &time_set1, &time_set2, &time_set3,
                                             on_gpu, queue, queue_hcmv, event_hcmv);
        #if defined(PROF_MAGMA_BATCH)
        magma_queue_sync( queue[0] );
        time_spmv += (MPI_Wtime()-tic);
        #endif
        c_hacapk_adot_cax_lfmtx_comm_gpu(flag, zakt[0],zau_cpu1,
                                         st_ctl,buffer,disps, wws_cpu,wwr_cpu, isct,irct,*nd, 
                                         &time_copy,&time_mpi, queue[0]);
        #if defined(PROF_MAGMA_BATCH)
        tic = MPI_Wtime();
        #endif
        #if 0
        if (nrank == 1) {
            magma_setdevice(gpu_id);
            magma_dgetvector( *nd, zakt[0], 1, zau_cpu1, 1, queue[0] );
        }
        for (d=1; d<gpus_per_proc; d++) {
            magma_setdevice(gpu_id+d);
            magma_dsetvector_async( *nd, zau_cpu1, 1, zakt[d], 1, queue[d] );
        }
        #else
        magma_event_record( event[0], queue[0] );
        for (d=1; d<gpus_per_proc; d++) {
            magma_setdevice(gpu_id+d);
            magma_queue_wait_event( queue[d], event[0] );
            cudaMemcpyPeerAsync(zakt[d], gpu_id+d, zakt[0], gpu_id, (*nd)*sizeof(double),
                                 magma_queue_get_cuda_stream( queue[d] ) );
            magma_event_record( event[d], queue[d] );
            magma_queue_wait_event( queue[0], event[d] );
        }
        #endif
        #if defined(PROF_MAGMA_BATCH)
        for (d=1; d<gpus_per_proc; d++) {
            magma_queue_sync( queue[d] );
        }
        toc = (MPI_Wtime()-tic);
        time_set  += toc;
        time_set4 += toc;

        tic = MPI_Wtime();
        #endif
        //
        magma_setdevice(gpu_id);
        znorm = magma_ddot( *nd, zakt[0], ione, zt[0], ione, queue[0] ); 
        zden = magma_ddot( *nd, zakt[0], ione, zakt[0], ione, queue[0] );
        zeta = znorm/zden;
        #if defined(PROF_MAGMA_BATCH)
        for (d=1; d<gpus_per_proc; d++) {
            magma_queue_sync( queue[d] );
        }
        time_dot += (MPI_Wtime()-tic);
        #endif
        // u(:nd) = u(:nd) + alpha*zkp(:nd) + zeta*zkt(:nd)
        for (d=0; d<gpus_per_proc; d++) {
            magma_setdevice(gpu_id+d);
            magma_daxpy( *nd, alpha, zkp[d], ione, u[d], ione, queue[d] );
            magma_daxpy( *nd, zeta,  zkt[d], ione, u[d], ione, queue[d] );
        }
        // zr(:nd) = zt(:nd) - zeta*zakt(:nd)
        zeta = -zeta;
        for (d=0; d<gpus_per_proc; d++) {
            magma_setdevice(gpu_id+d);
            magmablas_dlacpy( MagmaFull, *nd, ione, zt[d], *nd, zr[d], *nd, queue[d] );
            magma_daxpy( *nd, zeta, zakt[d], ione, zr[d], ione, queue[d] );
        }
        // beta = alpha/zeta * HACApK_dotp_d(nd,zshdw,zr)/znormold;
        #if defined(PROF_MAGMA_BATCH)
        for (d=1; d<gpus_per_proc; d++) {
            magma_queue_sync( queue[d] );
        }
        tic = MPI_Wtime();
        #endif
        magma_setdevice(gpu_id);
        beta = magma_ddot( *nd, zshdw[0], ione, zr[0], ione, queue[0]);
        beta = -alpha/zeta * beta/znormold;
        zrnorm = magma_ddot( *nd, zr[0], ione, zr[0], ione, queue[0] );
        zrnorm = sqrt(zrnorm);
        #if defined(PROF_MAGMA_BATCH)
        for (d=1; d<gpus_per_proc; d++) {
            magma_queue_sync( queue[d] );
        }
        time_dot += (MPI_Wtime()-tic);
        #endif
        *nstp = step;
        en_measure_time = MPI_Wtime();
        time = en_measure_time - st_measure_time;
        if (st_ctl->param[0] > 0 && mpinr == 0) {
            printf( " %d: time=%.2e log10(zrnorm/bnorm)=log10(%.2e/%.2e)=%.2e\n",step,time,zrnorm,bnorm,log10(zrnorm/bnorm) );
        }
    }
    magma_dgetvector( *nd, u[0], 1, u_cpu, 1, queue[0] );
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice(gpu_id+d);
        magma_queue_sync( queue[d] );
    }
    MPI_Barrier( icomm );
    en_measure_time = MPI_Wtime();
    time = en_measure_time - st_measure_time;
    if (st_ctl->param[0]>0) {
        //printf( " End: %d, %.2e\n",mpinr,time );
        if (mpinr == 0) {
            printf( "       BiCG       = %.5e\n", time );
            printf( "        time_mpi  = %.5e\n", time_mpi );
            printf( "        time_copy = %.5e\n", time_copy );
            printf( "        time_spmv = %.5e\n", time_spmv );
            printf( "        > time_batch = %.5e\n", time_batch );
            printf( "        > time_set   = %.5e\n", time_set );
            printf( "          + time_set1 = %.5e\n", time_set1 );
            printf( "          + time_set2 = %.5e\n", time_set2 );
            printf( "          + time_set3 = %.5e\n", time_set3 );
            printf( "          + time_set4 = %.5e\n", time_set4 );
            printf( "          +          += %.5e\n", time_set1+time_set2+time_set3+time_set4 );
            printf( "        > time_dot   = %.5e\n", time_dot );
            printf( "                     => %.5e\n", time_mpi+time_copy+time_spmv+time_dot );
        }
    }

    // free gpu memory
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice(gpu_id+d);
        magma_queue_sync( queue[d] );
        magma_queue_destroy( queue[d] );
        magma_queue_destroy( queue[d+gpus_per_proc] );

#if !defined(redundant_u)
        magma_free(u[d]);
#endif
        magma_free(b[d]);

        magma_free(zt[d]);
        magma_free(zr[d]);
        magma_free(zp[d]);
        magma_free(zkp[d]);
        magma_free(zakp[d]);
        magma_free(zkt[d]);
        magma_free(zakt[d]);
        magma_free(zshdw[d]);
    }
    // delete matrix
    c_hacapk_adot_body_lfdel_mgpu_(st_leafmtxp);

    magma_setdevice(gpu_id);
    // free cpu memory
    free(queue);
    if (buffer != NULL) free(buffer);
    free(disps);
#if !defined(redundant_u)
    free(u);
#endif
    free(b);
    free(zt);
    free(zr);
    free(zp);
    free(zkp);
    free(zakp);
    free(zkt);
    free(zakt);
    free(zshdw);

    magma_free_pinned(zau_cpu1);
    magma_free_pinned(zau_cpu2);
    magma_free_pinned(wws_cpu);
    magma_free_pinned(wwr_cpu);

    magma_free(wws);
    magma_free(wwr);
    // setup peers
    for (d=1; d<gpus_per_proc; d++) {
        magma_setdevice(gpu_id+d);
        cudaDeviceDisablePeerAccess( gpu_id );
    }
}

void device_daxpy(int n, double *alpha, double *x, int incx, double *y, int incy, magma_queue_t *queue) {
    cublasHandle_t handle = magma_queue_get_cublas_handle( *queue );
    cublasSetPointerMode( handle, CUBLAS_POINTER_MODE_DEVICE );
    cublasDaxpy( handle, n, alpha, x, incx, y, incy );
    cublasSetPointerMode( handle, CUBLAS_POINTER_MODE_HOST );
}

// BICG on multiple GPUs (distribute vector operation on local GPUs)
void c_hacapk_bicgstab_cax_lfmtx_mgpu3_(stc_HACApK_leafmtxp *st_leafmtxp, stc_HACApK_lcontrol *st_ctl,
                                        double *u_cpu, double *b_cpu, double*param, int *nd, int *nstp, int *lrtrn) {
    // local constants
    int ione = 1;
    double zero =  0.0;
    double one  =  1.0;
    double mone = -1.0;
    // local arrays
    int *lpmd = (int*)((void*)st_ctl->param + st_ctl->lpmd_offset);
    int isct[2], irct[2];
    // local variables
    double eps, beta, zeta, zz, znormold, bnorm, zrnorm;
    #define DDOT_REDUCE_ON_CPU
    #if defined(DDOT_REDUCE_ON_CPU)
    double alpha, zden, znorm;
    #endif
    double en_measure_time, st_measure_time, time;
    int info, step, mstep;
    int mpinr, nrank, ierr;
    double time_spmv, time_mpi, time_batch, time_set, time_copy, time_dot, tic, toc;
    double time_set1, time_set2, time_set3, time_set4; 

    MPI_Comm icomm = MPI_COMM_WORLD; //lpmd[0];
    mstep = param[82];
    eps = param[90];
    mpinr = lpmd[2]; 
    nrank = lpmd[1]; 
    MPI_Barrier( icomm );

    assert(gpus_per_proc == gpus_per_node);
    //int gpus_per_node = 4;
    //magma_device_t devices[ MagmaMaxGPUs ];
    //magma_getdevices( devices, MagmaMaxGPUs, &gpus_per_node );

    #if defined(ACCUM_ON_CPU)
    int flag = 0;
    #else
    int flag = 1;
    #endif
    int on_gpu = 1, d, gpu_id = (gpus_per_proc*get_device_id(st_leafmtxp))%gpus_per_node;
    magma_device_t cdev;
    magma_queue_t *queue = (magma_queue_t *)malloc(2*gpus_per_proc * sizeof(magma_queue_t));
    magma_event_t *event = (magma_event_t *)malloc(  gpus_per_proc * sizeof(magma_event_t));
    printf( " process %d on GPU(%d:%d)\n",mpinr,gpu_id,gpu_id+gpus_per_proc-1);
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice(gpu_id+d);
        magma_getdevice( &cdev );
        magma_queue_create( cdev, &queue[d] );
        magma_queue_create( cdev, &queue[d+gpus_per_proc] );
        magma_event_create( &event[d] );
    }
    // main GPU
    magma_setdevice(gpu_id);
    st_leafmtxp->iwork = (int*)malloc(2*gpus_per_proc * sizeof(int));

    // use pinned memory for CPU buffer
    double *zau_cpu1, *zau_cpu2, *wws_cpu, *wwr_cpu;
    magma_dmalloc_pinned(&zau_cpu1, *nd);
    magma_dmalloc_pinned(&zau_cpu2, *nd);
    magma_dmalloc_pinned(&wws_cpu,  *nd);
    magma_dmalloc_pinned(&wwr_cpu, (*nd)*gpus_per_proc);

    // allocate GPU vectors
    double *wws, *wwr;
    if (MAGMA_SUCCESS != magma_dmalloc(&wws, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&wwr, *nd)) {
      printf( " failed to allocate vectors (nd=%d)\n",*nd );
    }

    double **b = (double**)malloc(gpus_per_proc * sizeof(double*));
#define redundant_u
#if !defined(redundant_u)
    double **u = (double**)malloc(gpus_per_proc * sizeof(double*));
#endif
    double **zr = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **zp = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **zt = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **zkp = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **zkt = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **zakp =(double**)malloc(gpus_per_proc * sizeof(double*));
    double **zshdw = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **zakt = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **dBuffer = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **znorm_d = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **alpha_d = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **zeta_d  = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **beta_d  = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **znorm_h = (double**)malloc(gpus_per_proc * sizeof(double*));
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice(gpu_id+d);
        if (MAGMA_SUCCESS != magma_dmalloc(&b[d], *nd) ||
#if !defined(redundant_u)
            MAGMA_SUCCESS != magma_dmalloc(&u[d], *nd) ||
#endif
            MAGMA_SUCCESS != magma_dmalloc(&zt[d], 2*(*nd)) ||
            MAGMA_SUCCESS != magma_dmalloc(&zp[d], *nd) ||
            MAGMA_SUCCESS != magma_dmalloc(&zkp[d], *nd) ||
            MAGMA_SUCCESS != magma_dmalloc(&zkt[d], *nd) ||
            MAGMA_SUCCESS != magma_dmalloc(&zshdw[d], 3*(*nd)) ||
            MAGMA_SUCCESS != magma_dmalloc(&znorm_d[d], 4) ||
            MAGMA_SUCCESS != magma_dmalloc(&alpha_d[d], 2) ||
            MAGMA_SUCCESS != magma_dmalloc(&beta_d[d], 1) ||
            MAGMA_SUCCESS != magma_dmalloc(&zeta_d[d], 1)) {
          printf( " failed to allocate vectors (nd=%d)\n",*nd );
        }
        zr[d]   = &zshdw[d][*nd];
        zakp[d] = &zr[d][*nd];
        zakt[d] = &zt[d][*nd];
        magma_dmalloc_pinned(&znorm_h[d], 4);
    }
    magma_setdevice(gpu_id);
    for (d=0; d<gpus_per_proc; d++) {
        magma_dmalloc(&dBuffer[d], *nd);
    }
    // setup peers
    #if defined(DDOT_REDUCE_ON_CPU)
    for (d=1; d<gpus_per_proc; d++) {
        int canAccessPeer;
        magma_setdevice(gpu_id+d);
        cudaDeviceCanAccessPeer(&canAccessPeer, gpu_id+d, gpu_id);
        if (canAccessPeer == 1) {
            if (cudaDeviceEnablePeerAccess( gpu_id, 0 ) != cudaSuccess) {
                printf( " %d:%d: cudaDeviceEnablePeerAccess( %d ) failed\n",mpinr,gpu_id+d,gpu_id );
            }
        } else {
            printf( "cudaDeviceCanAccessPeer( %d, %d ) failed\n",gpu_id+d,gpu_id );
        }
    }
    #else
    for (d=0; d<gpus_per_proc; d++) {
        int dest;
        for (dest=0; dest<gpus_per_proc; dest++) {
            if (dest != d) {
                int canAccessPeer;
                magma_setdevice(gpu_id+d);
                cudaDeviceCanAccessPeer(&canAccessPeer, gpu_id+d, gpu_id+dest);
                if (canAccessPeer == 1) {
                    if (cudaDeviceEnablePeerAccess( gpu_id+dest, 0 ) != cudaSuccess) {
                        printf( " %d:%d: cudaDeviceEnablePeerAccess( %d ) failed\n",mpinr,gpu_id+d,gpu_id+dest );
                    }
                } else {
                    printf( "cudaDeviceCanAccessPeer( %d, %d ) failed\n",gpu_id+d,gpu_id+dest );
                }
            }
        }
    }
    #endif
    // MPI buffer
    double *buffer = NULL;
    int *disps = NULL;
    c_hacapk_adot_cax_lfmtx_comm_setup(st_ctl, &buffer, &disps);

    #ifdef WARMUP_MPI
    c_hacapk_adot_cax_lfmtx_warmup(st_ctl, zau_cpu, wws_cpu, wwr_cpu, *nd);
    #endif
    // copy matrix to GPU
    c_hacapk_adot_body_lfcpy_batch_sorted_mgpu_(nd, st_leafmtxp, queue);
#if defined(redundant_u)
    double **u = st_leafmtxp->zu_mgpu;
#endif
    //
    time_dot  = 0.0;
    time_spmv = 0.0;
    time_mpi = 0.0;
    time_batch = 0.0;
    time_set = 0.0;
    time_set1 = 0.0;
    time_set2 = 0.0;
    time_set3 = 0.0;
    time_set4 = 0.0;
    time_copy = 0.0;
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice(gpu_id+d);
        magma_queue_sync( queue[d] );
    }
    MPI_Barrier( icomm );
    st_measure_time = MPI_Wtime();
    // copy the input vector to GPU
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice(gpu_id+d);
        magma_dsetvector_async( *nd, b_cpu, 1, b[d], 1, queue[d] );
        magma_dsetvector_async( *nd, u_cpu, 1, u[d], 1, queue[d] );
    }
    // init
    #ifdef DDOT_REDUCE_ON_CPU
    alpha = 0.0; beta = 0.0; zeta = 0.0;
    #endif
    magma_setdevice(gpu_id);
    zz = magma_ddot(*nd, b[0], ione, b[0], ione, queue[0]); 
    bnorm=sqrt(zz);
    // .. SpMV ..
    int flag_set = 1;
    #if defined(PROF_MAGMA_BATCH)
    magma_queue_sync( queue[0] );
    tic = MPI_Wtime();
    #endif
    c_hacapk_adot_body_lfmtx_batch_mgpu2(0, zshdw[0],st_leafmtxp,st_ctl,u,wws, zau_cpu1,wwr_cpu,
                                         dBuffer, event, &time_batch,&time_set,&time_copy,
                                         &time_set1, &time_set2, &time_set3,
                                         on_gpu, queue, NULL, NULL);
    #if defined(PROF_MAGMA_BATCH)
    magma_queue_sync( queue[0] );
    time_spmv += (MPI_Wtime()-tic);
    #endif
    c_hacapk_adot_cax_lfmtx_comm_gpu(flag, zshdw[0], zau_cpu1,
                                     st_ctl,buffer,disps, wws_cpu, wwr_cpu, isct, irct, *nd, 
                                     &time_copy,&time_mpi, queue[0]);
    #if defined(PROF_MAGMA_BATCH)
    tic = MPI_Wtime();
    #endif
    #if 0
    if (nrank == 1) {
        magma_setdevice(gpu_id);
        magma_dgetvector( *nd, zshdw[0], 1, zau_cpu1, 1, queue[0] );
    }
    for (d=1; d<gpus_per_proc; d++) {
        magma_setdevice(gpu_id+d);
        magma_dsetvector_async( *nd, zau_cpu1, 1, zshdw[d], 1, queue[d] );
    }
    #else
    magma_event_record( event[0], queue[0] );
    for (d=1; d<gpus_per_proc; d++) {
        magma_setdevice(gpu_id+d);
        magma_queue_wait_event( queue[d], event[0] );
        cudaMemcpyPeerAsync(zshdw[d], gpu_id+d, zshdw[0], gpu_id, (*nd)*sizeof(double),
                            magma_queue_get_cuda_stream( queue[d] ) );
        magma_event_record( event[d], queue[d] );
        magma_queue_wait_event( queue[0], event[d] );
    }
    #endif
    #if defined(PROF_MAGMA_BATCH)
    for (d=1; d<gpus_per_proc; d++) {
        magma_queue_sync( queue[d] );
    }
    toc = (MPI_Wtime()-tic);
    time_set  += toc;
    time_set4 += toc;
    #endif
    //
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice(gpu_id+d);
        magmablas_dlacpy( MagmaFull, *nd, ione, b[d], *nd, zr[d], *nd, queue[d] );
        magma_daxpy( *nd, mone, zshdw[d], ione, zr[d], ione, queue[d] );
        magmablas_dlacpy( MagmaFull, *nd, ione, zr[d], *nd, zshdw[d], *nd, queue[d] );
    }
    magma_setdevice(gpu_id);
    zrnorm = magma_ddot(*nd, zr[0], ione, zr[0], ione, queue[0]); 
    zrnorm = sqrt(zrnorm);
    if (mpinr == 0) {
        printf( "\n ** BICG with MAGMA batched on multiple GPUs (%d GPUs) **\n",gpus_per_proc );
        printf( "\nOriginal relative residual norm = %.2e/%.2e = %.2e\n",zrnorm,bnorm,zrnorm/bnorm );
        printf( " first time_mpi=%.2e\n",time_mpi );
        printf( "HACApK_bicgstab_lfmtx_mgpu3 start\n" );
    }
    int nd_loc2 = (*nd + 1)/2;
    int nd_loc  = (*nd + 3)/4;
    for ( step=1; step<=mstep; step++ ) {
        if (zrnorm/bnorm < eps) break;
        // zp(:nd) = zr(:nd) + beta*(zp(:nd) - zeta*zakp(:nd))
        for (d=0; d<gpus_per_proc; d++) {
            magma_setdevice(gpu_id+d);
            if (step == 1) {
                magmablas_dlacpy( MagmaFull, *nd, ione, zr[d], *nd, zkp[d], *nd, queue[d] );
            } else {
                #ifdef DDOT_REDUCE_ON_CPU
                magma_daxpy( *nd, -zeta, zakp[d], ione, zp[d], ione, queue[d] );
                magmablas_dlacpy( MagmaFull, *nd, ione, zr[d], *nd, zkp[d], *nd, queue[d] );
                magma_daxpy( *nd, beta, zp[d], ione, zkp[d], ione, queue[d] );
                #else
                device_daxpy( *nd, zeta_d[d], zakp[d], ione, zp[d], ione, &queue[d] );
                magmablas_dlacpy( MagmaFull, *nd, ione, zr[d], *nd, zkp[d], *nd, queue[d] );
                device_daxpy( *nd, beta_d[d], zp[d], ione, zkp[d], ione, &queue[d] );
                #endif
            }
        }
        // zkp(:nd) = zp(:nd)
        for (d=0; d<gpus_per_proc; d++) {
            magma_setdevice(gpu_id+d);
            magmablas_dlacpy( MagmaFull, *nd, ione, zkp[d], *nd, zp[d], *nd, queue[d] );
        }
        //  .. SpMV ..
        magma_setdevice(gpu_id);
        #if defined(PROF_MAGMA_BATCH)
        magma_queue_sync( queue[0] );
        tic = MPI_Wtime();
        #endif
        c_hacapk_adot_body_lfmtx_batch_mgpu2(flag_set, zakp[0],st_leafmtxp,st_ctl, zkp,wws, zau_cpu2,wwr_cpu,
                                             dBuffer,event, &time_batch,&time_set,&time_copy, 
                                             &time_set1, &time_set2, &time_set3,
                                             on_gpu, queue, NULL, NULL);
        #if defined(PROF_MAGMA_BATCH)
        magma_queue_sync( queue[0] );
        time_spmv += (MPI_Wtime()-tic);
        #endif
        c_hacapk_adot_cax_lfmtx_comm_gpu(flag, zakp[0], zau_cpu2,
                                         st_ctl,buffer,disps, wws_cpu,wwr_cpu, isct,irct,*nd, 
                                         &time_copy,&time_mpi, queue[0]);
        #if defined(PROF_MAGMA_BATCH)
        tic = MPI_Wtime();
        #endif
        #if 0
        if (nrank == 1) {
            magma_setdevice(gpu_id);
            magma_dgetvector( *nd, zakp[0], 1, zau_cpu2, 1, queue[0] );
        }
        for (d=1; d<gpus_per_proc; d++) {
            magma_setdevice(gpu_id+d);
            magma_dsetvector_async( *nd, zau_cpu2, 1, zakp[d], 1, queue[d] );
        }
        #else
        magma_event_record( event[0], queue[0] );
        for (d=1; d<gpus_per_proc; d++) {
            magma_setdevice(gpu_id+d);
            magma_queue_wait_event( queue[d], event[0] );
            cudaMemcpyPeerAsync(zakp[d], gpu_id+d, zakp[0], gpu_id, (*nd)*sizeof(double),
                                magma_queue_get_cuda_stream( queue[d] ) );
            magma_event_record( event[d], queue[d] );
            magma_queue_wait_event( queue[0], event[d] );
        }
        #endif
        #if defined(PROF_MAGMA_BATCH)
        for (d=1; d<gpus_per_proc; d++) {
            magma_queue_sync( queue[d] );
        }
        toc = (MPI_Wtime()-tic);
        time_set  += toc;
        time_set4 += toc;

        tic = MPI_Wtime();
        #endif
        // distributed
        int dd0, inc_dd, swc_dd;
        #ifdef DDOT_REDUCE_ON_CPU
        dd0 = 2;
        inc_dd = 2;
        swc_dd = 0;
        #else
        dd0 = 1;
        inc_dd = 1;
        swc_dd = 1;
        #endif
        int offset = 0, dd;
#if 0
        magma_setdevice(gpu_id);
        znorm = magma_ddot( *nd, zshdw[0], ione, zr[0], ione, queue[0] ); 
        zden  = magma_ddot( *nd, zshdw[0], ione, zakp[0], ione, queue[0] );
        alpha = -znorm/zden;
        znormold = znorm;
#else
        for (dd=0; dd<gpus_per_proc; dd+=inc_dd) {
            d = (dd0+dd)%gpus_per_proc;
            int nd_d = (dd > swc_dd ? *nd-nd_loc2 : nd_loc2);
            magma_setdevice(gpu_id+d);
            #if 1 
            cublasSetPointerMode( magma_queue_get_cublas_handle( queue[d] ), CUBLAS_POINTER_MODE_DEVICE );
            cublasDdot( magma_queue_get_cublas_handle( queue[d] ), nd_d, 
                        &(zshdw[d][offset]), ione, &(zr[d][offset]), ione, &znorm_d[d][0] );
            cublasDdot( magma_queue_get_cublas_handle( queue[d] ), nd_d,
                        &(zshdw[d][offset]), ione, &(zakp[d][offset]), ione, &znorm_d[d][1] );
            cublasSetPointerMode(magma_queue_get_cublas_handle( queue[d] ), CUBLAS_POINTER_MODE_HOST);
            #else
            magma_dgemm( MagmaTrans, MagmaNoTrans,
                         2, 1, nd_d,
                         one, &(zr[d][offset]), *nd,
                              &(zshdw[d][offset]), *nd,
                         zero,  znorm_d[d], 2, queue[d] );
            #endif
            if (d == 1) { // GPU-1 -> GPU-3
                cudaMemcpyPeerAsync( &znorm_d[3][2], gpu_id+3, znorm_d[1], gpu_id+1, 2*sizeof(double),
                                     magma_queue_get_cuda_stream( queue[1] ) );
                magma_event_record( event[1], queue[1] );
            } else if (d == 2) { // GPU-2 -> GPU->0
                cudaMemcpyPeerAsync( &znorm_d[0][2], gpu_id+0, znorm_d[d], gpu_id+2, 2*sizeof(double),
                                     magma_queue_get_cuda_stream( queue[2] ) );
                magma_event_record( event[2], queue[2] );
            } else if (d == 3) {
                magma_queue_wait_event( queue[3], event[1] );
                magma_daxpy( 2, one,  &znorm_d[3][2], 1, &znorm_d[3][0], 1, queue[3] );

                // compute, copy it back, alpha=-znorm/zden
                // alpha[1] = znormold
                #if defined(DDOT_REDUCE_ON_CPU)
                magma_dgetvector( 2, znorm_d[3], 1, znorm_h[3], 1, queue[3] );
                znorm = znorm_h[3][0];
                zden  = znorm_h[3][1];
                alpha = -znorm/zden;
                znormold = znorm;
                #else
                magma_setdevice(gpu_id+3);
                magmablas_dlacpy( MagmaFull, 2, 1, znorm_d[3], 2, alpha_d[3], 2, queue[3] );
                magma_dtrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                             ione, ione,
                            -one, &znorm_d[3][1], ione,
                                   alpha_d[3], ione, queue[3] );
                cudaMemcpyPeerAsync( alpha_d[1], gpu_id+1, alpha_d[3], gpu_id+3, 2*sizeof(double),
                                     magma_queue_get_cuda_stream( queue[3] ) );
                magma_event_record( event[3], queue[3] );
                magma_queue_wait_event( queue[1], event[3] );
                #endif
            } else if (d == 0) {
                magma_queue_wait_event( queue[0], event[2] );
                magma_daxpy( 2, one,  &znorm_d[0][2], 1, &znorm_d[0][0], 1, queue[0] );

                // compute, copy it back, alpha[0]=-znorm/zden
                // alpha[1] = znormold
                #if defined(DDOT_REDUCE_ON_CPU)
                magma_dgetvector( 2, znorm_d[0], 1, znorm_h[0], 1, queue[0] );
                znorm = znorm_h[0][0];
                zden  = znorm_h[0][1];
                alpha = -znorm/zden;
                znormold = znorm;
                #else
                magmablas_dlacpy( MagmaFull, 2, 1, znorm_d[0], 2, alpha_d[0], 2, queue[0] );
                magma_dtrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                             ione, ione,
                            -one, &znorm_d[0][1], ione,
                                   alpha_d[0], ione, queue[0] );
                cudaMemcpyPeerAsync( alpha_d[2], gpu_id+2, alpha_d[0], gpu_id+0, 2*sizeof(double),
                                     magma_queue_get_cuda_stream( queue[0] ) );
                magma_event_record( event[0], queue[0] );
                magma_queue_wait_event( queue[2], event[0] );
                #endif
            }
            if (dd == swc_dd) offset += nd_d;
        }
#endif
        #if defined(PROF_MAGMA_BATCH)
        for (d=1; d<gpus_per_proc; d++) {
            magma_queue_sync( queue[d] );
        }
        time_dot += (MPI_Wtime()-tic);
        #endif
        //
        // zt(:nd) = zr(:nd) - alpha*zakp(:nd)
        for (d=0; d<gpus_per_proc; d++) {
            magma_setdevice(gpu_id+d);
            magmablas_dlacpy( MagmaFull, *nd, ione, zr[d], *nd, zt[d], *nd, queue[d] );
            #if defined(DDOT_REDUCE_ON_CPU)
            magma_daxpy( *nd, alpha, zakp[d], ione, zt[d], ione, queue[d] );
            #else
            device_daxpy( *nd, alpha_d[d], zakp[d], ione, zt[d], ione, &queue[d] );
            #endif
        }
        // zkt(:nd) = zt(:nd)
        for (d=0; d<gpus_per_proc; d++) {
            magma_setdevice(gpu_id+d);
            magmablas_dlacpy( MagmaFull, *nd, ione, zt[d], *nd, zkt[d], *nd, queue[d] );
        }
        //  .. SpMV ..
        magma_setdevice(gpu_id);
        #if defined(PROF_MAGMA_BATCH)
        magma_queue_sync( queue[0] );
        tic = MPI_Wtime();
        #endif
        c_hacapk_adot_body_lfmtx_batch_mgpu2(flag_set, zakt[0],st_leafmtxp,st_ctl, zkt,wws, zau_cpu1,wwr_cpu, 
                                             dBuffer,event, &time_batch,&time_set,&time_copy,
                                             &time_set1, &time_set2, &time_set3,
                                             on_gpu, queue, NULL, NULL);
        #if defined(PROF_MAGMA_BATCH)
        magma_queue_sync( queue[0] );
        time_spmv += (MPI_Wtime()-tic);
        #endif
        c_hacapk_adot_cax_lfmtx_comm_gpu(flag, zakt[0],zau_cpu1,
                                         st_ctl,buffer,disps, wws_cpu,wwr_cpu, isct,irct,*nd, 
                                         &time_copy,&time_mpi, queue[0]);
        #if defined(PROF_MAGMA_BATCH)
        tic = MPI_Wtime();
        #endif
        #if 0
        if (nrank == 1) {
            magma_setdevice(gpu_id);
            magma_dgetvector( *nd, zakt[0], 1, zau_cpu1, 1, queue[0] );
        }
        for (d=1; d<gpus_per_proc; d++) {
            magma_setdevice(gpu_id+d);
            magma_dsetvector_async( *nd, zau_cpu1, 1, zakt[d], 1, queue[d] );
        }
        #else
        magma_event_record( event[0], queue[0] );
        for (d=1; d<gpus_per_proc; d++) {
            magma_setdevice(gpu_id+d);
            magma_queue_wait_event( queue[d], event[0] );
            cudaMemcpyPeerAsync(zakt[d], gpu_id+d, zakt[0], gpu_id, (*nd)*sizeof(double),
                                magma_queue_get_cuda_stream( queue[d] ) );
            magma_event_record( event[d], queue[d] );
            magma_queue_wait_event( queue[0], event[d] );
        }
        #endif
        #if defined(PROF_MAGMA_BATCH)
        for (d=1; d<gpus_per_proc; d++) {
            magma_queue_sync( queue[d] );
        }
        toc = (MPI_Wtime()-tic);
        time_set  += toc;
        time_set4 += toc;

        tic = MPI_Wtime();
        #endif
        //
#if 0
        magma_setdevice(gpu_id);
        znorm = magma_ddot( *nd, zakt[0], ione, zt[0], ione, queue[0] ); 
        zden  = magma_ddot( *nd, zakt[0], ione, zakt[0], ione, queue[0] );
        zeta = znorm/zden;
#else
        offset = 0;
        for (dd=0; dd<gpus_per_proc; dd+=inc_dd) {
            d = (dd+dd0)%gpus_per_proc;
            int nd_d = (dd > swc_dd ? *nd-nd_loc2 : nd_loc2);
            magma_setdevice(gpu_id+d);
            #if 1
            cublasSetPointerMode( magma_queue_get_cublas_handle( queue[d] ), CUBLAS_POINTER_MODE_DEVICE );
            cublasDdot( magma_queue_get_cublas_handle( queue[d] ), nd_d, 
                        &(zakt[d][offset]), ione, &(zt[d][offset]), ione, &znorm_d[d][0] );
            cublasDdot( magma_queue_get_cublas_handle( queue[d] ), nd_d,
                        &(zakt[d][offset]), ione, &(zakt[d][offset]), ione, &znorm_d[d][1] );
            cublasSetPointerMode(magma_queue_get_cublas_handle( queue[d] ), CUBLAS_POINTER_MODE_HOST);
            #else
            magma_dgemm( MagmaTrans, MagmaNoTrans,
                         2, 1, nd_d,
                         one, &(zt[d][offset]), *nd,
                              &(zakt[d][offset]), *nd,
                         zero,  znorm_d[d], 2, queue[d] );
            #endif
            if (d == 1) { // GPU-1 -> GPU-3
                cudaMemcpyPeerAsync( &znorm_d[3][2], gpu_id+3, znorm_d[d], gpu_id+d, 2*sizeof(double),
                                     magma_queue_get_cuda_stream( queue[d] ) );
                magma_event_record( event[d], queue[d] );
            } else if (d == 2) { // GPU-2 -> GPU->0
                cudaMemcpyPeerAsync( &znorm_d[0][2], gpu_id+0, znorm_d[d], gpu_id+d, 2*sizeof(double),
                                     magma_queue_get_cuda_stream( queue[d] ) );
                magma_event_record( event[d], queue[d] );
            } else if (d == 3) {
                magma_queue_wait_event( queue[d], event[1] );
                magma_daxpy( 2, one,  &znorm_d[d][2], 1, &znorm_d[d][0], 1, queue[d] );

                // compute, copy it back, zeta = znorm/zden
                #if defined(DDOT_REDUCE_ON_CPU)
                magma_dgetvector( 2, znorm_d[3], 1, znorm_h[3], 1, queue[3] );
                znorm = znorm_h[3][0];
                zden  = znorm_h[3][1];
                zeta = znorm/zden;
                #else
                magmablas_dlacpy( MagmaFull, ione, ione, znorm_d[3], ione, zeta_d[3], ione, queue[3] );
                magma_dtrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                             ione, ione,
                             one, &znorm_d[3][1], ione,
                                   zeta_d[3], ione, queue[3] );
                cudaMemcpyPeerAsync( zeta_d[1], gpu_id+1, zeta_d[3], gpu_id+3, sizeof(double),
                                     magma_queue_get_cuda_stream( queue[3] ) );
                magma_event_record( event[3], queue[3] );
                magma_queue_wait_event( queue[1], event[3] );
                #endif
            } else if (d == 0) {
                magma_queue_wait_event( queue[d], event[2] );
                magma_daxpy( 2, one,  &znorm_d[d][2], 1, &znorm_d[d][0], 1, queue[d] );

                // compute, copy it back, zeta = znorm/zden
                #if defined(DDOT_REDUCE_ON_CPU)
                magma_dgetvector( 2, znorm_d[0], 1, znorm_h[0], 1, queue[0] );
                znorm = znorm_h[0][0];
                zden  = znorm_h[0][1];
                zeta = znorm/zden;
                #else
                magmablas_dlacpy( MagmaFull, ione, ione, znorm_d[0], ione, zeta_d[0], ione, queue[0] );
                magma_dtrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                             ione, ione,
                             one, &znorm_d[0][1], ione,
                                   zeta_d[0], ione, queue[0] );
                cudaMemcpyPeerAsync( zeta_d[2], gpu_id+2, zeta_d[0], gpu_id+0, sizeof(double),
                                     magma_queue_get_cuda_stream( queue[0] ) );
                magma_event_record( event[0], queue[0] );
                magma_queue_wait_event( queue[2], event[0] );
                #endif
            }
            if (dd == swc_dd) offset += nd_d;
        }
#endif
        #if defined(PROF_MAGMA_BATCH)
        for (d=1; d<gpus_per_proc; d++) {
            magma_queue_sync( queue[d] );
        }
        time_dot += (MPI_Wtime()-tic);
        #endif
        //
        // u(:nd) = u(:nd) + alpha*zkp(:nd) + zeta*zkt(:nd) (distributed)
#if 0
        for (d=0; d<gpus_per_proc; d++) {
            magma_setdevice(gpu_id+d);
            magma_daxpy( *nd, alpha, zkp[d], ione, u[d], ione, queue[d] );
            magma_daxpy( *nd, zeta,  zkt[d], ione, u[d], ione, queue[d] );
        }
#else
        offset = 0;
        for (d=0; d<gpus_per_proc; d++) {
            int nd_d = (d == gpus_per_proc-1 ? *nd-d*nd_loc : nd_loc);
            magma_setdevice(gpu_id+d);
            #if defined(DDOT_REDUCE_ON_CPU)
            magma_daxpy( nd_d, alpha, &(zkp[d][offset]), ione, &(u[d][offset]), ione, queue[d] );
            magma_daxpy( nd_d, zeta,  &(zkt[d][offset]), ione, &(u[d][offset]), ione, queue[d] );
            #else
            device_daxpy( nd_d, alpha_d[d], &(zkp[d][offset]), ione, &(u[d][offset]), ione, &queue[d] );
            device_daxpy( nd_d, zeta_d[d],  &(zkt[d][offset]), ione, &(u[d][offset]), ione, &queue[d] );
            #endif
            offset += nd_d;
        }
#endif
        // zr(:nd) = zt(:nd) - zeta*zakt(:nd)
        for (d=0; d<gpus_per_proc; d++) {
            magma_setdevice(gpu_id+d);
            magmablas_dlacpy( MagmaFull, *nd, ione, zt[d], *nd, zr[d], *nd, queue[d] );
            #if defined(DDOT_REDUCE_ON_CPU)
            magma_daxpy( *nd, -zeta, zakt[d], ione, zr[d], ione, queue[d] );
            #else
            magmablas_dlascl( MagmaFull, ione, ione, -one, one, ione, ione, zeta_d[d], ione, queue[d], &info );
            device_daxpy( *nd, zeta_d[d], zakt[d], ione, zr[d], ione, &queue[d] );
            #endif
        }
        // beta = alpha/zeta * HACApK_dotp_d(nd,zshdw,zr)/znormold;
        #if defined(PROF_MAGMA_BATCH)
        for (d=1; d<gpus_per_proc; d++) {
            magma_queue_sync( queue[d] );
        }
        tic = MPI_Wtime();
        #endif
#if 0
        magma_setdevice(gpu_id);
        //beta   = magma_ddot( *nd, zshdw[0], ione, zr[0], ione, queue[0]);
        //zrnorm = magma_ddot( *nd, zr[0],    ione, zr[0], ione, queue[0] );
        d = (dd0+2)%gpus_per_proc;
        znorm_h[d][0] = magma_ddot( *nd, zshdw[0], ione, zr[0], ione, queue[0]);
        znorm_h[d][1] = magma_ddot( *nd, zr[0],    ione, zr[0], ione, queue[0] );
#else
        offset = 0;
        for (dd=0; dd<gpus_per_proc; dd+=inc_dd) {
            d = (dd0+dd)%gpus_per_proc;
            int nd_d = (dd > swc_dd ? *nd-nd_loc2 : nd_loc2);
            magma_setdevice(gpu_id+d);
            #if 1
            cublasSetPointerMode( magma_queue_get_cublas_handle( queue[d] ), CUBLAS_POINTER_MODE_DEVICE );
            cublasDdot( magma_queue_get_cublas_handle( queue[d] ), nd_d, 
                        &(zr[d][offset]), ione, &(zshdw[d][offset]), ione, &znorm_d[d][0] );
            cublasDdot( magma_queue_get_cublas_handle( queue[d] ), nd_d,
                        &(zr[d][offset]), ione, &(zr[d][offset]), ione, &znorm_d[d][1] );
            cublasSetPointerMode(magma_queue_get_cublas_handle( queue[d] ), CUBLAS_POINTER_MODE_HOST);
            #else
            magma_dgemm( MagmaTrans, MagmaNoTrans, 2, 1, nd_d,
                         one, &(zshdw[d][offset]), *nd,
                              &(zr[d][offset]), *nd,
                         zero,  znorm_d[d], 2, queue[d] );
            #endif
            if (d == 1) { // GPU-1 -> GPU-3
                cudaMemcpyPeerAsync( &znorm_d[3][2], gpu_id+3, znorm_d[d], gpu_id+d, 2*sizeof(double),
                                     magma_queue_get_cuda_stream( queue[d] ) );
                magma_event_record( event[d], queue[d] );
            } else if (d == 2) { // GPU-2 -> GPU->0
                cudaMemcpyPeerAsync( &znorm_d[0][2], gpu_id+0, znorm_d[d], gpu_id+d, 2*sizeof(double),
                                     magma_queue_get_cuda_stream( queue[d] ) );
                magma_event_record( event[d], queue[d] );
            } else if (d == 3) {
                magma_queue_wait_event( queue[d], event[1] );
                magma_daxpy( 2, one,  &znorm_d[d][2], 1, &znorm_d[d][0], 1, queue[d] );
                // for convergence check.
                magma_dgetvector_async( 2, znorm_d[d], 1, znorm_h[d], 1, queue[d] );
                #if !defined(DDOT_REDUCE_ON_CPU)
                // compute, copy it back, alpha=-znorm/zden
                cudaMemcpyPeerAsync( znorm_d[1], gpu_id+1, znorm_d[3], gpu_id+3, sizeof(double),
                                     magma_queue_get_cuda_stream( queue[3] ) );
                magma_event_record( event[3], queue[3] );
                magma_queue_wait_event( queue[1], event[3] );
                #endif
            } else if (d == 0) {
                magma_queue_wait_event( queue[d], event[2] );
                magma_daxpy( 2, one,  &znorm_d[d][2], 1, &znorm_d[d][0], 1, queue[d] );
                // compute, copy it back, alpha=-znorm/zden
                #if defined(DDOT_REDUCE_ON_CPU)
                magma_dgetvector_async( 2, znorm_d[d], 1, znorm_h[d], 1, queue[d] );
                #else
                cudaMemcpyPeerAsync( znorm_d[2], gpu_id+2, znorm_d[0], gpu_id+0, sizeof(double),
                                     magma_queue_get_cuda_stream( queue[0] ) );
                magma_event_record( event[0], queue[0] );
                magma_queue_wait_event( queue[2], event[0] );
                #endif
            }
            if (dd == swc_dd) offset += nd_d;
        }
#endif
        #if defined(PROF_MAGMA_BATCH)
        for (d=1; d<gpus_per_proc; d++) {
            magma_queue_sync( queue[d] );
        }
        time_dot += (MPI_Wtime()-tic);
        #endif
        //
        //beta *= (alpha/zeta) / znormold;
        #if defined(DDOT_REDUCE_ON_CPU)
        d = (dd0+2)%gpus_per_proc;
        magma_setdevice(gpu_id+d);
        magma_queue_sync( queue[d] );
        beta = znorm_h[d][0];
        beta = -alpha/zeta * beta/znormold;
        zrnorm = sqrt(znorm_h[d][1]);
        #else
        for (d=0; d<gpus_per_proc; d++) {
            magma_setdevice(gpu_id+d);
            // znormold = zeta/znormold
            magma_dgemm( MagmaNoTrans, MagmaNoTrans, 1, 1, 1,
                         one,   zeta_d[d], 1,
                               &alpha_d[d][1], 1,
                         zero, &alpha_d[d][0], 1, queue[d] );
            // beta = beta / znormold
            magmablas_dlacpy( MagmaFull, 1, 1, znorm_d[d], 1, beta_d[d], 1, queue[d] );
            magma_dtrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                         ione, ione,
                        -one, &alpha_d[d][0], ione,
                               beta_d[d], ione, queue[d] );
        }
        magma_setdevice(gpu_id+3);
        magma_queue_sync( queue[3] );
        zrnorm = sqrt(znorm_h[3][1]);
        #endif
        // from GPU-3
        *nstp = step;
        en_measure_time = MPI_Wtime();
        time = en_measure_time - st_measure_time;
        if (st_ctl->param[0] > 0 && mpinr == 0) {
            printf( " %d: time=%.2e log10(zrnorm/bnorm)=log10(%.2e/%.2e)=%.2e\n",step,time,zrnorm,bnorm,log10(zrnorm/bnorm) );
        }
    }
    //magma_dgetvector( *nd, u[0], 1, u_cpu, 1, queue[0] );
    int offset = 0;
    for (d=0; d<gpus_per_proc; d++) {
        int nd_d = (d == gpus_per_proc-1 ? *nd-d*nd_loc : nd_loc);
        magma_setdevice(gpu_id+d);
        magma_dgetvector( nd_d, &(u[d][offset]), 1, &(u_cpu[offset]), 1, queue[d] );
    }
    MPI_Barrier( icomm );
    en_measure_time = MPI_Wtime();
    time = en_measure_time - st_measure_time;
    if (st_ctl->param[0]>0) {
        //printf( " End: %d, %.2e\n",mpinr,time );
        if (mpinr == 0) {
            printf( "       BiCG       = %.5e\n", time );
            printf( "        time_mpi  = %.5e\n", time_mpi );
            printf( "        time_copy = %.5e\n", time_copy );
            printf( "        time_spmv = %.5e\n", time_spmv );
            printf( "        > time_batch = %.5e\n", time_batch );
            printf( "        > time_set   = %.5e\n", time_set );
            printf( "          + time_set1 = %.5e\n", time_set1 );
            printf( "          + time_set2 = %.5e\n", time_set2 );
            printf( "          + time_set3 = %.5e\n", time_set3 );
            printf( "          + time_set4 = %.5e\n", time_set4 );
            printf( "          +          += %.5e\n", time_set1+time_set2+time_set3+time_set4 );
            printf( "        > time_dot   = %.5e\n", time_dot );
            printf( "                     => %.5e\n", time_mpi+time_copy+time_spmv+time_dot );
        }
    }

    // free gpu memory
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice(gpu_id+d);
        magma_queue_sync( queue[d] );
        magma_queue_destroy( queue[d] );
        magma_queue_destroy( queue[d+gpus_per_proc] );

#if !defined(redundant_u)
        magma_free(u[d]);
#endif
        magma_free(b[d]);

        magma_free(zt[d]);
        magma_free(zp[d]);
        magma_free(zkp[d]);
        magma_free(zakp[d]);
        magma_free(zkt[d]);
        magma_free(zakt[d]);
        magma_free(zshdw[d]);
    }
    // delete matrix
    c_hacapk_adot_body_lfdel_mgpu_(st_leafmtxp);

    magma_setdevice(gpu_id);
    // free cpu memory
    free(queue);
    if (buffer != NULL) free(buffer);
    free(disps);
#if !defined(redundant_u)
    free(u);
#endif
    free(b);
    free(zt);
    free(zr);
    free(zp);
    free(zkp);
    free(zkt);
    free(zshdw);

    magma_free_pinned(zau_cpu1);
    magma_free_pinned(zau_cpu2);
    magma_free_pinned(wws_cpu);
    magma_free_pinned(wwr_cpu);

    magma_free(wws);
    magma_free(wwr);
    // setup peers
    for (d=1; d<gpus_per_proc; d++) {
        magma_setdevice(gpu_id+d);
        int dest;
        for (dest=0; dest<gpus_per_proc; dest++) {
            if (dest != d) cudaDeviceDisablePeerAccess( gpu_id+dest );
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// pipelined version
// 
// on one GPU / proc
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
        MAGMA_SUCCESS != magma_dmalloc(&zw, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zq, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zt, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zr, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zp, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zz, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zs, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zb, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zx, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zy, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zv, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&wws, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&wwr, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&zr0, *nd)) {
      printf( " failed to allocate vectors (nd=%d)\n",*nd );
    }
    // use pinned memory for buffer
    //wws_cpu = (double*)malloc((*nd) * sizeof(double));
    //wwr_cpu = (double*)malloc((*nd) * sizeof(double));
    if (MAGMA_SUCCESS != magma_dmalloc_pinned(&wws_cpu, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc_pinned(&wwr_cpu, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc_pinned(&zau_cpu, *nd)) {
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
    bnorm = magma_ddot(*nd, b, ione, b, ione, queue); 
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
    znorm = magma_ddot(*nd, zr, ione, zr, ione, queue); 
    zrnorm = magma_ddot(*nd, zw, ione, zr, ione, queue); 
    alpha = znorm/zrnorm;
    //if (mpinr == 0) printf( " alpha=%.2e/%.2e=%.2e\n",znorm,zrnorm, alpha );
    zrnorm = sqrt(znorm);
    if (mpinr == 0) {
        printf( "\n ** pipelined BICG with MAGMA batched on GPU **\n" );
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
        zrnorm = magma_ddot(*nd, zy, ione, zq, ione, queue); 
        zden = magma_ddot( *nd, zy, ione, zy, ione, queue);
        zeta = zrnorm/zden;
        //if (mpinr == 0) printf( " %d: zeta=%.2e/%.2e=%.2e, alpha=%.2e\n",step,zrnorm,zden,zeta, alpha );
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
        znorm  = magma_ddot(*nd, zr, ione, zr0, ione, queue); 
        zwnorm = magma_ddot(*nd, zw, ione, zr0, ione, queue); 
        zsnorm = magma_ddot(*nd, zs, ione, zr0, ione, queue); 
        zznorm = magma_ddot(*nd, zz, ione, zr0, ione, queue); 
        //if (mpinr == 0) printf( " %d: znorm=%.2e zwnorm=%.2e zsnorm=%.2e nnzorm=%.2e\n",step,znorm,zwnorm,zsnorm,zznorm );
        // beta
        beta = (alpha/zeta)*(znorm/znormold);
        alpha = znorm/(zwnorm+beta*zsnorm-beta*zeta*zznorm);
        // > znorm = zr'*zr
        zrnorm = magma_ddot(*nd, zr, ione, zr, ione, queue); 
        zrnorm = sqrt(zrnorm);
        *nstp = step;
        en_measure_time = MPI_Wtime();
        time = en_measure_time - st_measure_time;
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

    magma_free(zz);
    magma_free(zw);

    magma_free(zr0);

    magma_free(zt);
    magma_free(zr);
    magma_free(zp);
    magma_free(zq);
    magma_free(zz);
    magma_free(zs);
    magma_free(zb);
    magma_free(zx);
    magma_free(zy);
    magma_free(zv);
}

#endif
