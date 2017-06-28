#include	<stdio.h>
#include	<stdlib.h>
#include	<time.h>
#include	"omp.h"
#include	"mpi.h"
#include	"HACApK_MAGMA.h"
#include        "magma_dlapack.h"

// !! BiCG in C !!
double c_hacapk_dotp_d(int nd, double *b, double *a) {
    double norm = 0.0;
    int ii;
    for (ii=0; ii<nd; ii++) {
        norm += b[ii]*a[ii];
    }
    return norm;
}


void c_hacapk_adot_cax_lfmtx_comm2(stc_HACApK_lcontrol *st_ctl,
                                   double *wws, double *wwr, int nd) {
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
        lapackf77_dlaset( "F", &nd, &ione, &one, &one, wws, &nd );
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

void c_hacapk_adot_cax_lfmtx_comm(double *zau, stc_HACApK_lcontrol *st_ctl,
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

#if 1
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
    }
}

void c_hacapk_adot_cax_lfmtx_comm_gpu(int flag, double *zau_gpu, double *zau,
                                      stc_HACApK_lcontrol *st_ctl,
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
        tic = MPI_Wtime();
#if 0
        magma_dgetvector( nd, zau_gpu, 1, zau, 1, queue );
#else
        if (flag == 1) {
            int ione = 1;
            double zero = 0.0;
            lapackf77_dlaset( "F", &nd, &ione, &zero, &zero, zau, &nd );
            magma_dgetvector( lnp[mpinr], &zau_gpu[lsp[mpinr]-1], 1, &zau[lsp[mpinr]-1], 1, queue );
        }
#endif
        *time_copy += MPI_Wtime()-tic;

        c_hacapk_adot_cax_lfmtx_comm(zau, st_ctl, wws,wwr, isct,irct, nd,time_mpi);

        magma_queue_sync( queue );
        tic = MPI_Wtime();
        magma_dsetvector( nd, zau, 1, zau_gpu, 1, queue );
        *time_copy += MPI_Wtime()-tic;
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
    int mpinr, nrank, icomm, ierr;
    double time_spmv, time_mpi, time_batch, time_set, time_copy, tic;
 
    mstep = param[82];
    eps = param[90];
    mpinr = lpmd[2]; 
    nrank = lpmd[1]; 
    icomm = MPI_COMM_WORLD; //lpmd[0];
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
    lapackf77_dlaset( "F", nd, &ione, &zero, &zero, zp, nd );
    lapackf77_dlaset( "F", nd, &ione, &zero, &zero, zakp, nd );
    lapackf77_dlacpy( "F", nd, &ione, b, nd, zr, nd );
    //  .. SpMV ..
    tic = MPI_Wtime();
    lapackf77_dlaset( "F", nd, &ione, &zero, &zero, zshdw, nd );
    c_hacapk_adot_body_lfmtx_batch_(zshdw,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy);
    time_spmv += (MPI_Wtime()-tic);
    c_hacapk_adot_cax_lfmtx_comm(zshdw, st_ctl, wws, wwr, isct, irct, *nd, &time_mpi);
    //
    blasf77_daxpy( nd, &mone, zshdw, &ione, zr, &ione );
    lapackf77_dlacpy( "F", nd, &ione, zr, nd, zshdw, nd );
    zrnorm = c_hacapk_dotp_d(*nd, zr, zr ); 
    zrnorm = sqrt(zrnorm);
    if (mpinr == 0) {
        printf( "\n ** BICG (c version, flag) **\n" );
        printf( "\nOriginal relative residual norm = %.2e/%.2e = %.2e\n",zrnorm,bnorm,zrnorm/bnorm );
        printf( "HACApK_bicgstab_lfmtx_flat start\n" );
    }
    for ( step=1; step<=mstep; step++ ) {
        if (zrnorm/bnorm < eps) break;
        // zp(:nd) = zr(:nd) + beta*(zp(:nd) - zeta*zakp(:nd))
        blasf77_daxpy( nd, &zeta, zakp, &ione, zp, &ione );
        lapackf77_dlascl( "G", &ione, &ione, &one, &beta, nd, &ione, zp, nd, &info );
        blasf77_daxpy( nd, &one, zr, &ione, zp, &ione );
        // zkp(:nd) = zp(:nd)
        lapackf77_dlacpy( "F", nd, &ione, zp, nd, zkp, nd );
        //  .. SpMV ..
        lapackf77_dlaset( "F", nd, &ione, &zero, &zero, zakp, nd );
        tic = MPI_Wtime();
        c_hacapk_adot_body_lfmtx_batch_(zakp,st_leafmtxp,zkp,wws, &time_batch,&time_set,&time_copy);
        time_spmv += (MPI_Wtime()-tic);
        c_hacapk_adot_cax_lfmtx_comm(zakp,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
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
        c_hacapk_adot_cax_lfmtx_comm(zakt,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
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
        if (st_ctl->param[0] > 0 && mpinr==0) {
            printf( " %d: time=%.2e log10(zrnorm/bnorm)=log10(%.2e/%.2e)=%.2e\n",step,time,zrnorm,bnorm,log10(zrnorm/bnorm) );
        }
    }
    MPI_Barrier( icomm );
    en_measure_time = MPI_Wtime();
    time = en_measure_time - st_measure_time;
    // delete matrix
    c_hacapk_adot_body_lfdel_batch_(st_leafmtxp);
    if (st_ctl->param[0]>0) {
        //printf( " End: %d, %.2e\n",mpinr,time );
        if (mpinr==0) {
            printf( "       BiCG        = %.5e\n", time );
            printf( "        time_mpi   = %.5e\n", time_mpi );
            printf( "        time_spmv  = %.5e\n", time_spmv );
            printf( "        > time_copy  = %.5e\n", time_copy );
            printf( "        > time_set   = %.5e\n", time_set );
            printf( "        > time_batch = %.5e\n", time_batch );
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
    int mpinr, nrank, icomm, ierr;
    double time_spmv, time_mpi, time_batch, time_set, time_copy, tic;
 
    mstep = param[82];
    eps = param[90];
    mpinr = lpmd[2]; 
    nrank = lpmd[1]; 
    icomm = MPI_COMM_WORLD; //lpmd[0];
    MPI_Barrier( icomm );

    int on_gpu = 1;
    magma_device_t cdev;
    magma_queue_t queue;

    magma_setdevice( get_device_id(st_leafmtxp) );
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

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
    magma_dmalloc_pinned(&wws_cpu, *nd);
    magma_dmalloc_pinned(&wwr_cpu, *nd);
    magma_dmalloc_pinned(&zau_cpu, *nd);

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
    magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zp, *nd, queue );
    magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zakp, *nd, queue );
    magma_queue_sync( queue );
    //  .. SpMV ..
    magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zshdw, *nd, queue );
    tic = MPI_Wtime();
    c_hacapk_adot_body_lfmtx_batch_queue(zshdw,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy, 
                                         on_gpu, queue);
    magma_queue_sync( queue );
    time_spmv += (MPI_Wtime()-tic);
    c_hacapk_adot_cax_lfmtx_comm_gpu(1, zshdw, zau_cpu, st_ctl, wws_cpu, wwr_cpu, isct, irct, *nd, 
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
        printf( "HACApK_bicgstab_lfmtx_flat start\n" );
    }
    for ( step=1; step<=mstep; step++ ) {
        if (zrnorm/bnorm < eps) break;
        // zp(:nd) = zr(:nd) + beta*(zp(:nd) - zeta*zakp(:nd))
        magma_daxpy( *nd, zeta, zakp, ione, zp, ione, queue );
        magmablas_dlascl( MagmaFull, ione, ione, one, beta, *nd, ione, zp, *nd, queue, &info );
        magma_daxpy( *nd, one, zr, ione, zp, ione, queue );
        // zkp(:nd) = zp(:nd)
        magmablas_dlacpy( MagmaFull, *nd, ione, zp, *nd, zkp, *nd, queue );
        //  .. SpMV ..
        magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zakp, *nd, queue );
        magma_queue_sync( queue );
        tic = MPI_Wtime();
        c_hacapk_adot_body_lfmtx_batch_queue(zakp,st_leafmtxp,zkp,wws, &time_batch,&time_set,&time_copy,
                                             on_gpu, queue);
        magma_queue_sync( queue );
        time_spmv += (MPI_Wtime()-tic);
        c_hacapk_adot_cax_lfmtx_comm_gpu(1, zakp, zau_cpu, st_ctl, wws_cpu,wwr_cpu, isct,irct,*nd, 
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
        magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zakt, *nd, queue );
        magma_queue_sync( queue );
        tic = MPI_Wtime();
        c_hacapk_adot_body_lfmtx_batch_queue(zakt,st_leafmtxp,zkt,wws, &time_batch,&time_set,&time_copy,
                                             on_gpu, queue);
        magma_queue_sync( queue );
        time_spmv += (MPI_Wtime()-tic);
        c_hacapk_adot_cax_lfmtx_comm_gpu(1, zakt,zau_cpu, st_ctl, wws_cpu,wwr_cpu, isct,irct,*nd, 
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
        if (st_ctl->param[0] > 0 && mpinr==0) {
            printf( " %d: time=%.2e log10(zrnorm/bnorm)=log10(%.2e/%.2e)=%.2e\n",step,time,zrnorm,bnorm,log10(zrnorm/bnorm) );
        }
    }
    magma_dgetvector( *nd, u, 1, u_cpu, 1, queue );
    MPI_Barrier( icomm );
    en_measure_time = MPI_Wtime();
    time = en_measure_time - st_measure_time;
    // delete matrix
    c_hacapk_adot_body_lfdel_batch_(st_leafmtxp);
    if (st_ctl->param[0]>0) {
        //printf( " End: %d, %.2e\n",mpinr,time );
        if (mpinr==0) {
            printf( "       BiCG        = %.5e\n", time );
            printf( "        time_mpi   = %.5e\n", time_mpi );
            printf( "        time_copy  = %.5e\n", time_copy );
            printf( "        time_spmv  = %.5e\n", time_spmv );
            printf( "        > time_batch = %.5e\n", time_batch );
            printf( "        > time_set   = %.5e\n", time_set );
        }
    }

    magma_queue_sync( queue );
    magma_queue_destroy( queue );
    //free(wws_cpu);
    //free(wwr_cpu);
    magma_free_pinned(wws_cpu);
    magma_free_pinned(wwr_cpu);
    magma_free_pinned(zau_cpu);

#if 1
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
#endif
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
    double *zr, *zshdw, *zp, *zt, *zkp, *zakp, *zkt, *zakt;
    double *b = NULL, *u = NULL, *wws, *wwr, *wws_cpu, *wwr_cpu;
    int *lpmd = (int*)((void*)st_ctl->param + st_ctl->lpmd_offset);
    int isct[2], irct[2];
    // local variables
    double eps, alpha, beta, zeta, zz, zden, znorm, znormold, bnorm, zrnorm;
    double en_measure_time, st_measure_time, time;
    int info, step, mstep;
    int mpinr, nrank, icomm, ierr;
    double time_spmv, time_mpi, time_batch, time_set, time_copy, tic;
 
    mstep = param[82];
    eps = param[90];
    mpinr = lpmd[2]; 
    nrank = lpmd[1]; 
    icomm = MPI_COMM_WORLD; //lpmd[0];
    MPI_Barrier( icomm );

    #define ACCUM_ON_CPU
    #if defined(ACCUM_ON_CPU)
    int flag = 1;
    #else
    int flag = 0;
    #endif
    int on_gpu = 1, d, gpu_id = get_device_id(st_leafmtxp);
    magma_device_t cdev;
    magma_queue_t *queue = (magma_queue_t *)malloc(gpus_per_proc * sizeof(magma_queue_t));
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice((gpu_id+d)%procs_per_node);
        magma_getdevice( &cdev );
        magma_queue_create( cdev, &queue[d] );
    }
    // main GPU
    magma_setdevice(gpu_id);

    // use pinned memory for CPU buffer
    magma_dmalloc_pinned(&wws_cpu, *nd);
    magma_dmalloc_pinned(&wwr_cpu, (*nd)*gpus_per_proc);

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
    // copy matrix to GPU
    c_hacapk_adot_body_lfcpy_batch_sorted_mgpu_(nd, st_leafmtxp, queue);
    //
    time_spmv = 0.0;
    time_mpi = 0.0;
    time_batch = 0.0;
    time_set = 0.0;
double time_set2 = 0.0;
double time_set3 = 0.0;
    time_copy = 0.0;
    MPI_Barrier( icomm );
    st_measure_time = MPI_Wtime();
    // copy the input vector to GPU
    magma_dsetvector_async( *nd, b_cpu, 1, b, 1, queue[0] );
    magma_dsetvector_async( *nd, u_cpu, 1, u, 1, queue[0] );
    // init
    alpha = 0.0; beta = 0.0; zeta = 0.0;
    zz = magma_ddot(*nd, b, ione, b, ione, queue[0]); 
    bnorm=sqrt(zz);
    magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zp, *nd, queue[0] );
    magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zakp, *nd, queue[0] );
    magma_queue_sync( queue[0] );
    // .. SpMV ..
    magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zshdw, *nd, queue[0] );
    tic = MPI_Wtime();
    c_hacapk_adot_body_lfmtx_batch_mgpu(zshdw,st_leafmtxp,st_ctl,u,wws, wws_cpu,wwr_cpu,
                                        &time_batch,&time_set,&time_copy,
&time_set2, &time_set3,
                                        on_gpu, queue);
    magma_queue_sync( queue[0] );
    time_spmv += (MPI_Wtime()-tic);
    c_hacapk_adot_cax_lfmtx_comm_gpu(flag, zshdw, wws_cpu,
                                     st_ctl, u_cpu, wwr_cpu, isct, irct, *nd, 
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
        printf( "HACApK_bicgstab_lfmtx_flat start\n" );
    }
    for ( step=1; step<=mstep; step++ ) {
        if (zrnorm/bnorm < eps) break;
        // zp(:nd) = zr(:nd) + beta*(zp(:nd) - zeta*zakp(:nd))
        magma_daxpy( *nd, zeta, zakp, ione, zp, ione, queue[0] );
        magmablas_dlascl( MagmaFull, ione, ione, one, beta, *nd, ione, zp, *nd, queue[0], &info );
        magma_daxpy( *nd, one, zr, ione, zp, ione, queue[0] );
        // zkp(:nd) = zp(:nd)
        magmablas_dlacpy( MagmaFull, *nd, ione, zp, *nd, zkp, *nd, queue[0] );
        //  .. SpMV ..
        magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zakp, *nd, queue[0] );
        magma_queue_sync( queue[0] );
        tic = MPI_Wtime();
        c_hacapk_adot_body_lfmtx_batch_mgpu(zakp,st_leafmtxp,st_ctl, zkp,wws, wws_cpu,wwr_cpu,
                                            &time_batch,&time_set,&time_copy, 
&time_set2, &time_set3,
                                            on_gpu, queue);
        magma_queue_sync( queue[0] );
        time_spmv += (MPI_Wtime()-tic);
        c_hacapk_adot_cax_lfmtx_comm_gpu(flag, zakp, wws_cpu,
                                         st_ctl, u_cpu,wwr_cpu, isct,irct,*nd, 
                                         &time_copy,&time_mpi, queue[0]);
        //
        znorm = magma_ddot( *nd, zshdw, ione, zr, ione, queue[0] ); 
        zden = magma_ddot( *nd, zshdw, ione, zakp, ione, queue[0] );
        alpha = -znorm/zden;
        znormold = znorm;
        // zt(:nd) = zr(:nd) - alpha*zakp(:nd)
        magmablas_dlacpy( MagmaFull, *nd, ione, zr, *nd, zt, *nd, queue[0] );
        magma_daxpy( *nd, alpha, zakp, ione, zt, ione, queue[0] );
        alpha = -alpha;
        // zkt(:nd) = zt(:nd)
        magmablas_dlacpy( MagmaFull, *nd, ione, zt, *nd, zkt, *nd, queue[0] );
        //  .. SpMV ..
        magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zakt, *nd, queue[0] );
        magma_queue_sync( queue[0] );
        tic = MPI_Wtime();
        c_hacapk_adot_body_lfmtx_batch_mgpu(zakt,st_leafmtxp,st_ctl, zkt,wws, wws_cpu,wwr_cpu, 
                                            &time_batch,&time_set,&time_copy,
&time_set2, &time_set3,
                                            on_gpu, queue);
        magma_queue_sync( queue[0] );
        time_spmv += (MPI_Wtime()-tic);
        c_hacapk_adot_cax_lfmtx_comm_gpu(flag, zakt,wws_cpu,
                                         st_ctl, u_cpu,wwr_cpu, isct,irct,*nd, 
                                         &time_copy,&time_mpi, queue[0]);
        //
        znorm = magma_ddot( *nd, zakt, ione, zt, ione, queue[0] ); 
        zden = magma_ddot( *nd, zakt, ione, zakt, ione, queue[0] );
        zeta = znorm/zden;
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
        if (st_ctl->param[0] > 0 && mpinr==0) {
            printf( " %d: time=%.2e log10(zrnorm/bnorm)=log10(%.2e/%.2e)=%.2e\n",step,time,zrnorm,bnorm,log10(zrnorm/bnorm) );
        }
    }
    magma_dgetvector( *nd, u, 1, u_cpu, 1, queue[0] );
    MPI_Barrier( icomm );
    en_measure_time = MPI_Wtime();
    time = en_measure_time - st_measure_time;
    // delete matrix
    c_hacapk_adot_body_lfdel_mgpu_(st_leafmtxp);
    if (st_ctl->param[0]>0) {
        //printf( " End: %d, %.2e\n",mpinr,time );
        if (mpinr==0) {
            printf( "       BiCG        = %.5e\n", time );
            printf( "        time_mpi   = %.5e\n", time_mpi );
            printf( "        time_copy  = %.5e\n", time_copy );
            printf( "        time_spmv  = %.5e\n", time_spmv );
            printf( "        > time_batch = %.5e\n", time_batch );
            printf( "        > time_set   = %.5e\n", time_set );
printf( " > time_set2  = %.5e\n", time_set2 );
printf( " > time_set3  = %.5e\n", time_set3 );
        }
    }

    for (d=0; d<gpus_per_proc; d++) {
        magma_queue_destroy( queue[d] );
    }
    free(queue);

    magma_free_pinned(wws_cpu);
    magma_free_pinned(wwr_cpu);

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
}


///////////////////////////////////////////////////////////////////////////
// pipelined version
// 
// on one GPU / proc
void c_hacapk_bicgstab_cax_lfmtx_pipe_(stc_HACApK_leafmtxp *st_leafmtxp, stc_HACApK_lcontrol *st_ctl,
                                       double *u_cpu, double *b_cpu, double*param, int *nd, int *nstp, int *lrtrn) {
    // local constants
    int ione = 1;
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
    int mpinr, nrank, icomm, ierr;
    double time_spmv, time_mpi, time_batch, time_set, time_copy, tic;
 
    mstep = param[82];
    eps = param[90];
    mpinr = lpmd[2]; 
    nrank = lpmd[1]; 
    icomm = MPI_COMM_WORLD; //lpmd[0];
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
    //lapackf77_dlaset( "F", nd, &ione, &zero, &zero, wws_cpu, nd );
    //lapackf77_dlaset( "F", nd, &ione, &zero, &zero, wwr_cpu, nd );
    //lapackf77_dlaset( "F", nd, &ione, &zero, &zero, zau_cpu, nd );
    //MPI_Barrier( icomm );
    #define WARMUP_MPI
    #ifdef WARMUP_MPI
    c_hacapk_adot_cax_lfmtx_comm2(st_ctl, wws_cpu, wwr_cpu, *nd);
    #endif
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
    bnorm = magma_ddot(*nd, b, ione, b, ione, queue); 
    bnorm=sqrt(bnorm);
    //  .. SpMV: zv=A*u ..
    magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zv, *nd, queue );
    magma_queue_sync( queue );
    tic = MPI_Wtime();
    c_hacapk_adot_body_lfmtx_batch_queue(zv,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy, 
                                         on_gpu, queue);
    magma_queue_sync( queue );
    time_spmv += (MPI_Wtime()-tic);
    c_hacapk_adot_cax_lfmtx_comm_gpu(1, zv, zau_cpu, st_ctl, 
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
    c_hacapk_adot_body_lfmtx_batch_queue(zw,st_leafmtxp,zr,wws, &time_batch,&time_set,&time_copy, 
                                         on_gpu, queue);
    magma_queue_sync( queue );
    time_spmv += (MPI_Wtime()-tic);
    c_hacapk_adot_cax_lfmtx_comm_gpu(1, zw, zau_cpu, st_ctl, 
                                     wws_cpu, wwr_cpu, isct, irct, *nd, 
                                     &time_copy,&time_mpi, queue);
    //  .. SpMV: zt = A*zw ..
    magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zt, *nd, queue );
    magma_queue_sync( queue );
    tic = MPI_Wtime();
    c_hacapk_adot_body_lfmtx_batch_queue(zt,st_leafmtxp,zw,wws, &time_batch,&time_set,&time_copy, 
                                         on_gpu, queue);
    magma_queue_sync( queue );
    time_spmv += (MPI_Wtime()-tic);
    c_hacapk_adot_cax_lfmtx_comm_gpu(1, zt, zau_cpu, st_ctl, 
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
        printf( "HACApK_bicgstab_lfmtx_flat start\n" );
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
        //if (mpinr == 0) printf( " zeta=%.2e/%.2e=%.2e, alpha=%.2e\n",zrnorm,zden,zeta, alpha );
        //  .. SpMV: zv = A*zz ..
        magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zv, *nd, queue );
        magma_queue_sync( queue );
        tic = MPI_Wtime();
        c_hacapk_adot_body_lfmtx_batch_queue(zv,st_leafmtxp,zz,wws, &time_batch,&time_set,&time_copy,
                                             on_gpu, queue);
        magma_queue_sync( queue );
        time_spmv += (MPI_Wtime()-tic);
        c_hacapk_adot_cax_lfmtx_comm_gpu(1, zv, zau_cpu, st_ctl,
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
        // beta
        beta = (alpha/zeta)*(znorm/znormold);
        alpha = znorm/(zwnorm+beta*zsnorm-beta*zeta*zznorm);
        // > znorm = zr'*zr
        zrnorm = magma_ddot(*nd, zr, ione, zr, ione, queue); 
        zrnorm = sqrt(zrnorm);
        *nstp = step;
        en_measure_time = MPI_Wtime();
        time = en_measure_time - st_measure_time;
        if (st_ctl->param[0] > 0 && mpinr==0) {
            printf( " %d: time=%.2e log10(zrnorm/bnorm)=log10(%.2e/%.2e)=%.2e\n",step,time,zrnorm,bnorm,log10(zrnorm/bnorm) );
        }
        //  .. SpMV: zt = A*zw ..
        magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zt, *nd, queue );
        magma_queue_sync( queue );
        tic = MPI_Wtime();
        c_hacapk_adot_body_lfmtx_batch_queue(zt,st_leafmtxp,zw,wws, &time_batch,&time_set,&time_copy,
                                             on_gpu, queue);
        magma_queue_sync( queue );
        time_spmv += (MPI_Wtime()-tic);
        c_hacapk_adot_cax_lfmtx_comm_gpu(1, zt,zau_cpu, st_ctl,
                                         wws_cpu,wwr_cpu, isct,irct,*nd, 
                                         &time_copy,&time_mpi, queue);
    }
    magma_dgetvector( *nd, u, 1, u_cpu, 1, queue );
    MPI_Barrier( icomm );
    en_measure_time = MPI_Wtime();
    time = en_measure_time - st_measure_time;
    // delete matrix
    c_hacapk_adot_body_lfdel_batch_(st_leafmtxp);
    if (st_ctl->param[0]>0) {
        //printf( " End: %d, %.2e\n",mpinr,time );
        if (mpinr==0) {
            printf( "       BiCG        = %.5e\n", time );
            printf( "        time_mpi   = %.5e\n", time_mpi );
            printf( "        time_copy  = %.5e\n", time_copy );
            printf( "        time_spmv  = %.5e\n", time_spmv );
            printf( "        > time_batch = %.5e\n", time_batch );
            printf( "        > time_set   = %.5e\n", time_set );
        }
    }

    magma_queue_destroy( queue );
    magma_free_pinned(wws_cpu);
    magma_free_pinned(wwr_cpu);
    magma_free_pinned(zau_cpu);

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
