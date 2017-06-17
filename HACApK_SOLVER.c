#include	<stdio.h>
#include	<stdlib.h>
#include	<time.h>
#include	"omp.h"
#include	"mpi.h"
#include	"HACApK_MAGMA.h"


// !! BiCG in C !!
double c_hacapk_dotp_d(int nd, double *b, double *a) {
    double norm = 0.0;
    int ii;
    for (ii=0; ii<nd; ii++) {
        norm += b[ii]*a[ii];
    }
    return norm;
}


void c_hacapk_adot_cax_lfmtx_comm(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, stc_HACApK_lcontrol *st_ctl,
                                  double *wws, double *wwr, int *isct, int *irct, int nd, double *time_mpi) {
    int ione = 1;
    double one = 1.0;

    double tic;
    int *lpmd = (int*)((void*)st_ctl->param + st_ctl->lpmd_offset); 
    int mpinr = lpmd[2]; 
    int nrank = lpmd[1]; 
   
    int *lsp = (int*)((void*)st_ctl->param + st_ctl->lsp_offset);
    int *lnp = (int*)((void*)st_ctl->param + st_ctl->lnp_offset);
    MPI_Comm icomm = MPI_COMM_WORLD;
    if (nrank > 1) {
        int ic;
        int ncdp = (mpinr+1)%nrank;       // my neighbor
        int ncsp = (mpinr+nrank-1)%nrank; // my neighbor
        isct[0] = lnp[mpinr];
        isct[1] = lsp[mpinr];

        dlacpy_( "F", &lnp[mpinr], &ione, &zau[lsp[mpinr]-1], &lnp[mpinr], wws, &lnp[mpinr] );

        for (ic=1; ic<nrank; ic++) {
           MPI_Status stat;

           tic = MPI_Wtime();
#if 1
           MPI_Sendrecv(isct, 2, MPI_INT, ncdp, 1,
                        irct, 2, MPI_INT, ncsp, 1, icomm, &stat);
          
#else // read offset/size from structure
          int nctp = (ncsp-ic+nrank+1)%nrank; // where it came from
          irct[0] = lnp[nctp];
          irct[1] = lsp[nctp];
#endif
           MPI_Sendrecv(wws, isct[0], MPI_DOUBLE, ncdp, 1,
                        wwr, irct[0], MPI_DOUBLE, ncsp, 1, icomm, &stat);
           *time_mpi += (MPI_Wtime()-tic);
           daxpy_( &irct[0], &one, wwr, &ione, &zau[irct[1]-1], &ione );
           dlacpy_( "F", &irct[0], &ione, wwr, &irct[0], wws, &irct[0] );
           isct[0] = irct[0];
           isct[1] = irct[1];
        }
    }
}

void c_hacapk_adot_cax_lfmtx_comm_gpu(int flag, double *zau_gpu, double *zau,
                                      stc_HACApK_leafmtxp *st_leafmtxp, stc_HACApK_lcontrol *st_ctl,
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
            dlaset_( "F", &nd, &ione, &zero, &zero, zau, &nd );
            magma_dgetvector( lnp[mpinr], &zau_gpu[lsp[mpinr]-1], 1, &zau[lsp[mpinr]-1], 1, queue );
        }
#endif
        *time_copy += MPI_Wtime()-tic;

        c_hacapk_adot_cax_lfmtx_comm(zau, st_leafmtxp, st_ctl, wws,wwr, isct,irct, nd,time_mpi);

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
    dlaset_( "F", nd, &ione, &zero, &zero, zp, nd );
    dlaset_( "F", nd, &ione, &zero, &zero, zakp, nd );
    dlacpy_( "F", nd, &ione, b, nd, zr, nd );
    //  .. SpMV ..
    tic = MPI_Wtime();
    dlaset_( "F", nd, &ione, &zero, &zero, zshdw, nd );
    c_hacapk_adot_body_lfmtx_batch_(zshdw,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy);
    time_spmv += (MPI_Wtime()-tic);
    c_hacapk_adot_cax_lfmtx_comm(zshdw, st_leafmtxp, st_ctl, wws, wwr, isct, irct, *nd, &time_mpi);
    //
    daxpy_( nd, &mone, zshdw, &ione, zr, &ione );
    dlacpy_( "F", nd, &ione, zr, nd, zshdw, nd );
    zrnorm = c_hacapk_dotp_d(*nd, zr, zr ); 
    zrnorm = sqrt(zrnorm);
    if (mpinr == 0) {
        printf( "\n ** BICG with MAGMA batched **\n" );
        printf( "\nOriginal relative residual norm = %.2e/%.2e = %.2e\n",zrnorm,bnorm,zrnorm/bnorm );
        printf( "HACApK_bicgstab_lfmtx_flat start\n" );
    }
    for ( step=1; step<=mstep; step++ ) {
        if (zrnorm/bnorm < eps) break;
        // zp(:nd) = zr(:nd) + beta*(zp(:nd) - zeta*zakp(:nd))
        daxpy_( nd, &zeta, zakp, &ione, zp, &ione );
        dlascl_( "G", &ione, &ione, &one, &beta, nd, &ione, zp, nd, &info );
        daxpy_( nd, &one, zr, &ione, zp, &ione );
        // zkp(:nd) = zp(:nd)
        dlacpy_( "F", nd, &ione, zp, nd, zkp, nd );
        //  .. SpMV ..
        dlaset_( "F", nd, &ione, &zero, &zero, zakp, nd );
        tic = MPI_Wtime();
        c_hacapk_adot_body_lfmtx_batch_(zakp,st_leafmtxp,zkp,wws, &time_batch,&time_set,&time_copy);
        time_spmv += (MPI_Wtime()-tic);
        c_hacapk_adot_cax_lfmtx_comm(zakp,st_leafmtxp,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
        //
        znorm = c_hacapk_dotp_d(*nd, zshdw, zr ); 
        zden = c_hacapk_dotp_d(*nd, zshdw, zakp );
        alpha = -znorm/zden;
        znormold = znorm;
        // zt(:nd) = zr(:nd) - alpha*zakp(:nd)
        dlacpy_( "F", nd, &ione, zr, nd, zt, nd );
        daxpy_( nd, &alpha, zakp, &ione, zt, &ione );
        alpha = -alpha;
        // zkt(:nd) = zt(:nd)
        dlacpy_( "F", nd, &ione, zt, nd, zkt, nd );
        //  .. SpMV ..
        dlaset_( "F", nd, &ione, &zero, &zero, zakt, nd );
        tic = MPI_Wtime();
        c_hacapk_adot_body_lfmtx_batch_(zakt,st_leafmtxp,zkt,wws, &time_batch,&time_set,&time_copy);
        time_spmv += (MPI_Wtime()-tic);
        c_hacapk_adot_cax_lfmtx_comm(zakt,st_leafmtxp,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
        //
        znorm = c_hacapk_dotp_d(*nd, zakt, zt ); 
        zden = c_hacapk_dotp_d( *nd, zakt, zakt );
        zeta = znorm/zden;
        // u(:nd) = u(:nd) + alpha*zkp(:nd) + zeta*zkt(:nd)
        daxpy_( nd, &alpha, zkp, &ione, u, &ione );
        daxpy_( nd, &zeta,  zkt, &ione, u, &ione );
        // zr(:nd) = zt(:nd) - zeta*zakt(:nd)
        zeta = -zeta;
        dlacpy_( "F", nd, &ione, zt, nd, zr, nd );
        daxpy_( nd, &zeta, zakt, &ione, zr, &ione );
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
    double *b, *u, *wws, *wwr, *wws_cpu, *wwr_cpu;
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
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    if (MAGMA_SUCCESS != magma_malloc((void**)&u, (*nd) * sizeof(double)) ||
        MAGMA_SUCCESS != magma_malloc((void**)&b, (*nd) * sizeof(double)) ) {
      printf( " failed to allocate u or b (nd=%d)\n",*nd );
    }
    // use pinned memory for buffer
    //wws_cpu = (double*)malloc((*nd) * sizeof(double));
    //wwr_cpu = (double*)malloc((*nd) * sizeof(double));
    magma_dmalloc_pinned(&wws_cpu, *nd);
    magma_dmalloc_pinned(&wwr_cpu, *nd);

    magma_malloc((void**)&wws, (*nd) * sizeof(double));
    magma_malloc((void**)&wwr, (*nd) * sizeof(double));

    magma_malloc((void**)&zt, (*nd) * sizeof(double));
    magma_malloc((void**)&zr, (*nd) * sizeof(double));
    magma_malloc((void**)&zp, (*nd) * sizeof(double));
    magma_malloc((void**)&zkp, (*nd) * sizeof(double));
    magma_malloc((void**)&zakp, (*nd) * sizeof(double));
    magma_malloc((void**)&zkt, (*nd) * sizeof(double));
    magma_malloc((void**)&zakt, (*nd) * sizeof(double));
    magma_malloc((void**)&zshdw, (*nd) * sizeof(double));
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
    magma_dsetvector( *nd, b_cpu, 1, b, 1, queue );
    magma_dsetvector( *nd, u_cpu, 1, u, 1, queue );
    // init
    alpha = 0.0; beta = 0.0; zeta = 0.0;
    zz = magma_ddot(*nd, b, ione, b, ione, queue); 
    bnorm=sqrt(zz);
    magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zp, *nd, queue );
    magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zakp, *nd, queue );
    magmablas_dlacpy( MagmaFull, *nd, ione, b, *nd, zr, *nd, queue );
    magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zshdw, *nd, queue );
    magma_queue_sync( queue );
    //  .. SpMV ..
    tic = MPI_Wtime();
    c_hacapk_adot_body_lfmtx_batch_queue(zshdw,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy, 
                                         on_gpu, queue);
    magma_queue_sync( queue );
    time_spmv += (MPI_Wtime()-tic);
    c_hacapk_adot_cax_lfmtx_comm_gpu(1, zshdw, wws_cpu, st_leafmtxp, st_ctl, u_cpu, wwr_cpu, isct, irct, *nd, 
                                     &time_copy,&time_mpi, queue);
    //
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
        c_hacapk_adot_cax_lfmtx_comm_gpu(1, zakp, wws_cpu, st_leafmtxp,st_ctl, u_cpu,wwr_cpu, isct,irct,*nd, 
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
        c_hacapk_adot_cax_lfmtx_comm_gpu(1, zakt,wws_cpu, st_leafmtxp,st_ctl, u_cpu,wwr_cpu, isct,irct,*nd, 
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

    magma_queue_destroy( queue );
    //free(wws_cpu);
    //free(wwr_cpu);
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

    magma_malloc((void**)&u, (*nd) * sizeof(double));
    magma_malloc((void**)&b, (*nd) * sizeof(double));
    magma_malloc((void**)&wws, (*nd) * sizeof(double));
    magma_malloc((void**)&wwr, (*nd) * sizeof(double));

    magma_malloc((void**)&zt, (*nd) * sizeof(double));
    magma_malloc((void**)&zr, (*nd) * sizeof(double));
    magma_malloc((void**)&zp, (*nd) * sizeof(double));
    magma_malloc((void**)&zkp, (*nd) * sizeof(double));
    magma_malloc((void**)&zakp, (*nd) * sizeof(double));
    magma_malloc((void**)&zkt, (*nd) * sizeof(double));
    magma_malloc((void**)&zakt, (*nd) * sizeof(double));
    magma_malloc((void**)&zshdw, (*nd) * sizeof(double));
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
    magma_dsetvector( *nd, b_cpu, 1, b, 1, queue[0] );
    magma_dsetvector( *nd, u_cpu, 1, u, 1, queue[0] );
    // init
    alpha = 0.0; beta = 0.0; zeta = 0.0;
    zz = magma_ddot(*nd, b, ione, b, ione, queue[0]); 
    bnorm=sqrt(zz);
    magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zp, *nd, queue[0] );
    magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zakp, *nd, queue[0] );
    magmablas_dlacpy( MagmaFull, *nd, ione, b, *nd, zr, *nd, queue[0] );
    magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zshdw, *nd, queue[0] );
    magma_queue_sync( queue[0] );
    // .. SpMV ..
    tic = MPI_Wtime();
    c_hacapk_adot_body_lfmtx_batch_mgpu(zshdw,st_leafmtxp,st_ctl,u,wws, wws_cpu,wwr_cpu,
                                        &time_batch,&time_set,&time_copy,
&time_set2, &time_set3,
                                        on_gpu, queue);
    magma_queue_sync( queue[0] );
    time_spmv += (MPI_Wtime()-tic);
    c_hacapk_adot_cax_lfmtx_comm_gpu(flag, zshdw, wws_cpu,
                                     st_leafmtxp, st_ctl, u_cpu, wwr_cpu, isct, irct, *nd, 
                                     &time_copy,&time_mpi, queue[0]);
    //
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
                                         st_leafmtxp,st_ctl, u_cpu,wwr_cpu, isct,irct,*nd, 
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
                                         st_leafmtxp,st_ctl, u_cpu,wwr_cpu, isct,irct,*nd, 
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
    //c_hacapk_adot_body_lfdel_batch_(st_leafmtxp);
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
