#include	<stdio.h>
#include	<stdlib.h>
#include	<time.h>
#include	"omp.h"
#include	"mpi.h"
#include	"HACApK_FPGA.h"


// !! BiCG in C !!
double c_hacapk_dotp_d(int nd, double *b, double *a) {
    double norm = 0.0;
    int ii;
    for (ii=0; ii<nd; ii++) norm += b[ii]*a[ii];
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
       int ncdp = (mpinr+1)%nrank;
       int ncsp = (mpinr+nrank-1)%nrank;
       isct[0] = lnp[mpinr];
       isct[1] = lsp[mpinr];

       dlacpy_( "F", &lnp[mpinr], &ione, &zau[lsp[mpinr]-1], &lnp[mpinr], wws, &lnp[mpinr] );

       int ic;
       for (ic=1; ic<nrank; ic++) {
           MPI_Status stat;

           tic = MPI_Wtime();
           MPI_Sendrecv(isct, 2, MPI_INT, ncdp, 1,
                        irct, 2, MPI_INT, ncsp, 1, icomm, &stat);
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
    zz=c_hacapk_dotp_d(*nd, b, b); 
    bnorm=sqrt(zz);
    dlaset_( "F", nd, &ione, &zero, &zero, zp, nd );
    dlaset_( "F", nd, &ione, &zero, &zero, zakp, nd );
    dlacpy_( "F", nd, &ione, b, nd, zr, nd );
    tic = MPI_Wtime();
    dlaset_( "F", nd, &ione, &zero, &zero, zshdw, nd );
    c_hacapk_adot_body_lfmtx_batch_(zshdw,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy);
    time_spmv += (MPI_Wtime()-tic);
    c_hacapk_adot_cax_lfmtx_comm(zshdw, st_leafmtxp, st_ctl, wws, wwr, isct, irct, *nd, &time_mpi);
    daxpy_( nd, &mone, zshdw, &ione, zr, &ione );
    dlacpy_( "F", nd, &ione, zr, nd, zshdw, nd );
    zrnorm = c_hacapk_dotp_d(*nd, zr, zr); 
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
        znorm = c_hacapk_dotp_d(*nd, zshdw, zr); 
        zden = c_hacapk_dotp_d(*nd, zshdw, zakp);
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
        znorm = c_hacapk_dotp_d(*nd, zakt, zt); 
        zden = c_hacapk_dotp_d( *nd, zakt, zakt);
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
        printf( " End: %d, %.2e\n",mpinr,time );
        if (mpinr==0) {
            printf( "       BiCG        = %.2e\n", time );
            printf( "        time_mpi   = %.2e\n", time_mpi );
            printf( "        time_spmv  = %.2e\n", time_spmv );
            printf( "        > time_copy  = %.2e\n", time_copy );
            printf( "        > time_set   = %.2e\n", time_set );
            printf( "        > time_batch = %.2e\n", time_batch );
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
