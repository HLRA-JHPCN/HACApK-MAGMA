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

#define PINNED_BUFFER
#define PORTABLE_BUFFER
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

#define WARMUP_MPI
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
           #if 1
           int nctp = (ncsp-ic+nrank+1)%nrank; // where it came from
           if (irct[0] != lnp[nctp]) printf( " irct0[%d,%d]: %d vs. %d\n",mpinr,ic,irct[0],lnp[nctp] );
           if (irct[1] != lsp[nctp]) printf( " irct1[%d,%d]: %d vs. %d\n",mpinr,ic,irct[1],lsp[nctp] );
           #endif
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

void  c_hacapk_adot_body_lfmtx_seq_calc
(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu,
 double *time_batch, double *time_set, double *time_copy) {
  register int ip,il,it;
  int nlf,ndl,ndt,nstrtl,nstrtt,kt,itl,itt,ill;
  int st_lf_stride = st_leafmtxp->st_lf_stride;
  size_t a1size;
 
  nlf=st_leafmtxp->nlf;
  //fprintf(stderr,"nlf=%d \n",nlf);

  for(ip=0; ip<nlf; ip++){
    //ip=0;{
    /**/
    stc_HACApK_leafmtx *sttmp;
    sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
    //fprintf(stderr, "%d: %p\n", ip, sttmp);
    /**/

    ndl   =sttmp->ndl; 
    ndt   =sttmp->ndt;
    nstrtl=sttmp->nstrtl; 
    nstrtt=sttmp->nstrtt;
    //fprintf(stderr,"ip=%d, ndl=%d, ndt=%d, nstrtl=%d, nstrtt=%d \n",ip,ndl,ndt,nstrtl,nstrtt);
    if(sttmp->ltmtx==1){
      /**/
      double *a2tmp = (double *)((void*)(sttmp->a1)+sttmp->a1size);
      /**/
      kt=sttmp->kt;

      for(il=0; il<kt; il++){
		zbu[il]=0.0;
		for(it=0; it<ndt; it++){
		  itt=it+nstrtt-1;
		  itl=it+il*ndt; 
		  zbu[il] += sttmp->a1[itl]*zu[itt];
		}
      }
      for(il=0; il<kt; il++){
		for(it=0; it<ndl; it++){
		  ill=it+nstrtl-1;
		  itl=it+il*ndl; 
		  zau[ill] += a2tmp[itl]*zbu[il];
		}
      }
    } else if(sttmp->ltmtx==2){
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

void c_hacapk_adot_cax_lfmtx_seq_comm(double *zau, stc_HACApK_lcontrol *st_ctl,
                                  double *wws, double *wwr, int *isct, int *irct, int nd, double *time_mpi) {
    int ione = 1;
    double one = 1.0;

    double tic;
    int *lpmd = (int*)((void*)st_ctl->param + st_ctl->lpmd_offset); 
    int mpinr = lpmd[2]; 
    int nrank = lpmd[1]; 
    int i;
   
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
        //dlacpy_( "F", &lnp[mpinr], &ione, &zau[lsp[mpinr]-1], &lnp[mpinr], wws, &lnp[mpinr] );
	for(i=0;i<lnp[mpinr];i++)wws[i]=zau[lsp[mpinr]-1+i];
        for (ic=1; ic<nrank; ic++) {
           MPI_Status stat;
           tic = MPI_Wtime();
	   // read offset/size from structure
           int nctp = (ncsp-ic+nrank+1)%nrank; // where it came from
           irct[0] = lnp[nctp];
           irct[1] = lsp[nctp];

           MPI_Status stats[2];
           MPI_Request reqs[2];
           if (MPI_SUCCESS != MPI_Isend(wws, isct[0], MPI_DOUBLE, ncdp, nrank+ic, MPI_COMM_WORLD, &reqs[0])) 
               printf( "MPI_Isend failed\n" );
           if (MPI_SUCCESS != MPI_Irecv(wwr, irct[0], MPI_DOUBLE, ncsp, nrank+ic, MPI_COMM_WORLD, &reqs[1]))
               printf( "MPI_Irecv failed\n" );
           if (MPI_SUCCESS != MPI_Waitall(2, reqs, stats))
               printf( "MPI_Waitall failed\n" );

           *time_mpi += (MPI_Wtime()-tic);
           //blasf77_daxpy( &irct[0], &one, wwr, &ione, &zau[irct[1]-1], &ione );
	   for(i=0;i<irct[0];i++)zau[irct[1]-1+i]+=wwr[i];

           //dlacpy_( "F", &irct[0], &ione, wwr, &irct[0], wws, &irct[0] );
	   for(i=0;i<irct[0];i++)wws[i]=wwr[i];
           isct[0] = irct[0];
           isct[1] = irct[1];
        }
    }
}

void c_hacapk_bicgstab_cax_lfmtx_seq_
(stc_HACApK_leafmtxp *st_leafmtxp, stc_HACApK_lcontrol *st_ctl,
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
  int i;
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
  // copy matrix to GPU
  //c_hacapk_adot_body_lfcpy_batch_sorted_(nd, st_leafmtxp);

  time_spmv = 0.0;
  time_mpi = 0.0;
  time_batch = 0.0;
  time_set = 0.0;
  time_copy = 0.0;
  MPI_Barrier( icomm );
  st_measure_time = MPI_Wtime();
  // init
  alpha = 0.0; beta = 0.0; zeta = 0.0;
  zz = 0.0; for(i=0;i<(*nd);i++)zz += b[i]*b[i];
  bnorm=sqrt(zz);
  printf("bnorm:%e\n",bnorm);
  for(i=0;i<(*nd);i++)zr[i]=b[i];
  //  .. MATVEC ..
  tic = MPI_Wtime();
  for(i=0;i<(*nd);i++)zshdw[i]=0.0;
  c_hacapk_adot_body_lfmtx_seq_calc(zshdw,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy);
  time_spmv += (MPI_Wtime()-tic);
  c_hacapk_adot_cax_lfmtx_seq_comm(zshdw, st_ctl, wws, wwr, isct, irct, *nd, &time_mpi);
  //
  for(i=0;i<(*nd);i++)zr[i]+=mone*zshdw[i];
  for(i=0;i<(*nd);i++)zshdw[i]=zr[i];
  zrnorm = 0.0; for(i=0;i<(*nd);i++)zrnorm += zr[i]*zr[i];
  zrnorm = sqrt(zrnorm);
  printf("zrnorm:%e",zrnorm);
  if (mpinr == 0) {
    printf( "\n ** BICG (c version, seq) **\n" );
    printf( "\nOriginal relative residual norm = %.2e/%.2e = %.2e\n",zrnorm,bnorm,zrnorm/bnorm );
    printf( "c_HACApK_bicgstab_cax_lfmtx_seq start\n" );
  }
  for ( step=1; step<=mstep; step++ ) {
    //for(step=1; step<=1; step++){
	if (zrnorm/bnorm < eps) break;
	// zp(:nd) = zr(:nd) + beta*(zp(:nd) - zeta*zakp(:nd))
	if (beta == zero) {
	  for(i=0;i<(*nd);i++)zp[i]=zr[i];
	} else {
	  for(i=0;i<(*nd);i++){
	    zp[i] = zr[i] + beta * (zp[i] + zeta*zakp[i]);
	  }
	}
	/*
	{
	  FILE *F;
	  F=fopen("seq-zp.dat","w");
	  for(i=0;i<(*nd);i++)fprintf(F,"%e\n", zp[i]);
	  fclose(F);
	}
	*/
	// zkp(:nd) = zp(:nd)
	for(i=0;i<(*nd);i++)zkp[i]=zp[i];
	//  .. MATVEC ..
	for(i=0;i<(*nd);i++)zakp[i]=0.0;
	tic = MPI_Wtime();
	c_hacapk_adot_body_lfmtx_seq_calc(zakp,st_leafmtxp,zkp,wws, &time_batch,&time_set,&time_copy);
	time_spmv += (MPI_Wtime()-tic);
	/*
	{
	  FILE *F;
	  F=fopen("seq-zakp.dat","w");
	  for(i=0;i<(*nd);i++)fprintf(F,"%e\n", zakp[i]);
	  fclose(F);
	}
	*/
	c_hacapk_adot_cax_lfmtx_seq_comm(zakp,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
	//
	znorm = 0.0; for(i=0;i<(*nd);i++)znorm += zshdw[i]*zr[i];
	zden = 0.0; for(i=0;i<(*nd);i++)zden += zshdw[i]*zakp[i];
	/*
	{
	  FILE *F;
	  F=fopen("seq.dat","w");
	  fprintf(F,"%e %e\n",znorm, zden);
	  fclose(F);
	}
	*/
	alpha = -znorm/zden;
	znormold = znorm;
	// zt(:nd) = zr(:nd) - alpha*zakp(:nd)
	for(i=0;i<(*nd);i++)zt[i]=zr[i];
	for(i=0;i<(*nd);i++)zt[i]+=alpha*zakp[i];
	alpha = -alpha;
	// zkt(:nd) = zt(:nd)
	for(i=0;i<(*nd);i++)zkt[i]=zt[i];
	//  .. MATVEC ..
	for(i=0;i<(*nd);i++)zakt[i]=0.0;
	tic = MPI_Wtime();
	c_hacapk_adot_body_lfmtx_seq_calc(zakt,st_leafmtxp,zkt,wws, &time_batch,&time_set,&time_copy);
	time_spmv += (MPI_Wtime()-tic);
	c_hacapk_adot_cax_lfmtx_seq_comm(zakt,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
	//
	znorm = 0.0; for(i=0;i<(*nd);i++)znorm += zakt[i]*zt[i];
	zden = 0.0; for(i=0;i<(*nd);i++)zden += zakt[i]*zakt[i];
	zeta = znorm/zden;
	/*
	{
	  FILE *F;
	  F=fopen("seq.dat","a");
	  fprintf(F,"%e %e\n",znorm, zden);
	  fclose(F);
	}
	*/
	// u(:nd) = u(:nd) + alpha*zkp(:nd) + zeta*zkt(:nd)
	for(i=0;i<(*nd);i++)u[i]+=alpha*zkp[i];
	for(i=0;i<(*nd);i++)u[i]+=zeta*zkt[i];
	// zr(:nd) = zt(:nd) - zeta*zakt(:nd)
	zeta = -zeta;
	for(i=0;i<(*nd);i++)zr[i]=zt[i];
	for(i=0;i<(*nd);i++)zr[i]+=zeta*zakt[i];
	// beta = alpha/zeta * HACApK_dotp_d(nd,zshdw,zr)/znormold;
	beta = 0.0; for(i=0;i<(*nd);i++)beta += zshdw[i]*zr[i];
	beta = -alpha/zeta * beta/znormold;
	zrnorm = 0.0; for(i=0;i<(*nd);i++)zrnorm += zr[i]*zr[i];
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
	for(i=0;i<nrank;i++){
	  if(i==mpinr){
		printf( "C-SEQ  %d BiCG        = %.5e\n", i, time );
		printf( "C-SEQ  %d time_mpi   = %.5e\n", i, time_mpi );
		printf( "C-SEQ  %d time_matvec  = %.5e\n", i, time_spmv );
		printf( "C-SEQ  %d >time_copy  = %.5e\n", i, time_copy );
		printf( "C-SEQ  %d >time_set   = %.5e\n", i, time_set );
		printf( "C-SEQ  %d >time_batch = %.5e\n", i, time_batch );
	  }
	  MPI_Barrier( icomm );
	}
  }
  // delete matrix
  //c_hacapk_adot_body_lfdel_batch_(st_leafmtxp);

  // free cpu memory
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

void  c_hacapk_adot_body_lfcpy_magma(int *nd, stc_HACApK_leafmtxp *st_leafmtxp) {
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
}

void  c_hacapk_adot_body_lfmtx_magma_calc
(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu,
 double *time_batch, double *time_set, double *time_copy, int nd) {
    // constants
    int ione = 1;
    double one = 1.0;
    double zero = 0.0;

    int ip;
    int nlf, ndl, ndt, nstrtl, nstrtt, kt;
    int st_lf_stride = st_leafmtxp->st_lf_stride;

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
        nstrtl = sttmp->nstrtl; // i: index of first row (base-1)
        nstrtt = sttmp->nstrtt; // j: index of first column (base-1)
        int stream_id = ip%num_streams;
        if (sttmp->ltmtx == 1) { // compressed
            /**/
            double *a2tmp = (double *)((void*)(sttmp->a1)+sttmp->a1size);
            /**/
            kt=sttmp->kt; // rank
            // zbu := V'*zu
            magmablas_dgemv(MagmaTrans, ndt, kt, 
                            one,  st_leafmtxp->mtx1_gpu[ip], ndt, 
                                 &(st_leafmtxp->zu_gpu[nstrtt-1]), ione,
                            zero, st_leafmtxp->zbu_gpu[stream_id], ione,
                            queue[stream_id] );

            // zau :+= U*zbu
            magmablas_dgemv(MagmaNoTrans, ndl, kt, 
                            one,   st_leafmtxp->mtx2_gpu[ip], ndl, 
                                   st_leafmtxp->zbu_gpu[stream_id], ione,
                            one, &(st_leafmtxp->zau_gpu[stream_id][nstrtl-1]), ione,
                            queue[stream_id] );
        } else if(sttmp->ltmtx == 2) { // full
            magmablas_dgemv(MagmaTrans, ndt, ndl, 
                            one,   st_leafmtxp->mtx1_gpu[ip], ndt, 
                                 &(st_leafmtxp->zu_gpu[nstrtt-1]), ione,
                            one, &(st_leafmtxp->zau_gpu[stream_id][nstrtl-1]), ione,
                            queue[stream_id] );
        }
    }
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
    if (st_leafmtxp->mpi_rank == 0) {
        printf( " time_gpu: %.2e seconds\n",MPI_Wtime()-tic );
    }
    // copy back
    magma_dgetvector( st_leafmtxp->m, st_leafmtxp->zau_gpu[0], 1, zau, 1, queue[0] );
    magma_queue_destroy( queue[0] );
}

void c_hacapk_adot_cax_lfmtx_magma_comm
(double *zau, stc_HACApK_lcontrol *st_ctl,
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
#if 1
	  int nctp = (ncsp-ic+nrank+1)%nrank; // where it came from
	  if (irct[0] != lnp[nctp]) printf( " irct0[%d,%d]: %d vs. %d\n",mpinr,ic,irct[0],lnp[nctp] );
	  if (irct[1] != lsp[nctp]) printf( " irct1[%d,%d]: %d vs. %d\n",mpinr,ic,irct[1],lsp[nctp] );
#endif
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

void c_hacapk_adot_body_lfdel_magma
(stc_HACApK_leafmtxp *st_leafmtxp) {
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

 void c_hacapk_bicgstab_cax_lfmtx_magma_
(stc_HACApK_leafmtxp *st_leafmtxp, stc_HACApK_lcontrol *st_ctl,
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
  int i;
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
  // copy matrix to GPU
  c_hacapk_adot_body_lfcpy_magma(nd, st_leafmtxp);

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
  //  .. MATVEC ..
  tic = MPI_Wtime();
  dlaset_( "F", nd, &ione, &zero, &zero, zshdw, nd );
  c_hacapk_adot_body_lfmtx_magma_calc(zshdw,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy,*nd);
  time_spmv += (MPI_Wtime()-tic);
  c_hacapk_adot_cax_lfmtx_magma_comm(zshdw, st_ctl, wws, wwr, isct, irct, *nd, &time_mpi);
  //
  daxpy_( nd, &mone, zshdw, &ione, zr, &ione );
  dlacpy_( "F", nd, &ione, zr, nd, zshdw, nd );
  zrnorm = c_hacapk_dotp_d(*nd, zr, zr); 
  zrnorm = sqrt(zrnorm);
  if (mpinr == 0) {
	printf( "\n ** BICG (c version, MAGMA) **\n" );
	printf( "\nOriginal relative residual norm = %.2e/%.2e = %.2e\n",zrnorm,bnorm,zrnorm/bnorm );
	printf( "c_hacapk_bicgstab_cax_lfmtx_magma start\n" );
  }
  for ( step=1; step<=mstep; step++ ) {
	if (zrnorm/bnorm < eps) break;
	// zp(:nd) = zr(:nd) + beta*(zp(:nd) - zeta*zakp(:nd))
	if (beta == zero) {
	  dlacpy_( "F", nd, &ione, zr, nd, zp, nd );
	} else {
	  daxpy_( nd, &zeta, zakp, &ione, zp, &ione );
	  dlascl_( "G", &ione, &ione, &one, &beta, nd, &ione, zp, nd, &info );
	  daxpy_( nd, &one, zr, &ione, zp, &ione );
	}
	// zkp(:nd) = zp(:nd)
	dlacpy_( "F", nd, &ione, zp, nd, zkp, nd );
	//  .. MATVEC ..
	dlaset_( "F", nd, &ione, &zero, &zero, zakp, nd );
	tic = MPI_Wtime();
	c_hacapk_adot_body_lfmtx_magma_calc(zakp,st_leafmtxp,zkp,wws, &time_batch,&time_set,&time_copy,*nd);
	time_spmv += (MPI_Wtime()-tic);
	c_hacapk_adot_cax_lfmtx_magma_comm(zakp,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
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
	//  .. MATVEC ..
	dlaset_( "F", nd, &ione, &zero, &zero, zakt, nd );
	tic = MPI_Wtime();
	c_hacapk_adot_body_lfmtx_magma_calc(zakt,st_leafmtxp,zkt,wws, &time_batch,&time_set,&time_copy,*nd);
	time_spmv += (MPI_Wtime()-tic);
	c_hacapk_adot_cax_lfmtx_magma_comm(zakt,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
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
	if (st_ctl->param[0] > 0 && mpinr == 0) {
	  printf( " %d: time=%.2e log10(zrnorm/bnorm)=log10(%.2e/%.2e)=%.2e\n",step,time,zrnorm,bnorm,log10(zrnorm/bnorm) );
	}
  }
  MPI_Barrier( icomm );
  en_measure_time = MPI_Wtime();
  time = en_measure_time - st_measure_time;
  if (st_ctl->param[0] > 0) {
	//printf( " End: %d, %.2e\n",mpinr,time );
	for(i=0;i<nrank;i++){
	  if (mpinr == i) {
		printf( "C-MAGMA %d   BiCG        = %.5e\n", i, time );
		printf( "C-MAGMA %d   time_mpi   = %.5e\n", i, time_mpi );
		printf( "C-MAGMA %d   time_matvec  = %.5e\n", i, time_spmv );
		printf( "C-MAGMA %d   >time_copy  = %.5e\n", i, time_copy );
		printf( "C-MAGMA %d   >time_set   = %.5e\n", i, time_set );
		printf( "C-MAGMA %d   >time_batch = %.5e\n", i, time_batch );
	  }
	  MPI_Barrier( icomm );
	}
  }
  // delete matrix
  c_hacapk_adot_body_lfdel_magma(st_leafmtxp);

  // free cpu memory
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

// C + OpenMP

void  c_hacapk_adot_body_lfmtx_hyp_calc
(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu,
 double *time_batch, double *time_set, double *time_copy, int nd) {
#pragma omp parallel
  {
	register int ip,il,it;
	int nlf,ndl,ndt,nstrtl,nstrtt,kt,itl,itt,ill;
	int st_lf_stride = st_leafmtxp->st_lf_stride;
	size_t a1size;
	int ith, nths, nthe;
	double *zaut, *zbut;
	int ls, le;
	int i;

#pragma omp for
	for(i=0;i<nd;i++)zau[i]=0.0;

	nlf=st_leafmtxp->nlf;
	//fprintf(stderr,"nlf=%d \n",nlf);

	zaut = (double*)malloc(sizeof(double)*nd);
	for(il=0;il<nd;il++)zaut[il]=0.0;
	//printf("st_leafmtxp->ktmax = %d\n",st_leafmtxp->ktmax);
	zbut = (double*)malloc(sizeof(double)*st_leafmtxp->ktmax);
	ls = nd;
	le = 1;
#pragma omp for
	for(ip=0; ip<nlf; ip++){
	  //ip=0;{
	  /**/
	  stc_HACApK_leafmtx *sttmp;
	  sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
	  //fprintf(stderr, "%d: %p\n", ip, sttmp);
	  /**/

	  ndl   =sttmp->ndl; 
	  ndt   =sttmp->ndt;
	  nstrtl=sttmp->nstrtl; 
	  nstrtt=sttmp->nstrtt;
	  //fprintf(stderr,"ip=%d, ndl=%d, ndt=%d, nstrtl=%d, nstrtt=%d \n",ip,ndl,ndt,nstrtl,nstrtt);
	  if(nstrtl<ls)ls=nstrtl;
	  if(nstrtl+ndl-1>le)le=nstrtl+ndl-1;
	  //printf("DBG: ltmtx=%d\n",sttmp->ltmtx);
	  if(sttmp->ltmtx==1){
		/**/
		double *a2tmp = (double *)((void*)(sttmp->a1)+sttmp->a1size);
		/**/
		kt=sttmp->kt;
		for(il=0;il<kt;il++)zbut[il]=0.0;
		for(il=0; il<kt; il++){
		  //zbu[il]=0.0;
		  for(it=0; it<ndt; it++){
			itt=it+nstrtt-1;
			itl=it+il*ndt; 
			zbut[il] += sttmp->a1[itl]*zu[itt];
		  }
		}
		for(il=0; il<kt; il++){
		  for(it=0; it<ndl; it++){
			ill=it+nstrtl-1;
			itl=it+il*ndl; 
			zaut[ill] += a2tmp[itl]*zbut[il];
		  }
		}
	  } else if(sttmp->ltmtx==2){
		for(il=0; il<ndl; il++){
		  ill=il+nstrtl-1; 
		  for(it=0; it<ndt; it++){
			itt=it+nstrtt-1; 
			itl=it+il*ndt;
			zaut[ill] += sttmp->a1[itl]*zu[itt];
		  }
		}
	  }
	}
	for(il=ls-1;il<=le-1;il++){
#pragma omp atomic
	  zau[il] += zaut[il];
	}
	free(zaut); free(zbut);
  }
}

void c_hacapk_adot_cax_lfmtx_hyp_comm
(double *zau, stc_HACApK_lcontrol *st_ctl,
 double *wws, double *wwr, int *isct, int *irct, int nd, double *time_mpi) {
  int ione = 1;
  double one = 1.0;

  double tic;
  int *lpmd = (int*)((void*)st_ctl->param + st_ctl->lpmd_offset); 
  int mpinr = lpmd[2]; 
  int nrank = lpmd[1]; 
  int i;
   
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
	//dlacpy_( "F", &lnp[mpinr], &ione, &zau[lsp[mpinr]-1], &lnp[mpinr], wws, &lnp[mpinr] );
	for(i=0;i<lnp[mpinr];i++)wws[i]=zau[lsp[mpinr]-1+i];
	for (ic=1; ic<nrank; ic++) {
	  MPI_Status stat;
	  tic = MPI_Wtime();
	  // read offset/size from structure
	  int nctp = (ncsp-ic+nrank+1)%nrank; // where it came from
	  irct[0] = lnp[nctp];
	  irct[1] = lsp[nctp];

	  MPI_Status stats[2];
	  MPI_Request reqs[2];
	  if (MPI_SUCCESS != MPI_Isend(wws, isct[0], MPI_DOUBLE, ncdp, nrank+ic, MPI_COMM_WORLD, &reqs[0])) 
		printf( "MPI_Isend failed\n" );
	  if (MPI_SUCCESS != MPI_Irecv(wwr, irct[0], MPI_DOUBLE, ncsp, nrank+ic, MPI_COMM_WORLD, &reqs[1]))
		printf( "MPI_Irecv failed\n" );
	  if (MPI_SUCCESS != MPI_Waitall(2, reqs, stats))
		printf( "MPI_Waitall failed\n" );

	  *time_mpi += (MPI_Wtime()-tic);
	  //blasf77_daxpy( &irct[0], &one, wwr, &ione, &zau[irct[1]-1], &ione );
	  for(i=0;i<irct[0];i++)zau[irct[1]-1+i]+=wwr[i];

	  //dlacpy_( "F", &irct[0], &ione, wwr, &irct[0], wws, &irct[0] );
	  for(i=0;i<irct[0];i++)wws[i]=wwr[i];
	  isct[0] = irct[0];
	  isct[1] = irct[1];
	}
  }
}

void c_hacapk_bicgstab_cax_lfmtx_hyp_
(stc_HACApK_leafmtxp *st_leafmtxp, stc_HACApK_lcontrol *st_ctl,
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
  int i, tid;
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
  // copy matrix to GPU
  //c_hacapk_adot_body_lfcpy_batch_sorted_(nd, st_leafmtxp);

  time_spmv = 0.0;
  time_mpi = 0.0;
  time_batch = 0.0;
  time_set = 0.0;
  time_copy = 0.0;
  MPI_Barrier( icomm );
  st_measure_time = MPI_Wtime();
  // init
  alpha = 0.0; beta = 0.0; zeta = 0.0;
  zz = 0.0;
#pragma omp parallel for reduction(+:zz)
  for(i=0;i<(*nd);i++){zz += b[i]*b[i];}
  bnorm=sqrt(zz);
  printf("bnorm:%e\n",bnorm);
#pragma omp parallel for
  for(i=0;i<(*nd);i++)zr[i]=b[i];
  //  .. MATVEC ..
  tic = MPI_Wtime();
  for(i=0;i<(*nd);i++)zshdw[i]=0.0;
  c_hacapk_adot_body_lfmtx_hyp_calc(zshdw,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy,*nd);
  /*
  {
    FILE *F;
    F=fopen("omp1.dat","w");
    for(i=0;i<(*nd);i++){
      fprintf(F,"%e\n",zshdw[i]);
    }
  }
  */
  time_spmv += (MPI_Wtime()-tic);
  c_hacapk_adot_cax_lfmtx_hyp_comm(zshdw, st_ctl, wws, wwr, isct, irct, *nd, &time_mpi);
  //
  /*
  {
    FILE *F;
    F=fopen("omp2.dat","w");
    for(i=0;i<(*nd);i++){
      fprintf(F,"%e\n",zshdw[i]);
    }
  }
  */
#pragma omp parallel for
  for(i=0;i<(*nd);i++){
	zr[i]+=mone*zshdw[i];
	zshdw[i]=zr[i];
  }
  zrnorm = 0.0;
#pragma omp parallel for reduction(+:zrnorm)
  for(i=0;i<(*nd);i++)zrnorm += zr[i]*zr[i];
  zrnorm = sqrt(zrnorm);
  printf("zrnorm:%e\n",zrnorm);
  //return;
  if (mpinr == 0) {
	printf( "\n ** BICG (c version, OMP) **\n" );
	printf( "\nOriginal relative residual norm = %.2e/%.2e = %.2e\n",zrnorm,bnorm,zrnorm/bnorm );
	printf( "c_HACApK_bicgstab_cax_lfmtx_hyp start\n" );
  }
  for ( step=1; step<=mstep; step++ ) {
    //for(step=1; step<=1; step++){
	if (zrnorm/bnorm < eps) break;
	// zp(:nd) = zr(:nd) + beta*(zp(:nd) - zeta*zakp(:nd))
	if (beta == zero) {
#pragma omp parallel for
	  for(i=0;i<(*nd);i++)zp[i]=zr[i];
	} else {
#pragma omp parallel for
	  for(i=0;i<(*nd);i++){
	    zp[i] = zr[i] + beta * (zp[i] + zeta*zakp[i]);
	  }
	}
	/*
	{
	  FILE *F;
	  F=fopen("omp-zp.dat","w");
	  for(i=0;i<(*nd);i++)fprintf(F,"%e\n", zp[i]);
	  fclose(F);
	}
	*/
	// zkp(:nd) = zp(:nd)
#pragma omp parallel for
	for(i=0;i<(*nd);i++)zkp[i]=zp[i];
	//  .. MATVEC ..
	//for(i=0;i<(*nd);i++)zakp[i]=0.0;
	tic = MPI_Wtime();
	c_hacapk_adot_body_lfmtx_hyp_calc(zakp,st_leafmtxp,zkp,wws, &time_batch,&time_set,&time_copy,*nd);
	time_spmv += (MPI_Wtime()-tic);
	/*
	{
	  FILE *F;
	  F=fopen("omp-zakp.dat","w");
	  for(i=0;i<(*nd);i++)fprintf(F,"%e\n", zakp[i]);
	  fclose(F);
	}
	*/
	c_hacapk_adot_cax_lfmtx_hyp_comm(zakp,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
	//
	znorm = 0.0;
#pragma omp parallel for reduction(+:znorm)
	for(i=0;i<(*nd);i++)znorm += zshdw[i]*zr[i];
	zden = 0.0;
#pragma omp parallel for reduction(+:zden)
	for(i=0;i<(*nd);i++)zden += zshdw[i]*zakp[i];
	printf("znorm:%e\n",znorm);
	printf("zden:%e\n",zden);
	/*
	{
	  FILE *F;
	  F=fopen("omp.dat","w");
	  fprintf(F,"%e %e\n",znorm, zden);
	  fclose(F);
	}
	*/
	alpha = -znorm/zden;
	znormold = znorm;
	// zt(:nd) = zr(:nd) - alpha*zakp(:nd)
#pragma omp parallel for
	for(i=0;i<(*nd);i++)zt[i]=zr[i]+alpha*zakp[i];
	alpha = -alpha;
	// zkt(:nd) = zt(:nd)
#pragma omp parallel for
	for(i=0;i<(*nd);i++)zkt[i]=zt[i];
	//  .. MATVEC ..
	//for(i=0;i<(*nd);i++)zakt[i]=0.0;
	tic = MPI_Wtime();
	c_hacapk_adot_body_lfmtx_hyp_calc(zakt,st_leafmtxp,zkt,wws, &time_batch,&time_set,&time_copy,*nd);
	time_spmv += (MPI_Wtime()-tic);
	c_hacapk_adot_cax_lfmtx_hyp_comm(zakt,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
	//
	znorm = 0.0;
#pragma omp parallel for reduction(+:znorm)
	for(i=0;i<(*nd);i++)znorm += zakt[i]*zt[i];
	zden = 0.0;
#pragma omp parallel for reduction(+:zden)
	for(i=0;i<(*nd);i++)zden += zakt[i]*zakt[i];
	printf("znorm:%e\n",znorm);
	printf("zden:%e\n",zden);
	zeta = znorm/zden;
	printf("zeta:%e\n",zeta);
	/*
	{
	  FILE *F;
	  F=fopen("omp.dat","a");
	  fprintf(F,"%e %e\n",znorm, zden);
	  fclose(F);
	}
	*/
	// u(:nd) = u(:nd) + alpha*zkp(:nd) + zeta*zkt(:nd)
#pragma omp parallel for
	for(i=0;i<(*nd);i++){
	  u[i] += alpha*zkp[i] + zeta*zkt[i];
	}
	// zr(:nd) = zt(:nd) - zeta*zakt(:nd)
	zeta = -zeta;
#pragma omp parallel
	for(i=0;i<(*nd);i++){
	  zr[i]=zt[i] + zeta*zakt[i];
	}
	// beta = alpha/zeta * HACApK_dotp_d(nd,zshdw,zr)/znormold;
	beta = 0.0;
#pragma omp parallel for reduction(+:beta)
	for(i=0;i<(*nd);i++)beta += zshdw[i]*zr[i];
	beta = -alpha/zeta * beta/znormold;
	zrnorm = 0.0;
#pragma omp parallel for reduction(+:zrnorm)
	for(i=0;i<(*nd);i++)zrnorm += zr[i]*zr[i];
	printf("beta:%e\n",beta);
	printf("zrnorm:%e\n",zrnorm);
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
	  for(i=0;i<nrank;i++){
		if(i==mpinr){
			printf( "C-OMP  %d  BiCG        = %.5e\n", i, time );
			printf( "C-OMP  %d  time_mpi   = %.5e\n", i, time_mpi );
			printf( "C-OMP  %d  time_matvec  = %.5e\n", i, time_spmv );
			printf( "C-OMP  %d  >time_copy  = %.5e\n", i, time_copy );
			printf( "C-OMP  %d  >time_set   = %.5e\n", i, time_set );
			printf( "C-OMP  %d  >time_batch = %.5e\n", i, time_batch );
		}
		MPI_Barrier( icomm );
	  }
    }
    // delete matrix
    //c_hacapk_adot_body_lfdel_batch_(st_leafmtxp);

    // free cpu memory
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
    int i;
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
    c_hacapk_adot_cax_lfmtx_comm(zshdw, st_ctl, wws, wwr, isct, irct, *nd, &time_mpi);
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
        if (st_ctl->param[0] > 0 && mpinr == 0) {
            printf( " %d: time=%.2e log10(zrnorm/bnorm)=log10(%.2e/%.2e)=%.2e\n",step,time,zrnorm,bnorm,log10(zrnorm/bnorm) );
        }
    }
    MPI_Barrier( icomm );
    en_measure_time = MPI_Wtime();
    time = en_measure_time - st_measure_time;
    if (st_ctl->param[0] > 0) {
        //printf( " End: %d, %.2e\n",mpinr,time );
	  for(i=0;i<nrank;i++){
        if (i==mpinr) {
		  printf( "C-FLAT  %d   BiCG        = %.5e\n", i, time );
		  printf( "C-FLAT  %d   time_mpi   = %.5e\n", i, time_mpi );
		  printf( "C-FLAT  %d   time_matvec  = %.5e\n", i, time_spmv );
		  printf( "C-FLAT  %d   >time_copy  = %.5e\n", i, time_copy );
		  printf( "C-FLAT  %d   >time_set   = %.5e\n", i, time_set );
		  printf( "C-FLAT  %d   >time_batch = %.5e\n", i, time_batch );
        }
		MPI_Barrier( icomm );
	  }
    }
    // delete matrix
    c_hacapk_adot_body_lfdel_batch_(st_leafmtxp);

    // free cpu memory
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
    int i; 
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
    magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zshdw, *nd, queue );
    magma_queue_sync( queue );
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
        printf( " first time_mpi=%.2e\n",time_mpi );
        printf( "HACApK_bicgstab_lfmtx_gpu start\n" );
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
      for(i=0;i<nrank;i++){
        if (mpinr == i) {
		  printf( "C-ALL   %d  BiCG        = %.5e\n", i, time );
		  printf( "C-ALL   %d  time_mpi   = %.5e\n", i, time_mpi );
		  printf( "C-ALL   %d  time_copy  = %.5e\n", i, time_copy );
		  printf( "C-ALL   %d  time_spmv  = %.5e\n", i, time_spmv );
		  printf( "C-ALL   %d  > time_batch = %.5e\n", i, time_batch );
		  printf( "C-ALL   %d  > time_set   = %.5e\n", i, time_set );
        }
		MPI_Barrier( icomm );
      }
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
    double time_set2, time_set3; 
    int i;
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
    magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zshdw, *nd, queue[0] );
    magma_queue_sync( queue[0] );
    tic = MPI_Wtime();
    c_hacapk_adot_body_lfmtx_batch_mgpu(flag_set, zshdw,st_leafmtxp,st_ctl,u,wws, zau_cpu,wwr_cpu,
                                        &time_batch,&time_set,&time_copy,
                                        &time_set2, &time_set3,
                                        on_gpu, queue);
    magma_queue_sync( queue[0] );
    time_spmv += (MPI_Wtime()-tic);
    c_hacapk_adot_cax_lfmtx_comm_gpu(flag, zshdw, zau_cpu,
                                     st_ctl, wws_cpu, wwr_cpu, isct, irct, *nd, 
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
//if (mpinr == 0) printf( "zkp: %.2e\n",magma_ddot( *nd, zkp, ione, zkp, ione, queue[0]) ); 
        //  .. SpMV ..
        magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zakp, *nd, queue[0] );
        magma_queue_sync( queue[0] );
        tic = MPI_Wtime();
        c_hacapk_adot_body_lfmtx_batch_mgpu(flag_set, zakp,st_leafmtxp,st_ctl, zkp,wws, zau_cpu,wwr_cpu,
                                            &time_batch,&time_set,&time_copy, 
                                            &time_set2, &time_set3,
                                            on_gpu, queue);
        magma_queue_sync( queue[0] );
        time_spmv += (MPI_Wtime()-tic);
        c_hacapk_adot_cax_lfmtx_comm_gpu(flag, zakp, zau_cpu,
                                         st_ctl, wws_cpu,wwr_cpu, isct,irct,*nd, 
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
        magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zakt, *nd, queue[0] );
        magma_queue_sync( queue[0] );
        tic = MPI_Wtime();
        c_hacapk_adot_body_lfmtx_batch_mgpu(flag_set, zakt,st_leafmtxp,st_ctl, zkt,wws, zau_cpu,wwr_cpu, 
                                            &time_batch,&time_set,&time_copy,
                                            &time_set2, &time_set3,
                                            on_gpu, queue);
        magma_queue_sync( queue[0] );
        time_spmv += (MPI_Wtime()-tic);
        c_hacapk_adot_cax_lfmtx_comm_gpu(flag, zakt,zau_cpu,
                                         st_ctl, wws_cpu,wwr_cpu, isct,irct,*nd, 
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
	  for(i=0;i<nrank;i++){
        if (mpinr == i) {
		  printf( "  %d    BiCG        = %.5e\n", i, time );
		  printf( "  %d    time_mpi   = %.5e\n", i, time_mpi );
		  printf( "  %d    time_copy  = %.5e\n", i, time_copy );
		  printf( "  %d    time_spmv  = %.5e\n", i, time_spmv );
		  printf( "  %d    > time_batch = %.5e\n", i, time_batch );
		  printf( "  %d    > time_set   = %.5e\n", i, time_set );
		  printf( "  %d    + time_set2 = %.5e\n", i, time_set2 );
		  printf( "  %d    + time_set3 = %.5e\n", i, time_set3 );
        }
		MPI_Barrier( icomm );
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
    double time_spmv, time_mpi, time_batch, time_set, time_copy, tic;
    double time_set2, time_set3; 
    int i;
    MPI_Comm icomm = MPI_COMM_WORLD; //lpmd[0];
    mstep = param[82];
    eps = param[90];
    mpinr = lpmd[2]; 
    nrank = lpmd[1]; 
    MPI_Barrier( icomm );

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
    magma_dmalloc_pinned(&zau_cpu,  *nd);
    magma_dmalloc_pinned(&wws_cpu,  *nd);
    magma_dmalloc_pinned(&wwr_cpu, (*nd)*gpus_per_proc);

    // allocate GPU vectors
    double *wws, *wwr;
    if (MAGMA_SUCCESS != magma_dmalloc(&wws, *nd) ||
        MAGMA_SUCCESS != magma_dmalloc(&wwr, *nd)) {
      printf( " failed to allocate vectors (nd=%d)\n",*nd );
    }

    double **b = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **u = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **zr = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **zp = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **zt = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **zkp = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **zkt = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **zakp =(double**)malloc(gpus_per_proc * sizeof(double*));
    double **zshdw = (double**)malloc(gpus_per_proc * sizeof(double*));
    double **zakt = (double**)malloc(gpus_per_proc * sizeof(double*));
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice((gpu_id+d)%procs_per_node);
        if (MAGMA_SUCCESS != magma_dmalloc(&u[d], *nd) ||
            MAGMA_SUCCESS != magma_dmalloc(&b[d], *nd) ||
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
    #ifdef WARMUP_MPI
    c_hacapk_adot_cax_lfmtx_warmup(st_ctl, zau_cpu, wws_cpu, wwr_cpu, *nd);
    #endif
    // copy matrix to GPU
    c_hacapk_adot_body_lfcpy_batch_sorted_mgpu_(nd, st_leafmtxp, queue);
    //
    time_spmv = 0.0;
    time_mpi = 0.0;
    time_batch = 0.0;
    time_set = 0.0;
    time_set2 = 0.0;
    time_set3 = 0.0;
    time_copy = 0.0;
    MPI_Barrier( icomm );
    st_measure_time = MPI_Wtime();
    // copy the input vector to GPU
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice((gpu_id+d)%procs_per_node);
        magma_dsetvector_async( *nd, b_cpu, 1, b[d], 1, queue[d] );
        magma_dsetvector_async( *nd, u_cpu, 1, u[d], 1, queue[d] );
    }
    // init
    alpha = 0.0; beta = 0.0; zeta = 0.0;
    magma_setdevice(gpu_id);
    zz = magma_ddot(*nd, b[0], ione, b[0], ione, queue[0]); 
    bnorm=sqrt(zz);
    // .. SpMV ..
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice((gpu_id+d)%procs_per_node);
        magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zshdw[d], *nd, queue[d] );
    }
    int flag_set = 1;
    magma_queue_sync( queue[0] );
    tic = MPI_Wtime();
    c_hacapk_adot_body_lfmtx_batch_mgpu(flag_set, zshdw[0],st_leafmtxp,st_ctl,u[0],wws, zau_cpu,wwr_cpu,
                                        &time_batch,&time_set,&time_copy,
                                        &time_set2, &time_set3,
                                        on_gpu, queue);
    magma_queue_sync( queue[0] );
    time_spmv += (MPI_Wtime()-tic);
    c_hacapk_adot_cax_lfmtx_comm_gpu(flag, zshdw[0], zau_cpu,
                                     st_ctl, wws_cpu, wwr_cpu, isct, irct, *nd, 
                                     &time_copy,&time_mpi, queue[0]);
    //
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice((gpu_id+d)%procs_per_node);
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
            magma_setdevice((gpu_id+d)%procs_per_node);
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
            magma_setdevice((gpu_id+d)%procs_per_node);
            magmablas_dlacpy( MagmaFull, *nd, ione, zp[d], *nd, zkp[d], *nd, queue[d] );
        }
//magma_setdevice(gpu_id);
//if (mpinr == 0) printf( "zkp: %.2e\n",magma_ddot( *nd, zkp[0], ione, zkp[0], ione, queue[0]) ); 
        //  .. SpMV ..
        for (d=0; d<gpus_per_proc; d++) {
            magma_setdevice((gpu_id+d)%procs_per_node);
            magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zakp[d], *nd, queue[d] );
        }
        magma_setdevice(gpu_id);
        magma_queue_sync( queue[0] );
        tic = MPI_Wtime();
        c_hacapk_adot_body_lfmtx_batch_mgpu(flag_set, zakp[0],st_leafmtxp,st_ctl, zkp[0],wws, zau_cpu,wwr_cpu,
                                            &time_batch,&time_set,&time_copy, 
                                            &time_set2, &time_set3,
                                            on_gpu, queue);
        magma_queue_sync( queue[0] );
        time_spmv += (MPI_Wtime()-tic);
        c_hacapk_adot_cax_lfmtx_comm_gpu(flag, zakp[0], zau_cpu,
                                         st_ctl, wws_cpu,wwr_cpu, isct,irct,*nd, 
                                         &time_copy,&time_mpi, queue[0]);
        //
        magma_setdevice(gpu_id);
        znorm = magma_ddot( *nd, zshdw[0], ione, zr[0], ione, queue[0] ); 
        zden = magma_ddot( *nd, zshdw[0], ione, zakp[0], ione, queue[0] );
        alpha = -znorm/zden;
//if (mpinr == 0) printf( " alpha=%.2e znorm=%.2e, zden=%.2e\n",alpha,znorm,zden );
        znormold = znorm;
        // zt(:nd) = zr(:nd) - alpha*zakp(:nd)
        for (d=0; d<gpus_per_proc; d++) {
            magma_setdevice((gpu_id+d)%procs_per_node);
            magmablas_dlacpy( MagmaFull, *nd, ione, zr[d], *nd, zt[d], *nd, queue[d] );
            magma_daxpy( *nd, alpha, zakp[d], ione, zt[d], ione, queue[d] );
        }
        alpha = -alpha;
        // zkt(:nd) = zt(:nd)
        for (d=0; d<gpus_per_proc; d++) {
            magma_setdevice((gpu_id+d)%procs_per_node);
            magmablas_dlacpy( MagmaFull, *nd, ione, zt[d], *nd, zkt[d], *nd, queue[d] );
        }
        //  .. SpMV ..
        for (d=0; d<gpus_per_proc; d++) {
            magma_setdevice((gpu_id+d)%procs_per_node);
            magmablas_dlaset( MagmaFull, *nd, ione, zero, zero, zakt[d], *nd, queue[d] );
        }
        magma_setdevice(gpu_id);
        magma_queue_sync( queue[0] );
        tic = MPI_Wtime();
        c_hacapk_adot_body_lfmtx_batch_mgpu(flag_set, zakt[0],st_leafmtxp,st_ctl, zkt[0],wws, zau_cpu,wwr_cpu, 
                                            &time_batch,&time_set,&time_copy,
                                            &time_set2, &time_set3,
                                            on_gpu, queue);
        magma_queue_sync( queue[0] );
        time_spmv += (MPI_Wtime()-tic);
        c_hacapk_adot_cax_lfmtx_comm_gpu(flag, zakt[0],zau_cpu,
                                         st_ctl, wws_cpu,wwr_cpu, isct,irct,*nd, 
                                         &time_copy,&time_mpi, queue[0]);
        //
        magma_setdevice(gpu_id);
        znorm = magma_ddot( *nd, zakt[0], ione, zt[0], ione, queue[0] ); 
        zden = magma_ddot( *nd, zakt[0], ione, zakt[0], ione, queue[0] );
        zeta = znorm/zden;
//if (mpinr == 0) printf( " zeta=%.2e znorm=%.2e, zden=%.2e\n",zeta,znorm,zden );
        // u(:nd) = u(:nd) + alpha*zkp(:nd) + zeta*zkt(:nd)
        for (d=0; d<gpus_per_proc; d++) {
            magma_setdevice((gpu_id+d)%procs_per_node);
            magma_daxpy( *nd, alpha, zkp[d], ione, u[d], ione, queue[d] );
            magma_daxpy( *nd, zeta,  zkt[d], ione, u[d], ione, queue[d] );
        }
        // zr(:nd) = zt(:nd) - zeta*zakt(:nd)
        zeta = -zeta;
        for (d=0; d<gpus_per_proc; d++) {
            magma_setdevice((gpu_id+d)%procs_per_node);
            magmablas_dlacpy( MagmaFull, *nd, ione, zt[d], *nd, zr[d], *nd, queue[d] );
            magma_daxpy( *nd, zeta, zakt[d], ione, zr[d], ione, queue[d] );
        }
        // beta = alpha/zeta * HACApK_dotp_d(nd,zshdw,zr)/znormold;
        magma_setdevice(gpu_id);
        beta = magma_ddot( *nd, zshdw[0], ione, zr[0], ione, queue[0]);
        beta = -alpha/zeta * beta/znormold;
        zrnorm = magma_ddot( *nd, zr[0], ione, zr[0], ione, queue[0] );
        zrnorm = sqrt(zrnorm);
        *nstp = step;
        en_measure_time = MPI_Wtime();
        time = en_measure_time - st_measure_time;
        if (st_ctl->param[0] > 0 && mpinr == 0) {
            printf( " %d: time=%.2e log10(zrnorm/bnorm)=log10(%.2e/%.2e)=%.2e\n",step,time,zrnorm,bnorm,log10(zrnorm/bnorm) );
        }
    }
    magma_dgetvector( *nd, u[0], 1, u_cpu, 1, queue[0] );
    MPI_Barrier( icomm );
    en_measure_time = MPI_Wtime();
    time = en_measure_time - st_measure_time;
    if (st_ctl->param[0]>0) {
        //printf( " End: %d, %.2e\n",mpinr,time );
	  for(i=0;i<nrank;i++){
        if (mpinr == i) {
		  printf( " %d     BiCG       = %.5e\n", i, time );
		  printf( " %d     time_mpi  = %.5e\n", i, time_mpi );
		  printf( " %d     time_copy = %.5e\n", i, time_copy );
		  printf( " %d     time_spmv = %.5e\n", i, time_spmv );
		  printf( " %d     > time_batch = %.5e\n", i, time_batch );
		  printf( " %d     > time_set   = %.5e\n", i, time_set );
		  printf( " %d       + time_set2 = %.5e\n", i, time_set2 );
		  printf( " %d       + time_set3 = %.5e\n", i, time_set3 );
        }
		MPI_Barrier( icomm );
	  }
    }

    // free gpu memory
    for (d=0; d<gpus_per_proc; d++) {
        magma_setdevice((gpu_id+d)%procs_per_node);
        magma_queue_sync( queue[d] );
        magma_queue_destroy( queue[d] );
        magma_queue_destroy( queue[d+gpus_per_proc] );

        magma_free(u[d]);
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
    free(u);
    free(b);
    free(zt);
    free(zr);
    free(zp);
    free(zkp);
    free(zakp);
    free(zkt);
    free(zakt);
    free(zshdw);

    magma_free_pinned(zau_cpu);
    magma_free_pinned(wws_cpu);
    magma_free_pinned(wwr_cpu);

    magma_free(wws);
    magma_free(wwr);
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
    int mpinr, nrank, ierr;
    double time_spmv, time_mpi, time_batch, time_set, time_copy, tic;
    int i;
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
    MPI_Barrier( icomm );
    #ifdef WARMUP_MPI
    c_hacapk_adot_cax_lfmtx_warmup(st_ctl, zau_cpu, wws_cpu, wwr_cpu, *nd);
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
        if (st_ctl->param[0] > 0 && mpinr == 0) {
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
    if (st_ctl->param[0] > 0) {
        //printf( " End: %d, %.2e\n",mpinr,time );
	  for(i=0;i<nrank;i++){
	    if (mpinr == i) {
	      printf( "   %d    BiCG        = %.5e\n", i, time );
	      printf( "   %d    time_mpi   = %.5e\n", i, time_mpi );
	      printf( "   %d    time_copy  = %.5e\n", i, time_copy );
	      printf( "   %d    time_spmv  = %.5e\n", i, time_spmv );
	      printf( "   %d    > time_batch = %.5e\n", i, time_batch );
	      printf( "   %d    > time_set   = %.5e\n", i, time_set );
	    }
	    MPI_Barrier( icomm );
	  }
    }
    magma_queue_destroy( queue );
    // delete matrix
    c_hacapk_adot_body_lfdel_batch_(st_leafmtxp);

    // free cpu memory
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
