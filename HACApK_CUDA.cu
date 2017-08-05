// -*- c++ -*-
#include	<stdio.h>
#include	<stdlib.h>
#include	<time.h>
#include	"omp.h"
#include	"mpi.h"
#include	"HACApK_MAGMA.h"
//#include        "magma_dlapack.h"

__global__ void cuda_matvec_a1
(int kt, int ndt, int nstrtt, double *d_zbut, double *d_a1, double *d_zu)
{
  int il, it, itt, itl;
  for(il=0; il<kt; il++){
    for(it=0; it<ndt; it++){
      itt=it+nstrtt-1;
      itl=it+il*ndt; 
      d_zbut[il] += d_a1[itl]*d_zu[itt];
    }
  }
}
template <int THREADS_PER_BLOCK>
__global__ void cuda_matvec_a1_2
(int kt, int ndt, int nstrtt, double *d_zbut, double *d_a1, double *d_zu)
{
  int il, it, itt, itl;
  int gid = blockIdx.x;
  int glen = gridDim.x;
  int tid = threadIdx.x;
  int tlen = blockDim.x;
  double tmp=0.0;
  __shared__ double smTmp[THREADS_PER_BLOCK*2];
  for(il=gid; il<kt; il+=glen){
    tmp = 0.0;
    for(it=tid; it<ndt; it+=tlen){
      itt=it+nstrtt-1;
      itl=it+il*ndt; 
      tmp += d_a1[itl]*d_zu[itt];
    }
    smTmp[tid] = tmp;
    smTmp[tid+tlen] = 0.0;
    __syncthreads();
    if(THREADS_PER_BLOCK > 512){    if(tid<512)smTmp[tid] = tmp = tmp + smTmp[tid+512];    __syncthreads();  }
    if(THREADS_PER_BLOCK > 256){    if(tid<256)smTmp[tid] = tmp = tmp + smTmp[tid+256];    __syncthreads();  }
    if(THREADS_PER_BLOCK > 128){    if(tid<128)smTmp[tid] = tmp = tmp + smTmp[tid+128];    __syncthreads();  }
    if(THREADS_PER_BLOCK >  64){    if(tid< 64)smTmp[tid] = tmp = tmp + smTmp[tid+ 64];    __syncthreads();  }
    if(THREADS_PER_BLOCK >  32){    if(tid< 32)smTmp[tid] = tmp = tmp + smTmp[tid+ 32];    __syncthreads();  }
    if(tid<32){
      for (int offset = warpSize/2; offset > 0; offset /= 2){
	tmp += __shfl_down(tmp, offset);
      }
    }
    if(tid==0)d_zbut[il] += tmp;
  }
}
__global__ void cuda_matvec_a2
(int kt, int ndl, int nstrtl, double *d_zaut, double *d_a2tmp, double *d_zbut)
{
  int il, it, ill, itl;
  for(il=0; il<kt; il++){
    for(it=0; it<ndl; it++){
      ill=it+nstrtl-1;
      itl=it+il*ndl; 
      d_zaut[ill] += d_a2tmp[itl]*d_zbut[il];
    }
  }
}
__global__ void cuda_matvec_a2_2a
(int kt, int ndl, int nstrtl, double *d_zaut, double *d_a2tmp, double *d_zbut)
{
  int il, it, ill, itl;
  int gid = blockIdx.x;
  int glen = gridDim.x;
  int tid = threadIdx.x;
  int tlen = blockDim.x;
  for(il=gid; il<kt; il+=glen){
    for(it=tid; it<ndl; it+=tlen){
      ill=it+nstrtl-1;
      itl=it+il*ndl;
      atomicAdd(&d_zaut[ill], d_a2tmp[itl]*d_zbut[il]);
    }
  }
}
template <int THREADS_PER_BLOCK>
__global__ void cuda_matvec_a2_2b
(int kt, int ndl, int nstrtl, double *d_zaut, double *d_a2tmp, double *d_zbut)
{
  int il, it, ill, itl;
  int gid = blockIdx.x;
  int glen = gridDim.x;
  int tid = threadIdx.x;
  int tlen = blockDim.x;
  double tmp=0.0;
  __shared__ double smTmp[THREADS_PER_BLOCK*2];
  for(it=gid; it<ndl; it+=glen){
    tmp = 0.0;
    ill=it+nstrtl-1;
    for(il=tid; il<kt; il+=tlen){
      itl=it+il*ndl; 
      tmp += d_a2tmp[itl]*d_zbut[il];
    }
    smTmp[tid] = tmp;
    smTmp[tid+tlen] = 0.0;
    __syncthreads();
    if(THREADS_PER_BLOCK > 512){    if(tid<512)smTmp[tid] = tmp = tmp + smTmp[tid+512];    __syncthreads();  }
    if(THREADS_PER_BLOCK > 256){    if(tid<256)smTmp[tid] = tmp = tmp + smTmp[tid+256];    __syncthreads();  }
    if(THREADS_PER_BLOCK > 128){    if(tid<128)smTmp[tid] = tmp = tmp + smTmp[tid+128];    __syncthreads();  }
    if(THREADS_PER_BLOCK >  64){    if(tid< 64)smTmp[tid] = tmp = tmp + smTmp[tid+ 64];    __syncthreads();  }
    if(THREADS_PER_BLOCK >  32){    if(tid< 32)smTmp[tid] = tmp = tmp + smTmp[tid+ 32];    __syncthreads();  }
    if(tid<32){
      for (int offset = warpSize/2; offset > 0; offset /= 2){
	tmp += __shfl_down(tmp, offset);
      }
    }
    if(tid==0)d_zaut[ill] += tmp;
  }
}

__global__ void cuda_matvec_s
(int ndl, int ndt, int nstrtl, int nstrtt, double *d_zaut, double *d_a1, double *d_zu)
{
  int il, it, ill, itt, itl;
  for(il=0; il<ndl; il++){
    ill=il+nstrtl-1; 
    for(it=0; it<ndt; it++){
      itt=it+nstrtt-1; 
      itl=it+il*ndt;
      d_zaut[ill] += d_a1[itl]*d_zu[itt];
    }
  }
}
template <int THREADS_PER_BLOCK>
__global__ void cuda_matvec_s_2
(int ndl, int ndt, int nstrtl, int nstrtt, double *d_zaut, double *d_a1, double *d_zu)
{
  int il, it, ill, itt, itl;
  int gid = blockIdx.x;
  int glen = gridDim.x;
  int tid = threadIdx.x;
  int tlen = blockDim.x;
  double tmp=0.0;
  __shared__ double smTmp[THREADS_PER_BLOCK*2];
  for(il=gid; il<ndl; il+=glen){
    tmp = 0.0;
    ill=il+nstrtl-1;
    for(it=tid; it<ndt; it+=tlen){
      itt=it+nstrtt-1; 
      itl=it+il*ndt;
      tmp += d_a1[itl]*d_zu[itt];
    }
    smTmp[tid] = tmp;
    smTmp[tid+tlen] = 0.0;
    __syncthreads();
    if(THREADS_PER_BLOCK > 512){    if(tid<512)smTmp[tid] = tmp = tmp + smTmp[tid+512];    __syncthreads();  }
    if(THREADS_PER_BLOCK > 256){    if(tid<256)smTmp[tid] = tmp = tmp + smTmp[tid+256];    __syncthreads();  }
    if(THREADS_PER_BLOCK > 128){    if(tid<128)smTmp[tid] = tmp = tmp + smTmp[tid+128];    __syncthreads();  }
    if(THREADS_PER_BLOCK >  64){    if(tid< 64)smTmp[tid] = tmp = tmp + smTmp[tid+ 64];    __syncthreads();  }
    if(THREADS_PER_BLOCK >  32){    if(tid< 32)smTmp[tid] = tmp = tmp + smTmp[tid+ 32];    __syncthreads();  }
    if(tid<32){
      for (int offset = warpSize/2; offset > 0; offset /= 2){
	tmp += __shfl_down(tmp, offset);
      }
    }
    if(tid==0)d_zaut[ill] += tmp;
  }
}

void  c_hacapk_adot_body_lfmtx_cuda_calc
(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu,
 double *time_batch, double *time_set, double *time_copy, int nd) {
  register int ip,il,it;
  int nlf,ndl,ndt,nstrtl,nstrtt,kt,itl,itt,ill;
  int st_lf_stride = st_leafmtxp->st_lf_stride;
  size_t a1size;
  int ith, nths, nthe;
  double *zaut, *zbut;
  int ls, le;
  int i;


  double *d_zaut, *d_zbut;
  double *d_zau, *d_zu;
  double *d_a1, *d_a2tmp;
  cudaMalloc(&d_zaut, sizeof(double)*nd);
  cudaMalloc(&d_zbut, sizeof(double)*st_leafmtxp->ktmax);
  cudaMalloc(&d_zau, sizeof(double)*nd);
  cudaMalloc(&d_zu, sizeof(double)*nd);

  for(i=0;i<nd;i++)zau[i]=0.0;
  cudaMemcpy(d_zau, zau, sizeof(double)*nd, cudaMemcpyHostToDevice);
  cudaMemcpy(d_zu, zu, sizeof(double)*nd, cudaMemcpyHostToDevice);

  nlf=st_leafmtxp->nlf;
  //fprintf(stderr,"nlf=%d \n",nlf);

  zaut = (double*)malloc(sizeof(double)*nd);
  for(il=0;il<nd;il++)zaut[il]=0.0;
  cudaMemcpy(d_zaut, zaut, sizeof(double)*nd, cudaMemcpyHostToDevice);
  //printf("st_leafmtxp->ktmax = %d\n",st_leafmtxp->ktmax);
  zbut = (double*)malloc(sizeof(double)*st_leafmtxp->ktmax);
  ls = nd;
  le = 1;
  for(ip=0; ip<nlf; ip++){
    //ip=0;{
    /**/
    stc_HACApK_leafmtx *sttmp;
    sttmp = (stc_HACApK_leafmtx *)((size_t)((void *)(st_leafmtxp->st_lf)) + st_lf_stride * ip);
    //fprintf(stderr, "%d: %p\n", ip, sttmp);
    /**/

    ndl   =sttmp->ndl; 
    ndt   =sttmp->ndt;
    nstrtl=sttmp->nstrtl; 
    nstrtt=sttmp->nstrtt;
    //fprintf(stderr,"ip=%d, ndl=%d, ndt=%d, nstrtl=%d, nstrtt=%d \n",ip,ndl,ndt,nstrtl,nstrtt);
    cudaMalloc(&d_a2tmp, sizeof(double)*ndl*ndt);
    if(nstrtl<ls)ls=nstrtl;
    if(nstrtl+ndl-1>le)le=nstrtl+ndl-1;
    //printf("DBG: ltmtx=%d\n",sttmp->ltmtx);
    if(sttmp->ltmtx==1){
      /**/
      double *a2tmp = (double *)((size_t)((void*)(sttmp->a1))+sttmp->a1size);
      /**/
      kt=sttmp->kt;
      for(il=0;il<kt;il++)zbut[il]=0.0;
      cudaMemcpy(d_zbut, zbut, sizeof(double)*kt, cudaMemcpyHostToDevice);
      cudaMemcpy(d_a2tmp, a2tmp, sizeof(double)*ndl*ndt, cudaMemcpyHostToDevice);
      cudaMalloc(&d_a1, sizeof(double)*ndt*kt);
      cudaMemcpy(d_a1, sttmp->a1, sizeof(double)*ndt*kt, cudaMemcpyHostToDevice);
      //cuda_matvec_a1<<<1,1>>>(kt,ndt,nstrtt,d_zbut,d_a1,d_zu);
      cuda_matvec_a1_2<128><<<112,128>>>(kt,ndt,nstrtt,d_zbut,d_a1,d_zu);
      /*
	for(il=0; il<kt; il++){
	  for(it=0; it<ndt; it++){
	    itt=it+nstrtt-1;
	    itl=it+il*ndt; 
	    zbut[il] += sttmp->a1[itl]*zu[itt];
	  }
	}
      */
      //cuda_matvec_a2<<<1,1>>>(kt,ndl,nstrtl,d_zaut,d_a2tmp,d_zbut);
      //cuda_matvec_a2_2a<<<112,128>>>(kt,ndl,nstrtl,d_zaut,d_a2tmp,d_zbut);
      cuda_matvec_a2_2b<128><<<112,128>>>(kt,ndl,nstrtl,d_zaut,d_a2tmp,d_zbut);
      /*
	for(il=0; il<kt; il++){
	  for(it=0; it<ndl; it++){
	    ill=it+nstrtl-1;
	    itl=it+il*ndl; 
	    zaut[ill] += a2tmp[itl]*zbut[il];
	  }
	}
      */
    } else if(sttmp->ltmtx==2){
      cudaMalloc(&d_a1, sizeof(double)*ndt*ndl);
      cudaMemcpy(d_a1, sttmp->a1, sizeof(double)*ndt*ndl, cudaMemcpyHostToDevice);
      //cuda_matvec_s<<<1,1>>>(ndl,ndt,nstrtl,nstrtt,d_zaut,d_a1,d_zu);
      cuda_matvec_s_2<128><<<112,128>>>(ndl,ndt,nstrtl,nstrtt,d_zaut,d_a1,d_zu);
      /*
	for(il=0; il<ndl; il++){
	  ill=il+nstrtl-1; 
	  for(it=0; it<ndt; it++){
	    itt=it+nstrtt-1; 
	    itl=it+il*ndt;
	    zaut[ill] += sttmp->a1[itl]*zu[itt];
	  }
	}
      */
    }
    cudaFree(d_a1);
    cudaFree(d_a2tmp);
  }
  cudaMemcpy(zaut, d_zaut, sizeof(double)*nd, cudaMemcpyDeviceToHost);
  for(il=ls-1;il<=le-1;il++){
    zau[il] += zaut[il];
  }
  /*
    for(il=ls-1;il<=le-1;il++){
    #pragma omp atomic
    zau[il] += zaut[il];
    }
  */
  free(zaut); free(zbut);
  cudaFree(d_zaut); cudaFree(d_zbut); cudaFree(d_zau); cudaFree(d_zu);
}

void c_hacapk_adot_cax_lfmtx_cuda_comm
(double *zau, stc_HACApK_lcontrol *st_ctl,
 double *wws, double *wwr, int *isct, int *irct, int nd, double *time_mpi) {
  int ione = 1;
  double one = 1.0;

  double tic;
  int *lpmd = (int*)((size_t)((void*)st_ctl->param) + st_ctl->lpmd_offset); 
  int mpinr = lpmd[2]; 
  int nrank = lpmd[1]; 
  int i;
   
  if (nrank > 1) {
    int *lsp = (int*)((size_t)((void*)st_ctl->param) + st_ctl->lsp_offset);
    int *lnp = (int*)((size_t)((void*)st_ctl->param) + st_ctl->lnp_offset);
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

extern "C"
void c_hacapk_bicgstab_cax_lfmtx_cuda_
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
  int *lpmd = (int*)((size_t)((void*)st_ctl->param) + st_ctl->lpmd_offset);
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
  //for(i=0;i<(*nd);i++)zshdw[i]=0.0;
  c_hacapk_adot_body_lfmtx_cuda_calc(zshdw,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy,*nd);
  /*
  {
    FILE *F;
    F=fopen("cuda1.dat","w");
    for(i=0;i<(*nd);i++){
      fprintf(F,"%e\n",zshdw[i]);
    }
  }
  */
  time_spmv += (MPI_Wtime()-tic);
  c_hacapk_adot_cax_lfmtx_cuda_comm(zshdw, st_ctl, wws, wwr, isct, irct, *nd, &time_mpi);
  //
  /*
  {
    FILE *F;
    F=fopen("cuda2.dat","w");
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
  printf("zrnorm:%e",zrnorm);
  //return;
  if (mpinr == 0) {
	printf( "\n ** BICG (c version, CUDA) **\n" );
	printf( "\nOriginal relative residual norm = %.2e/%.2e = %.2e\n",zrnorm,bnorm,zrnorm/bnorm );
	printf( "c_HACApK_bicgstab_cax_lfmtx_cuda start\n" );
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
	  F=fopen("cuda-zp.dat","w");
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
	c_hacapk_adot_body_lfmtx_cuda_calc(zakp,st_leafmtxp,zkp,wws, &time_batch,&time_set,&time_copy,*nd);
	time_spmv += (MPI_Wtime()-tic);
	/*
	{
	  FILE *F;
	  F=fopen("cuda-zakp.dat","w");
	  for(i=0;i<(*nd);i++)fprintf(F,"%e\n", zakp[i]);
	  fclose(F);
	}
	*/
	c_hacapk_adot_cax_lfmtx_cuda_comm(zakp,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
	//
	znorm = 0.0;
#pragma omp parallel for reduction(+:znorm)
	for(i=0;i<(*nd);i++)znorm += zshdw[i]*zr[i];
	zden = 0.0;
#pragma omp parallel for reduction(+:zden)
	for(i=0;i<(*nd);i++)zden += zshdw[i]*zakp[i];
	/*
	{
	  FILE *F;
	  F=fopen("cuda.dat","w");
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
	c_hacapk_adot_body_lfmtx_cuda_calc(zakt,st_leafmtxp,zkt,wws, &time_batch,&time_set,&time_copy,*nd);
	time_spmv += (MPI_Wtime()-tic);
	c_hacapk_adot_cax_lfmtx_cuda_comm(zakt,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
	//
	znorm = 0.0;
#pragma omp parallel for reduction(+:znorm)
	for(i=0;i<(*nd);i++)znorm += zakt[i]*zt[i];
	zden = 0.0;
#pragma omp parallel for reduction(+:zden)
	for(i=0;i<(*nd);i++)zden += zakt[i]*zakt[i];
	zeta = znorm/zden;
	/*
	{
	  FILE *F;
	  F=fopen("cuda.dat","a");
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
			printf( "C-CUDA  %d  BiCG        = %.5e\n", i, time );
			printf( "C-CUDA  %d  time_mpi   = %.5e\n", i, time_mpi );
			printf( "C-CUDA  %d  time_matvec  = %.5e\n", i, time_spmv );
			printf( "C-CUDA  %d  >time_copy  = %.5e\n", i, time_copy );
			printf( "C-CUDA  %d  >time_set   = %.5e\n", i, time_set );
			printf( "C-CUDA  %d  >time_batch = %.5e\n", i, time_batch );
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


void  c_hacapk_adot_body_lfmtx_warp_calc
(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu,
 double *time_batch, double *time_set, double *time_copy, int nd) {
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
  for(ip=0; ip<nlf; ip++){
	/**/
	stc_HACApK_leafmtx *sttmp;
	sttmp = (stc_HACApK_leafmtx *)((size_t)((void *)(st_leafmtxp->st_lf)) + st_lf_stride * ip);
	//fprintf(stderr, "%d: %p\n", ip, sttmp);
	/**/

	ndl   =sttmp->ndl; 
	ndt   =sttmp->ndt;
	nstrtl=sttmp->nstrtl; 
	nstrtt=sttmp->nstrtt;
	//fprintf(stderr,"ip=%d, ndl=%d, ndt=%d, nstrtl=%d, nstrtt=%d \n",ip,ndl,ndt,nstrtl,nstrtt);
	if(nstrtl<ls)ls=nstrtl;
	if(nstrtl+ndl-1>le)le=nstrtl+ndl-1;
	if(sttmp->ltmtx==1){
	  /**/
	  double *a2tmp = (double *)((size_t)((void*)(sttmp->a1))+sttmp->a1size);
	  /**/
	  kt=sttmp->kt;
	  for(il=0;il<kt;il++)zbut[il]=0.0;
	  for(il=0; il<kt; il++){
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

void c_hacapk_adot_cax_lfmtx_warp_comm
(double *zau, stc_HACApK_lcontrol *st_ctl,
 double *wws, double *wwr, int *isct, int *irct, int nd, double *time_mpi) {
  int ione = 1;
  double one = 1.0;

  double tic;
  int *lpmd = (int*)((size_t)((void*)st_ctl->param) + st_ctl->lpmd_offset); 
  int mpinr = lpmd[2]; 
  int nrank = lpmd[1]; 
  int i;
   
  if (nrank > 1) {
	int *lsp = (int*)((size_t)((void*)st_ctl->param) + st_ctl->lsp_offset);
	int *lnp = (int*)((size_t)((void*)st_ctl->param) + st_ctl->lnp_offset);
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

extern "C"
void c_hacapk_bicgstab_cax_lfmtx_warp_
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
  int *lpmd = (int*)((size_t)((void*)st_ctl->param) + st_ctl->lpmd_offset);
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
#pragma omp parallel for
  for(i=0;i<(*nd);i++)zr[i]=b[i];
  //  .. MATVEC ..
  tic = MPI_Wtime();
  for(i=0;i<(*nd);i++)zshdw[i]=0.0;
  c_hacapk_adot_body_lfmtx_cuda_calc(zshdw,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy,*nd);
  time_spmv += (MPI_Wtime()-tic);
  c_hacapk_adot_cax_lfmtx_cuda_comm(zshdw, st_ctl, wws, wwr, isct, irct, *nd, &time_mpi);
  //
#pragma omp parallel for
  for(i=0;i<(*nd);i++){
	zr[i]+=mone*zshdw[i];
	zshdw[i]=zr[i];
  }
  zrnorm = 0.0;
#pragma omp parallel for reduction(+:zrnorm)
  for(i=0;i<(*nd);i++)zrnorm += zr[i]*zr[i];
  zrnorm = sqrt(zrnorm);
  if (mpinr == 0) {
	printf( "\n ** BICG (c version, WARP) **\n" );
	printf( "\nOriginal relative residual norm = %.2e/%.2e = %.2e\n",zrnorm,bnorm,zrnorm/bnorm );
	printf( "c_HACApK_bicgstab_cax_lfmtx_warp start\n" );
  }
  for ( step=1; step<=mstep; step++ ) {
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
	// zkp(:nd) = zp(:nd)
#pragma omp parallel for
	for(i=0;i<(*nd);i++)zkp[i]=zp[i];
	//  .. MATVEC ..
	//for(i=0;i<(*nd);i++)zakp[i]=0.0;
	tic = MPI_Wtime();
	c_hacapk_adot_body_lfmtx_cuda_calc(zakp,st_leafmtxp,zkp,wws, &time_batch,&time_set,&time_copy,*nd);
	time_spmv += (MPI_Wtime()-tic);
	c_hacapk_adot_cax_lfmtx_cuda_comm(zakp,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
	//
	znorm = 0.0;
#pragma omp parallel for reduction(+:znorm)
	for(i=0;i<(*nd);i++)znorm += zshdw[i]*zr[i];
	zden = 0.0;
#pragma omp parallel for reduction(+:zden)
	for(i=0;i<(*nd);i++)zden += zshdw[i]*zakp[i];
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
	c_hacapk_adot_body_lfmtx_cuda_calc(zakt,st_leafmtxp,zkt,wws, &time_batch,&time_set,&time_copy,*nd);
	time_spmv += (MPI_Wtime()-tic);
	c_hacapk_adot_cax_lfmtx_cuda_comm(zakt,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
	//
	znorm = 0.0;
#pragma omp parallel for reduction(+:znorm)
	for(i=0;i<(*nd);i++)znorm += zakt[i]*zt[i];
	zden = 0.0;
#pragma omp parallel for reduction(+:zden)
	for(i=0;i<(*nd);i++)zden += zakt[i]*zakt[i];
	zeta = znorm/zden;
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
			printf( "C-WARP  %d  BiCG        = %.5e\n", i, time );
			printf( "C-WARP  %d  time_mpi   = %.5e\n", i, time_mpi );
			printf( "C-WARP  %d  time_matvec  = %.5e\n", i, time_spmv );
			printf( "C-WARP  %d  >time_copy  = %.5e\n", i, time_copy );
			printf( "C-WARP  %d  >time_set   = %.5e\n", i, time_set );
			printf( "C-WARP  %d  >time_batch = %.5e\n", i, time_batch );
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
