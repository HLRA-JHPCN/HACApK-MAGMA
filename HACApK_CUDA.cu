// -*- c++ -*-
#include	<stdio.h>
#include	<stdlib.h>
#include	<time.h>
#include	"omp.h"
#include	"mpi.h"
#include	"HACApK_MAGMA.h"
//#include        "magma_dlapack.h"
#include <cublas_v2.h>

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
    if(tid==0){
      atomicAdd(&d_zbut[il], tmp);
      //d_zbut[il] += tmp;
    }
  }
}
template <int THREADS_PER_BLOCK>
__global__ void cuda_matvec_a1_2_smo
(int kt, int ndt, int nstrtt, double *d_zbut, double *d_zu, double *d_mat, int head)
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
      tmp += d_mat[head+itl]*d_zu[itt];
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
    if(tid==0){
      //atomicAdd(&d_zbut[il], tmp);
      d_zbut[il] = tmp;
    }
  }
}
__global__ void cuda_matvec_a1_3_smo
(int kt, int ndt, int nstrtt, double *d_zbut, double *d_zu, double *d_mat, int head)
{ // 128threads = 32(=1warp) * 4pairs
  int il, it, itt, itl;
  int wid = threadIdx.x/32;
  int wlen = 4;
  int tid = threadIdx.x;
  int tlen = blockDim.x;
  int xid = threadIdx.x%32;
  int xlen = 32;
  double tmp=0.0;
  for(il=wid; il<kt; il+=wlen){
    tmp = 0.0;
    for(it=xid; it<ndt; it+=xlen){
      itt=it+nstrtt-1;
      itl=it+il*ndt; 
      tmp += d_mat[head+itl]*d_zu[itt];
    }
    //__syncthreads();
    for (int offset = warpSize/2; offset > 0; offset /= 2){
      tmp += __shfl_down(tmp, offset);
    }
    if(xid==0){
      //atomicAdd(&d_zbut[il], tmp);
      d_zbut[il] = tmp;
    }
  }
}

__global__ void cuda_matvec_a12_3_smo
(int ndt, int ndl, int kt, int nstrtt, int nstrtl, double *d_zaut, double *d_mat, double *d_zu, int a1, int a2)
{ // 128threads = 32(=1warp) * 4pairs
  int il, it, itt, itl, ill;
  int wid = threadIdx.x/32;
  int wlen = 4;
  int tid = threadIdx.x;
  int tlen = blockDim.x;
  int xid = threadIdx.x%32;
  int xlen = 32;
  double tmp=0.0;
  extern __shared__ double tmpbut[];
  for(il=wid; il<kt; il+=wlen){
    tmp = 0.0;
    for(it=xid; it<ndt; it+=xlen){
      itt=it+nstrtt-1;
      itl=it+il*ndt; 
      tmp += d_mat[a1+itl]*d_zu[itt];
    }
    for (int offset = warpSize/2; offset > 0; offset /= 2){
      tmp += __shfl_down(tmp, offset);
    }
    if(xid==0)tmpbut[il] = tmp;
  }
  __syncthreads();
  for(it=wid; it<ndl; it+=wlen){
    tmp = 0.0;
    ill=it+nstrtl-1;
    for(il=xid; il<kt; il+=xlen){
      itl=it+il*ndl; 
      tmp += d_mat[a2+itl]*tmpbut[il];
    }
    for (int offset = warpSize/2; offset > 0; offset /= 2){
      tmp += __shfl_down(tmp, offset);
    }
    if(xid==0)atomicAdd(&d_zaut[ill], tmp);
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
      d_zaut[ill] += d_a2tmp[itl]*d_zbut[il];
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
    if(tid==0){
      atomicAdd(&d_zaut[ill], tmp);
      //d_zaut[ill] += tmp;
    }
  }
}
template <int THREADS_PER_BLOCK>
__global__ void cuda_matvec_a2_2b_smo
(int kt, int ndl, int nstrtl, double *d_zaut, double *d_zbut, double *d_mat, size_t head)
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
      tmp += d_mat[head+itl]*d_zbut[il];
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
    if(tid==0){
      atomicAdd(&d_zaut[ill], tmp);
      //d_zaut[ill] += tmp;
    }
  }
}
__global__ void cuda_matvec_a2_3_smo
(int kt, int ndl, int nstrtl, double *d_zaut, double *d_zbut, double *d_mat, size_t head)
{
  int il, it, ill, itl;
  int wid = threadIdx.x/32;
  int wlen = 4;
  int tid = threadIdx.x;
  int tlen = blockDim.x;
  int xid = threadIdx.x%32;
  int xlen = 32;
  double tmp=0.0;
  for(it=wid; it<ndl; it+=wlen){
    tmp = 0.0;
    ill=it+nstrtl-1;
    for(il=xid; il<kt; il+=xlen){
      itl=it+il*ndl; 
      tmp += d_mat[head+itl]*d_zbut[il];
    }
    //__syncthreads();
    for (int offset = warpSize/2; offset > 0; offset /= 2){
      tmp += __shfl_down(tmp, offset);
    }
    if(xid==0){
      atomicAdd(&d_zaut[ill], tmp);
      //d_zaut[ill] += tmp;
    }
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
    if(tid==0){
      atomicAdd(&d_zaut[ill], tmp);
      //d_zaut[ill] += tmp;
    }
  }
}
template <int THREADS_PER_BLOCK>
__global__ void cuda_matvec_s_2_smo
(int ndl, int ndt, int nstrtl, int nstrtt, double *d_zaut, double *d_zu, double *d_mat, size_t head)
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
      tmp += d_mat[head+itl]*d_zu[itt];
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
    if(tid==0){
      atomicAdd(&d_zaut[ill], tmp);
      //d_zaut[ill] += tmp;
    }
  }
}
__global__ void cuda_matvec_s_3_smo
(int ndl, int ndt, int nstrtl, int nstrtt, double *d_zaut, double *d_zu, double *d_mat, size_t head)
{
  int il, it, ill, itt, itl;
  int wid = threadIdx.x/32;
  int wlen = 4;
  int tid = threadIdx.x;
  int tlen = blockDim.x;
  int xid = threadIdx.x%32;
  int xlen = 32;
  double tmp=0.0;
  for(il=wid; il<ndl; il+=wlen){
    tmp = 0.0;
    ill=il+nstrtl-1;
    for(it=xid; it<ndt; it+=xlen){
      itt=it+nstrtt-1; 
      itl=it+il*ndt;
      tmp += d_mat[head+itl]*d_zu[itt];
    }
    __syncthreads();
    for (int offset = warpSize/2; offset > 0; offset /= 2){
      tmp += __shfl_down(tmp, offset);
    }
    if(xid==0){
      atomicAdd(&d_zaut[ill], tmp);
      //d_zaut[ill] += tmp;
    }
  }
}

void  c_hacapk_adot_body_lfmtx_cuda_calc
(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu,
 double *time_batch, double *time_set, double *time_copy, int nd) {
  register int ip,il;
  int nlf,ndl,ndt,nstrtl,nstrtt,kt;
  int st_lf_stride = st_leafmtxp->st_lf_stride;
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
      cudaMalloc(&d_a2tmp, sizeof(double)*ndl*kt);
      cudaMemcpy(d_a2tmp, a2tmp, sizeof(double)*ndl*kt, cudaMemcpyHostToDevice);
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
      cudaFree(d_a1);
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
      cudaFree(d_a2tmp);
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
      cudaFree(d_a1);
    }
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

__global__ void myCudaFunc1
(double *zshdw, double *zr, int nd)
{
  int gid = blockIdx.x;
  int glen = gridDim.x;
  int tid = threadIdx.x;
  int tlen = blockDim.x;
  int i, id;
  id = gid*tlen + tid;
  for(i=id;i<nd;i+=glen*tlen){
    zr[i]+=-1.0*zshdw[i];
    zshdw[i]=zr[i];
  }
}
__global__ void myCudaFunc2
(double *zp, double *zr, double *zakp, double beta, double zeta, int nd)
{
  int gid = blockIdx.x;
  int glen = gridDim.x;
  int tid = threadIdx.x;
  int tlen = blockDim.x;
  int i, id;
  id = gid*tlen + tid;
  for(i=id;i<nd;i+=glen*tlen){
    zp[i] = zr[i] + beta * (zp[i] + zeta*zakp[i]);
  }
}
__global__ void myCudaFunc3
(double *zt, double *zr, double alpha, double *zakp, int nd)
{
  int gid = blockIdx.x;
  int glen = gridDim.x;
  int tid = threadIdx.x;
  int tlen = blockDim.x;
  int i, id;
  id = gid*tlen + tid;
  for(i=id;i<nd;i+=glen*tlen){
    zt[i]=zr[i]+alpha*zakp[i];
  }
}
__global__ void myCudaFunc4
(double *u, double alpha, double *zkp, double zeta, double *zkt, int nd)
{
  int gid = blockIdx.x;
  int glen = gridDim.x;
  int tid = threadIdx.x;
  int tlen = blockDim.x;
  int i, id;
  id = gid*tlen + tid;
  for(i=id;i<nd;i+=glen*tlen){
    u[i] += alpha*zkp[i] + zeta*zkt[i];
  }
}
__global__ void myCudaFunc5
(double *zr, double *zt, double zeta, double *zakt, int nd)
{
  int gid = blockIdx.x;
  int glen = gridDim.x;
  int tid = threadIdx.x;
  int tlen = blockDim.x;
  int i, id;
  id = gid*tlen + tid;
  for(i=id;i<nd;i+=glen*tlen){
    zr[i]=zt[i] + zeta*zakt[i];
  }
}

template <int THREADS_PER_BLOCK>
__global__ void myCudaReductionZero
(double *dst, double *src, int n)
{
  int il, it, ill, itt, itl;
  int gid = blockIdx.x;
  int glen = gridDim.x;
  int tid = threadIdx.x;
  int tlen = blockDim.x;
  double tmp=0.0;
  __shared__ double smTmp[THREADS_PER_BLOCK*2];
  int i, id;
  id = gid*tlen + tid;
  if(id==0)dst[0] = 0.0;
  for(i=id; i<n; i+=glen*tlen){
    tmp += src[i]*src[i];
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
  if(tid==0)atomicAdd(&dst[0], tmp);
}
template <int THREADS_PER_BLOCK>
__global__ void myCudaReductionZero
(double *dst, double *src1, double *src2, int n)
{
  int il, it, ill, itt, itl;
  int gid = blockIdx.x;
  int glen = gridDim.x;
  int tid = threadIdx.x;
  int tlen = blockDim.x;
  double tmp=0.0;
  __shared__ double smTmp[THREADS_PER_BLOCK*2];
  int i, id;
  id = gid*tlen + tid;
  if(id==0)dst[0] = 0.0;
  for(i=id; i<n; i+=glen*tlen){
    tmp += src1[i]*src2[i];
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
  if(tid==0)atomicAdd(&dst[0], tmp);
}

__global__ void cuda_vadd(double *a, double *b, int n)
{
  int gid = blockIdx.x;
  int glen = gridDim.x;
  int tid = threadIdx.x;
  int tlen = blockDim.x;
  int id = gid*tlen + tid;
  int i;
  for(i=id; i<n; i+=glen*tlen){
    a[i] += b[i];
  }
}

struct stc_matrixoffset{
  size_t a1, a2;
};
struct stc_matrixoffsetd{
  size_t *a1, *a2;
};
struct stc_cublasmatrices{
  double *d1, *d2;
};

size_t getAllLength(stc_HACApK_leafmtxp *st_leafmtxp)
{
  int nlf, ip, ndt, ndl, kt;
  int st_lf_stride = st_leafmtxp->st_lf_stride;
  size_t offset;
  nlf=st_leafmtxp->nlf;
  printf("getAllLength: check total memory\n"); fflush(stdout);
  offset = 0;
  for(ip=0; ip<nlf; ip++){
    stc_HACApK_leafmtx *sttmp;
    sttmp = (stc_HACApK_leafmtx *)((size_t)((void *)(st_leafmtxp->st_lf)) + st_lf_stride * ip);
    ndl   =sttmp->ndl; 
    ndt   =sttmp->ndt;
    if(sttmp->ltmtx==1){
      kt=sttmp->kt;
      offset += kt*ndt;
      offset += kt*ndl;
      //sttmp->a1;
      //a2tmp;
    } else if(sttmp->ltmtx==2){
      offset += ndl*ndt;
      //sttmp->a1;
    }
  }
  printf("getAllLength: total %lld (%lld byte)\n", offset, offset*sizeof(double)); fflush(stdout);
  return offset;
}
void myCudaCopyAll(struct stc_matrixoffset *smo, stc_HACApK_leafmtxp *st_leafmtxp, double *d_mat)
{
  int nlf, ip, ndt, ndl, kt;
  int st_lf_stride = st_leafmtxp->st_lf_stride;
  size_t offset;
  nlf=st_leafmtxp->nlf;
  printf("myCudaCopyAll: begin\n"); fflush(stdout);
  //printf("myCudaCopyAll: %d nodes\n", nlf); fflush(stdout);
  //printf("myCudaCopyAll: check total memory\n"); fflush(stdout);
  offset = 0;
  for(ip=0; ip<nlf; ip++){
    stc_HACApK_leafmtx *sttmp;
    sttmp = (stc_HACApK_leafmtx *)((size_t)((void *)(st_leafmtxp->st_lf)) + st_lf_stride * ip);
    ndl   =sttmp->ndl; 
    ndt   =sttmp->ndt;
    if(sttmp->ltmtx==1){
      double *a2tmp = (double *)((size_t)((void*)(sttmp->a1))+sttmp->a1size);
      kt=sttmp->kt;
      cudaMemcpy(&d_mat[offset],sttmp->a1,sizeof(double)*kt*ndt,cudaMemcpyHostToDevice);
      smo[ip].a1 = offset;
      offset += kt*ndt;
      cudaMemcpy(&d_mat[offset],a2tmp,sizeof(double)*kt*ndl,cudaMemcpyHostToDevice);
      smo[ip].a2 = offset;
      offset += kt*ndl;
      //sttmp->a1;
      //a2tmp;
    } else if(sttmp->ltmtx==2){
      cudaMemcpy(&d_mat[offset],sttmp->a1,sizeof(double)*ndl*ndt,cudaMemcpyHostToDevice);
      smo[ip].a1 = offset;
      offset += ndl*ndt;
      //sttmp->a1;
    }
  }
  printf("myCudaCopyAll: end\n"); fflush(stdout);
}

void myCublasMakeMatrix(stc_HACApK_leafmtxp *st_leafmtxp, struct stc_cublasmatrices *scm)
{
  int nlf, ip, ndt, ndl, kt;
  int st_lf_stride = st_leafmtxp->st_lf_stride;
  nlf=st_leafmtxp->nlf;
  printf("myCublasMakeMatrix: begin\n"); fflush(stdout);
  //printf("myCudaCopyAll: %d nodes\n", nlf); fflush(stdout);
  //printf("myCudaCopyAll: check total memory\n"); fflush(stdout);
  for(ip=0; ip<nlf; ip++){
    stc_HACApK_leafmtx *sttmp;
    sttmp = (stc_HACApK_leafmtx *)((size_t)((void *)(st_leafmtxp->st_lf)) + st_lf_stride * ip);
    ndl   =sttmp->ndl; 
    ndt   =sttmp->ndt;
    if(sttmp->ltmtx==1){
      double *a2tmp = (double *)((size_t)((void*)(sttmp->a1))+sttmp->a1size);
      kt=sttmp->kt;
      cudaMalloc(&scm[ip].d1, sizeof(double)*kt*ndt);
      //cudaMemcpy(scm[ip].d1, sttmp->a1, sizeof(double)*kt*ndt, cudaMemcpyHostToDevice);
      cudaMalloc(&scm[ip].d2, sizeof(double)*kt*ndl);
      //cudaMemcpy(scm[ip].d2, a2tmp, sizeof(double)*kt*ndl, cudaMemcpyHostToDevice);
    } else if(sttmp->ltmtx==2){
      cudaMalloc(&scm[ip].d1, sizeof(double)*ndl*ndt);
      //cudaMemcpy(scm[ip].a1, sttmp->a1,sizeof(double)*ndl*ndt,cudaMemcpyHostToDevice);
    }
  }
  printf("myCublasMakeMatrix: end\n"); fflush(stdout);
}
void myCublasDestroyMatrix(struct stc_matrixoffset *smo, stc_HACApK_leafmtxp *st_leafmtxp, struct stc_cublasmatrices *scm)
{
  int nlf, ip, ndt, ndl, kt;
  int st_lf_stride = st_leafmtxp->st_lf_stride;
  size_t offset;
  nlf=st_leafmtxp->nlf;
  for(ip=0; ip<nlf; ip++){
    stc_HACApK_leafmtx *sttmp;
    sttmp = (stc_HACApK_leafmtx *)((size_t)((void *)(st_leafmtxp->st_lf)) + st_lf_stride * ip);
    if(sttmp->ltmtx==1){
      cudaFree(&scm[ip].d1);
      cudaFree(&scm[ip].d2);
    } else if(sttmp->ltmtx==2){
      cudaFree(&scm[ip].d1);
    }
  }
}

int myCublasSetMatrix(stc_HACApK_leafmtxp *st_leafmtxp, struct stc_cublasmatrices *scm)
{
#define CHK_ERROR(st) \
  switch(st){ \
 case CUBLAS_STATUS_NOT_INITIALIZED: printf("CUBLAS_STATUS_NOT_INITIALIZED\n"); break; \
 case CUBLAS_STATUS_INVALID_VALUE: printf("CUBLAS_STATUS_INVALID_VALUE\n"); break; \
 case CUBLAS_STATUS_MAPPING_ERROR: printf("CUBLAS_STATUS_MAPPING_ERROR\n"); break; \
}
  cublasStatus_t st;
  int nlf, ip, ndt, ndl, kt;
  int st_lf_stride = st_leafmtxp->st_lf_stride;
  nlf=st_leafmtxp->nlf;
  printf("myCublasSetMatrix: begin\n"); fflush(stdout);
  for(ip=0; ip<nlf; ip++){
    stc_HACApK_leafmtx *sttmp;
    sttmp = (stc_HACApK_leafmtx *)((size_t)((void *)(st_leafmtxp->st_lf)) + st_lf_stride * ip);
    ndl   =sttmp->ndl; 
    ndt   =sttmp->ndt;
    if(sttmp->ltmtx==1){
      double *a2tmp = (double *)((size_t)((void*)(sttmp->a1))+sttmp->a1size);
      kt=sttmp->kt;
      st = cublasSetMatrix(ndt,kt,sizeof(double),sttmp->a1,ndt,scm[ip].d1,ndt);
      if(st!=CUBLAS_STATUS_SUCCESS){printf("cublasSetMatrix: failed (%d,d1)\n", ip); CHK_ERROR(st); return -1;}
      st = cublasSetMatrix(ndl,kt,sizeof(double),a2tmp,ndl,scm[ip].d2,ndl);
      if(st!=CUBLAS_STATUS_SUCCESS){printf("cublasSetMatrix: failed (%d,d2)\n", ip); CHK_ERROR(st); return -1;}
    } else if(sttmp->ltmtx==2){
      st = cublasSetMatrix(ndt,ndl,sizeof(double),sttmp->a1,ndt,scm[ip].d1,ndt);
      if(st!=CUBLAS_STATUS_SUCCESS){printf("cublasSetMatrix: failed (%d)\n", ip); CHK_ERROR(st); return -1;}
    }
  }
  printf("myCublasSetMatrix: end\n"); fflush(stdout);
  return 0;
}

void  c_hacapk_adot_body_lfmtx_cuda_calc_device
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
  double *d_a1, *d_a2tmp;
  cudaMalloc(&d_zaut, sizeof(double)*nd);
  cudaMalloc(&d_zbut, sizeof(double)*st_leafmtxp->ktmax);

  //for(i=0;i<nd;i++)zau[i]=0.0;
  cudaMemset(zau, 0.0, sizeof(double)*nd);

  nlf=st_leafmtxp->nlf;
  //fprintf(stderr,"nlf=%d \n",nlf);

  zaut = (double*)malloc(sizeof(double)*nd);
  for(il=0;il<nd;il++)zaut[il]=0.0;
  cudaMemset(d_zaut, 0.0, sizeof(double)*nd);
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
      cudaMalloc(&d_a2tmp, sizeof(double)*ndl*kt);
      cudaMemcpy(d_a2tmp, a2tmp, sizeof(double)*ndl*kt, cudaMemcpyHostToDevice);
      cudaMalloc(&d_a1, sizeof(double)*ndt*kt);
      cudaMemcpy(d_a1, sttmp->a1, sizeof(double)*ndt*kt, cudaMemcpyHostToDevice);
      //cuda_matvec_a1<<<1,1>>>(kt,ndt,nstrtt,d_zbut,d_a1,d_zu);
      cuda_matvec_a1_2<128><<<112,128>>>(kt,ndt,nstrtt,d_zbut,d_a1,zu);
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
      cudaFree(d_a1);
      cudaFree(d_a2tmp);
    } else if(sttmp->ltmtx==2){
      cudaMalloc(&d_a1, sizeof(double)*ndt*ndl);
      cudaMemcpy(d_a1, sttmp->a1, sizeof(double)*ndt*ndl, cudaMemcpyHostToDevice);
      //cuda_matvec_s<<<1,1>>>(ndl,ndt,nstrtl,nstrtt,d_zaut,d_a1,d_zu);
      cuda_matvec_s_2<128><<<112,128>>>(ndl,ndt,nstrtl,nstrtt,d_zaut,d_a1,zu);
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
      cudaFree(d_a1);
    }
  }
  /*
  cudaMemcpy(zaut, d_zaut, sizeof(double)*nd, cudaMemcpyDeviceToHost);
  for(il=ls-1;il<=le-1;il++){
    zau[il] += zaut[il];
  }
  */
  cuda_vadd<<<112,128>>>(zau,d_zaut,nd);
  /*
    for(il=ls-1;il<=le-1;il++){
    #pragma omp atomic
    zau[il] += zaut[il];
    }
  */
  free(zaut); free(zbut);
  cudaFree(d_zaut); cudaFree(d_zbut);
}

void  c_hacapk_adot_body_lfmtx_cuda_calc_device_smo
(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu,
 double *time_batch, double *time_set, double *time_copy, int nd, struct stc_matrixoffset *smo, double *d_mat) {
  register int ip,il,it;
  int nlf,ndl,ndt,nstrtl,nstrtt,kt,itl,itt,ill;
  int st_lf_stride = st_leafmtxp->st_lf_stride;
  size_t a1size;
  int ith, nths, nthe;
  double *zaut;//, *zbut;
  int ls, le;
  int i;

  double *d_zaut, *d_zbut;
  cudaMalloc(&d_zaut, sizeof(double)*nd);
  cudaMalloc(&d_zbut, sizeof(double)*st_leafmtxp->ktmax);

  //for(i=0;i<nd;i++)zau[i]=0.0;
  cudaMemset(zau, 0.0, sizeof(double)*nd);

  nlf=st_leafmtxp->nlf;
  //fprintf(stderr,"nlf=%d \n",nlf);
  /*
  zaut = (double*)malloc(sizeof(double)*nd);
  for(il=0;il<nd;il++)zaut[il]=0.0;
  */
  cudaMemset(d_zaut, 0.0, sizeof(double)*nd);
  //printf("st_leafmtxp->ktmax = %d\n",st_leafmtxp->ktmax);
  //zbut = (double*)malloc(sizeof(double)*st_leafmtxp->ktmax);
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
    if(nstrtl<ls)ls=nstrtl;
    if(nstrtl+ndl-1>le)le=nstrtl+ndl-1;
    //printf("DBG: ltmtx=%d\n",sttmp->ltmtx);
    if(sttmp->ltmtx==1){
      /**/
      //double *a2tmp = (double *)((size_t)((void*)(sttmp->a1))+sttmp->a1size);
      /**/
      kt=sttmp->kt;
      //for(il=0;il<kt;il++)zbut[il]=0.0;
      //cudaMemcpy(d_zbut, zbut, sizeof(double)*kt, cudaMemcpyHostToDevice);
      //cudaMemset(d_zbut, 0.0, sizeof(double)*kt);
      cuda_matvec_a1_2_smo<128><<<112,128>>>(kt,ndt,nstrtt,d_zbut,zu, d_mat, smo[ip].a1);
      /*
	for(il=0; il<kt; il++){
	  for(it=0; it<ndt; it++){
	    itt=it+nstrtt-1;
	    itl=it+il*ndt; 
	    zbut[il] += sttmp->a1[itl]*zu[itt];
	  }
	}
      */
      cuda_matvec_a2_2b_smo<128><<<112,128>>>(kt,ndl,nstrtl,d_zaut,d_zbut, d_mat, smo[ip].a2);
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
      cuda_matvec_s_2_smo<128><<<112,128>>>(ndl,ndt,nstrtl,nstrtt,d_zaut,zu, d_mat, smo[ip].a1);
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
  }
  /*
  cudaMemcpy(zaut, d_zaut, sizeof(double)*nd, cudaMemcpyDeviceToHost);
  for(il=ls-1;il<=le-1;il++){
    zau[il] += zaut[il];
  }
  */
  cuda_vadd<<<112,128>>>(zau,d_zaut,nd);
  /*
    for(il=ls-1;il<=le-1;il++){
    #pragma omp atomic
    zau[il] += zaut[il];
    }
  */
  //free(zaut); free(zbut);
  cudaFree(d_zaut); cudaFree(d_zbut);
}

void  c_hacapk_adot_body_lfmtx_cuda_calc_device_smo_cublas
(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu,
 double *time_batch, double *time_set, double *time_copy, int nd, struct stc_matrixoffset *smo,
 cublasHandle_t handle, struct stc_cublasmatrices *scm, double *d_mat, double *d_zaut, double *d_zbut) {
  register int ip,il,it;
  int nlf,ndl,ndt,nstrtl,nstrtt,kt,itl,itt,ill;
  int st_lf_stride = st_leafmtxp->st_lf_stride;
  int ith, nths, nthe;
  double *zaut;//, *zbut;
  int ls, le;
  int i;
  double done=1.0, dzero=0.0;

  //double *d_zaut, *d_zbut;
  /*
  cudaMalloc(&d_zaut, sizeof(double)*nd);
  cudaMalloc(&d_zbut, sizeof(double)*st_leafmtxp->ktmax);
  */
  cudaMemset(zau, 0.0, sizeof(double)*nd);

  nlf=st_leafmtxp->nlf;
  cudaMemset(d_zaut, 0.0, sizeof(double)*nd);
  ls = nd;
  le = 1;
  for(ip=0; ip<nlf; ip++){
    stc_HACApK_leafmtx *sttmp;
    sttmp = (stc_HACApK_leafmtx *)((size_t)((void *)(st_leafmtxp->st_lf)) + st_lf_stride * ip);
    ndl   =sttmp->ndl; 
    ndt   =sttmp->ndt;
    nstrtl=sttmp->nstrtl; 
    nstrtt=sttmp->nstrtt;
    if(nstrtl<ls)ls=nstrtl;
    if(nstrtl+ndl-1>le)le=nstrtl+ndl-1;
    if(sttmp->ltmtx==1){
      kt=sttmp->kt;
      //cuda_matvec_a1_2_smo<128><<<112,128>>>(kt,ndt,nstrtt,d_zbut,zu, d_mat, smo[ip].a1);
      cudaMemset(d_zbut, 0.0, sizeof(double)*kt);
      cublasDgemv(handle,CUBLAS_OP_T, ndt,kt, &done,scm[ip].d1, ndt,&zu[nstrtt-1],1,&done,d_zbut,1);
      //cuda_matvec_a2_2b_smo<128><<<112,128>>>(kt,ndl,nstrtl,d_zaut,d_zbut, d_mat, smo[ip].a2);
      cublasDgemv(handle,CUBLAS_OP_N, ndl,kt, &done,scm[ip].d2, ndl,d_zbut,1,&done,&d_zaut[nstrtl-1],1);
    } else if(sttmp->ltmtx==2){
      //cuda_matvec_s_2_smo<128><<<112,128>>>(ndl,ndt,nstrtl,nstrtt,d_zaut,zu, d_mat, smo[ip].a1);
      cublasDgemv(handle,CUBLAS_OP_T, ndt,ndl, &done,scm[ip].d1, ndt,&zu[nstrtt-1],1,&done,&d_zaut[nstrtl-1],1);
    }
  }
  cuda_vadd<<<112,128>>>(zau,d_zaut,nd);
  //cudaFree(d_zaut); cudaFree(d_zbut);
}

void  c_hacapk_adot_body_lfmtx_cuda_calc_device_smo_1blk
(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu,
 double *time_batch, double *time_set, double *time_copy, int nd, struct stc_matrixoffset *smo, double *d_mat) {
  register int ip,il,it;
  int nlf,ndl,ndt,nstrtl,nstrtt,kt,itl,itt,ill;
  int st_lf_stride = st_leafmtxp->st_lf_stride;
  size_t a1size;
  int ith, nths, nthe;
  double *zaut;//, *zbut;
  int ls, le;
  int i;

  double *d_zaut, *d_zbut;
  cudaMalloc(&d_zaut, sizeof(double)*nd);
  cudaMalloc(&d_zbut, sizeof(double)*st_leafmtxp->ktmax);
  cudaMemset(zau, 0.0, sizeof(double)*nd);
  nlf=st_leafmtxp->nlf;
  cudaMemset(d_zaut, 0.0, sizeof(double)*nd);
  ls = nd;
  le = 1;
  for(ip=0; ip<nlf; ip++){
    stc_HACApK_leafmtx *sttmp;
    sttmp = (stc_HACApK_leafmtx *)((size_t)((void *)(st_leafmtxp->st_lf)) + st_lf_stride * ip);
    ndl   =sttmp->ndl; 
    ndt   =sttmp->ndt;
    nstrtl=sttmp->nstrtl; 
    nstrtt=sttmp->nstrtt;
    if(nstrtl<ls)ls=nstrtl;
    if(nstrtl+ndl-1>le)le=nstrtl+ndl-1;
    if(sttmp->ltmtx==1){
      kt=sttmp->kt;
      cuda_matvec_a1_2_smo<128><<<1,128>>>(kt,ndt,nstrtt,d_zbut,zu, d_mat, smo[ip].a1);
      cuda_matvec_a2_2b_smo<128><<<1,128>>>(kt,ndl,nstrtl,d_zaut,d_zbut, d_mat, smo[ip].a2);
    } else if(sttmp->ltmtx==2){
      cuda_matvec_s_2_smo<128><<<1,128>>>(ndl,ndt,nstrtl,nstrtt,d_zaut,zu, d_mat, smo[ip].a1);
    }
  }
  cuda_vadd<<<112,128>>>(zau,d_zaut,nd);
  cudaFree(d_zaut); cudaFree(d_zbut);
}

void c_hacapk_adot_body_lfmtx_cuda_calc_device_smo_async
(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu,
 double *time_batch, double *time_set, double *time_copy, int nd, struct stc_matrixoffset *smo, double *d_mat,
 cudaStream_t *s, int streams, double *d_zaut, double **d_zbut) {
  register int ip,il,it;
  int nlf,ndl,ndt,nstrtl,nstrtt,kt,itl,itt,ill;
  int st_lf_stride = st_leafmtxp->st_lf_stride;
  size_t a1size;
  int ith, nths, nthe;
  double *zaut;
  int ls, le;
  int i;
  cudaMemset(zau, 0.0, sizeof(double)*nd);
  nlf=st_leafmtxp->nlf;
  cudaMemset(d_zaut, 0.0, sizeof(double)*nd);
  ls = nd;
  le = 1;
  for(ip=0; ip<nlf; ip++){
    stc_HACApK_leafmtx *sttmp;
    sttmp = (stc_HACApK_leafmtx *)((size_t)((void *)(st_leafmtxp->st_lf)) + st_lf_stride * ip);
    ndl   =sttmp->ndl; 
    ndt   =sttmp->ndt;
    nstrtl=sttmp->nstrtl; 
    nstrtt=sttmp->nstrtt;
    if(nstrtl<ls)ls=nstrtl;
    if(nstrtl+ndl-1>le)le=nstrtl+ndl-1;
    if(sttmp->ltmtx==1){
      kt=sttmp->kt;
      cuda_matvec_a1_2_smo<128><<<1,128,0,s[ip%streams]>>>(kt,ndt,nstrtt,d_zbut[ip%streams],zu, d_mat, smo[ip].a1);
      cuda_matvec_a2_2b_smo<128><<<1,128,0,s[ip%streams]>>>(kt,ndl,nstrtl,d_zaut,d_zbut[ip%streams], d_mat, smo[ip].a2);
    } else if(sttmp->ltmtx==2){
      cuda_matvec_s_2_smo<128><<<1,128,0,s[ip%streams]>>>(ndl,ndt,nstrtl,nstrtt,d_zaut,zu, d_mat, smo[ip].a1);
    }
  }
  cudaThreadSynchronize();
  //for(i=0;i<streams;i++){cudaStreamSynchronize(s[i]);}
  cuda_vadd<<<112,128>>>(zau,d_zaut,nd);
}

#if 0
void c_hacapk_adot_body_lfmtx_cuda_calc_device_smo_async2
(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu,
 double *time_batch, double *time_set, double *time_copy, int nd, struct stc_matrixoffset *smo, double *d_mat,
 cudaStream_t *s, int streams, double *d_zaut, double **d_zbut) {
  register int ip,il,it;
  int nlf,ndl,ndt,nstrtl,nstrtt,kt,itl,itt,ill;
  int st_lf_stride = st_leafmtxp->st_lf_stride;
  size_t a1size;
  int ith, nths, nthe;
  double *zaut;
  int ls, le;
  int i;
  cudaMemset(zau, 0.0, sizeof(double)*nd);
  nlf=st_leafmtxp->nlf;
  cudaMemset(d_zaut, 0.0, sizeof(double)*nd);
  ls = nd;
  le = 1;
  for(ip=0; ip<nlf; ip++){
    stc_HACApK_leafmtx *sttmp;
    sttmp = (stc_HACApK_leafmtx *)((size_t)((void *)(st_leafmtxp->st_lf)) + st_lf_stride * ip);
    ndl   =sttmp->ndl; 
    ndt   =sttmp->ndt;
    nstrtl=sttmp->nstrtl; 
    nstrtt=sttmp->nstrtt;
    if(nstrtl<ls)ls=nstrtl;
    if(nstrtl+ndl-1>le)le=nstrtl+ndl-1;
    if(sttmp->ltmtx==1){
      kt=sttmp->kt;
      cuda_matvec_a1_3_smo<128><<<1,128,0,s[ip%streams]>>>(kt,ndt,nstrtt,d_zbut[ip%streams],zu, d_mat, smo[ip].a1);
      //cuda_matvec_a2_3_smo<128><<<1,128,0,s[ip%streams]>>>(kt,ndl,nstrtl,d_zaut,d_zbut[ip%streams], d_mat, smo[ip].a2);
      cuda_matvec_a2_3_smo<128><<<1,128,0,s[ip%streams]>>>(kt,ndl,nstrtl,zau,d_zbut[ip%streams], d_mat, smo[ip].a2);
    } else if(sttmp->ltmtx==2){
      //cuda_matvec_s_3_smo<128><<<1,128,0,s[ip%streams]>>>(ndl,ndt,nstrtl,nstrtt,d_zaut,zu, d_mat, smo[ip].a1);
      cuda_matvec_s_3_smo<128><<<1,128,0,s[ip%streams]>>>(ndl,ndt,nstrtl,nstrtt,zau,zu, d_mat, smo[ip].a1);
    }
  }
  cudaThreadSynchronize();
  //for(i=0;i<streams;i++){cudaStreamSynchronize(s[i]);}
  //cuda_vadd<<<112,128>>>(zau,d_zaut,nd);
}
#endif
// merge kernel
void c_hacapk_adot_body_lfmtx_cuda_calc_device_smo_async2
(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu,
 double *time_batch, double *time_set, double *time_copy, int nd, struct stc_matrixoffset *smo, double *d_mat,
 cudaStream_t *s, int streams, double *d_zaut, double **d_zbut, int merge) {
  register int ip,il,it;
  int nlf,ndl,ndt,nstrtl,nstrtt,kt,itl,itt,ill;
  int st_lf_stride = st_leafmtxp->st_lf_stride;
  size_t a1size;
  int ith, nths, nthe;
  double *zaut;
  int ls, le;
  int i;
  cudaMemset(zau, 0.0, sizeof(double)*nd);
  nlf=st_leafmtxp->nlf;
  cudaMemset(d_zaut, 0.0, sizeof(double)*nd);
  ls = nd;
  le = 1;
  for(ip=0; ip<nlf; ip++){
    stc_HACApK_leafmtx *sttmp;
    sttmp = (stc_HACApK_leafmtx *)((size_t)((void *)(st_leafmtxp->st_lf)) + st_lf_stride * ip);
    ndl   =sttmp->ndl; 
    ndt   =sttmp->ndt;
    nstrtl=sttmp->nstrtl; 
    nstrtt=sttmp->nstrtt;
    if(nstrtl<ls)ls=nstrtl;
    if(nstrtl+ndl-1>le)le=nstrtl+ndl-1;
    if(sttmp->ltmtx==1){
      kt=sttmp->kt;
      if(merge==0){
	cuda_matvec_a1_3_smo<<<1,128,0,s[ip%streams]>>>(kt,ndt,nstrtt,d_zbut[ip%streams],zu, d_mat, smo[ip].a1);
	cuda_matvec_a2_3_smo<<<1,128,0,s[ip%streams]>>>(kt,ndl,nstrtl,zau,d_zbut[ip%streams], d_mat, smo[ip].a2);
      }else{
	cuda_matvec_a12_3_smo<<<1,128,st_leafmtxp->ktmax*sizeof(double),s[ip%streams]>>>
	  (ndt,ndl,kt,nstrtt,nstrtl,zau,d_mat,zu,smo[ip].a1,smo[ip].a2);
      }
    } else if(sttmp->ltmtx==2){
      cuda_matvec_s_3_smo<<<1,128,0,s[ip%streams]>>>(ndl,ndt,nstrtl,nstrtt,zau,zu, d_mat, smo[ip].a1);
    }
  }
  cudaThreadSynchronize();
  //for(i=0;i<streams;i++){cudaStreamSynchronize(s[i]);}
  //cuda_vadd<<<112,128>>>(zau,d_zaut,nd);
}
/*
void c_hacapk_adot_body_lfmtx_cuda_calc_device_smo_async
(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu,
 double *time_batch, double *time_set, double *time_copy, int nd, struct stc_matrixoffset *smo, double *d_mat,
 cudaStream_t *s, int streams, double *d_zaut, double **d_zbut) {
  register int ip,il,it;
  int nlf,ndl,ndt,nstrtl,nstrtt,kt,itl,itt,ill;
  int st_lf_stride = st_leafmtxp->st_lf_stride;
  size_t a1size;
  int ith, nths, nthe;
  double *zaut;
  int ls, le;
  int i;
  cudaMemset(zau, 0.0, sizeof(double)*nd);
  nlf=st_leafmtxp->nlf;
  cudaMemset(d_zaut, 0.0, sizeof(double)*nd);
  ls = nd;
  le = 1;
  for(ip=0; ip<nlf; ip++){
    stc_HACApK_leafmtx *sttmp;
    sttmp = (stc_HACApK_leafmtx *)((size_t)((void *)(st_leafmtxp->st_lf)) + st_lf_stride * ip);
    ndl   =sttmp->ndl; 
    ndt   =sttmp->ndt;
    nstrtl=sttmp->nstrtl; 
    nstrtt=sttmp->nstrtt;
    if(nstrtl<ls)ls=nstrtl;
    if(nstrtl+ndl-1>le)le=nstrtl+ndl-1;
    if(sttmp->ltmtx==1){
      kt=sttmp->kt;
      cuda_matvec_a1_2_smo<128><<<1,128,0,s[ip%streams]>>>(kt,ndt,nstrtt,d_zbut[ip%streams],zu, d_mat, smo[ip].a1);
      cuda_matvec_a2_2b_smo<128><<<1,128,0,s[ip%streams]>>>(kt,ndl,nstrtl,d_zaut,d_zbut[ip%streams], d_mat, smo[ip].a2);
    } else if(sttmp->ltmtx==2){
      cuda_matvec_s_2_smo<128><<<1,128,0,s[ip%streams]>>>(ndl,ndt,nstrtl,nstrtt,d_zaut,zu, d_mat, smo[ip].a1);
    }
  }
  cudaThreadSynchronize();
  //for(i=0;i<streams;i++){cudaStreamSynchronize(s[i]);}
  cuda_vadd<<<112,128>>>(zau,d_zaut,nd);
}
*/
struct stc_1kernel_info{
  int *ltmtx;
  int *kt;
  int *ndl, *ndt;
  int *nstrtl, *nstrtt;
};
void myMake1kInfo(struct stc_1kernel_info *dinfo, stc_HACApK_leafmtxp *st_leafmtxp, struct stc_matrixoffset *smo, struct stc_matrixoffsetd *smd)
{
  int st_lf_stride = st_leafmtxp->st_lf_stride;
  int ip, nlf;
  struct stc_1kernel_info info;
  printf("myMake1kInfo: begin\n"); fflush(stdout);
  nlf=st_leafmtxp->nlf;
  info.ltmtx  = (int*)malloc(sizeof(int)*nlf);
  info.kt     = (int*)malloc(sizeof(int)*nlf);
  info.ndl    = (int*)malloc(sizeof(int)*nlf);
  info.ndt    = (int*)malloc(sizeof(int)*nlf);
  info.nstrtl = (int*)malloc(sizeof(int)*nlf);
  info.nstrtt = (int*)malloc(sizeof(int)*nlf);
  cudaMalloc(&dinfo->ltmtx,sizeof(int)*nlf);
  cudaMalloc(&dinfo->ltmtx,sizeof(int)*nlf);
  cudaMalloc(&dinfo->kt,sizeof(int)*nlf);
  cudaMalloc(&dinfo->ndl,sizeof(int)*nlf);
  cudaMalloc(&dinfo->ndt,sizeof(int)*nlf);
  cudaMalloc(&dinfo->nstrtl,sizeof(int)*nlf);
  cudaMalloc(&dinfo->nstrtt,sizeof(int)*nlf);
  for(ip=0; ip<nlf; ip++){
    stc_HACApK_leafmtx *sttmp;
    sttmp = (stc_HACApK_leafmtx *)((size_t)((void *)(st_leafmtxp->st_lf)) + st_lf_stride * ip);
    info.ltmtx[ip]  = sttmp->ltmtx;
    info.ndl[ip]    = sttmp->ndl; 
    info.ndt[ip]    = sttmp->ndt;
    info.nstrtl[ip] = sttmp->nstrtl; 
    info.nstrtt[ip] = sttmp->nstrtt; 
    if(sttmp->ltmtx==1){
      info.kt[ip] = sttmp->kt;
    }else{
      info.kt[ip] = -1;
    }
  }
  cudaMemcpy(dinfo->ltmtx, info.ltmtx, sizeof(int)*nlf, cudaMemcpyHostToDevice);
  cudaMemcpy(dinfo->kt, info.kt, sizeof(int)*nlf, cudaMemcpyHostToDevice);
  cudaMemcpy(dinfo->ndl, info.ndl, sizeof(int)*nlf, cudaMemcpyHostToDevice);
  cudaMemcpy(dinfo->ndt, info.ndt, sizeof(int)*nlf, cudaMemcpyHostToDevice);
  cudaMemcpy(dinfo->nstrtl, info.nstrtl, sizeof(int)*nlf, cudaMemcpyHostToDevice);
  cudaMemcpy(dinfo->nstrtt, info.nstrtt, sizeof(int)*nlf, cudaMemcpyHostToDevice);
  free(info.ltmtx); free(info.kt); free(info.ndl); free(info.ndt); free(info.nstrtl); free(info.nstrtt);

  struct stc_matrixoffsetd tmpsmd;
  tmpsmd.a1 = (size_t*)malloc(sizeof(size_t)*nlf);
  tmpsmd.a2 = (size_t*)malloc(sizeof(size_t)*nlf);
  cudaMalloc(&smd->a1, sizeof(size_t)*nlf);
  cudaMalloc(&smd->a2, sizeof(size_t)*nlf);
  for(ip=0; ip<nlf; ip++){
    tmpsmd.a1[ip] = smo[ip].a1;
    tmpsmd.a2[ip] = smo[ip].a2;
  }
  cudaMemcpy(smd->a1, tmpsmd.a1, sizeof(size_t)*nlf, cudaMemcpyHostToDevice);
  cudaMemcpy(smd->a2, tmpsmd.a2, sizeof(size_t)*nlf, cudaMemcpyHostToDevice);
  printf("myMake1kInfo: end\n"); fflush(stdout);
}
void myDestroy1kInfo(struct stc_1kernel_info *dinfo)
{
  cudaFree(dinfo->ltmtx);
  cudaFree(dinfo->kt);
  cudaFree(dinfo->ndl);
  cudaFree(dinfo->ndt);
  cudaFree(dinfo->nstrtl);
  cudaFree(dinfo->nstrtt);
}
__global__ void cuda_matvec_1kernel0
(struct stc_1kernel_info info, struct stc_matrixoffsetd smd,
 double *d_zaut, double *d_zu, double *d_mat, int nlf, int ktmax)
{ // X blocks, 128 threads
  int gid  = blockIdx.x;
  int glen = gridDim.x;
  int tid  = threadIdx.x;
  int tlen = blockDim.x;
  int ndl, ndt, nstrtl, nstrtt, ltmtx;
  int wid  = threadIdx.x/32;
  int wlen = 4;
  int xid  = threadIdx.x%32;
  int xlen = 32;
  int ip, kt, il, it, itt, itl, ill;
  size_t head;
  double tmp;
  extern __shared__ double tmpbut[];
  ip = gid;
  {
    ndl = info.ndl[ip];
    ndt = info.ndt[ip];
    nstrtl = info.nstrtl[ip];
    nstrtt = info.nstrtt[ip];
    ltmtx = info.ltmtx[ip];
    if(ltmtx==1){
      kt = info.kt[ip];
      //cuda_matvec_1k_a1(kt,ndt,nstrtt,d_zbut[ip/X],zu, d_mat, smo[ip].a1);
      head = smd.a1[ip];
      for(il=wid; il<kt; il+=wlen){
	tmp = 0.0;
	for(it=xid; it<ndt; it+=xlen){
	  itt=it+nstrtt-1;
	  itl=it+il*ndt; 
	  tmp += d_mat[head+itl]*d_zu[itt];
	}
	for (int offset = warpSize/2; offset > 0; offset /= 2){
	  tmp += __shfl_down(tmp, offset);
	}
	if(xid==0){
	  tmpbut[il] = tmp;
	}
      }
      __syncthreads();
      //cuda_matvec_1k_a2(kt,ndl,nstrtl,zau,d_zbut[ip/X], d_mat, smo[ip].a2);
      head = smd.a2[ip];
      for(it=wid; it<ndl; it+=wlen){
	tmp = 0.0;
	ill=it+nstrtl-1;
	for(il=xid; il<kt; il+=xlen){
	  itl=it+il*ndl; 
	  tmp += d_mat[head+itl]*tmpbut[il];
	}
	for (int offset = warpSize/2; offset > 0; offset /= 2){
	  tmp += __shfl_down(tmp, offset);
	}
	if(xid==0){
	  atomicAdd(&d_zaut[ill], tmp);
	}
      }
    } else if(ltmtx==2){
      //cuda_matvec_1k_s(ndl,ndt,nstrtl,nstrtt,zau,zu, d_mat, smo[ip].a1);
      head = smd.a1[ip];
      for(il=wid; il<ndl; il+=wlen){
	tmp = 0.0;
	ill=il+nstrtl-1;
	for(it=xid; it<ndt; it+=xlen){
	  itt=it+nstrtt-1; 
	  itl=it+il*ndt;
	  tmp += d_mat[head+itl]*d_zu[itt];
	}
	for (int offset = warpSize/2; offset > 0; offset /= 2){
	  tmp += __shfl_down(tmp, offset);
	}
	if(xid==0){
	  atomicAdd(&d_zaut[ill], tmp);
	}
      }
    }
  }
}
__global__ void cuda_matvec_1kernel0a
(struct stc_1kernel_info info, struct stc_matrixoffsetd smd,
 double *d_zaut, double *d_zu, double *d_mat, int nlf, int ktmax)
{ // X blocks, 128 threads
  int gid  = blockIdx.x;
  int glen = gridDim.x;
  int tid  = threadIdx.x;
  int tlen = blockDim.x;
  int ndl, ndt, nstrtl, nstrtt, ltmtx;
  int wid  = threadIdx.x/32;
  int wlen = 4;
  int xid  = threadIdx.x%32;
  int xlen = 32;
  int ip, kt, il, it, itt, itl, ill;
  size_t head;
  double tmp;
  extern __shared__ double tmpbut[];
  ip = gid;
  {
    ndl = info.ndl[ip];
    ndt = info.ndt[ip];
    nstrtl = info.nstrtl[ip];
    nstrtt = info.nstrtt[ip];
    ltmtx = info.ltmtx[ip];
    if(ltmtx==1){
      kt = info.kt[ip];
      //cuda_matvec_1k_a1(kt,ndt,nstrtt,d_zbut[ip/X],zu, d_mat, smo[ip].a1);
      head = smd.a1[ip];
      for(il=wid; il<kt; il+=wlen){
	tmp = 0.0;
	for(it=xid; it<ndt; it+=xlen){
	  itt=it+nstrtt-1;
	  itl=it+il*ndt; 
	  tmp += d_mat[head+itl]*d_zu[itt];
	}
	for (int offset = warpSize/2; offset > 0; offset /= 2){
	  tmp += __shfl_down(tmp, offset);
	}
	if(xid==0){
	  tmpbut[il] = tmp;
	}
      }
      __syncthreads();
      //cuda_matvec_1k_a2(kt,ndl,nstrtl,zau,d_zbut[ip/X], d_mat, smo[ip].a2);
      head = smd.a2[ip];
      for(il=tid; il<ndl; il+=tlen){
	tmp = 0.0;
	ill = il+nstrtl-1;
	for(it=0; it<kt; it++){
	  itl = il+it*ndl;
	  tmp += d_mat[head+itl]*tmpbut[it];
	}
	atomicAdd(&d_zaut[ill], tmp);
      }
    } else if(ltmtx==2){
      //cuda_matvec_1k_s(ndl,ndt,nstrtl,nstrtt,zau,zu, d_mat, smo[ip].a1);
      head = smd.a1[ip];
      for(il=wid; il<ndl; il+=wlen){
	tmp = 0.0;
	ill=il+nstrtl-1;
	for(it=xid; it<ndt; it+=xlen){
	  itt=it+nstrtt-1; 
	  itl=it+il*ndt;
	  tmp += d_mat[head+itl]*d_zu[itt];
	}
	for (int offset = warpSize/2; offset > 0; offset /= 2){
	  tmp += __shfl_down(tmp, offset);
	}
	if(xid==0){
	  atomicAdd(&d_zaut[ill], tmp);
	}
      }
    }
  }
}
__global__ void cuda_matvec_1kernel
(struct stc_1kernel_info info, struct stc_matrixoffsetd smd,
 double *d_zaut, double *d_zu, double *d_mat, int nlf, int ktmax)
{ // X blocks, 128 threads
  int gid  = blockIdx.x;
  int glen = gridDim.x;
  int tid  = threadIdx.x;
  int tlen = blockDim.x;
  int ndl, ndt, nstrtl, nstrtt, ltmtx;
  int wid  = threadIdx.x/32;
  int wlen = 4;
  int xid  = threadIdx.x%32;
  int xlen = 32;
  int ip, kt, il, it, itt, itl, ill;
  size_t head;
  double tmp;
  extern __shared__ double tmpbut[];
  for(ip=gid; ip<nlf; ip+=glen){
    ndl = info.ndl[ip];
    ndt = info.ndt[ip];
    nstrtl = info.nstrtl[ip];
    nstrtt = info.nstrtt[ip];
    ltmtx = info.ltmtx[ip];
    if(ltmtx==1){
      kt = info.kt[ip];
      //cuda_matvec_1k_a1(kt,ndt,nstrtt,d_zbut[ip/X],zu, d_mat, smo[ip].a1);
      head = smd.a1[ip];
      for(il=wid; il<kt; il+=wlen){
	tmp = 0.0;
	for(it=xid; it<ndt; it+=xlen){
	  itt=it+nstrtt-1;
	  itl=it+il*ndt; 
	  tmp += d_mat[head+itl]*d_zu[itt];
	}
	for (int offset = warpSize/2; offset > 0; offset /= 2){
	  tmp += __shfl_down(tmp, offset);
	}
	if(xid==0){
	  tmpbut[il] = tmp;
	}
      }
      __syncthreads();
      //cuda_matvec_1k_a2(kt,ndl,nstrtl,zau,d_zbut[ip/X], d_mat, smo[ip].a2);
      head = smd.a2[ip];
      for(it=wid; it<ndl; it+=wlen){
	tmp = 0.0;
	ill=it+nstrtl-1;
	for(il=xid; il<kt; il+=xlen){
	  itl=it+il*ndl; 
	  tmp += d_mat[head+itl]*tmpbut[il];
	}
	for (int offset = warpSize/2; offset > 0; offset /= 2){
	  tmp += __shfl_down(tmp, offset);
	}
	if(xid==0){
	  atomicAdd(&d_zaut[ill], tmp);
	}
      }
    } else if(ltmtx==2){
      //cuda_matvec_1k_s(ndl,ndt,nstrtl,nstrtt,zau,zu, d_mat, smo[ip].a1);
      head = smd.a1[ip];
      for(il=wid; il<ndl; il+=wlen){
	tmp = 0.0;
	ill=il+nstrtl-1;
	for(it=xid; it<ndt; it+=xlen){
	  itt=it+nstrtt-1; 
	  itl=it+il*ndt;
	  tmp += d_mat[head+itl]*d_zu[itt];
	}
	for (int offset = warpSize/2; offset > 0; offset /= 2){
	  tmp += __shfl_down(tmp, offset);
	}
	if(xid==0){
	  atomicAdd(&d_zaut[ill], tmp);
	}
      }
    }
  }
}

void c_hacapk_adot_body_lfmtx_cuda_calc_device_smo_1kernel
(double *zau, stc_HACApK_leafmtxp *st_leafmtxp, double *zu, double *zbu,
 double *time_batch, double *time_set, double *time_copy, int nd, struct stc_matrixoffsetd smd, double *d_mat,
 struct stc_1kernel_info info, int nblocks) {
  cudaMemset(zau, 0.0, sizeof(double)*nd);
  if(nblocks==0){
    cuda_matvec_1kernel0<<<st_leafmtxp->nlf,128,st_leafmtxp->ktmax*sizeof(double)>>>
      (info, smd, zau, zu, d_mat, st_leafmtxp->nlf, st_leafmtxp->ktmax);
  }else if(nblocks<0){
    cuda_matvec_1kernel0a<<<st_leafmtxp->nlf,128,st_leafmtxp->ktmax*sizeof(double)>>>
      (info, smd, zau, zu, d_mat, st_leafmtxp->nlf, st_leafmtxp->ktmax);
  }else{
    cuda_matvec_1kernel<<<nblocks,128,st_leafmtxp->ktmax*sizeof(double)>>>
      (info, smd, zau, zu, d_mat, st_leafmtxp->nlf, st_leafmtxp->ktmax);
  }
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
  //printf("bnorm:%e\n",bnorm);
#pragma omp parallel for
  for(i=0;i<(*nd);i++)zr[i]=b[i];
  //  .. MATVEC ..
  tic = MPI_Wtime();
  //for(i=0;i<(*nd);i++)zshdw[i]=0.0;
  c_hacapk_adot_body_lfmtx_cuda_calc(zshdw,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy,*nd);
  cudaThreadSynchronize();
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
  //printf("zrnorm:%e\n",zrnorm);
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
	cudaThreadSynchronize();
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
	//printf("znorm:%e\n",znorm);
	//printf("zden:%e\n",zden);
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
	cudaThreadSynchronize();
	time_spmv += (MPI_Wtime()-tic);
	c_hacapk_adot_cax_lfmtx_cuda_comm(zakt,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
	//
	znorm = 0.0;
#pragma omp parallel for reduction(+:znorm)
	for(i=0;i<(*nd);i++)znorm += zakt[i]*zt[i];
	zden = 0.0;
#pragma omp parallel for reduction(+:zden)
	for(i=0;i<(*nd);i++)zden += zakt[i]*zakt[i];
	//printf("znorm:%e\n",znorm);
	//printf("zden:%e\n",zden);
	zeta = znorm/zden;
	//printf("zeta:%e\n",zeta);
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
	//printf("beta:%e\n",beta);
	//printf("zrnorm:%e\n",zrnorm);
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
	      printf( "C-CUDA  %d  time_mpi    = %.5e\n", i, time_mpi );
	      printf( "C-CUDA  %d  time_matvec = %.5e\n", i, time_spmv );
	      printf( "C-CUDA  %d  >time_copy  = %.5e\n", i, time_copy );
	      printf( "C-CUDA  %d  >time_set   = %.5e\n", i, time_set );
	      printf( "C-CUDA  %d  >time_batch = %.5e\n", i, time_batch );
	      printf( "C-CUDA  %d  iteration   = %d\n", i, step);
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

extern "C"
void c_hacapk_bicgstab_cax_lfmtx_cuda2_
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

  double *d_zz, *d_b, *d_zr, *d_zshdw, *d_u, *d_wws, *d_zrnorm;
  double *d_zp, *d_zakp, *d_zkp, *d_znorm, *d_zden, *d_zt, *d_zkt, *d_zakt, *d_beta;
  cudaMalloc(&d_zz, sizeof(double));
  cudaMalloc(&d_b, sizeof(double)*(*nd)); cudaMemcpy(d_b, b, sizeof(double)*(*nd), cudaMemcpyHostToDevice);
  cudaMalloc(&d_zr, sizeof(double)*(*nd));
  cudaMalloc(&d_zshdw, sizeof(double)*(*nd));
  cudaMalloc(&d_u, sizeof(double)*(*nd)); cudaMemcpy(d_u, u, sizeof(double)*(*nd), cudaMemcpyHostToDevice);
  cudaMalloc(&d_wws, sizeof(double)*(*nd));
  cudaMalloc(&d_zrnorm, sizeof(double));
  cudaMalloc(&d_zp, sizeof(double)*(*nd));
  cudaMalloc(&d_zakp, sizeof(double)*(*nd));
  cudaMalloc(&d_zkp, sizeof(double)*(*nd));
  cudaMalloc(&d_znorm, sizeof(double));
  cudaMalloc(&d_zden, sizeof(double));
  cudaMalloc(&d_zt, sizeof(double)*(*nd));
  cudaMalloc(&d_zakt, sizeof(double)*(*nd));
  cudaMalloc(&d_zkt, sizeof(double)*(*nd));
  cudaMalloc(&d_beta, sizeof(double));

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
  //#pragma omp parallel for reduction(+:zz)
  //for(i=0;i<(*nd);i++){zz += b[i]*b[i];}
  myCudaReductionZero<128><<<112,128>>>(d_zz, d_b, (*nd));
  cudaMemcpy(&zz,d_zz,sizeof(double),cudaMemcpyDeviceToHost);
  bnorm=sqrt(zz);
  //printf("bnorm:%e\n",bnorm);
  //#pragma omp parallel for
  //for(i=0;i<(*nd);i++)zr[i]=b[i];
  cudaMemcpy(d_zr,d_b,sizeof(double)*(*nd),cudaMemcpyDeviceToDevice);
  //  .. MATVEC ..
  tic = MPI_Wtime();
  //for(i=0;i<(*nd);i++)zshdw[i]=0.0;
  //c_hacapk_adot_body_lfmtx_cuda_calc(zshdw,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy,*nd);
  c_hacapk_adot_body_lfmtx_cuda_calc_device(d_zshdw,st_leafmtxp,d_u, wws,&time_batch,&time_set,&time_copy,*nd);
  cudaThreadSynchronize();
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
  //c_hacapk_adot_cax_lfmtx_cuda_comm(zshdw, st_ctl, wws, wwr, isct, irct, *nd, &time_mpi);
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
  /*
#pragma omp parallel for
  for(i=0;i<(*nd);i++){
    zr[i]+=mone*zshdw[i];
    zshdw[i]=zr[i];
  }
  */
  myCudaFunc1<<<112,128>>>(d_zshdw,d_zr,(*nd));
  /*
  zrnorm = 0.0;
#pragma omp parallel for reduction(+:zrnorm)
  for(i=0;i<(*nd);i++)zrnorm += zr[i]*zr[i];
  */
  myCudaReductionZero<128><<<112,128>>>(d_zrnorm, d_zr, (*nd));
  cudaMemcpy(&zrnorm,d_zrnorm,sizeof(double),cudaMemcpyDeviceToHost);
  zrnorm = sqrt(zrnorm);
  //printf("zrnorm:%e\n",zrnorm);
  //return;
  if (mpinr == 0) {
	printf( "\n ** BICG (c version, CUDA2) **\n" );
	printf( "\nOriginal relative residual norm = %.2e/%.2e = %.2e\n",zrnorm,bnorm,zrnorm/bnorm );
	printf( "c_HACApK_bicgstab_cax_lfmtx_cuda2 start\n" );
  }
  for ( step=1; step<=mstep; step++ ) {
    //for(step=1; step<=1; step++){
    if (zrnorm/bnorm < eps) break;
    // zp(:nd) = zr(:nd) + beta*(zp(:nd) - zeta*zakp(:nd))
    if (beta == zero) {
      //#pragma omp parallel for
      //for(i=0;i<(*nd);i++)zp[i]=zr[i];
      cudaMemcpy(d_zp,d_zr,sizeof(double)*(*nd), cudaMemcpyDeviceToDevice);
    } else {
      /*
#pragma omp parallel for
	  for(i=0;i<(*nd);i++){
	    zp[i] = zr[i] + beta * (zp[i] + zeta*zakp[i]);
	  }
      */
      myCudaFunc2<<<112,128>>>(d_zp,d_zr,d_zakp,beta,zeta,(*nd));
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
    //#pragma omp parallel for
    //for(i=0;i<(*nd);i++)zkp[i]=zp[i];
    cudaMemcpy(d_zkp,d_zp,sizeof(double)*(*nd),cudaMemcpyDeviceToDevice);
    //  .. MATVEC ..
    //for(i=0;i<(*nd);i++)zakp[i]=0.0;
    tic = MPI_Wtime();
    //c_hacapk_adot_body_lfmtx_cuda_calc(zakp,st_leafmtxp,zkp,wws, &time_batch,&time_set,&time_copy,*nd);
    c_hacapk_adot_body_lfmtx_cuda_calc_device(d_zakp,st_leafmtxp,d_zkp,wws, &time_batch,&time_set,&time_copy,*nd);
    cudaThreadSynchronize();
    time_spmv += (MPI_Wtime()-tic);
    /*
      {
	  FILE *F;
	  F=fopen("cuda-zakp.dat","w");
	  for(i=0;i<(*nd);i++)fprintf(F,"%e\n", zakp[i]);
	  fclose(F);
	  }
    */
    //c_hacapk_adot_cax_lfmtx_cuda_comm(zakp,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
    //
    /*
	znorm = 0.0;
#pragma omp parallel for reduction(+:znorm)
	for(i=0;i<(*nd);i++)znorm += zshdw[i]*zr[i];
	zden = 0.0;
#pragma omp parallel for reduction(+:zden)
	for(i=0;i<(*nd);i++)zden += zshdw[i]*zakp[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_znorm,d_zshdw,d_zr,(*nd));
    myCudaReductionZero<128><<<112,128>>>(d_zden,d_zshdw,d_zakp,(*nd));
    /*
      {
	  FILE *F;
	  F=fopen("cuda.dat","w");
	  fprintf(F,"%e %e\n",znorm, zden);
	  fclose(F);
	  }
    */
    cudaMemcpy(&znorm,d_znorm,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(&zden,d_zden,sizeof(double),cudaMemcpyDeviceToHost);
    //printf("znorm:%e\n",znorm);
    //printf("zden:%e\n",zden);
    alpha = -znorm/zden;
    znormold = znorm;
    // zt(:nd) = zr(:nd) - alpha*zakp(:nd)
    //#pragma omp parallel for
    //for(i=0;i<(*nd);i++)zt[i]=zr[i]+alpha*zakp[i];
    myCudaFunc3<<<112,128>>>(d_zt,d_zr,alpha,d_zakp,*nd);
    alpha = -alpha;
    // zkt(:nd) = zt(:nd)
    //#pragma omp parallel for
    //for(i=0;i<(*nd);i++)zkt[i]=zt[i];
    cudaMemcpy(d_zkt,d_zt,sizeof(double)*(*nd),cudaMemcpyDeviceToDevice);
    //  .. MATVEC ..
    //for(i=0;i<(*nd);i++)zakt[i]=0.0;
    tic = MPI_Wtime();
    //c_hacapk_adot_body_lfmtx_cuda_calc(zakt,st_leafmtxp,zkt,wws, &time_batch,&time_set,&time_copy,*nd);
    c_hacapk_adot_body_lfmtx_cuda_calc_device(d_zakt,st_leafmtxp,d_zkt,wws, &time_batch,&time_set,&time_copy,*nd);
    cudaThreadSynchronize();
    time_spmv += (MPI_Wtime()-tic);
    //c_hacapk_adot_cax_lfmtx_cuda_comm(zakt,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
    //
    /*
	znorm = 0.0;
#pragma omp parallel for reduction(+:znorm)
	for(i=0;i<(*nd);i++)znorm += zakt[i]*zt[i];
	zden = 0.0;
#pragma omp parallel for reduction(+:zden)
	for(i=0;i<(*nd);i++)zden += zakt[i]*zakt[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_znorm,d_zakt,d_zt,(*nd));
    myCudaReductionZero<128><<<112,128>>>(d_zden,d_zakt,(*nd));
    cudaMemcpy(&znorm,d_znorm,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(&zden,d_zden,sizeof(double),cudaMemcpyDeviceToHost);
    //printf("znorm:%e\n",znorm);
    //printf("zden:%e\n",zden);
    zeta = znorm/zden;
    //printf("zeta:%e\n",zeta);
    /*
      {
	  FILE *F;
	  F=fopen("cuda.dat","a");
	  fprintf(F,"%e %e\n",znorm, zden);
	  fclose(F);
	  }
    */
    // u(:nd) = u(:nd) + alpha*zkp(:nd) + zeta*zkt(:nd)
    /*
#pragma omp parallel for
	for(i=0;i<(*nd);i++){
	  u[i] += alpha*zkp[i] + zeta*zkt[i];
	}
    */
    myCudaFunc4<<<112,128>>>(d_u,alpha,d_zkp,zeta,d_zkt,*nd);
    // zr(:nd) = zt(:nd) - zeta*zakt(:nd)
    zeta = -zeta;
    /*
#pragma omp parallel
	for(i=0;i<(*nd);i++){
	  zr[i]=zt[i] + zeta*zakt[i];
	}
    */
    myCudaFunc5<<<112,128>>>(d_zr,d_zt,zeta,d_zakt,*nd);
    // beta = alpha/zeta * HACApK_dotp_d(nd,zshdw,zr)/znormold;
    /*
	beta = 0.0;
#pragma omp parallel for reduction(+:beta)
	for(i=0;i<(*nd);i++)beta += zshdw[i]*zr[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_beta,d_zshdw,d_zr,*nd);
    cudaMemcpy(&beta,d_beta,sizeof(double),cudaMemcpyDeviceToHost);
    beta = -alpha/zeta * beta/znormold;
    /*
	zrnorm = 0.0;
#pragma omp parallel for reduction(+:zrnorm)
	for(i=0;i<(*nd);i++)zrnorm += zr[i]*zr[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_zrnorm,d_zr,*nd);
    cudaMemcpy(&zrnorm,d_zrnorm,sizeof(double),cudaMemcpyDeviceToHost);
    //printf("beta:%e\n",beta);
    //printf("zrnorm:%e\n",zrnorm);
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
	printf( "C-CUDA2  %d  BiCG        = %.5e\n", i, time );
	printf( "C-CUDA2  %d  time_mpi    = %.5e\n", i, time_mpi );
	printf( "C-CUDA2  %d  time_matvec = %.5e\n", i, time_spmv );
	printf( "C-CUDA2  %d  >time_copy  = %.5e\n", i, time_copy );
	printf( "C-CUDA2  %d  >time_set   = %.5e\n", i, time_set );
	printf( "C-CUDA2  %d  >time_batch = %.5e\n", i, time_batch );
	printf( "C-CUDA2  %d  iteration   = %d\n", i, step );
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

extern "C"
void c_hacapk_bicgstab_cax_lfmtx_cuda3_
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
  struct stc_matrixoffset *smo;
  double *d_mat;
  int len;

  double *d_zz, *d_b, *d_zr, *d_zshdw, *d_u, *d_wws, *d_zrnorm;
  double *d_zp, *d_zakp, *d_zkp, *d_znorm, *d_zden, *d_zt, *d_zkt, *d_zakt, *d_beta;
  cudaMalloc(&d_zz, sizeof(double));
  cudaMalloc(&d_b, sizeof(double)*(*nd)); cudaMemcpy(d_b, b, sizeof(double)*(*nd), cudaMemcpyHostToDevice);
  cudaMalloc(&d_zr, sizeof(double)*(*nd));
  cudaMalloc(&d_zshdw, sizeof(double)*(*nd));
  cudaMalloc(&d_u, sizeof(double)*(*nd)); cudaMemcpy(d_u, u, sizeof(double)*(*nd), cudaMemcpyHostToDevice);
  cudaMalloc(&d_wws, sizeof(double)*(*nd));
  cudaMalloc(&d_zrnorm, sizeof(double));
  cudaMalloc(&d_zp, sizeof(double)*(*nd));
  cudaMalloc(&d_zakp, sizeof(double)*(*nd));
  cudaMalloc(&d_zkp, sizeof(double)*(*nd));
  cudaMalloc(&d_znorm, sizeof(double));
  cudaMalloc(&d_zden, sizeof(double));
  cudaMalloc(&d_zt, sizeof(double)*(*nd));
  cudaMalloc(&d_zakt, sizeof(double)*(*nd));
  cudaMalloc(&d_zkt, sizeof(double)*(*nd));
  cudaMalloc(&d_beta, sizeof(double));

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

  len = getAllLength(st_leafmtxp);
  cudaMalloc(&d_mat, sizeof(double)*len);
  smo=(struct stc_matrixoffset *)malloc(sizeof(struct stc_matrixoffset)*st_leafmtxp->nlf);
  myCudaCopyAll(smo, st_leafmtxp, d_mat);

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
  //#pragma omp parallel for reduction(+:zz)
  //for(i=0;i<(*nd);i++){zz += b[i]*b[i];}
  myCudaReductionZero<128><<<112,128>>>(d_zz, d_b, (*nd));
  cudaMemcpy(&zz,d_zz,sizeof(double),cudaMemcpyDeviceToHost);
  bnorm=sqrt(zz);
  //printf("bnorm:%e\n",bnorm);
  //#pragma omp parallel for
  //for(i=0;i<(*nd);i++)zr[i]=b[i];
  cudaMemcpy(d_zr,d_b,sizeof(double)*(*nd),cudaMemcpyDeviceToDevice);
  //  .. MATVEC ..
  tic = MPI_Wtime();
  //for(i=0;i<(*nd);i++)zshdw[i]=0.0;
  //c_hacapk_adot_body_lfmtx_cuda_calc(zshdw,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy,*nd);
  c_hacapk_adot_body_lfmtx_cuda_calc_device_smo
    (d_zshdw,st_leafmtxp,d_u, wws,&time_batch,&time_set,&time_copy,*nd, smo, d_mat);
  cudaThreadSynchronize();
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
  //c_hacapk_adot_cax_lfmtx_cuda_comm(zshdw, st_ctl, wws, wwr, isct, irct, *nd, &time_mpi);
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
  /*
#pragma omp parallel for
  for(i=0;i<(*nd);i++){
    zr[i]+=mone*zshdw[i];
    zshdw[i]=zr[i];
  }
  */
  myCudaFunc1<<<112,128>>>(d_zshdw,d_zr,(*nd));
  /*
  zrnorm = 0.0;
#pragma omp parallel for reduction(+:zrnorm)
  for(i=0;i<(*nd);i++)zrnorm += zr[i]*zr[i];
  */
  myCudaReductionZero<128><<<112,128>>>(d_zrnorm, d_zr, (*nd));
  cudaMemcpy(&zrnorm,d_zrnorm,sizeof(double),cudaMemcpyDeviceToHost);
  zrnorm = sqrt(zrnorm);
  //printf("zrnorm:%e\n",zrnorm);
  //return;
  if (mpinr == 0) {
	printf( "\n ** BICG (c version, CUDA3) **\n" );
	printf( "\nOriginal relative residual norm = %.2e/%.2e = %.2e\n",zrnorm,bnorm,zrnorm/bnorm );
	printf( "c_HACApK_bicgstab_cax_lfmtx_cuda3 start\n" );
  }
  for ( step=1; step<=mstep; step++ ) {
    //for(step=1; step<=1; step++){
    if (zrnorm/bnorm < eps) break;
    // zp(:nd) = zr(:nd) + beta*(zp(:nd) - zeta*zakp(:nd))
    if (beta == zero) {
      //#pragma omp parallel for
      //for(i=0;i<(*nd);i++)zp[i]=zr[i];
      cudaMemcpy(d_zp,d_zr,sizeof(double)*(*nd), cudaMemcpyDeviceToDevice);
    } else {
      /*
#pragma omp parallel for
	  for(i=0;i<(*nd);i++){
	    zp[i] = zr[i] + beta * (zp[i] + zeta*zakp[i]);
	  }
      */
      myCudaFunc2<<<112,128>>>(d_zp,d_zr,d_zakp,beta,zeta,(*nd));
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
    //#pragma omp parallel for
    //for(i=0;i<(*nd);i++)zkp[i]=zp[i];
    cudaMemcpy(d_zkp,d_zp,sizeof(double)*(*nd),cudaMemcpyDeviceToDevice);
    //  .. MATVEC ..
    //for(i=0;i<(*nd);i++)zakp[i]=0.0;
    tic = MPI_Wtime();
    //c_hacapk_adot_body_lfmtx_cuda_calc(zakp,st_leafmtxp,zkp,wws, &time_batch,&time_set,&time_copy,*nd);
    c_hacapk_adot_body_lfmtx_cuda_calc_device_smo
      (d_zakp,st_leafmtxp,d_zkp,wws, &time_batch,&time_set,&time_copy,*nd, smo, d_mat);
    cudaThreadSynchronize();
    time_spmv += (MPI_Wtime()-tic);
    /*
      {
	  FILE *F;
	  F=fopen("cuda-zakp.dat","w");
	  for(i=0;i<(*nd);i++)fprintf(F,"%e\n", zakp[i]);
	  fclose(F);
	  }
    */
    //c_hacapk_adot_cax_lfmtx_cuda_comm(zakp,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
    //
    /*
	znorm = 0.0;
#pragma omp parallel for reduction(+:znorm)
	for(i=0;i<(*nd);i++)znorm += zshdw[i]*zr[i];
	zden = 0.0;
#pragma omp parallel for reduction(+:zden)
	for(i=0;i<(*nd);i++)zden += zshdw[i]*zakp[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_znorm,d_zshdw,d_zr,(*nd));
    myCudaReductionZero<128><<<112,128>>>(d_zden,d_zshdw,d_zakp,(*nd));
    /*
      {
	  FILE *F;
	  F=fopen("cuda.dat","w");
	  fprintf(F,"%e %e\n",znorm, zden);
	  fclose(F);
	  }
    */
    cudaMemcpy(&znorm,d_znorm,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(&zden,d_zden,sizeof(double),cudaMemcpyDeviceToHost);
    //printf("znorm:%e\n",znorm);
    //printf("zden:%e\n",zden);
    alpha = -znorm/zden;
    znormold = znorm;
    // zt(:nd) = zr(:nd) - alpha*zakp(:nd)
    //#pragma omp parallel for
    //for(i=0;i<(*nd);i++)zt[i]=zr[i]+alpha*zakp[i];
    myCudaFunc3<<<112,128>>>(d_zt,d_zr,alpha,d_zakp,*nd);
    alpha = -alpha;
    // zkt(:nd) = zt(:nd)
    //#pragma omp parallel for
    //for(i=0;i<(*nd);i++)zkt[i]=zt[i];
    cudaMemcpy(d_zkt,d_zt,sizeof(double)*(*nd),cudaMemcpyDeviceToDevice);
    //  .. MATVEC ..
    //for(i=0;i<(*nd);i++)zakt[i]=0.0;
    tic = MPI_Wtime();
    //c_hacapk_adot_body_lfmtx_cuda_calc(zakt,st_leafmtxp,zkt,wws, &time_batch,&time_set,&time_copy,*nd);
    c_hacapk_adot_body_lfmtx_cuda_calc_device_smo
      (d_zakt,st_leafmtxp,d_zkt,wws, &time_batch,&time_set,&time_copy,*nd, smo, d_mat);
    cudaThreadSynchronize();
    time_spmv += (MPI_Wtime()-tic);
    //c_hacapk_adot_cax_lfmtx_cuda_comm(zakt,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
    //
    /*
	znorm = 0.0;
#pragma omp parallel for reduction(+:znorm)
	for(i=0;i<(*nd);i++)znorm += zakt[i]*zt[i];
	zden = 0.0;
#pragma omp parallel for reduction(+:zden)
	for(i=0;i<(*nd);i++)zden += zakt[i]*zakt[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_znorm,d_zakt,d_zt,(*nd));
    myCudaReductionZero<128><<<112,128>>>(d_zden,d_zakt,(*nd));
    cudaMemcpy(&znorm,d_znorm,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(&zden,d_zden,sizeof(double),cudaMemcpyDeviceToHost);
    //printf("znorm:%e\n",znorm);
    //printf("zden:%e\n",zden);
    zeta = znorm/zden;
    //printf("zeta:%e\n",zeta);
    /*
      {
	  FILE *F;
	  F=fopen("cuda.dat","a");
	  fprintf(F,"%e %e\n",znorm, zden);
	  fclose(F);
	  }
    */
    // u(:nd) = u(:nd) + alpha*zkp(:nd) + zeta*zkt(:nd)
    /*
#pragma omp parallel for
	for(i=0;i<(*nd);i++){
	  u[i] += alpha*zkp[i] + zeta*zkt[i];
	}
    */
    myCudaFunc4<<<112,128>>>(d_u,alpha,d_zkp,zeta,d_zkt,*nd);
    // zr(:nd) = zt(:nd) - zeta*zakt(:nd)
    zeta = -zeta;
    /*
#pragma omp parallel
	for(i=0;i<(*nd);i++){
	  zr[i]=zt[i] + zeta*zakt[i];
	}
    */
    myCudaFunc5<<<112,128>>>(d_zr,d_zt,zeta,d_zakt,*nd);
    // beta = alpha/zeta * HACApK_dotp_d(nd,zshdw,zr)/znormold;
    /*
	beta = 0.0;
#pragma omp parallel for reduction(+:beta)
	for(i=0;i<(*nd);i++)beta += zshdw[i]*zr[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_beta,d_zshdw,d_zr,*nd);
    cudaMemcpy(&beta,d_beta,sizeof(double),cudaMemcpyDeviceToHost);
    beta = -alpha/zeta * beta/znormold;
    /*
	zrnorm = 0.0;
#pragma omp parallel for reduction(+:zrnorm)
	for(i=0;i<(*nd);i++)zrnorm += zr[i]*zr[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_zrnorm,d_zr,*nd);
    cudaMemcpy(&zrnorm,d_zrnorm,sizeof(double),cudaMemcpyDeviceToHost);
    //printf("beta:%e\n",beta);
    //printf("zrnorm:%e\n",zrnorm);
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
	printf( "C-CUDA3  %d  BiCG        = %.5e\n", i, time );
	printf( "C-CUDA3  %d  time_mpi    = %.5e\n", i, time_mpi );
	printf( "C-CUDA3  %d  time_matvec = %.5e\n", i, time_spmv );
	printf( "C-CUDA3  %d  >time_copy  = %.5e\n", i, time_copy );
	printf( "C-CUDA3  %d  >time_set   = %.5e\n", i, time_set );
	printf( "C-CUDA3  %d  >time_batch = %.5e\n", i, time_batch );
	printf( "C-CUDA3  %d  iteration   = %d\n", i, step );
      }
      MPI_Barrier( icomm );
    }
  }
    // delete matrix
    //c_hacapk_adot_body_lfdel_batch_(st_leafmtxp);
  cudaFree(d_mat);

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

extern "C"
void c_hacapk_bicgstab_cax_lfmtx_cuda4_
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
  struct stc_matrixoffset *smo;
  double *d_mat;
  int len;

  double *d_zz, *d_b, *d_zr, *d_zshdw, *d_u, *d_wws, *d_zrnorm;
  double *d_zp, *d_zakp, *d_zkp, *d_znorm, *d_zden, *d_zt, *d_zkt, *d_zakt, *d_beta;
  cudaMalloc(&d_zz, sizeof(double));
  cudaMalloc(&d_b, sizeof(double)*(*nd)); cudaMemcpy(d_b, b, sizeof(double)*(*nd), cudaMemcpyHostToDevice);
  cudaMalloc(&d_zr, sizeof(double)*(*nd));
  cudaMalloc(&d_zshdw, sizeof(double)*(*nd));
  cudaMalloc(&d_u, sizeof(double)*(*nd)); cudaMemcpy(d_u, u, sizeof(double)*(*nd), cudaMemcpyHostToDevice);
  cudaMalloc(&d_wws, sizeof(double)*(*nd));
  cudaMalloc(&d_zrnorm, sizeof(double));
  cudaMalloc(&d_zp, sizeof(double)*(*nd));
  cudaMalloc(&d_zakp, sizeof(double)*(*nd));
  cudaMalloc(&d_zkp, sizeof(double)*(*nd));
  cudaMalloc(&d_znorm, sizeof(double));
  cudaMalloc(&d_zden, sizeof(double));
  cudaMalloc(&d_zt, sizeof(double)*(*nd));
  cudaMalloc(&d_zakt, sizeof(double)*(*nd));
  cudaMalloc(&d_zkt, sizeof(double)*(*nd));
  cudaMalloc(&d_beta, sizeof(double));

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

  len = getAllLength(st_leafmtxp);
  cudaMalloc(&d_mat, sizeof(double)*len);
  smo=(struct stc_matrixoffset *)malloc(sizeof(struct stc_matrixoffset)*st_leafmtxp->nlf);
  myCudaCopyAll(smo, st_leafmtxp, d_mat);

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
  //#pragma omp parallel for reduction(+:zz)
  //for(i=0;i<(*nd);i++){zz += b[i]*b[i];}
  myCudaReductionZero<128><<<112,128>>>(d_zz, d_b, (*nd));
  cudaMemcpy(&zz,d_zz,sizeof(double),cudaMemcpyDeviceToHost);
  bnorm=sqrt(zz);
  //printf("bnorm:%e\n",bnorm);
  //#pragma omp parallel for
  //for(i=0;i<(*nd);i++)zr[i]=b[i];
  cudaMemcpy(d_zr,d_b,sizeof(double)*(*nd),cudaMemcpyDeviceToDevice);
  //  .. MATVEC ..
  tic = MPI_Wtime();
  //for(i=0;i<(*nd);i++)zshdw[i]=0.0;
  //c_hacapk_adot_body_lfmtx_cuda_calc(zshdw,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy,*nd);
  c_hacapk_adot_body_lfmtx_cuda_calc_device_smo_1blk
    (d_zshdw,st_leafmtxp,d_u, wws,&time_batch,&time_set,&time_copy,*nd, smo, d_mat);
  cudaThreadSynchronize();
  time_spmv += (MPI_Wtime()-tic);
  if(0){
    FILE *F;
    F=fopen("cuda1blk.dat","w");
    for(i=0;i<(*nd);i++){
      fprintf(F,"%e\n",zshdw[i]);
    }
    return;
  }
  //c_hacapk_adot_cax_lfmtx_cuda_comm(zshdw, st_ctl, wws, wwr, isct, irct, *nd, &time_mpi);
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
  /*
#pragma omp parallel for
  for(i=0;i<(*nd);i++){
    zr[i]+=mone*zshdw[i];
    zshdw[i]=zr[i];
  }
  */
  myCudaFunc1<<<112,128>>>(d_zshdw,d_zr,(*nd));
  /*
  zrnorm = 0.0;
#pragma omp parallel for reduction(+:zrnorm)
  for(i=0;i<(*nd);i++)zrnorm += zr[i]*zr[i];
  */
  myCudaReductionZero<128><<<112,128>>>(d_zrnorm, d_zr, (*nd));
  cudaMemcpy(&zrnorm,d_zrnorm,sizeof(double),cudaMemcpyDeviceToHost);
  zrnorm = sqrt(zrnorm);
  //printf("zrnorm:%e\n",zrnorm);
  //return;
  if (mpinr == 0) {
	printf( "\n ** BICG (c version, CUDA4) **\n" );
	printf( "\nOriginal relative residual norm = %.2e/%.2e = %.2e\n",zrnorm,bnorm,zrnorm/bnorm );
	printf( "c_HACApK_bicgstab_cax_lfmtx_cuda4 start\n" );
  }
  for ( step=1; step<=mstep; step++ ) {
    //for(step=1; step<=1; step++){
    if (zrnorm/bnorm < eps) break;
    // zp(:nd) = zr(:nd) + beta*(zp(:nd) - zeta*zakp(:nd))
    if (beta == zero) {
      //#pragma omp parallel for
      //for(i=0;i<(*nd);i++)zp[i]=zr[i];
      cudaMemcpy(d_zp,d_zr,sizeof(double)*(*nd), cudaMemcpyDeviceToDevice);
    } else {
      /*
#pragma omp parallel for
	  for(i=0;i<(*nd);i++){
	    zp[i] = zr[i] + beta * (zp[i] + zeta*zakp[i]);
	  }
      */
      myCudaFunc2<<<112,128>>>(d_zp,d_zr,d_zakp,beta,zeta,(*nd));
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
    //#pragma omp parallel for
    //for(i=0;i<(*nd);i++)zkp[i]=zp[i];
    cudaMemcpy(d_zkp,d_zp,sizeof(double)*(*nd),cudaMemcpyDeviceToDevice);
    //  .. MATVEC ..
    //for(i=0;i<(*nd);i++)zakp[i]=0.0;
    tic = MPI_Wtime();
    //c_hacapk_adot_body_lfmtx_cuda_calc(zakp,st_leafmtxp,zkp,wws, &time_batch,&time_set,&time_copy,*nd);
    c_hacapk_adot_body_lfmtx_cuda_calc_device_smo_1blk
      (d_zakp,st_leafmtxp,d_zkp,wws, &time_batch,&time_set,&time_copy,*nd, smo, d_mat);
    cudaThreadSynchronize();
    time_spmv += (MPI_Wtime()-tic);
    /*
      {
	  FILE *F;
	  F=fopen("cuda-zakp.dat","w");
	  for(i=0;i<(*nd);i++)fprintf(F,"%e\n", zakp[i]);
	  fclose(F);
	  }
    */
    //c_hacapk_adot_cax_lfmtx_cuda_comm(zakp,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
    //
    /*
	znorm = 0.0;
#pragma omp parallel for reduction(+:znorm)
	for(i=0;i<(*nd);i++)znorm += zshdw[i]*zr[i];
	zden = 0.0;
#pragma omp parallel for reduction(+:zden)
	for(i=0;i<(*nd);i++)zden += zshdw[i]*zakp[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_znorm,d_zshdw,d_zr,(*nd));
    myCudaReductionZero<128><<<112,128>>>(d_zden,d_zshdw,d_zakp,(*nd));
    /*
      {
	  FILE *F;
	  F=fopen("cuda.dat","w");
	  fprintf(F,"%e %e\n",znorm, zden);
	  fclose(F);
	  }
    */
    cudaMemcpy(&znorm,d_znorm,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(&zden,d_zden,sizeof(double),cudaMemcpyDeviceToHost);
    //printf("znorm:%e\n",znorm);
    //printf("zden:%e\n",zden);
    alpha = -znorm/zden;
    znormold = znorm;
    // zt(:nd) = zr(:nd) - alpha*zakp(:nd)
    //#pragma omp parallel for
    //for(i=0;i<(*nd);i++)zt[i]=zr[i]+alpha*zakp[i];
    myCudaFunc3<<<112,128>>>(d_zt,d_zr,alpha,d_zakp,*nd);
    alpha = -alpha;
    // zkt(:nd) = zt(:nd)
    //#pragma omp parallel for
    //for(i=0;i<(*nd);i++)zkt[i]=zt[i];
    cudaMemcpy(d_zkt,d_zt,sizeof(double)*(*nd),cudaMemcpyDeviceToDevice);
    //  .. MATVEC ..
    //for(i=0;i<(*nd);i++)zakt[i]=0.0;
    tic = MPI_Wtime();
    //c_hacapk_adot_body_lfmtx_cuda_calc(zakt,st_leafmtxp,zkt,wws, &time_batch,&time_set,&time_copy,*nd);
    c_hacapk_adot_body_lfmtx_cuda_calc_device_smo_1blk
      (d_zakt,st_leafmtxp,d_zkt,wws, &time_batch,&time_set,&time_copy,*nd, smo, d_mat);
    cudaThreadSynchronize();
    time_spmv += (MPI_Wtime()-tic);
    //c_hacapk_adot_cax_lfmtx_cuda_comm(zakt,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
    //
    /*
	znorm = 0.0;
#pragma omp parallel for reduction(+:znorm)
	for(i=0;i<(*nd);i++)znorm += zakt[i]*zt[i];
	zden = 0.0;
#pragma omp parallel for reduction(+:zden)
	for(i=0;i<(*nd);i++)zden += zakt[i]*zakt[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_znorm,d_zakt,d_zt,(*nd));
    myCudaReductionZero<128><<<112,128>>>(d_zden,d_zakt,(*nd));
    cudaMemcpy(&znorm,d_znorm,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(&zden,d_zden,sizeof(double),cudaMemcpyDeviceToHost);
    //printf("znorm:%e\n",znorm);
    //printf("zden:%e\n",zden);
    zeta = znorm/zden;
    //printf("zeta:%e\n",zeta);
    /*
      {
	  FILE *F;
	  F=fopen("cuda.dat","a");
	  fprintf(F,"%e %e\n",znorm, zden);
	  fclose(F);
	  }
    */
    // u(:nd) = u(:nd) + alpha*zkp(:nd) + zeta*zkt(:nd)
    /*
#pragma omp parallel for
	for(i=0;i<(*nd);i++){
	  u[i] += alpha*zkp[i] + zeta*zkt[i];
	}
    */
    myCudaFunc4<<<112,128>>>(d_u,alpha,d_zkp,zeta,d_zkt,*nd);
    // zr(:nd) = zt(:nd) - zeta*zakt(:nd)
    zeta = -zeta;
    /*
#pragma omp parallel
	for(i=0;i<(*nd);i++){
	  zr[i]=zt[i] + zeta*zakt[i];
	}
    */
    myCudaFunc5<<<112,128>>>(d_zr,d_zt,zeta,d_zakt,*nd);
    // beta = alpha/zeta * HACApK_dotp_d(nd,zshdw,zr)/znormold;
    /*
	beta = 0.0;
#pragma omp parallel for reduction(+:beta)
	for(i=0;i<(*nd);i++)beta += zshdw[i]*zr[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_beta,d_zshdw,d_zr,*nd);
    cudaMemcpy(&beta,d_beta,sizeof(double),cudaMemcpyDeviceToHost);
    beta = -alpha/zeta * beta/znormold;
    /*
	zrnorm = 0.0;
#pragma omp parallel for reduction(+:zrnorm)
	for(i=0;i<(*nd);i++)zrnorm += zr[i]*zr[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_zrnorm,d_zr,*nd);
    cudaMemcpy(&zrnorm,d_zrnorm,sizeof(double),cudaMemcpyDeviceToHost);
    //printf("beta:%e\n",beta);
    //printf("zrnorm:%e\n",zrnorm);
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
	printf( "C-CUDA4  %d  BiCG        = %.5e\n", i, time );
	printf( "C-CUDA4  %d  time_mpi    = %.5e\n", i, time_mpi );
	printf( "C-CUDA4  %d  time_matvec = %.5e\n", i, time_spmv );
	printf( "C-CUDA4  %d  >time_copy  = %.5e\n", i, time_copy );
	printf( "C-CUDA4  %d  >time_set   = %.5e\n", i, time_set );
	printf( "C-CUDA4  %d  >time_batch = %.5e\n", i, time_batch );
	printf( "C-CUDA4  %d  iteration   = %d\n", i, step );
      }
      MPI_Barrier( icomm );
    }
  }
    // delete matrix
    //c_hacapk_adot_body_lfdel_batch_(st_leafmtxp);
  cudaFree(d_mat);

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

extern "C"
void c_hacapk_bicgstab_cax_lfmtx_cuda5_
(stc_HACApK_leafmtxp *st_leafmtxp, stc_HACApK_lcontrol *st_ctl,
 double *u, double *b, double*param, int *nd, int *nstp, int *lrtrn, int *streams, int *opt, int *merge) {
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
  double time_spmv, time_mpi, time_batch, time_set, time_copy, tic, time_setmatrix;
  int i, tid;
  MPI_Comm icomm = MPI_COMM_WORLD; //lpmd[0];
  struct stc_matrixoffset *smo;
  double *d_mat;
  int len;
  cudaStream_t *s;
  double *d_zaut, **d_zbut;

  double *d_zz, *d_b, *d_zr, *d_zshdw, *d_u, *d_wws, *d_zrnorm;
  double *d_zp, *d_zakp, *d_zkp, *d_znorm, *d_zden, *d_zt, *d_zkt, *d_zakt, *d_beta;
  cudaMalloc(&d_zz, sizeof(double));
  cudaMalloc(&d_b, sizeof(double)*(*nd)); cudaMemcpy(d_b, b, sizeof(double)*(*nd), cudaMemcpyHostToDevice);
  cudaMalloc(&d_zr, sizeof(double)*(*nd));
  cudaMalloc(&d_zshdw, sizeof(double)*(*nd));
  cudaMalloc(&d_u, sizeof(double)*(*nd)); cudaMemcpy(d_u, u, sizeof(double)*(*nd), cudaMemcpyHostToDevice);
  cudaMalloc(&d_wws, sizeof(double)*(*nd));
  cudaMalloc(&d_zrnorm, sizeof(double));
  cudaMalloc(&d_zp, sizeof(double)*(*nd));
  cudaMalloc(&d_zakp, sizeof(double)*(*nd));
  cudaMalloc(&d_zkp, sizeof(double)*(*nd));
  cudaMalloc(&d_znorm, sizeof(double));
  cudaMalloc(&d_zden, sizeof(double));
  cudaMalloc(&d_zt, sizeof(double)*(*nd));
  cudaMalloc(&d_zakt, sizeof(double)*(*nd));
  cudaMalloc(&d_zkt, sizeof(double)*(*nd));
  cudaMalloc(&d_beta, sizeof(double));

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

  st_measure_time = MPI_Wtime();
  len = getAllLength(st_leafmtxp);
  cudaMalloc(&d_mat, sizeof(double)*len);
  smo=(struct stc_matrixoffset *)malloc(sizeof(struct stc_matrixoffset)*st_leafmtxp->nlf);
  myCudaCopyAll(smo, st_leafmtxp, d_mat);
  //printf("c_hacapk_bicgstab_cax_lfmtx_cuda5_: %d streams create\n", (*streams));
  s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*(*streams));
  for(i=0;i<(*streams);i++){
    cudaStreamCreate(&s[i]);
  }
  //printf("c_hacapk_bicgstab_cax_lfmtx_cuda5_: %d streams created\n", streams);
  cudaMalloc(&d_zaut, sizeof(double)*(*nd));
  d_zbut = (double**)malloc(sizeof(double*)*(*streams));
  for(i=0;i<(*streams);i++){
    cudaMalloc(&d_zbut[i], sizeof(double)*st_leafmtxp->ktmax);
  }
  time_setmatrix = MPI_Wtime() - st_measure_time;

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
  //#pragma omp parallel for reduction(+:zz)
  //for(i=0;i<(*nd);i++){zz += b[i]*b[i];}
  myCudaReductionZero<128><<<112,128>>>(d_zz, d_b, (*nd));
  cudaMemcpy(&zz,d_zz,sizeof(double),cudaMemcpyDeviceToHost);
  bnorm=sqrt(zz);
  //printf("bnorm:%e\n",bnorm);
  //#pragma omp parallel for
  //for(i=0;i<(*nd);i++)zr[i]=b[i];
  cudaMemcpy(d_zr,d_b,sizeof(double)*(*nd),cudaMemcpyDeviceToDevice);
  //  .. MATVEC ..
  tic = MPI_Wtime();
  //for(i=0;i<(*nd);i++)zshdw[i]=0.0;
  //c_hacapk_adot_body_lfmtx_cuda_calc(zshdw,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy,*nd);
  if(*opt==0){
    c_hacapk_adot_body_lfmtx_cuda_calc_device_smo_async
      (d_zshdw,st_leafmtxp,d_u, wws,&time_batch,&time_set,&time_copy,*nd, smo, d_mat, s, *streams, d_zaut, d_zbut);
  }else{
    c_hacapk_adot_body_lfmtx_cuda_calc_device_smo_async2
      (d_zshdw,st_leafmtxp,d_u, wws,&time_batch,&time_set,&time_copy,*nd, smo, d_mat, s, *streams, d_zaut, d_zbut, *merge);
  }
  cudaThreadSynchronize();
  time_spmv += (MPI_Wtime()-tic);
  //c_hacapk_adot_cax_lfmtx_cuda_comm(zshdw, st_ctl, wws, wwr, isct, irct, *nd, &time_mpi);
  //
  /*
#pragma omp parallel for
  for(i=0;i<(*nd);i++){
    zr[i]+=mone*zshdw[i];
    zshdw[i]=zr[i];
  }
  */
  myCudaFunc1<<<112,128>>>(d_zshdw,d_zr,(*nd));
  /*
  zrnorm = 0.0;
#pragma omp parallel for reduction(+:zrnorm)
  for(i=0;i<(*nd);i++)zrnorm += zr[i]*zr[i];
  */
  myCudaReductionZero<128><<<112,128>>>(d_zrnorm, d_zr, (*nd));
  cudaMemcpy(&zrnorm,d_zrnorm,sizeof(double),cudaMemcpyDeviceToHost);
  zrnorm = sqrt(zrnorm);
  //printf("zrnorm:%e\n",zrnorm);
  //return;
  if (mpinr == 0) {
    if(*opt==0){
      printf( "\n ** BICG (c version, CUDA5, async) **\n" );
    }else{
      if(*merge==0) printf( "\n ** BICG (c version, CUDA5a, async2) **\n" );
      else printf( "\n ** BICG (c version, CUDA5a, async2-merge) **\n" );
    } 
    printf( "\nOriginal relative residual norm = %.2e/%.2e = %.2e\n",zrnorm,bnorm,zrnorm/bnorm );
    printf( "c_HACApK_bicgstab_cax_lfmtx_cuda5 start\n" );
  }
  for ( step=1; step<=mstep; step++ ) {
    //for(step=1; step<=1; step++){
    if (zrnorm/bnorm < eps) break;
    // zp(:nd) = zr(:nd) + beta*(zp(:nd) - zeta*zakp(:nd))
    if (beta == zero) {
      //#pragma omp parallel for
      //for(i=0;i<(*nd);i++)zp[i]=zr[i];
      cudaMemcpy(d_zp,d_zr,sizeof(double)*(*nd), cudaMemcpyDeviceToDevice);
    } else {
      /*
#pragma omp parallel for
	  for(i=0;i<(*nd);i++){
	    zp[i] = zr[i] + beta * (zp[i] + zeta*zakp[i]);
	  }
      */
      myCudaFunc2<<<112,128>>>(d_zp,d_zr,d_zakp,beta,zeta,(*nd));
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
    //#pragma omp parallel for
    //for(i=0;i<(*nd);i++)zkp[i]=zp[i];
    cudaMemcpy(d_zkp,d_zp,sizeof(double)*(*nd),cudaMemcpyDeviceToDevice);
    //  .. MATVEC ..
    //for(i=0;i<(*nd);i++)zakp[i]=0.0;
    tic = MPI_Wtime();
    //c_hacapk_adot_body_lfmtx_cuda_calc(zakp,st_leafmtxp,zkp,wws, &time_batch,&time_set,&time_copy,*nd);
    if(*opt==0){
      c_hacapk_adot_body_lfmtx_cuda_calc_device_smo_async
	(d_zakp,st_leafmtxp,d_zkp,wws, &time_batch,&time_set,&time_copy,*nd, smo, d_mat, s, *streams, d_zaut, d_zbut);
    }else{
      c_hacapk_adot_body_lfmtx_cuda_calc_device_smo_async2
	(d_zakp,st_leafmtxp,d_zkp,wws, &time_batch,&time_set,&time_copy,*nd, smo, d_mat, s, *streams, d_zaut, d_zbut, *merge);
    }
    cudaThreadSynchronize();
    time_spmv += (MPI_Wtime()-tic);
    /*
      {
	  FILE *F;
	  F=fopen("cuda-zakp.dat","w");
	  for(i=0;i<(*nd);i++)fprintf(F,"%e\n", zakp[i]);
	  fclose(F);
	  }
    */
    //c_hacapk_adot_cax_lfmtx_cuda_comm(zakp,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
    //
    /*
	znorm = 0.0;
#pragma omp parallel for reduction(+:znorm)
	for(i=0;i<(*nd);i++)znorm += zshdw[i]*zr[i];
	zden = 0.0;
#pragma omp parallel for reduction(+:zden)
	for(i=0;i<(*nd);i++)zden += zshdw[i]*zakp[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_znorm,d_zshdw,d_zr,(*nd));
    myCudaReductionZero<128><<<112,128>>>(d_zden,d_zshdw,d_zakp,(*nd));
    /*
      {
	  FILE *F;
	  F=fopen("cuda.dat","w");
	  fprintf(F,"%e %e\n",znorm, zden);
	  fclose(F);
	  }
    */
    cudaMemcpy(&znorm,d_znorm,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(&zden,d_zden,sizeof(double),cudaMemcpyDeviceToHost);
    //printf("znorm:%e\n",znorm);
    //printf("zden:%e\n",zden);
    alpha = -znorm/zden;
    znormold = znorm;
    // zt(:nd) = zr(:nd) - alpha*zakp(:nd)
    //#pragma omp parallel for
    //for(i=0;i<(*nd);i++)zt[i]=zr[i]+alpha*zakp[i];
    myCudaFunc3<<<112,128>>>(d_zt,d_zr,alpha,d_zakp,*nd);
    alpha = -alpha;
    // zkt(:nd) = zt(:nd)
    //#pragma omp parallel for
    //for(i=0;i<(*nd);i++)zkt[i]=zt[i];
    cudaMemcpy(d_zkt,d_zt,sizeof(double)*(*nd),cudaMemcpyDeviceToDevice);
    //  .. MATVEC ..
    //for(i=0;i<(*nd);i++)zakt[i]=0.0;
    tic = MPI_Wtime();
    //c_hacapk_adot_body_lfmtx_cuda_calc(zakt,st_leafmtxp,zkt,wws, &time_batch,&time_set,&time_copy,*nd);
    if(*opt==0){
      c_hacapk_adot_body_lfmtx_cuda_calc_device_smo_async
	(d_zakt,st_leafmtxp,d_zkt,wws, &time_batch,&time_set,&time_copy,*nd, smo, d_mat, s, *streams, d_zaut, d_zbut);
    }else{
      c_hacapk_adot_body_lfmtx_cuda_calc_device_smo_async2
	(d_zakt,st_leafmtxp,d_zkt,wws, &time_batch,&time_set,&time_copy,*nd, smo, d_mat, s, *streams, d_zaut, d_zbut, *merge);
    }
    cudaThreadSynchronize();
    time_spmv += (MPI_Wtime()-tic);
    //c_hacapk_adot_cax_lfmtx_cuda_comm(zakt,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
    //
    /*
	znorm = 0.0;
#pragma omp parallel for reduction(+:znorm)
	for(i=0;i<(*nd);i++)znorm += zakt[i]*zt[i];
	zden = 0.0;
#pragma omp parallel for reduction(+:zden)
	for(i=0;i<(*nd);i++)zden += zakt[i]*zakt[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_znorm,d_zakt,d_zt,(*nd));
    myCudaReductionZero<128><<<112,128>>>(d_zden,d_zakt,(*nd));
    cudaMemcpy(&znorm,d_znorm,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(&zden,d_zden,sizeof(double),cudaMemcpyDeviceToHost);
    //printf("znorm:%e\n",znorm);
    //printf("zden:%e\n",zden);
    zeta = znorm/zden;
    //printf("zeta:%e\n",zeta);
    /*
      {
	  FILE *F;
	  F=fopen("cuda.dat","a");
	  fprintf(F,"%e %e\n",znorm, zden);
	  fclose(F);
	  }
    */
    // u(:nd) = u(:nd) + alpha*zkp(:nd) + zeta*zkt(:nd)
    /*
#pragma omp parallel for
	for(i=0;i<(*nd);i++){
	  u[i] += alpha*zkp[i] + zeta*zkt[i];
	}
    */
    myCudaFunc4<<<112,128>>>(d_u,alpha,d_zkp,zeta,d_zkt,*nd);
    // zr(:nd) = zt(:nd) - zeta*zakt(:nd)
    zeta = -zeta;
    /*
#pragma omp parallel
	for(i=0;i<(*nd);i++){
	  zr[i]=zt[i] + zeta*zakt[i];
	}
    */
    myCudaFunc5<<<112,128>>>(d_zr,d_zt,zeta,d_zakt,*nd);
    // beta = alpha/zeta * HACApK_dotp_d(nd,zshdw,zr)/znormold;
    /*
	beta = 0.0;
#pragma omp parallel for reduction(+:beta)
	for(i=0;i<(*nd);i++)beta += zshdw[i]*zr[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_beta,d_zshdw,d_zr,*nd);
    cudaMemcpy(&beta,d_beta,sizeof(double),cudaMemcpyDeviceToHost);
    beta = -alpha/zeta * beta/znormold;
    /*
	zrnorm = 0.0;
#pragma omp parallel for reduction(+:zrnorm)
	for(i=0;i<(*nd);i++)zrnorm += zr[i]*zr[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_zrnorm,d_zr,*nd);
    cudaMemcpy(&zrnorm,d_zrnorm,sizeof(double),cudaMemcpyDeviceToHost);
    //printf("beta:%e\n",beta);
    //printf("zrnorm:%e\n",zrnorm);
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
	char str[32];
	if(*opt==0)snprintf(str,32,"C-CUDA5(async)");else snprintf(str,32,"C-CUDA5a(async)");
	printf( "%s  %d  BiCG           = %.5e\n", str, i, time );
	printf( "%s  %d  time_mpi       = %.5e\n", str, i, time_mpi );
	printf( "%s  %d  time_matvec    = %.5e\n", str, i, time_spmv );
	printf( "%s  %d  >time_copy     = %.5e\n", str, i, time_copy );
	printf( "%s  %d  >time_set      = %.5e\n", str, i, time_set );
	printf( "%s  %d  >time_batch    = %.5e\n", str, i, time_batch );
	printf( "%s  %d  time_setmatrix = %.5e\n", str, i, time_setmatrix );
	printf( "%s  %d  iteration      = %d\n", str, i, step );
      }
      MPI_Barrier( icomm );
    }
  }
    // delete matrix
    //c_hacapk_adot_body_lfdel_batch_(st_leafmtxp);
  cudaFree(d_mat);
  for(i=0;i<(*streams);i++){
    cudaStreamDestroy(s[i]);
  }
  for(i=0;i<(*streams);i++)cudaFree(d_zbut[i]); free(d_zbut);
  cudaFree(d_zaut);
  cudaFree(d_mat);
  cudaFree(d_beta);
  cudaFree(d_zkt);
  cudaFree(d_zakt);
  cudaFree(d_zt);
  cudaFree(d_zden);
  cudaFree(d_znorm);
  cudaFree(d_zkp);
  cudaFree(d_zakp);
  cudaFree(d_zp);
  cudaFree(d_znorm);
  cudaFree(d_wws);
  cudaFree(d_u);
  cudaFree(d_zshdw);
  cudaFree(d_zr);
  cudaFree(d_b);
  cudaFree(d_zz);
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

// + CUBLAS
extern "C"
void c_hacapk_bicgstab_cax_lfmtx_cuda6_
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
  double time_spmv, time_mpi, time_batch, time_set, time_copy, tic, time_setmatrix;
  int i, tid;
  MPI_Comm icomm = MPI_COMM_WORLD; //lpmd[0];
  struct stc_matrixoffset *smo;
  double *d_mat;
  int len;
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  struct stc_cublasmatrices *scm;
  int ret;
  double *d_zaut, *d_zbut;

  double *d_zz, *d_b, *d_zr, *d_zshdw, *d_u, *d_wws, *d_zrnorm;
  double *d_zp, *d_zakp, *d_zkp, *d_znorm, *d_zden, *d_zt, *d_zkt, *d_zakt, *d_beta;
  cudaMalloc(&d_zz, sizeof(double));
  cudaMalloc(&d_b, sizeof(double)*(*nd)); cudaMemcpy(d_b, b, sizeof(double)*(*nd), cudaMemcpyHostToDevice);
  cudaMalloc(&d_zr, sizeof(double)*(*nd));
  cudaMalloc(&d_zshdw, sizeof(double)*(*nd));
  cudaMalloc(&d_u, sizeof(double)*(*nd)); cudaMemcpy(d_u, u, sizeof(double)*(*nd), cudaMemcpyHostToDevice);
  cudaMalloc(&d_wws, sizeof(double)*(*nd));
  cudaMalloc(&d_zrnorm, sizeof(double));
  cudaMalloc(&d_zp, sizeof(double)*(*nd));
  cudaMalloc(&d_zakp, sizeof(double)*(*nd));
  cudaMalloc(&d_zkp, sizeof(double)*(*nd));
  cudaMalloc(&d_znorm, sizeof(double));
  cudaMalloc(&d_zden, sizeof(double));
  cudaMalloc(&d_zt, sizeof(double)*(*nd));
  cudaMalloc(&d_zakt, sizeof(double)*(*nd));
  cudaMalloc(&d_zkt, sizeof(double)*(*nd));
  cudaMalloc(&d_beta, sizeof(double));

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

  st_measure_time = MPI_Wtime();

  len = getAllLength(st_leafmtxp);
  cudaMalloc(&d_mat, sizeof(double)*len);
  smo=(struct stc_matrixoffset *)malloc(sizeof(struct stc_matrixoffset)*st_leafmtxp->nlf);
  myCudaCopyAll(smo, st_leafmtxp, d_mat);

  scm=(struct stc_cublasmatrices *)malloc(sizeof(struct stc_cublasmatrices)*st_leafmtxp->nlf);
  myCublasMakeMatrix(st_leafmtxp, scm);
  stat = cublasCreate(&handle);
  if(stat!=cudaSuccess){
    printf("cublasCreate failed\n");
    return;
  }
  ret = myCublasSetMatrix(st_leafmtxp, scm);
  if(ret!=0)return;
  cudaMalloc(&d_zaut, sizeof(double)*(*nd));
  cudaMalloc(&d_zbut, sizeof(double)*st_leafmtxp->ktmax);
  time_setmatrix = MPI_Wtime() - st_measure_time;

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
  //#pragma omp parallel for reduction(+:zz)
  //for(i=0;i<(*nd);i++){zz += b[i]*b[i];}
  myCudaReductionZero<128><<<112,128>>>(d_zz, d_b, (*nd));
  cudaMemcpy(&zz,d_zz,sizeof(double),cudaMemcpyDeviceToHost);
  bnorm=sqrt(zz);
  //printf("bnorm:%e\n",bnorm);
  //#pragma omp parallel for
  //for(i=0;i<(*nd);i++)zr[i]=b[i];
  cudaMemcpy(d_zr,d_b,sizeof(double)*(*nd),cudaMemcpyDeviceToDevice);
  //  .. MATVEC ..
  tic = MPI_Wtime();
  //for(i=0;i<(*nd);i++)zshdw[i]=0.0;
  //c_hacapk_adot_body_lfmtx_cuda_calc(zshdw,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy,*nd);
  c_hacapk_adot_body_lfmtx_cuda_calc_device_smo_cublas
    (d_zshdw,st_leafmtxp,d_u, wws,&time_batch,&time_set,&time_copy,*nd, smo, handle, scm, d_mat, d_zaut, d_zbut);
  cudaThreadSynchronize();
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
  //c_hacapk_adot_cax_lfmtx_cuda_comm(zshdw, st_ctl, wws, wwr, isct, irct, *nd, &time_mpi);
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
  /*
#pragma omp parallel for
  for(i=0;i<(*nd);i++){
    zr[i]+=mone*zshdw[i];
    zshdw[i]=zr[i];
  }
  */
  myCudaFunc1<<<112,128>>>(d_zshdw,d_zr,(*nd));
  /*
  zrnorm = 0.0;
#pragma omp parallel for reduction(+:zrnorm)
  for(i=0;i<(*nd);i++)zrnorm += zr[i]*zr[i];
  */
  myCudaReductionZero<128><<<112,128>>>(d_zrnorm, d_zr, (*nd));
  cudaMemcpy(&zrnorm,d_zrnorm,sizeof(double),cudaMemcpyDeviceToHost);
  zrnorm = sqrt(zrnorm);
  //printf("zrnorm:%e\n",zrnorm);
  //return;
  if (mpinr == 0) {
	printf( "\n ** BICG (c version, CUDA6) **\n" );
	printf( "\nOriginal relative residual norm = %.2e/%.2e = %.2e\n",zrnorm,bnorm,zrnorm/bnorm );
	printf( "c_HACApK_bicgstab_cax_lfmtx_cuda6 start\n" );
  }
  for ( step=1; step<=mstep; step++ ) {
    //for(step=1; step<=1; step++){
    if (zrnorm/bnorm < eps) break;
    // zp(:nd) = zr(:nd) + beta*(zp(:nd) - zeta*zakp(:nd))
    if (beta == zero) {
      //#pragma omp parallel for
      //for(i=0;i<(*nd);i++)zp[i]=zr[i];
      cudaMemcpy(d_zp,d_zr,sizeof(double)*(*nd), cudaMemcpyDeviceToDevice);
    } else {
      /*
#pragma omp parallel for
	  for(i=0;i<(*nd);i++){
	    zp[i] = zr[i] + beta * (zp[i] + zeta*zakp[i]);
	  }
      */
      myCudaFunc2<<<112,128>>>(d_zp,d_zr,d_zakp,beta,zeta,(*nd));
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
    //#pragma omp parallel for
    //for(i=0;i<(*nd);i++)zkp[i]=zp[i];
    cudaMemcpy(d_zkp,d_zp,sizeof(double)*(*nd),cudaMemcpyDeviceToDevice);
    //  .. MATVEC ..
    //for(i=0;i<(*nd);i++)zakp[i]=0.0;
    tic = MPI_Wtime();
    //c_hacapk_adot_body_lfmtx_cuda_calc(zakp,st_leafmtxp,zkp,wws, &time_batch,&time_set,&time_copy,*nd);
    c_hacapk_adot_body_lfmtx_cuda_calc_device_smo_cublas
      (d_zakp,st_leafmtxp,d_zkp,wws, &time_batch,&time_set,&time_copy,*nd, smo, handle, scm, d_mat, d_zaut, d_zbut);
    cudaThreadSynchronize();
    time_spmv += (MPI_Wtime()-tic);
    /*
      {
	  FILE *F;
	  F=fopen("cuda-zakp.dat","w");
	  for(i=0;i<(*nd);i++)fprintf(F,"%e\n", zakp[i]);
	  fclose(F);
	  }
    */
    //c_hacapk_adot_cax_lfmtx_cuda_comm(zakp,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
    //
    /*
	znorm = 0.0;
#pragma omp parallel for reduction(+:znorm)
	for(i=0;i<(*nd);i++)znorm += zshdw[i]*zr[i];
	zden = 0.0;
#pragma omp parallel for reduction(+:zden)
	for(i=0;i<(*nd);i++)zden += zshdw[i]*zakp[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_znorm,d_zshdw,d_zr,(*nd));
    myCudaReductionZero<128><<<112,128>>>(d_zden,d_zshdw,d_zakp,(*nd));
    /*
      {
	  FILE *F;
	  F=fopen("cuda.dat","w");
	  fprintf(F,"%e %e\n",znorm, zden);
	  fclose(F);
	  }
    */
    cudaMemcpy(&znorm,d_znorm,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(&zden,d_zden,sizeof(double),cudaMemcpyDeviceToHost);
    //printf("znorm:%e\n",znorm);
    //printf("zden:%e\n",zden);
    alpha = -znorm/zden;
    znormold = znorm;
    // zt(:nd) = zr(:nd) - alpha*zakp(:nd)
    //#pragma omp parallel for
    //for(i=0;i<(*nd);i++)zt[i]=zr[i]+alpha*zakp[i];
    myCudaFunc3<<<112,128>>>(d_zt,d_zr,alpha,d_zakp,*nd);
    alpha = -alpha;
    // zkt(:nd) = zt(:nd)
    //#pragma omp parallel for
    //for(i=0;i<(*nd);i++)zkt[i]=zt[i];
    cudaMemcpy(d_zkt,d_zt,sizeof(double)*(*nd),cudaMemcpyDeviceToDevice);
    //  .. MATVEC ..
    //for(i=0;i<(*nd);i++)zakt[i]=0.0;
    tic = MPI_Wtime();
    //c_hacapk_adot_body_lfmtx_cuda_calc(zakt,st_leafmtxp,zkt,wws, &time_batch,&time_set,&time_copy,*nd);
    c_hacapk_adot_body_lfmtx_cuda_calc_device_smo_cublas
      (d_zakt,st_leafmtxp,d_zkt,wws, &time_batch,&time_set,&time_copy,*nd, smo, handle, scm, d_mat, d_zaut, d_zbut);
    cudaThreadSynchronize();
    time_spmv += (MPI_Wtime()-tic);
    //c_hacapk_adot_cax_lfmtx_cuda_comm(zakt,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
    //
    /*
	znorm = 0.0;
#pragma omp parallel for reduction(+:znorm)
	for(i=0;i<(*nd);i++)znorm += zakt[i]*zt[i];
	zden = 0.0;
#pragma omp parallel for reduction(+:zden)
	for(i=0;i<(*nd);i++)zden += zakt[i]*zakt[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_znorm,d_zakt,d_zt,(*nd));
    myCudaReductionZero<128><<<112,128>>>(d_zden,d_zakt,(*nd));
    cudaMemcpy(&znorm,d_znorm,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(&zden,d_zden,sizeof(double),cudaMemcpyDeviceToHost);
    //printf("znorm:%e\n",znorm);
    //printf("zden:%e\n",zden);
    zeta = znorm/zden;
    //printf("zeta:%e\n",zeta);
    /*
      {
	  FILE *F;
	  F=fopen("cuda.dat","a");
	  fprintf(F,"%e %e\n",znorm, zden);
	  fclose(F);
	  }
    */
    // u(:nd) = u(:nd) + alpha*zkp(:nd) + zeta*zkt(:nd)
    /*
#pragma omp parallel for
	for(i=0;i<(*nd);i++){
	  u[i] += alpha*zkp[i] + zeta*zkt[i];
	}
    */
    myCudaFunc4<<<112,128>>>(d_u,alpha,d_zkp,zeta,d_zkt,*nd);
    // zr(:nd) = zt(:nd) - zeta*zakt(:nd)
    zeta = -zeta;
    /*
#pragma omp parallel
	for(i=0;i<(*nd);i++){
	  zr[i]=zt[i] + zeta*zakt[i];
	}
    */
    myCudaFunc5<<<112,128>>>(d_zr,d_zt,zeta,d_zakt,*nd);
    // beta = alpha/zeta * HACApK_dotp_d(nd,zshdw,zr)/znormold;
    /*
	beta = 0.0;
#pragma omp parallel for reduction(+:beta)
	for(i=0;i<(*nd);i++)beta += zshdw[i]*zr[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_beta,d_zshdw,d_zr,*nd);
    cudaMemcpy(&beta,d_beta,sizeof(double),cudaMemcpyDeviceToHost);
    beta = -alpha/zeta * beta/znormold;
    /*
	zrnorm = 0.0;
#pragma omp parallel for reduction(+:zrnorm)
	for(i=0;i<(*nd);i++)zrnorm += zr[i]*zr[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_zrnorm,d_zr,*nd);
    cudaMemcpy(&zrnorm,d_zrnorm,sizeof(double),cudaMemcpyDeviceToHost);
    //printf("beta:%e\n",beta);
    //printf("zrnorm:%e\n",zrnorm);
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
	char str[16];
	strncpy(str, "C-CUDA6(CUBLAS)", 16);
	printf( "%s  %d  BiCG           = %.5e\n", str, i, time );
	printf( "%s  %d  time_mpi       = %.5e\n", str, i, time_mpi );
	printf( "%s  %d  time_matvec    = %.5e\n", str, i, time_spmv );
	printf( "%s  %d  >time_copy     = %.5e\n", str, i, time_copy );
	printf( "%s  %d  >time_set      = %.5e\n", str, i, time_set );
	printf( "%s  %d  >time_batch    = %.5e\n", str, i, time_batch );
	printf( "%s  %d  time_setmatrix = %.5e\n", str, i, time_setmatrix );
	printf( "%s  %d  iteration      = %d\n", str, i, step );
      }
      MPI_Barrier( icomm );
    }
  }
    // delete matrix
    //c_hacapk_adot_body_lfdel_batch_(st_leafmtxp);
  cudaFree(d_zaut); cudaFree(d_zbut);
  myCublasDestroyMatrix(smo,st_leafmtxp,scm);
  cublasDestroy(handle);
  cudaFree(d_mat);
  cudaFree(d_beta);
  cudaFree(d_zkt);
  cudaFree(d_zakt);
  cudaFree(d_zt);
  cudaFree(d_zden);
  cudaFree(d_znorm);
  cudaFree(d_zkp);
  cudaFree(d_zakp);
  cudaFree(d_zp);
  cudaFree(d_znorm);
  cudaFree(d_wws);
  cudaFree(d_u);
  cudaFree(d_zshdw);
  cudaFree(d_zr);
  cudaFree(d_b);
  cudaFree(d_zz);
  free(scm);
  free(smo);

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

extern "C"
void c_hacapk_bicgstab_cax_lfmtx_cuda7_
(stc_HACApK_leafmtxp *st_leafmtxp, stc_HACApK_lcontrol *st_ctl,
 double *u, double *b, double*param, int *nd, int *nstp, int *lrtrn, int *blocks) {
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
  double time_spmv, time_mpi, time_batch, time_set, time_copy, tic, time_setmatrix;
  int i, tid;
  MPI_Comm icomm = MPI_COMM_WORLD; //lpmd[0];
  struct stc_matrixoffset *smo;
  struct stc_matrixoffsetd smd;
  double *d_mat;
  int len;
  struct stc_1kernel_info s1info;

  double *d_zz, *d_b, *d_zr, *d_zshdw, *d_u, *d_wws, *d_zrnorm;
  double *d_zp, *d_zakp, *d_zkp, *d_znorm, *d_zden, *d_zt, *d_zkt, *d_zakt, *d_beta;
  cudaMalloc(&d_zz, sizeof(double));
  cudaMalloc(&d_b, sizeof(double)*(*nd)); cudaMemcpy(d_b, b, sizeof(double)*(*nd), cudaMemcpyHostToDevice);
  cudaMalloc(&d_zr, sizeof(double)*(*nd));
  cudaMalloc(&d_zshdw, sizeof(double)*(*nd));
  cudaMalloc(&d_u, sizeof(double)*(*nd)); cudaMemcpy(d_u, u, sizeof(double)*(*nd), cudaMemcpyHostToDevice);
  cudaMalloc(&d_wws, sizeof(double)*(*nd));
  cudaMalloc(&d_zrnorm, sizeof(double));
  cudaMalloc(&d_zp, sizeof(double)*(*nd));
  cudaMalloc(&d_zakp, sizeof(double)*(*nd));
  cudaMalloc(&d_zkp, sizeof(double)*(*nd));
  cudaMalloc(&d_znorm, sizeof(double));
  cudaMalloc(&d_zden, sizeof(double));
  cudaMalloc(&d_zt, sizeof(double)*(*nd));
  cudaMalloc(&d_zakt, sizeof(double)*(*nd));
  cudaMalloc(&d_zkt, sizeof(double)*(*nd));
  cudaMalloc(&d_beta, sizeof(double));

  printf("DBG: %e %e %e %e\n",u[0],u[1],u[2],u[3]);
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

  st_measure_time = MPI_Wtime();
  len = getAllLength(st_leafmtxp);
  cudaMalloc(&d_mat, sizeof(double)*len);
  smo=(struct stc_matrixoffset *)malloc(sizeof(struct stc_matrixoffset)*st_leafmtxp->nlf);
  myCudaCopyAll(smo, st_leafmtxp, d_mat);
  myMake1kInfo(&s1info, st_leafmtxp, smo, &smd);
  time_setmatrix = MPI_Wtime() - st_measure_time;

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
  //#pragma omp parallel for reduction(+:zz)
  //for(i=0;i<(*nd);i++){zz += b[i]*b[i];}
  myCudaReductionZero<128><<<112,128>>>(d_zz, d_b, (*nd));
  cudaMemcpy(&zz,d_zz,sizeof(double),cudaMemcpyDeviceToHost);
  bnorm=sqrt(zz);
  //printf("bnorm:%e\n",bnorm);
  //#pragma omp parallel for
  //for(i=0;i<(*nd);i++)zr[i]=b[i];
  cudaMemcpy(d_zr,d_b,sizeof(double)*(*nd),cudaMemcpyDeviceToDevice);
  //  .. MATVEC ..
  tic = MPI_Wtime();
  //for(i=0;i<(*nd);i++)zshdw[i]=0.0;
  //c_hacapk_adot_body_lfmtx_cuda_calc(zshdw,st_leafmtxp,u,wws, &time_batch,&time_set,&time_copy,*nd);
    c_hacapk_adot_body_lfmtx_cuda_calc_device_smo_1kernel
      (d_zshdw,st_leafmtxp,d_u, wws,&time_batch,&time_set,&time_copy,*nd, smd, d_mat, s1info, *blocks);
    cudaThreadSynchronize();
  time_spmv += (MPI_Wtime()-tic);
  if(0){
    FILE *F;
    F=fopen("cuda1kernel.dat","w");
    cudaMemcpy(zshdw,d_zshdw,sizeof(double)*(*nd),cudaMemcpyDeviceToHost);
    for(i=0;i<(*nd);i++){
      fprintf(F,"%e\n",zshdw[i]);
    }
    return;
  }
  //c_hacapk_adot_cax_lfmtx_cuda_comm(zshdw, st_ctl, wws, wwr, isct, irct, *nd, &time_mpi);
  //
  /*
#pragma omp parallel for
  for(i=0;i<(*nd);i++){
    zr[i]+=mone*zshdw[i];
    zshdw[i]=zr[i];
  }
  */
  myCudaFunc1<<<112,128>>>(d_zshdw,d_zr,(*nd));
  /*
  zrnorm = 0.0;
#pragma omp parallel for reduction(+:zrnorm)
  for(i=0;i<(*nd);i++)zrnorm += zr[i]*zr[i];
  */
  myCudaReductionZero<128><<<112,128>>>(d_zrnorm, d_zr, (*nd));
  cudaMemcpy(&zrnorm,d_zrnorm,sizeof(double),cudaMemcpyDeviceToHost);
  zrnorm = sqrt(zrnorm);
  //printf("zrnorm:%e\n",zrnorm);
  //return;
  if (mpinr == 0) {
    printf( "\n ** BICG (c version, CUDA7, 1kernel(%d)) **\n", (*blocks) );
    printf( "\nOriginal relative residual norm = %.2e/%.2e = %.2e\n",zrnorm,bnorm,zrnorm/bnorm );
    printf( "c_HACApK_bicgstab_cax_lfmtx_cuda7(%d) start\n", (*blocks) );
  }
  for ( step=1; step<=mstep; step++ ) {
    //for(step=1; step<=1; step++){
    if (zrnorm/bnorm < eps) break;
    // zp(:nd) = zr(:nd) + beta*(zp(:nd) - zeta*zakp(:nd))
    if (beta == zero) {
      //#pragma omp parallel for
      //for(i=0;i<(*nd);i++)zp[i]=zr[i];
      cudaMemcpy(d_zp,d_zr,sizeof(double)*(*nd), cudaMemcpyDeviceToDevice);
    } else {
      /*
#pragma omp parallel for
	  for(i=0;i<(*nd);i++){
	    zp[i] = zr[i] + beta * (zp[i] + zeta*zakp[i]);
	  }
      */
      myCudaFunc2<<<112,128>>>(d_zp,d_zr,d_zakp,beta,zeta,(*nd));
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
    //#pragma omp parallel for
    //for(i=0;i<(*nd);i++)zkp[i]=zp[i];
    cudaMemcpy(d_zkp,d_zp,sizeof(double)*(*nd),cudaMemcpyDeviceToDevice);
    //  .. MATVEC ..
    //for(i=0;i<(*nd);i++)zakp[i]=0.0;
    tic = MPI_Wtime();
    //c_hacapk_adot_body_lfmtx_cuda_calc(zakp,st_leafmtxp,zkp,wws, &time_batch,&time_set,&time_copy,*nd);
    c_hacapk_adot_body_lfmtx_cuda_calc_device_smo_1kernel
      (d_zakp,st_leafmtxp,d_zkp,wws, &time_batch,&time_set,&time_copy,*nd, smd, d_mat, s1info, *blocks);
    cudaThreadSynchronize();
    time_spmv += (MPI_Wtime()-tic);
    /*
      {
	  FILE *F;
	  F=fopen("cuda-zakp.dat","w");
	  for(i=0;i<(*nd);i++)fprintf(F,"%e\n", zakp[i]);
	  fclose(F);
	  }
    */
    //c_hacapk_adot_cax_lfmtx_cuda_comm(zakp,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
    //
    /*
	znorm = 0.0;
#pragma omp parallel for reduction(+:znorm)
	for(i=0;i<(*nd);i++)znorm += zshdw[i]*zr[i];
	zden = 0.0;
#pragma omp parallel for reduction(+:zden)
	for(i=0;i<(*nd);i++)zden += zshdw[i]*zakp[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_znorm,d_zshdw,d_zr,(*nd));
    myCudaReductionZero<128><<<112,128>>>(d_zden,d_zshdw,d_zakp,(*nd));
    /*
      {
	  FILE *F;
	  F=fopen("cuda.dat","w");
	  fprintf(F,"%e %e\n",znorm, zden);
	  fclose(F);
	  }
    */
    cudaMemcpy(&znorm,d_znorm,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(&zden,d_zden,sizeof(double),cudaMemcpyDeviceToHost);
    //printf("znorm:%e\n",znorm);
    //printf("zden:%e\n",zden);
    alpha = -znorm/zden;
    znormold = znorm;
    // zt(:nd) = zr(:nd) - alpha*zakp(:nd)
    //#pragma omp parallel for
    //for(i=0;i<(*nd);i++)zt[i]=zr[i]+alpha*zakp[i];
    myCudaFunc3<<<112,128>>>(d_zt,d_zr,alpha,d_zakp,*nd);
    alpha = -alpha;
    // zkt(:nd) = zt(:nd)
    //#pragma omp parallel for
    //for(i=0;i<(*nd);i++)zkt[i]=zt[i];
    cudaMemcpy(d_zkt,d_zt,sizeof(double)*(*nd),cudaMemcpyDeviceToDevice);
    //  .. MATVEC ..
    //for(i=0;i<(*nd);i++)zakt[i]=0.0;
    tic = MPI_Wtime();
    //c_hacapk_adot_body_lfmtx_cuda_calc(zakt,st_leafmtxp,zkt,wws, &time_batch,&time_set,&time_copy,*nd);
    c_hacapk_adot_body_lfmtx_cuda_calc_device_smo_1kernel
      (d_zakt,st_leafmtxp,d_zkt,wws, &time_batch,&time_set,&time_copy,*nd, smd, d_mat, s1info, *blocks);
    cudaThreadSynchronize();
    time_spmv += (MPI_Wtime()-tic);
    //c_hacapk_adot_cax_lfmtx_cuda_comm(zakt,st_ctl,wws,wwr,isct,irct,*nd, &time_mpi);
    //
    /*
	znorm = 0.0;
#pragma omp parallel for reduction(+:znorm)
	for(i=0;i<(*nd);i++)znorm += zakt[i]*zt[i];
	zden = 0.0;
#pragma omp parallel for reduction(+:zden)
	for(i=0;i<(*nd);i++)zden += zakt[i]*zakt[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_znorm,d_zakt,d_zt,(*nd));
    myCudaReductionZero<128><<<112,128>>>(d_zden,d_zakt,(*nd));
    cudaMemcpy(&znorm,d_znorm,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(&zden,d_zden,sizeof(double),cudaMemcpyDeviceToHost);
    //printf("znorm:%e\n",znorm);
    //printf("zden:%e\n",zden);
    zeta = znorm/zden;
    //printf("zeta:%e\n",zeta);
    /*
      {
	  FILE *F;
	  F=fopen("cuda.dat","a");
	  fprintf(F,"%e %e\n",znorm, zden);
	  fclose(F);
	  }
    */
    // u(:nd) = u(:nd) + alpha*zkp(:nd) + zeta*zkt(:nd)
    /*
#pragma omp parallel for
	for(i=0;i<(*nd);i++){
	  u[i] += alpha*zkp[i] + zeta*zkt[i];
	}
    */
    myCudaFunc4<<<112,128>>>(d_u,alpha,d_zkp,zeta,d_zkt,*nd);
    // zr(:nd) = zt(:nd) - zeta*zakt(:nd)
    zeta = -zeta;
    /*
#pragma omp parallel
	for(i=0;i<(*nd);i++){
	  zr[i]=zt[i] + zeta*zakt[i];
	}
    */
    myCudaFunc5<<<112,128>>>(d_zr,d_zt,zeta,d_zakt,*nd);
    // beta = alpha/zeta * HACApK_dotp_d(nd,zshdw,zr)/znormold;
    /*
	beta = 0.0;
#pragma omp parallel for reduction(+:beta)
	for(i=0;i<(*nd);i++)beta += zshdw[i]*zr[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_beta,d_zshdw,d_zr,*nd);
    cudaMemcpy(&beta,d_beta,sizeof(double),cudaMemcpyDeviceToHost);
    beta = -alpha/zeta * beta/znormold;
    /*
	zrnorm = 0.0;
#pragma omp parallel for reduction(+:zrnorm)
	for(i=0;i<(*nd);i++)zrnorm += zr[i]*zr[i];
    */
    myCudaReductionZero<128><<<112,128>>>(d_zrnorm,d_zr,*nd);
    cudaMemcpy(&zrnorm,d_zrnorm,sizeof(double),cudaMemcpyDeviceToHost);
    //printf("beta:%e\n",beta);
    //printf("zrnorm:%e\n",zrnorm);
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
	char str[32];
	snprintf(str, 32, "C-CUDA7(1kernel-%d)", (*blocks));
	printf( "%s  %d  BiCG           = %.5e\n", str, i, time );
	printf( "%s  %d  time_mpi       = %.5e\n", str, i, time_mpi );
	printf( "%s  %d  time_matvec    = %.5e\n", str, i, time_spmv );
	printf( "%s  %d  >time_copy     = %.5e\n", str, i, time_copy );
	printf( "%s  %d  >time_set      = %.5e\n", str, i, time_set );
	printf( "%s  %d  >time_batch    = %.5e\n", str, i, time_batch );
	printf( "%s  %d  time_setmatrix = %.5e\n", str, i, time_setmatrix );
	printf( "%s  %d  iteration      = %d\n", str, i, step );
      }
      MPI_Barrier( icomm );
    }
  }
    // delete matrix
    //c_hacapk_adot_body_lfdel_batch_(st_leafmtxp);
  cudaFree(d_mat);
  myDestroy1kInfo(&s1info);

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
  cudaThreadSynchronize();
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
	cudaThreadSynchronize();
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
	cudaThreadSynchronize();
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
			printf( "C-WARP  %d  time_mpi    = %.5e\n", i, time_mpi );
			printf( "C-WARP  %d  time_matvec = %.5e\n", i, time_spmv );
			printf( "C-WARP  %d  >time_copy  = %.5e\n", i, time_copy );
			printf( "C-WARP  %d  >time_set   = %.5e\n", i, time_set );
			printf( "C-WARP  %d  >time_batch = %.5e\n", i, time_batch );
			printf( "C-WARP  %d  iteration   = %d\n", i, step );
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
