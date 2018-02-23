#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct stc_HACApK_leafmtx {
  int ltmtx;
  int kt;
  int nstrtl, ndl;
  int nstrtt, ndt;
  size_t a1size;
  double *a1;
  double *a2;
} stc_HACApK_leafmtx;

typedef struct stc_HACApK_leafmtxp {
  int nd;
  int nlf;
  int nlfkt;
  int ktmax;
  int st_lf_stride;
  stc_HACApK_leafmtx *st_lf;
} stc_HACApK_leafmtxp;

void hmvm
(double *zau, // result vector
 stc_HACApK_leafmtxp *st_leafmtxp, // h-matrix
 double *zu, // input vector
 double *zbu // tmp vector
) {
  register int ip,il,it;
  int nlf,ndl,ndt,nstrtl,nstrtt,kt,itl,itt,ill;
  int st_lf_stride = st_leafmtxp->st_lf_stride;
  size_t a1size;

  nlf=st_leafmtxp->nlf;

  for(ip=0; ip<nlf; ip++){
    /**/
    stc_HACApK_leafmtx *sttmp;
    sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
    ndl   =sttmp->ndl;
    ndt   =sttmp->ndt;
    nstrtl=sttmp->nstrtl;
    nstrtt=sttmp->nstrtt;
    if(sttmp->ltmtx==1){
      double *a2tmp = (double *)((void*)(sttmp->a1)+sttmp->a1size);
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

int bicgsolve
(
 stc_HACApK_leafmtxp *st_leafmtxp, // h-matrix
 double *u, // vector
 double *b,
 int mstep,
 double eps,
 int ND,
 int zzz
)
{
  // local arrays
  double *zr, *zshdw, *zp, *zt, *zkp, *zakp, *zkt, *zakt;
  double *wws, *wwr;
  int isct[2], irct[2];
  // local variables
  double alpha, beta, zeta, zz, zden, znorm, znormold, bnorm, zrnorm;
  int step;
  int i, converged;

  wws = (double*)malloc(ND * sizeof(double));
  wwr = (double*)malloc(ND * sizeof(double));

  zt = (double*)malloc(ND * sizeof(double));
  zr = (double*)malloc(ND * sizeof(double));
  zp = (double*)malloc(ND * sizeof(double));
  zkp = (double*)malloc(ND * sizeof(double));
  zakp = (double*)malloc(ND * sizeof(double));
  zkt = (double*)malloc(ND * sizeof(double));
  zakt= (double*)malloc(ND * sizeof(double));
  zshdw = (double*)malloc(ND * sizeof(double));

  // init
  alpha = 0.0; beta = 0.0; zeta = 0.0;
  zz = 0.0;
  for(i=0;i<ND;i++)zz += b[i]*b[i];
  bnorm=sqrt(zz);
  //printf("bnorm:%e\n",bnorm);
  for(i=0;i<ND;i++)zr[i]=b[i];

  //  .. MATVEC ..
  for(i=0;i<ND;i++)zshdw[i]=0.0;
  hmvm(zshdw,st_leafmtxp,u,wws);
  //c_hacapk_adot_cax_lfmtx_seq_comm(zshdw, st_ctl, wws, wwr, isct, irct, ND, &time_mpi);

  for(i=0;i<ND;i++)zr[i]+=-1.0*zshdw[i];
  for(i=0;i<ND;i++)zshdw[i]=zr[i];
  zrnorm = 0.0;
  for(i=0;i<ND;i++)zrnorm += zr[i]*zr[i];
  zrnorm = sqrt(zrnorm);
  //printf("zrnorm:%e",zrnorm);

  // iteration
  for ( step=1; step<=mstep; step++ ) {
	// converged?
    if(zrnorm/bnorm < eps){converged++; break;}

	if (beta == 0.0) {
	  for(i=0;i<ND;i++)zp[i]=zr[i];
	} else {
	  for(i=0;i<ND;i++){
	    zp[i] = zr[i] + beta * (zp[i] + zeta*zakp[i]);
	  }
	}
	for(i=0;i<ND;i++)zkp[i]=zp[i];

	//  .. MATVEC ..
	for(i=0;i<ND;i++)zakp[i]=0.0;
	hmvm(zakp,st_leafmtxp,zkp,wws);
	//c_hacapk_adot_cax_lfmtx_seq_comm(zakp,st_ctl,wws,wwr,isct,irct,ND, &time_mpi);

	znorm = 0.0; for(i=0;i<ND;i++)znorm += zshdw[i]*zr[i];
	zden = 0.0; for(i=0;i<ND;i++)zden += zshdw[i]*zakp[i];

	alpha = -znorm/zden;
	znormold = znorm;

	for(i=0;i<ND;i++)zt[i]=zr[i];
	for(i=0;i<ND;i++)zt[i]+=alpha*zakp[i];
	alpha = -alpha;

	for(i=0;i<ND;i++)zkt[i]=zt[i];

	//  .. MATVEC ..
	for(i=0;i<ND;i++)zakt[i]=0.0;
	hmvm(zakt,st_leafmtxp,zkt,wws);
	//c_hacapk_adot_cax_lfmtx_seq_comm(zakt,st_ctl,wws,wwr,isct,irct,ND, &time_mpi);

	znorm = 0.0; for(i=0;i<ND;i++)znorm += zakt[i]*zt[i];
	zden = 0.0; for(i=0;i<ND;i++)zden += zakt[i]*zakt[i];
	zeta = znorm/zden;

	for(i=0;i<ND;i++)u[i]+=alpha*zkp[i];
	for(i=0;i<ND;i++)u[i]+=zeta*zkt[i];
	zeta = -zeta;
	for(i=0;i<ND;i++)zr[i]=zt[i];
	for(i=0;i<ND;i++)zr[i]+=zeta*zakt[i];

	beta = 0.0;
	for(i=0;i<ND;i++)beta += zshdw[i]*zr[i];
	beta = -alpha/zeta * beta/znormold;
	zrnorm = 0.0;
	for(i=0;i<ND;i++)zrnorm += zr[i]*zr[i];
	zrnorm = sqrt(zrnorm);
	printf( " %d: log10(zrnorm/bnorm)=log10(%.2e/%.2e)=%.2e\n",step,zrnorm,bnorm,log10(zrnorm/bnorm) );
  }
  // converged?
  if(converged==0)step--;


  printf( "C-TEST  iteration   = %d\n", step );

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

  return 0;
}
