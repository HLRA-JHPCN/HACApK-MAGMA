#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/*
Fortranから呼び出す場合とC言語単体テストの場合では配列参照手順に違いが生じるので気を付けること
 F
  int st_lf_stride = st_leafmtxp->st_lf_stride;
  double *a2tmp = (double *)((void*)(sttmp->a1)+sttmp->a1size);
  zau[ill] += a2tmp[itl]*zbu[il];
C
  zau[ill] += sttmp->a2[itl]*zbu[il];
*/

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

// sequential
void hmvm
(double *zau, // result vector
 stc_HACApK_leafmtxp *st_leafmtxp, // h-matrix
 double *zu, // input vector
 double *zbu, // tmp vector
 double *time_batch, double *time_set, double *time_copy
) {
  register int ip,il,it;
  int nlf,ndl,ndt,nstrtl,nstrtt,kt,itl,itt,ill;
  //int st_lf_stride = st_leafmtxp->st_lf_stride;
  //size_t a1size;

  nlf=st_leafmtxp->nlf;

  for(ip=0; ip<nlf; ip++){
    /**/
    stc_HACApK_leafmtx *sttmp;
    //sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
	sttmp = &st_leafmtxp->st_lf[ip];
    ndl   =sttmp->ndl;
    ndt   =sttmp->ndt;
    nstrtl=sttmp->nstrtl;
    nstrtt=sttmp->nstrtt;
    if(sttmp->ltmtx==1){
      //double *a2tmp = (double *)((void*)(sttmp->a1)+sttmp->a1size);
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
		  //zau[ill] += a2tmp[itl]*zbu[il];
		  zau[ill] += sttmp->a2[itl]*zbu[il];
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
 int nd
)
{
  // local arrays
  double *zr, *zshdw, *zp, *zt, *zkp, *zakp, *zkt, *zakt;
  double *wws, *wwr;
  int isct[2], irct[2];
  // local variables
  double alpha, beta, zeta, zz, zden, znorm, znormold, bnorm, zrnorm;
  int step;
  int i, converged=0;

  double en_measure_time, st_measure_time, time;
  double time_matvec, time_mpi, time_batch, time_set, time_copy, tic;

  printf("bicgsolve\n");

  wws = (double*)malloc(nd * sizeof(double));
  wwr = (double*)malloc(nd * sizeof(double));

  zt = (double*)malloc(nd * sizeof(double));
  zr = (double*)malloc(nd * sizeof(double));
  zp = (double*)malloc(nd * sizeof(double));
  zkp = (double*)malloc(nd * sizeof(double));
  zakp = (double*)malloc(nd * sizeof(double));
  zkt = (double*)malloc(nd * sizeof(double));
  zakt= (double*)malloc(nd * sizeof(double));
  zshdw = (double*)malloc(nd * sizeof(double));

  time_matvec = 0.0;
  time_mpi = 0.0;
  time_batch = 0.0;
  time_set = 0.0;
  time_copy = 0.0;
  st_measure_time = omp_get_wtime();

  // init
  alpha = 0.0; beta = 0.0; zeta = 0.0;
  zz = 0.0;
  for(i=0;i<nd;i++)zz += b[i]*b[i];
  bnorm=sqrt(zz);
  //printf("bnorm:%e\n",bnorm);
  for(i=0;i<nd;i++)zr[i]=b[i];

  //  .. MATVEC ..
  tic = omp_get_wtime();
  for(i=0;i<nd;i++)zshdw[i]=0.0;
  hmvm(zshdw,st_leafmtxp,u,wws, &time_batch, &time_set, &time_copy);
  //c_hacapk_adot_cax_lfmtx_seq_comm(zshdw, st_ctl, wws, wwr, isct, irct, nd, &time_mpi);
  time_matvec += omp_get_wtime() - tic;

  for(i=0;i<nd;i++)zr[i]+=-1.0*zshdw[i];
  for(i=0;i<nd;i++)zshdw[i]=zr[i];
  zrnorm = 0.0;
  for(i=0;i<nd;i++)zrnorm += zr[i]*zr[i];
  zrnorm = sqrt(zrnorm);
  //printf("zrnorm:%e",zrnorm);

  // iteration
  for ( step=1; step<=mstep; step++ ) {
	// converged?
    if(zrnorm/bnorm < eps){converged++; break;}

	if (beta == 0.0) {
	  for(i=0;i<nd;i++)zp[i]=zr[i];
	} else {
	  for(i=0;i<nd;i++){
	    zp[i] = zr[i] + beta * (zp[i] + zeta*zakp[i]);
	  }
	}
	for(i=0;i<nd;i++)zkp[i]=zp[i];

	//  .. MATVEC ..
	tic = omp_get_wtime();
	for(i=0;i<nd;i++)zakp[i]=0.0;
	hmvm(zakp,st_leafmtxp,zkp,wws, &time_batch, &time_set, &time_copy);
	//c_hacapk_adot_cax_lfmtx_seq_comm(zakp,st_ctl,wws,wwr,isct,irct,nd, &time_mpi);
	time_matvec += omp_get_wtime() - tic;

	znorm = 0.0; for(i=0;i<nd;i++)znorm += zshdw[i]*zr[i];
	zden = 0.0; for(i=0;i<nd;i++)zden += zshdw[i]*zakp[i];

	alpha = -znorm/zden;
	znormold = znorm;

	for(i=0;i<nd;i++)zt[i]=zr[i];
	for(i=0;i<nd;i++)zt[i]+=alpha*zakp[i];
	alpha = -alpha;

	for(i=0;i<nd;i++)zkt[i]=zt[i];

	//  .. MATVEC ..
	for(i=0;i<nd;i++)zakt[i]=0.0;
	tic = omp_get_wtime();
	hmvm(zakt,st_leafmtxp,zkt,wws, &time_batch, &time_set, &time_copy);
	time_matvec += omp_get_wtime() - tic;
	//c_hacapk_adot_cax_lfmtx_seq_comm(zakt,st_ctl,wws,wwr,isct,irct,nd, &time_mpi);

	znorm = 0.0; for(i=0;i<nd;i++)znorm += zakt[i]*zt[i];
	zden = 0.0; for(i=0;i<nd;i++)zden += zakt[i]*zakt[i];
	zeta = znorm/zden;

	for(i=0;i<nd;i++)u[i]+=alpha*zkp[i];
	for(i=0;i<nd;i++)u[i]+=zeta*zkt[i];
	zeta = -zeta;
	for(i=0;i<nd;i++)zr[i]=zt[i];
	for(i=0;i<nd;i++)zr[i]+=zeta*zakt[i];

	beta = 0.0;
	for(i=0;i<nd;i++)beta += zshdw[i]*zr[i];
	beta = -alpha/zeta * beta/znormold;
	zrnorm = 0.0;
	for(i=0;i<nd;i++)zrnorm += zr[i]*zr[i];
	zrnorm = sqrt(zrnorm);
	en_measure_time = omp_get_wtime();
	time = en_measure_time - st_measure_time;
	printf( " %d: log10(zrnorm/bnorm)=log10(%.2e/%.2e)=%.2e\n",step,zrnorm,bnorm,log10(zrnorm/bnorm) );
  }
  // converged?
  if(converged==0)step--;
  en_measure_time = omp_get_wtime();
  time = en_measure_time - st_measure_time;

  printf( "C-SEQ  BiCG        = %.5e\n", time );
  printf( "C-SEQ  time_mpi    = %.5e\n", time_mpi );
  printf( "C-SEQ  time_matvec = %.5e\n", time_matvec );
  printf( "C-SEQ  >time_copy  = %.5e\n", time_copy );
  printf( "C-SEQ  >time_set   = %.5e\n", time_set );
  printf( "C-SEQ  >time_batch = %.5e\n", time_batch );
  printf( "C-SEQ  iteration   = %d\n", step );

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

int load1
(
const char *file_h
 )
{
  int nd;
  FILE *F;
  printf("load nd\n");
  F = fopen(file_h, "rb");
  if(F==NULL){printf("can't open %s for input\n", file_h);return -1;}
  fread(&nd, sizeof(int), 1, F);
  fclose(F);

  printf("load1 finished\n");
  printf("nd = %d\n", nd);
  return nd;
}

int load2
(
 int nd,
 const char *file_h,
 const char *file_u,
 const char *file_b,
 stc_HACApK_leafmtxp *st_leafmtxp,
 double *u, double *b
)
{
  int i, ip;
  FILE *F;
  int approx=0, dense=0;

  printf("load u\n");
  F = fopen(file_u, "rb");
  if(F==NULL){printf("can't open %s for input\n", file_u);return -1;}
  fread(u, sizeof(double), nd, F);
  fclose(F);

  printf("load b\n");
  F = fopen(file_b, "rb");
  if(F==NULL){printf("can't open %s for input\n", file_b);return -1;}
  fread(b, sizeof(double), nd, F);
  fclose(F);

  printf("load h-matrix\n");
  F = fopen(file_h, "rb");
  if(F==NULL){printf("can't open %s for input\n", file_h);return -1;}
  fread(&st_leafmtxp->nd, sizeof(int), 1, F);
  fread(&st_leafmtxp->nlf, sizeof(int), 1, F);
  fread(&st_leafmtxp->nlfkt, sizeof(int), 1, F);
  fread(&st_leafmtxp->ktmax, sizeof(int), 1, F);
  fread(&st_leafmtxp->st_lf_stride, sizeof(int), 1, F);
  int nlf = st_leafmtxp->nlf;
  int st_lf_stride = st_leafmtxp->st_lf_stride;
  printf("info: %d %d %d %d %d\n", st_leafmtxp->nd, st_leafmtxp->nlf, st_leafmtxp->nlfkt, st_leafmtxp->ktmax, st_leafmtxp->st_lf_stride);

  st_leafmtxp->st_lf = (stc_HACApK_leafmtx*)malloc(sizeof(stc_HACApK_leafmtx)*nlf);
  for(ip=0; ip<nlf; ip++){
	//printf("%d: ", ip+1);
	stc_HACApK_leafmtx *sttmp;
	//sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
	sttmp = &st_leafmtxp->st_lf[ip];
	int ndl;
	int ndt;
	int nstrtl;
	int nstrtt;
	int ltmtx;
	size_t a1size = 0;
	fread(&ndl, sizeof(int), 1, F); sttmp->ndl = ndl;
	fread(&ndt, sizeof(int), 1, F); sttmp->ndt = ndt;
	fread(&nstrtl, sizeof(int), 1, F); sttmp->nstrtl = nstrtl;
	fread(&nstrtt, sizeof(int), 1, F); sttmp->nstrtt = nstrtt;
	fread(&ltmtx, sizeof(int), 1, F); sttmp->ltmtx = ltmtx;
	//fread(&a1size, sizeof(size_t), 1, F); sttmp->a1size = a1size;
	//printf("%d %d %d %d %d %zu", ndl, ndt, nstrtl, nstrtt, ltmtx, a1size);
	if(sttmp->ltmtx==1){
	  approx++;
	  //double *a2tmp = (double *)((void*)(sttmp->a1)+sttmp->a1size);
	  fread(&sttmp->kt, sizeof(int), 1, F);
	  int kt = sttmp->kt;
	  sttmp->a1 = (double*)malloc(sizeof(double)*kt*ndt);
	  sttmp->a2 = (double*)malloc(sizeof(double)*kt*ndl);
	  fread(sttmp->a1, sizeof(double), kt*ndt, F);
	  fread(sttmp->a2, sizeof(double), kt*ndl, F);
	  //printf(" : (%d x %d) x (%d x %d)\n", kt, ndt, kt, ndl);
	} else if(sttmp->ltmtx==2){
	  dense++;
	  sttmp->a1 = (double*)malloc(sizeof(double)*ndl*ndt);
	  fread(sttmp->a1, sizeof(double), ndl*ndt, F);
	  //printf(" : %d x %d\n", ndl, ndt);
	}
  }
  fclose(F);

  printf("number of approximate matrices pairs: %d\n", approx);
  printf("number of small dense matrices: %d\n", dense);
  printf("load2 finished\n");
  fflush(stdout);

  return 0;
}

int unload
(
 stc_HACApK_leafmtxp *st_leafmtxp
)
{
  int ip;
  int nlf = st_leafmtxp->nlf;
  printf("unload"); fflush(stdout);
  for(ip=0; ip<nlf; ip++){
	//printf(" %d",ip); fflush(stdout);
	stc_HACApK_leafmtx *sttmp;
	//sttmp = (void *)(st_leafmtxp->st_lf) + st_lf_stride * ip;
	sttmp = &st_leafmtxp->st_lf[ip];
	if(sttmp->ltmtx==1){
	  free(sttmp->a2);
	  free(sttmp->a1);
	} else if(sttmp->ltmtx==2){
	  free(sttmp->a1);
	}
  }
  free(st_leafmtxp->st_lf);
  printf(" finished\n"); fflush(stdout);
  return 0;
}

int main(int argc, char **argv)
{
  char *file_h;
  char *file_u;
  char *file_b;
  int nd;
  double *u, *b;
  double eps;
  int maxiter;

  if(argc!=6){
	printf("usage: %s file_h file_u file_b eps maxiter\n", argv[0]);
	return -1;
  }
  file_h = argv[1];
  file_u = argv[2];
  file_b = argv[3];
  eps = atof(argv[4]);
  maxiter = atoi(argv[5]);

  printf("settings: %s %s %s %e %d\n", file_h, file_u, file_b, eps, maxiter);
  nd = load1(file_h);
  u = (double*)malloc(sizeof(double)*nd);
  b = (double*)malloc(sizeof(double)*nd);
  stc_HACApK_leafmtxp xp;
  load2(nd, file_h, file_u, file_b, &xp, u, b);

  bicgsolve(&xp, u, b, maxiter, eps, nd);

  unload(&xp);

  free(b);
  free(u);

  return 0;
}
