//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================

// WENO routines copied from BAM TODO
//========================================================================================

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "reconstruction.hpp"

#define SGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define USE_WEIGHTS_OPTIMAL 0

double rec1d_m_weno5(double uimt, double uimo, double ui, double uipo, double uipt);
double rec1d_p_weno5(double uimt, double uimo, double ui, double uipo, double uipt);
double rec1d_m_wenoz(double uimt, double uimo, double ui, double uipo, double uipt);
double rec1d_p_wenoz(double uimt, double uimo, double ui, double uipo, double uipt);
double rec1d_m_mp5(double uimt, double uimo, double ui, double uipo, double uipt);
double rec1d_p_mp5(double uimt, double uimo, double ui, double uipo, double uipt);
double rec1d_m_ceno3(double uimt, double uimo, double ui, double uipo, double uipt);
double rec1d_p_ceno3(double uimt, double uimo, double ui, double uipo, double uipt);
double mp5lim( double u,
               double uimt,double uimo,double ui,double uipo,double uipt );
double Median(double a, double b, double c);
double MM2( double x, double y );
double MC2( double x, double y );

double ceno3lim( double d[3] );


static double cc [32] = { 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.,
                          11., 12., 13., 14., 15., 16., 17., 18., 19., 20.,
                          21., 22., 23., 24., 25., 26., 27., 28., 29., 30.,
                          31};
static double oocc [32] = { 1e99, 1., 1./2., 1./3., 1./4., 1./5., 1./6., 1./7., 1./8., 1./9., 1./10.,
                            1./11., 1./12., 1./13., 1./14., 1./15., 1./16., 1./17., 1./18., 1./19., 1./20.,
                            1./21., 1./22., 1./23., 1./24., 1./25., 1./26., 1./27., 1./28., 1./29., 1./30.,
                            1./31};

const double othreeotwo  = 13./12.;
const double EPSL        = 1e-40; //1e-6
const double alpha       = 0.7; // CENO coef
static double optimw[3]  = {1./10., 3./5., 3./10.};// WENO optimal weights

static double cl5[5] = {2./60., -13./60., 47./60., 27./60., -3./60.};





void Reconstruction::WenoX1(
    const int k, const int j, const int il, const int iu,
    const AthenaArray<Real> &q,
    AthenaArray<Real> &ql, AthenaArray<Real> &qr) {
  Coordinates *pco = pmy_block_->pcoord;
  // set work arrays to shallow copies of scratch arrays
  AthenaArray<Real> &qc = scr1_ni_, &dql = scr2_ni_, &dqr = scr3_ni_,
                   &dqm = scr4_ni_;
  const int nu = q.GetDim4() - 1;

  // compute L/R slopes for each variable
  for (int n=0; n<=nu; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      Real luimt = q(n,k,j,i-3);
      Real luimo = q(n,k,j,i-2);
      Real lui = q(n,k,j,i-1);
      Real luipo = q(n,k,j,i);
      Real luipt = q(n,k,j,i+1);
      Real ruimt = q(n,k,j,i-2);
      Real ruimo = q(n,k,j,i-1);
      Real rui = q(n,k,j,i);
      Real ruipo = q(n,k,j,i+1);
      Real ruipt = q(n,k,j,i+2);
      ql(n,i) = rec1d_p_weno5(luimt,luimo,lui,luipo,luipt);
      qr(n,i) = rec1d_m_weno5(ruimt,ruimo,rui,ruipo,ruipt);
//    possible other reconstruction routines for testing - to be separated into separate callable reconstruction routines
//      ql(n,i) = rec1d_p_wenoz(luimt,luimo,lui,luipo,luipt);
//      qr(n,i) = rec1d_m_wenoz(ruimt,ruimo,rui,ruipo,ruipt);
//      ql(n,i) = rec1d_p_mp5(luimt,luimo,lui,luipo,luipt);
//      qr(n,i) = rec1d_m_mp5(ruimt,ruimo,rui,ruipo,ruipt);
//      ql(n,i) = rec1d_p_ceno3(luimt,luimo,lui,luipo,luipt);
//      qr(n,i) = rec1d_m_ceno3(ruimt,ruimo,rui,ruipo,ruipt);
    }
  }
  return;
}

void Reconstruction::WenoX2(
    const int k, const int j, const int il, const int iu,
    const AthenaArray<Real> &q,
    AthenaArray<Real> &ql, AthenaArray<Real> &qr) {
  Coordinates *pco = pmy_block_->pcoord;
  // set work arrays to shallow copies of scratch arrays
  AthenaArray<Real> &qc = scr1_ni_, &dql = scr2_ni_, &dqr = scr3_ni_,
                   &dqm = scr4_ni_;
  const int nu = q.GetDim4() - 1;

  // compute L/R slopes for each variable
  for (int n=0; n<=nu; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      Real luimt = q(n,k,j-3,i);
      Real luimo = q(n,k,j-2,i);
      Real lui = q(n,k,j-1,i);
      Real luipo = q(n,k,j,i);
      Real luipt = q(n,k,j+1,i);
      Real ruimt = q(n,k,j-2,i);
      Real ruimo = q(n,k,j-1,i);
      Real rui = q(n,k,j,i);
      Real ruipo = q(n,k,j+1,i);
      Real ruipt = q(n,k,j+2,i);
      ql(n,i) = rec1d_p_weno5(luimt,luimo,lui,luipo,luipt);
      qr(n,i) = rec1d_m_weno5(ruimt,ruimo,rui,ruipo,ruipt);
//      ql(n,i) = rec1d_p_wenoz(luimt,luimo,lui,luipo,luipt);
//      qr(n,i) = rec1d_m_wenoz(ruimt,ruimo,rui,ruipo,ruipt);
//      ql(n,i) = rec1d_p_mp5(luimt,luimo,lui,luipo,luipt);
//      qr(n,i) = rec1d_m_mp5(ruimt,ruimo,rui,ruipo,ruipt);
//      ql(n,i) = rec1d_p_ceno3(luimt,luimo,lui,luipo,luipt);
//      qr(n,i) = rec1d_m_ceno3(ruimt,ruimo,rui,ruipo,ruipt);
    }
  }
  return;
}
//----------------------------------------------------------------------------------------
//  \brief

void Reconstruction::WenoX3(
    const int k, const int j, const int il, const int iu,
    const AthenaArray<Real> &q,
    AthenaArray<Real> &ql, AthenaArray<Real> &qr) {
  Coordinates *pco = pmy_block_->pcoord;
  // set work arrays to shallow copies of scratch arrays
  AthenaArray<Real> &qc = scr1_ni_, &dql = scr2_ni_, &dqr = scr3_ni_,
                   &dqm = scr4_ni_;
  const int nu = q.GetDim4() - 1;

  // compute L/R slopes for each variable
  for (int n=0; n<=nu; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      Real luimt = q(n,k-3,j,i);
      Real luimo = q(n,k-2,j,i);
      Real lui = q(n,k-1,j,i);
      Real luipo = q(n,k,j,i);
      Real luipt = q(n,k+1,j,i);
      Real ruimt = q(n,k-2,j,i);
      Real ruimo = q(n,k-1,j,i);
      Real rui = q(n,k,j,i);
      Real ruipo = q(n,k+1,j,i);
      Real ruipt = q(n,k+2,j,i);
      ql(n,i) = rec1d_p_weno5(luimt,luimo,lui,luipo,luipt);
      qr(n,i) = rec1d_m_weno5(ruimt,ruimo,rui,ruipo,ruipt);
//      ql(n,i) = rec1d_p_wenoz(luimt,luimo,lui,luipo,luipt);
//      qr(n,i) = rec1d_m_wenoz(ruimt,ruimo,rui,ruipo,ruipt);
//      ql(n,i) = rec1d_p_mp5(luimt,luimo,lui,luipo,luipt);
//      qr(n,i) = rec1d_m_mp5(ruimt,ruimo,rui,ruipo,ruipt);
//      ql(n,i) = rec1d_p_ceno3(luimt,luimo,lui,luipo,luipt);
//      qr(n,i) = rec1d_m_ceno3(ruimt,ruimo,rui,ruipo,ruipt);
    }
  }
  return;
}



//double rec1d_p_weno5(double *u, int i)
double rec1d_p_weno5(double uimt, double uimo, double ui, double uipo, double uipt)
{
  double up;
/*
i  double uimt = u [i-2];
  double uimo = u [i-1];
  double ui   = u [i];
  double uipo = u [i+1];
  double uipt = u [i+2];
*/
  double uk[3], a[3], b[3], w[3], dsa;
  int j;

#if (USE_WEIGHTS_OPTIMAL)
  for( j = 0 ; j<3; j++) w[j] = optimw[j];
#else
//   smoothness coefs, Jiag & Shu '96
     b[0] = othreeotwo * SQR(( uimt-cc[2]*uimo+ui   )) + oocc[4] * SQR((uimt-cc[4]*uimo+cc[3]*ui));
     b[1] = othreeotwo * SQR(( uimo-cc[2]*ui  +uipo )) + oocc[4] * SQR((uimo-uipo ) );
     b[2] = othreeotwo * SQR(( ui  -cc[2]*uipo+uipt )) + oocc[4] * SQR((cc[3]*ui-cc[4]*uipo+uipt));
  
     for( j = 0 ; j<3; j++) a[j] = optimw[j]/( SQR(( EPSL + b[j])) );
         dsa = cc[1]/( a[0] + a[1] + a[2] );
         for( j = 0 ; j<3; j++) w[j] = a[j] * dsa;
#endif
     uk[0] = oocc[6]*(   cc[2]*uimt - cc[7]*uimo + cc[11]*ui   );
     uk[1] = oocc[6]*( - cc[1]*uimo + cc[5]*ui   + cc[2] *uipo );
     uk[2] = oocc[6]*(   cc[2]*ui   + cc[5]*uipo -        uipt );
  
     up = w[0]*uk[0] + w[1]*uk[1] + w[2]*uk[2];
     return up;
}


double rec1d_m_weno5(double uimt, double uimo, double ui, double uipo, double uipt)
{
  double um;
/*
  double uimt = u [i-2];
  double uimo = u [i-1];
  double ui   = u [i];
  double uipo = u [i+1];
  double uipt = u [i+2];
*/
  double uk[3], a[3], b[3], w[3], dsa;
  int j;

#if (USE_WEIGHTS_OPTIMAL)
  for( j = 0 ; j<3; j++) w[j] = optimw[j];
#else
  b[0] = othreeotwo * SQR(( uipt-cc[2]*uipo+ui   )) + oocc[4] * SQR((uipt-cc[4]*uipo+cc[3]*ui)); 
  b[1] = othreeotwo * SQR(( uipo-cc[2]*ui  +uimo )) + oocc[4] * SQR((uipo-uimo )); 
  b[2] = othreeotwo * SQR(( ui  -cc[2]*uimo+uimt )) + oocc[4] * SQR((cc[3]*ui-cc[4]*uimo+uimt)); 
  
  for( j = 0 ; j<3; j++) a[j] = optimw[j]/( SQR(( EPSL + b[j])) );
  dsa = cc[1]/( a[0] + a[1] + a[2] );
  for( j = 0 ; j<3; j++) w[j] = a[j] * dsa;
#endif   
  
  uk[0] = oocc[6]*( cc[2]*uipt - cc[7]*uipo + cc[11]*ui   );
  uk[1] = oocc[6]*( -     uipo + cc[5]*ui   + cc[2] *uimo );
  uk[2] = oocc[6]*( cc[2]*ui   + cc[5]*uimo -        uimt );

  um = w[0]*uk[0] + w[1]*uk[1] + w[2]*uk[2];

  return um;
}

double rec1d_p_wenoz(double uimt, double uimo, double ui, double uipo, double uipt)
{
  double up;
/*
  double uimt = u [i-2];
  double uimo = u [i-1];
  double ui   = u [i];
  double uipo = u [i+1];
  double uipt = u [i+2];
*/
  double uk[3], a[3], b[3], w[3], dsa;
  int j;

#if (USE_WEIGHTS_OPTIMAL)
  for( j = 0 ; j<3; j++) w[j] = optimw[j];
#else
  // smoothness coefs, Jiag & Shu '96
  b[0] = othreeotwo * SQR(( uimt-cc[2]*uimo+ui   )) + oocc[4] * SQR((uimt-cc[4]*uimo+cc[3]*ui));
  b[1] = othreeotwo * SQR(( uimo-cc[2]*ui  +uipo )) + oocc[4] * SQR((uimo-uipo ) );
  b[2] = othreeotwo * SQR(( ui  -cc[2]*uipo+uipt )) + oocc[4] * SQR((cc[3]*ui-cc[4]*uipo+uipt));

  for( j = 0 ; j<3; j++) a[j] = optimw[j]*( cc[1] + fabs(b[0]-b[2])/( EPSL + b[j]) );
  dsa = cc[1]/( a[0] + a[1] + a[2] );
  for( j = 0 ; j<3; j++) w[j] = a[j] * dsa;
#endif

  uk[0] = oocc[6]*(   cc[2]*uimt - cc[7]*uimo + cc[11]*ui   );
  uk[1] = oocc[6]*( - cc[1]*uimo + cc[5]*ui   + cc[2] *uipo );
  uk[2] = oocc[6]*(   cc[2]*ui   + cc[5]*uipo -        uipt );

  up = w[0]*uk[0] + w[1]*uk[1] + w[2]*uk[2];

  return up;
}
// WENOZ 
double rec1d_m_wenoz(double uimt, double uimo, double ui, double uipo, double uipt)
{
  double um;
/*
  double uimt = u [i-2];
  double uimo = u [i-1];
  double ui   = u [i];
  double uipo = u [i+1];
  double uipt = u [i+2];
*/
  double uk[3], a[3], b[3], w[3], dsa;
  int j;

#if (USE_WEIGHTS_OPTIMAL)
  for( j = 0 ; j<3; j++) w[j] = optimw[j];
#else
  // smoothness coefs, Jiag & Shu '96
  b[0] = othreeotwo * SQR(( uipt-cc[2]*uipo+ui   )) + oocc[4] * SQR((uipt-cc[4]*uipo+cc[3]*ui));
  b[1] = othreeotwo * SQR(( uipo-cc[2]*ui  +uimo )) + oocc[4] * SQR((uipo-uimo ));
  b[2] = othreeotwo * SQR(( ui  -cc[2]*uimo+uimt )) + oocc[4] * SQR((cc[3]*ui-cc[4]*uimo+uimt));

  for( j = 0 ; j<3; j++) a[j] = optimw[j]*( cc[1] + fabs(b[0]-b[2])/( EPSL + b[j]) );
  dsa = cc[1]/( a[0] + a[1] + a[2] );
  for( j = 0 ; j<3; j++) w[j] = a[j] * dsa;
#endif

  uk[0] = oocc[6]*( cc[2]*uipt - cc[7]*uipo + cc[11]*ui   );
  uk[1] = oocc[6]*( -     uipo + cc[5]*ui   + cc[2] *uimo );
  uk[2] = oocc[6]*( cc[2]*ui   + cc[5]*uimo -        uimt );

  um = w[0]*uk[0] + w[1]*uk[1] + w[2]*uk[2];

  return um;
}

double rec1d_m_mp5(double uimt, double uimo, double ui, double uipo, double uipt)
{
/*	
  double uimt = u [i-2];
  double uimo = u [i-1];
  double ui   = u [i];
  double uipo = u [i+1];
  double uipt = u [i+2];
*/
  double ulim  = cl5[0]*uipt + cl5[1]*uipo + cl5[2]*ui + cl5[3]*uimo + cl5[4]*uimt;
  return mp5lim(ulim, uipt,uipo,ui,uimo,uimt);
}
double rec1d_p_mp5(double uimt, double uimo, double ui, double uipo, double uipt)
{
/*
  double uimt = u [i-2];
  double uimo = u [i-1];
  double ui   = u [i];
  double uipo = u [i+1];
  double uipt = u [i+2];
*/
  double ulim  = cl5[0]*uimt + cl5[1]*uimo + cl5[2]*ui + cl5[3]*uipo + cl5[4]*uipt;
  return mp5lim(ulim, uimt,uimo,ui,uipo,uipt);
}

double mp5lim( double u,
               double uimt,double uimo,double ui,double uipo,double uipt )
{

  double ump,umin,umax, fUL,fAV,fMD,fLC;
  double d2m,d2c,d2p, dMMm,dMMp;
  double tmp1,tmp2, mp5;

  /*static double alpha = 2.0;*/
  static double alpha = 4.0;
  static double eps = 1e-20;
  static double fot = 1.3333333333333333333; // 4/3

  mp5 = u;
  ump = ui + MM2( (uipo-ui), alpha*(ui-uimo) );

  if( ((u-ui)*(u-ump)) <= eps ) return mp5;

  d2m = uimt -2.*uimo +ui;
  d2c = uimo -2.*ui   +uipo;
  d2p = ui   -2.*uipo +uipt;

  tmp1 = MM2(4.*d2c - d2p, 4.*d2p - d2c);
  tmp2 = MM2(d2c, d2p);
  dMMp = MM2(tmp1,tmp2);

  tmp1 = MM2(4.*d2m - d2c, 4.*d2c - d2m);
  tmp2 = MM2(d2c, d2m);
  dMMm = MM2(tmp1,tmp2);

  fUL = ui + alpha*(ui-uimo);
  fAV = 0.5*(ui + uipo);
  fMD = fAV - 0.5*dMMp;
  fLC = 0.5*(3.0*ui - uimo) + fot*dMMm;

  tmp1 = fmin(ui, uipo); tmp1 = fmin(tmp1, fMD);
  tmp2 = fmin(ui, fUL);  tmp2 = fmin(tmp2, fLC);
  umin = fmax(tmp1, tmp2);

  tmp1 = fmax(ui, uipo); tmp1 = fmax(tmp1, fMD);
  tmp2 = fmax(ui, fUL);  tmp2 = fmax(tmp2, fLC);
  umax = fmin(tmp1, tmp2);

  mp5 = Median(mp5, umin, umax);

  return mp5;
}

double Median(double a, double b, double c)
{
  return (a + MM2(b-a,c-a));
}

double MM2( double x, double y )
{
  double s1 = SGN( cc[1], x );
  double s2 = SGN( cc[1], y );

  return( oocc[2] * (s1+s2) * fmin(fabs(x),fabs(y)) );
}

double rec1d_p_ceno3(double uimt, double uimo, double ui, double uipo, double uipt)
{
  double up;
/*
  double uipt = u [i+2];
  double uipo = u [i+1];
  double ui   = u [i];
  double uimo = u [i-1];
  double uimt = u [i-2];
*/
  double slope = oocc[2] * MC2( ( ui - uimo ), ( uipo - ui ) );

  double tmpL;
  double tmpd[3];    // these are d^k_i with k = -1,0,1 

  tmpL    = ui + slope;
  tmpd[0] = ( cc[3]*uimt - cc[10]*uimo + cc[15]*ui   )*oocc[8] - tmpL;
  tmpd[1] = ( -     uimo + cc[6] *ui   + cc[3] *uipo )*oocc[8] - tmpL;
  tmpd[2] = ( cc[3]*ui   + cc[6] *uipo -        uipt )*oocc[8] - tmpL;

  up  = tmpL + ceno3lim(tmpd);

  return up;
}
double rec1d_m_ceno3(double uimt, double uimo, double ui, double uipo, double uipt)
{
  double um;
/*
  double uipt = u [i+2];
  double uipo = u [i+1];
  double ui   = u [i];
  double uimo = u [i-1];
  double uimt = u [i-2];
*/
  double slope = oocc[2] * MC2( ( ui - uimo ), ( uipo - ui ) );

  double tmpL;
  double tmpd[3];    // these are d^k_i with k = -1,0,1 

  tmpL    = ui - slope;
  tmpd[2] = ( cc[3]*uipt - cc[10]*uipo + cc[15]*ui   )*oocc[8] - tmpL;
  tmpd[1] = ( -     uipo + cc[6] *ui   + cc[3] *uimo )*oocc[8] - tmpL;
  tmpd[0] = ( cc[3]*ui   + cc[6] *uimo -        uimt )*oocc[8] - tmpL;

  um  = tmpL + ceno3lim(tmpd);

  return um;
}

double MC2( double x, double y )
{

  double s1  = SGN( cc[1], x );
  double s2  = SGN( cc[1], y );
  double min = fmin( (cc[2]*fabs(x)), (cc[2]*fabs(y)) );

  return( oocc[2]*(s1+s2) * fmin( min, (oocc[2]*fabs(x+y)) ) );
}

double ceno3lim( double d[3] )
{

    double o3term = 0.0;
    double absd[3];
    int kmin;

    if ( ((d[0]>=0.) && (d[1]>=0.) && (d[2]>=0.)) ||
         ((d[0]<0.) && (d[1]<0.) && (d[2]<0.))  ) {

        absd[0] = fabs( d[0] );
        absd[1] = fabs( alpha*d[1] );
        absd[2] = fabs( d[2] );

        kmin = 0;
        if( absd[1] < absd[kmin] ) kmin = 1;
        if( absd[2] < absd[kmin] ) kmin = 2;

        o3term = d[kmin];

    }

    return( o3term );

}


