#ifndef CHARACTERISTIC_FIELDS_HPP_
#define CHARACTERISTIC_FIELDS_HPP_

// C headers

// C++ headers

// Athena++ headers
#include "../athena_aliases.hpp"
#include "../utils/linear_algebra.hpp"
#include "hydro.hpp"

#define ONE 1.0
#define ZERO 0.
#define TWO 2.
#define BIG 1e32
#define TINY 1e-32

enum  fi{gxx,gyy,gzz,gxy,gxz,gyz,gxx_u,gyy_u,gzz_u,det_g,bx,a}; // Field Indexes
enum  hi{vx,vy,vz,vx_d,vy_d,vz_d,v2,rho,p,epsl,W,h,T}; // Hydro Indexes
enum  eig{kappa,chi,cs2,L0,Lp,Lm};

namespace characterisiticfields {

static Real cc[32] = { 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 
			11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 
			21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 
			31};
static Real oocc[32] = { 1e99, 1., 1./2., 1./3., 1./4., 1./5., 1./6., 1./7., 1./8., 1./9., 1./10.,
			   1./11., 1./12., 1./13., 1./14., 1./15., 1./16., 1./17., 1./18., 1./19., 1./20., 
			   1./21., 1./22., 1./23., 1./24., 1./25., 1./26., 1./27., 1./28., 1./29., 1./30., 
			   1./31};
static Real EPSL = 1e-40; //1e-6
static Real optimw[3] = {1./10., 3./5., 3./10.};// WENO optimal weights
static Real othreeotwo  = 13./12.;

using namespace gra::aliases;
using namespace LinearAlgebra;
Real Reconstruct(Real char_flx[5])
{
  static Real othreeotwo  = 13./12.;
  Real a[3], b[3], w[3], fk[3], dsa;
  Real fimt, fimo, fi, fipo, fipt;
  Real fLipoh;
  int j;
  fipt = char_flx[4];
  fipo = char_flx[3];
  fi   = char_flx[2];
  fimo = char_flx[1];
  fimt = char_flx[0];
#if 1
  for( j = 0 ; j<3; j++) w[j] = optimw[j];
#endif
#if 0
// smoothness coefs, Jiag & Shu '96
  b[0] = othreeotwo * SQR(( fimt - cc[2]*fimo + fi   )) + oocc[4] * SQR((       fimt - cc[4]*fimo + cc[3]*fi   ) ); 
  b[1] = othreeotwo * SQR(( fimo - cc[2]*fi   + fipo )) + oocc[4] * SQR((       fimo -       fipo              ) ); 
  b[2] = othreeotwo * SQR(( fi   - cc[2]*fipo + fipt )) + oocc[4] * SQR(( cc[3]*fi   - cc[4]*fipo +       fipt ) );   
  for( j = 0 ; j<3; j++) a[j] = optimw[j]/(SQR(( EPSL + b[j])));
  dsa = 1./( a[0] + a[1] + a[2] );
  for( j = 0 ; j<3; j++) w[j] = a[j] * dsa;
#endif
#if 0
  // smoothness coefs, Borges '08 (wenoZ)
  b[0] = othreeotwo * SQR(( fimt - cc[2]*fimo + fi   )) + oocc[4] * SQR((       fimt - cc[4]*fimo + cc[3]*fi   ) ); 
  b[1] = othreeotwo * SQR(( fimo - cc[2]*fi   + fipo )) + oocc[4] * SQR((       fimo -       fipo              ) ); 
  b[2] = othreeotwo * SQR(( fi   - cc[2]*fipo + fipt )) + oocc[4] * SQR(( cc[3]*fi   - cc[4]*fipo +       fipt ) );   
  for( j = 0 ; j<3; j++) a[j] = optimw[j]*( 1.+ std::abs(b[0]-b[2])/(b[j]+EPSL) );
  dsa = 1./( a[0] + a[1] + a[2] );
  for( j = 0 ; j<3; j++) w[j] = a[j] * dsa;
#endif  
  fk[0] = oocc[6]*(   cc[2]*fimt - cc[7]*fimo + cc[11]*fi   );
  fk[1] = oocc[6]*( - cc[1]*fimo + cc[5]*fi   + cc[2] *fipo );
  fk[2] = oocc[6]*(   cc[2]*fi   + cc[5]*fipo -        fipt );

  fLipoh = w[0]*fk[0] + w[1]*fk[1] + w[2]*fk[2];

  return fLipoh;
}

void GetEulerianVelocity(const int il, const int iu,
                        const int k, const int j,
                        AthenaTensor<Real, TensorSymm::SYM2, 3, 2> const &met,
                        AthenaTensor<Real, TensorSymm::NONE, 3, 1> const &w_util_u,
                        AthenaTensor<Real, TensorSymm::NONE, 3, 0> &W,
                        AthenaTensor<Real, TensorSymm::NONE, 3, 1> &w_v_u
                        )
{

  // Lorentz factors
  for (int i=il; i<=iu; ++i)
  {
    W(k,j,i) = std::sqrt(1. + InnerProductVecMetric(
      w_util_u, met, k,j,i));
  }

  // Eulerian velocity centred contravariant componenets
  for (int a=0; a<NDIM; ++a)
  {
    for (int i=il; i<= iu; ++i)
    {
      w_v_u(a,k,j,i) = w_util_u(a,k,j,i) / W(k,j,i);
    }
  }
  return;
}

void GetFluxesGRHD(
                   const int k, const int j,
                   const int ivx,const int il,const int iu,
                   AthenaTensor<Real, TensorSymm::NONE, 3, 0> const &sqrt_detmet,
                   AthenaTensor<Real, TensorSymm::NONE, 3, 0> const &alpha,
                   AthenaTensor<Real, TensorSymm::NONE, 3, 1> const &beta_u,
                   AthenaTensor<Real, TensorSymm::NONE, 3, 0> const &pgas,
                   AthenaTensor<Real, TensorSymm::NONE, 3, 1> const &v_u,
                   AthenaArray<Real> const &cons,
                   AthenaTensor<Real, TensorSymm::NONE, 5,   1> &f
                   )
{

  // calculate flux
  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    f(IDN,k,j,i) = cons(IDN,k,j,i) * alpha(k,j,i) * (
      v_u(ivx-1,k,j,i) - beta_u(ivx-1,k,j,i)/alpha(k,j,i));

    f(IEN,k,j,i) = cons(IEN,k,j,i) * alpha(k,j,i) * 
    (v_u(ivx-1,k,j,i) - beta_u(ivx-1,k,j,i)/alpha(k,j,i)) 
      + alpha(k,j,i)*sqrt_detmet(k,j,i)*pgas(k,j,i)*v_u(ivx-1,k,j,i);
  }

  for (int a=0; a<NDIM; ++a)
  {
    #pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      f(IVX+a,k,j,i) = (
        cons(IVX+a,k,j,i) * alpha(k,j,i) *
        (v_u(ivx-1,k,j,i) -
         beta_u(ivx-1,k,j,i)/alpha(k,j,i))
      );

    }
  }

  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    f(ivx,k,j,i) += pgas(k,j,i) * alpha(k,j,i) * sqrt_detmet(k,j,i);
  }

  return;

}

void GetFluxesSRHD(
                   const int k, const int j,
                   const int ivx,const int il,const int iu,
                   AthenaTensor<Real, TensorSymm::NONE, 3, 0> const &sqrt_detmet,
                   AthenaTensor<Real, TensorSymm::NONE, 3, 0> const &alpha,
                   AthenaTensor<Real, TensorSymm::NONE, 3, 1> const &beta_u,
                   AthenaTensor<Real, TensorSymm::NONE, 3, 0> const &pgas,
                   AthenaTensor<Real, TensorSymm::NONE, 3, 1> const &v_u,
                   AthenaArray<Real> const &cons,
                   AthenaTensor<Real, TensorSymm::NONE, 5,   1> &f
                   )
{

  // calculate flux
  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    f(IDN,i) = cons(IDN,k,j,i) * v_u(ivx-1,i);

    f(IEN,i) = cons(ivx,k,j,i) - cons(IDN,k,j,i)*v_u(ivx-1,i);
  }

  for (int a=0; a<NDIM; ++a)
  {
    #pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      f(IVX+a,i) = cons(IVX+a,k,j,i) * v_u(ivx-1,i);
    }
  }

  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    f(ivx,i) += pgas(k,j,i);
  }

  return;

}

void GetEigenVectorSRHD(MeshBlock * pmb,
                    const int dir, const Real avg_field[12],
                    const Real avg_hq[13], const Real avg_eig[6],
                    Real (&L)[NHYDRO][NHYDRO], 
                    Real (&R)[NHYDRO][NHYDRO])
{
Real K,Am,Ap,delta,v2,W,vx,vy,vz,lamm,lamp,h;

int ivx,ivy,ivz;

switch(dir)
{
case 1:
    ivx = 0;
    ivy = 1;
    ivz = 2;
    break;
case 2:
    ivx = 1; 
    ivy = 0;
    ivz = 2;
    break;
case 3:
    ivx = 2;
    ivy = 1;
    ivz = 0;
    break;
}

#if USETM
  const Real Gamma = pmb->peos->GetEOS().GetGamma();
#else
  const Real Gamma = pmb->peos.GetGamma();
#endif

  h=avg_hq[hi::h];
  W=avg_hq[hi::W];
  vx=avg_hq[ivx];
  vy=avg_hq[ivy];
  vz=avg_hq[ivz];
  v2=avg_hq[hi::v2];

  lamp = avg_eig[eig::Lp];
  lamm = avg_eig[eig::Lm];

  K  = h; 
  Ap = (ONE - vx*vx )/(ONE - vx*lamp);
  Am = (ONE - vx*vx )/(ONE - vx*lamm);
  delta = h*h*h*W*(K-ONE)*(ONE-vx*vx)*(Ap*lamp-Am*lamm);


  // Right Eigen Vectors

  R[IDN][0] = K/(h*W);
  R[ivx+1][0] = vx;
  R[ivy+1][0] = vy;
  R[ivz+1][0] = vz;
  R[IEN][0]  = ONE-K/(h*W);

  R[IDN][1]   = W*vy;
  R[ivx+1][1] = TWO*h*W*W*vx*vy;
  R[ivy+1][1] = h*(ONE + TWO*W*W*vy*vy);
  R[ivz+1][1] = TWO*h*W*W*vy*vz;
  R[IEN][1]   = TWO*h*W*W*vy-W*vy;

  R[IDN][2]   = W*vz;
  R[ivx+1][2] = TWO*h*W*W*vx*vz;
  R[ivy+1][2] = TWO*h*W*W*vy*vz;
  R[ivz+1][2] = h*(ONE + TWO*W*W*vz*vz);
  R[IEN][2]  = TWO*h*W*W*vz-W*vz;

  R[IDN][3]   = ONE;
  R[ivx+1][3] = h*W*Am*lamm;
  R[ivy+1][3] = h*W*vy;
  R[ivz+1][3] = h*W*vz;
  R[IEN][3]   = h*W*Am - ONE;

  R[IDN][4]   = ONE;
  R[ivx+1][4] = h*W*Ap*lamp;
  R[ivy+1][4] = h*W*vy;
  R[ivz+1][4] = h*W*vz;
  R[IEN][4]   = h*W*Ap - ONE;

  L[0][IDN]   = W/(K-ONE)*(h-W);
  L[0][ivx+1] = W/(K-ONE)*(W*vx);
  L[0][ivy+1] = W/(K-ONE)*(W*vy);
  L[0][ivz+1] = W/(K-ONE)*(W*vz);
  L[0][IEN]   = W/(K-ONE)*(-W);

  L[1][IDN]   = ONE/(h*(ONE-vx*vx))*(-vy);
  L[1][ivx+1] = ONE/(h*(ONE-vx*vx))*(vx*vy);
  L[1][ivy+1] = ONE/(h*(ONE-vx*vx))*(ONE-vx*vx);
  L[1][ivz+1] = ONE/(h*(ONE-vx*vx))*(ZERO);
  L[1][IEN]    = ONE/(h*(ONE-vx*vx))*(-vy);

  L[2][IDN]   = ONE/(h*(ONE-vx*vx))*(-vz);
  L[2][ivx+1] = ONE/(h*(ONE-vx*vx))*(vx*vz);
  L[2][ivy+1] = ONE/(h*(ONE-vx*vx))*(ZERO);
  L[2][ivz+1] = ONE/(h*(ONE-vx*vx))*(ONE-vx*vx);
  L[2][IEN]   = ONE/(h*(ONE-vx*vx))*(-vz);

  L[3][IDN]   = h*h/delta*( h*W*Ap*(vx-lamp)-vx-W*W*(v2-vx*vx)*(TWO*K-ONE)*(vx-Ap*lamp)+K*Ap*lamp );
  L[3][ivx+1] = h*h/delta*( ONE+W*W*(v2-vx*vx)*(TWO*K-ONE)*(ONE-Ap)-K*Ap );
  L[3][ivy+1] = h*h/delta*( W*W*vy*(TWO*K-ONE)*Ap*(vx-lamp) );
  L[3][ivz+1] = h*h/delta*( W*W*vz*(TWO*K-ONE)*Ap*(vx-lamp) );
  L[3][IEN]   = h*h/delta*( -vx-W*W*(v2-vx*vx)*(TWO*K-ONE)*(vx-Ap*lamp)+K*Ap*lamp );


  L[4][IDN]   = -h*h/delta*( h*W*Am*(vx-lamm)-vx-W*W*(v2-vx*vx)*(TWO*K-ONE)*(vx-Am*lamm)+K*Am*lamm );
  L[4][ivx+1] = -h*h/delta*( ONE+W*W*(v2-vx*vx)*(TWO*K-ONE)*(ONE-Am)-K*Am );
  L[4][ivy+1] = -h*h/delta*( W*W*vy*(TWO*K-ONE)*Am*(vx-lamm) );
  L[4][ivz+1] = -h*h/delta*( W*W*vz*(TWO*K-ONE)*Am*(vx-lamm) );
  L[4][IEN]   = -h*h/delta*( -vx-W*W*(v2-vx*vx)*(TWO*K-ONE)*(vx-Am*lamm)+K*Am*lamm );

  return;

}
void GetEigenVectorGRHD(MeshBlock * pmb,
                    const int dir, const Real avg_field[12],
                    const Real avg_hq[13], const Real avg_eig[6],
                    Real (&L)[NHYDRO][NHYDRO], 
                    Real (&R)[NHYDRO][NHYDRO])
{
  Real W2, tW2, h2, hW, oohW;
  Real tmpLm, tmpLp;
  Real gxxup_vxtmpLm, gxxup_vxtmpLp;
  Real Vm, Vp;
  Real Cp, Cm;
  Real Am, Ap;
  Real kk, K;  
  Real tKmo, omKAp, omKAm, omK;
  Real WoKmo; 
  Real W2oKmo;
  Real cxx, cxy, cxz;
  Real xsi, W2xsi, hxsi;
  Real oohxsi;
  Real delta, h2odelta;
  Real tmpL5m, tmpL5p;
  Real tmpx, tmpy, tmpz;
  Real vx2mo;

  Real gxx,gxy,gxz,gyy,gyz,gzz,gxxup,detg,h,W,rho,cs2,
      kappa,vx,vy,vz,vlowx,vlowy,vlowz,lamm,lamp,alpha,betax;
      

  int ivx,ivy,ivz;

  switch(dir)
  {
  case 1:
      ivx = 0;
      ivy = 1;
      ivz = 2;
      break;
  case 2:
      ivx = 1; 
      ivy = 0;
      ivz = 2;
      break;
  case 3:
      ivx = 2;
      ivy = 1;
      ivz = 0;
      break;
  }

#if USETM
  const Real Gamma = pmb->peos->GetEOS().GetGamma();
#else 
  const Real Gamma=pmb->peos.GetGamma()
#endif

  gxx=avg_field[fi::gxx+ivx];
  gyy=avg_field[fi::gxx+ivy];
  gzz=avg_field[fi::gxx+ivz];
  gxy=(dir == 1 || dir ==2) ? avg_field[fi::gxy] : avg_field[fi::gyz] ;
  gxz=(dir == 1 || dir ==3) ? avg_field[fi::gxz] : avg_field[fi::gyz];  
  gyz=avg_field[fi::gyz-ivx];
  gxxup=avg_field[fi::gxx_u + dir - 1];
  detg= avg_field[fi::det_g];
  betax=avg_field[fi::bx];
  alpha=avg_field[fi::a];
  rho=avg_hq[hi::rho];
  vx=avg_hq[ivx];
  vy=avg_hq[ivy];
  vz=avg_hq[ivz];
  vlowx=avg_hq[hi::vx_d+ivx];
  vlowy=avg_hq[hi::vx_d+ivy];
  vlowz=avg_hq[hi::vx_d+ivz];
  h=avg_hq[hi::h];
  W=avg_hq[hi::W];
  cs2=avg_eig[eig::cs2];
  lamp=avg_eig[eig::Lp];
  lamm=avg_eig[eig::Lm];

  W2  = W*W;
  tW2 = TWO*W2;
  h2  = h*h;
  hW = h*W;
  if(hW!=0.0)  oohW = 1.0/hW;
  else         oohW = BIG;
  tmpLm = (alpha<=ZERO) ? BIG : (lamm+betax)/alpha;
  tmpLp = (alpha<=ZERO) ? BIG : (lamp+betax)/alpha;
  gxxup_vxtmpLm = gxxup-vx*tmpLm;
  gxxup_vxtmpLp = gxxup-vx*tmpLp;

  if(gxxup_vxtmpLm==0.0)  gxxup_vxtmpLm = TINY;
  if(gxxup_vxtmpLp==0.0)  gxxup_vxtmpLp = TINY;

  Vm = (vx-tmpLm)/(gxxup_vxtmpLm);
  Vp = (vx-tmpLp)/(gxxup_vxtmpLp);

  Cp = vlowx-Vp;
  Cm = vlowx-Vm;

  Am = (gxxup-vx*vx)/(gxxup_vxtmpLm);
  Ap = (gxxup-vx*vx)/(gxxup_vxtmpLp);

  kappa = avg_eig[eig::kappa];
  if(rho!=0.0)  kk = kappa/rho;
  else          kk = BIG;
  if(kk!=cs2)   K  = kk/(kk-cs2);
  else          K  = BIG;

  tKmo  = (TWO * K - ONE);
  omKAp = (ONE - K * Ap);
  omKAm = (ONE - K * Am);
  omK   = (ONE - K);

  /* WoKmo = -W/omK; */
  if(omK!=0.0)  WoKmo = -W/omK;
  else          WoKmo = BIG;

  W2oKmo = WoKmo*W;

  cxx = gyy * gzz - gyz * gyz;
  cxy = gxz * gyz - gxy * gzz;
  cxz = gxy * gyz - gxz * gyy;

  xsi    = cxx - detg * vx * vx;
  W2xsi  = W2 * xsi;
  hxsi   = h*xsi;
  //oohxsi = ONE/hxsi; 
  oohxsi = (hxsi==ZERO) ? BIG : ONE/hxsi;

  delta    = h2 * hW * omK * (Cm - Cp) * xsi; 
  //h2odelta = h2 / delta; 
  h2odelta = (delta==ZERO) ? BIG : h2/delta;

  // note:    m <---------> p
  tmpL5m = ( omK * (Vp * (W2xsi - cxx) - detg * vx) - K * W2xsi * Vp );
  tmpL5p = ( omK * (Vm * (W2xsi - cxx) - detg * vx) - K * W2xsi * Vm );

  tmpx = tKmo * (W2xsi * vx - cxx * vx);
  tmpy = tKmo * (W2xsi * vy - cxy * vx);
  tmpz = tKmo * (W2xsi * vz - cxz * vx);

  vx2mo = (vx * vlowx - ONE);

  // RIGHT EIGEN VECTORS

  R[IDN][0] = K*oohW;
  R[ivx+1][0] = vlowx;
  R[ivy+1][0] = vlowy;
  R[ivz+1][0] = vlowz;
  R[IEN][0]  = ONE-K*oohW;

  R[IDN][1]  =  W * vlowy;
  R[ivx+1][1] =  h * (gxy + tW2 * vlowy * vlowx);
  R[ivy+1][1] =  h * (gyy + tW2 * vlowy * vlowy);
  R[ivz+1][1]=  h * (gyz + tW2 * vlowy * vlowz);
  R[IEN][1]   =  W * vlowy * (TWO * hW - ONE);

  R[IDN][2]   = W * vlowz;
  R[ivx+1][2] = h * (gxz + tW2 * vlowz * vlowx);
  R[ivy+1][2] = h * (gyz + tW2 * vlowz * vlowy);
  R[ivz+1][2] = h * (gzz + tW2 * vlowz * vlowz);
  R[IEN][2]  = W * vlowz * (TWO * hW - ONE);

  R[IDN][3]   = ONE;
  R[ivx+1][3] = hW * Cm;
  R[ivy+1][3] = hW * vlowy;
  R[ivz+1][3] = hW * vlowz;
  R[IEN][3]   = hW * Am - ONE;

  R[IDN][4]   = ONE;
  R[ivx+1][4] = hW * Cp;
  R[ivy+1][4] = hW * vlowy;
  R[ivz+1][4] = hW * vlowz;
  R[IEN][4]   = hW * Ap - ONE;

  // LEFT EIGEN VECTORS

  L[0][IDN]   = WoKmo * (h - W);
  L[0][ivx+1] = W2oKmo * vx;
  L[0][ivy+1] = W2oKmo * vy;
  L[0][ivz+1] = W2oKmo * vz;
  L[0][IEN]  = -W2oKmo;

  L[1][IDN]  = oohxsi * (gyz *   vlowz  - gzz * vlowy     );
  L[1][ivx+1] = oohxsi * (gzz *   vlowy  - gyz * vlowz     ) * vx;
  L[1][ivy+1] = oohxsi * (gzz * (-vx2mo) + gxz * vlowz * vx);
  L[1][ivz+1] = oohxsi * (gyz *   vx2mo  - gxz * vlowy * vx);
  L[1][IEN]   = oohxsi * (gyz *   vlowz  - gzz * vlowy     );

  L[2][IDN]   = oohxsi * (gyz *   vlowy  - gyy * vlowz     );
  L[2][ivx+1] = oohxsi * (gyy *   vlowz  - gyz * vlowy     ) * vx;
  L[2][ivy+1] = oohxsi * (gyz *   vx2mo  - gxy * vlowz * vx);
  L[2][ivz+1] = oohxsi * (gyy * (-vx2mo) + gxy * vlowy * vx);
  L[2][IEN]   = oohxsi * (gyz *   vlowy  - gyy * vlowz     );

  L[3][IDN]   = h2odelta * ( hW  * Vp * xsi + tmpL5m );
  L[3][ivx+1] = h2odelta * ( cxx * omKAp + Vp * tmpx );
  L[3][ivy+1] = h2odelta * ( cxy * omKAp + Vp * tmpy );
  L[3][ivz+1] = h2odelta * ( cxz * omKAp + Vp * tmpz );
  L[3][IEN]   = h2odelta * ( tmpL5m );

  L[4][IDN]   = - h2odelta * ( hW  * Vm * xsi + tmpL5p );
  L[4][ivx+1] = - h2odelta * ( cxx * omKAm + Vm * tmpx );
  L[4][ivy+1] = - h2odelta * ( cxy * omKAm + Vm * tmpy );
  L[4][ivz+1] = - h2odelta * ( cxz * omKAm + Vm * tmpz );
  L[4][IEN]  = - h2odelta * ( tmpL5p );

   return;
}

void GetEigenVectorGRHD1(MeshBlock * pmb,
                    const int k, const int j,
                    const int i,
                    const int dir, const Real avg_field[12],
                    const Real avg_hq[13], const Real avg_eig[6],
                    AthenaTensor<Real, TensorSymm::NONE, 5,2> &L, 
                    AthenaTensor<Real, TensorSymm::NONE, 5,2> &R)
{
Real e, K,kk,L_p,L_m, A_p,A_m,A_p_,A_m_ ,V_p,V_m, N_p, N_m,C_p,C_m,
det, E, G_xx, G_xy, G_xz ;
Real gxx,gxy,gxz,gyy,gyz,gzz,gxxup,detg,h,W,W2,hW,rho,cs2,
    kappa,vx,vy,vz,vlowx,vlowy,vlowz,lamm,lamp,alpha,betax;

Real  m_fac1,m_fac2,m_fac3;

int ivx,ivy,ivz;

switch(dir)
{
case 1:
    ivx = 0;
    ivy = 1;
    ivz = 2;
    break;
case 2:
    ivx = 1; 
    ivy = 0;
    ivz = 2;
    break;
case 3:
    ivx = 2;
    ivy = 1;
    ivz = 0;
    break;
}

  gxx=avg_field[fi::gxx+ivx];
  gyy=avg_field[fi::gxx+ivy];
  gzz=avg_field[fi::gxx+ivz];
  gxy=(dir == 1 || dir ==2) ? avg_field[fi::gxy] : avg_field[fi::gyz] ;
  gxz=(dir == 1 || dir ==3) ? avg_field[fi::gxz] : avg_field[fi::gyz];  
  gyz=avg_field[fi::gyz-ivx];
  gxxup=avg_field[fi::gxx_u + dir - 1];
  detg= avg_field[fi::det_g];
  betax=avg_field[fi::bx];
  alpha=avg_field[fi::a];
  rho=avg_hq[hi::rho];
  vx=avg_hq[ivx];
  vy=avg_hq[ivy];
  vz=avg_hq[ivz];
  vlowx=avg_hq[hi::vx_d+ivx];
  vlowy=avg_hq[hi::vx_d+ivy];
  vlowz=avg_hq[hi::vx_d+ivz];
  h=avg_hq[hi::h];
  W=avg_hq[hi::W];
  cs2=avg_eig[eig::cs2];
  L_p=avg_eig[eig::Lp];
  L_m=avg_eig[eig::Lm];

  W2  = W*W;
  Real tW2 = TWO*W2;
  Real h2  = h*h;
  hW = h*W;

  kappa = avg_eig[eig::kappa];
  if(rho!=0.0)  kk = kappa/rho;
  else          kk = BIG;
  if(kk!=cs2)   K  = kk/(kk-cs2);
  else          K  = BIG;

  A_p = (L_p + betax) / alpha;
  A_m = (L_m + betax) / alpha;
  A_p_= (gxxup - vx*vx) / (gxxup - vx*A_p);
  A_m_= (gxxup - vx*vx) / (gxxup - vx*A_m);
  V_p = (vx - A_p) / (gxxup - vx*A_p);
  V_m = (vx - A_m) / (gxxup - vx*A_m);

    // Right Eigen Vectors

    R(IDN,IDN,i)   =   1.;
    R(ivx+1,IDN,i) = h* W * ( vlowx - V_m);
    R(ivy+1,IDN,i) = h * W * vlowy;
    R(ivz+1,IDN,i) = h * W * vlowz;
    R(IEN,IDN,i)   = h * W * A_m_  - 1. ;

    R(IDN, IVX, i) = K/(h* W);
    R(ivx+1,IVX,i) = vlowx;
    R(ivy+1,IVX,i) = vlowy;
    R(ivz+1,IVX,i) = vlowz;
    R(IEN,IVX,i)   = 1. - K/(h*W);

    R(IDN,IVY,i)   = W* vlowy;
    R(ivx+1,IVY,i) = h*(gxy + 2. * W*W * vlowx*vlowy);
    R(ivy+1,IVY,i) = h*(gyy + 2. * W*W* vlowy*vlowy);
    R(ivz+1,IVY,i) = h*(gyz + 2. * W*W* vlowz*vlowy);
    R(IEN,IVY,i)   = W*vlowy*(2. * h *W - 1.);

    R(IDN,IVZ,i)   = W*vlowz;
    R(ivx+1,IVZ,i) = h*(gxz + 2. * W*W * vlowx*vlowz);
    R(ivy+1,IVZ,i) = h*(gyz + 2. * W*W * vlowy*vlowz);
    R(ivz+1,IVZ,i) = h*(gzz+ 2. * W*W * vlowz*vlowz);
    R(IEN, IVZ,i)  = W*vlowz*(2. * h *W - 1.);

    R(IDN,IEN,i)   = 1.;
    R(ivx+1,IEN,i) =  h* W * ( vlowx - V_p);
    R(ivy+1,IEN,i) =  h * W * vlowy;
    R(ivz+1,IEN,i) =   h * W * vlowz;
    R(IEN,IEN,i)   =  h * W * A_p_  - 1.;

    // LEFT Eigen Vectors
    G_xx = gyy*gzz -gyz*gyz;
    G_xy = -(gxy*gzz - gxz*gyz);
    G_xz = gxy*gyz -gyy*gxz;
    E    = G_xx - detg* vx*vx;
    C_p  = vlowx - V_p;
    C_m  = vlowx - V_m;
    N_p  = (1. -K)*(-detg*vx + V_m*(W*W*E -G_xx )) - K*W*W*V_m*E;
    N_m  = (1. -K)*(-detg*vx + V_p*(W*W*E -G_xx)) - K*W*W*V_p*E;
    
    // Determinant of eigen vector matrix
    det  = h*h*h *W*(K-1.)*(C_p -C_m)*E;

    m_fac1 = (h*h)/det;
    m_fac2 = W/(K-1.);
    m_fac3 = 1./(h*E);
    
    
    L(IDN,IDN,i)   = m_fac1 * ( h*W*V_p*E - N_m);
    L(IDN,ivx+1,i) = m_fac1 * (G_xx*(1. - K*A_p_) + (2.*K - 1.)* V_p * (W*W * vx * E - G_xx * vx) );
    L(IDN,ivy+1,i) = m_fac1 * (G_xy*(1. - K*A_p_) + (2.*K - 1.)* V_p * (W*W * vy * E - G_xy * vx) );
    L(IDN,ivz+1,i) = m_fac1 * (G_xz*(1. - K*A_p_) + (2.*K - 1.)* V_p * (W*W * vz * E - G_xz * vx) );
    L(IDN,IEN,i)   = m_fac1*N_m;

    L(IVX,IDN,i)   = m_fac2 *(h-W);
    L(IVX,ivx+1,i) = m_fac2 *(W*vx);
    L(IVX,ivy+1,i) = m_fac2 *(W*vy);
    L(IVX,ivz+1,i) = m_fac2 *(W*vz);
    L(IVX,IEN,i)   = m_fac2 *(-W);

    L(IVY,IDN,i)   = m_fac3 *(-gzz*vlowy + gyz*vlowz);
    L(IVY,ivx+1,i) = m_fac3 *vx*(gzz*vlowy - gyz*vlowz);
    L(IVY,ivy+1,i) = m_fac3 *(gzz * (1. - vlowx*vx) + gxz*vlowz*vx);
    L(IVY,ivz+1,i) = m_fac3 *(-gyz * (1. - vlowx*vx) - gxz*vlowy*vx);
    L(IVY,IEN,i)   = m_fac3 *(-gzz*vlowy + gyz*vlowz);

    L(IVZ,IDN,i)   = m_fac3 *(-gyy*vlowz + gyz*vlowy);
    L(IVZ,ivx+1,i) = m_fac3 *vx*(gyy*vlowz - gyz*vlowy);
    L(IVZ,ivy+1,i) = m_fac3 *(-gyz * (1. - vlowx*vx) - gxy*vlowz*vx);
    L(IVZ,ivz+1,i) = m_fac3 *(gyy * (1. - vlowx*vx) + gxy*vlowy*vx);
    L(IVZ,IEN,i)   = m_fac3 *(-gyy*vlowz + gyz*vlowy);

    L(IEN,IDN,i)   = -m_fac1 * ( h*W*V_m*E - N_p);
    L(IEN,ivx+1,i) = -m_fac1 * (G_xx*(1. - K*A_m_) + (2.*K - 1.)* V_m * (W*W * vx * E - G_xx * vx) );
    L(IEN,ivy+1,i) = -m_fac1 * (G_xy*(1. - K*A_m_) + (2.*K - 1.)* V_m * (W*W * vy * E - G_xy * vx) );
    L(IEN,ivz+1,i) = -m_fac1 * (G_xz*(1. - K*A_m_) + (2.*K - 1.)* V_m * (W*W * vz * E - G_xz * vx) );
    L(IEN,IEN,i)   = -m_fac1*N_p;

   return;

}


void GetEigenValues(MeshBlock * pmb,
          const int k, const int j,
          const int il, const int iu,
          const int ivx,
          AthenaTensor<Real, TensorSymm::SYM2, 3, 2> const &met,
          AthenaTensor<Real, TensorSymm::SYM2, 3, 2> const &met_inv,
          AthenaTensor<Real, TensorSymm::NONE, 3, 0> const &alpha,
          AthenaTensor<Real, TensorSymm::NONE, 3, 0> const &rho,
          AthenaTensor<Real, TensorSymm::NONE, 3, 0> const &pgas,
          AthenaTensor<Real, TensorSymm::NONE, 3, 1> const &beta,
          AthenaTensor<Real, TensorSymm::NONE, 3, 1> const &v_u,
          AthenaTensor<Real, TensorSymm::NONE, 3, 0> const &w_norm2_v_,
          AthenaTensor<Real, TensorSymm::NONE, 5,   1> &lambda
        )
{
  Real w_hrho;
  AT_N_sca  lambda_p(iu+1);
  AT_N_sca  lambda_m(iu+1);
  
#if USETM
  const Real mb = pmb->peos->GetEOS().GetBaryonMass();
  Real Y[MAX_SPECIES] = {0.0};
  Real Gamma = pmb->peos->GetEOS().GetGamma();
  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  { 
#if 0  
    Real Csqrt,epsl,cs2,h;
    epsl = pgas(k,j,i)/((Gamma-1)*rho(k,j,i));
    h=1.0 + epsl + pgas(k,j,i)/rho(k,j,i);
    Real kappa = (Gamma-1.0)*rho(k,j,i);
    Real chi   = (Gamma -1.0)*epsl;
    cs2 = ( chi + (pgas(k,j,i)*kappa)/SQR(rho(k,j,i)) )/h;
    Csqrt  = sqrt(std::abs(cs2));
    lambda_m(i)  = (v_u(ivx-1,k,j,i) - Csqrt)/(ONE - v_u(ivx-1,k,j,i)*Csqrt); // lam-
    lambda_p(i)  = (v_u(ivx-1,k,j,i) + Csqrt)/(ONE + v_u(ivx-1,k,j,i)*Csqrt); // lam+
#endif
#if 1
    const Real n = rho(k,j,i) / mb;

    for (int m=0; m<NSCALARS; m++)
    {
      Y[m] = pmb->pscalars->r(m,k,j,i);
    }
    const Real T = pmb->peos->GetEOS().GetTemperatureFromP(
      n, pgas(k,j,i), Y);
    pmb->peos->SoundSpeedsGR(n, T,
                             v_u(ivx-1,k,j,i),
                             w_norm2_v_(k,j,i),
                             alpha(k,j,i),
                             beta(ivx-1,k,j,i),
                             met_inv(ivx-1,ivx-1,k,j,i),
                             &lambda_p(i),
                             &lambda_m(i),
                             Y);
#endif
  }

#else
  const Real Gamma = pmb->peos->GetGamma();
  
  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    w_hrho = rho(k,j,i)`;

    pmb->peos->SoundSpeedsGR(w_hrho,
                             pgas(k,j,i),
                             v_u(ivx-1,k,j,i),
                             w_norm2_v_(k,j,i),
                             alpha(k,j,i),
                             beta(ivx-1,k,j,i),
                             met_inv(ivx-1,ivx-1,i),
                             &lambda_p(i),
                             &lambda_m(i));
  }
#endif

  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {

    lambda(0,k,j,i) = lambda_m(i);

    // lambda_0 (three-fold degeneracy)
    lambda(1,k,j,i) = alpha(k,j,i) * v_u(ivx-1,k,j,i) - beta(ivx-1,k,j,i);
    lambda(2,k,j,i) = lambda(1,k,j,i);
    lambda(3,k,j,i) = lambda(1,k,j,i);

    lambda(4,k,j,i) = lambda_p(i);
  }

  return;

}

void GetMaximalWaveSpeed(const int k, const int j,const int i,
                        const int stensil, const int ivx,
                        AthenaTensor<Real, TensorSymm::NONE, 5,1> const &lambda,
                        Real lambda_max[NHYDRO])
{  
  switch(ivx)
  {
    case(1): 
    {
      Real amax[3]={};
      for (int n = 0; n<3;++n) amax[n]=0.;
      for(int s = i-stensil; s<=i+stensil ; ++s)
      {
        amax[0]=std::max(amax[0], std::abs(lambda(0,k,j,s)));
        amax[1]=std::max(amax[1], std::abs(lambda(4,k,j,s)));
        amax[2]=std::max(amax[2], std::abs(lambda(1,k,j,s)));
      }
      lambda_max[0] = amax[2];
      lambda_max[1] = amax[2];
      lambda_max[2] = amax[2];
      lambda_max[3] = amax[0];
      lambda_max[4] = amax[1];
      break;
    }

    case(2):
    {
      Real amax[3]={};
      for (int n = 0; n<3;++n) amax[n]=0.;
      for(int s = j-stensil; s<=j+stensil ; ++s)
      {
        amax[0]=std::max(amax[0], std::abs(lambda(0,k,s,i)));
        amax[1]=std::max(amax[1], std::abs(lambda(4,k,s,i)));
        amax[2]=std::max(amax[2], std::abs(lambda(1,k,s,i)));
      }
      lambda_max[0] = amax[2];
      lambda_max[1] = amax[2];
      lambda_max[2] = amax[2];
      lambda_max[3] = amax[0];
      lambda_max[4] = amax[1];
      break;
    }

    case(3):
    {
      Real amax[3]={};
      for (int n = 0; n<3;++n) amax[n]=0.;
      for(int s = k-stensil; s<=k+stensil ; ++s)
      {
        amax[0]=std::max(amax[0], std::abs(lambda(0,s,j,i)));
        amax[1]=std::max(amax[1], std::abs(lambda(4,s,j,i)));
        amax[2]=std::max(amax[2], std::abs(lambda(1,s,j,i)));
      }
      lambda_max[0] = amax[2];
      lambda_max[1] = amax[2];
      lambda_max[2] = amax[2];
      lambda_max[3] = amax[0];
      lambda_max[4] = amax[1];
      break;
    }

  }
  return;      
}                      

void ReconCharFields(const int k, const int j,
                    const int i,const int stensil,const int ivx,
                    AthenaTensor<Real, TensorSymm::NONE, NHYDRO,1> const &flx,
                    Real lambda_max[NHYDRO],
                    AthenaArray<Real> const &cons,
                    const Real (&L_eig)[NHYDRO][NHYDRO],
                    Real char_flx[NHYDRO]
                    )
{
  Real fac = 0.5;
  Real flux_stensil_p[5]={};
  Real flux_stensil_m[5]={};
  switch(ivx)
  {
    case (1):
    {
      for( int m = 0 ; m< NHYDRO; ++m )
      {
        int i_p = i-stensil;
        int i_m = i+stensil-1;

        for(int s = 0; s<5;++s)
        {
          flux_stensil_p[s]=0.;
          flux_stensil_m[s]=0.;
          for( int l = 0; l<NHYDRO ;++l)
          {
            flux_stensil_p[s] += fac * L_eig[m][l]*(flx(l,k,j,i_p) + lambda_max[m]*cons(l,k,j,i_p));
            flux_stensil_m[s] += fac * L_eig[m][l]*(flx(l,k,j,i_m) - lambda_max[m]*cons(l,k,j,i_m));
          }
          i_p+=1;
          i_m-=1;
        }

        char_flx[m] = Reconstruct(flux_stensil_p) + Reconstruct(flux_stensil_m);
      }
      break;
    }

    case (2):
    {
      for( int m = 0 ; m< NHYDRO; ++m )
      {
        int j_p = j-stensil;
        int j_m = j+stensil-1;

        for(int s = 0; s<5;++s)
        {
          flux_stensil_p[s]=0.;
          flux_stensil_m[s]=0.;
          for( int l = 0; l<NHYDRO ;++l)
          {
            flux_stensil_p[s] += fac * L_eig[m][l]*(flx(l,k,j_p,i) + lambda_max[m]*cons(l,k,j_p,i));
            flux_stensil_m[s] += fac * L_eig[m][l]*(flx(l,k,j_m,i) - lambda_max[m]*cons(l,k,j_m,i));
          }
          j_p+=1;
          j_m-=1;
        }

        char_flx[m] = Reconstruct(flux_stensil_p) + Reconstruct(flux_stensil_m);
      }
      break;
    }

    case (3):
    {
      for( int m = 0 ; m< NHYDRO; ++m )
      {
        int k_p = k-stensil;
        int k_m = k+stensil-1;

        for(int s = 0; s<5;++s)
        {
          flux_stensil_p[s]=0.;
          flux_stensil_m[s]=0.;
          for( int l = 0; l<NHYDRO ;++l)
          {
            flux_stensil_p[s] += fac * L_eig[m][l]*(flx(l,k_p,j,i) + lambda_max[m]*cons(l,k_p,j,i));
            flux_stensil_m[s] += fac * L_eig[m][l]*(flx(l,k_m,j,i) - lambda_max[m]*cons(l,k_m,j,i));
          }
          k_p+=1;
          k_m-=1;
        }

        char_flx[m] = Reconstruct(flux_stensil_p) + Reconstruct(flux_stensil_m);
      }
      break;
    }
  }

  return;
}

void ReconFlux(const int k, const int j,const int i,const int ivx,
              const Real char_flx[NHYDRO],
              const Real (&R_eig)[NHYDRO][NHYDRO],
              AthenaArray<Real> &flx
              )
{
  Real rflx;
  for (int m=0 ; m<NHYDRO ;++m)
  {
    rflx=0.;
    for (int n=0 ; n<NHYDRO ;++n)
    {
      rflx += R_eig[m][n] *char_flx[n];
    }
    flx(m,k,j,i)=rflx;
  }

  return;
}

}
namespace averages::grhd{
using namespace characterisiticfields;

void metcontractionpt(Real vx, Real vy,Real vz,
                      Real gxx,Real gxy,Real gxz,
                      Real gyy,Real gyz,Real gzz,
                      Real *vx_d, Real *vy_d, Real *vz_d,
                      Real *v2)
{
  Real vlx = gxx*vx + gxy*vy + gxz*vz;
  Real vly = gxy*vx + gyy*vy + gyz*vz; 
  Real vlz = gxz*vx + gyz*vy + gzz*vz;

  *vx_d = vlx;
  *vy_d = vly;
  *vz_d = vlz;
  *v2   = vlx*vx + vly*vy + vlz*vz;
}

void GetAvgs(MeshBlock * pmb,int k, int j,int i, const int ivx,
        AthenaTensor<Real, TensorSymm::SYM2, 3, 2> const &met,
        AthenaTensor<Real, TensorSymm::SYM2, 3, 2> const &met_inv,
        AthenaTensor<Real, TensorSymm::NONE, 3, 0> const &det_met,
        AthenaTensor<Real, TensorSymm::NONE, 3, 1> const &betacc,
        AthenaTensor<Real, TensorSymm::NONE, 3, 0> const &alphacc,
        AthenaTensor<Real, TensorSymm::NONE, 3, 0> const &rhocc,
        AthenaTensor<Real, TensorSymm::NONE, 3, 0> const &pcc,
        AthenaTensor<Real, TensorSymm::NONE, 3, 1> const &vucc,
        AthenaTensor<Real, TensorSymm::NONE, 3, 1> const &vd,
        AthenaTensor<Real, TensorSymm::NONE, 3, 0> const &vnorm,
        AthenaTensor<Real, TensorSymm::NONE, 3, 0> const &Wcc,
        Real avg_field[12], Real avg_hq[13], Real avg_eig[6])
{

  int i1,i2,k1,k2,j1,j2;
  const Real Gamma = pmb->peos->GetEOS().GetGamma();
  const Real mb = pmb->peos->GetEOS().GetBaryonMass();
  Real Gammamo=Gamma - 1.0;
  Real tmp1,tmp2,tmp3;
  const Real vmax = 0.999;
  const Real Wmax = 1e8;

  Real gxx,gxy,gxz,gyy,gyz,gzz,gxx_u,gyy_u,gzz_u,det_g,betax,alpha;
  Real vx,vy,vz,vx_d,vy_d,vz_d,v2,rho,p,epsl,W,h,T,n;
  Real h1,h2, epsl1,epsl2;
  Real kappa,chi,cs,cs2,L0,Lp,Lm;
  Real Y[MAX_SPECIES] = {0.0};
  int choice=1;
  switch(ivx)
  {
  case  (1):
    {
      i1=i-1;i2=i;j1=j;j2=j;k1=k;k2=k;
      break;
    }
  case (2):
    {
      i1=i;i2=i;j1=j-1;j2=j;k1=k;k2=k;
      break;
    }
  case (3):
    {
      i1=i;i2=i;j1=j;j2=j;k1=k-1;k2=k;
      break;
    }
  }
  
  // Field Averages
  gxx  = 0.5*(met(0,0,k1,j1,i1) +met(0,0,k2,j2,i2));
  gxy  = 0.5*(met(0,1,k1,j1,i1) +met(0,1,k2,j2,i2));
  gxz  = 0.5*(met(0,2,k1,j1,i1)  +met(0,2,k2,j2,i2));
  gyy  = 0.5*(met(1,1,k1,j1,i1) +met(1,1,k2,j2,i2));
  gyz  = 0.5*(met(1,2,k1,j1,i1) +met(1,2,k2,j2,i2));
  gzz  = 0.5*(met(2,2,k1,j1,i1) +met(2,2,k2,j2,i2));
  betax = 0.5*(betacc(ivx-1,k1,j1,i1) + betacc(ivx-1,k2,j2,i2));
  alpha = 0.5*(alphacc(k1,j1,i1) + alphacc(k2,j2,i2));

  // Hydro Quantities
  rho  = 0.5*(rhocc(k2,j2,i2) + rhocc(k1,j1,i1));
  epsl1 = pcc(k1,j1,i1)/(Gammamo*rhocc(k1,j1,i1)) ;
  epsl2 = pcc(k2,j2,i2)/(Gammamo*rhocc(k2,j2,i2)) ;
  epsl = 0.5*(epsl1 + epsl2);
  vx   = 0.5*(vucc(0,k2,j2,i2) + vucc(0,k1,j1,i1));
  vy   = 0.5*(vucc(1,k2,j2,i2) + vucc(1,k1,j1,i1));
  vz   = 0.5*(vucc(2,k2,j2,i2) + vucc(2,k1,j1,i1));

  //Calculate other quantities from these averages

  if (choice == 1 )
  {
    // inverse metric and detg
    det_g = Det3Metric(gxx, gxy, gxz, gyy, gyz, gzz);

    Inv3Metric(1.0/det_g,gxx,gxy, gxz,gyy,gyz,gzz,
        &gxx_u,&tmp1, &tmp2,&gyy_u,&tmp2, &gzz_u);
    
    metcontractionpt(vx, vy,vz,gxx,gxy,gxz,gyy,gyz,gzz,
                   &vx_d, &vy_d, &vz_d,&v2);

    p  = Gammamo * rho * epsl;
    h  = 1.0 + epsl +  p/rho;
    v2 = std::min(v2, vmax);
    if (v2 == vmax) W = Wmax;
    else W = ONE/std::sqrt(ONE - v2 );
    n = rho / mb;
    T = pmb->peos->GetEOS().GetTemperatureFromP(n, p, Y);

    chi    = Gammamo*epsl;
    kappa  = Gammamo*rho;
    //cs2    = (chi + (p*kappa/SQR(rho)) ) / h;
    //Real cs = std::sqrt(std::abs(cs2));
    cs = pmb->peos->GetEOS().GetSoundSpeed(n,T,Y);
    cs2 = SQR(cs);
  }

  if (choice == 2)
  {
    gxx_u = 0.5 * (met_inv(0,0,k1,j1,i1) + met_inv(0,0,k2,j2,i2));
    gyy_u = 0.5 * (met_inv(1,1,k1,j1,i1) + met_inv(1,1,k2,j2,i2));
    gyy_u = 0.5 * (met_inv(2,2,k1,j1,i1) + met_inv(2,2,k2,j2,i2));
    det_g = 0.5 *(det_met(k1,j1,i1)  + det_met(k2,j2,i2));

    metcontractionpt(vx, vy,vz,gxx,gxy,gxz,gyy,gyz,gzz,
                   &vx_d, &vy_d, &vz_d,&v2);
    
    p  = Gammamo * rho * epsl;
    h  = 1.0 + epsl +  p/rho;
    v2 = std::min(v2, vmax);
    if (v2 == vmax) W = Wmax;
    else W = ONE/std::sqrt(ONE - v2 );
    n = rho / mb;
    T = pmb->peos->GetEOS().GetTemperatureFromP(n, p, Y);

    chi    = Gammamo*epsl;
    kappa  = Gammamo*rho;
    //cs2    = (chi + (p*kappa/SQR(rho)) ) / h;
    //Real cs = std::sqrt(std::abs(cs2));
    cs = pmb->peos->GetEOS().GetSoundSpeed(n,T,Y);
    cs2 = SQR(cs);
  }

  if (choice == 3)
  {
    gxx_u = 0.5 * (met_inv(0,0,k1,j1,i1) + met_inv(0,0,k2,j2,i2));
    gyy_u = 0.5 * (met_inv(1,1,k1,j1,i1) + met_inv(1,1,k2,j2,i2));
    gyy_u = 0.5 * (met_inv(2,2,k1,j1,i1) + met_inv(2,2,k2,j2,i2));
    det_g = 0.5 *(det_met(k1,j1,i1)  + det_met(k2,j2,i2));

    v2 = 0.5 * (vnorm(k1,j1,i1) + vnorm(k2,j2,i1));
    vx_d = 0.5 * (vd(0,k1,j1,i1) + vd(0,k2,j2,i2));
    vy_d = 0.5 * (vd(1,k1,j1,i1) + vd(1,k2,j2,i2));
    vz_d = 0.5 * (vd(2,k1,j1,i1) + vd(2,k2,j2,i2));
    W = 0.5 *(Wcc(k1,j1,i1) + Wcc(k2,j2,i2));

    v2 = std::min(v2, vmax);
    if (v2 == vmax) W = Wmax;
    else W = ONE/std::sqrt(ONE - v2 );

    p = 0.5 *(pcc(k1,j1,i1) + pcc(k2,j2,i2));
    h1 = 1.0 + epsl1 + pcc(k1,j1,i1)/ rhocc(k1,j1,i1);
    h2 = 1.0 + epsl2 + pcc(k2,j2,i2)/ rhocc(k2,j2,i2);
    h = 0.5 *(h1 + h2);
    n = rho / mb;
    T = pmb->peos->GetEOS().GetTemperatureFromP(n, p, Y);

    chi    = Gammamo*epsl;
    kappa  = Gammamo*rho;
    //cs2    = (chi + (p*kappa/SQR(rho)) ) / h;
    //Real cs = std::sqrt(std::abs(cs2));
    cs = pmb->peos->GetEOS().GetSoundSpeed(n,T,Y);
    cs2 = SQR(cs);
  }

  // Set Field values
  avg_field[fi::gxx] = gxx;
  avg_field[fi::gxy] = gxy;
  avg_field[fi::gxz] = gxz;
  avg_field[fi::gyy] = gyy;
  avg_field[fi::gyz] = gyz;
  avg_field[fi::gzz] = gzz;
  avg_field[fi::gxx_u] = gxx_u;
  avg_field[fi::gyy_u] = gyy_u;
  avg_field[fi::gzz_u] = gzz_u;
  avg_field[fi::det_g] = det_g;
  avg_field[fi::bx]  = betax;
  avg_field[fi::a]   = alpha;

  // Set Hydro Quantities
  avg_hq[hi::rho]    = rho;
  avg_hq[hi::epsl]   = epsl;
  avg_hq[hi::p]      = p;
  avg_hq[hi::h]      = h;
  avg_hq[hi::vx]     = vx;
  avg_hq[hi::vy]     = vy;
  avg_hq[hi::vz]     = vz;
  avg_hq[hi::vx_d]   = vx_d;
  avg_hq[hi::vy_d]   = vy_d;
  avg_hq[hi::vz_d]   = vz_d;
  avg_hq[hi::v2]     = v2;
  avg_hq[hi::W]      = W;
  avg_hq[hi::T]      = T;

  // sound speed and Eigen values at interaface
  for (int m=0; m<NSCALARS; m++)
  {
    Y[m] = pmb->pscalars->r(m,k,j,i);
  }
  pmb->peos->SoundSpeedsGR(n, T,avg_hq[hi::vx +ivx-1],
                           v2,alpha,betax,avg_field[fi::gxx_u +ivx -1], &Lp,&Lm,Y);
  //pmb->peos->SoundSpeedsSR(n,T,vx,SQR(W),&Lp,&Lm,Y);
  L0 = alpha*avg_hq[hi::vx +ivx-1] - betax;
  //L0=vx;
  avg_eig[eig::chi]   = chi;
  avg_eig[eig::kappa] = kappa;
  avg_eig[eig::cs2]   = cs2;
  avg_eig[eig::L0]    = L0;
  avg_eig[eig::Lm]    = Lm;
  avg_eig[eig::Lp]    = Lp;

  return;
}

}


#endif // CHARACTERISTIC_FIELDS_HPP_
