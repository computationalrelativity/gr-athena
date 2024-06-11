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
double rec1d_m_mp3(double eps, double uimt, double uimo, double ui, double uipo, double uipt);
double rec1d_p_mp3(double eps, double uimt, double uimo, double ui, double uipo, double uipt);
double rec1d_m_mp5(double eps, double uimt, double uimo, double ui, double uipo, double uipt);
double rec1d_p_mp5(double eps, double uimt, double uimo, double ui, double uipo, double uipt);
Real rec1d_m_mp7(Real eps, Real uim3, Real uimt, Real uimo, Real ui, Real uipo, Real uipt, Real uip3);
Real rec1d_p_mp7(Real eps, Real uim3, Real uimt, Real uimo, Real ui, Real uipo, Real uipt, Real uip3);
double rec1d_m_ceno3(double uimt, double uimo, double ui, double uipo, double uipt);
double rec1d_p_ceno3(double uimt, double uimo, double ui, double uipo, double uipt);

double rec1d_p_weno5d_si(double uimt, double uimo, double ui, double uipo, double uipt);
// const double W5D_SI_EPSL = 1e-40;
const double W5D_SI_EPSL = 1e-12;
const double W5D_SI_p    = 2;
const double W5D_SI_s    = 1;
const double W5D_SI_mu_0 = 1e-40;

Real mpnlimiter(Real eps_mpn,
                Real u,
                Real uimt,Real uimo,Real ui,Real uipo,Real uipt);

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

// point-wise
static double optimw_pw[3]  = {1./16., 5./8., 5./16.};

// for mpn schemes
static double cl3[3] = {-1./6., 5./6., 2./6.};

static double cl5[5] = {2./60., -13./60., 47./60., 27./60., -3./60.};
// point-wise
// static double cl5[5] = {3./128.,
//                         -20./128.,
//                         90./128.,
//                         60./128.,
//                        -5./128.};

static double cl7[7] = {-3./420.,
                         25./420.,
                        -101./420.,
                         319./420.,
                         214./420.,
                        -38./420.,
                         4./420.};


void Reconstruction::WenoX1(
    const int k, const int j, const int il, const int iu,
    const AthenaArray<Real> &q, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &ql, AthenaArray<Real> &qr,
    const bool enforce_floors) {
  Coordinates *pco = pmy_block_->pcoord;
  // set work arrays to shallow copies of scratch arrays
  AthenaArray<Real> &qc = scr1_ni_, &dql = scr2_ni_, &dqr = scr3_ni_,
                   &dqm = scr4_ni_;
//  const int nu = q.GetDim4() - 1;

  /*
  int nu;
  if(eps_rec){
     nu = NHYDRO-1;
  }
  else{
     nu = NHYDRO;
  }
  */

  const int nu = q.GetDim4();

  // compute L/R slopes for each variable
  for (int n=0; n< nu; ++n)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      /*
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

//      ql(n,i) = rec1d_p_weno5(luimt,luimo,lui,luipo,luipt);
//      qr(n,i) = rec1d_m_weno5(ruimt,ruimo,rui,ruipo,ruipt);
//    possible other reconstruction routines for testing - to be separated into separate callable reconstruction routines
      ql(n,i) = rec1d_p_wenoz(luimt,luimo,lui,luipo,luipt);
      qr(n,i) = rec1d_m_wenoz(ruimt,ruimo,rui,ruipo,ruipt);
//      ql(n,i) = rec1d_p_mp5(luimt,luimo,lui,luipo,luipt);
//      qr(n,i) = rec1d_m_mp5(ruimt,ruimo,rui,ruipo,ruipt);
//      ql(n,i) = rec1d_p_ceno3(luimt,luimo,lui,luipo,luipt);
//      qr(n,i) = rec1d_m_ceno3(ruimt,ruimo,rui,ruipo,ruipt);
      */

      const Real uimt = q(n,k,j,i-2);
      const Real uimo = q(n,k,j,i-1);
      const Real ui   = q(n,k,j,i);
      const Real uipo = q(n,k,j,i+1);
      const Real uipt = q(n,k,j,i+2);

      // N.B offset for x-dir rec.
      switch (xorder_style)  // should check if this gets loop-lifted...
      {
        case ReconstructionVariant::ceno3:
        {
          ql(n,i+1) = rec1d_p_ceno3(uimt,uimo,ui,uipo,uipt);
          qr(n,i  ) = rec1d_p_ceno3(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::mp3:
        {
          ql(n,i+1) = rec1d_p_mp3(xorder_eps,uimt,uimo,ui,uipo,uipt);
          qr(n,i  ) = rec1d_p_mp3(xorder_eps,uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::mp5:
        {
          ql(n,i+1) = rec1d_p_mp5(xorder_eps,uimt,uimo,ui,uipo,uipt);
          qr(n,i  ) = rec1d_p_mp5(xorder_eps,uipt,uipo,ui,uimo,uimt);

          /*
          const bool rpf = (
            pmy_block_->peos->RequirePrimitiveFloor(ql,k,j,i+1) ||
            pmy_block_->peos->RequirePrimitiveFloor(qr,k,j,i));

          if (rpf)
          {
            ql(n,i+1) = rec1d_p_mp3(xorder_eps,uimt,uimo,ui,uipo,uipt);
            qr(n,i  ) = rec1d_p_mp3(xorder_eps,uipt,uipo,ui,uimo,uimt);
          }
          */

          break;
        }
        case ReconstructionVariant::mp7:
        {
          const Real uim3 = q(n,k,j,i-3);
          const Real uip3 = q(n,k,j,i+3);
          ql(n,i+1) = rec1d_p_mp7(xorder_eps,uim3,uimt,uimo,ui,uipo,uipt,uip3);
          qr(n,i  ) = rec1d_p_mp7(xorder_eps,uip3,uipt,uipo,ui,uimo,uimt,uim3);

          /*
          const bool rpf_7 = (
            pmy_block_->peos->RequirePrimitiveFloor(ql,k,j,i+1) ||
            pmy_block_->peos->RequirePrimitiveFloor(qr,k,j,i));

          if (rpf_7)
          {
            ql(n,i+1) = rec1d_p_mp5(xorder_eps,uimt,uimo,ui,uipo,uipt);
            qr(n,i  ) = rec1d_p_mp5(xorder_eps,uipt,uipo,ui,uimo,uimt);
          }

          const bool rpf_5 = (
            pmy_block_->peos->RequirePrimitiveFloor(ql,k,j,i+1) ||
            pmy_block_->peos->RequirePrimitiveFloor(qr,k,j,i));

          if (rpf_5)
          {
            ql(n,i+1) = rec1d_p_mp3(xorder_eps,uimt,uimo,ui,uipo,uipt);
            qr(n,i  ) = rec1d_p_mp3(xorder_eps,uipt,uipo,ui,uimo,uimt);
          }
          */

          break;
        }
        case ReconstructionVariant::weno5:
        {
          ql(n,i+1) = rec1d_p_weno5(uimt,uimo,ui,uipo,uipt);
          qr(n,i  ) = rec1d_p_weno5(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5z:
        {
          ql(n,i+1) = rec1d_p_wenoz(uimt,uimo,ui,uipo,uipt);
          qr(n,i)   = rec1d_p_wenoz(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5z_r:
        {
          ql(n,i+1) = rec1d_p_wenoz(uimt,uimo,ui,uipo,uipt);
          qr(n,i  ) = rec1d_p_wenoz(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5d_si:
        {
          ql(n,i+1) = rec1d_p_weno5d_si(uimt,uimo,ui,uipo,uipt);
          qr(n,i  ) = rec1d_p_weno5d_si(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::none:
        {
          break;
        }
      }

    }
  }

  if(eps_rec)
  {
    // BD: This should not live here.
    // If you want to reconstruct mapped variables map first then pass to
    // The reconstruction procedure.
    std::cout << "TO FIX: see weno.cpp" << std::endl;
    std::exit(0);

    /*
    #pragma omp simd
        for (int i=il; i<=iu; ++i) {
          Real luimt = q(IPR,k,j,i-3)/q(IDN,k,j,i-3);
          Real luimo = q(IPR,k,j,i-2)/q(IDN,k,j,i-2);
          Real lui = q(IPR,k,j,i-1)/q(IDN,k,j,i-1);
          Real luipo = q(IPR,k,j,i)/q(IDN,k,j,i);
          Real luipt = q(IPR,k,j,i+1)/q(IDN,k,j,i+1);
          Real ruimt = q(IPR,k,j,i-2)/q(IDN,k,j,i-2);
          Real ruimo = q(IPR,k,j,i-1)/q(IDN,k,j,i-1);
          Real rui = q(IPR,k,j,i)/q(IDN,k,j,i);
          Real ruipo = q(IPR,k,j,i+1)/q(IDN,k,j,i+1);
          Real ruipt = q(IPR,k,j,i+2)/q(IDN,k,j,i+2);
    //      ql(n,i) = rec1d_p_weno5(luimt,luimo,lui,luipo,luipt);
    //      qr(n,i) = rec1d_m_weno5(ruimt,ruimo,rui,ruipo,ruipt);
    //    possible other reconstruction routines for testing - to be separated into separate callable reconstruction routines
          ql(IPR,i) = rec1d_p_wenoz(luimt,luimo,lui,luipo,luipt)*ql(IDN,i);
          qr(IPR,i) = rec1d_m_wenoz(ruimt,ruimo,rui,ruipo,ruipt)*qr(IDN,i);
    //      ql(n,i) = rec1d_p_mp5(luimt,luimo,lui,luipo,luipt);
    //      qr(n,i) = rec1d_m_mp5(ruimt,ruimo,rui,ruipo,ruipt);
    //      ql(n,i) = rec1d_p_ceno3(luimt,luimo,lui,luipo,luipt);
    //      qr(n,i) = rec1d_m_ceno3(ruimt,ruimo,rui,ruipo,ruipt);
        }
    */
  }

  if(MAGNETIC_FIELDS_ENABLED)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      // Real luimt = bcc(IB2,k,j,i-3);
      // Real luimo = bcc(IB2,k,j,i-2);
      // Real lui = bcc(IB2,k,j,i-1);
      // Real luipo = bcc(IB2,k,j,i);
      // Real luipt = bcc(IB2,k,j,i+1);
      // Real ruimt = bcc(IB2,k,j,i-2);
      // Real ruimo = bcc(IB2,k,j,i-1);
      // Real rui = bcc(IB2,k,j,i);
      // Real ruipo = bcc(IB2,k,j,i+1);
      // Real ruipt = bcc(IB2,k,j,i+2);

      // ql(IBY,i) = rec1d_p_wenoz(luimt,luimo,lui,luipo,luipt);
      // qr(IBY,i) = rec1d_m_wenoz(ruimt,ruimo,rui,ruipo,ruipt);

      const Real uimt = bcc(IB2,k,j,i-2);
      const Real uimo = bcc(IB2,k,j,i-1);
      const Real ui   = bcc(IB2,k,j,i);
      const Real uipo = bcc(IB2,k,j,i+1);
      const Real uipt = bcc(IB2,k,j,i+2);

      // N.B offset for x-dir rec.
      switch (xorder_style)  // should check if this gets loop-lifted...
      {
        case ReconstructionVariant::ceno3:
        {
          ql(IBY,i+1) = rec1d_p_ceno3(uimt,uimo,ui,uipo,uipt);
          qr(IBY,i  ) = rec1d_p_ceno3(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::mp3:
        {
          ql(IBY,i+1) = rec1d_p_mp3(xorder_eps,uimt,uimo,ui,uipo,uipt);
          qr(IBY,i  ) = rec1d_p_mp3(xorder_eps,uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::mp5:
        {
          ql(IBY,i+1) = rec1d_p_mp5(xorder_eps,uimt,uimo,ui,uipo,uipt);
          qr(IBY,i  ) = rec1d_p_mp5(xorder_eps,uipt,uipo,ui,uimo,uimt);

          break;
        }
        case ReconstructionVariant::mp7:
        {
          const Real uim3 = q(IBY,k,j,i-3);
          const Real uip3 = q(IBY,k,j,i+3);
          ql(IBY,i+1) = rec1d_p_mp7(xorder_eps,uim3,uimt,uimo,ui,uipo,uipt,uip3);
          qr(IBY,i  ) = rec1d_p_mp7(xorder_eps,uip3,uipt,uipo,ui,uimo,uimt,uim3);

          break;
        }
        case ReconstructionVariant::weno5:
        {
          ql(IBY,i+1) = rec1d_p_weno5(uimt,uimo,ui,uipo,uipt);
          qr(IBY,i  ) = rec1d_p_weno5(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5z:
        {
          ql(IBY,i+1) = rec1d_p_wenoz(uimt,uimo,ui,uipo,uipt);
          qr(IBY,i)   = rec1d_p_wenoz(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5z_r:
        {
          ql(IBY,i+1) = rec1d_p_wenoz(uimt,uimo,ui,uipo,uipt);
          qr(IBY,i  ) = rec1d_p_wenoz(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5d_si:
        {
          ql(IBY,i+1) = rec1d_p_weno5d_si(uimt,uimo,ui,uipo,uipt);
          qr(IBY,i  ) = rec1d_p_weno5d_si(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::none:
        {
          break;
        }
      }

    }

    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      // Real luimt = bcc(IB3,k,j,i-3);
      // Real luimo = bcc(IB3,k,j,i-2);
      // Real lui = bcc(IB3,k,j,i-1);
      // Real luipo = bcc(IB3,k,j,i);
      // Real luipt = bcc(IB3,k,j,i+1);
      // Real ruimt = bcc(IB3,k,j,i-2);
      // Real ruimo = bcc(IB3,k,j,i-1);
      // Real rui = bcc(IB3,k,j,i);
      // Real ruipo = bcc(IB3,k,j,i+1);
      // Real ruipt = bcc(IB3,k,j,i+2);

      // ql(IBZ,i) = rec1d_p_wenoz(luimt,luimo,lui,luipo,luipt);
      // qr(IBZ,i) = rec1d_m_wenoz(ruimt,ruimo,rui,ruipo,ruipt);

      const Real uimt = bcc(IB3,k,j,i-2);
      const Real uimo = bcc(IB3,k,j,i-1);
      const Real ui   = bcc(IB3,k,j,i);
      const Real uipo = bcc(IB3,k,j,i+1);
      const Real uipt = bcc(IB3,k,j,i+2);

      // N.B offset for x-dir rec.
      switch (xorder_style)  // should check if this gets loop-lifted...
      {
        case ReconstructionVariant::ceno3:
        {
          ql(IBZ,i+1) = rec1d_p_ceno3(uimt,uimo,ui,uipo,uipt);
          qr(IBZ,i  ) = rec1d_p_ceno3(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::mp3:
        {
          ql(IBZ,i+1) = rec1d_p_mp3(xorder_eps,uimt,uimo,ui,uipo,uipt);
          qr(IBZ,i  ) = rec1d_p_mp3(xorder_eps,uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::mp5:
        {
          ql(IBZ,i+1) = rec1d_p_mp5(xorder_eps,uimt,uimo,ui,uipo,uipt);
          qr(IBZ,i  ) = rec1d_p_mp5(xorder_eps,uipt,uipo,ui,uimo,uimt);

          break;
        }
        case ReconstructionVariant::mp7:
        {
          const Real uim3 = q(IBZ,k,j,i-3);
          const Real uip3 = q(IBZ,k,j,i+3);
          ql(IBZ,i+1) = rec1d_p_mp7(xorder_eps,uim3,uimt,uimo,ui,uipo,uipt,uip3);
          qr(IBZ,i  ) = rec1d_p_mp7(xorder_eps,uip3,uipt,uipo,ui,uimo,uimt,uim3);

          break;
        }
        case ReconstructionVariant::weno5:
        {
          ql(IBZ,i+1) = rec1d_p_weno5(uimt,uimo,ui,uipo,uipt);
          qr(IBZ,i  ) = rec1d_p_weno5(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5z:
        {
          ql(IBZ,i+1) = rec1d_p_wenoz(uimt,uimo,ui,uipo,uipt);
          qr(IBZ,i)   = rec1d_p_wenoz(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5z_r:
        {
          ql(IBZ,i+1) = rec1d_p_wenoz(uimt,uimo,ui,uipo,uipt);
          qr(IBZ,i  ) = rec1d_p_wenoz(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5d_si:
        {
          ql(IBZ,i+1) = rec1d_p_weno5d_si(uimt,uimo,ui,uipo,uipt);
          qr(IBZ,i  ) = rec1d_p_weno5d_si(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::none:
        {
          break;
        }
      }

    }

  }

  if (xorder_fallback)
  for (int i=il; i<=iu; ++i)
  {
    bool rpf = (
      pmy_block_->peos->RequirePrimitiveFloor(ql,k,j,i+1) ||
      pmy_block_->peos->RequirePrimitiveFloor(qr,k,j,i));

    if (rpf)
    {
      // revert
      ReconstructionVariant xorder_style_ = xorder_style;
      xorder_style = ReconstructionVariant::ceno3;
      xorder_fallback = false;
      WenoX1(k, j, i, i, q, bcc, ql, qr, false);

      xorder_style = xorder_style_;
      xorder_fallback = true;


      rpf = (
        pmy_block_->peos->RequirePrimitiveFloor(ql,k,j,i+1) ||
        pmy_block_->peos->RequirePrimitiveFloor(qr,k,j,i));

      if (rpf)
      {
        PiecewiseLinearX1(k, j, i, i, q, bcc, ql, qr);
      }
    }
  }


  if (enforce_floors)
  {
#if USETM
for (int l=0; l<NSCALARS; l++){
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
      scalar_l(l,i+1) = pmy_block_->pscalars->r(l,k,j,i);
      scalar_r(l,i) = pmy_block_->pscalars->r(l,k,j,i);
    }
  }
#endif

    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
#if USETM
      pmy_block_->peos->ApplyPrimitiveFloors(ql, scalar_l, k ,j, i+1);
      pmy_block_->peos->ApplyPrimitiveFloors(qr, scalar_r, k ,j, i);
#else
      pmy_block_->peos->ApplyPrimitiveFloors(ql,k,j,i+1);
      pmy_block_->peos->ApplyPrimitiveFloors(qr,k,j,i);
#endif
    }
  }

  return;
}

void Reconstruction::WenoX2(
    const int k, const int j, const int il, const int iu,
    const AthenaArray<Real> &q,  const AthenaArray<Real> &bcc,
    AthenaArray<Real> &ql, AthenaArray<Real> &qr,
    const bool enforce_floors)
{
  Coordinates *pco = pmy_block_->pcoord;
  // set work arrays to shallow copies of scratch arrays
  AthenaArray<Real> &qc = scr1_ni_, &dql = scr2_ni_, &dqr = scr3_ni_,
                   &dqm = scr4_ni_;
//  const int nu = q.GetDim4() - 1;

  // compute L/R slopes for each variable
//  for (int n=0; n<NHYDRO-1; ++n) {

  const int nu = q.GetDim4();

  /*
  int nu;
  if(eps_rec){
     nu = NHYDRO-1;
  }
  else{
     nu = NHYDRO;
  }
  */

  for (int n=0; n<nu; ++n)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      /*
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
      //      ql(n,i) = rec1d_p_weno5(ruimt,ruimo,rui,ruipo,ruipt);
      //      qr(n,i) = rec1d_m_weno5(ruimt,ruimo,rui,ruipo,ruipt);
            ql(n,i) = rec1d_p_wenoz(ruimt,ruimo,rui,ruipo,ruipt);
            qr(n,i) = rec1d_m_wenoz(ruimt,ruimo,rui,ruipo,ruipt);
      //      ql(n,i) = rec1d_p_mp5(ruimt,ruimo,rui,ruipo,ruipt);
      //      qr(n,i) = rec1d_m_mp5(ruimt,ruimo,rui,ruipo,ruipt);
      //      ql(n,i) = rec1d_p_ceno3(ruimt,ruimo,rui,ruipo,ruipt);
      //      qr(n,i) = rec1d_m_ceno3(ruimt,ruimo,rui,ruipo,ruipt);
      */

      const Real uimt = q(n,k,j-2,i);
      const Real uimo = q(n,k,j-1,i);
      const Real ui   = q(n,k,j,  i);
      const Real uipo = q(n,k,j+1,i);
      const Real uipt = q(n,k,j+2,i);

      switch (xorder_style)  // should check if this gets loop-lifted...
      {
        case ReconstructionVariant::ceno3:
        {
          ql(n,i) = rec1d_p_ceno3(uimt,uimo,ui,uipo,uipt);
          qr(n,i) = rec1d_p_ceno3(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::mp3:
        {
          ql(n,i) = rec1d_p_mp3(xorder_eps,uimt,uimo,ui,uipo,uipt);
          qr(n,i) = rec1d_p_mp3(xorder_eps,uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::mp5:
        {

          ql(n,i) = rec1d_p_mp5(xorder_eps,uimt,uimo,ui,uipo,uipt);
          qr(n,i) = rec1d_p_mp5(xorder_eps,uipt,uipo,ui,uimo,uimt);

          /*
          const bool rpf = (pmy_block_->peos->RequirePrimitiveFloor(ql,k,j,i) ||
                            pmy_block_->peos->RequirePrimitiveFloor(qr,k,j,i));

          if (rpf)
          {
            ql(n,i) = rec1d_p_mp3(xorder_eps,uimt,uimo,ui,uipo,uipt);
            qr(n,i) = rec1d_p_mp3(xorder_eps,uipt,uipo,ui,uimo,uimt);
          }
          */

          break;
        }
        case ReconstructionVariant::mp7:
        {
          const Real uim3 = q(n,k,j-3,i);
          const Real uip3 = q(n,k,j+3,i);
          ql(n,i) = rec1d_p_mp7(xorder_eps,uim3,uimt,uimo,ui,uipo,uipt,uip3);
          qr(n,i) = rec1d_p_mp7(xorder_eps,uip3,uipt,uipo,ui,uimo,uimt,uim3);

          /*
          const bool rpf_7 = (
            pmy_block_->peos->RequirePrimitiveFloor(ql,k,j,i) ||
            pmy_block_->peos->RequirePrimitiveFloor(qr,k,j,i));

          if (rpf_7)
          {
            ql(n,i) = rec1d_p_mp5(xorder_eps,uimt,uimo,ui,uipo,uipt);
            qr(n,i) = rec1d_p_mp5(xorder_eps,uipt,uipo,ui,uimo,uimt);
          }

          const bool rpf_5 = (
            pmy_block_->peos->RequirePrimitiveFloor(ql,k,j,i) ||
            pmy_block_->peos->RequirePrimitiveFloor(qr,k,j,i));

          if (rpf_5)
          {
            ql(n,i) = rec1d_p_mp3(xorder_eps,uimt,uimo,ui,uipo,uipt);
            qr(n,i) = rec1d_p_mp3(xorder_eps,uipt,uipo,ui,uimo,uimt);
          }
          */

          break;
        }
        case ReconstructionVariant::weno5:
        {
          ql(n,i) = rec1d_p_weno5(uimt,uimo,ui,uipo,uipt);
          qr(n,i) = rec1d_p_weno5(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5z:
        {
          ql(n,i) = rec1d_p_wenoz(uimt,uimo,ui,uipo,uipt);
          qr(n,i) = rec1d_p_wenoz(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5z_r:
        {
          ql(n,i) = rec1d_p_wenoz(uimt,uimo,ui,uipo,uipt);
          qr(n,i) = rec1d_p_wenoz(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5d_si:
        {
          ql(n,i) = rec1d_p_weno5d_si(uimt,uimo,ui,uipo,uipt);
          qr(n,i) = rec1d_p_weno5d_si(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::none:
        {
          break;
        }
      }

    }
  }

  if(eps_rec)
  {
    // BD: This should not live here.
    // If you want to reconstruct mapped variables map first then pass to
    // The reconstruction procedure.
    std::cout << "TO FIX: see weno.cpp" << std::endl;
    std::exit(0);

    /*
        #pragma omp simd
        for (int i=il; i<=iu; ++i) {
          Real ruimt = q(IPR,k,j-2,i)/q(IDN,k,j-2,i);
          Real ruimo = q(IPR,k,j-1,i)/q(IDN,k,j-1,i);
          Real rui = q(IPR,k,j,i)/q(IDN,k,j,i);
          Real ruipo = q(IPR,k,j+1,i)/q(IDN,k,j+1,i);
          Real ruipt = q(IPR,k,j+2,i)/q(IDN,k,j+2,i);
    //      ql(n,i) = rec1d_p_weno5(ruimt,ruimo,rui,ruipo,ruipt);
    //      qr(n,i) = rec1d_m_weno5(ruimt,ruimo,rui,ruipo,ruipt);
          ql(IPR,i) = rec1d_p_wenoz(ruimt,ruimo,rui,ruipo,ruipt)*ql(IDN,i);
          qr(IPR,i) = rec1d_m_wenoz(ruimt,ruimo,rui,ruipo,ruipt)*qr(IDN,i);
    //      ql(n,i) = rec1d_p_mp5(ruimt,ruimo,rui,ruipo,ruipt);
    //      qr(n,i) = rec1d_m_mp5(ruimt,ruimo,rui,ruipo,ruipt);
    //      ql(n,i) = rec1d_p_ceno3(ruimt,ruimo,rui,ruipo,ruipt);
    //      qr(n,i) = rec1d_m_ceno3(ruimt,ruimo,rui,ruipo,ruipt);
        }
    */

  }


  if(MAGNETIC_FIELDS_ENABLED)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      // Real luimt = bcc(IB3,k,j-3,i);
      // Real luimo = bcc(IB3,k,j-2,i);
      // Real lui = bcc(IB3,k,j-1,i);
      // Real luipo = bcc(IB3,k,j,i);
      // Real luipt = bcc(IB3,k,j+1,i);
      // Real ruimt = bcc(IB3,k,j-2,i);
      // Real ruimo = bcc(IB3,k,j-1,i);
      // Real rui = bcc(IB3,k,j,i);
      // Real ruipo = bcc(IB3,k,j+1,i);
      // Real ruipt = bcc(IB3,k,j+2,i);

      // ql(IBY,i) = rec1d_p_wenoz(ruimt,ruimo,rui,ruipo,ruipt);
      // qr(IBY,i) = rec1d_m_wenoz(ruimt,ruimo,rui,ruipo,ruipt);

      const Real uimt = bcc(IB3,k,j-2,i);
      const Real uimo = bcc(IB3,k,j-1,i);
      const Real ui   = bcc(IB3,k,j,i);
      const Real uipo = bcc(IB3,k,j+1,i);
      const Real uipt = bcc(IB3,k,j+2,i);

      switch (xorder_style)  // should check if this gets loop-lifted...
      {
        case ReconstructionVariant::ceno3:
        {
          ql(IBY,i) = rec1d_p_ceno3(uimt,uimo,ui,uipo,uipt);
          qr(IBY,i) = rec1d_p_ceno3(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::mp3:
        {
          ql(IBY,i) = rec1d_p_mp3(xorder_eps,uimt,uimo,ui,uipo,uipt);
          qr(IBY,i) = rec1d_p_mp3(xorder_eps,uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::mp5:
        {

          ql(IBY,i) = rec1d_p_mp5(xorder_eps,uimt,uimo,ui,uipo,uipt);
          qr(IBY,i) = rec1d_p_mp5(xorder_eps,uipt,uipo,ui,uimo,uimt);

          break;
        }
        case ReconstructionVariant::mp7:
        {
          const Real uim3 = q(IBY,k,j-3,i);
          const Real uip3 = q(IBY,k,j+3,i);
          ql(IBY,i) = rec1d_p_mp7(xorder_eps,uim3,uimt,uimo,ui,uipo,uipt,uip3);
          qr(IBY,i) = rec1d_p_mp7(xorder_eps,uip3,uipt,uipo,ui,uimo,uimt,uim3);

          break;
        }
        case ReconstructionVariant::weno5:
        {
          ql(IBY,i) = rec1d_p_weno5(uimt,uimo,ui,uipo,uipt);
          qr(IBY,i) = rec1d_p_weno5(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5z:
        {
          ql(IBY,i) = rec1d_p_wenoz(uimt,uimo,ui,uipo,uipt);
          qr(IBY,i) = rec1d_p_wenoz(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5z_r:
        {
          ql(IBY,i) = rec1d_p_wenoz(uimt,uimo,ui,uipo,uipt);
          qr(IBY,i) = rec1d_p_wenoz(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5d_si:
        {
          ql(IBY,i) = rec1d_p_weno5d_si(uimt,uimo,ui,uipo,uipt);
          qr(IBY,i) = rec1d_p_weno5d_si(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::none:
        {
          break;
        }
      }
    }

    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      // Real luimt = bcc(IB1,k,j-3,i);
      // Real luimo = bcc(IB1,k,j-2,i);
      // Real lui = bcc(IB1,k,j-1,i);
      // Real luipo = bcc(IB1,k,j,i);
      // Real luipt = bcc(IB1,k,j+1,i);
      // Real ruimt = bcc(IB1,k,j-2,i);
      // Real ruimo = bcc(IB1,k,j-1,i);
      // Real rui = bcc(IB1,k,j,i);
      // Real ruipo = bcc(IB1,k,j+1,i);
      // Real ruipt = bcc(IB1,k,j+2,i);

      // ql(IBZ,i) = rec1d_p_wenoz(ruimt,ruimo,rui,ruipo,ruipt);
      // qr(IBZ,i) = rec1d_m_wenoz(ruimt,ruimo,rui,ruipo,ruipt);

      const Real uimt = bcc(IB1,k,j-2,i);
      const Real uimo = bcc(IB1,k,j-1,i);
      const Real ui   = bcc(IB1,k,j,i);
      const Real uipo = bcc(IB1,k,j+1,i);
      const Real uipt = bcc(IB1,k,j+2,i);

      switch (xorder_style)  // should check if this gets loop-lifted...
      {
        case ReconstructionVariant::ceno3:
        {
          ql(IBZ,i) = rec1d_p_ceno3(uimt,uimo,ui,uipo,uipt);
          qr(IBZ,i) = rec1d_p_ceno3(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::mp3:
        {
          ql(IBZ,i) = rec1d_p_mp3(xorder_eps,uimt,uimo,ui,uipo,uipt);
          qr(IBZ,i) = rec1d_p_mp3(xorder_eps,uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::mp5:
        {

          ql(IBZ,i) = rec1d_p_mp5(xorder_eps,uimt,uimo,ui,uipo,uipt);
          qr(IBZ,i) = rec1d_p_mp5(xorder_eps,uipt,uipo,ui,uimo,uimt);

          break;
        }
        case ReconstructionVariant::mp7:
        {
          const Real uim3 = q(IBZ,k,j-3,i);
          const Real uip3 = q(IBZ,k,j+3,i);
          ql(IBZ,i) = rec1d_p_mp7(xorder_eps,uim3,uimt,uimo,ui,uipo,uipt,uip3);
          qr(IBZ,i) = rec1d_p_mp7(xorder_eps,uip3,uipt,uipo,ui,uimo,uimt,uim3);

          break;
        }
        case ReconstructionVariant::weno5:
        {
          ql(IBZ,i) = rec1d_p_weno5(uimt,uimo,ui,uipo,uipt);
          qr(IBZ,i) = rec1d_p_weno5(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5z:
        {
          ql(IBZ,i) = rec1d_p_wenoz(uimt,uimo,ui,uipo,uipt);
          qr(IBZ,i) = rec1d_p_wenoz(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5z_r:
        {
          ql(IBZ,i) = rec1d_p_wenoz(uimt,uimo,ui,uipo,uipt);
          qr(IBZ,i) = rec1d_p_wenoz(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5d_si:
        {
          ql(IBZ,i) = rec1d_p_weno5d_si(uimt,uimo,ui,uipo,uipt);
          qr(IBZ,i) = rec1d_p_weno5d_si(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::none:
        {
          break;
        }
      }

    }
  }

  if (xorder_fallback)
  for (int i=il; i<=iu; ++i)
  {
    bool rpf = (
      pmy_block_->peos->RequirePrimitiveFloor(ql,k,j,i) ||
      pmy_block_->peos->RequirePrimitiveFloor(qr,k,j,i));

    if (rpf)
    {
      // revert
      ReconstructionVariant xorder_style_ = xorder_style;
      xorder_style = ReconstructionVariant::ceno3;
      xorder_fallback = false;
      WenoX2(k, j, i, i, q, bcc, ql, qr, false);

      xorder_style = xorder_style_;
      xorder_fallback = true;

      rpf = (
        pmy_block_->peos->RequirePrimitiveFloor(ql,k,j,i) ||
        pmy_block_->peos->RequirePrimitiveFloor(qr,k,j,i));

      if (rpf)
      {
        PiecewiseLinearX2(k, j, i, i, q, bcc, ql, qr);
      }
    }
  }

  if (enforce_floors)
  {
#if USETM
    for (int l=0; l<NSCALARS; l++)
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      scalar_l(l,i) = pmy_block_->pscalars->r(l,k,j,i);
      scalar_r(l,i) = pmy_block_->pscalars->r(l,k,j,i);
    }
#endif

    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
#if USETM
      pmy_block_->peos->ApplyPrimitiveFloors(ql, scalar_l, k ,j, i);
      pmy_block_->peos->ApplyPrimitiveFloors(qr, scalar_r, k ,j, i);
#else
      pmy_block_->peos->ApplyPrimitiveFloors(ql, k ,j, i);
      pmy_block_->peos->ApplyPrimitiveFloors(qr, k ,j, i);
#endif
    }
  }

  return;
}
//----------------------------------------------------------------------------------------
//  \brief

void Reconstruction::WenoX3(
    const int k, const int j, const int il, const int iu,
    const AthenaArray<Real> &q, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &ql, AthenaArray<Real> &qr,
    const bool enforce_floors)
{
  Coordinates *pco = pmy_block_->pcoord;
  // set work arrays to shallow copies of scratch arrays
  AthenaArray<Real> &qc = scr1_ni_, &dql = scr2_ni_, &dqr = scr3_ni_,
                   &dqm = scr4_ni_;
//  const int nu = q.GetDim4() - 1;

  // compute L/R slopes for each variable
//  for (int n=0; n<NHYDRO-1; ++n) {

  /*
  int nu;
  if(eps_rec){
     nu = NHYDRO-1;
  }
  else{
     nu = NHYDRO;
  }
  */

  const int nu = q.GetDim4();


  for (int n=0; n<nu; ++n)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {

      const Real uimt = q(n,k-2,j,i);
      const Real uimo = q(n,k-1,j,i);
      const Real ui   = q(n,k,  j,i);
      const Real uipo = q(n,k+1,j,i);
      const Real uipt = q(n,k+2,j,i);


      switch (xorder_style)  // should check if this gets loop-lifted...
      {
        case ReconstructionVariant::ceno3:
        {
          ql(n,i) = rec1d_p_ceno3(uimt,uimo,ui,uipo,uipt);
          qr(n,i) = rec1d_p_ceno3(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::mp3:
        {
          ql(n,i) = rec1d_p_mp3(xorder_eps,uimt,uimo,ui,uipo,uipt);
          qr(n,i) = rec1d_p_mp3(xorder_eps,uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::mp5:
        {
          ql(n,i) = rec1d_p_mp5(xorder_eps,uimt,uimo,ui,uipo,uipt);
          qr(n,i) = rec1d_p_mp5(xorder_eps,uipt,uipo,ui,uimo,uimt);

          /*
          const bool rpf = (pmy_block_->peos->RequirePrimitiveFloor(ql,k,j,i) ||
                            pmy_block_->peos->RequirePrimitiveFloor(qr,k,j,i));

          if (rpf)
          {
            ql(n,i) = rec1d_p_mp3(xorder_eps,uimt,uimo,ui,uipo,uipt);
            qr(n,i) = rec1d_p_mp3(xorder_eps,uipt,uipo,ui,uimo,uimt);
          }
          */

          break;
        }
        case ReconstructionVariant::mp7:
        {
          const Real uim3 = q(n,k-3,j,i);
          const Real uip3 = q(n,k+3,j,i);
          ql(n,i) = rec1d_p_mp7(xorder_eps,uim3,uimt,uimo,ui,uipo,uipt,uip3);
          qr(n,i) = rec1d_p_mp7(xorder_eps,uip3,uipt,uipo,ui,uimo,uimt,uim3);

          /*
          const bool rpf_7 = (
            pmy_block_->peos->RequirePrimitiveFloor(ql,k,j,i) ||
            pmy_block_->peos->RequirePrimitiveFloor(qr,k,j,i));

          if (rpf_7)
          {
            ql(n,i) = rec1d_p_mp5(xorder_eps,uimt,uimo,ui,uipo,uipt);
            qr(n,i) = rec1d_p_mp5(xorder_eps,uipt,uipo,ui,uimo,uimt);
          }

          const bool rpf_5 = (
            pmy_block_->peos->RequirePrimitiveFloor(ql,k,j,i) ||
            pmy_block_->peos->RequirePrimitiveFloor(qr,k,j,i));

          if (rpf_5)
          {
            ql(n,i) = rec1d_p_mp3(xorder_eps,uimt,uimo,ui,uipo,uipt);
            qr(n,i) = rec1d_p_mp3(xorder_eps,uipt,uipo,ui,uimo,uimt);
          }
          */

          break;
        }
        case ReconstructionVariant::weno5:
        {
          ql(n,i) = rec1d_p_weno5(uimt,uimo,ui,uipo,uipt);
          qr(n,i) = rec1d_p_weno5(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5z:
        {
          ql(n,i) = rec1d_p_wenoz(uimt,uimo,ui,uipo,uipt);
          qr(n,i) = rec1d_p_wenoz(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5z_r:
        {
          ql(n,i) = rec1d_p_wenoz(uimt,uimo,ui,uipo,uipt);
          qr(n,i) = rec1d_p_wenoz(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5d_si:
        {
          ql(n,i) = rec1d_p_weno5d_si(uimt,uimo,ui,uipo,uipt);
          qr(n,i) = rec1d_p_weno5d_si(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::none:
        {
          break;
        }
      }
      // ql(n,i) = rec1d_p_mp5(uimt,uimo,ui,uipo,uipt);
      // qr(n,i) = rec1d_m_mp5(uimt,uimo,ui,uipo,uipt);

    }
  }

  if(eps_rec)
  {
    // BD: This should not live here.
    // If you want to reconstruct mapped variables map first then pass to
    // The reconstruction procedure.
    std::cout << "TO FIX: see weno.cpp" << std::endl;
    std::exit(0);

    /*
        #pragma omp simd
        for (int i=il; i<=iu; ++i)
        {
          Real ruimt = q(IPR,k-2,j,i)/q(IDN,k-2,j,i);
          Real ruimo = q(IPR,k-1,j,i)/q(IDN,k-1,j,i);
          Real rui = q(IPR,k,j,i)/q(IDN,k,j,i);
          Real ruipo = q(IPR,k+1,j,i)/q(IDN,k+1,j,i);
          Real ruipt = q(IPR,k+2,j,i)/q(IDN,k+2,j,i);
    //      ql(n,i) = rec1d_p_weno5(ruimt,ruimo,rui,ruipo,ruipt);
    //      qr(n,i) = rec1d_m_weno5(ruimt,ruimo,rui,ruipo,ruipt);
          ql(IPR,i) = rec1d_p_wenoz(ruimt,ruimo,rui,ruipo,ruipt)*ql(IDN,i);
          qr(IPR,i) = rec1d_m_wenoz(ruimt,ruimo,rui,ruipo,ruipt)*qr(IDN,i);
    //      ql(n,i) = rec1d_p_mp5(ruimt,ruimo,rui,ruipo,ruipt);
    //      qr(n,i) = rec1d_m_mp5(ruimt,ruimo,rui,ruipo,ruipt);
    //      ql(n,i) = rec1d_p_ceno3(ruimt,ruimo,rui,ruipo,ruipt);
    //      qr(n,i) = rec1d_m_ceno3(ruimt,ruimo,rui,ruipo,ruipt);
        }
    */

  }

  if(MAGNETIC_FIELDS_ENABLED)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      // Real luimt = bcc(IB1,k-3,j,i);
      // Real luimo = bcc(IB1,k-2,j,i);
      // Real lui = bcc(IB1,k-1,j,i);
      // Real luipo = bcc(IB1,k,j,i);
      // Real luipt = bcc(IB1,k+1,j,i);
      // Real ruimt = bcc(IB1,k-2,j,i);
      // Real ruimo = bcc(IB1,k-1,j,i);
      // Real rui = bcc(IB1,k,j,i);
      // Real ruipo = bcc(IB1,k+1,j,i);
      // Real ruipt = bcc(IB1,k+2,j,i);

      // ql(IBY,i) = rec1d_p_wenoz(ruimt,ruimo,rui,ruipo,ruipt);
      // qr(IBY,i) = rec1d_m_wenoz(ruimt,ruimo,rui,ruipo,ruipt);

      const Real uimt = bcc(IB1,k-2,j,i);
      const Real uimo = bcc(IB1,k-1,j,i);
      const Real ui   = bcc(IB1,k,j,i);
      const Real uipo = bcc(IB1,k+1,j,i);
      const Real uipt = bcc(IB1,k+2,j,i);

      switch (xorder_style)  // should check if this gets loop-lifted...
      {
        case ReconstructionVariant::ceno3:
        {
          ql(IBY,i) = rec1d_p_ceno3(uimt,uimo,ui,uipo,uipt);
          qr(IBY,i) = rec1d_p_ceno3(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::mp3:
        {
          ql(IBY,i) = rec1d_p_mp3(xorder_eps,uimt,uimo,ui,uipo,uipt);
          qr(IBY,i) = rec1d_p_mp3(xorder_eps,uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::mp5:
        {
          ql(IBY,i) = rec1d_p_mp5(xorder_eps,uimt,uimo,ui,uipo,uipt);
          qr(IBY,i) = rec1d_p_mp5(xorder_eps,uipt,uipo,ui,uimo,uimt);

          break;
        }
        case ReconstructionVariant::mp7:
        {
          const Real uim3 = q(IBY,k-3,j,i);
          const Real uip3 = q(IBY,k+3,j,i);
          ql(IBY,i) = rec1d_p_mp7(xorder_eps,uim3,uimt,uimo,ui,uipo,uipt,uip3);
          qr(IBY,i) = rec1d_p_mp7(xorder_eps,uip3,uipt,uipo,ui,uimo,uimt,uim3);

          break;
        }
        case ReconstructionVariant::weno5:
        {
          ql(IBY,i) = rec1d_p_weno5(uimt,uimo,ui,uipo,uipt);
          qr(IBY,i) = rec1d_p_weno5(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5z:
        {
          ql(IBY,i) = rec1d_p_wenoz(uimt,uimo,ui,uipo,uipt);
          qr(IBY,i) = rec1d_p_wenoz(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5z_r:
        {
          ql(IBY,i) = rec1d_p_wenoz(uimt,uimo,ui,uipo,uipt);
          qr(IBY,i) = rec1d_p_wenoz(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5d_si:
        {
          ql(IBY,i) = rec1d_p_weno5d_si(uimt,uimo,ui,uipo,uipt);
          qr(IBY,i) = rec1d_p_weno5d_si(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::none:
        {
          break;
        }
      }

    }

    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      // Real luimt = bcc(IB2,k-3,j,i);
      // Real luimo = bcc(IB2,k-2,j,i);
      // Real lui = bcc(IB2,k-1,j,i);
      // Real luipo = bcc(IB2,k,j,i);
      // Real luipt = bcc(IB2,k+1,j,i);
      // Real ruimt = bcc(IB2,k-2,j,i);
      // Real ruimo = bcc(IB2,k-1,j,i);
      // Real rui = bcc(IB2,k,j,i);
      // Real ruipo = bcc(IB2,k+1,j,i);
      // Real ruipt = bcc(IB2,k+2,j,i);

      // ql(IBZ,i) = rec1d_p_wenoz(ruimt,ruimo,rui,ruipo,ruipt);
      // qr(IBZ,i) = rec1d_m_wenoz(ruimt,ruimo,rui,ruipo,ruipt);

      const Real uimt = bcc(IB2,k-2,j,i);
      const Real uimo = bcc(IB2,k-1,j,i);
      const Real ui   = bcc(IB2,k,j,i);
      const Real uipo = bcc(IB2,k+1,j,i);
      const Real uipt = bcc(IB2,k+2,j,i);

      switch (xorder_style)  // should check if this gets loop-lifted...
      {
        case ReconstructionVariant::ceno3:
        {
          ql(IBZ,i) = rec1d_p_ceno3(uimt,uimo,ui,uipo,uipt);
          qr(IBZ,i) = rec1d_p_ceno3(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::mp3:
        {
          ql(IBZ,i) = rec1d_p_mp3(xorder_eps,uimt,uimo,ui,uipo,uipt);
          qr(IBZ,i) = rec1d_p_mp3(xorder_eps,uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::mp5:
        {
          ql(IBZ,i) = rec1d_p_mp5(xorder_eps,uimt,uimo,ui,uipo,uipt);
          qr(IBZ,i) = rec1d_p_mp5(xorder_eps,uipt,uipo,ui,uimo,uimt);

          break;
        }
        case ReconstructionVariant::mp7:
        {
          const Real uim3 = q(IBZ,k-3,j,i);
          const Real uip3 = q(IBZ,k+3,j,i);
          ql(IBZ,i) = rec1d_p_mp7(xorder_eps,uim3,uimt,uimo,ui,uipo,uipt,uip3);
          qr(IBZ,i) = rec1d_p_mp7(xorder_eps,uip3,uipt,uipo,ui,uimo,uimt,uim3);

          break;
        }
        case ReconstructionVariant::weno5:
        {
          ql(IBZ,i) = rec1d_p_weno5(uimt,uimo,ui,uipo,uipt);
          qr(IBZ,i) = rec1d_p_weno5(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5z:
        {
          ql(IBZ,i) = rec1d_p_wenoz(uimt,uimo,ui,uipo,uipt);
          qr(IBZ,i) = rec1d_p_wenoz(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5z_r:
        {
          ql(IBZ,i) = rec1d_p_wenoz(uimt,uimo,ui,uipo,uipt);
          qr(IBZ,i) = rec1d_p_wenoz(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::weno5d_si:
        {
          ql(IBZ,i) = rec1d_p_weno5d_si(uimt,uimo,ui,uipo,uipt);
          qr(IBZ,i) = rec1d_p_weno5d_si(uipt,uipo,ui,uimo,uimt);
          break;
        }
        case ReconstructionVariant::none:
        {
          break;
        }
      }

    }
  }

  if (xorder_fallback)
  for (int i=il; i<=iu; ++i)
  {
    bool rpf = (
      pmy_block_->peos->RequirePrimitiveFloor(ql,k,j,i) ||
      pmy_block_->peos->RequirePrimitiveFloor(qr,k,j,i));

    if (rpf)
    {
      // revert
      ReconstructionVariant xorder_style_ = xorder_style;
      xorder_style = ReconstructionVariant::ceno3;
      xorder_fallback = false;
      WenoX3(k, j, i, i, q, bcc, ql, qr, false);

      xorder_style = xorder_style_;
      xorder_fallback = true;

      rpf = (
        pmy_block_->peos->RequirePrimitiveFloor(ql,k,j,i) ||
        pmy_block_->peos->RequirePrimitiveFloor(qr,k,j,i));

      if (rpf)
      {
        PiecewiseLinearX3(k, j, i, i, q, bcc, ql, qr);
      }
    }
  }


  if (enforce_floors)
  {
#if USETM
  for (int l=0; l<NSCALARS; l++){
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      scalar_l(l,i) = pmy_block_->pscalars->r(l,k,j,i);
      scalar_r(l,i) = pmy_block_->pscalars->r(l,k,j,i);
    }
  }
#endif

    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
#if USETM
      pmy_block_->peos->ApplyPrimitiveFloors(ql, scalar_l, k ,j, i);
      pmy_block_->peos->ApplyPrimitiveFloors(qr, scalar_r, k ,j, i);
#else
      pmy_block_->peos->ApplyPrimitiveFloors(ql, k ,j, i);
      pmy_block_->peos->ApplyPrimitiveFloors(qr, k ,j, i);
#endif
    }
  }

  return;
}



//double rec1d_p_weno5(double *u, int i)
double rec1d_p_weno5(double uimt, double uimo, double ui, double uipo, double uipt)
{
/*
  // Computes u[i + 1/2]
  double uimt = u [i-2];
  double uimo = u [i-1];
  double ui   = u [i];
  double uipo = u [i+1];
  double uipt = u [i+2];
*/
  double up;

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
/*
  // Computes u[i - 1/2]
  double uimt = u [i-2];
  double uimo = u [i-1];
  double ui   = u [i];
  double uipo = u [i+1];
  double uipt = u [i+2];
*/
  double um;

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

#pragma omp declare simd
double rec1d_p_wenoz(double uimt, double uimo, double ui, double uipo, double uipt)
{
/*
  // Computes u[i + 1/2]
  double uimt = u [i-2];
  double uimo = u [i-1];
  double ui   = u [i];
  double uipo = u [i+1];
  double uipt = u [i+2];
*/
  double up;

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

  /*
  uk[0] = oocc[8]*(   cc[3]*uimt - cc[10]*uimo + cc[15]*ui   );
  uk[1] = oocc[8]*( - cc[1]*uimo + cc[6]*ui    + cc[3] *uipo );
  uk[2] = oocc[8]*(   cc[3]*ui   + cc[6]*uipo  -        uipt );
  up = w[0]*uk[0] + w[1]*uk[1] + w[2]*uk[2];
  */

  return up;
}
// WENOZ 
#pragma omp declare simd
double rec1d_m_wenoz(double uimt, double uimo, double ui, double uipo, double uipt)
{
/*
  // Computes u[i - 1/2]
  double uimt = u [i-2];
  double uimo = u [i-1];
  double ui   = u [i];
  double uipo = u [i+1];
  double uipt = u [i+2];
*/
  double um;

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

#pragma omp declare simd
double rec1d_p_weno5d_si(double uimt, double uimo, double ui, double uipo, double uipt)
{
/*
  // Computes u[i + 1/2]
  double uimt = u [i-2];
  double uimo = u [i-1];
  double ui   = u [i];
  double uipo = u [i+1];
  double uipt = u [i+2];
*/
  double up;

  double uk[3], a[3], b[3], w[3], dsa;
  int j;

#if (USE_WEIGHTS_OPTIMAL)
  for( j = 0 ; j<3; ++j) w[j] = optimw[j];
#else
  // smoothness coefs, Jiag & Shu '96
  b[0] = othreeotwo * SQR(( uimt-cc[2]*uimo+ui   )) + oocc[4] * SQR((uimt-cc[4]*uimo+cc[3]*ui));
  b[1] = othreeotwo * SQR(( uimo-cc[2]*ui  +uipo )) + oocc[4] * SQR((uimo-uipo ) );
  b[2] = othreeotwo * SQR(( ui  -cc[2]*uipo+uipt )) + oocc[4] * SQR((cc[3]*ui-cc[4]*uipo+uipt));

  const double phi = std::sqrt(std::abs(b[0]-2.*b[1]+b[2]));
  const double tau = std::abs(b[0]-b[2]);

  // local descaling function
  const int r = 3;
  const double xi = (std::abs(uimt) +
                     std::abs(uimo) +
                     std::abs(ui) +
                     std::abs(uipo) +
                     std::abs(uipt)) / (2. * (r) - 1.);

  const double mu  = xi + W5D_SI_mu_0;
  const double mu2 = SQR(mu);
  const double Phi = std::min(1., phi / mu);

  const double eps_mu2 = W5D_SI_EPSL * mu2;

  for( j = 0; j<3; ++j)
  {
    const double Z_j = std::pow(tau / (b[j] + eps_mu2), W5D_SI_p);
    const double Gam_j = Phi * Z_j;
    a[j] = optimw[j] * std::pow((1. + Gam_j), W5D_SI_s);
  }

  dsa = cc[1]/( a[0] + a[1] + a[2] );

  for( j = 0 ; j<3; ++j)
  {
    w[j] = a[j] * dsa;
  }
#endif

  uk[0] = oocc[6]*(   cc[2]*uimt - cc[7]*uimo + cc[11]*ui   );
  uk[1] = oocc[6]*( - cc[1]*uimo + cc[5]*ui   + cc[2] *uipo );
  uk[2] = oocc[6]*(   cc[2]*ui   + cc[5]*uipo -        uipt );

  /*
  // point-wise
  uk[0] = oocc[8]*(   cc[3]*uimt - cc[10]*uimo + cc[15]*ui   );
  uk[1] = oocc[8]*( - cc[1]*uimo + cc[6]*ui    + cc[3] *uipo );
  uk[2] = oocc[8]*(   cc[3]*ui   + cc[6]*uipo  -        uipt );
  */

  up = w[0]*uk[0] + w[1]*uk[1] + w[2]*uk[2];

  return up;
}

double rec1d_m_mp3(double eps, double uimt, double uimo, double ui, double uipo, double uipt)
{
  // Computes u[i - 1/2]
  // double uimt = u [i-2];
  // double uimo = u [i-1];
  // double ui   = u [i];
  // double uipo = u [i+1];
  // double uipt = u [i+2];
  double ulim  = cl3[0]*uipo + cl3[1]*ui + cl3[2]*uimo;
  return mpnlimiter(eps,ulim,uipt,uipo,ui,uimo,uimt);
}
double rec1d_p_mp3(double eps, double uimt, double uimo, double ui, double uipo, double uipt)
{

  // Computes u[i + 1/2]
  // double uimt = u [i-2];
  // double uimo = u [i-1];
  // double ui   = u [i];
  // double uipo = u [i+1];
  // double uipt = u [i+2];
  double ulim  = cl3[0]*uimo + cl3[1]*ui + cl3[2]*uipo;
  return mpnlimiter(eps,ulim,uimt,uimo,ui,uipo,uipt);
}

double rec1d_m_mp5(double eps, double uimt, double uimo, double ui, double uipo, double uipt)
{
  // Computes u[i - 1/2]
  // double uimt = u [i-2];
  // double uimo = u [i-1];
  // double ui   = u [i];
  // double uipo = u [i+1];
  // double uipt = u [i+2];
  double ulim  = cl5[0]*uipt + cl5[1]*uipo + cl5[2]*ui + cl5[3]*uimo + cl5[4]*uimt;
  return mpnlimiter(eps,ulim,uipt,uipo,ui,uimo,uimt);
}
double rec1d_p_mp5(double eps, double uimt, double uimo, double ui, double uipo, double uipt)
{

  // Computes u[i + 1/2]
  // double uimt = u [i-2];
  // double uimo = u [i-1];
  // double ui   = u [i];
  // double uipo = u [i+1];
  // double uipt = u [i+2];
  double ulim  = cl5[0]*uimt + cl5[1]*uimo + cl5[2]*ui + cl5[3]*uipo + cl5[4]*uipt;
  return mpnlimiter(eps,ulim,uimt,uimo,ui,uipo,uipt);
}

Real rec1d_m_mp7(
  Real eps,
  Real uim3, Real uimt, Real uimo, Real ui, Real uipo, Real uipt, Real uip3)
{
  // Computes u[i - 1/2]
  // double uim3 = u [i-3];
  // double uimt = u [i-2];
  // double uimo = u [i-1];
  // double ui   = u [i];
  // double uipo = u [i+1];
  // double uip2 = u [i+2];
  // double uip3 = u [i+3];
  double ulim  = (cl7[0]*uip3 +
                  cl7[1]*uipt +
                  cl7[2]*uipo +
                  cl7[3]*ui +
                  cl7[4]*uimo +
                  cl7[5]*uimt +
                  cl7[6]*uim3);
  return mpnlimiter(eps,ulim,uipt,uipo,ui,uimo,uimt);
}
Real rec1d_p_mp7(
  Real eps,
  Real uim3, Real uimt, Real uimo, Real ui, Real uipo, Real uipt, Real uip3)
{

  // Computes u[i + 1/2]
  // double uim3 = u [i-3];
  // double uimt = u [i-2];
  // double uimo = u [i-1];
  // double ui   = u [i];
  // double uipo = u [i+1];
  // double uip2 = u [i+2];
  // double uip3 = u [i+3];
  double ulim  = (cl7[0]*uim3 +
                  cl7[1]*uimt +
                  cl7[2]*uimo +
                  cl7[3]*ui +
                  cl7[4]*uipo +
                  cl7[5]*uipt +
                  cl7[6]*uip3);
  return mpnlimiter(eps,ulim,uimt,uimo,ui,uipo,uipt);
}

Real mpnlimiter(Real eps_mpn,
                Real u,
                Real uimt,Real uimo,Real ui,Real uipo,Real uipt)
{
  const Real oo2 = 1./2.;
  const Real oo8 = 1./8.;
  const Real fot = 4./3.;
  const Real alphatil = 4.;

  auto sign = [&](Real a)
  {
    return (a >= 0) ? 1. : -1.;
  };

  auto min_3 = [&](Real a, Real b, Real c)
  {
    return std::min(std::min(a, b), c);
  };

  auto max_3 = [&](Real a, Real b, Real c)
  {
    return std::max(std::max(a, b), c);
  };

  auto min_abs_2 = [&](Real a, Real b)
  {
    return std::min(std::abs(a), std::abs(b));
  };

  auto min_abs_4 = [&](Real a, Real b, Real c, Real d)
  {
    return std::min(
      std::min(std::abs(a), std::abs(b)),
      std::min(std::abs(c), std::abs(d))
    );
  };

  auto minmod2 = [&](Real x, Real y)
  {
    return oo2 * (sign(x) + sign(y)) * min_abs_2(x, y);
  };

  auto minmod4 = [&](Real w, Real x, Real y, Real z)
  {
    return oo8 * (sign(w) + sign(x)) * std::abs(
      (sign(w) + sign(y)) * (sign(w) + sign(z))
    ) * min_abs_4(w, x, y, z);
  };


  if (eps_mpn > 0)
  {
    const Real U_L2 = std::sqrt(SQR(uimt) + SQR(uimo) +
                                SQR(ui) +
                                SQR(uipo) + SQR(uipt));
    const Real u_MP = ui + minmod2(uipo-ui, alphatil * (ui-uimo));

    // check whether we should apply limiter
    if ((u-ui)*(u-u_MP) <= eps_mpn * U_L2)
      return u;
  }

  const Real dm = uimt - 2.*uimo + ui;
  const Real d0 = uimo - 2.*ui   + uipo;
  const Real dp = ui   - 2.*uipo + uipt;

  const Real dm4p = minmod4(4.*d0-dp, 4.*dp-d0, d0, dp);
  const Real dm4m = minmod4(4.*d0-dm, 4.*dm-d0, d0, dm);

  const Real u_ul = ui + alphatil * (ui - uimo);  // c.f. code of Suresh '97 (which has uimo)
  const Real u_av = oo2 * (ui + uipo);
  const Real u_md = u_av - oo2 * dm4p;
  const Real u_lc = ui + oo2 * (ui - uimo) + fot * dm4m;

  const Real u_min = std::max(min_3(ui, uipo, u_md), min_3(ui, u_ul, u_lc));
  const Real u_max = std::min(max_3(ui, uipo, u_md), max_3(ui, u_ul, u_lc));

  return u + minmod2(u_min-u, u_max-u);

}

// implementation from bam, not working?
/*
double mp5lim( double u,
               double uimt,double uimo,double ui,double uipo,double uipt )
{

  double ump,umin,umax, fUL,fAV,fMD,fLC;
  double d2m,d2c,d2p, dMMm,dMMp;
  double tmp1,tmp2, mp5;

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
*/

double rec1d_p_ceno3(double uimt, double uimo, double ui, double uipo, double uipt)
{
/*
  // Computes u[i + 1/2]
  double uipt = u [i+2];
  double uipo = u [i+1];
  double ui   = u [i];
  double uimo = u [i-1];
  double uimt = u [i-2];
*/
  double up;

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
/*
  // Computes u[i - 1/2]
  double uipt = u [i+2];
  double uipo = u [i+1];
  double ui   = u [i];
  double uimo = u [i-1];
  double uimt = u [i-2];
*/
  double um;

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


