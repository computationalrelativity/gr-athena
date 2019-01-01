//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c.cpp
//  \brief implementation of functions in the Z4c class

// Athena++ headers
#include "z4c.hpp"
#include "z4c_macro.hpp"
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"

#define SQ(X) ((X)*(X))

// constructor, initializes data structures and parameters

Z4c::Z4c(MeshBlock *pmb, ParameterInput *pin)
{
  pmy_block = pmb;
  Coordinates * pco = pmb->pcoord;

  // Allocate memory for the solution and its time derivative
  int ncells1 = pmy_block->block_size.nx1 + 2*(NGHOST);
  int ncells2 = 1, ncells3 = 1;
  if(pmy_block->block_size.nx2 > 1) ncells2 = pmy_block->block_size.nx2 + 2*(NGHOST);
  if(pmy_block->block_size.nx3 > 1) ncells3 = pmy_block->block_size.nx3 + 2*(NGHOST);

  storage.u.NewAthenaArray(N_Z4c, ncells3, ncells2, ncells1);
  storage.u1.NewAthenaArray(N_Z4c, ncells3, ncells2, ncells1);
  // If user-requested time integrator is type 3S*, allocate additional memory registers
  std::string integrator = pin->GetOrAddString("time","integrator","vl2");
  if (integrator == "ssprk5_4") storage.u2.NewAthenaArray(N_Z4c, ncells3, ncells2, ncells1);
  storage.rhs.NewAthenaArray(N_Z4c, ncells3, ncells2, ncells1);
  storage.adm.NewAthenaArray(N_ADM, ncells3, ncells2, ncells1);

  dt1_.NewAthenaArray(ncells1);
  dt2_.NewAthenaArray(ncells1);
  dt3_.NewAthenaArray(ncells1);

  // Parameters
  opt.chi_psi_power = pin->GetOrAddReal("z4c", "chi_psi_power", -4.0);
  opt.chi_div_floor = pin->GetOrAddReal("z4c", "chi_div_floor", -1000.0);
  opt.eps_floor = pin->GetOrAddReal("z4c", "eps_floor", 1e-12);
  opt.z4c_kappa_damp1 = pin->GetOrAddReal("z4c", "z4c_kappa_damp1", 0.0);
  opt.z4c_kappa_damp2 = pin->GetOrAddReal("z4c", "z4c_kappa_damp2", 0.0);

  // Set aliases
  SetZ4cAliases(storage.rhs, rhs);
  SetADMAliases(storage.adm, adm);

  // Allocate memory for aux 1D vars
  detg.NewAthenaTensor(ncells1);
  oopsi4.NewAthenaTensor(ncells1);
  A.NewAthenaTensor(ncells1);
  R.NewAthenaTensor(ncells1);
  K.NewAthenaTensor(ncells1);
  KK.NewAthenaTensor(ncells1);
  Mom_u.NewAthenaTensor(ncells1);
  g_uu.NewAthenaTensor(ncells1);
  R_dd.NewAthenaTensor(ncells1);
  Kt_dd.NewAthenaTensor(ncells1);
  K_ud.NewAthenaTensor(ncells1);
  Gamma_ddd.NewAthenaTensor(ncells1);
  Gamma_udd.NewAthenaTensor(ncells1);
  DK_ddd.NewAthenaTensor(ncells1);
  DK_udd.NewAthenaTensor(ncells1);

  dalpha_d.NewAthenaTensor(ncells1);
  dchi_d.NewAthenaTensor(ncells1);
  dKhat_d.NewAthenaTensor(ncells1);
  dTheta_d.NewAthenaTensor(ncells1);
  ddalpha_dd.NewAthenaTensor(ncells1);
  dbeta_du.NewAthenaTensor(ncells1);
  ddchi_dd.NewAthenaTensor(ncells1);
  dGam_du.NewAthenaTensor(ncells1);
  dg_ddd.NewAthenaTensor(ncells1);
  dK_ddd.NewAthenaTensor(ncells1);
  dA_ddd.NewAthenaTensor(ncells1);
  ddbeta_ddu.NewAthenaTensor(ncells1);
  ddg_dddd.NewAthenaTensor(ncells1);

  Lchi.NewAthenaTensor(ncells1);
  LKhat.NewAthenaTensor(ncells1);
  LTheta.NewAthenaTensor(ncells1);
  Lalpha.NewAthenaTensor(ncells1);
  LGam_u.NewAthenaTensor(ncells1);
  Lbeta_u.NewAthenaTensor(ncells1);
  Lg_dd.NewAthenaTensor(ncells1);
  LA_dd.NewAthenaTensor(ncells1);

  // Setup finite differencing kernel
  FD.stride[0] = 1;
  FD.stride[1] = 0;
  FD.stride[2] = 0;
  FD.idx[0] = 1.0/pco->dx1v(0);
  FD.idx[1] = 0.0;
  FD.idx[2] = 0.0;
  if(ncells2 > 1) {
    FD.stride[1] = ncells1;
    FD.idx[1] = 1.0/pco->dx2v(0);
  }
  if(ncells3 > 1) {
    FD.stride[2] = ncells2*ncells1;
    FD.idx[2] = 1.0/pco->dx3v(0);
  }
}

// destructor

Z4c::~Z4c()
{
  storage.u.DeleteAthenaArray();
  storage.u1.DeleteAthenaArray();
  storage.u2.DeleteAthenaArray();
  storage.rhs.DeleteAthenaArray();
  storage.adm.DeleteAthenaArray();

  dt1_.DeleteAthenaArray();
  dt2_.DeleteAthenaArray();
  dt3_.DeleteAthenaArray();

  detg.DeleteAthenaTensor();
  oopsi4.DeleteAthenaTensor();
  A.DeleteAthenaTensor();
  R.DeleteAthenaTensor();
  K.DeleteAthenaTensor();
  KK.DeleteAthenaTensor();
  Mom_u.DeleteAthenaTensor();
  g_uu.DeleteAthenaTensor();
  R_dd.DeleteAthenaTensor();
  Kt_dd.DeleteAthenaTensor();
  K_ud.DeleteAthenaTensor();
  Gamma_ddd.DeleteAthenaTensor();
  Gamma_udd.DeleteAthenaTensor();
  DK_ddd.DeleteAthenaTensor();
  DK_udd.DeleteAthenaTensor();

  dalpha_d.DeleteAthenaTensor();
  dchi_d.DeleteAthenaTensor();
  dKhat_d.DeleteAthenaTensor();
  dTheta_d.DeleteAthenaTensor();
  ddalpha_dd.DeleteAthenaTensor();
  dbeta_du.DeleteAthenaTensor();
  ddchi_dd.DeleteAthenaTensor();
  dGam_du.DeleteAthenaTensor();
  dg_ddd.DeleteAthenaTensor();
  dK_ddd.DeleteAthenaTensor();
  dA_ddd.DeleteAthenaTensor();
  ddbeta_ddu.DeleteAthenaTensor();
  ddg_dddd.DeleteAthenaTensor();

  Lchi.DeleteAthenaTensor();
  LKhat.DeleteAthenaTensor();
  LTheta.DeleteAthenaTensor();
  Lalpha.DeleteAthenaTensor();
  LGam_u.DeleteAthenaTensor();
  Lbeta_u.DeleteAthenaTensor();
  Lg_dd.DeleteAthenaTensor();
  LA_dd.DeleteAthenaTensor();
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::SetZ4cAliases(AthenaArray<Real> & u, Z4c_vars & z4c)
// \brief Set Z4c aliases

void Z4c::SetZ4cAliases(AthenaArray<Real> & u, Z4c::Z4c_vars & z4c)
{
  z4c.chi.InitWithShallowSlice(u, I_Z4c_chi);
  z4c.Khat.InitWithShallowSlice(u, I_Z4c_Khat);
  z4c.Theta.InitWithShallowSlice(u, I_Z4c_Theta);
  z4c.alpha.InitWithShallowSlice(u, I_Z4c_alpha);
  z4c.Gam_u.InitWithShallowSlice(u, I_Z4c_Gamx);
  z4c.beta_u.InitWithShallowSlice(u, I_Z4c_betax);
  z4c.g_dd.InitWithShallowSlice(u, I_Z4c_gxx);
  z4c.A_dd.InitWithShallowSlice(u, I_Z4c_Axx);
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::SetADMAliases(AthenaArray<Real> & u, ADM_vars & z4c)
// \brief Set ADM aliases

void Z4c::SetADMAliases(AthenaArray<Real> & u_adm, Z4c::ADM_vars & adm)
{
  adm.psi4.InitWithShallowSlice(u_adm, I_ADM_psi4);
  adm.H.InitWithShallowSlice(u_adm, I_ADM_Ham);
  adm.Mom_d.InitWithShallowSlice(u_adm, I_ADM_Momx);
  adm.g_dd.InitWithShallowSlice(u_adm, I_ADM_gxx);
  adm.K_dd.InitWithShallowSlice(u_adm, I_ADM_Kxx);
  adm.rho.InitWithShallowSlice(u_adm, I_ADM_rho);
  adm.S_d.InitWithShallowSlice(u_adm, I_ADM_Sx);
  adm.S_dd.InitWithShallowSlice(u_adm, I_ADM_Sxx);
}

//----------------------------------------------------------------------------------------
// \!fn Real Z4c::SpatialDet(Real gxx, ... , Real gzz)
// \brief returns determinant of 3-metric

Real Z4c::SpatialDet(Real const gxx, Real const gxy, Real const gxz,
                     Real const gyy, Real const gyz, Real const gzz)
{
  return - SQ(gxz)*gyy + 2*gxy*gxz*gyz - gxx*SQ(gyz) - SQ(gxy)*gzz + gxx*gyy*gzz;
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::SpatialInv(Real const detginv,
//           Real const gxx, Real const gxy, Real const gxz,
//           Real const gyy, Real const gyz, Real const gzz,
//           Real * uxx, Real * uxy, Real * uxz,
//           Real * uyy, Real * uyz, Real * uzz)
// \brief returns inverse of 3-metric

void Z4c::SpatialInv(Real const detginv,
                     Real const gxx, Real const gxy, Real const gxz,
                     Real const gyy, Real const gyz, Real const gzz,
                     Real * uxx, Real * uxy, Real * uxz,
                     Real * uyy, Real * uyz, Real * uzz)
{
  *uxx = (-SQ(gyz) + gyy*gzz)*detginv;
  *uxy = (gxz*gyz  - gxy*gzz)*detginv;
  *uyy = (-SQ(gxz) + gxx*gzz)*detginv;
  *uxz = (-gxz*gyy + gxy*gyz)*detginv;
  *uyz = (gxy*gxz  - gxx*gyz)*detginv;
  *uzz = (-SQ(gxy) + gxx*gyy)*detginv;
  return;
}

//----------------------------------------------------------------------------------------
// \!fn Real Z4c::Trace(Real detginv, Real gxx, ... , Real gzz, Real Axx, ..., Real Azz)
// \brief returns Trace of extrinsic curvature

Real Z4c::Trace(Real const detginv,
                Real const gxx, Real const gxy, Real const gxz,
                Real const gyy, Real const gyz, Real const gzz,
                Real const Axx, Real const Axy, Real const Axz,
                Real const Ayy, Real const Ayz, Real const Azz)
{
  return (detginv*(
       - 2.*Ayz*gxx*gyz + Axx*gyy*gzz +  gxx*(Azz*gyy + Ayy*gzz)
       + 2.*(gxz*(Ayz*gxy - Axz*gyy + Axy*gyz) + gxy*(Axz*gyz - Axy*gzz))
       - Azz*SQ(gxy) - Ayy*SQ(gxz) - Axx*SQ(gyz)
       ));
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::AlgConstr(AthenaArray<Real> & u)
// \brief algebraic constraints projection
//
// This function operates only on the interior points of the MeshBlock

void Z4c::AlgConstr(AthenaArray<Real> & u)
{
  Z4c_vars z4c;
  SetZ4cAliases(u, z4c);

  LOOP2(k,j) {
    // compute determinant and "conformal conformal factor"
    LOOP1(i) {
      detg(i) = SpatialDet(z4c.g_dd,k,j,i);
      detg(i) = detg(i) > 0. ? detg(i) : 1.;
      Real eps = detg(i) - 1.;
      oopsi4(i) = (eps < opt.eps_floor) ? (1. - opt.eps_floor/3.) : (pow(1./detg(i), 3));
    }
    // enforce unitary determinant for conformal metric
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      LOOP1(i) {
        z4c.g_dd(a,b,k,j,j,i) *= oopsi4(i);
      }
    }

    // compute trace of A
    LOOP1(i) {
      // note: here we are assuming that det g = 1, which we enforced above
      A(i) = Trace(1.0,
          z4c.g_dd(0,0,k,j,i), z4c.g_dd(0,1,k,j,i), z4c.g_dd(0,2,k,j,i),
          z4c.g_dd(1,1,k,j,i), z4c.g_dd(1,2,k,j,i), z4c.g_dd(2,2,k,j,i),
          z4c.A_dd(0,0,k,j,i), z4c.A_dd(0,1,k,j,i), z4c.A_dd(0,2,k,j,i),
          z4c.A_dd(1,1,k,j,i), z4c.A_dd(1,2,k,j,i), z4c.A_dd(2,2,k,j,i));
    }
    // enforce trace of A to be zero
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      LOOP1(i) {
        z4c.A_dd(a,b,k,j,i) -= (1.0/3.0) * A(i) * z4c.g_dd(a,b,k,j,i);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::GaugeGeodesic(AthenaArray<Real> & u)
// \brief Initialize lapse to 1 and shift to 0

void Z4c::GaugeGeodesic(AthenaArray<Real> & u)
{
  Z4c_vars z4c;
  SetZ4cAliases(u, z4c);
  z4c.alpha.Fill(1.);
  z4c.beta_u.Fill(0.);
}
