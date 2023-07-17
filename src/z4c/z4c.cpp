//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c.cpp
//  \brief implementation of functions in the Z4c class

#include <iostream>
#include <fstream>

// Athena++ headers
#include "z4c.hpp"
#include "z4c_macro.hpp"
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../outputs/outputs.hpp"

// constructor, initializes data structures and parameters

char const * const Z4c::Z4c_names[Z4c::N_Z4c] = {
  "z4c.chi",
  "z4c.gxx", "z4c.gxy", "z4c.gxz", "z4c.gyy", "z4c.gyz", "z4c.gzz",
  "z4c.Khat",
  "z4c.Axx", "z4c.Axy", "z4c.Axz", "z4c.Ayy", "z4c.Ayz", "z4c.Azz",
  "z4c.Gamx", "z4c.Gamy", "z4c.Gamz",
  "z4c.Theta",
  "z4c.alpha",
  "z4c.betax", "z4c.betay", "z4c.betaz",
};

char const * const Z4c::ADM_names[Z4c::N_ADM] = {
  "adm.gxx", "adm.gxy", "adm.gxz", "adm.gyy", "adm.gyz", "adm.gzz",
  "adm.Kxx", "adm.Kxy", "adm.Kxz", "adm.Kyy", "adm.Kyz", "adm.Kzz",
  "adm.psi4",
};

char const * const Z4c::Constraint_names[Z4c::N_CON] = {
  "con.C",
  "con.H",
  "con.M",
  "con.Z",
  "con.Mx", "con.My", "con.Mz",
};

char const * const Z4c::Matter_names[Z4c::N_MAT] = {
  "mat.rho",
  "mat.Sx", "mat.Sy", "mat.Sz",
  "mat.Sxx", "mat.Sxy", "mat.Sxz", "mat.Syy", "mat.Syz", "mat.Szz",
};

char const * const Z4c::Weyl_names[Z4c::N_WEY] = {
  "weyl.rpsi4","weyl.ipsi4",
};

Z4c::Z4c(MeshBlock *pmb, ParameterInput *pin) :
  pmy_block(pmb),
  coarse_u_(N_Z4c, pmb->ncv3, pmb->ncv2, pmb->ncv1,
            (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
             AthenaArray<Real>::DataStatus::empty)),
  storage{{N_Z4c, pmb->nverts3, pmb->nverts2, pmb->nverts1}, // u
          {N_Z4c, pmb->nverts3, pmb->nverts2, pmb->nverts1}, // u1
          {},                                                // u2
          {N_Z4c, pmb->nverts3, pmb->nverts2, pmb->nverts1}, // rhs
          {N_ADM, pmb->nverts3, pmb->nverts2, pmb->nverts1}, // adm
          {N_CON, pmb->nverts3, pmb->nverts2, pmb->nverts1}, // con
          {N_MAT, pmb->nverts3, pmb->nverts2, pmb->nverts1}, // mat
          {N_WEY, pmb->nverts3, pmb->nverts2, pmb->nverts1}, // weyl
  },
  empty_flux{AthenaArray<Real>(), AthenaArray<Real>(), AthenaArray<Real>()},
  ubvar(pmb, &storage.u, &coarse_u_, empty_flux)
{
  Mesh *pm = pmy_block->pmy_mesh;
  Coordinates * pco = pmb->pcoord;

  //---------------------------------------------------------------------------

  // inform MeshBlock that this array is the "primary" representation
  // Used for:
  // (1) load-balancing
  // (2) (future) dumping to restart file
  pmb->RegisterMeshBlockDataVC(storage.u);

  // "Enroll" in SMR/AMR by adding to vector of pointers in MeshRefinement class
  if (pm->multilevel) {
    refinement_idx = pmy_block->pmr->AddToRefinementVC(&storage.u, &coarse_u_);
  }

  // If user-requested time integrator is type 3S* allocate additional memory
  std::string integrator = pin->GetOrAddString("time", "integrator", "vl2");
  if (integrator == "ssprk5_4")
    storage.u2.NewAthenaArray(N_Z4c, pmb->nverts3, pmb->nverts2, pmb->nverts1);

  // enroll CellCenteredBoundaryVariable / VertexCenteredBoundaryVariable object
  ubvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&ubvar);
  pmb->pbval->bvars_main_int_vc.push_back(&ubvar);

  dt1_.NewAthenaArray(pmb->nverts1);
  dt2_.NewAthenaArray(pmb->nverts1);
  dt3_.NewAthenaArray(pmb->nverts1);

  // BD: TODO shift defaults to header [C++11 so can use def. declaration]..
  // Parameters
  opt.chi_psi_power = pin->GetOrAddReal("z4c", "chi_psi_power", -4.0);
  opt.chi_div_floor = pin->GetOrAddReal("z4c", "chi_div_floor", -1000.0);
  opt.diss = pin->GetOrAddReal("z4c", "diss", 0.0);
  opt.eps_floor = pin->GetOrAddReal("z4c", "eps_floor", 1e-12);
  opt.damp_kappa1 = pin->GetOrAddReal("z4c", "damp_kappa1", 0.0);
  opt.damp_kappa2 = pin->GetOrAddReal("z4c", "damp_kappa2", 0.0);
  // Gauge conditions (default to moving puncture gauge)
  opt.lapse_harmonicf = pin->GetOrAddReal("z4c", "lapse_harmonicf", 1.0);
  opt.lapse_harmonic = pin->GetOrAddReal("z4c", "lapse_harmonic", 0.0);
  opt.lapse_oplog = pin->GetOrAddReal("z4c", "lapse_oplog", 2.0);
  opt.lapse_advect = pin->GetOrAddReal("z4c", "lapse_advect", 1.0);
  opt.shift_Gamma = pin->GetOrAddReal("z4c", "shift_Gamma", 1.0);
  opt.shift_advect = pin->GetOrAddReal("z4c", "shift_advect", 1.0);

  opt.shift_alpha2Gamma = pin->GetOrAddReal("z4c", "shift_alpha2Gamma", 0.0);
  opt.shift_H = pin->GetOrAddReal("z4c", "shift_H", 0.0);

  opt.shift_eta = pin->GetOrAddReal("z4c", "shift_eta", 2.0);

  // Problem-specific parameters
  // Two punctures parameters

  // AwA parameters (default to linear wave test)

  opt.AwA_amplitude = pin->GetOrAddReal("z4c", "AwA_amplitude", 1e-10);
  opt.AwA_d_x = pin->GetOrAddReal("z4c", "AwA_d_x", 1.0);
  opt.AwA_d_y = pin->GetOrAddReal("z4c", "AwA_d_y", 1.0);
  opt.AwA_Gaussian_w = pin->GetOrAddReal("z4c", "AwA_Gaussian_w", 0.5);
  opt.AwA_polarised_Gowdy_t0 = pin->GetOrAddReal("z4c",
    "AwA_polarised_Gowdy_t0", 9.8753205829098);

  // sphere-zone refinement test [disabled by default]
  opt.sphere_zone_number = pin->GetOrAddInteger("z4c", "sphere_zone_number", 0);
  if (opt.sphere_zone_number > 0) {
    opt.sphere_zone_levels.NewAthenaArray(opt.sphere_zone_number);
    opt.sphere_zone_radii.NewAthenaArray(opt.sphere_zone_number);
    opt.sphere_zone_puncture.NewAthenaArray(opt.sphere_zone_number);
    opt.sphere_zone_center1.NewAthenaArray(opt.sphere_zone_number);
    opt.sphere_zone_center2.NewAthenaArray(opt.sphere_zone_number);
    opt.sphere_zone_center3.NewAthenaArray(opt.sphere_zone_number);

    for (int i=0; i<opt.sphere_zone_number; ++i) {
      opt.sphere_zone_levels(i) = pin->GetOrAddInteger("z4c",
        "sphere_zone_level_" + std::to_string(i), 0);
      opt.sphere_zone_radii(i) = pin->GetOrAddReal("z4c",
        "sphere_zone_radius_" + std::to_string(i), 0.);
      opt.sphere_zone_puncture(i) = pin->GetOrAddInteger("z4c",
        "sphere_zone_puncture_" + std::to_string(i), -1);
      // populate centers
      opt.sphere_zone_center1(i) = pin->GetOrAddReal("z4c",
        "sphere_zone_center1_" + std::to_string(i), 0.);
      opt.sphere_zone_center2(i) = pin->GetOrAddReal("z4c",
        "sphere_zone_center2_" + std::to_string(i), 0.);
      opt.sphere_zone_center3(i) = pin->GetOrAddReal("z4c",
        "sphere_zone_center3_" + std::to_string(i), 0.);
    }
  }

  //---------------------------------------------------------------------------
  // Set aliases
  SetADMAliases(storage.adm, adm);
  SetConstraintAliases(storage.con, con);
  SetMatterAliases(storage.mat, mat);
  SetZ4cAliases(storage.rhs, rhs);
  SetZ4cAliases(storage.u, z4c);
  SetWeylAliases(storage.weyl, weyl);
  // Allocate memory for aux 1D vars
  r.NewAthenaTensor(pmb->nverts1);
  detg.NewAthenaTensor(pmb->nverts1);
  chi_guarded.NewAthenaTensor(pmb->nverts1);
  oopsi4.NewAthenaTensor(pmb->nverts1);
  A.NewAthenaTensor(pmb->nverts1);
  AA.NewAthenaTensor(pmb->nverts1);
  R.NewAthenaTensor(pmb->nverts1);
  Ht.NewAthenaTensor(pmb->nverts1);
  K.NewAthenaTensor(pmb->nverts1);
  KK.NewAthenaTensor(pmb->nverts1);
  Ddalpha.NewAthenaTensor(pmb->nverts1);
  S.NewAthenaTensor(pmb->nverts1);
  M_u.NewAthenaTensor(pmb->nverts1);
  Gamma_u.NewAthenaTensor(pmb->nverts1);
  DA_u.NewAthenaTensor(pmb->nverts1);
  s_u.NewAthenaTensor(pmb->nverts1);
  g_uu.NewAthenaTensor(pmb->nverts1);
  A_uu.NewAthenaTensor(pmb->nverts1);
  AA_dd.NewAthenaTensor(pmb->nverts1);
  R_dd.NewAthenaTensor(pmb->nverts1);
  Rphi_dd.NewAthenaTensor(pmb->nverts1);
  Kt_dd.NewAthenaTensor(pmb->nverts1);
  K_ud.NewAthenaTensor(pmb->nverts1);
  Ddalpha_dd.NewAthenaTensor(pmb->nverts1);
  Ddphi_dd.NewAthenaTensor(pmb->nverts1);
  Gamma_ddd.NewAthenaTensor(pmb->nverts1);
  Gamma_udd.NewAthenaTensor(pmb->nverts1);
  DK_ddd.NewAthenaTensor(pmb->nverts1);
  DK_udd.NewAthenaTensor(pmb->nverts1);

#if defined(Z4C_ETA_CONF) || defined(Z4C_ETA_TRACK_TP)
  eta_damp.NewAthenaTensor(pmb->nverts1);
#endif // Z4C_ETA_CONF, Z4C_ETA_TRACK_TP

  dbeta.NewAthenaTensor(pmb->nverts1);
  dalpha_d.NewAthenaTensor(pmb->nverts1);
  ddbeta_d.NewAthenaTensor(pmb->nverts1);
  dchi_d.NewAthenaTensor(pmb->nverts1);
  dphi_d.NewAthenaTensor(pmb->nverts1);
  dK_d.NewAthenaTensor(pmb->nverts1);
  dKhat_d.NewAthenaTensor(pmb->nverts1);
  dTheta_d.NewAthenaTensor(pmb->nverts1);
  ddalpha_dd.NewAthenaTensor(pmb->nverts1);
  dbeta_du.NewAthenaTensor(pmb->nverts1);
  ddchi_dd.NewAthenaTensor(pmb->nverts1);
  dGam_du.NewAthenaTensor(pmb->nverts1);
  dg_ddd.NewAthenaTensor(pmb->nverts1);
  dK_ddd.NewAthenaTensor(pmb->nverts1);
  dA_ddd.NewAthenaTensor(pmb->nverts1);
  ddbeta_ddu.NewAthenaTensor(pmb->nverts1);
  ddg_dddd.NewAthenaTensor(pmb->nverts1);

  Lchi.NewAthenaTensor(pmb->nverts1);
  LKhat.NewAthenaTensor(pmb->nverts1);
  LTheta.NewAthenaTensor(pmb->nverts1);
  Lalpha.NewAthenaTensor(pmb->nverts1);
  LGam_u.NewAthenaTensor(pmb->nverts1);
  Lbeta_u.NewAthenaTensor(pmb->nverts1);
  Lg_dd.NewAthenaTensor(pmb->nverts1);
  LA_dd.NewAthenaTensor(pmb->nverts1);

  uvec.NewAthenaTensor(pmb->nverts1);
  vvec.NewAthenaTensor(pmb->nverts1);
  wvec.NewAthenaTensor(pmb->nverts1);
  dotp1.NewAthenaTensor(pmb->nverts1);
  dotp2.NewAthenaTensor(pmb->nverts1);
  Riem3_dddd.NewAthenaTensor(pmb->nverts1);
  Riemm4_dddd.NewAthenaTensor(pmb->nverts1);
  Riemm4_ddd.NewAthenaTensor(pmb->nverts1);
  Riemm4_dd.NewAthenaTensor(pmb->nverts1);

//<<<<<<< HEAD
//=======
//WGC wext
  uvec.NewAthenaTensor(nn1);
  vvec.NewAthenaTensor(nn1);
  wvec.NewAthenaTensor(nn1);
  dotp1.NewAthenaTensor(nn1);
  dotp2.NewAthenaTensor(nn1);
  Riem3_dddd.NewAthenaTensor(nn1);
  Riemm4_dddd.NewAthenaTensor(nn1);
  Riemm4_ddd.NewAthenaTensor(nn1);
  Riemm4_dd.NewAthenaTensor(nn1);
//WGC end

//
//>>>>>>> b72e67bb (Add yet-another way to do intergrid interpolation)
  // Set up finite difference operators
  Real dx1, dx2, dx3;
  dx1 = pco->dx1f(0); dx2 = pco->dx2f(0); dx3 = pco->dx3f(0);

  FD.stride[0] = 1;
  FD.stride[1] = 0;
  FD.stride[2] = 0;
  FD.idx[0] = 1.0 / dx1;
  FD.idx[1] = 0.0;
  FD.idx[2] = 0.0;
  if(pmb->nverts2 > 1) {
    FD.stride[1] = pmb->nverts1;
    FD.idx[1] = 1.0 / dx2;
  }
  if(pmb->nverts3 > 1) {
    FD.stride[2] = pmb->nverts2*pmb->nverts1;
    FD.idx[2] = 1.0 / dx3;
  }
  FD.diss = opt.diss*pow(2, -2*NGHOST)*(NGHOST % 2 == 0 ? -1 : 1);
}

// destructor
Z4c::~Z4c()
{
  storage.u.DeleteAthenaArray();
  storage.u1.DeleteAthenaArray();
  storage.u2.DeleteAthenaArray();
  storage.rhs.DeleteAthenaArray();
  storage.adm.DeleteAthenaArray();
  storage.con.DeleteAthenaArray();
  storage.mat.DeleteAthenaArray();
  storage.weyl.DeleteAthenaArray();
  dt1_.DeleteAthenaArray();
  dt2_.DeleteAthenaArray();
  dt3_.DeleteAthenaArray();

  r.DeleteAthenaTensor();
  detg.DeleteAthenaTensor();
  chi_guarded.DeleteAthenaTensor();
  oopsi4.DeleteAthenaTensor();
  A.DeleteAthenaTensor();
  AA.DeleteAthenaTensor();
  R.DeleteAthenaTensor();
  Ht.DeleteAthenaTensor();
  K.DeleteAthenaTensor();
  KK.DeleteAthenaTensor();
  Ddalpha.DeleteAthenaTensor();
  S.DeleteAthenaTensor();
  M_u.DeleteAthenaTensor();
  Gamma_u.DeleteAthenaTensor();
  DA_u.DeleteAthenaTensor();
  s_u.DeleteAthenaTensor();
  g_uu.DeleteAthenaTensor();
  A_uu.DeleteAthenaTensor();
  AA_dd.DeleteAthenaTensor();
  R_dd.DeleteAthenaTensor();
  Rphi_dd.DeleteAthenaTensor();
  Kt_dd.DeleteAthenaTensor();
  K_ud.DeleteAthenaTensor();
  Ddalpha_dd.DeleteAthenaTensor();
  Ddphi_dd.DeleteAthenaTensor();
  Gamma_ddd.DeleteAthenaTensor();
  Gamma_udd.DeleteAthenaTensor();
  DK_ddd.DeleteAthenaTensor();
  DK_udd.DeleteAthenaTensor();

#if defined(Z4C_ETA_CONF) || defined(Z4C_ETA_TRACK_TP)
  eta_damp.DeleteAthenaTensor();
#endif // Z4C_ETA_CONF, Z4C_ETA_TRACK_TP

  dbeta.DeleteAthenaTensor();
  dalpha_d.DeleteAthenaTensor();
  ddbeta_d.DeleteAthenaTensor();
  dchi_d.DeleteAthenaTensor();
  dphi_d.DeleteAthenaTensor();
  dK_d.DeleteAthenaTensor();
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
  uvec.DeleteAthenaTensor();
  vvec.DeleteAthenaTensor();
  wvec.DeleteAthenaTensor();
  dotp1.DeleteAthenaTensor();
  dotp2.DeleteAthenaTensor();
  Riem3_dddd.DeleteAthenaTensor();
  Riemm4_dddd.DeleteAthenaTensor();
  Riemm4_ddd.DeleteAthenaTensor();
  Riemm4_dd.DeleteAthenaTensor();
//<<<<<<< HEAD

  if (opt.sphere_zone_number > 0) {
    opt.sphere_zone_levels.DeleteAthenaArray();
    opt.sphere_zone_radii.DeleteAthenaArray();
    opt.sphere_zone_puncture.DeleteAthenaArray();
    opt.sphere_zone_center1.DeleteAthenaArray();
    opt.sphere_zone_center2.DeleteAthenaArray();
    opt.sphere_zone_center3.DeleteAthenaArray();
  }
//=======
//WGC end
//>>>>>>> b72e67bb (Add yet-another way to do intergrid interpolation)
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::SetADMAliases(AthenaArray<Real> & u, ADM_vars & z4c)
// \brief Set ADM aliases

void Z4c::SetADMAliases(AthenaArray<Real> & u_adm, Z4c::ADM_vars & adm)
{
  adm.psi4.InitWithShallowSlice(u_adm, I_ADM_psi4);
  adm.g_dd.InitWithShallowSlice(u_adm, I_ADM_gxx);
  adm.K_dd.InitWithShallowSlice(u_adm, I_ADM_Kxx);
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::SetConstraintAliases(AthenaArray<Real> & u, ADM_vars & z4c)
// \brief Set ADM aliases

void Z4c::SetConstraintAliases(AthenaArray<Real> & u_con, Z4c::Constraint_vars & con)
{
  con.C.InitWithShallowSlice(u_con, I_CON_C);
  con.H.InitWithShallowSlice(u_con, I_CON_H);
  con.M.InitWithShallowSlice(u_con, I_CON_M);
  con.Z.InitWithShallowSlice(u_con, I_CON_Z);
  con.M_d.InitWithShallowSlice(u_con, I_CON_Mx);
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::SetMatterAliases(AthenaArray<Real> & u_mat, Matter_vars & mat)
// \brief Set matter aliases

void Z4c::SetMatterAliases(AthenaArray<Real> & u_mat, Z4c::Matter_vars & mat)
{
  mat.rho.InitWithShallowSlice(u_mat, I_MAT_rho);
  mat.S_d.InitWithShallowSlice(u_mat, I_MAT_Sx);
  mat.S_dd.InitWithShallowSlice(u_mat, I_MAT_Sxx);
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
// \!fn void Z4c::SetWeylAliases(AthenaArray<Real> & u, Weyl_vars & weyl)
// \brief Set Weyl aliases

void Z4c::SetWeylAliases(AthenaArray<Real> & u, Z4c::Weyl_vars & weyl)
{
  weyl.rpsi4.InitWithShallowSlice(u, I_WEY_rpsi4);
  weyl.ipsi4.InitWithShallowSlice(u, I_WEY_ipsi4);
}

//----------------------------------------------------------------------------------------
// \!fn Real Z4c::SpatialDet(Real gxx, ... , Real gzz)
// \brief returns determinant of 3-metric

Real Z4c::SpatialDet(Real const gxx, Real const gxy, Real const gxz,
                     Real const gyy, Real const gyz, Real const gzz)
{
  return - SQR(gxz)*gyy + 2*gxy*gxz*gyz - gxx*SQR(gyz) - SQR(gxy)*gzz + gxx*gyy*gzz;
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
  *uxx = (-SQR(gyz) + gyy*gzz)*detginv;
  *uxy = (gxz*gyz  - gxy*gzz)*detginv;
  *uyy = (-SQR(gxz) + gxx*gzz)*detginv;
  *uxz = (-gxz*gyy + gxy*gyz)*detginv;
  *uyz = (gxy*gxz  - gxx*gyz)*detginv;
  *uzz = (-SQR(gxy) + gxx*gyy)*detginv;
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
       - Azz*SQR(gxy) - Ayy*SQR(gxz) - Axx*SQR(gyz)
       ));
}

//----------------------------------------------------------------------------------------
// \!fn Real Z4c::GetMatter(Real detginv, Real gxx, ... , Real gzz, Real Axx, ..., Real Azz)
// \brief Update matter variables from hydro 

#if USETM
void Z4c::GetMatter(AthenaArray<Real> & u_mat, AthenaArray<Real> & u_adm, AthenaArray<Real> & w, AthenaArray<Real> & r, AthenaArray<Real> &bb_cc)
#else
void Z4c::GetMatter(AthenaArray<Real> & u_mat, AthenaArray<Real> & u_adm, AthenaArray<Real> & w, AthenaArray<Real> &bb_cc)
#endif
{
    MeshBlock * pmb = pmy_block;
    Matter_vars mat;
    SetMatterAliases(u_mat, mat);
#if USETM
    Real mb = pmy_block->peos->GetEOS().GetBaryonMass();
#else
    Real gamma_adi = pmy_block->peos->GetGamma(); //NB specific to EOS
#endif
    AthenaArray<Real> epscc;
    int nn1 = pmb->ncells1;
    int nn2 = pmb->ncells2;
    int nn3 = pmb->ncells3;
    if(opt.epsinterp==1){
      epscc.NewAthenaArray(pmb->ncells3,pmb->ncells2,pmb->ncells1);
    }
    AthenaArray<Real> vcgamma_xx,vcgamma_xy,vcgamma_xz,vcgamma_yy;
    AthenaArray<Real> vcgamma_yz,vcgamma_zz,vcbeta_x,vcbeta_y;
    AthenaArray<Real> vcbeta_z, alpha;

    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> W_lor, rhoadm; //lapse
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> beta_u, v_u, v_d, Siadm_d, utilde_u; //lapse
    AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> gamma_dd, Sijadm_dd; //lapse

    ADM_vars adm;
     Z4c_vars z4c;
     SetZ4cAliases(storage.u,z4c);
     
    if(opt.fix_admsource==0){
      SetADMAliases(u_adm,adm);
    } else if (opt.fix_admsource==1){
      SetADMAliases(storage.adm_init,adm);
    }
// Cell centred hydro vars
    if(opt.fix_admsource==0){
      rhocc.InitWithShallowSlice(w,IDN,1);
      pgascc.InitWithShallowSlice(w,IPR,1);
      utilde1cc.InitWithShallowSlice(w,IVX,1);
      utilde2cc.InitWithShallowSlice(w,IVY,1);
      utilde3cc.InitWithShallowSlice(w,IVZ,1);
      for (int Yidx=0; Yidx<NSCALARS; Yidx++) {
        rscalarcc[Yidx].InitWithShallowSlice(r,Yidx,1);
      }
      #if MAGNETIC_FIELDS_ENABLED
      bb1cc.InitWithShallowSlice(bb_cc,IB1,1);
      bb2cc.InitWithShallowSlice(bb_cc,IB2,1);
      bb3cc.InitWithShallowSlice(bb_cc,IB3,1);   //check this!
      #endif
    } else if(opt.fix_admsource==1){
      rhocc.InitWithShallowSlice(pmb->phydro->w_init,IDN,1);
      pgascc.InitWithShallowSlice(pmb->phydro->w_init,IPR,1);
      utilde1cc.InitWithShallowSlice(pmb->phydro->w_init,IVX,1);
      utilde2cc.InitWithShallowSlice(pmb->phydro->w_init,IVY,1);
      utilde3cc.InitWithShallowSlice(pmb->phydro->w_init,IVZ,1);
      for (int Yidx=0; Yidx<NSCALARS; Yidx++) {
        rscalarcc[Yidx].InitWithShallowSlice(r,Yidx,1);
      }
    }

    if(opt.Tmunuinterp==0){
//     Real rhovc, pgasvc, utilde1vc, utilde2vc, utilde3vc, wgas,tmp, gamma_lor, v1,v2,v3,v_1,v_2,v_3,epsvc, bb1vc,bb2vc,bb3vc;
      AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> rhovc, pgasvc, utilde1vc, utilde2vc, utilde3vc, epsvc, bb1vc,bb2vc,bb3vc, tmp, wgas, gamma_lor, v1,v2,v3, detgamma, detg, bsq, b0_u;
      AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> v_d, v_u,  bb_u, bi_u, bi_d, utildevc_u; 
      rhovc.NewAthenaTensor(nn1);
      pgasvc.NewAthenaTensor(nn1);
      epsvc.NewAthenaTensor(nn1);
      utilde1vc.NewAthenaTensor(nn1);
      utilde2vc.NewAthenaTensor(nn1);
      utilde3vc.NewAthenaTensor(nn1);
      #if MAGNETIC_FIELDS_ENABLED
      bb1vc.NewAthenaTensor(nn1);
      bb2vc.NewAthenaTensor(nn1);
      bb3vc.NewAthenaTensor(nn1);
      #endif
      v1.NewAthenaTensor(nn1);
      v2.NewAthenaTensor(nn1);
      v3.NewAthenaTensor(nn1);
      tmp.NewAthenaTensor(nn1);
      wgas.NewAthenaTensor(nn1);
      gamma_lor.NewAthenaTensor(nn1);
      detgamma.NewAthenaTensor(nn1);
      detg.NewAthenaTensor(nn1);
      bsq.NewAthenaTensor(nn1);
      b0_u.NewAthenaTensor(nn1);
      v_d.NewAthenaTensor(nn1);
      v_u.NewAthenaTensor(nn1);
      #if MAGNETIC_FIELDS_ENABLED
      bb_u.NewAthenaTensor(nn1);
      bi_u.NewAthenaTensor(nn1);
      bi_d.NewAthenaTensor(nn1);
      #endif
      utildevc_u.NewAthenaTensor(nn1);
      alpha.InitWithShallowSlice(pmy_block->pz4c->storage.u,Z4c::I_Z4c_alpha,1);
      // interpolate to VC
      ILOOP2(k,j){
        ILOOP1(i){
          rhovc(i) = CCInterpolation(rhocc, k, j, i);
          if(opt.epsinterp==0){
            pgasvc(i) = CCInterpolation(pgascc, k, j, i);
          } else {
            epsvc(i) = CCInterpolation(epscc, k, j, i);
          }
          utilde1vc(i) = CCInterpolation(utilde1cc, k, j, i);
          utilde2vc(i) = CCInterpolation(utilde2cc, k, j, i);
          utilde3vc(i) = CCInterpolation(utilde3cc, k, j, i);
          #if MAGNETIC_FIELDS_ENABLED
          bb1vc(i) = CCInterpolation(bb1cc, k, j, i);
          bb2vc(i) = CCInterpolation(bb2cc, k, j, i);
          bb3vc(i) = CCInterpolation(bb3cc, k, j, i);
          #endif
     
          //   NB specific to EOS
          #if USETM
          Real n = rhovc(i)/mb;
          // FIXME: Generalize to work with EOSs accepting particle fractions.
          Real Y[MAX_SPECIES] = {0.0};
          for (int Yidx=0; Yidx<NSCALARS; Yidx++) {
            Y[Yidx] = CCInterpolation(rscalarcc[Yidx], k, j, i);
          }
          Real T = pmy_block->peos->GetEOS().GetTemperatureFromP(n, pgasvc(i), Y);
          wgas(i) = rhovc(i)*pmy_block->peos->GetEOS().GetEnthalpy(n, T, Y);
          #else
          if(opt.epsinterp==1){
            pgasvc(i) = epsvc(i)*rhovc(i)*(gamma_adi-1.0);
          }
          wgas(i) = rhovc(i) + gamma_adi/(gamma_adi-1.0) * pgasvc(i);
          #endif
          tmp(i) = utilde1vc(i)*utilde1vc(i)*adm.g_dd(0,0,k,j,i)
                 + utilde2vc(i)*utilde2vc(i)*adm.g_dd(1,1,k,j,i) 
                 + utilde3vc(i)*utilde3vc(i)*adm.g_dd(2,2,k,j,i) 
                 + 2.0*utilde1vc(i)*utilde2vc(i)*adm.g_dd(0,1,k,j,i)
                 + 2.0*utilde1vc(i)*utilde3vc(i)*adm.g_dd(0,2,k,j,i) 
                 + 2.0*utilde2vc(i)*utilde3vc(i)*adm.g_dd(1,2,k,j,i);
          gamma_lor(i) = sqrt(1.0+tmp(i));
          //   convert to 3-velocity
          v1(i) = utilde1vc(i)/gamma_lor(i);
          v2(i) = utilde2vc(i)/gamma_lor(i);
          v3(i) = utilde3vc(i)/gamma_lor(i);
     
          v_d(0,i) = v1(i)*adm.g_dd(0,0,k,j,i) + v2(i)*adm.g_dd(0,1,k,j,i) +v3(i)*adm.g_dd(0,2,k,j,i);
          v_d(1,i) = v1(i)*adm.g_dd(0,1,k,j,i) + v2(i)*adm.g_dd(1,1,k,j,i) +v3(i)*adm.g_dd(1,2,k,j,i);
          v_d(2,i) = v1(i)*adm.g_dd(0,2,k,j,i) + v2(i)*adm.g_dd(1,2,k,j,i) +v3(i)*adm.g_dd(2,2,k,j,i);

          detgamma(i) = SpatialDet(adm.g_dd(0,0,k,j,i),adm.g_dd(0,1,k,j,i), adm.g_dd(0,2,k,j,i), 
                                   adm.g_dd(1,1,k,j,i), adm.g_dd(1,2,k,j,i), adm.g_dd(2,2,k,j,i));
          detg(i) = alpha(k,j,i)*detgamma(i);

          #if MAGNETIC_FIELDS_ENABLED
          bb_u(0,i) = bb1vc(i)/detg(i);
          bb_u(1,i) = bb2vc(i)/detg(i);
          bb_u(2,i) = bb3vc(i)/detg(i);
          #endif
        }
        //b0_u = 0.0;
        #if MAGNETIC_FIELDS_ENABLED
        b0_u.ZeroClear();
        for(int a=0;a<NDIM;++a){
          ILOOP1(i){
            b0_u(i) += gamma_lor(i)*bb_u(a,i)*v_d(a,i)/alpha(k,j,i);
          }
        }
        #endif

        ILOOP1(i){
          utildevc_u(0,i) = utilde1vc(i);
          utildevc_u(1,i) = utilde2vc(i);
          utildevc_u(2,i) = utilde3vc(i);
          v_u(0,i)        = v1(i);
          v_u(1,i)        = v2(i);
          v_u(2,i)        = v3(i);
        }

        #if MAGNETIC_FIELDS_ENABLED
        for(int a=0;a<NDIM;++a){
          ILOOP1(i){
            bi_u(a,i) = (bb_u(a,i) + alpha(k,j,i)*b0_u(i)*gamma_lor(i)*(v_u(a,i) - z4c.beta_u(a,k,j,i)/alpha(i)))/gamma_lor(i);
          }
        }
        bi_d.ZeroClear();
        for(int a=0;a<NDIM;++a){
          for(int b=0;b<NDIM;++b){
            ILOOP1(i){
              bi_d(a,i) += bi_u(b,i)*adm.g_dd(a,b,k,j,i);
            }
          }
        }
        ILOOP1(i){
          bsq(i) = alpha(k,j,i)*alpha(k,j,i)*b0_u(i)*b0_u(i)/(gamma_lor(i)*gamma_lor(i));
        }
        for(int a=0;a<NDIM;++a){
          for(int b=0;b<NDIM;++b){
            ILOOP1(i){
              bsq(i) += bb_u(a,i)*bb_u(b,i)*adm.g_dd(a,b,k,j,i)/(gamma_lor(i)*gamma_lor(i));
            }
          }
        }
        #endif



        #if MAGNETIC_FIELDS_ENABLED
        ILOOP1(i){
          mat.rho(k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i)) - (pgasvc(i) + bsq(i)/2.0) - alpha(k,j,i)*alpha(k,j,i)*b0_u(i)*b0_u(i);
          mat.S_d(0,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))*v_d(0,i) - b0_u(i)*bi_d(0,i)*alpha(i);
          mat.S_d(1,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))*v_d(1,i) - b0_u(i)*bi_d(1,i)*alpha(i);
          mat.S_d(2,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))*v_d(2,i) - b0_u(i)*bi_d(2,i)*alpha(i);
          mat.S_dd(0,0,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))* v_d(0,i)*v_d(0,i) + (pgasvc(i)+bsq(i)/2.0)*adm.g_dd(0,0,k,j,i) - bi_d(0,i)*bi_d(0,i);
          mat.S_dd(0,1,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))* v_d(0,i)*v_d(1,i) + (pgasvc(i)+bsq(i)/2.0)*adm.g_dd(0,1,k,j,i) - bi_d(0,i)*bi_d(1,i);
          mat.S_dd(0,2,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))* v_d(0,i)*v_d(2,i) + (pgasvc(i)+bsq(i)/2.0)*adm.g_dd(0,2,k,j,i) - bi_d(0,i)*bi_d(2,i);
          mat.S_dd(1,1,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))* v_d(1,i)*v_d(1,i) + (pgasvc(i)+bsq(i)/2.0)*adm.g_dd(1,1,k,j,i) - bi_d(1,i)*bi_d(1,i);
          mat.S_dd(1,2,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))* v_d(1,i)*v_d(2,i) + (pgasvc(i)+bsq(i)/2.0)*adm.g_dd(1,2,k,j,i) - bi_d(1,i)*bi_d(2,i);
          mat.S_dd(2,2,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))* v_d(2,i)*v_d(2,i) + (pgasvc(i)+bsq(i)/2.0)*adm.g_dd(2,2,k,j,i) - bi_d(2,i)*bi_d(2,i);
        }
        #else
        ILOOP1(i) {
          mat.rho(k,j,i) = wgas(i)*SQR(gamma_lor(i)) - pgasvc(i);
          mat.S_d(0,k,j,i) = wgas(i)*SQR(gamma_lor(i))*v_d(0,i);
          mat.S_d(1,k,j,i) = wgas(i)*SQR(gamma_lor(i))*v_d(1,i);
          mat.S_d(2,k,j,i) = wgas(i)*SQR(gamma_lor(i))*v_d(2,i);
          mat.S_dd(0,0,k,j,i) = wgas(i)*SQR(gamma_lor(i))*v_d(0,i)*v_d(0,i) + pgasvc(i)*adm.g_dd(0,0,k,j,i);
          mat.S_dd(0,1,k,j,i) = wgas(i)*SQR(gamma_lor(i))*v_d(0,i)*v_d(1,i) + pgasvc(i)*adm.g_dd(0,1,k,j,i);
          mat.S_dd(0,2,k,j,i) = wgas(i)*SQR(gamma_lor(i))*v_d(0,i)*v_d(2,i) + pgasvc(i)*adm.g_dd(0,2,k,j,i);
          mat.S_dd(1,1,k,j,i) = wgas(i)*SQR(gamma_lor(i))*v_d(1,i)*v_d(1,i) + pgasvc(i)*adm.g_dd(1,1,k,j,i);
          mat.S_dd(1,2,k,j,i) = wgas(i)*SQR(gamma_lor(i))*v_d(1,i)*v_d(2,i) + pgasvc(i)*adm.g_dd(1,2,k,j,i);
          mat.S_dd(2,2,k,j,i) = wgas(i)*SQR(gamma_lor(i))*v_d(2,i)*v_d(2,i) + pgasvc(i)*adm.g_dd(2,2,k,j,i);
        }
        #endif
      }
    }
}

//----------------------------------------------------------------------------------------
//>>>>>>> b72e67bb (Add yet-another way to do intergrid interpolation)
// \!fn void Z4c::AlgConstr(AthenaArray<Real> & u)
// \brief algebraic constraints projection
//
// This function operates on all grid points of the MeshBlock

void Z4c::AlgConstr(AthenaArray<Real> & u)
{
  Z4c_vars z4c;
  SetZ4cAliases(u, z4c);

  GLOOP2(k,j) {

    // compute determinant and "conformal conformal factor"
    GLOOP1(i) {
      detg(i) = SpatialDet(z4c.g_dd,k,j,i);
      detg(i) = detg(i) > 0. ? detg(i) : 1.;
      Real eps = detg(i) - 1.;
      oopsi4(i) = (eps < opt.eps_floor) ? (1. - opt.eps_floor/3.) : (pow(1./detg(i), 1./3.));
    }
    // enforce unitary determinant for conformal metric
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      GLOOP1(i) {
        z4c.g_dd(a,b,k,j,i) *= oopsi4(i);
      }
    }

    // compute trace of A
    GLOOP1(i) {
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
      GLOOP1(i) {
        z4c.A_dd(a,b,k,j,i) -= (1.0/3.0) * A(i) * z4c.g_dd(a,b,k,j,i);
      }
    }
  }
}
