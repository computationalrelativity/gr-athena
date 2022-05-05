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
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../utils/interp_intergrid.hpp"

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
// WGC wext
char const * const Z4c::Weyl_names[Z4c::N_WEY] = {
  "weyl.rpsi4","weyl.ipsi4",
};
//WGC end
Z4c::Z4c(MeshBlock *pmb, ParameterInput *pin) :
  pmy_block(pmb),
#if PREFER_VC
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
//WGC wext
          {N_WEY, pmb->nverts3, pmb->nverts2, pmb->nverts1}, // weyl
//init buffers
          {N_Z4c, pmb->nverts3, pmb->nverts2, pmb->nverts1},
          {N_ADM, pmb->nverts3, pmb->nverts2, pmb->nverts1},
//WGC end
  },
#else
  coarse_u_(N_Z4c, pmb->ncc3, pmb->ncc2, pmb->ncc1,
            (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
             AthenaArray<Real>::DataStatus::empty)),
  storage{{N_Z4c, pmb->ncells3, pmb->ncells2, pmb->ncells1}, // u
          {N_Z4c, pmb->ncells3, pmb->ncells2, pmb->ncells1}, // u1
          {},                                                // u2
          {N_Z4c, pmb->ncells3, pmb->ncells2, pmb->ncells1}, // rhs
          {N_ADM, pmb->ncells3, pmb->ncells2, pmb->ncells1}, // adm
          {N_CON, pmb->ncells3, pmb->ncells2, pmb->ncells1}, // con
          {N_MAT, pmb->ncells3, pmb->ncells2, pmb->ncells1}, // mat
//WGC wext
          {N_WEY, pmb->ncells3, pmb->ncells2, pmb->ncells1}, // weyl
//WGC end
  },
#endif
  empty_flux{AthenaArray<Real>(), AthenaArray<Real>(), AthenaArray<Real>()},
  ubvar(pmb, &storage.u, &coarse_u_, empty_flux)
{
  Mesh *pm = pmy_block->pmy_mesh;
  Coordinates * pco = pmb->pcoord;

  // dimensions required for data allocation
  if (PREFER_VC) {
    mbi.nn1 = pmb->nverts1;
    mbi.nn2 = pmb->nverts2;
    mbi.nn3 = pmb->nverts3;
  } else {
    mbi.nn1 = pmb->ncells1;
    mbi.nn2 = pmb->ncells2;
    mbi.nn3 = pmb->ncells3;
  }
  int nn1 = mbi.nn1, nn2 = mbi.nn2, nn3 = mbi.nn3;

  // convenience for per-block iteration (private Wave scope)
  mbi.il = pmb->is; mbi.jl = pmb->js; mbi.kl = pmb->ks;
  if (PREFER_VC) {
    mbi.iu = pmb->ive; mbi.ju = pmb->jve; mbi.ku = pmb->kve;
  } else {
    mbi.iu = pmb->ie; mbi.ju = pmb->je; mbi.ku = pmb->ke;
  }

  // point to appropriate grid
  if (PREFER_VC) {
    mbi.x1.InitWithShallowSlice(pco->x1f, 1, 0, nn1);
    mbi.x2.InitWithShallowSlice(pco->x2f, 1, 0, nn2);
    mbi.x3.InitWithShallowSlice(pco->x3f, 1, 0, nn3);
  } else {
    mbi.x1.InitWithShallowSlice(pco->x1v, 1, 0, nn1);
    mbi.x2.InitWithShallowSlice(pco->x2v, 1, 0, nn2);
    mbi.x3.InitWithShallowSlice(pco->x3v, 1, 0, nn3);
  }
  //---------------------------------------------------------------------------

  // inform MeshBlock that this array is the "primary" representation
  // Used for:
  // (1) load-balancing
  // (2) (future) dumping to restart file
  // WGC separated VC and CC
  pmb->RegisterMeshBlockDataVC(storage.u);

  // "Enroll" in SMR/AMR by adding to vector of pointers in MeshRefinement class
  if (pm->multilevel) {
    refinement_idx = pmy_block->pmr->AddToRefinementVC(&storage.u, &coarse_u_);
  }


  // If user-requested time integrator is type 3S* allocate additional memory
  std::string integrator = pin->GetOrAddString("time", "integrator", "vl2");
  if (integrator == "ssprk5_4")
    storage.u2.NewAthenaArray(N_Z4c, nn3, nn2, nn1);

  // enroll CellCenteredBoundaryVariable / VertexCenteredBoundaryVariable object
  ubvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&ubvar);
  if (PREFER_VC) {
    pmb->pbval->bvars_main_int_vc.push_back(&ubvar);
  } else {
    pmb->pbval->bvars_main_int.push_back(&ubvar);
  }

  dt1_.NewAthenaArray(nn1);
  dt2_.NewAthenaArray(nn1);
  dt3_.NewAthenaArray(nn1);

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
  opt.shift_advect = pin->GetOrAddReal("z4c", "shift_advect", 1.0);

  opt.shift_eta = pin->GetOrAddReal("z4c", "shift_eta", 2.0);

#if defined(Z4C_ETA_CONF)
  // Spatially dependent shift damping [based on conf. factor]
  opt.shift_eta_a = pin->GetOrAddReal("z4c", "shift_eta_a", 2.);
  opt.shift_eta_b = pin->GetOrAddReal("z4c", "shift_eta_b", 2.);
  opt.shift_eta_R_0 = pin->GetOrAddReal("z4c", "shift_eta_R_0", 1.31);
#elif defined(Z4C_ETA_TRACK_TP)
  // Spatially dependent shift damping [based on puncture dist.]
  opt.shift_eta_w = pin->GetOrAddReal("z4c", "shift_eta_w", 2.67);
  opt.shift_eta_delta = pin->GetOrAddReal("z4c", "shift_eta_delta", 2.0);
  opt.shift_eta_P = pin->GetOrAddReal("z4c", "shift_eta_P", 3);
  opt.shift_eta_TP_ix = pin->GetOrAddInteger("z4c", "shift_eta_TP_ix", 0);
#endif // Z4C_ETA_CONF, Z4C_ETA_TRACK_TP

  // Problem-specific parameters
  // Two punctures parameters

  // AwA parameters (default to linear wave test)

  opt.AwA_amplitude = pin->GetOrAddReal("z4c", "AwA_amplitude", 1e-10);
  opt.AwA_d_x = pin->GetOrAddReal("z4c", "AwA_d_x", 1.0);
  opt.AwA_d_y = pin->GetOrAddReal("z4c", "AwA_d_y", 1.0);
  opt.AwA_Gaussian_w = pin->GetOrAddReal("z4c", "AwA_Gaussian_w", 0.5);
  opt.AwA_polarised_Gowdy_t0 = pin->GetOrAddReal("z4c",
    "AwA_polarised_Gowdy_t0", 9.8753205829098);
  
  // Matter parameters
  opt.cowling = pin->GetOrAddInteger("z4c", "cowling_true", 0);
  opt.rhstheta0 = pin->GetOrAddInteger("z4c", "rhstheta0", 0);
  opt.fixedgauge = pin->GetOrAddInteger("z4c", "fixedgauge", 0);
  opt.fix_admsource = pin->GetOrAddInteger("z4c", "fix_admsource", 0);
  opt.Tmunuinterp = pin->GetOrAddInteger("z4c", "Tmunuinterp", 0); // interpolate components of Tmunu if 1 (if 0 interpolate primitives)
  opt.epsinterp = pin->GetOrAddInteger("z4c", "epsinterp", 0); // interpolate internal energy eps instead of pressure p.
  //---------------------------------------------------------------------------

  // Set aliases
  SetADMAliases(storage.adm, adm);
  SetConstraintAliases(storage.con, con);
  SetMatterAliases(storage.mat, mat);
  SetZ4cAliases(storage.rhs, rhs);
  SetZ4cAliases(storage.u, z4c);
//WGC wext
  SetWeylAliases(storage.weyl, weyl);
//WGC end
  // Allocate memory for aux 1D vars
  r.NewAthenaTensor(nn1);
  detg.NewAthenaTensor(nn1);
  chi_guarded.NewAthenaTensor(nn1);
  oopsi4.NewAthenaTensor(nn1);
  A.NewAthenaTensor(nn1);
  AA.NewAthenaTensor(nn1);
  R.NewAthenaTensor(nn1);
  Ht.NewAthenaTensor(nn1);
  K.NewAthenaTensor(nn1);
  KK.NewAthenaTensor(nn1);
  Ddalpha.NewAthenaTensor(nn1);
  S.NewAthenaTensor(nn1);
  M_u.NewAthenaTensor(nn1);
  Gamma_u.NewAthenaTensor(nn1);
  DA_u.NewAthenaTensor(nn1);
  s_u.NewAthenaTensor(nn1);
  g_uu.NewAthenaTensor(nn1);
  A_uu.NewAthenaTensor(nn1);
  AA_dd.NewAthenaTensor(nn1);
  R_dd.NewAthenaTensor(nn1);
  Rphi_dd.NewAthenaTensor(nn1);
  Kt_dd.NewAthenaTensor(nn1);
  K_ud.NewAthenaTensor(nn1);
  Ddalpha_dd.NewAthenaTensor(nn1);
  Ddphi_dd.NewAthenaTensor(nn1);
  Gamma_ddd.NewAthenaTensor(nn1);
  Gamma_udd.NewAthenaTensor(nn1);
  DK_ddd.NewAthenaTensor(nn1);
  DK_udd.NewAthenaTensor(nn1);

// testcons
  tGamma_ddd.NewAthenaTensor(nn1);
  tGamma_udd.NewAthenaTensor(nn1);
  tdg_ddd.NewAthenaTensor(nn1);
  tdetg.NewAthenaTensor(nn1);
  tg_uu.NewAthenaTensor(nn1);
  tGamma_u.NewAthenaTensor(nn1);

#if defined(Z4C_ETA_CONF) || defined(Z4C_ETA_TRACK_TP)
  eta_damp.NewAthenaTensor(nn1);
#endif // Z4C_ETA_CONF, Z4C_ETA_TRACK_TP


  dbeta.NewAthenaTensor(nn1);
  dalpha_d.NewAthenaTensor(nn1);
  ddbeta_d.NewAthenaTensor(nn1);
  dchi_d.NewAthenaTensor(nn1);
  dphi_d.NewAthenaTensor(nn1);
  dK_d.NewAthenaTensor(nn1);
  dKhat_d.NewAthenaTensor(nn1);
  dTheta_d.NewAthenaTensor(nn1);
  ddalpha_dd.NewAthenaTensor(nn1);
  dbeta_du.NewAthenaTensor(nn1);
  ddchi_dd.NewAthenaTensor(nn1);
  dGam_du.NewAthenaTensor(nn1);
  dg_ddd.NewAthenaTensor(nn1);
  dg_duu.NewAthenaTensor(nn1);
  dK_ddd.NewAthenaTensor(nn1);
  dA_ddd.NewAthenaTensor(nn1);
  ddbeta_ddu.NewAthenaTensor(nn1);
  ddg_dddd.NewAthenaTensor(nn1);

  Lchi.NewAthenaTensor(nn1);
  LKhat.NewAthenaTensor(nn1);
  LTheta.NewAthenaTensor(nn1);
  Lalpha.NewAthenaTensor(nn1);
  LGam_u.NewAthenaTensor(nn1);
  Lbeta_u.NewAthenaTensor(nn1);
  Lg_dd.NewAthenaTensor(nn1);
  LA_dd.NewAthenaTensor(nn1);

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

//Intergrid communication
  int N[] = {pmb->block_size.nx1, pmb->block_size.nx2, pmb->block_size.nx3};
  Real rdx[] = {
    1./pmb->pcoord->dx1f(0), 1./pmb->pcoord->dx2f(0), 1./pmb->pcoord->dx3f(0)
  };
  ig = new InterpIntergridLocal(NDIM, &N[0], &rdx[0]);
if(pmb->pmy_mesh->multilevel){
  int N_coarse[] = {pmb->ncv1, pmb->ncv2, pmb->ncv3};
  Real rdx_coarse[] = {
    1./pmb->pmr->pcoarsec->dx1f(0), 1./pmb->pmr->pcoarsec->dx2f(0), 1./pmb->pmr->pcoarsec->dx3f(0)
  };
  ig_coarse = new InterpIntergridLocal(NDIM, &N_coarse[0], &rdx_coarse[0]);

}

//
  // Set up finite difference operators
  Real dx1, dx2, dx3;
  if (PREFER_VC) {
    dx1 = pco->dx1f(0); dx2 = pco->dx2f(0); dx3 = pco->dx3f(0);
  } else {
    dx1 = pco->dx1v(0); dx2 = pco->dx2v(0); dx3 = pco->dx3v(0);
  }

  FD.stride[0] = 1;
  FD.stride[1] = 0;
  FD.stride[2] = 0;
  FD.idx[0] = 1.0 / dx1;
  FD.idx[1] = 0.0;
  FD.idx[2] = 0.0;
  if(nn2 > 1) {
    FD.stride[1] = nn1;
    FD.idx[1] = 1.0 / dx2;
  }
  if(nn3 > 1) {
    FD.stride[2] = nn2*nn1;
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
//WGC wext
  storage.weyl.DeleteAthenaArray();
//WGC end
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
  dg_duu.DeleteAthenaTensor();
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
//WGC wext
  uvec.DeleteAthenaTensor();
  vvec.DeleteAthenaTensor();
  wvec.DeleteAthenaTensor();
  dotp1.DeleteAthenaTensor();
  dotp2.DeleteAthenaTensor();
  Riem3_dddd.DeleteAthenaTensor();
  Riemm4_dddd.DeleteAthenaTensor();
  Riemm4_ddd.DeleteAthenaTensor();
  Riemm4_dd.DeleteAthenaTensor();
//WGC end
  delete ig;
if(pmy_block->pmy_mesh->multilevel){
  delete ig_coarse;
}
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
//WGC wext
//----------------------------------------------------------------------------------------
// \!fn void Z4c::SetWeylAliases(AthenaArray<Real> & u, Weyl_vars & weyl)
// \brief Set Weyl aliases

void Z4c::SetWeylAliases(AthenaArray<Real> & u, Z4c::Weyl_vars & weyl)
{
  weyl.rpsi4.InitWithShallowSlice(u, I_WEY_rpsi4);
  weyl.ipsi4.InitWithShallowSlice(u, I_WEY_ipsi4);
}
//WGC end

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

void Z4c::GetMatter(AthenaArray<Real> & u_mat, AthenaArray<Real> & u_adm, AthenaArray<Real> & w, AthenaArray<Real> &bb_cc)
{
     MeshBlock * pmb = pmy_block;
     Matter_vars mat;
     SetMatterAliases(u_mat, mat);
     Real gamma_adi = pmy_block->peos->GetGamma(); //NB specific to EOS
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
     bb1cc.InitWithShallowSlice(bb_cc,IB1,1);
     bb2cc.InitWithShallowSlice(bb_cc,IB2,1);
     bb3cc.InitWithShallowSlice(bb_cc,IB3,1);   //check this!
     } else if(opt.fix_admsource==1){
     rhocc.InitWithShallowSlice(pmb->phydro->w_init,IDN,1);
     pgascc.InitWithShallowSlice(pmb->phydro->w_init,IPR,1);
     utilde1cc.InitWithShallowSlice(pmb->phydro->w_init,IVX,1);
     utilde2cc.InitWithShallowSlice(pmb->phydro->w_init,IVY,1);
     utilde3cc.InitWithShallowSlice(pmb->phydro->w_init,IVZ,1); 
     }

if(opt.Tmunuinterp==0){
//     Real rhovc, pgasvc, utilde1vc, utilde2vc, utilde3vc, wgas,tmp, gamma_lor, v1,v2,v3,v_1,v_2,v_3,epsvc, bb1vc,bb2vc,bb3vc;
AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> rhovc, pgasvc, utilde1vc, utilde2vc, utilde3vc, epsvc, bb1vc,bb2vc,bb3vc, tmp, wgas, gamma_lor, v1,v2,v3, detgamma, detg, bsq, b0_u;
AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> v_d, bb_u, bi_u, bi_d, utildevc_u; 
rhovc.NewAthenaTensor(nn1);
pgasvc.NewAthenaTensor(nn1);
epsvc.NewAthenaTensor(nn1);
utilde1vc.NewAthenaTensor(nn1);
utilde2vc.NewAthenaTensor(nn1);
utilde3vc.NewAthenaTensor(nn1);
bb1vc.NewAthenaTensor(nn1);
bb2vc.NewAthenaTensor(nn1);
bb3vc.NewAthenaTensor(nn1);
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
bb_u.NewAthenaTensor(nn1);
bi_u.NewAthenaTensor(nn1);
bi_d.NewAthenaTensor(nn1);
utildevc_u.NewAthenaTensor(nn1);
alpha.InitWithShallowSlice(pmy_block->pz4c->storage.u,Z4c::I_Z4c_alpha,1);
// interpolate to VC
     ILOOP2(k,j){
     ILOOP1(i){
     rhovc(i) = ig->map3d_CC2VC(rhocc(k,j,i));
     if(opt.epsinterp==0){
     pgasvc(i) = ig->map3d_CC2VC(pgascc(k,j,i));
     } else {
     epsvc(i) = ig->map3d_CC2VC(epscc(k,j,i));
     }
     utilde1vc(i) = ig->map3d_CC2VC(utilde1cc(k,j,i));
     utilde2vc(i) = ig->map3d_CC2VC(utilde2cc(k,j,i));
     utilde3vc(i) = ig->map3d_CC2VC(utilde3cc(k,j,i));
     bb1vc(i)     = ig->map3d_CC2VC(bb1cc(k,j,i));
     bb2vc(i)     = ig->map3d_CC2VC(bb2cc(k,j,i));
     bb3vc(i)     = ig->map3d_CC2VC(bb3cc(k,j,i));
     
//   NB specific to EOS
     if(opt.epsinterp==1){
     pgasvc(i) = epsvc(i)*rhovc(i)*(gamma_adi-1.0);
     }
     wgas(i) = rhovc(i) + gamma_adi/(gamma_adi-1.0) * pgasvc(i);
     tmp(i) = utilde1vc(i)*utilde1vc(i)*adm.g_dd(0,0,k,j,i) + utilde2vc(i)*utilde2vc(i)*adm.g_dd(1,1,k,j,i) 
                + utilde3vc(i)*utilde3vc(i)*adm.g_dd(2,2,k,j,i) + 2.0*utilde1vc(i)*utilde2vc(i)*adm.g_dd(0,1,k,j,i)
                + 2.0*utilde1vc(i)*utilde3vc(i)*adm.g_dd(0,2,k,j,i) + 
                2.0*utilde2vc(i)*utilde3vc(i)*adm.g_dd(1,2,k,j,i);
     gamma_lor(i) = sqrt(1.0+tmp(i));
//   convert to 3-velocity
     v1(i) = utilde1vc(i)/gamma_lor(i);
     v2(i) = utilde2vc(i)/gamma_lor(i);
     v3(i) = utilde3vc(i)/gamma_lor(i);

     v_d(0,i) = v1(i)*adm.g_dd(0,0,k,j,i) + v2(i)*adm.g_dd(0,1,k,j,i) +v3(i)*adm.g_dd(0,2,k,j,i);
     v_d(1,i) = v1(i)*adm.g_dd(0,1,k,j,i) + v2(i)*adm.g_dd(1,1,k,j,i) +v3(i)*adm.g_dd(1,2,k,j,i);
     v_d(2,i) = v1(i)*adm.g_dd(0,2,k,j,i) + v2(i)*adm.g_dd(1,2,k,j,i) +v3(i)*adm.g_dd(2,2,k,j,i);

     detgamma(i) = SpatialDet(adm.g_dd(0,0,k,j,i),adm.g_dd(0,1,k,j,i), adm.g_dd(0,2,k,j,i), adm.g_dd(1,1,k,j,i), adm.g_dd(1,2,k,j,i), adm.g_dd(2,2,k,j,i));
     detg(i) = alpha(k,j,i)*detgamma(i);

     bb_u(0,i) = bb1vc(i)/detg(i);
     bb_u(1,i) = bb2vc(i)/detg(i);
     bb_u(2,i) = bb3vc(i)/detg(i);
     }
//     b0_u = 0.0;
     b0_u.ZeroClear();
     for(int a=0;a<NDIM;++a){
     ILOOP1(i){
     b0_u(i) += gamma_lor(i)*bb_u(a,i)*v_d(a,i)/alpha(k,j,i);
     }
     }

     ILOOP1(i){
     utildevc_u(0,i) = utilde1vc(i);
     utildevc_u(1,i) = utilde2vc(i);
     utildevc_u(2,i) = utilde3vc(i);
     }

     for(int a=0;a<NDIM;++a){
      ILOOP1(i){
          bi_u(a,i) = (bb_u(a,i) + alpha(k,j,i)*b0_u(i)*utildevc_u(a,i))/gamma_lor(i);
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



     ILOOP1(i){
     mat.rho(k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i)) - (pgasvc(i) + bsq(i)/2.0) - alpha(k,j,i)*alpha(k,j,i)*b0_u(i)*b0_u(i);
     mat.S_d(0,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))*v_d(0,i) - b0_u(i)*bi_d(0,i);
     mat.S_d(1,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))*v_d(1,i) - b0_u(i)*bi_d(1,i);
     mat.S_d(2,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))*v_d(2,i) - b0_u(i)*bi_d(2,i);
     mat.S_dd(0,0,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))* v_d(0,i)*v_d(0,i) + (pgasvc(i)+bsq(i)/2.0)*adm.g_dd(0,0,k,j,i) - bi_d(0,i)*bi_d(0,i);
     mat.S_dd(0,1,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))* v_d(0,i)*v_d(1,i) + (pgasvc(i)+bsq(i)/2.0)*adm.g_dd(0,1,k,j,i) - bi_d(0,i)*bi_d(1,i);
     mat.S_dd(0,2,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))* v_d(0,i)*v_d(2,i) + (pgasvc(i)+bsq(i)/2.0)*adm.g_dd(0,2,k,j,i) - bi_d(0,i)*bi_d(2,i);
     mat.S_dd(1,1,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))* v_d(1,i)*v_d(1,i) + (pgasvc(i)+bsq(i)/2.0)*adm.g_dd(1,1,k,j,i) - bi_d(1,i)*bi_d(1,i);
     mat.S_dd(1,2,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))* v_d(1,i)*v_d(2,i) + (pgasvc(i)+bsq(i)/2.0)*adm.g_dd(1,2,k,j,i) - bi_d(1,i)*bi_d(2,i);
     mat.S_dd(2,2,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))* v_d(2,i)*v_d(2,i) + (pgasvc(i)+bsq(i)/2.0)*adm.g_dd(2,2,k,j,i) - bi_d(2,i)*bi_d(2,i);
}
}
}
}

//----------------------------------------------------------------------------------------
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
