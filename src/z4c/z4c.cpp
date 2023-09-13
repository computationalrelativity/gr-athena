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
#include "z4c_amr.hpp"
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../outputs/outputs.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../utils/linear_algebra.hpp"
#include "../utils/interp_intergrid.hpp" //SB FIXME imported from matter_tracker_extrema

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
  pz4c(this),
  pmy_mesh(pmb->pmy_mesh),
  pmy_block(pmb),
  pmy_coord(pmy_block->pcoord),
  mbi{
    1, pmy_mesh->f2, pmy_mesh->f3,                         // f1, f2, f3
    SW_CC_CX_VC(pmb->is, pmb->cx_is, pmb->ivs),            // il
    SW_CC_CX_VC(pmb->ie, pmb->cx_ie, pmb->ive),            // iu
    SW_CC_CX_VC(pmb->js, pmb->cx_js, pmb->jvs),            // jl
    SW_CC_CX_VC(pmb->je, pmb->cx_je, pmb->jve),            // ju
    SW_CC_CX_VC(pmb->ks, pmb->cx_ks, pmb->kvs),            // kl
    SW_CC_CX_VC(pmb->ke, pmb->cx_ke, pmb->kve),            // ku
    SW_CCX_VC(pmb->ncells1, pmb->nverts1),                 // nn1
    SW_CCX_VC(pmb->ncells2, pmb->nverts2),                 // nn2
    SW_CCX_VC(pmb->ncells3, pmb->nverts3),                 // nn3
    SW_CC_CX_VC(pmb->ncc1, pmb->cx_ncc1, pmb->ncv1),       // cnn1
    SW_CC_CX_VC(pmb->ncc2, pmb->cx_ncc2, pmb->ncv2),       // cnn2
    SW_CC_CX_VC(pmb->ncc3, pmb->cx_ncc3, pmb->ncv3),       // cnn3
    NGHOST,                                                // ng
    SW_CC_CX_VC(NCGHOST, NCGHOST_CX, NCGHOST),             // cng
    (pmy_mesh->f3) ? 3 : (pmy_mesh->f2) ? 2 : 1            // ndim
  },
  coarse_u_(N_Z4c, mbi.cnn3, mbi.cnn2, mbi.cnn1,
            (pmb->pmy_mesh->multilevel ?
             AthenaArray<Real>::DataStatus::allocated :
             AthenaArray<Real>::DataStatus::empty)),
  storage{{N_Z4c, mbi.nn3, mbi.nn2, mbi.nn1},              // u
          {N_Z4c, mbi.nn3, mbi.nn2, mbi.nn1},              // u1
          {},                                              // u2
          {N_Z4c, mbi.nn3, mbi.nn2, mbi.nn1},              // rhs
          {N_ADM, mbi.nn3, mbi.nn2, mbi.nn1},              // adm
          {N_CON, mbi.nn3, mbi.nn2, mbi.nn1},              // con
          {N_MAT, mbi.nn3, mbi.nn2, mbi.nn1},              // mat
          {N_WEY, mbi.nn3, mbi.nn2, mbi.nn1},              // weyl
	        {N_Z4c, mbi.nn3, mbi.nn2, mbi.nn1},              // init buffers
          {N_ADM, mbi.nn3, mbi.nn2, mbi.nn1},
  },
  empty_flux{AthenaArray<Real>(), AthenaArray<Real>(), AthenaArray<Real>()},
  ubvar(pmb, &storage.u, &coarse_u_, empty_flux),
  coarse_a_(N_WEY, mbi.cnn3, mbi.cnn2, mbi.cnn1,
            (pmb->pmy_mesh->multilevel ?
             AthenaArray<Real>::DataStatus::allocated :
             AthenaArray<Real>::DataStatus::empty)),
  abvar(pmb, &storage.weyl, &coarse_a_, empty_flux)
#if defined(Z4C_CX_ENABLED)
  ,rbvar(pmb, &storage.u, &coarse_u_, empty_flux)
#endif
{
  Mesh *pm = pmy_block->pmy_mesh;
  Coordinates * pco = pmb->pcoord;

  //---------------------------------------------------------------------------
  // set up sampling
  //---------------------------------------------------------------------------
  mbi.x1.InitWithShallowSlice(SW_CCX_VC(pco->x1v, pco->x1f), 1, 0, mbi.nn1);
  mbi.x2.InitWithShallowSlice(SW_CCX_VC(pco->x2v, pco->x2f), 1, 0, mbi.nn2);
  mbi.x3.InitWithShallowSlice(SW_CCX_VC(pco->x3v, pco->x3f), 1, 0, mbi.nn3);

  // sizes are the same in either case
  mbi.dx1.InitWithShallowSlice(SW_CCX_VC(pco->dx1v, pco->dx1f), 1, 0, mbi.nn1);
  mbi.dx2.InitWithShallowSlice(SW_CCX_VC(pco->dx2v, pco->dx2f), 1, 0, mbi.nn2);
  mbi.dx3.InitWithShallowSlice(SW_CCX_VC(pco->dx3v, pco->dx3f), 1, 0, mbi.nn3);
  //---------------------------------------------------------------------------

  // now init. amr (requires sampling)
  pz4c_amr = new Z4c_AMR(pz4c,pmb,pin);

  // inform MeshBlock that this array is the "primary" representation
  // Used for:
  // (1) load-balancing
  // (2) (future) dumping to restart file
  FCN_CC_CX_VC(
    pmb->RegisterMeshBlockDataCC(storage.u),
    pmb->RegisterMeshBlockDataCX(storage.u),
    pmb->RegisterMeshBlockDataVC(storage.u)
  );

  // "Enroll" in SMR/AMR by adding to vector of pointers in MeshRefinement class
  if (pm->multilevel) {
    refinement_idx = FCN_CC_CX_VC(
      pmy_block->pmr->AddToRefinementCC,
      pmy_block->pmr->AddToRefinementCX,
      pmy_block->pmr->AddToRefinementVC
    )(&storage.u, &coarse_u_);

  }

  // register auxiliaries
  FCN_CC_CX_VC(
    pmb->RegisterMeshBlockDataCC(storage.weyl),
    pmb->RegisterMeshBlockDataCX(storage.weyl),
    pmb->RegisterMeshBlockDataVC(storage.weyl)
  );

  // if (pm->multilevel) {
  //   refinement_idx = FCN_CC_CX_VC(
  //     pmy_block->pmr->AddToRefinementCC,
  //     pmy_block->pmr->AddToRefinementCX,
  //     pmy_block->pmr->AddToRefinementVC
  //   )(&storage.weyl, &coarse_a_);
  // }

#if defined(DBG_REDUCE_AUX_COMM)
  if (pm->multilevel) {
    refinement_idx = FCN_CC_CX_VC(
      pmy_block->pmr->AddToRefinementAuxCC,
      pmy_block->pmr->AddToRefinementAuxCX,
      pmy_block->pmr->AddToRefinementAuxVC
    )(&storage.weyl, &coarse_a_);
  }
#else
  if (pm->multilevel) {
    refinement_idx = FCN_CC_CX_VC(
      pmy_block->pmr->AddToRefinementCC,
      pmy_block->pmr->AddToRefinementCX,
      pmy_block->pmr->AddToRefinementVC
    )(&storage.weyl, &coarse_a_);
  }
#endif // DBG_REDUCE_AUX_COMM

  // If user-requested time integrator is type 3S* allocate additional memory
  std::string integrator = pin->GetOrAddString("time", "integrator", "vl2");
  if (integrator == "ssprk5_4")
    storage.u2.NewAthenaArray(N_Z4c, mbi.nn3, mbi.nn2, mbi.nn1);

  // enroll BoundaryVariable object
  ubvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&ubvar);

  FCN_CC_CX_VC(
    pmb->pbval->bvars_main_int.push_back(&ubvar),
    pmb->pbval->bvars_main_int_cx.push_back(&ubvar),
    pmb->pbval->bvars_main_int_vc.push_back(&ubvar)
  );

#if defined(Z4C_CX_ENABLED)
  rbvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&rbvar);
  pmb->pbval->bvars_rbc.push_back(&rbvar);
#endif

  abvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&abvar);
  pmb->pbval->bvars_aux.push_back(&abvar);

  dt1_.NewAthenaArray(mbi.nn1);
  dt2_.NewAthenaArray(mbi.nn2);
  dt3_.NewAthenaArray(mbi.nn3);

  // BD: TODO shift defaults to header [C++11 so can use def. declaration]..
  // Parameters
  opt.chi_psi_power = pin->GetOrAddReal("z4c", "chi_psi_power", -4.0);
  opt.chi_div_floor = pin->GetOrAddReal("z4c", "chi_div_floor", -1000.0);

  const Real diss_val = pin->GetOrAddReal("z4c", "diss", 0.0);
  opt.diss = diss_val*pow(2, -2*NGHOST)*(NGHOST % 2 == 0 ? -1 : 1);

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

  // Matter parameters
  opt.cowling = pin->GetOrAddInteger("z4c", "cowling_true", 0);
  opt.bssn = pin->GetOrAddInteger("z4c", "bssn", 0);
  opt.rhstheta0 = pin->GetOrAddInteger("z4c", "rhstheta0", 0);
  opt.fixedgauge = pin->GetOrAddInteger("z4c", "fixedgauge", 0);
  opt.fix_admsource = pin->GetOrAddInteger("z4c", "fix_admsource", 0);
  opt.Tmunuinterp = pin->GetOrAddInteger("z4c", "Tmunuinterp", 0); // interpolate components of Tmunu if 1 (if 0 interpolate primitives)
  opt.epsinterp = pin->GetOrAddInteger("z4c", "epsinterp", 0); // interpolate internal energy eps instead of pressure p.

  if (opt.epsinterp == 1)
  {
    std::cout << "<z4c> epsinterp = 1 not supported" << std::endl;
  }

  // Problem-specific parameters
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

#ifdef TWO_PUNCTURES
  // Two punctures parameters
  opt.impose_bitant_id = pin->GetOrAddBoolean(
      "problem", "impose_bitant_id", false);
#endif


  //---------------------------------------------------------------------------
  // Set aliases
  SetADMAliases(storage.adm, adm);
  SetConstraintAliases(storage.con, con);
  SetMatterAliases(storage.mat, mat);
  SetZ4cAliases(storage.rhs, rhs);
  SetZ4cAliases(storage.u, z4c);
  SetWeylAliases(storage.weyl, weyl);
  // Allocate memory for aux 1D vars
  r.NewAthenaTensor(mbi.nn1);
  detg.NewAthenaTensor(mbi.nn1);
  chi_guarded.NewAthenaTensor(mbi.nn1);
  oopsi4.NewAthenaTensor(mbi.nn1);
  A.NewAthenaTensor(mbi.nn1);
  AA.NewAthenaTensor(mbi.nn1);
  R.NewAthenaTensor(mbi.nn1);
  Ht.NewAthenaTensor(mbi.nn1);
  K.NewAthenaTensor(mbi.nn1);
  KK.NewAthenaTensor(mbi.nn1);
  Ddalpha.NewAthenaTensor(mbi.nn1);
  S.NewAthenaTensor(mbi.nn1);
  M_u.NewAthenaTensor(mbi.nn1);
  Gamma_u.NewAthenaTensor(mbi.nn1);
  DA_u.NewAthenaTensor(mbi.nn1);
  s_u.NewAthenaTensor(mbi.nn1);
  g_uu.NewAthenaTensor(mbi.nn1);
  A_uu.NewAthenaTensor(mbi.nn1);
  AA_dd.NewAthenaTensor(mbi.nn1);
  R_dd.NewAthenaTensor(mbi.nn1);
  Rphi_dd.NewAthenaTensor(mbi.nn1);
  Kt_dd.NewAthenaTensor(mbi.nn1);
  K_ud.NewAthenaTensor(mbi.nn1);
  Ddalpha_dd.NewAthenaTensor(mbi.nn1);
  Ddphi_dd.NewAthenaTensor(mbi.nn1);
  Gamma_ddd.NewAthenaTensor(mbi.nn1);
  Gamma_udd.NewAthenaTensor(mbi.nn1);
  DK_ddd.NewAthenaTensor(mbi.nn1);
  DK_udd.NewAthenaTensor(mbi.nn1);
  // testcons //SB not used
  // tGamma_ddd.NewAthenaTensor(mbi.nn1);
  // tGamma_udd.NewAthenaTensor(mbi.nn1);
  // tdg_ddd.NewAthenaTensor(mbi.nn1);
  // tdetg.NewAthenaTensor(mbi.nn1);
  // tg_uu.NewAthenaTensor(mbi.nn1);
  // tGamma_u.NewAthenaTensor(mbi.nn1);

#if defined(Z4C_ETA_CONF) || defined(Z4C_ETA_TRACK_TP)
  eta_damp.NewAthenaTensor(mbi.nn1);
#endif // Z4C_ETA_CONF, Z4C_ETA_TRACK_TP

  dbeta.NewAthenaTensor(mbi.nn1);
  dalpha_d.NewAthenaTensor(mbi.nn1);
  ddbeta_d.NewAthenaTensor(mbi.nn1);
  dchi_d.NewAthenaTensor(mbi.nn1);
  dphi_d.NewAthenaTensor(mbi.nn1);
  dK_d.NewAthenaTensor(mbi.nn1);
  dKhat_d.NewAthenaTensor(mbi.nn1);
  dTheta_d.NewAthenaTensor(mbi.nn1);
  ddalpha_dd.NewAthenaTensor(mbi.nn1);
  dbeta_du.NewAthenaTensor(mbi.nn1);
  ddchi_dd.NewAthenaTensor(mbi.nn1);
  dGam_du.NewAthenaTensor(mbi.nn1);
  dg_ddd.NewAthenaTensor(mbi.nn1);
  dK_ddd.NewAthenaTensor(mbi.nn1);
  dA_ddd.NewAthenaTensor(mbi.nn1);
  ddbeta_ddu.NewAthenaTensor(mbi.nn1);
  ddg_dddd.NewAthenaTensor(mbi.nn1);

  Lchi.NewAthenaTensor(mbi.nn1);
  LKhat.NewAthenaTensor(mbi.nn1);
  LTheta.NewAthenaTensor(mbi.nn1);
  Lalpha.NewAthenaTensor(mbi.nn1);
  LGam_u.NewAthenaTensor(mbi.nn1);
  Lbeta_u.NewAthenaTensor(mbi.nn1);
  Lg_dd.NewAthenaTensor(mbi.nn1);
  LA_dd.NewAthenaTensor(mbi.nn1);

  uvec.NewAthenaTensor(mbi.nn1);
  vvec.NewAthenaTensor(mbi.nn1);
  wvec.NewAthenaTensor(mbi.nn1);
  dotp1.NewAthenaTensor(mbi.nn1);
  dotp2.NewAthenaTensor(mbi.nn1);
  Riem3_dddd.NewAthenaTensor(mbi.nn1);
  Riemm4_dddd.NewAthenaTensor(mbi.nn1);
  Riemm4_ddd.NewAthenaTensor(mbi.nn1);
  Riemm4_dd.NewAthenaTensor(mbi.nn1);

  // Intergrid communication
  //SB TODO make this optional for matter?
  //SB FIXME pcoarsec in mesh/mesh_refinement.hpp should be private
  int N[] = {pmb->block_size.nx1, pmb->block_size.nx2, pmb->block_size.nx3};
  Real rdx[] = {
    1./(SW_CCX_VC(pco->dx1v(0), pco->dx1f(0))),
    1./(SW_CCX_VC(pco->dx2v(0), pco->dx2f(0))),
    1./(SW_CCX_VC(pco->dx3v(0), pco->dx3f(0)))
  };
  ig = new InterpIntergridLocal(NDIM, &N[0], &rdx[0]);
  if(pmb->pmy_mesh->multilevel){
    int N_coarse[] = {pmb->block_size.nx1/2, pmb->block_size.nx2/2, pmb->block_size.nx3/2};
    Real rdx_coarse[] = {
      1./(SW_CCX_VC(pmb->pmr->pcoarsec->dx1v(0), pmb->pmr->pcoarsec->dx1f(0))),
      1./(SW_CCX_VC(pmb->pmr->pcoarsec->dx2v(0), pmb->pmr->pcoarsec->dx2f(0))),
      1./(SW_CCX_VC(pmb->pmr->pcoarsec->dx3v(0), pmb->pmr->pcoarsec->dx3f(0)))
    };
    ig_coarse = new InterpIntergridLocal(NDIM, &N_coarse[0], &rdx_coarse[0]);
  }


  // To handle inter-grid interpolation ---------------------------------------
  if (Z4C_ENABLED && FLUID_ENABLED)
  {
    rho.NewAthenaTensor(mbi.nn1);
    pgas.NewAthenaTensor(mbi.nn1);
    utilde.NewAthenaTensor(mbi.nn1);

#if MAGNETIC_FIELDS_ENABLED
    bb.NewAthenaTensor(mbi.nn1);
#endif
  }

  // Finite differencing alias ------------------------------------------------
#if defined (Z4C_CC_ENABLED)
  this->fd = pmb->pcoord->fd_cc;
#elif defined (Z4C_CX_ENABLED)
  this->fd = pmb->pcoord->fd_cx;
#else  // Z4C_VC_ENABLED
  this->fd = pmb->pcoord->fd_vc;
#endif

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

  //SB TODO make this optional for matter?
  delete ig;
  if(pmy_block->pmy_mesh->multilevel){
    delete ig_coarse;
  }

  if (opt.sphere_zone_number > 0) {
    opt.sphere_zone_levels.DeleteAthenaArray();
    opt.sphere_zone_radii.DeleteAthenaArray();
    opt.sphere_zone_puncture.DeleteAthenaArray();
    opt.sphere_zone_center1.DeleteAthenaArray();
    opt.sphere_zone_center2.DeleteAthenaArray();
    opt.sphere_zone_center3.DeleteAthenaArray();
  }

  // To handle inter-grid interpolation ---------------------------------------
  if (Z4C_ENABLED && FLUID_ENABLED)
  {
    rho.DeleteAthenaTensor();
    pgas.DeleteAthenaTensor();
    utilde.DeleteAthenaTensor();

#if MAGNETIC_FIELDS_ENABLED
    bb.DeleteAthenaTensor();
#endif

  }

  delete pz4c_amr;
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
// \!fn void Z4c::AlgConstr(AthenaArray<Real> & u)
// \brief algebraic constraints projection
//
// This function operates on all grid points of the MeshBlock

void Z4c::AlgConstr(AthenaArray<Real> & u)
{
  using namespace LinearAlgebra;

  Z4c_vars z4c;
  SetZ4cAliases(u, z4c);

  GLOOP2(k,j) {

    // compute determinant and "conformal conformal factor"
    GLOOP1(i) {
      detg(i) = Det3Metric(z4c.g_dd,k,j,i);
      detg(i) = detg(i) > 0. ? detg(i) : 1.;
      Real eps = detg(i) - 1.;
      // oopsi4(i) = (eps < opt.eps_floor) ? (1. - opt.eps_floor/3.) : (pow(1./detg(i), 1./3.));
      oopsi4(i) = std::cbrt(1./detg(i));
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
      A(i) = TraceRank2(1.0,
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