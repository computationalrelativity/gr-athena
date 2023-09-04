//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file wave.cpp
//  \brief implementation of functions in the Wave class

// C++ headers
#include <iostream>
#include <string>
// #include <algorithm>  // min()
// #include <cmath>      // fabs(), sqrt()
#include <limits>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"

#include "wave.hpp"

// constructor, initializes data structures and parameters

Wave::Wave(MeshBlock *pmb, ParameterInput *pin) :
  pmy_block(pmb),
  u(NWAVE_CPT,
    (WAVE_VC_ENABLED) ? pmb->nverts3 : pmb->ncells3,
    (WAVE_VC_ENABLED) ? pmb->nverts2 : pmb->ncells2,
    (WAVE_VC_ENABLED) ? pmb->nverts1 : pmb->ncells1),
  coarse_u_(NWAVE_CPT,
            (WAVE_VC_ENABLED) ? pmb->ncv3 : ((WAVE_CX_ENABLED) ? pmb->cx_ncc3 : pmb->ncc3),
            (WAVE_VC_ENABLED) ? pmb->ncv2 : ((WAVE_CX_ENABLED) ? pmb->cx_ncc2 : pmb->ncc2),
            (WAVE_VC_ENABLED) ? pmb->ncv1 : ((WAVE_CX_ENABLED) ? pmb->cx_ncc1 : pmb->ncc1),
            (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
             AthenaArray<Real>::DataStatus::empty)),
  empty_flux{AthenaArray<Real>(), AthenaArray<Real>(), AthenaArray<Real>()},
  ubvar_cc(pmb, &u, &coarse_u_, empty_flux),  // dirty but safe as only _one_ is registered
  ubvar_vc(pmb, &u, &coarse_u_, empty_flux),
  ubvar_cx(pmb, &u, &coarse_u_, empty_flux)
{
  Mesh *pm = pmb->pmy_mesh;
  Coordinates * pco = pmb->pcoord;

  // dimensions required for data allocation
  mbi.nn1 = (WAVE_VC_ENABLED) ? pmb->nverts1 : pmb->ncells1;
  mbi.nn2 = (WAVE_VC_ENABLED) ? pmb->nverts2 : pmb->ncells2;
  mbi.nn3 = (WAVE_VC_ENABLED) ? pmb->nverts3 : pmb->ncells3;

  int nn1 = mbi.nn1, nn2 = mbi.nn2, nn3 = mbi.nn3;

  // convenience for per-block iteration (private Wave scope)
  mbi.il = pmb->is;
  mbi.iu = (WAVE_VC_ENABLED) ? pmb->ive : pmb->ie;

  mbi.jl = pmb->js;
  mbi.ju = (WAVE_VC_ENABLED) ? pmb->jve : pmb->je;

  mbi.kl = pmb->ks;
  mbi.ku = (WAVE_VC_ENABLED) ? pmb->kve : pmb->ke;

  // point to appropriate grid
  if (WAVE_CC_ENABLED || WAVE_CX_ENABLED)
  {
    mbi.x1.InitWithShallowSlice(pco->x1v, 1, 0, nn1);
    mbi.x2.InitWithShallowSlice(pco->x2v, 1, 0, nn2);
    mbi.x3.InitWithShallowSlice(pco->x3v, 1, 0, nn3);
  }
  else if (WAVE_VC_ENABLED)
  {
    mbi.x1.InitWithShallowSlice(pco->x1f, 1, 0, nn1);
    mbi.x2.InitWithShallowSlice(pco->x2f, 1, 0, nn2);
    mbi.x3.InitWithShallowSlice(pco->x3f, 1, 0, nn3);
  }

  // inform MeshBlock that this array is the "primary" representation
  // Used for:
  // (1) load-balancing
  // (2) (future) dumping to restart file
  if (WAVE_CC_ENABLED)
    pmb->RegisterMeshBlockDataCC(u);

  if (WAVE_VC_ENABLED)
    pmb->RegisterMeshBlockDataVC(u);

  if (WAVE_CX_ENABLED)
    pmb->RegisterMeshBlockDataCX(u);

  // Allocate memory for the solution and its time derivative
  // u.NewAthenaArray(NWAVE_CPT, nn3, nn2, nn1);
  u1.NewAthenaArray(NWAVE_CPT, nn3, nn2, nn1);
  rhs.NewAthenaArray(NWAVE_CPT, nn3, nn2, nn1);

  exact.NewAthenaArray(nn3, nn2, nn1);
  error.NewAthenaArray(nn3, nn2, nn1);

  // If user-requested time integrator is type 3S* allocate additional memory
  std::string integrator = pin->GetOrAddString("time", "integrator", "vl2");
  if (integrator == "ssprk5_4")
    u2.NewAthenaArray(NWAVE_CPT, nn3, nn2, nn1);

  c = pin->GetOrAddReal("wave", "c", 1.0);

  // debug control
  debug_inspect_error = pin->GetOrAddBoolean("wave", "debug_inspect_err", false);
  debug_abort_threshold = pin->GetOrAddReal("wave", "debug_abort_threshold",
    std::numeric_limits<Real>::max());

  // Additional boundary condition control
  std::string boundary_type = pin->GetOrAddString("wave", "boundary_type", "none");

  if (boundary_type == "Dirichlet") {
    use_Dirichlet = true;

    if (pmb->pmy_mesh->ndim == 1) {

      Lx1_ = pin->GetOrAddReal("wave", "Lx1", 1.);
      M_ = pin->GetOrAddInteger("wave", "M_cutoff", 1.);

      A_.NewAthenaArray(M_);
      B_.NewAthenaArray(M_);

      for (int m=1; m<=M_; ++m) {
        std::string m_str = std::to_string(m);
        A_(m-1) = pin->GetOrAddReal("wave", "A_" + m_str, 0.);
        B_(m-1) = pin->GetOrAddReal("wave", "B_" + m_str, 0.);
      }

    } else if (pmb->pmy_mesh->ndim == 2) {

      Lx1_ = pin->GetOrAddReal("wave", "Lx1", 1.);
      Lx2_ = pin->GetOrAddReal("wave", "Lx2", 1.);

      M_ = pin->GetOrAddInteger("wave", "M_cutoff", 1.);
      N_ = pin->GetOrAddInteger("wave", "N_cutoff", 1.);

      A_.NewAthenaArray(M_, N_);
      B_.NewAthenaArray(M_, N_);

      for (int n=1; n<=N_; ++n) {
        std::string n_str = std::to_string(n);

        for (int m=1; m<=M_; ++m) {
          std::string m_str = std::to_string(m);
          A_(m-1, n-1) = pin->GetOrAddReal("wave", "A_" + m_str + n_str, 0.);
          B_(m-1, n-1) = pin->GetOrAddReal("wave", "B_" + m_str + n_str, 0.);
        }
      }

    } else {
      Lx1_ = pin->GetOrAddReal("wave", "Lx1", 1.);
      Lx2_ = pin->GetOrAddReal("wave", "Lx2", 1.);
      Lx3_ = pin->GetOrAddReal("wave", "Lx3", 1.);

      M_ = pin->GetOrAddInteger("wave", "M_cutoff", 1.);
      N_ = pin->GetOrAddInteger("wave", "N_cutoff", 1.);
      O_ = pin->GetOrAddInteger("wave", "O_cutoff", 1.);

      A_.NewAthenaArray(M_, N_, O_);
      B_.NewAthenaArray(M_, N_, O_);

      for (int o=1; o<=O_; ++o) {
        std::string o_str = std::to_string(o);
        for (int n=1; n<=N_; ++n) {
          std::string n_str = std::to_string(n);
          for (int m=1; m<=M_; ++m) {
            std::string m_str = std::to_string(m);
            A_(m-1, n-1, o-1) = pin->GetOrAddReal(
              "wave", "A_" + m_str + n_str + o_str, 0.);
            B_(m-1, n-1, o-1) = pin->GetOrAddReal(
              "wave", "B_" + m_str + n_str + o_str, 0.);
          }
        }
      }

    }

  } else if (boundary_type == "Sommerfeld")
    use_Sommerfeld = true;

  // "Enroll" in SMR/AMR by adding to vector of pointers in MeshRefinement class
  if (pm->multilevel) {
    if (WAVE_CC_ENABLED)
      refinement_idx = pmb->pmr->AddToRefinementCC(&u, &coarse_u_);

    if (WAVE_VC_ENABLED)
      refinement_idx = pmb->pmr->AddToRefinementVC(&u, &coarse_u_);

    if (WAVE_CX_ENABLED)
      refinement_idx = pmb->pmr->AddToRefinementCX(&u, &coarse_u_);

  }

  // enroll CellCenteredBoundaryVariable / VertexCenteredBoundaryVariable object
  if (WAVE_CC_ENABLED)
  {
    ubvar_cc.bvar_index = pmb->pbval->bvars.size();
    pmb->pbval->bvars.push_back(&ubvar_cc);
    pmb->pbval->bvars_main_int.push_back(&ubvar_cc);
  }
  else if (WAVE_VC_ENABLED)
  {
    ubvar_vc.bvar_index = pmb->pbval->bvars.size();
    pmb->pbval->bvars.push_back(&ubvar_vc);
    pmb->pbval->bvars_main_int_vc.push_back(&ubvar_vc);
  }
  else if (WAVE_CX_ENABLED)
  {
    ubvar_cx.bvar_index = pmb->pbval->bvars.size();
    pmb->pbval->bvars.push_back(&ubvar_cx);
    pmb->pbval->bvars_main_int_cx.push_back(&ubvar_cx);
  }

  // Allocate memory for scratch arrays
  dt1_.NewAthenaArray(nn1);
  dt2_.NewAthenaArray(nn1);
  dt3_.NewAthenaArray(nn1);

  if (WAVE_CC_ENABLED)
  {
    this->fd = pmb->pcoord->fd_cc;
  }
  else if (WAVE_CX_ENABLED)
  {
    this->fd = pmb->pcoord->fd_cx;
  }
  else
  {
    this->fd = pmb->pcoord->fd_vc;
  }
}


// destructor
Wave::~Wave()
{
  u.DeleteAthenaArray();

  dt1_.DeleteAthenaArray();
  dt2_.DeleteAthenaArray();
  dt3_.DeleteAthenaArray();
  u1.DeleteAthenaArray();
  u2.DeleteAthenaArray(); // only allocated in case of 3S*-type of integrator

  rhs.DeleteAthenaArray();

  exact.DeleteAthenaArray();
  error.DeleteAthenaArray();

  if (use_Dirichlet) {
    A_.DeleteAthenaArray();
    B_.DeleteAthenaArray();
  }

  // note: do not include x1_, x2_, x3_
}


//----------------------------------------------------------------------------------------
//! \fn  void Wave::AddWaveRHS
//  \brief Adds RHS to weighted average of variables from
//  previous step(s) of time integrator algorithm

void Wave::AddWaveRHS(const Real wght, AthenaArray<Real> &u_out) {
  MeshBlock *pmb=pmy_block;

  for (int k=mbi.kl; k<=mbi.ku; ++k) {
    for (int j=mbi.jl; j<=mbi.ju; ++j) {
      // update variables
      for (int n=0; n<NWAVE_CPT; ++n) {
#pragma omp simd
        for (int i=mbi.il; i<=mbi.iu; ++i) {
          u_out(n, k, j, i) += wght*(pmb->pmy_mesh->dt)*rhs(n, k, j, i);
        }
        }
      }
    }

  return;
}
