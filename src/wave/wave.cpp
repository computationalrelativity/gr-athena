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

#include "wave.hpp"

#if WAVE_CC_ENABLED
  #define WAVE_SW_CC_CX_VC(a, b, c)                                                 \
    a
#elif WAVE_CX_ENABLED
  #define WAVE_SW_CC_CX_VC(a, b, c)                                                 \
    b
#else
  #define WAVE_SW_CC_CX_VC(a, b, c)                                                 \
    c
#endif

#if WAVE_CC_ENABLED || WAVE_CX_ENABLED
  #define WAVE_SW_CCX_VC(a, b)                                                      \
    a
#else
  #define WAVE_SW_CCX_VC(a, b)                                                      \
    b
#endif

// constructor, initializes data structures and parameters

Wave::Wave(MeshBlock *pmb, ParameterInput *pin) :
  pmy_mesh(pmb->pmy_mesh),
  pmy_block(pmb),
  mbi{
    1, pmy_mesh->f2, pmy_mesh->f3,                         // f1, f2, f3
    WAVE_SW_CC_CX_VC(pmb->is, pmb->cx_is, pmb->ivs),       // il
    WAVE_SW_CC_CX_VC(pmb->ie, pmb->cx_ie, pmb->ive),       // iu
    WAVE_SW_CC_CX_VC(pmb->js, pmb->cx_js, pmb->jvs),       // jl
    WAVE_SW_CC_CX_VC(pmb->je, pmb->cx_je, pmb->jve),       // ju
    WAVE_SW_CC_CX_VC(pmb->ks, pmb->cx_ks, pmb->kvs),       // kl
    WAVE_SW_CC_CX_VC(pmb->ke, pmb->cx_ke, pmb->kve),       // ku
    WAVE_SW_CCX_VC(pmb->ncells1, pmb->nverts1),            // nn1
    WAVE_SW_CCX_VC(pmb->ncells2, pmb->nverts2),            // nn2
    WAVE_SW_CCX_VC(pmb->ncells3, pmb->nverts3),            // nn3
    WAVE_SW_CC_CX_VC(pmb->ncc1, pmb->cx_ncc1, pmb->ncv1),  // cnn1
    WAVE_SW_CC_CX_VC(pmb->ncc2, pmb->cx_ncc2, pmb->ncv2),  // cnn2
    WAVE_SW_CC_CX_VC(pmb->ncc3, pmb->cx_ncc3, pmb->ncv3),  // cnn3
    NGHOST,                                                // ng
    WAVE_SW_CC_CX_VC(NCGHOST, NCGHOST_CX, NCGHOST),        // cng
    (pmy_mesh->f3) ? 3 : (pmy_mesh->f2) ? 2 : 1            // ndim
  },
  u(NWAVE_CPT, mbi.nn3, mbi.nn2, mbi.nn1),
  ref_tra(mbi.nn3, mbi.nn2, mbi.nn1),  // tracker debug
  coarse_u_(NWAVE_CPT, mbi.cnn3, mbi.cnn2, mbi.cnn1,
            (pmb->pmy_mesh->multilevel ?
             AthenaArray<Real>::DataStatus::allocated :
             AthenaArray<Real>::DataStatus::empty)),
  empty_flux{AthenaArray<Real>(),
             AthenaArray<Real>(),
             AthenaArray<Real>()},
  ubvar_cc(pmb, &u, &coarse_u_, empty_flux),  // dirty but safe as only _one_ is registered
  ubvar_vc(pmb, &u, &coarse_u_, empty_flux),
  ubvar_cx(pmb, &u, &coarse_u_, empty_flux)
{
  Mesh *pm = pmb->pmy_mesh;
  Coordinates * pco = pmb->pcoord;

  //---------------------------------------------------------------------------
  // set up sampling
  //---------------------------------------------------------------------------
  mbi.x1.InitWithShallowSlice(WAVE_SW_CCX_VC(pco->x1v, pco->x1f),
                              1, 0, mbi.nn1);
  mbi.x2.InitWithShallowSlice(WAVE_SW_CCX_VC(pco->x2v, pco->x2f),
                              1, 0, mbi.nn2);
  mbi.x3.InitWithShallowSlice(WAVE_SW_CCX_VC(pco->x3v, pco->x3f),
                              1, 0, mbi.nn3);

  // sizes are the same in either case
  mbi.dx1.InitWithShallowSlice(WAVE_SW_CCX_VC(pco->dx1v, pco->dx1f),
                               1, 0, mbi.nn1);
  mbi.dx2.InitWithShallowSlice(WAVE_SW_CCX_VC(pco->dx2v, pco->dx2f),
                               1, 0, mbi.nn2);
  mbi.dx3.InitWithShallowSlice(WAVE_SW_CCX_VC(pco->dx3v, pco->dx3f),
                               1, 0, mbi.nn3);
  //---------------------------------------------------------------------------

  // dimensions required for data allocation
  int nn1 = mbi.nn1, nn2 = mbi.nn2, nn3 = mbi.nn3;

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
  std::string integrator = pin->GetOrAddString("time", "integrator", "vl2");
  if (integrator.find("bt_") != std::string::npos)
  {
    // Butcher-Tableaux style...
    // .. deal with coefficients and allocation of stage scratches in task-list
  }
  else
  {
    // Allocate memory for the solution and its time derivative
    rhs.NewAthenaArray(NWAVE_CPT, nn3, nn2, nn1);

    // Low storage style...
    u1.NewAthenaArray(NWAVE_CPT, nn3, nn2, nn1);

    // If user-requested time integrator is type 3S* allocate additional memory
    if (integrator == "ssprk5_4")
      u2.NewAthenaArray(NWAVE_CPT, nn3, nn2, nn1);
  }

  // If user-requested time integrator is type 3S* allocate additional memory
  if (integrator == "ssprk5_4")
    u2.NewAthenaArray(NWAVE_CPT, nn3, nn2, nn1);

  // Error monitoring
  exact.NewAthenaArray(nn3, nn2, nn1);
  error.NewAthenaArray(nn3, nn2, nn1);

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
