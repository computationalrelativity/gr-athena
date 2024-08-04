//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file scalars.cpp
//  \brief implementation of functions in class PassiveScalars

// C headers

// C++ headers
#include <algorithm>
#include <string>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../mesh/mesh.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "scalars.hpp"

// constructor, initializes data structures and parameters

PassiveScalars::PassiveScalars(MeshBlock *pmb, ParameterInput *pin)  :
    s(NSCALARS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    s1(NSCALARS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    r(NSCALARS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    s_flux{ {NSCALARS, pmb->ncells3, pmb->ncells2, pmb->ncells1+1},
            {NSCALARS, pmb->ncells3, pmb->ncells2+1, pmb->ncells1,
             (pmb->pmy_mesh->f2 ? AthenaArray<Real>::DataStatus::allocated :
              AthenaArray<Real>::DataStatus::empty)},
            {NSCALARS, pmb->ncells3+1, pmb->ncells2, pmb->ncells1,
             (pmb->pmy_mesh->f3 ? AthenaArray<Real>::DataStatus::allocated :
              AthenaArray<Real>::DataStatus::empty)}
    },
    coarse_s_(NSCALARS, pmb->ncc3, pmb->ncc2, pmb->ncc1,
              (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
               AthenaArray<Real>::DataStatus::empty)),
    coarse_r_(NSCALARS, pmb->ncc3, pmb->ncc2, pmb->ncc1,
              (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
               AthenaArray<Real>::DataStatus::empty)),
    sbvar(pmb, &s, &coarse_s_, s_flux),
    nu_scalar_iso{pin->GetOrAddReal("problem", "nu_scalar_iso", 0.0)},
    //nu_scalar_aniso{pin->GetOrAddReal("problem", "nu_scalar_aniso", 0.0)},
    scalar_diffusion_defined{(nu_scalar_iso > 0.0 ? true : false)},
    pmy_block(pmb)
{
  int nc1 = pmb->ncells1, nc2 = pmb->ncells2, nc3 = pmb->ncells3;
  Mesh *pm = pmy_block->pmy_mesh;

  pmb->RegisterMeshBlockDataCC(s);

  // If user-requested time integrator is type 3S*, allocate additional memory registers
  std::string integrator = pin->GetOrAddString("time", "integrator", "vl2");
  if (integrator == "ssprk5_4" || STS_ENABLED)
    // future extension may add "int nregister" to Hydro class
    s2.NewAthenaArray(NSCALARS, nc3, nc2, nc1);

  // "Enroll" in SMR/AMR by adding to vector of pointers in MeshRefinement class
  if (pm->multilevel) {
    refinement_idx = pmy_block->pmr->AddToRefinementCC(&s, &coarse_s_);
  }

  // enroll CellCenteredBoundaryVariable object
  sbvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&sbvar);
  pmb->pbval->bvars_main_int.push_back(&sbvar);

  // Allocate memory for scratch arrays
  rl_.NewAthenaArray(NSCALARS, nc1);
  rr_.NewAthenaArray(NSCALARS, nc1);
  rlb_.NewAthenaArray(NSCALARS, nc1);

  if (pmy_block->precon->xorder_use_fb)
  {
    r_rl_.NewAthenaArray(NSCALARS, nc1);
    r_rr_.NewAthenaArray(NSCALARS, nc1);
    r_rlb_.NewAthenaArray(NSCALARS, nc1);
  }

  dflx_.NewAthenaArray(NSCALARS, nc1);

  if (scalar_diffusion_defined)
  {
    diffusion_flx[X1DIR].NewAthenaArray(NSCALARS, nc3, nc2, nc1+1);
    diffusion_flx[X2DIR].NewAthenaArray(NSCALARS, nc3, nc2+1, nc1);
    diffusion_flx[X3DIR].NewAthenaArray(NSCALARS, nc3+1, nc2, nc1);
    //nu_scalar.NewAthenaArray(2, nc3, nc2, nc1);
    dx1_.NewAthenaArray(nc1);
    dx2_.NewAthenaArray(nc1);
    dx3_.NewAthenaArray(nc1);
    // nu_scalar_tot_.NewAthenaArray(nc1);
  }
}
