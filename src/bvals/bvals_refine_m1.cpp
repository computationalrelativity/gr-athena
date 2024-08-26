// Treatment of refinement for M1
//
// Overall idea is as in Hydro treatment, but, hopefully less awkward.

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>
#include <iterator>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
// #include "../eos/eos.hpp"
#include "../field/field.hpp"
// #include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
// #include "../scalars/scalars.hpp"
#include "bvals.hpp"
// #include "cc/hydro/bvals_hydro.hpp"
#include "fc/bvals_fc.hpp"
#include "vc/bvals_vc.hpp"
#include "cx/bvals_cx.hpp"

#include "../m1/m1.hpp"

void BoundaryValues::ProlongateBoundariesM1(const Real time, const Real dt)
{
  MeshBlock *pmb = pmy_block_;
  int &mylevel = pmb->loc.level;

  // For each finer neighbor, to prolongate a boundary we need to fill one more cell
  // surrounding the boundary zone to calculate the slopes ("ghost-ghost zone"). 3x steps:
  for (int n=0; n<nneighbor; n++) {
    NeighborBlock& nb = neighbor[n];
    if (nb.snb.level >= mylevel) continue;
    // fill the required ghost-ghost zone
    int nis, nie, njs, nje, nks, nke;
    nis = std::max(nb.ni.ox1-1, -1);
    nie = std::min(nb.ni.ox1+1, 1);
    if (pmb->block_size.nx2 == 1) {
      njs = 0;
      nje = 0;
    } else {
      njs = std::max(nb.ni.ox2-1, -1);
      nje = std::min(nb.ni.ox2+1, 1);
    }

    if (pmb->block_size.nx3 == 1) {
      nks = 0;
      nke = 0;
    } else {
      nks = std::max(nb.ni.ox3-1, -1);
      nke = std::min(nb.ni.ox3+1, 1);
    }

    // Step 1. Apply necessary variable restrictions when ghost-ghost zone is on same lvl
    for (int nk=nks; nk<=nke; nk++) {
      for (int nj=njs; nj<=nje; nj++) {
        for (int ni=nis; ni<=nie; ni++) {
          int ntype = std::abs(ni) + std::abs(nj) + std::abs(nk);
          // skip myself or coarse levels; only the same level must be restricted
          if (ntype == 0 || nblevel[nk+1][nj+1][ni+1] != mylevel) continue;

          // this neighbor block is on the same level
          // and needs to be restricted for prolongation
          RestrictGhostCellsOnSameLevelM1(nb, nk, nj, ni);
        }
      }
    }

    // calculate the loop limits for the ghost zones
    int cn = pmb->cnghost - 1;
    int si, ei, sj, ej, sk, ek;
    if (nb.ni.ox1 == 0) {
      std::int64_t &lx1 = pmb->loc.lx1;
      si = pmb->cis, ei = pmb->cie;
      if ((lx1 & 1LL) == 0LL) ei += cn;
      else             si -= cn;
    } else if (nb.ni.ox1 > 0) { si = pmb->cie + 1,  ei = pmb->cie + cn;}
    else              si = pmb->cis-cn, ei = pmb->cis-1;
    if (nb.ni.ox2 == 0) {
      sj = pmb->cjs, ej = pmb->cje;
      if (pmb->block_size.nx2 > 1) {
        std::int64_t &lx2 = pmb->loc.lx2;
        if ((lx2 & 1LL) == 0LL) ej += cn;
        else             sj -= cn;
      }
    } else if (nb.ni.ox2 > 0) { sj = pmb->cje + 1,  ej = pmb->cje + cn;}
    else              sj = pmb->cjs-cn, ej = pmb->cjs-1;
    if (nb.ni.ox3 == 0) {
      sk = pmb->cks, ek = pmb->cke;
      if (pmb->block_size.nx3 > 1) {
        std::int64_t &lx3 = pmb->loc.lx3;
        if ((lx3 & 1LL) == 0LL) ek += cn;
        else             sk -= cn;
      }
    } else if (nb.ni.ox3 > 0) { sk = pmb->cke + 1,  ek = pmb->cke + cn;}
    else              sk = pmb->cks-cn, ek = pmb->cks-1;

    // Step 2. Re-apply physical boundaries on the coarse boundary:
    pmb->pm1->enable_user_bc = true;
    ApplyPhysicalBoundariesOnCoarseLevelM1(nb, time, dt, si, ei, sj, ej, sk, ek);
    pmb->pm1->enable_user_bc = false;

    // Step 3. Finally, the ghost-ghost zones are ready for prolongation:
    ProlongateGhostCellsM1(nb, si, ei, sj, ej, sk, ek);

  } // end loop over nneighbor

  return;

}


void BoundaryValues::RestrictGhostCellsOnSameLevelM1(
  const NeighborBlock& nb, int nk,
  int nj, int ni)
{
  MeshBlock *pmb = pmy_block_;
  MeshRefinement *pmr = pmb->pmr;

  int ris, rie, rjs, rje, rks, rke;
  if (ni == 0) {
    ris = pmb->cis;
    rie = pmb->cie;
    if (nb.ni.ox1 == 1) {
      ris = pmb->cie;
    } else if (nb.ni.ox1 == -1) {
      rie = pmb->cis;
    }
  } else if (ni == 1) {
    ris = pmb->cie + 1, rie = pmb->cie + 1;
  } else { //(ni ==  - 1)
    ris = pmb->cis - 1, rie = pmb->cis - 1;
  }
  if (nj == 0) {
    rjs = pmb->cjs, rje = pmb->cje;
    if (nb.ni.ox2 == 1) rjs = pmb->cje;
    else if (nb.ni.ox2 == -1) rje = pmb->cjs;
  } else if (nj == 1) {
    rjs = pmb->cje + 1, rje = pmb->cje + 1;
  } else { //(nj == -1)
    rjs = pmb->cjs - 1, rje = pmb->cjs - 1;
  }
  if (nk == 0) {
    rks = pmb->cks, rke = pmb->cke;
    if (nb.ni.ox3 == 1) rks = pmb->cke;
    else if (nb.ni.ox3 == -1) rke = pmb->cks;
  } else if (nk == 1) {
    rks = pmb->cke + 1, rke = pmb->cke + 1;
  } else { //(nk == -1)
    rks = pmb->cks - 1, rke = pmb->cks - 1;
  }

  for (auto cc_pair : pmr->pvars_m1_cc_) {
    AthenaArray<Real> *var_cc = std::get<0>(cc_pair);
    AthenaArray<Real> *coarse_cc = std::get<1>(cc_pair);
    int nu = var_cc->GetDim4() - 1;
    pmb->pmr->RestrictCellCenteredValues(*var_cc, *coarse_cc, 0, nu,
                                         ris, rie, rjs, rje, rks, rke);
  }

  return;
}

void BoundaryValues::ApplyPhysicalBoundariesOnCoarseLevelM1(
  const NeighborBlock& nb, const Real time, const Real dt,
  int si, int ei, int sj, int ej, int sk, int ek)
{

  MeshBlock *pmb = pmy_block_;
  MeshRefinement *pmr = pmb->pmr;

  M1::M1 *pm1 = pmb->pm1;
  Field *pf = nullptr;

  if (nb.ni.ox1 == 0) {
    if (apply_bndry_fn_[BoundaryFace::inner_x1]) {
      DispatchBoundaryFunctions(pmb, pmr->pcoarsec, time, dt,
                                pmb->cis, pmb->cie, sj, ej, sk, ek, 1,
                                BoundaryFace::inner_x1,
                                bvars_m1);
    }
    if (apply_bndry_fn_[BoundaryFace::outer_x1]) {
      DispatchBoundaryFunctions(pmb, pmr->pcoarsec, time, dt,
                                pmb->cis, pmb->cie, sj, ej, sk, ek, 1,
                                BoundaryFace::outer_x1,
                                bvars_m1);
    }
  }
  if (nb.ni.ox2 == 0 && pmb->block_size.nx2 > 1) {
    if (apply_bndry_fn_[BoundaryFace::inner_x2]) {
      DispatchBoundaryFunctions(pmb, pmr->pcoarsec, time, dt,
                                si, ei, pmb->cjs, pmb->cje, sk, ek, 1,
                                BoundaryFace::inner_x2,
                                bvars_m1);
    }
    if (apply_bndry_fn_[BoundaryFace::outer_x2]) {
      DispatchBoundaryFunctions(pmb, pmr->pcoarsec, time, dt,
                                si, ei, pmb->cjs, pmb->cje, sk, ek, 1,
                                BoundaryFace::outer_x2,
                                bvars_m1);
    }
  }
  if (nb.ni.ox3 == 0 && pmb->block_size.nx3 > 1) {
    if (apply_bndry_fn_[BoundaryFace::inner_x3]) {
      DispatchBoundaryFunctions(pmb, pmr->pcoarsec, time, dt,
                                si, ei, sj, ej, pmb->cks, pmb->cke, 1,
                                BoundaryFace::inner_x3,
                                bvars_m1);
    }
    if (apply_bndry_fn_[BoundaryFace::outer_x3]) {
      DispatchBoundaryFunctions(pmb, pmr->pcoarsec, time, dt,
                                si, ei, sj, ej, pmb->cks, pmb->cke, 1,
                                BoundaryFace::outer_x3,
                                bvars_m1);
    }
  }
}

void BoundaryValues::ProlongateGhostCellsM1(
  const NeighborBlock& nb,
  int si, int ei, int sj, int ej,
  int sk, int ek)
{

  MeshBlock *pmb = pmy_block_;
  MeshRefinement *pmr = pmb->pmr;

  // prolongate cell-centered S/AMR-enrolled quantities

  for (auto cc_pair : pmr->pvars_m1_cc_) {
    AthenaArray<Real> *var_cc = std::get<0>(cc_pair);
    AthenaArray<Real> *coarse_cc = std::get<1>(cc_pair);
    int nu = var_cc->GetDim4() - 1;

    pmr->ProlongateCellCenteredValues(*coarse_cc, *var_cc, 0, nu,
                                      si, ei, sj, ej, sk, ek);
  }
}

//
// :D
//