//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file refinement_ops.cpp
//  \brief Prolongation and restriction dispatch.
//
//  Routes prolongation/restriction calls to MeshRefinement operators based on
//  ProlongOp/RestrictOp enums stored in CommSpec.  Replaces the virtual dispatch
//  through BoundaryVariable subclasses.

#include "refinement_ops.hpp"

#include <sstream>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/mesh_refinement.hpp"
#include "comm_spec.hpp"
#include "neighbor_connectivity.hpp"

namespace comm {

//----------------------------------------------------------------------------------------
// Compute the 3D coarse-grid prolongation index range for one neighbor.
//
// The index range identifies the coarse-level cells whose prolongation will fill the
// fine-level ghost zones adjacent to this neighbor.  The per-dimension helper
// (ProlongationRange, inlined from the header) is identical across CC/CX/VC - only
// the base coarse indices and pcng width differ per sampling type.

idx::IndexRange3D ProlongationIndices(const MeshBlock *pmb,
                                      const NeighborBlock &nb,
                                      Sampling sampling) {
  idx::IndexRange3D r;

  // Per-sampling coarse interior bounds and prolongation ghost width.
  int cvs1, cve1, cvs2, cve2, cvs3, cve3;
  int pcng;

  switch (sampling) {
    case Sampling::CC:
      cvs1 = pmb->cis;  cve1 = pmb->cie;
      cvs2 = pmb->cjs;  cve2 = pmb->cje;
      cvs3 = pmb->cks;  cve3 = pmb->cke;
      pcng = pmb->cnghost - 1;
      break;
    case Sampling::CX:
      cvs1 = pmb->cx_cis;  cve1 = pmb->cx_cie;
      cvs2 = pmb->cx_cjs;  cve2 = pmb->cx_cje;
      cvs3 = pmb->cx_cks;  cve3 = pmb->cx_cke;
      // CX/VC use half the fine ghost width, rounded up.
      pcng = pmb->ng / 2 + (pmb->ng % 2 != 0);
      break;
    case Sampling::VC:
      cvs1 = pmb->civs;  cve1 = pmb->cive;
      cvs2 = pmb->cjvs;  cve2 = pmb->cjve;
      cvs3 = pmb->ckvs;  cve3 = pmb->ckve;
      pcng = pmb->ng / 2 + (pmb->ng % 2 != 0);
      break;
    case Sampling::FC:
      // FC prolongation uses CC coarse indices; the staggered offset is handled
      // internally by MeshRefinement::ProlongateSharedField*.
      cvs1 = pmb->cis;  cve1 = pmb->cie;
      cvs2 = pmb->cjs;  cve2 = pmb->cje;
      cvs3 = pmb->cks;  cve3 = pmb->cke;
      pcng = pmb->cnghost - 1;
      break;
  }

  // Logical coordinates for the even/odd child test in ox==0 case.
  std::int64_t lx1 = pmb->loc.lx1;
  std::int64_t lx2 = pmb->loc.lx2;
  std::int64_t lx3 = pmb->loc.lx3;

  ProlongationRange(lx1, nb.ni.ox1, pcng, cvs1, cve1, r.si, r.ei, true);
  ProlongationRange(lx2, nb.ni.ox2, pcng, cvs2, cve2, r.sj, r.ej,
                    pmb->block_size.nx2 > 1);
  ProlongationRange(lx3, nb.ni.ox3, pcng, cvs3, cve3, r.sk, r.ek,
                    pmb->block_size.nx3 > 1);

  return r;
}

//----------------------------------------------------------------------------------------
// Prolongate one neighbor's ghost zone slab.
// Dispatches to the correct MeshRefinement operator based on ProlongOp.

void ProlongateNeighbor(MeshBlock *pmb,
                        const CommSpec &spec,
                        const idx::IndexRange3D &r) {
  MeshRefinement *pmr = pmb->pmr;
  AthenaArray<Real> &coarse = *spec.coarse_var;
  AthenaArray<Real> &fine = *spec.var;
  const int nu = spec.nvar - 1;

  switch (spec.prolong_op) {
    case ProlongOp::MinmodLinear:
      // CC: minmod-limited piecewise linear interpolation.
      pmr->ProlongateCellCenteredValues(coarse, fine, 0, nu,
                                        r.si, r.ei, r.sj, r.ej, r.sk, r.ek);
      break;
    case ProlongOp::LagrangeUniform:
      // VC: symmetric Lagrange on uniform grid (inject coincident + interpolate).
      pmr->ProlongateVertexCenteredValues(coarse, fine, 0, nu,
                                          r.si, r.ei, r.sj, r.ej, r.sk, r.ek);
      break;
    case ProlongOp::LagrangeChildren:
      // CX: Lagrange children interpolation (interior order).
      pmr->ProlongateCellCenteredXValues(coarse, fine, 0, nu,
                                         r.si, r.ei, r.sj, r.ej, r.sk, r.ek);
      break;
    case ProlongOp::LagrangeChildrenBC:
      // CX: Lagrange children (boundary-compatible, lower order near edges).
      pmr->ProlongateCellCenteredXBCValues(coarse, fine, 0, nu,
                                           r.si, r.ei, r.sj, r.ej, r.sk, r.ek);
      break;
    case ProlongOp::FaceSharedMinmod:
    case ProlongOp::FaceDivPreserving:
      // FC prolongation uses ProlongateNeighborFC, not this function.
      // Reaching here indicates a caller error.
      {
        std::stringstream msg;
        msg << "### FATAL ERROR in ProlongateNeighbor\n"
            << "FC prolongation must use ProlongateNeighborFC()." << std::endl;
        ATHENA_ERROR(msg);
      }
      break;
    case ProlongOp::None:
      // No prolongation requested (uniform-only variable).
      break;
  }
}

//----------------------------------------------------------------------------------------
// Prolongate all coarser neighbors for one channel.
// Replaces the per-subclass ProlongateBoundaries() virtual method.
//
// The loop structure is identical to the old system: iterate over all neighbors,
// skip same-level or finer, compute coarse-grid index range, call operator.

void ProlongateBoundaries(MeshBlock *pmb,
                          const CommSpec &spec,
                          const NeighborConnectivity &nc) {
  if (spec.prolong_op == ProlongOp::None) return;

  // FC needs coarse_fc; CC/VC/CX need coarse_var.
  if (spec.sampling == Sampling::FC) {
    if (spec.coarse_fc == nullptr) return;
  } else {
    if (spec.coarse_var == nullptr) return;
  }

  const int mylevel = pmb->loc.level;
  const bool is_fc = (spec.sampling == Sampling::FC);

  for (int n = 0; n < nc.num_neighbors(); ++n) {
    const NeighborBlock &nb = nc.neighbor(n);
    if (nb.snb.level >= mylevel) continue;  // only prolongate from coarser neighbors

    idx::IndexRange3D r = ProlongationIndices(pmb, nb, spec.sampling);

    if (is_fc) {
      // FC: compute shared-face extended range, then 4-step Toth & Roe.
      idx::IndexRange3D shared = ProlongationSharedIndices(pmb, nb, r);
      ProlongateNeighborFC(pmb, spec, r, shared);
    } else {
      ProlongateNeighbor(pmb, spec, r);
    }
  }
}

//----------------------------------------------------------------------------------------
// Restrict the fine interior into the coarse buffer.
// Called before PackAndSend so the coarse payload is available for same-level neighbors
// and the to-coarser message.

void RestrictInterior(MeshBlock *pmb, const CommSpec &spec) {
  if (spec.restrict_op == RestrictOp::None) return;

  // FC uses var_fc/coarse_fc; all others use var/coarse_var.
  if (spec.restrict_op == RestrictOp::AreaWeightedFace) {
    if (spec.var_fc == nullptr || spec.coarse_fc == nullptr) return;
  } else {
    if (spec.coarse_var == nullptr) return;
  }

  MeshRefinement *pmr = pmb->pmr;

  switch (spec.restrict_op) {
    case RestrictOp::VolumeWeighted: {
      // CC: volume-weighted average of 2^d fine cells.
      AthenaArray<Real> &fine = *spec.var;
      AthenaArray<Real> &coarse = *spec.coarse_var;
      const int nu = spec.nvar - 1;
      pmr->RestrictCellCenteredValues(fine, coarse, 0, nu,
                                      pmb->cis, pmb->cie,
                                      pmb->cjs, pmb->cje,
                                      pmb->cks, pmb->cke);
      break;
    }
    case RestrictOp::Injection: {
      // VC: direct injection of coincident vertex values.
      AthenaArray<Real> &fine = *spec.var;
      AthenaArray<Real> &coarse = *spec.coarse_var;
      const int nu = spec.nvar - 1;
      pmr->RestrictVertexCenteredValues(fine, coarse, 0, nu,
                                        pmb->civs, pmb->cive,
                                        pmb->cjvs, pmb->cjve,
                                        pmb->ckvs, pmb->ckve);
      break;
    }
    case RestrictOp::LagrangeUniform: {
      // CX: symmetric Lagrange on uniform grid.
      AthenaArray<Real> &fine = *spec.var;
      AthenaArray<Real> &coarse = *spec.coarse_var;
      const int nu = spec.nvar - 1;
      pmr->RestrictCellCenteredXWithInteriorValues(fine, coarse, 0, nu);
      break;
    }
    case RestrictOp::Barycentric: {
      // CX: Floater-Hormann barycentric rational restriction.
      // Currently shares the same entry point as LagrangeUniform; the operator
      // selection is made inside MeshRefinement based on compile-time config.
      AthenaArray<Real> &fine = *spec.var;
      AthenaArray<Real> &coarse = *spec.coarse_var;
      const int nu = spec.nvar - 1;
      pmr->RestrictCellCenteredXWithInteriorValues(fine, coarse, 0, nu);
      break;
    }
    case RestrictOp::AreaWeightedFace: {
      // FC: three separate restriction calls, one per face direction.
      // Each has staggering: +1 in its own direction.
      if (spec.var_fc[0] == nullptr || spec.coarse_fc[0] == nullptr) break;
      AthenaArray<Real> &fine_x1f = *spec.var_fc[0];
      AthenaArray<Real> &fine_x2f = *spec.var_fc[1];
      AthenaArray<Real> &fine_x3f = *spec.var_fc[2];
      AthenaArray<Real> &coarse_x1f = *spec.coarse_fc[0];
      AthenaArray<Real> &coarse_x2f = *spec.coarse_fc[1];
      AthenaArray<Real> &coarse_x3f = *spec.coarse_fc[2];
      int f2 = pmb->pmy_mesh->f2;
      int f3 = pmb->pmy_mesh->f3;

      // x1-faces: staggered in x1 -> i-range [cis, cie+1]
      pmr->RestrictFieldX1(fine_x1f, coarse_x1f,
                           pmb->cis, pmb->cie + 1,
                           pmb->cjs, pmb->cje,
                           pmb->cks, pmb->cke);

      // x2-faces: staggered in x2 -> j-range [cjs, cje+f2]
      pmr->RestrictFieldX2(fine_x2f, coarse_x2f,
                           pmb->cis, pmb->cie,
                           pmb->cjs, pmb->cje + f2,
                           pmb->cks, pmb->cke);
      // 1D degenerate: copy the extra x2f face.
      if (pmb->block_size.nx2 == 1) {
        for (int i = pmb->cis; i <= pmb->cie; ++i)
          coarse_x2f(pmb->cks, pmb->cjs + 1, i) = coarse_x2f(pmb->cks, pmb->cjs, i);
      }

      // x3-faces: staggered in x3 -> k-range [cks, cke+f3]
      pmr->RestrictFieldX3(fine_x3f, coarse_x3f,
                           pmb->cis, pmb->cie,
                           pmb->cjs, pmb->cje,
                           pmb->cks, pmb->cke + f3);
      // 1D/2D degenerate: copy the extra x3f face.
      if (pmb->block_size.nx3 == 1) {
        for (int j = pmb->cjs; j <= pmb->cje; ++j)
          for (int i = pmb->cis; i <= pmb->cie; ++i)
            coarse_x3f(pmb->cks + 1, j, i) = coarse_x3f(pmb->cks, j, i);
      }
      break;
    }
    case RestrictOp::LagrangeFull: {
      // CX: symmetric Lagrange polynomial restriction using ghost data.
      // Used by the Z4c iterated boundary path (SendBoundaryBuffersFullRestriction)
      // where ghosts are already filled and contribute to the restriction stencil.
      AthenaArray<Real> &fine = *spec.var;
      AthenaArray<Real> &coarse = *spec.coarse_var;
      const int nu = spec.nvar - 1;
      pmr->RestrictCellCenteredX<NGHOST>(fine, coarse, 0, nu,
                                         pmb->cx_cis, pmb->cx_cie,
                                         pmb->cx_cjs, pmb->cx_cje,
                                         pmb->cx_cks, pmb->cx_cke);
      break;
    }
    case RestrictOp::None:
      break;
  }
}

//----------------------------------------------------------------------------------------
// Compute FC shared-face extended prolongation indices.
// Starting from the base ProlongationIndices range (si..ei, sj..ej, sk..ek), extend each
// non-degenerate dimension by +1.  Then trim shared faces where a same-or-finer-level
// neighbor already exists (those faces were already set by normal boundary comm and must
// not be overwritten by prolongation).
//
// This mirrors FaceCenteredBoundaryVariable::CalculateProlongationSharedIndices exactly.

idx::IndexRange3D ProlongationSharedIndices(const MeshBlock *pmb,
                                             const NeighborBlock &nb,
                                             const idx::IndexRange3D &base) {
  const int mylevel = pmb->loc.level;
  const NeighborConnectivity &nc = pmb->nc();

  idx::IndexRange3D sh;

  // x1: always non-trivial; extend then trim.
  sh.si = base.si;
  sh.ei = base.ei + 1;
  // Trim lower shared face if a same/finer neighbor exists on that side.
  if ((nb.ni.ox1 >= 0) &&
      (nc.neighbor_level(nb.ni.ox1 - 1, nb.ni.ox2, nb.ni.ox3) >= mylevel))
    sh.si++;
  // Trim upper shared face.
  if ((nb.ni.ox1 <= 0) &&
      (nc.neighbor_level(nb.ni.ox1 + 1, nb.ni.ox2, nb.ni.ox3) >= mylevel))
    sh.ei--;

  // x2: extend only if non-degenerate.
  if (pmb->block_size.nx2 > 1) {
    sh.sj = base.sj;
    sh.ej = base.ej + 1;
    if ((nb.ni.ox2 >= 0) &&
        (nc.neighbor_level(nb.ni.ox1, nb.ni.ox2 - 1, nb.ni.ox3) >= mylevel))
      sh.sj++;
    if ((nb.ni.ox2 <= 0) &&
        (nc.neighbor_level(nb.ni.ox1, nb.ni.ox2 + 1, nb.ni.ox3) >= mylevel))
      sh.ej--;
  } else {
    sh.sj = base.sj;
    sh.ej = base.ej;
  }

  // x3: extend only if non-degenerate.
  if (pmb->block_size.nx3 > 1) {
    sh.sk = base.sk;
    sh.ek = base.ek + 1;
    if ((nb.ni.ox3 >= 0) &&
        (nc.neighbor_level(nb.ni.ox1, nb.ni.ox2, nb.ni.ox3 - 1) >= mylevel))
      sh.sk++;
    if ((nb.ni.ox3 <= 0) &&
        (nc.neighbor_level(nb.ni.ox1, nb.ni.ox2, nb.ni.ox3 + 1) >= mylevel))
      sh.ek--;
  } else {
    sh.sk = base.sk;
    sh.ek = base.ek;
  }

  return sh;
}

//----------------------------------------------------------------------------------------
// FC prolongation: 4-step Toth & Roe divergence-preserving sequence.
// Step 1-3: prolongate shared faces in x1, x2, x3 using the extended shared indices
//           in the stagger dimension and base indices in the transverse dimensions.
// Step 4:   fill internal faces using divergence-preserving interpolation on the base
//           index range.  This uses the shared-face values computed in steps 1-3.

void ProlongateNeighborFC(MeshBlock *pmb,
                          const CommSpec &spec,
                          const idx::IndexRange3D &base,
                          const idx::IndexRange3D &shared) {
  MeshRefinement *pmr = pmb->pmr;
  AthenaArray<Real> &coarse_x1f = *spec.coarse_fc[0];
  AthenaArray<Real> &coarse_x2f = *spec.coarse_fc[1];
  AthenaArray<Real> &coarse_x3f = *spec.coarse_fc[2];
  AthenaArray<Real> &fine_x1f = *spec.var_fc[0];
  AthenaArray<Real> &fine_x2f = *spec.var_fc[1];
  AthenaArray<Real> &fine_x3f = *spec.var_fc[2];

  // Step 1: x1 shared faces - i uses shared range (il..iu), j/k use base range.
  pmr->ProlongateSharedFieldX1(coarse_x1f, fine_x1f,
                               shared.si, shared.ei,
                               base.sj, base.ej,
                               base.sk, base.ek);

  // Step 2: x2 shared faces - j uses shared range (jl..ju), i/k use base range.
  pmr->ProlongateSharedFieldX2(coarse_x2f, fine_x2f,
                               base.si, base.ei,
                               shared.sj, shared.ej,
                               base.sk, base.ek);

  // Step 3: x3 shared faces - k uses shared range (kl..ku), i/j use base range.
  pmr->ProlongateSharedFieldX3(coarse_x3f, fine_x3f,
                               base.si, base.ei,
                               base.sj, base.ej,
                               shared.sk, shared.ek);

  // Step 4: divergence-preserving internal field fill on the base range.
  // ProlongateInternalField takes FaceField&, so build a temporary shallow alias.
  // TODO: add an overload that takes 3 AthenaArrays after old bvals is retired.
  FaceField fine_alias;
  fine_alias.x1f.InitWithShallowSlice(fine_x1f);
  fine_alias.x2f.InitWithShallowSlice(fine_x2f);
  fine_alias.x3f.InitWithShallowSlice(fine_x3f);
  pmr->ProlongateInternalField(fine_alias,
                               base.si, base.ei,
                               base.sj, base.ej,
                               base.sk, base.ek);
}

//----------------------------------------------------------------------------------------
// Convert coarse-grid prolongation indices to fine-grid indices.
// Useful for post-prolongation operations (PrimitiveToConserved on prolonged slabs).
//
// The formula is: fine_index = (coarse_index - coarse_start) * 2 + fine_start
// with a +1 on the end index because each coarse cell maps to two fine cells.
// In degenerate dimensions (nx==1), the fine range spans the full single cell.

idx::IndexRange3D ProlongationIndicesFine(const MeshBlock *pmb,
                                          const idx::IndexRange3D &coarse_r,
                                          Sampling sampling) {
  idx::IndexRange3D fr;

  // Select base coarse/fine index pairs per sampling.
  int cbase1, fbase1, cbase2, fbase2, cbase3, fbase3;

  switch (sampling) {
    case Sampling::CC:
    case Sampling::FC:
      cbase1 = pmb->cis;  fbase1 = pmb->is;
      cbase2 = pmb->cjs;  fbase2 = pmb->js;
      cbase3 = pmb->cks;  fbase3 = pmb->ks;
      break;
    case Sampling::CX:
      cbase1 = pmb->cx_cis;  fbase1 = pmb->cx_is;
      cbase2 = pmb->cx_cjs;  fbase2 = pmb->cx_js;
      cbase3 = pmb->cx_cks;  fbase3 = pmb->cx_ks;
      break;
    case Sampling::VC:
      cbase1 = pmb->civs;  fbase1 = pmb->ivs;
      cbase2 = pmb->cjvs;  fbase2 = pmb->jvs;
      cbase3 = pmb->ckvs;  fbase3 = pmb->kvs;
      break;
  }

  // x1 is always non-trivial.
  fr.si = (coarse_r.si - cbase1) * 2 + fbase1;
  fr.ei = (coarse_r.ei - cbase1) * 2 + fbase1 + 1;

  if (pmb->block_size.nx2 > 1) {
    fr.sj = (coarse_r.sj - cbase2) * 2 + fbase2;
    fr.ej = (coarse_r.ej - cbase2) * 2 + fbase2 + 1;
  } else {
    fr.sj = pmb->js;
    fr.ej = pmb->je;
  }

  if (pmb->block_size.nx3 > 1) {
    fr.sk = (coarse_r.sk - cbase3) * 2 + fbase3;
    fr.ek = (coarse_r.ek - cbase3) * 2 + fbase3 + 1;
  } else {
    fr.sk = pmb->ks;
    fr.ek = pmb->ke;
  }

  return fr;
}

} // namespace comm
