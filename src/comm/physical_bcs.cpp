//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file physical_bcs.cpp
//  \brief Physical boundary condition implementations and dispatch.
//
//  Each BC implementation fills ghost zones at one domain face using only the
//  array and index ranges.  The dispatch layer (ApplyPhysicalBCFace) handles
//  parity sign flips for Reflect and PolarWedge BCs using the parity module.
//
//  Face ordering in ApplyPhysicalBCs / ApplyPhysicalBCsOnCoarseLevel matches
//  the old system exactly: x1 first, then x2 (with extended x1 ghost limits),
//  then x3 (with extended x1 + x2 ghost limits).

#include "physical_bcs.hpp"

#include <sstream>
#include <vector>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/mesh_refinement.hpp"
#include "../utils/interp_univariate.hpp"
#include "comm_spec.hpp"
#include "parity.hpp"

namespace comm
{

//========================================================================================
// Individual BC implementations
//========================================================================================

//----------------------------------------------------------------------------------------
// Reflect: ghost cell = mirror-image interior cell.
// Parity sign flips are applied by the dispatch layer after this function.

void ReflectBC(AthenaArray<Real>& var,
               int nvar,
               BoundaryFace face,
               int il,
               int iu,
               int jl,
               int ju,
               int kl,
               int ku,
               int ngh)
{
  switch (face)
  {
    case BoundaryFace::inner_x1:
      for (int n = 0; n < nvar; ++n)
        for (int k = kl; k <= ku; ++k)
          for (int j = jl; j <= ju; ++j)
#pragma omp simd
            for (int i = 1; i <= ngh; ++i)
              var(n, k, j, il - i) = var(n, k, j, il + i - 1);
      break;
    case BoundaryFace::outer_x1:
      for (int n = 0; n < nvar; ++n)
        for (int k = kl; k <= ku; ++k)
          for (int j = jl; j <= ju; ++j)
#pragma omp simd
            for (int i = 1; i <= ngh; ++i)
              var(n, k, j, iu + i) = var(n, k, j, iu - i + 1);
      break;
    case BoundaryFace::inner_x2:
      for (int n = 0; n < nvar; ++n)
        for (int k = kl; k <= ku; ++k)
          for (int j = 1; j <= ngh; ++j)
#pragma omp simd
            for (int i = il; i <= iu; ++i)
              var(n, k, jl - j, i) = var(n, k, jl + j - 1, i);
      break;
    case BoundaryFace::outer_x2:
      for (int n = 0; n < nvar; ++n)
        for (int k = kl; k <= ku; ++k)
          for (int j = 1; j <= ngh; ++j)
#pragma omp simd
            for (int i = il; i <= iu; ++i)
              var(n, k, ju + j, i) = var(n, k, ju - j + 1, i);
      break;
    case BoundaryFace::inner_x3:
      for (int n = 0; n < nvar; ++n)
        for (int k = 1; k <= ngh; ++k)
          for (int j = jl; j <= ju; ++j)
#pragma omp simd
            for (int i = il; i <= iu; ++i)
              var(n, kl - k, j, i) = var(n, kl + k - 1, j, i);
      break;
    case BoundaryFace::outer_x3:
      for (int n = 0; n < nvar; ++n)
        for (int k = 1; k <= ngh; ++k)
          for (int j = jl; j <= ju; ++j)
#pragma omp simd
            for (int i = il; i <= iu; ++i)
              var(n, ku + k, j, i) = var(n, ku - k + 1, j, i);
      break;
    default:
      break;
  }
}

//----------------------------------------------------------------------------------------
// Outflow: zeroth-order extrapolation (all ghost cells = boundary cell value).

void OutflowBC(AthenaArray<Real>& var,
               int nvar,
               BoundaryFace face,
               int il,
               int iu,
               int jl,
               int ju,
               int kl,
               int ku,
               int ngh)
{
  switch (face)
  {
    case BoundaryFace::inner_x1:
      for (int n = 0; n < nvar; ++n)
        for (int k = kl; k <= ku; ++k)
          for (int j = jl; j <= ju; ++j)
#pragma omp simd
            for (int i = 1; i <= ngh; ++i)
              var(n, k, j, il - i) = var(n, k, j, il);
      break;
    case BoundaryFace::outer_x1:
      for (int n = 0; n < nvar; ++n)
        for (int k = kl; k <= ku; ++k)
          for (int j = jl; j <= ju; ++j)
#pragma omp simd
            for (int i = 1; i <= ngh; ++i)
              var(n, k, j, iu + i) = var(n, k, j, iu);
      break;
    case BoundaryFace::inner_x2:
      for (int n = 0; n < nvar; ++n)
        for (int k = kl; k <= ku; ++k)
          for (int j = 1; j <= ngh; ++j)
#pragma omp simd
            for (int i = il; i <= iu; ++i)
              var(n, k, jl - j, i) = var(n, k, jl, i);
      break;
    case BoundaryFace::outer_x2:
      for (int n = 0; n < nvar; ++n)
        for (int k = kl; k <= ku; ++k)
          for (int j = 1; j <= ngh; ++j)
#pragma omp simd
            for (int i = il; i <= iu; ++i)
              var(n, k, ju + j, i) = var(n, k, ju, i);
      break;
    case BoundaryFace::inner_x3:
      for (int n = 0; n < nvar; ++n)
        for (int k = 1; k <= ngh; ++k)
          for (int j = jl; j <= ju; ++j)
#pragma omp simd
            for (int i = il; i <= iu; ++i)
              var(n, kl - k, j, i) = var(n, kl, j, i);
      break;
    case BoundaryFace::outer_x3:
      for (int n = 0; n < nvar; ++n)
        for (int k = 1; k <= ngh; ++k)
          for (int j = jl; j <= ju; ++j)
#pragma omp simd
            for (int i = il; i <= iu; ++i)
              var(n, ku + k, j, i) = var(n, ku, j, i);
      break;
    default:
      break;
  }
}

//----------------------------------------------------------------------------------------
// ExtrapolateOutflow: polynomial extrapolation of order NEXTRAPOLATE-1.
// Ghost cells are filled sequentially from the boundary outward (cascading).
// Extrapolation order is compile-time configurable via configure.py
// --nextrapolate. Coefficients are the signed binomial coefficients from
// BiasR<NEXTRAPOLATE>.
//
// NOTE: the x1 inner loops have a loop-carried dependency (each ghost cell
// reads the one filled in the previous iteration). The #pragma omp simd is
// inherited from the old system; compilers typically detect the dependency and
// ignore the hint. The x2/x3 cases are safe because the cascading loop is an
// outer loop, not the simd-annotated one.

void ExtrapolateOutflowBC(AthenaArray<Real>& var,
                          int nvar,
                          BoundaryFace face,
                          int il,
                          int iu,
                          int jl,
                          int ju,
                          int kl,
                          int ku,
                          int ngh)
{
  using InterpUniform::InterpolateLagrangeUniformBiasR;
  constexpr int NE = NEXTRAPOLATE;
  const Real* c    = InterpolateLagrangeUniformBiasR<NE>::coeff;

  switch (face)
  {
    case BoundaryFace::inner_x1:
      for (int n = 0; n < nvar; ++n)
        for (int k = kl; k <= ku; ++k)
          for (int j = jl; j <= ju; ++j)
            for (int i = il - 1; i >= il - ngh; --i)
            {
              Real v = 0.0;
              for (int p = 0; p < NE; ++p)
                v += c[p] * var(n, k, j, i + p + 1);
              var(n, k, j, i) = v;
            }
      break;
    case BoundaryFace::outer_x1:
      for (int n = 0; n < nvar; ++n)
        for (int k = kl; k <= ku; ++k)
          for (int j = jl; j <= ju; ++j)
            for (int i = iu + 1; i <= iu + ngh; ++i)
            {
              Real v = 0.0;
              for (int p = 0; p < NE; ++p)
                v += c[p] * var(n, k, j, i - p - 1);
              var(n, k, j, i) = v;
            }
      break;
    case BoundaryFace::inner_x2:
      for (int n = 0; n < nvar; ++n)
        for (int k = kl; k <= ku; ++k)
          for (int j = jl - 1; j >= jl - ngh; --j)
#pragma omp simd
            for (int i = il; i <= iu; ++i)
            {
              Real v = 0.0;
              for (int p = 0; p < NE; ++p)
                v += c[p] * var(n, k, j + p + 1, i);
              var(n, k, j, i) = v;
            }
      break;
    case BoundaryFace::outer_x2:
      for (int n = 0; n < nvar; ++n)
        for (int k = kl; k <= ku; ++k)
          for (int j = ju + 1; j <= ju + ngh; ++j)
#pragma omp simd
            for (int i = il; i <= iu; ++i)
            {
              Real v = 0.0;
              for (int p = 0; p < NE; ++p)
                v += c[p] * var(n, k, j - p - 1, i);
              var(n, k, j, i) = v;
            }
      break;
    case BoundaryFace::inner_x3:
      for (int n = 0; n < nvar; ++n)
        for (int k = kl - 1; k >= kl - ngh; --k)
          for (int j = jl; j <= ju; ++j)
#pragma omp simd
            for (int i = il; i <= iu; ++i)
            {
              Real v = 0.0;
              for (int p = 0; p < NE; ++p)
                v += c[p] * var(n, k + p + 1, j, i);
              var(n, k, j, i) = v;
            }
      break;
    case BoundaryFace::outer_x3:
      for (int n = 0; n < nvar; ++n)
        for (int k = ku + 1; k <= ku + ngh; ++k)
          for (int j = jl; j <= ju; ++j)
#pragma omp simd
            for (int i = il; i <= iu; ++i)
            {
              Real v = 0.0;
              for (int p = 0; p < NE; ++p)
                v += c[p] * var(n, k - p - 1, j, i);
              var(n, k, j, i) = v;
            }
      break;
    default:
      break;
  }
}

//----------------------------------------------------------------------------------------
// PolarWedge: mirror copy with j-reversal. Only valid on inner/outer x2.
// This is the geometric fill only - parity sign flips are applied by the
// dispatch layer. Matches the old polarwedge_cc.cpp pattern but without the
// hardcoded flip array.

void PolarWedgeBC(AthenaArray<Real>& var,
                  int nvar,
                  BoundaryFace face,
                  int il,
                  int iu,
                  int jl,
                  int ju,
                  int kl,
                  int ku,
                  int ngh)
{
  if (face != BoundaryFace::inner_x2 && face != BoundaryFace::outer_x2)
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in PolarWedgeBC\n"
        << "PolarWedge BC is only valid on inner/outer x2 faces." << std::endl;
    ATHENA_ERROR(msg);
  }

  // Mirror copy across the pole: ghost zone j reflects about the boundary.
  // Identical to ReflectBC for x2, but the parity sign flips will use Polar
  // context rather than ReflectX2 context (different sign rules).
  if (face == BoundaryFace::inner_x2)
  {
    for (int n = 0; n < nvar; ++n)
      for (int k = kl; k <= ku; ++k)
        for (int j = 1; j <= ngh; ++j)
#pragma omp simd
          for (int i = il; i <= iu; ++i)
            var(n, k, jl - j, i) = var(n, k, jl + j - 1, i);
  }
  else
  {
    for (int n = 0; n < nvar; ++n)
      for (int k = kl; k <= ku; ++k)
        for (int j = 1; j <= ngh; ++j)
#pragma omp simd
          for (int i = il; i <= iu; ++i)
            var(n, k, ju + j, i) = var(n, k, ju - j + 1, i);
  }
}

//========================================================================================
// Ghost-zone index range helpers for parity sign application
//========================================================================================

// Compute the ghost-zone index range that a BC fill wrote to, for applying
// signs. The BC fills ghost zones outside [il..iu] x [jl..ju] x [kl..ku] by
// ngh cells on the face side.  This returns the range of the filled ghost
// region.

static void GhostRange(BoundaryFace face,
                       int il,
                       int iu,
                       int jl,
                       int ju,
                       int kl,
                       int ku,
                       int ngh,
                       int& si,
                       int& ei,
                       int& sj,
                       int& ej,
                       int& sk,
                       int& ek)
{
  si = il;
  ei = iu;
  sj = jl;
  ej = ju;
  sk = kl;
  ek = ku;
  switch (face)
  {
    case BoundaryFace::inner_x1:
      si = il - ngh;
      ei = il - 1;
      break;
    case BoundaryFace::outer_x1:
      si = iu + 1;
      ei = iu + ngh;
      break;
    case BoundaryFace::inner_x2:
      sj = jl - ngh;
      ej = jl - 1;
      break;
    case BoundaryFace::outer_x2:
      sj = ju + 1;
      ej = ju + ngh;
      break;
    case BoundaryFace::inner_x3:
      sk = kl - ngh;
      ek = kl - 1;
      break;
    case BoundaryFace::outer_x3:
      sk = ku + 1;
      ek = ku + ngh;
      break;
    default:
      break;
  }
}

//========================================================================================
// Single-face dispatch
//========================================================================================

//----------------------------------------------------------------------------------------
// Apply the appropriate BC on one face.
// GRSommerfeld is not a separate implementation - it maps to Outflow (CC/FC)
// or ExtrapolateOutflow (CX/VC), matching the old system's delegation pattern.
//
// For Reflect and PolarWedge, parity sign flips are applied after the
// geometric fill.

void ApplyPhysicalBCFace(AthenaArray<Real>& var,
                         BoundaryFace face,
                         int il,
                         int iu,
                         int jl,
                         int ju,
                         int kl,
                         int ku,
                         int ngh,
                         PhysicalBC bc,
                         const CommSpec& spec)
{
  const int nvar          = spec.nvar;
  const Sampling sampling = spec.sampling;

  switch (bc)
  {
    case PhysicalBC::Reflect:
    {
      ReflectBC(var, nvar, face, il, iu, jl, ju, kl, ku, ngh);
      // Apply parity sign flips: vector normals and tensor off-diagonals.
      if (!spec.component_groups.empty())
      {
        std::vector<Real> signs = ComputeSignArray(spec, ReflectContext(face));
        int si, ei, sj, ej, sk, ek;
        GhostRange(face, il, iu, jl, ju, kl, ku, ngh, si, ei, sj, ej, sk, ek);
        ApplyParitySigns(var, nvar, signs, si, ei, sj, ej, sk, ek);
      }
      break;
    }
    case PhysicalBC::Outflow:
      OutflowBC(var, nvar, face, il, iu, jl, ju, kl, ku, ngh);
      break;
    case PhysicalBC::ExtrapolateOutflow:
      ExtrapolateOutflowBC(var, nvar, face, il, iu, jl, ju, kl, ku, ngh);
      break;
    case PhysicalBC::GRSommerfeld:
      // GRSommerfeld delegates to sampling-appropriate fallback.
      // CC/FC -> Outflow (zeroth-order copy).
      // CX/VC -> ExtrapolateOutflow (4th-order polynomial).
      if (sampling == Sampling::CC || sampling == Sampling::FC)
      {
        OutflowBC(var, nvar, face, il, iu, jl, ju, kl, ku, ngh);
      }
      else
      {
        ExtrapolateOutflowBC(var, nvar, face, il, iu, jl, ju, kl, ku, ngh);
      }
      break;
    case PhysicalBC::PolarWedge:
    {
      PolarWedgeBC(var, nvar, face, il, iu, jl, ju, kl, ku, ngh);
      // Apply polar parity sign flips: theta/phi vector components and mixed
      // tensors.
      if (!spec.component_groups.empty())
      {
        std::vector<Real> signs = ComputeSignArray(spec, FlipContext::Polar);
        int si, ei, sj, ej, sk, ek;
        GhostRange(face, il, iu, jl, ju, kl, ku, ngh, si, ei, sj, ej, sk, ek);
        ApplyParitySigns(var, nvar, signs, si, ei, sj, ej, sk, ek);
      }
      break;
    }
    case PhysicalBC::User:
      // User BCs are handled by the caller (they call a per-face enrolled
      // function).
      break;
    case PhysicalBC::None:
      break;
  }
}

//========================================================================================
// Multi-face dispatch
//========================================================================================

//----------------------------------------------------------------------------------------
// Helper: determine whether a face needs physical BC dispatch.
// Uses the same logic as the old apply_bndry_fn_[] array: true for reflect,
// outflow, extrapolate_outflow, gr_sommerfeld, polar_wedge, user.  False for
// block/periodic/polar.

static bool FaceNeedsBC(MeshBlock* pmb, BoundaryFace face)
{
  BoundaryFlag bf = pmb->nc().boundary_flag(face);
  return (bf == BoundaryFlag::reflect || bf == BoundaryFlag::outflow ||
          bf == BoundaryFlag::extrapolate_outflow ||
          bf == BoundaryFlag::gr_sommerfeld ||
          bf == BoundaryFlag::polar_wedge || bf == BoundaryFlag::user);
}

//----------------------------------------------------------------------------------------
// Apply physical BCs on all active faces for one channel, fine level.
//
// Face ordering and transverse-limit extension match the old system exactly:
//   1. X1 faces (inner, outer) - transverse limits extended for periodic x2/x3
//   2. X2 faces (inner, outer) - i-range includes x1 ghost zones; k extended
//   for periodic x3
//   3. X3 faces (inner, outer) - i-range and j-range include ghost zones

void ApplyPhysicalBCs(MeshBlock* pmb, const CommSpec& spec, Real time, Real dt)
{
  if (!spec.HasAnyPhysicalBC())
    return;

  AthenaArray<Real>& var = *spec.var;
  const int ng           = spec.nghost;

  // Interior bounds for this variable (fine level).
  const int is = pmb->is, ie = pmb->ie;
  const int js = pmb->js, je = pmb->je;
  const int ks = pmb->ks, ke = pmb->ke;

  // Start with transverse limits = interior, then extend for periodic faces.
  int bis = is - ng, bie = ie + ng;
  int bjs = js, bje = je;
  int bks = ks, bke = ke;

  if (!FaceNeedsBC(pmb, BoundaryFace::inner_x2) && pmb->block_size.nx2 > 1)
    bjs = js - ng;
  if (!FaceNeedsBC(pmb, BoundaryFace::outer_x2) && pmb->block_size.nx2 > 1)
    bje = je + ng;
  if (!FaceNeedsBC(pmb, BoundaryFace::inner_x3) && pmb->block_size.nx3 > 1)
    bks = ks - ng;
  if (!FaceNeedsBC(pmb, BoundaryFace::outer_x3) && pmb->block_size.nx3 > 1)
    bke = ke + ng;

  // X1 faces.
  if (FaceNeedsBC(pmb, BoundaryFace::inner_x1))
  {
    PhysicalBC bc = spec.physical_bc[static_cast<int>(BoundaryFace::inner_x1)];
    if (bc == PhysicalBC::User && spec.physical_bc_fn != nullptr)
    {
      spec.physical_bc_fn(
        pmb, pmb->pcoord, var, time, dt, is, ie, bjs, bje, bks, bke, ng);
    }
    else
    {
      ApplyPhysicalBCFace(
        var, BoundaryFace::inner_x1, is, ie, bjs, bje, bks, bke, ng, bc, spec);
    }
  }
  if (FaceNeedsBC(pmb, BoundaryFace::outer_x1))
  {
    PhysicalBC bc = spec.physical_bc[static_cast<int>(BoundaryFace::outer_x1)];
    if (bc == PhysicalBC::User && spec.physical_bc_fn != nullptr)
    {
      spec.physical_bc_fn(
        pmb, pmb->pcoord, var, time, dt, is, ie, bjs, bje, bks, bke, ng);
    }
    else
    {
      ApplyPhysicalBCFace(
        var, BoundaryFace::outer_x1, is, ie, bjs, bje, bks, bke, ng, bc, spec);
    }
  }

  // X2 faces (only in 2D+).
  if (pmb->block_size.nx2 > 1)
  {
    if (FaceNeedsBC(pmb, BoundaryFace::inner_x2))
    {
      PhysicalBC bc =
        spec.physical_bc[static_cast<int>(BoundaryFace::inner_x2)];
      if (bc == PhysicalBC::User && spec.physical_bc_fn != nullptr)
      {
        spec.physical_bc_fn(
          pmb, pmb->pcoord, var, time, dt, bis, bie, js, je, bks, bke, ng);
      }
      else
      {
        ApplyPhysicalBCFace(var,
                            BoundaryFace::inner_x2,
                            bis,
                            bie,
                            js,
                            je,
                            bks,
                            bke,
                            ng,
                            bc,
                            spec);
      }
    }
    if (FaceNeedsBC(pmb, BoundaryFace::outer_x2))
    {
      PhysicalBC bc =
        spec.physical_bc[static_cast<int>(BoundaryFace::outer_x2)];
      if (bc == PhysicalBC::User && spec.physical_bc_fn != nullptr)
      {
        spec.physical_bc_fn(
          pmb, pmb->pcoord, var, time, dt, bis, bie, js, je, bks, bke, ng);
      }
      else
      {
        ApplyPhysicalBCFace(var,
                            BoundaryFace::outer_x2,
                            bis,
                            bie,
                            js,
                            je,
                            bks,
                            bke,
                            ng,
                            bc,
                            spec);
      }
    }
  }

  // X3 faces (only in 3D).
  if (pmb->block_size.nx3 > 1)
  {
    // After x2 BCs are applied, extend j-range to include x2 ghost zones.
    bjs = js - ng;
    bje = je + ng;

    if (FaceNeedsBC(pmb, BoundaryFace::inner_x3))
    {
      PhysicalBC bc =
        spec.physical_bc[static_cast<int>(BoundaryFace::inner_x3)];
      if (bc == PhysicalBC::User && spec.physical_bc_fn != nullptr)
      {
        spec.physical_bc_fn(
          pmb, pmb->pcoord, var, time, dt, bis, bie, bjs, bje, ks, ke, ng);
      }
      else
      {
        ApplyPhysicalBCFace(var,
                            BoundaryFace::inner_x3,
                            bis,
                            bie,
                            bjs,
                            bje,
                            ks,
                            ke,
                            ng,
                            bc,
                            spec);
      }
    }
    if (FaceNeedsBC(pmb, BoundaryFace::outer_x3))
    {
      PhysicalBC bc =
        spec.physical_bc[static_cast<int>(BoundaryFace::outer_x3)];
      if (bc == PhysicalBC::User && spec.physical_bc_fn != nullptr)
      {
        spec.physical_bc_fn(
          pmb, pmb->pcoord, var, time, dt, bis, bie, bjs, bje, ks, ke, ng);
      }
      else
      {
        ApplyPhysicalBCFace(var,
                            BoundaryFace::outer_x3,
                            bis,
                            bie,
                            bjs,
                            bje,
                            ks,
                            ke,
                            ng,
                            bc,
                            spec);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
// Apply physical BCs on the coarse representation for prolongation.
//
// This replaces ApplyPhysicalBoundariesOnCoarseLevel().  Instead of swapping
// var/coarse pointers (InterchangeFundamentalCoarse), we pass the coarse array
// directly. The coarse coordinates come from MeshRefinement::pcoarsec.
//
// The caller provides the coarse interior bounds (cis/cie/cjs/cje/cks/cke) and
// the coarse ghost width (cng).  These vary per sampling type:
//   CC  -> pmb->cis/cie/cjs/cje/cks/cke, cng = NGHOST
//   CX  -> pz4c->mbi.cil/ciu/cjl/cju/ckl/cku, cng = pz4c->mbi.cng
//   VC  -> same as CX with VC-specific bounds

void ApplyPhysicalBCsOnCoarseLevel(MeshBlock* pmb,
                                   const CommSpec& spec,
                                   Real time,
                                   Real dt,
                                   int cis,
                                   int cie,
                                   int cjs,
                                   int cje,
                                   int cks,
                                   int cke,
                                   int cng)
{
  if (!spec.HasAnyPhysicalBC())
    return;
  if (spec.coarse_var == nullptr)
    return;

  AthenaArray<Real>& cvar = *spec.coarse_var;
  Coordinates* pco        = pmb->pmr->pcoarsec;

  // Transverse limit extension for periodic faces (same logic as fine level).
  int bis = cis - cng, bie = cie + cng;
  int bjs = cjs, bje = cje;
  int bks = cks, bke = cke;

  if (!FaceNeedsBC(pmb, BoundaryFace::inner_x2) && pmb->block_size.nx2 > 1)
    bjs = cjs - cng;
  if (!FaceNeedsBC(pmb, BoundaryFace::outer_x2) && pmb->block_size.nx2 > 1)
    bje = cje + cng;
  if (!FaceNeedsBC(pmb, BoundaryFace::inner_x3) && pmb->block_size.nx3 > 1)
    bks = cks - cng;
  if (!FaceNeedsBC(pmb, BoundaryFace::outer_x3) && pmb->block_size.nx3 > 1)
    bke = cke + cng;

  // X1 faces.
  if (FaceNeedsBC(pmb, BoundaryFace::inner_x1))
  {
    PhysicalBC bc = spec.physical_bc[static_cast<int>(BoundaryFace::inner_x1)];
    if (bc == PhysicalBC::User && spec.physical_bc_fn != nullptr)
    {
      spec.physical_bc_fn(
        pmb, pco, cvar, time, dt, cis, cie, bjs, bje, bks, bke, cng);
    }
    else
    {
      ApplyPhysicalBCFace(cvar,
                          BoundaryFace::inner_x1,
                          cis,
                          cie,
                          bjs,
                          bje,
                          bks,
                          bke,
                          cng,
                          bc,
                          spec);
    }
  }
  if (FaceNeedsBC(pmb, BoundaryFace::outer_x1))
  {
    PhysicalBC bc = spec.physical_bc[static_cast<int>(BoundaryFace::outer_x1)];
    if (bc == PhysicalBC::User && spec.physical_bc_fn != nullptr)
    {
      spec.physical_bc_fn(
        pmb, pco, cvar, time, dt, cis, cie, bjs, bje, bks, bke, cng);
    }
    else
    {
      ApplyPhysicalBCFace(cvar,
                          BoundaryFace::outer_x1,
                          cis,
                          cie,
                          bjs,
                          bje,
                          bks,
                          bke,
                          cng,
                          bc,
                          spec);
    }
  }

  // X2 faces.
  if (pmb->block_size.nx2 > 1)
  {
    if (FaceNeedsBC(pmb, BoundaryFace::inner_x2))
    {
      PhysicalBC bc =
        spec.physical_bc[static_cast<int>(BoundaryFace::inner_x2)];
      if (bc == PhysicalBC::User && spec.physical_bc_fn != nullptr)
      {
        spec.physical_bc_fn(
          pmb, pco, cvar, time, dt, bis, bie, cjs, cje, bks, bke, cng);
      }
      else
      {
        ApplyPhysicalBCFace(cvar,
                            BoundaryFace::inner_x2,
                            bis,
                            bie,
                            cjs,
                            cje,
                            bks,
                            bke,
                            cng,
                            bc,
                            spec);
      }
    }
    if (FaceNeedsBC(pmb, BoundaryFace::outer_x2))
    {
      PhysicalBC bc =
        spec.physical_bc[static_cast<int>(BoundaryFace::outer_x2)];
      if (bc == PhysicalBC::User && spec.physical_bc_fn != nullptr)
      {
        spec.physical_bc_fn(
          pmb, pco, cvar, time, dt, bis, bie, cjs, cje, bks, bke, cng);
      }
      else
      {
        ApplyPhysicalBCFace(cvar,
                            BoundaryFace::outer_x2,
                            bis,
                            bie,
                            cjs,
                            cje,
                            bks,
                            bke,
                            cng,
                            bc,
                            spec);
      }
    }
  }

  // X3 faces.
  if (pmb->block_size.nx3 > 1)
  {
    bjs = cjs - cng;
    bje = cje + cng;

    if (FaceNeedsBC(pmb, BoundaryFace::inner_x3))
    {
      PhysicalBC bc =
        spec.physical_bc[static_cast<int>(BoundaryFace::inner_x3)];
      if (bc == PhysicalBC::User && spec.physical_bc_fn != nullptr)
      {
        spec.physical_bc_fn(
          pmb, pco, cvar, time, dt, bis, bie, bjs, bje, cks, cke, cng);
      }
      else
      {
        ApplyPhysicalBCFace(cvar,
                            BoundaryFace::inner_x3,
                            bis,
                            bie,
                            bjs,
                            bje,
                            cks,
                            cke,
                            cng,
                            bc,
                            spec);
      }
    }
    if (FaceNeedsBC(pmb, BoundaryFace::outer_x3))
    {
      PhysicalBC bc =
        spec.physical_bc[static_cast<int>(BoundaryFace::outer_x3)];
      if (bc == PhysicalBC::User && spec.physical_bc_fn != nullptr)
      {
        spec.physical_bc_fn(
          pmb, pco, cvar, time, dt, bis, bie, bjs, bje, cks, cke, cng);
      }
      else
      {
        ApplyPhysicalBCFace(cvar,
                            BoundaryFace::outer_x3,
                            bis,
                            bie,
                            bjs,
                            bje,
                            cks,
                            cke,
                            cng,
                            bc,
                            spec);
      }
    }
  }
}

//========================================================================================
// Face-centered (FC) boundary condition implementations
//========================================================================================
//
// FC BCs operate on all 3 components of a FaceField simultaneously.  Each
// component's loop range extends by +1 in its own stagger direction (x1f: i+1,
// x2f: j+1, x3f: k+1). Sign flips are hardcoded here (not via parity module)
// because the FC field has only 3 components with known geometric meaning: B1
// = normal on x1 faces, etc.

//----------------------------------------------------------------------------------------
// ReflectBC_FC: mirror-image copy with normal component negation.
//
// For x1 faces (boundary in i-direction):
//   x1f (normal): ghost = -interior (sign flip), source offset by +1 vs CC
//   (face stagger) x2f (tangential): ghost = +interior, j-range extended by +1
//   (x2f stagger) x3f (tangential): ghost = +interior, k-range extended by +1
//   (x3f stagger)
// Analogous patterns for x2/x3 faces, with the normal component rotating
// accordingly.

void ReflectBC_FC(AthenaArray<Real>* fc[3],
                  BoundaryFace face,
                  int il,
                  int iu,
                  int jl,
                  int ju,
                  int kl,
                  int ku,
                  int ngh)
{
  AthenaArray<Real>& x1f = *fc[0];
  AthenaArray<Real>& x2f = *fc[1];
  AthenaArray<Real>& x3f = *fc[2];

  switch (face)
  {
    case BoundaryFace::inner_x1:
      // x1f (normal): negate, source at il+i (face stagger, not il+i-1)
      for (int k = kl; k <= ku; ++k)
        for (int j = jl; j <= ju; ++j)
#pragma omp simd
          for (int i = 1; i <= ngh; ++i)
            x1f(k, j, il - i) = -x1f(k, j, il + i);
      // x2f (tangential): copy, j-range +1 for x2f stagger
      for (int k = kl; k <= ku; ++k)
        for (int j = jl; j <= ju + 1; ++j)
#pragma omp simd
          for (int i = 1; i <= ngh; ++i)
            x2f(k, j, il - i) = x2f(k, j, il + i - 1);
      // x3f (tangential): copy, k-range +1 for x3f stagger
      for (int k = kl; k <= ku + 1; ++k)
        for (int j = jl; j <= ju; ++j)
#pragma omp simd
          for (int i = 1; i <= ngh; ++i)
            x3f(k, j, il - i) = x3f(k, j, il + i - 1);
      break;

    case BoundaryFace::outer_x1:
      // x1f (normal): negate, outer face stagger: src = iu-i+1, dst = iu+i+1
      for (int k = kl; k <= ku; ++k)
        for (int j = jl; j <= ju; ++j)
#pragma omp simd
          for (int i = 1; i <= ngh; ++i)
            x1f(k, j, iu + i + 1) = -x1f(k, j, iu - i + 1);
      // x2f (tangential): j-range +1
      for (int k = kl; k <= ku; ++k)
        for (int j = jl; j <= ju + 1; ++j)
#pragma omp simd
          for (int i = 1; i <= ngh; ++i)
            x2f(k, j, iu + i) = x2f(k, j, iu - i + 1);
      // x3f (tangential): k-range +1
      for (int k = kl; k <= ku + 1; ++k)
        for (int j = jl; j <= ju; ++j)
#pragma omp simd
          for (int i = 1; i <= ngh; ++i)
            x3f(k, j, iu + i) = x3f(k, j, iu - i + 1);
      break;

    case BoundaryFace::inner_x2:
      // x1f (tangential): i-range +1 for x1f stagger
      for (int k = kl; k <= ku; ++k)
        for (int j = 1; j <= ngh; ++j)
#pragma omp simd
          for (int i = il; i <= iu + 1; ++i)
            x1f(k, jl - j, i) = x1f(k, jl + j - 1, i);
      // x2f (normal): negate, source at jl+j (face stagger)
      for (int k = kl; k <= ku; ++k)
        for (int j = 1; j <= ngh; ++j)
#pragma omp simd
          for (int i = il; i <= iu; ++i)
            x2f(k, jl - j, i) = -x2f(k, jl + j, i);
      // x3f (tangential): k-range +1 for x3f stagger
      for (int k = kl; k <= ku + 1; ++k)
        for (int j = 1; j <= ngh; ++j)
#pragma omp simd
          for (int i = il; i <= iu; ++i)
            x3f(k, jl - j, i) = x3f(k, jl + j - 1, i);
      break;

    case BoundaryFace::outer_x2:
      // x1f (tangential): i-range +1
      for (int k = kl; k <= ku; ++k)
        for (int j = 1; j <= ngh; ++j)
#pragma omp simd
          for (int i = il; i <= iu + 1; ++i)
            x1f(k, ju + j, i) = x1f(k, ju - j + 1, i);
      // x2f (normal): negate, outer face stagger: dst = ju+j+1, src = ju-j+1
      for (int k = kl; k <= ku; ++k)
        for (int j = 1; j <= ngh; ++j)
#pragma omp simd
          for (int i = il; i <= iu; ++i)
            x2f(k, ju + j + 1, i) = -x2f(k, ju - j + 1, i);
      // x3f (tangential): k-range +1
      for (int k = kl; k <= ku + 1; ++k)
        for (int j = 1; j <= ngh; ++j)
#pragma omp simd
          for (int i = il; i <= iu; ++i)
            x3f(k, ju + j, i) = x3f(k, ju - j + 1, i);
      break;

    case BoundaryFace::inner_x3:
      // x1f (tangential): i-range +1
      for (int k = 1; k <= ngh; ++k)
        for (int j = jl; j <= ju; ++j)
#pragma omp simd
          for (int i = il; i <= iu + 1; ++i)
            x1f(kl - k, j, i) = x1f(kl + k - 1, j, i);
      // x2f (tangential): j-range +1
      for (int k = 1; k <= ngh; ++k)
        for (int j = jl; j <= ju + 1; ++j)
#pragma omp simd
          for (int i = il; i <= iu; ++i)
            x2f(kl - k, j, i) = x2f(kl + k - 1, j, i);
      // x3f (normal): negate, source at kl+k (face stagger)
      for (int k = 1; k <= ngh; ++k)
        for (int j = jl; j <= ju; ++j)
#pragma omp simd
          for (int i = il; i <= iu; ++i)
            x3f(kl - k, j, i) = -x3f(kl + k, j, i);
      break;

    case BoundaryFace::outer_x3:
      // x1f (tangential): i-range +1
      for (int k = 1; k <= ngh; ++k)
        for (int j = jl; j <= ju; ++j)
#pragma omp simd
          for (int i = il; i <= iu + 1; ++i)
            x1f(ku + k, j, i) = x1f(ku - k + 1, j, i);
      // x2f (tangential): j-range +1
      for (int k = 1; k <= ngh; ++k)
        for (int j = jl; j <= ju + 1; ++j)
#pragma omp simd
          for (int i = il; i <= iu; ++i)
            x2f(ku + k, j, i) = x2f(ku - k + 1, j, i);
      // x3f (normal): negate, outer face stagger: dst = ku+k+1, src = ku-k+1
      for (int k = 1; k <= ngh; ++k)
        for (int j = jl; j <= ju; ++j)
#pragma omp simd
          for (int i = il; i <= iu; ++i)
            x3f(ku + k + 1, j, i) = -x3f(ku - k + 1, j, i);
      break;

    default:
      break;
  }
}

//----------------------------------------------------------------------------------------
// OutflowBC_FC: zeroth-order extrapolation for face fields.
//
// Normal component copies from the boundary face (e.g. x1f at il for inner_x1,
// x1f at iu+1 for outer_x1).  Tangential components copy from the boundary
// cell (e.g. x2f at il for inner_x1, x2f at iu for outer_x1). Each component
// extends its own stagger direction by +1 in the loop range.

void OutflowBC_FC(AthenaArray<Real>* fc[3],
                  BoundaryFace face,
                  int il,
                  int iu,
                  int jl,
                  int ju,
                  int kl,
                  int ku,
                  int ngh)
{
  AthenaArray<Real>& x1f = *fc[0];
  AthenaArray<Real>& x2f = *fc[1];
  AthenaArray<Real>& x3f = *fc[2];

  switch (face)
  {
    case BoundaryFace::inner_x1:
      // x1f (normal): copy from boundary face at il
      for (int k = kl; k <= ku; ++k)
        for (int j = jl; j <= ju; ++j)
#pragma omp simd
          for (int i = 1; i <= ngh; ++i)
            x1f(k, j, il - i) = x1f(k, j, il);
      // x2f (tangential): copy from boundary cell at il, j-range +1
      for (int k = kl; k <= ku; ++k)
        for (int j = jl; j <= ju + 1; ++j)
#pragma omp simd
          for (int i = 1; i <= ngh; ++i)
            x2f(k, j, il - i) = x2f(k, j, il);
      // x3f (tangential): copy from boundary cell at il, k-range +1
      for (int k = kl; k <= ku + 1; ++k)
        for (int j = jl; j <= ju; ++j)
#pragma omp simd
          for (int i = 1; i <= ngh; ++i)
            x3f(k, j, il - i) = x3f(k, j, il);
      break;

    case BoundaryFace::outer_x1:
      // x1f (normal): copy from boundary face at iu+1, dst = iu+i+1
      for (int k = kl; k <= ku; ++k)
        for (int j = jl; j <= ju; ++j)
#pragma omp simd
          for (int i = 1; i <= ngh; ++i)
            x1f(k, j, iu + i + 1) = x1f(k, j, iu + 1);
      // x2f (tangential): copy from boundary cell at iu, j-range +1
      for (int k = kl; k <= ku; ++k)
        for (int j = jl; j <= ju + 1; ++j)
#pragma omp simd
          for (int i = 1; i <= ngh; ++i)
            x2f(k, j, iu + i) = x2f(k, j, iu);
      // x3f (tangential): copy from boundary cell at iu, k-range +1
      for (int k = kl; k <= ku + 1; ++k)
        for (int j = jl; j <= ju; ++j)
#pragma omp simd
          for (int i = 1; i <= ngh; ++i)
            x3f(k, j, iu + i) = x3f(k, j, iu);
      break;

    case BoundaryFace::inner_x2:
      // x1f (tangential): copy from boundary cell at jl, i-range +1
      for (int k = kl; k <= ku; ++k)
        for (int j = 1; j <= ngh; ++j)
#pragma omp simd
          for (int i = il; i <= iu + 1; ++i)
            x1f(k, jl - j, i) = x1f(k, jl, i);
      // x2f (normal): copy from boundary face at jl
      for (int k = kl; k <= ku; ++k)
        for (int j = 1; j <= ngh; ++j)
#pragma omp simd
          for (int i = il; i <= iu; ++i)
            x2f(k, jl - j, i) = x2f(k, jl, i);
      // x3f (tangential): copy from boundary cell at jl, k-range +1
      for (int k = kl; k <= ku + 1; ++k)
        for (int j = 1; j <= ngh; ++j)
#pragma omp simd
          for (int i = il; i <= iu; ++i)
            x3f(k, jl - j, i) = x3f(k, jl, i);
      break;

    case BoundaryFace::outer_x2:
      // x1f (tangential): copy from boundary cell at ju, i-range +1
      for (int k = kl; k <= ku; ++k)
        for (int j = 1; j <= ngh; ++j)
#pragma omp simd
          for (int i = il; i <= iu + 1; ++i)
            x1f(k, ju + j, i) = x1f(k, ju, i);
      // x2f (normal): copy from boundary face at ju+1, dst = ju+j+1
      for (int k = kl; k <= ku; ++k)
        for (int j = 1; j <= ngh; ++j)
#pragma omp simd
          for (int i = il; i <= iu; ++i)
            x2f(k, ju + j + 1, i) = x2f(k, ju + 1, i);
      // x3f (tangential): copy from boundary cell at ju, k-range +1
      for (int k = kl; k <= ku + 1; ++k)
        for (int j = 1; j <= ngh; ++j)
#pragma omp simd
          for (int i = il; i <= iu; ++i)
            x3f(k, ju + j, i) = x3f(k, ju, i);
      break;

    case BoundaryFace::inner_x3:
      // x1f (tangential): copy from boundary cell at kl, i-range +1
      for (int k = 1; k <= ngh; ++k)
        for (int j = jl; j <= ju; ++j)
#pragma omp simd
          for (int i = il; i <= iu + 1; ++i)
            x1f(kl - k, j, i) = x1f(kl, j, i);
      // x2f (tangential): copy from boundary cell at kl, j-range +1
      for (int k = 1; k <= ngh; ++k)
        for (int j = jl; j <= ju + 1; ++j)
#pragma omp simd
          for (int i = il; i <= iu; ++i)
            x2f(kl - k, j, i) = x2f(kl, j, i);
      // x3f (normal): copy from boundary face at kl
      for (int k = 1; k <= ngh; ++k)
        for (int j = jl; j <= ju; ++j)
#pragma omp simd
          for (int i = il; i <= iu; ++i)
            x3f(kl - k, j, i) = x3f(kl, j, i);
      break;

    case BoundaryFace::outer_x3:
      // x1f (tangential): copy from boundary cell at ku, i-range +1
      for (int k = 1; k <= ngh; ++k)
        for (int j = jl; j <= ju; ++j)
#pragma omp simd
          for (int i = il; i <= iu + 1; ++i)
            x1f(ku + k, j, i) = x1f(ku, j, i);
      // x2f (tangential): copy from boundary cell at ku, j-range +1
      for (int k = 1; k <= ngh; ++k)
        for (int j = jl; j <= ju + 1; ++j)
#pragma omp simd
          for (int i = il; i <= iu; ++i)
            x2f(ku + k, j, i) = x2f(ku, j, i);
      // x3f (normal): copy from boundary face at ku+1, dst = ku+k+1
      for (int k = 1; k <= ngh; ++k)
        for (int j = jl; j <= ju; ++j)
#pragma omp simd
          for (int i = il; i <= iu; ++i)
            x3f(ku + k + 1, j, i) = x3f(ku + 1, j, i);
      break;

    default:
      break;
  }
}

//----------------------------------------------------------------------------------------
// PolarWedgeBC_FC: mirror copy across x2 boundary with per-component sign
// flips.
//
// Sign convention from flip_across_pole_field[] = {false, true, true}:
//   B1 (x1f): sign = +1   (r-component unchanged across pole)
//   B2 (x2f): sign = -1   (theta-component flips across pole)
//   B3 (x3f): sign = -1   (phi-component flips across pole)
//
// Additionally zeros the normal face (x2f) at the pole boundary:
//   inner x2: x2f(k, jl, i) = 0
//   outer x2: x2f(k, ju+1, i) = 0
//
// The mirror geometry is identical to ReflectBC_FC for x2, with the normal
// component using face-stagger source offset (jl+j for inner, ju-j+1 for
// outer). Only valid for inner/outer x2 faces.

void PolarWedgeBC_FC(AthenaArray<Real>* fc[3],
                     BoundaryFace face,
                     int il,
                     int iu,
                     int jl,
                     int ju,
                     int kl,
                     int ku,
                     int ngh)
{
  if (face != BoundaryFace::inner_x2 && face != BoundaryFace::outer_x2)
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in PolarWedgeBC_FC\n"
        << "PolarWedge BC is only valid on inner/outer x2 faces." << std::endl;
    ATHENA_ERROR(msg);
  }

  AthenaArray<Real>& x1f = *fc[0];
  AthenaArray<Real>& x2f = *fc[1];
  AthenaArray<Real>& x3f = *fc[2];

  // Hardcoded signs: {B1=+1, B2=-1, B3=-1}
  constexpr Real sign_b1 = 1.0;
  constexpr Real sign_b2 = -1.0;
  constexpr Real sign_b3 = -1.0;

  if (face == BoundaryFace::inner_x2)
  {
    // x1f: i-range +1, mirror with sign_b1
    for (int k = kl; k <= ku; ++k)
      for (int j = 1; j <= ngh; ++j)
#pragma omp simd
        for (int i = il; i <= iu + 1; ++i)
          x1f(k, jl - j, i) = sign_b1 * x1f(k, jl + j - 1, i);
    // x2f (normal): face-stagger source (jl+j), then zero pole face
    for (int k = kl; k <= ku; ++k)
      for (int j = 1; j <= ngh; ++j)
#pragma omp simd
        for (int i = il; i <= iu; ++i)
          x2f(k, jl - j, i) = sign_b2 * x2f(k, jl + j, i);
    // Zero normal face at pole
    for (int k = kl; k <= ku; ++k)
#pragma omp simd
      for (int i = il; i <= iu; ++i)
        x2f(k, jl, i) = 0.0;
    // x3f: k-range +1, mirror with sign_b3
    for (int k = kl; k <= ku + 1; ++k)
      for (int j = 1; j <= ngh; ++j)
#pragma omp simd
        for (int i = il; i <= iu; ++i)
          x3f(k, jl - j, i) = sign_b3 * x3f(k, jl + j - 1, i);
  }
  else
  {
    // x1f: i-range +1, mirror with sign_b1
    for (int k = kl; k <= ku; ++k)
      for (int j = 1; j <= ngh; ++j)
#pragma omp simd
        for (int i = il; i <= iu + 1; ++i)
          x1f(k, ju + j, i) = sign_b1 * x1f(k, ju - j + 1, i);
    // x2f (normal): outer face-stagger: dst = ju+j+1, src = ju-j+1
    for (int k = kl; k <= ku; ++k)
      for (int j = 1; j <= ngh; ++j)
#pragma omp simd
        for (int i = il; i <= iu; ++i)
          x2f(k, ju + j + 1, i) = sign_b2 * x2f(k, ju - j + 1, i);
    // Zero normal face at pole
    for (int k = kl; k <= ku; ++k)
#pragma omp simd
      for (int i = il; i <= iu; ++i)
        x2f(k, ju + 1, i) = 0.0;
    // x3f: k-range +1, mirror with sign_b3
    for (int k = kl; k <= ku + 1; ++k)
      for (int j = 1; j <= ngh; ++j)
#pragma omp simd
        for (int i = il; i <= iu; ++i)
          x3f(k, ju + j, i) = sign_b3 * x3f(k, ju - j + 1, i);
  }
}

//========================================================================================
// FC dispatch
//========================================================================================

//----------------------------------------------------------------------------------------
// Single-face dispatch for FC variables.
// GRSommerfeld delegates to OutflowBC_FC (confirmed from old
// gr_sommerfeld_fc.cpp). ExtrapolateOutflow and User are not implemented for
// FC - runtime error.

void ApplyPhysicalBCFace_FC(AthenaArray<Real>* fc[3],
                            BoundaryFace face,
                            int il,
                            int iu,
                            int jl,
                            int ju,
                            int kl,
                            int ku,
                            int ngh,
                            PhysicalBC bc)
{
  switch (bc)
  {
    case PhysicalBC::Reflect:
      ReflectBC_FC(fc, face, il, iu, jl, ju, kl, ku, ngh);
      break;
    case PhysicalBC::Outflow:
      OutflowBC_FC(fc, face, il, iu, jl, ju, kl, ku, ngh);
      break;
    case PhysicalBC::GRSommerfeld:
      // FC GRSommerfeld always delegates to Outflow (no extrapolation
      // variant).
      OutflowBC_FC(fc, face, il, iu, jl, ju, kl, ku, ngh);
      break;
    case PhysicalBC::PolarWedge:
      PolarWedgeBC_FC(fc, face, il, iu, jl, ju, kl, ku, ngh);
      break;
    case PhysicalBC::ExtrapolateOutflow:
    {
      std::stringstream msg;
      msg
        << "### FATAL ERROR in ApplyPhysicalBCFace_FC\n"
        << "ExtrapolateOutflow is not implemented for face-centered variables."
        << std::endl;
      ATHENA_ERROR(msg);
      break;
    }
    case PhysicalBC::User:
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in ApplyPhysicalBCFace_FC\n"
          << "User BCs are not supported for face-centered variables."
          << std::endl;
      ATHENA_ERROR(msg);
      break;
    }
    case PhysicalBC::None:
      break;
  }
}

//----------------------------------------------------------------------------------------
// Multi-face dispatch for FC variables, fine level.
// Same face ordering and transverse-limit extension logic as CC
// ApplyPhysicalBCs, but operates on FaceField and uses FC-specific dispatch.

void ApplyPhysicalBCs_FC(MeshBlock* pmb,
                         const CommSpec& spec,
                         Real time,
                         Real dt)
{
  if (!spec.HasAnyPhysicalBC())
    return;
  // FC data stored as three independent AthenaArray pointers; null check on
  // first.
  if (spec.var_fc[0] == nullptr)
    return;

  AthenaArray<Real>* fc[3] = { spec.var_fc[0],
                               spec.var_fc[1],
                               spec.var_fc[2] };
  const int ng             = spec.nghost;

  const int is = pmb->is, ie = pmb->ie;
  const int js = pmb->js, je = pmb->je;
  const int ks = pmb->ks, ke = pmb->ke;

  // Transverse limits, extended for periodic faces.
  int bis = is - ng, bie = ie + ng;
  int bjs = js, bje = je;
  int bks = ks, bke = ke;

  if (!FaceNeedsBC(pmb, BoundaryFace::inner_x2) && pmb->block_size.nx2 > 1)
    bjs = js - ng;
  if (!FaceNeedsBC(pmb, BoundaryFace::outer_x2) && pmb->block_size.nx2 > 1)
    bje = je + ng;
  if (!FaceNeedsBC(pmb, BoundaryFace::inner_x3) && pmb->block_size.nx3 > 1)
    bks = ks - ng;
  if (!FaceNeedsBC(pmb, BoundaryFace::outer_x3) && pmb->block_size.nx3 > 1)
    bke = ke + ng;

  // X1 faces.
  if (FaceNeedsBC(pmb, BoundaryFace::inner_x1))
  {
    PhysicalBC bc = spec.physical_bc[static_cast<int>(BoundaryFace::inner_x1)];
    ApplyPhysicalBCFace_FC(
      fc, BoundaryFace::inner_x1, is, ie, bjs, bje, bks, bke, ng, bc);
  }
  if (FaceNeedsBC(pmb, BoundaryFace::outer_x1))
  {
    PhysicalBC bc = spec.physical_bc[static_cast<int>(BoundaryFace::outer_x1)];
    ApplyPhysicalBCFace_FC(
      fc, BoundaryFace::outer_x1, is, ie, bjs, bje, bks, bke, ng, bc);
  }

  // X2 faces (2D+).
  if (pmb->block_size.nx2 > 1)
  {
    if (FaceNeedsBC(pmb, BoundaryFace::inner_x2))
    {
      PhysicalBC bc =
        spec.physical_bc[static_cast<int>(BoundaryFace::inner_x2)];
      ApplyPhysicalBCFace_FC(
        fc, BoundaryFace::inner_x2, bis, bie, js, je, bks, bke, ng, bc);
    }
    if (FaceNeedsBC(pmb, BoundaryFace::outer_x2))
    {
      PhysicalBC bc =
        spec.physical_bc[static_cast<int>(BoundaryFace::outer_x2)];
      ApplyPhysicalBCFace_FC(
        fc, BoundaryFace::outer_x2, bis, bie, js, je, bks, bke, ng, bc);
    }
  }

  // X3 faces (3D). After x2, extend j-range to include x2 ghost zones.
  if (pmb->block_size.nx3 > 1)
  {
    bjs = js - ng;
    bje = je + ng;

    if (FaceNeedsBC(pmb, BoundaryFace::inner_x3))
    {
      PhysicalBC bc =
        spec.physical_bc[static_cast<int>(BoundaryFace::inner_x3)];
      ApplyPhysicalBCFace_FC(
        fc, BoundaryFace::inner_x3, bis, bie, bjs, bje, ks, ke, ng, bc);
    }
    if (FaceNeedsBC(pmb, BoundaryFace::outer_x3))
    {
      PhysicalBC bc =
        spec.physical_bc[static_cast<int>(BoundaryFace::outer_x3)];
      ApplyPhysicalBCFace_FC(
        fc, BoundaryFace::outer_x3, bis, bie, bjs, bje, ks, ke, ng, bc);
    }
  }
}

//----------------------------------------------------------------------------------------
// Multi-face dispatch for FC variables, coarse level (for prolongation).
// Operates on spec.coarse_fc with coarse-level bounds.

void ApplyPhysicalBCsOnCoarseLevel_FC(MeshBlock* pmb,
                                      const CommSpec& spec,
                                      Real time,
                                      Real dt,
                                      int cis,
                                      int cie,
                                      int cjs,
                                      int cje,
                                      int cks,
                                      int cke,
                                      int cng)
{
  if (!spec.HasAnyPhysicalBC())
    return;
  // FC coarse data stored as three independent AthenaArray pointers; null
  // check on first.
  if (spec.coarse_fc[0] == nullptr)
    return;

  AthenaArray<Real>* cfc[3] = { spec.coarse_fc[0],
                                spec.coarse_fc[1],
                                spec.coarse_fc[2] };

  // Transverse limit extension for periodic faces.
  int bis = cis - cng, bie = cie + cng;
  int bjs = cjs, bje = cje;
  int bks = cks, bke = cke;

  if (!FaceNeedsBC(pmb, BoundaryFace::inner_x2) && pmb->block_size.nx2 > 1)
    bjs = cjs - cng;
  if (!FaceNeedsBC(pmb, BoundaryFace::outer_x2) && pmb->block_size.nx2 > 1)
    bje = cje + cng;
  if (!FaceNeedsBC(pmb, BoundaryFace::inner_x3) && pmb->block_size.nx3 > 1)
    bks = cks - cng;
  if (!FaceNeedsBC(pmb, BoundaryFace::outer_x3) && pmb->block_size.nx3 > 1)
    bke = cke + cng;

  // X1 faces.
  if (FaceNeedsBC(pmb, BoundaryFace::inner_x1))
  {
    PhysicalBC bc = spec.physical_bc[static_cast<int>(BoundaryFace::inner_x1)];
    ApplyPhysicalBCFace_FC(
      cfc, BoundaryFace::inner_x1, cis, cie, bjs, bje, bks, bke, cng, bc);
  }
  if (FaceNeedsBC(pmb, BoundaryFace::outer_x1))
  {
    PhysicalBC bc = spec.physical_bc[static_cast<int>(BoundaryFace::outer_x1)];
    ApplyPhysicalBCFace_FC(
      cfc, BoundaryFace::outer_x1, cis, cie, bjs, bje, bks, bke, cng, bc);
  }

  // X2 faces.
  if (pmb->block_size.nx2 > 1)
  {
    if (FaceNeedsBC(pmb, BoundaryFace::inner_x2))
    {
      PhysicalBC bc =
        spec.physical_bc[static_cast<int>(BoundaryFace::inner_x2)];
      ApplyPhysicalBCFace_FC(
        cfc, BoundaryFace::inner_x2, bis, bie, cjs, cje, bks, bke, cng, bc);
    }
    if (FaceNeedsBC(pmb, BoundaryFace::outer_x2))
    {
      PhysicalBC bc =
        spec.physical_bc[static_cast<int>(BoundaryFace::outer_x2)];
      ApplyPhysicalBCFace_FC(
        cfc, BoundaryFace::outer_x2, bis, bie, cjs, cje, bks, bke, cng, bc);
    }
  }

  // X3 faces.
  if (pmb->block_size.nx3 > 1)
  {
    bjs = cjs - cng;
    bje = cje + cng;

    if (FaceNeedsBC(pmb, BoundaryFace::inner_x3))
    {
      PhysicalBC bc =
        spec.physical_bc[static_cast<int>(BoundaryFace::inner_x3)];
      ApplyPhysicalBCFace_FC(
        cfc, BoundaryFace::inner_x3, bis, bie, bjs, bje, cks, cke, cng, bc);
    }
    if (FaceNeedsBC(pmb, BoundaryFace::outer_x3))
    {
      PhysicalBC bc =
        spec.physical_bc[static_cast<int>(BoundaryFace::outer_x3)];
      ApplyPhysicalBCFace_FC(
        cfc, BoundaryFace::outer_x3, bis, bie, bjs, bje, cks, cke, cng, bc);
    }
  }
}

}  // namespace comm
