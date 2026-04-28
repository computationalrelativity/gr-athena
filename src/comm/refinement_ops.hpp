#ifndef COMM_REFINEMENT_OPS_HPP_
#define COMM_REFINEMENT_OPS_HPP_
//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file refinement_ops.hpp
//  \brief Prolongation and restriction dispatch for the data-driven comm system.
//
//  Replaces the per-subclass ProlongateBoundaries() / RestrictNonGhost() virtual calls
//  with free functions dispatched by ProlongOp / RestrictOp enums.
//
//  Prolongation:
//    For each neighbor at a coarser level, compute the coarse-grid index range that
//    needs interpolation, then call the appropriate MeshRefinement operator to fill
//    the fine-level ghost zones from the coarse buffer.
//
//  Restriction:
//    Pre-restrict the entire fine interior into the coarse buffer so that same-level
//    neighbors can receive the coarse payload.

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "comm_enums.hpp"
#include "index_utilities.hpp"  // IndexRange3D

// forward declarations
class MeshBlock;
struct NeighborBlock;
struct NeighborIndexes;

namespace comm {

struct CommSpec;
class NeighborConnectivity;

//----------------------------------------------------------------------------------------
// Per-dimension prolongation index helper.
// Identical logic for CC, CX, VC (FC handled separately).
// Determines which coarse-grid cells need prolongation for one dimension.
//
//   lx     - block's logical coordinate in this dimension
//   ox     - neighbor offset (-1, 0, +1)
//   pcng   - prolongation coarse ghost width (cnghost-1 for CC, ng/2+ceil for CX/VC)
//   cvs/cve - coarse interior start/end indices
//   svs/sve - output: prolongation start/end in coarse coords
//   active  - whether this dimension is non-trivial (nx > 1)

inline void ProlongationRange(std::int64_t lx, int ox, int pcng,
                              int cvs, int cve,
                              int &svs, int &sve,
                              bool active) {
  if (ox > 0) {
    svs = cve + 1;
    sve = cve + pcng;
  } else if (ox < 0) {
    svs = cvs - pcng;
    sve = cvs - 1;
  } else {
    svs = cvs;
    sve = cve;
    if (active) {
      // For ox==0, the prolongation region extends into the half of the block
      // that borders the coarser neighbor.  Which half depends on whether this
      // block is the left or right child (even/odd logical coordinate).
      if ((lx & 1LL) == 0LL) {
        sve += pcng;
      } else {
        svs -= pcng;
      }
    }
  }
}

//----------------------------------------------------------------------------------------
// Compute the 3D coarse-grid prolongation index range for one neighbor.
// Returns coarse-level indices.  The caller passes these to MeshRefinement::Prolongate*.

idx::IndexRange3D ProlongationIndices(const MeshBlock *pmb,
                                      const NeighborBlock &nb,
                                      Sampling sampling);

//----------------------------------------------------------------------------------------
// Prolongate one neighbor's ghost zone slab from coarse_buf into the fine variable.
// Dispatches to the correct MeshRefinement operator based on ProlongOp.
// For FC, use ProlongateNeighborFC instead.

void ProlongateNeighbor(MeshBlock *pmb,
                        const CommSpec &spec,
                        const idx::IndexRange3D &r);

//----------------------------------------------------------------------------------------
// Compute the shared-face extended index range for FC prolongation.
// Extends the base range by +1 in each non-degenerate dimension, then trims shared
// faces where a same-or-finer-level neighbor exists (preventing overwrite of data
// that was already set by normal boundary communication).

idx::IndexRange3D ProlongationSharedIndices(const MeshBlock *pmb,
                                            const NeighborBlock &nb,
                                            const idx::IndexRange3D &base);

//----------------------------------------------------------------------------------------
// FC prolongation: 4-step Toth & Roe divergence-preserving interpolation.
// 1. ProlongateSharedFieldX1 on shared i-range (il..iu)
// 2. ProlongateSharedFieldX2 on shared j-range (jl..ju)
// 3. ProlongateSharedFieldX3 on shared k-range (kl..ku)
// 4. ProlongateInternalField on base range (si..ei, sj..ej, sk..ek)

void ProlongateNeighborFC(MeshBlock *pmb,
                          const CommSpec &spec,
                          const idx::IndexRange3D &base,
                          const idx::IndexRange3D &shared);

//----------------------------------------------------------------------------------------
// Prolongate all coarser neighbors' ghost zones for a single channel.
// This is the top-level call that replaces BoundaryVariable::ProlongateBoundaries().

void ProlongateBoundaries(MeshBlock *pmb,
                          const CommSpec &spec,
                          const NeighborConnectivity &nc);

//----------------------------------------------------------------------------------------
// Restrict the fine interior into the coarse buffer.
// Replaces BoundaryVariable::RestrictNonGhost().
// Called before PackAndSend so that coarse payload is available.

void RestrictInterior(MeshBlock *pmb, const CommSpec &spec);

//----------------------------------------------------------------------------------------
// Convert prolongation coarse-grid indices to fine-grid indices.
// Needed by physics modules that do post-prolongation work (e.g. PrimitiveToConserved
// on prolonged boundary slabs).

idx::IndexRange3D ProlongationIndicesFine(const MeshBlock *pmb,
                                          const idx::IndexRange3D &coarse_r,
                                          Sampling sampling);

} // namespace comm

#endif // COMM_REFINEMENT_OPS_HPP_
