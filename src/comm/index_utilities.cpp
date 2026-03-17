//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file index_utilities.cpp
//  \brief Implementation of 3D index range functions for pack/unpack.
//
//  Each function assembles three per-dimension calls into an IndexRange3D.
//  The per-dimension helpers (LoadRangeCC, SetRangeCC, etc.) are inlined from the header.
//
//  Ghost widths by sampling:
//    CC:
//      - Fine level: nghost (== NGHOST).
//      - Coarse level: pmb->cnghost == (NGHOST+1)/2+1.
//      - LoadToCoarser uses nghost as slab width, indices in coarse buffer (cis..cie).
//      - LoadToFiner uses cnghost as slab width, indices in fine array (is..ie).
//    VC:
//      - Fine level: pmb->ng (== NGHOST).
//      - Coarse level: pmb->cng (== NCGHOST).
//      - LoadToCoarser uses pmb->ng as slab width, indices in coarse buffer (civs..cive).
//      - LoadToFiner uses pmb->cng as slab width, indices in fine array (ivs..ive).
//      - Shared vertex included in load/set; unpack is additive (handled by caller).
//    CX:
//      - Fine level: NGHOST.
//      - Coarse level: NCGHOST_CX.
//      - LoadToCoarser uses cx_dng offset from cx_cige/cx_cigs.
//      - LoadToFiner uses cx_dng offset + NCGHOST_CX as half_size.
//      - Set has a gap at boundary (ie+1/is-1), like CC.

#include "index_utilities.hpp"

#include <algorithm>  // std::max
#include <sstream>    // std::stringstream

#include "../defs.hpp"  // NCGHOST_CX, ATHENA_ERROR
#include "../mesh/mesh.hpp"  // MeshBlock, LogicalLocation, RegionSize

namespace comm {
namespace idx {

//========================================================================================
// LoadSameLevel: pack source indices for same-level communication.
//   is_coarse=false -> fine array, is_coarse=true -> coarse buffer.
//   CC: arithmetic offset from is/ie by nghost.
//   VC: precomputed ghost-load boundaries (ige/igs), includes shared vertex.
//   CX: precomputed ghost-load boundaries (cx_ige/cx_igs), no shared vertex.
//========================================================================================

IndexRange3D LoadSameLevel(const MeshBlock *pmb, const NeighborIndexes &ni,
                           int nghost, bool is_coarse, Sampling sampling,
                           int stagger_axis) {
  IndexRange3D r;

  switch (sampling) {
  case Sampling::CC:
    if (!is_coarse) {
      LoadRangeCC(ni.ox1, r.si, r.ei, pmb->is, pmb->ie, nghost);
      LoadRangeCC(ni.ox2, r.sj, r.ej, pmb->js, pmb->je, nghost);
      LoadRangeCC(ni.ox3, r.sk, r.ek, pmb->ks, pmb->ke, nghost);
    } else {
      int cng = pmb->cnghost;
      LoadRangeCC(ni.ox1, r.si, r.ei, pmb->cis, pmb->cie, cng);
      LoadRangeCC(ni.ox2, r.sj, r.ej, pmb->cjs, pmb->cje, cng);
      LoadRangeCC(ni.ox3, r.sk, r.ek, pmb->cks, pmb->cke, cng);
    }
    break;

  case Sampling::VC:
    if (!is_coarse) {
      // VC includes shared vertex at boundary in loads
      LoadRangeVC(ni.ox1, r.si, r.ei, pmb->ivs, pmb->ive, pmb->igs, pmb->ige);
      LoadRangeVC(ni.ox2, r.sj, r.ej, pmb->jvs, pmb->jve, pmb->jgs, pmb->jge);
      LoadRangeVC(ni.ox3, r.sk, r.ek, pmb->kvs, pmb->kve, pmb->kgs, pmb->kge);
    } else {
      // Coarse VC buffer uses coarse ghost-load indices
      LoadRangeVC(ni.ox1, r.si, r.ei,
                  pmb->civs, pmb->cive, pmb->cigs, pmb->cige);
      LoadRangeVC(ni.ox2, r.sj, r.ej,
                  pmb->cjvs, pmb->cjve, pmb->cjgs, pmb->cjge);
      LoadRangeVC(ni.ox3, r.sk, r.ek,
                  pmb->ckvs, pmb->ckve, pmb->ckgs, pmb->ckge);
    }
    break;

  case Sampling::CX:
    if (!is_coarse) {
      LoadRangeCX(ni.ox1, r.si, r.ei,
                  pmb->cx_is, pmb->cx_ie, pmb->cx_igs, pmb->cx_ige);
      LoadRangeCX(ni.ox2, r.sj, r.ej,
                  pmb->cx_js, pmb->cx_je, pmb->cx_jgs, pmb->cx_jge);
      LoadRangeCX(ni.ox3, r.sk, r.ek,
                  pmb->cx_ks, pmb->cx_ke, pmb->cx_kgs, pmb->cx_kge);
    } else {
      // Coarse CX buffer uses coarse CX ghost-load indices
      LoadRangeCX(ni.ox1, r.si, r.ei,
                  pmb->cx_cis, pmb->cx_cie, pmb->cx_cigs, pmb->cx_cige);
      LoadRangeCX(ni.ox2, r.sj, r.ej,
                  pmb->cx_cjs, pmb->cx_cje, pmb->cx_cjgs, pmb->cx_cjge);
      LoadRangeCX(ni.ox3, r.sk, r.ek,
                  pmb->cx_cks, pmb->cx_cke, pmb->cx_ckgs, pmb->cx_ckge);
    }
    break;

  case Sampling::FC:
    if (!is_coarse) {
      // Fine grid: per-dimension helper selects stagger vs CC logic
      int ng = nghost;
      LoadRangeFC(ni.ox1, r.si, r.ei, pmb->is, pmb->ie, ng,
                  stagger_axis == 0, true);
      LoadRangeFC(ni.ox2, r.sj, r.ej, pmb->js, pmb->je, ng,
                  stagger_axis == 1, pmb->block_size.nx2 > 1);
      LoadRangeFC(ni.ox3, r.sk, r.ek, pmb->ks, pmb->ke, ng,
                  stagger_axis == 2, pmb->block_size.nx3 > 1);
      // Same-level edge/corner overlap is disabled for FC - overwrites EMF-corrected
      // face values.  See old bvals_fc.cpp for rationale.
    } else {
      // Coarse same-level payload: uses cnghost with degenerate-dimension scaling
      int cng = pmb->cnghost;
      int f2 = (pmb->block_size.nx2 > 1) ? 1 : 0;
      int f3 = (pmb->block_size.nx3 > 1) ? 1 : 0;
      LoadRangeFC(ni.ox1, r.si, r.ei, pmb->cis, pmb->cie, cng,
                  stagger_axis == 0, true);
      LoadRangeFC(ni.ox2, r.sj, r.ej, pmb->cjs, pmb->cje, cng * f2,
                  stagger_axis == 1, pmb->block_size.nx2 > 1);
      LoadRangeFC(ni.ox3, r.sk, r.ek, pmb->cks, pmb->cke, cng * f3,
                  stagger_axis == 2, pmb->block_size.nx3 > 1);
    }
    break;

  default: {
    std::stringstream msg;
    msg << "### FATAL ERROR in LoadSameLevel: unsupported Sampling type" << std::endl;
    ATHENA_ERROR(msg);
  }
  }

  return r;
}

//========================================================================================
// SetSameLevel: unpack destination indices for same-level communication.
//   type=1 -> fine array ghost zones, type=2 -> coarse buffer.
//   CC: ghost zone = [ve+1, ve+ng] / [vs-ng, vs-1].
//   VC: includes shared vertex in set range (additive unpack done by caller).
//   CX: gap at boundary [ie+1, pe] / [ms, is-1].
//========================================================================================

IndexRange3D SetSameLevel(const MeshBlock *pmb, const NeighborIndexes &ni,
                          int nghost, int type, Sampling sampling,
                          int stagger_axis) {
  IndexRange3D r;

  switch (sampling) {
  case Sampling::CC:
    if (type == 1) {
      SetRangeCC(ni.ox1, r.si, r.ei, pmb->is, pmb->ie, nghost);
      SetRangeCC(ni.ox2, r.sj, r.ej, pmb->js, pmb->je, nghost);
      SetRangeCC(ni.ox3, r.sk, r.ek, pmb->ks, pmb->ke, nghost);
    } else {
      int cng = pmb->cnghost;
      SetRangeCC(ni.ox1, r.si, r.ei, pmb->cis, pmb->cie, cng);
      SetRangeCC(ni.ox2, r.sj, r.ej, pmb->cjs, pmb->cje, cng);
      SetRangeCC(ni.ox3, r.sk, r.ek, pmb->cks, pmb->cke, cng);
    }
    break;

  case Sampling::VC:
    if (type == 1) {
      // VC set includes shared vertex: ox>0 -> [ve, pe], ox<0 -> [ms, vs]
      SetRangeVC(ni.ox1, r.si, r.ei,
                 pmb->ivs, pmb->ive, pmb->ims, pmb->ipe);
      SetRangeVC(ni.ox2, r.sj, r.ej,
                 pmb->jvs, pmb->jve, pmb->jms, pmb->jpe);
      SetRangeVC(ni.ox3, r.sk, r.ek,
                 pmb->kvs, pmb->kve, pmb->kms, pmb->kpe);
    } else {
      // Coarse VC buffer
      SetRangeVC(ni.ox1, r.si, r.ei,
                 pmb->civs, pmb->cive, pmb->cims, pmb->cipe);
      SetRangeVC(ni.ox2, r.sj, r.ej,
                 pmb->cjvs, pmb->cjve, pmb->cjms, pmb->cjpe);
      SetRangeVC(ni.ox3, r.sk, r.ek,
                 pmb->ckvs, pmb->ckve, pmb->ckms, pmb->ckpe);
    }
    break;

  case Sampling::CX:
    if (type == 1) {
      // CX set has gap: ox>0 -> [ie+1, pe], ox<0 -> [ms, is-1]
      SetRangeCX(ni.ox1, r.si, r.ei,
                 pmb->cx_is, pmb->cx_ie, pmb->cx_ims, pmb->cx_ipe);
      SetRangeCX(ni.ox2, r.sj, r.ej,
                 pmb->cx_js, pmb->cx_je, pmb->cx_jms, pmb->cx_jpe);
      SetRangeCX(ni.ox3, r.sk, r.ek,
                 pmb->cx_ks, pmb->cx_ke, pmb->cx_kms, pmb->cx_kpe);
    } else {
      // Coarse CX buffer
      SetRangeCX(ni.ox1, r.si, r.ei,
                 pmb->cx_cis, pmb->cx_cie, pmb->cx_cims, pmb->cx_cipe);
      SetRangeCX(ni.ox2, r.sj, r.ej,
                 pmb->cx_cjs, pmb->cx_cje, pmb->cx_cjms, pmb->cx_cjpe);
      SetRangeCX(ni.ox3, r.sk, r.ek,
                 pmb->cx_cks, pmb->cx_cke, pmb->cx_ckms, pmb->cx_ckpe);
    }
    break;

  case Sampling::FC:
    if (type == 1) {
      // Fine grid: stagger dim skips shared boundary face on ox>0
      int ng = nghost;
      SetRangeFC(ni.ox1, r.si, r.ei, pmb->is, pmb->ie, ng,
                 stagger_axis == 0, true);
      SetRangeFC(ni.ox2, r.sj, r.ej, pmb->js, pmb->je, ng,
                 stagger_axis == 1, pmb->block_size.nx2 > 1);
      SetRangeFC(ni.ox3, r.sk, r.ek, pmb->ks, pmb->ke, ng,
                 stagger_axis == 2, pmb->block_size.nx3 > 1);
      // Same-level edge/corner overlap disabled for FC (see LoadSameLevel).
    } else {
      // Coarse same-level payload: degenerate-dimension ghost scaling
      int cng = pmb->cnghost;
      int f2 = (pmb->block_size.nx2 > 1) ? 1 : 0;
      int f3 = (pmb->block_size.nx3 > 1) ? 1 : 0;
      SetRangeFC(ni.ox1, r.si, r.ei, pmb->cis, pmb->cie, cng,
                 stagger_axis == 0, true);
      SetRangeFC(ni.ox2, r.sj, r.ej, pmb->cjs, pmb->cje, cng * f2,
                 stagger_axis == 1, pmb->block_size.nx2 > 1);
      SetRangeFC(ni.ox3, r.sk, r.ek, pmb->cks, pmb->cke, cng * f3,
                 stagger_axis == 2, pmb->block_size.nx3 > 1);
    }
    break;

  default: {
    std::stringstream msg;
    msg << "### FATAL ERROR in SetSameLevel: unsupported Sampling type" << std::endl;
    ATHENA_ERROR(msg);
  }
  }

  return r;
}

//========================================================================================
// LoadToCoarser: pack source indices from the coarse buffer for sending to a coarser
// neighbor.
//   CC: uses nghost as slab width from coarse buffer (cis..cie).
//   VC: uses pmb->ng as slab width from coarse VC buffer (civs..cive).
//   CX: uses cx_dng offset from coarse CX ghost-load boundaries.
//========================================================================================

IndexRange3D LoadToCoarser(const MeshBlock *pmb, const NeighborIndexes &ni,
                           int nghost, Sampling sampling,
                           int stagger_axis) {
  IndexRange3D r;

  switch (sampling) {
  case Sampling::CC:
    // CC packs from coarse buffer with fine-level ghost width
    LoadRangeCC(ni.ox1, r.si, r.ei, pmb->cis, pmb->cie, nghost);
    LoadRangeCC(ni.ox2, r.sj, r.ej, pmb->cjs, pmb->cje, nghost);
    LoadRangeCC(ni.ox3, r.sk, r.ek, pmb->cks, pmb->cke, nghost);
    break;

  case Sampling::VC: {
    // VC packs from coarse VC buffer with fine-level ghost width pmb->ng.
    // Includes shared vertex - slab width is ng+1 (not ng like CC).
    int ng = pmb->ng;
    LoadToCoarserRangeVC(ni.ox1, r.si, r.ei, pmb->civs, pmb->cive, ng);
    LoadToCoarserRangeVC(ni.ox2, r.sj, r.ej, pmb->cjvs, pmb->cjve, ng);
    LoadToCoarserRangeVC(ni.ox3, r.sk, r.ek, pmb->ckvs, pmb->ckve, ng);
    break;
  }

  case Sampling::CX: {
    // CX accounts for ghost-width mismatch between levels via cx_dng
    int dng = pmb->cx_dng;
    LoadToCoarserRangeCX(ni.ox1, r.si, r.ei,
                         pmb->cx_cis, pmb->cx_cie, pmb->cx_cigs, pmb->cx_cige, dng);
    LoadToCoarserRangeCX(ni.ox2, r.sj, r.ej,
                         pmb->cx_cjs, pmb->cx_cje, pmb->cx_cjgs, pmb->cx_cjge, dng);
    LoadToCoarserRangeCX(ni.ox3, r.sk, r.ek,
                         pmb->cx_cks, pmb->cx_cke, pmb->cx_ckgs, pmb->cx_ckge, dng);
    break;
  }

  case Sampling::FC: {
    // FC packs from coarse buffer with NGHOST slab width, then applies
    // edge/corner overlap: re-includes boundary face for non-face neighbors.
    int cng = nghost;
    LoadRangeFC(ni.ox1, r.si, r.ei, pmb->cis, pmb->cie, cng,
                stagger_axis == 0, true);
    LoadRangeFC(ni.ox2, r.sj, r.ej, pmb->cjs, pmb->cje, cng,
                stagger_axis == 1, pmb->block_size.nx2 > 1);
    LoadRangeFC(ni.ox3, r.sk, r.ek, pmb->cks, pmb->cke, cng,
                stagger_axis == 2, pmb->block_size.nx3 > 1);
    // Edge/corner overlap: re-include boundary face for non-face neighbors
    if (ni.type != NeighborConnect::face) {
      if (stagger_axis == 0) {
        if (ni.ox1 > 0) r.ei++;
        else if (ni.ox1 < 0) r.si--;
      } else if (stagger_axis == 1) {
        if (ni.ox2 > 0) r.ej++;
        else if (ni.ox2 < 0) r.sj--;
      } else {
        if (ni.ox3 > 0) r.ek++;
        else if (ni.ox3 < 0) r.sk--;
      }
    }
    break;
  }

  default: {
    std::stringstream msg;
    msg << "### FATAL ERROR in LoadToCoarser: unsupported Sampling type" << std::endl;
    ATHENA_ERROR(msg);
  }
  }

  return r;
}

//========================================================================================
// SetFromCoarser: unpack destination indices into the coarse buffer when receiving from
// a coarser neighbor.
//   CC: uses cnghost as ghost extension with lx-parity.
//   VC: uses precomputed coarse ghost ranges (cims/cime/cips/cipe) with lx-parity.
//   CX: uses precomputed coarse CX ghost ranges (cx_cims/etc.) with lx-parity.
//========================================================================================

IndexRange3D SetFromCoarser(const MeshBlock *pmb, const NeighborIndexes &ni,
                            int nghost, Sampling sampling,
                            int stagger_axis) {
  IndexRange3D r;

  switch (sampling) {
  case Sampling::CC: {
    int cng = pmb->cnghost;
    SetFromCoarserRangeCC(ni.ox1, r.si, r.ei,
                        pmb->cis, pmb->cie, cng, pmb->loc.lx1, true);
    SetFromCoarserRangeCC(ni.ox2, r.sj, r.ej,
                        pmb->cjs, pmb->cje, cng, pmb->loc.lx2,
                        pmb->block_size.nx2 > 1);
    SetFromCoarserRangeCC(ni.ox3, r.sk, r.ek,
                        pmb->cks, pmb->cke, cng, pmb->loc.lx3,
                        pmb->block_size.nx3 > 1);
    break;
  }

  case Sampling::VC:
    // VC uses precomputed coarse ghost-set ranges with lx-parity
    SetFromCoarserRangeVC(ni.ox1, r.si, r.ei,
                          pmb->civs, pmb->cive,
                          pmb->cims, pmb->cime, pmb->cips, pmb->cipe,
                          pmb->loc.lx1, true);
    SetFromCoarserRangeVC(ni.ox2, r.sj, r.ej,
                          pmb->cjvs, pmb->cjve,
                          pmb->cjms, pmb->cjme, pmb->cjps, pmb->cjpe,
                          pmb->loc.lx2, pmb->block_size.nx2 > 1);
    SetFromCoarserRangeVC(ni.ox3, r.sk, r.ek,
                          pmb->ckvs, pmb->ckve,
                          pmb->ckms, pmb->ckme, pmb->ckps, pmb->ckpe,
                          pmb->loc.lx3, pmb->block_size.nx3 > 1);
    break;

  case Sampling::CX:
    // CX uses precomputed coarse CX ghost-set ranges with lx-parity
    SetFromCoarserRangeCX(ni.ox1, r.si, r.ei,
                          pmb->cx_cis, pmb->cx_cie,
                          pmb->cx_cims, pmb->cx_cime,
                          pmb->cx_cips, pmb->cx_cipe,
                          pmb->loc.lx1, true);
    SetFromCoarserRangeCX(ni.ox2, r.sj, r.ej,
                          pmb->cx_cjs, pmb->cx_cje,
                          pmb->cx_cjms, pmb->cx_cjme,
                          pmb->cx_cjps, pmb->cx_cjpe,
                          pmb->loc.lx2, pmb->block_size.nx2 > 1);
    SetFromCoarserRangeCX(ni.ox3, r.sk, r.ek,
                          pmb->cx_cks, pmb->cx_cke,
                          pmb->cx_ckms, pmb->cx_ckme,
                          pmb->cx_ckps, pmb->cx_ckpe,
                          pmb->loc.lx3, pmb->block_size.nx3 > 1);
    break;

  case Sampling::FC: {
    // FC uses cnghost with stagger-aware per-dimension helper, then parity extend
    int cng = pmb->cnghost;
    SetFromCoarserRangeFC(ni.ox1, r.si, r.ei,
                          pmb->cis, pmb->cie, cng, pmb->loc.lx1, true,
                          stagger_axis == 0);
    SetFromCoarserRangeFC(ni.ox2, r.sj, r.ej,
                          pmb->cjs, pmb->cje, cng, pmb->loc.lx2,
                          pmb->block_size.nx2 > 1,
                          stagger_axis == 1);
    SetFromCoarserRangeFC(ni.ox3, r.sk, r.ek,
                          pmb->cks, pmb->cke, cng, pmb->loc.lx3,
                          pmb->block_size.nx3 > 1,
                          stagger_axis == 2);
    break;
  }

  default: {
    std::stringstream msg;
    msg << "### FATAL ERROR in SetFromCoarser: unsupported Sampling type" << std::endl;
    ATHENA_ERROR(msg);
  }
  }

  return r;
}

//========================================================================================
// LoadToFiner: pack source indices from the fine array for sending to a finer neighbor.
//   CC: slab width = cnghost, indices in fine array (is..ie).
//   VC: slab width = cng, indices in fine VC array (ivs..ive), excludes boundary vertex.
//   CX: uses cx_dng offset, half_size = NCGHOST_CX.
//
// fi1/fi2 assignment for ox==0:
//   x1: always fi1 (first resolved dimension).
//   x2: fi1 when ox1!=0 (x1 already selected), fi2 otherwise.
//   x3: fi1 when ox1!=0 && ox2!=0, fi2 otherwise.
//========================================================================================

IndexRange3D LoadToFiner(const MeshBlock *pmb, const NeighborIndexes &ni,
                         int nghost, Sampling sampling,
                         int stagger_axis) {
  IndexRange3D r;

  switch (sampling) {
  case Sampling::CC: {
    int cng = pmb->cnghost;
    // x1: always nontrivial, always fi1
    FinerRangeCC(ni.ox1, r.si, r.ei, pmb->is, pmb->ie, cng,
               ni.fi1, ni.fi2, pmb->block_size.nx1/2 - cng,
               true, true, /*is_load=*/true);
    // x2: fi1 when ox1 selects x1
    FinerRangeCC(ni.ox2, r.sj, r.ej, pmb->js, pmb->je, cng,
               ni.fi1, ni.fi2, pmb->block_size.nx2/2 - cng,
               pmb->block_size.nx2 > 1, ni.ox1 != 0, /*is_load=*/true);
    // x3: fi1 when both ox1 and ox2 nonzero
    FinerRangeCC(ni.ox3, r.sk, r.ek, pmb->ks, pmb->ke, cng,
               ni.fi1, ni.fi2, pmb->block_size.nx3/2 - cng,
               pmb->block_size.nx3 > 1, (ni.ox1 != 0 && ni.ox2 != 0),
               /*is_load=*/true);
    break;
  }

  case Sampling::VC: {
    // VC excludes boundary vertex for ox!=0; slab width = pmb->cng
    int cng_vc = pmb->cng;
    int hs = pmb->block_size.nx1/2 - cng_vc;  // half_size for ox==0

    LoadToFinerRangeVC(ni.ox1, r.si, r.ei, pmb->ivs, pmb->ive, cng_vc,
                       ni.fi1, ni.fi2, hs,
                       true, true);
    hs = pmb->block_size.nx2/2 - cng_vc;
    LoadToFinerRangeVC(ni.ox2, r.sj, r.ej, pmb->jvs, pmb->jve, cng_vc,
                       ni.fi1, ni.fi2, hs,
                       pmb->block_size.nx2 > 1, ni.ox1 != 0);
    hs = pmb->block_size.nx3/2 - cng_vc;
    LoadToFinerRangeVC(ni.ox3, r.sk, r.ek, pmb->kvs, pmb->kve, cng_vc,
                       ni.fi1, ni.fi2, hs,
                       pmb->block_size.nx3 > 1, (ni.ox1 != 0 && ni.ox2 != 0));
    break;
  }

  case Sampling::CX: {
    // CX uses cx_dng offset and NCGHOST_CX as half_size
    int dng = pmb->cx_dng;
    int hs_cx = NCGHOST_CX;  // half_size for CX finer ranges

    LoadToFinerRangeCX(ni.ox1, r.si, r.ei,
                       pmb->cx_is, pmb->cx_ie, pmb->cx_igs, pmb->cx_ige, dng,
                       ni.fi1, ni.fi2,
                       pmb->block_size.nx1/2 - hs_cx,
                       true, true);
    LoadToFinerRangeCX(ni.ox2, r.sj, r.ej,
                       pmb->cx_js, pmb->cx_je, pmb->cx_jgs, pmb->cx_jge, dng,
                       ni.fi1, ni.fi2,
                       pmb->block_size.nx2/2 - hs_cx,
                       pmb->block_size.nx2 > 1, ni.ox1 != 0);
    LoadToFinerRangeCX(ni.ox3, r.sk, r.ek,
                       pmb->cx_ks, pmb->cx_ke, pmb->cx_kgs, pmb->cx_kge, dng,
                       ni.fi1, ni.fi2,
                       pmb->block_size.nx3/2 - hs_cx,
                       pmb->block_size.nx3 > 1, (ni.ox1 != 0 && ni.ox2 != 0));
    break;
  }

  case Sampling::FC: {
    // FC: slab width = cnghost; stagger dim includes boundary face (span = cng+1).
    int cng = pmb->cnghost;
    int nx1 = pmb->block_size.nx1;
    int nx2 = pmb->block_size.nx2;
    int nx3 = pmb->block_size.nx3;
    // x1: always nontrivial, always fi1
    LoadToFinerRangeFC(ni.ox1, r.si, r.ei, pmb->is, pmb->ie, cng,
                       ni.fi1, ni.fi2, nx1/2 - cng,
                       true, true,
                       stagger_axis == 0);
    // x2: fi1 when ox1 selects x1
    LoadToFinerRangeFC(ni.ox2, r.sj, r.ej, pmb->js, pmb->je, cng,
                       ni.fi1, ni.fi2, nx2/2 - cng,
                       nx2 > 1, ni.ox1 != 0,
                       stagger_axis == 1);
    // x3: fi1 when both ox1 and ox2 nonzero
    LoadToFinerRangeFC(ni.ox3, r.sk, r.ek, pmb->ks, pmb->ke, cng,
                       ni.fi1, ni.fi2, nx3/2 - cng,
                       nx3 > 1, (ni.ox1 != 0 && ni.ox2 != 0),
                       stagger_axis == 2);
    break;
  }

  default: {
    std::stringstream msg;
    msg << "### FATAL ERROR in LoadToFiner: unsupported Sampling type" << std::endl;
    ATHENA_ERROR(msg);
  }
  }

  return r;
}

//========================================================================================
// SetFromFiner: unpack destination indices into the fine array when receiving from a
// finer neighbor.
//   CC: ghost width = nghost, half_size = nx/2.
//   VC: includes shared vertex, half_size = nx/2.
//   CX: gap at boundary, half_size = nx/2.
//========================================================================================

IndexRange3D SetFromFiner(const MeshBlock *pmb, const NeighborIndexes &ni,
                          int nghost, Sampling sampling,
                          int stagger_axis) {
  IndexRange3D r;

  switch (sampling) {
  case Sampling::CC:
    // x1: always nontrivial, always fi1
    FinerRangeCC(ni.ox1, r.si, r.ei, pmb->is, pmb->ie, nghost,
               ni.fi1, ni.fi2, pmb->block_size.nx1/2,
               true, true, /*is_load=*/false);
    // x2
    FinerRangeCC(ni.ox2, r.sj, r.ej, pmb->js, pmb->je, nghost,
               ni.fi1, ni.fi2, pmb->block_size.nx2/2,
               pmb->block_size.nx2 > 1, ni.ox1 != 0, /*is_load=*/false);
    // x3
    FinerRangeCC(ni.ox3, r.sk, r.ek, pmb->ks, pmb->ke, nghost,
               ni.fi1, ni.fi2, pmb->block_size.nx3/2,
               pmb->block_size.nx3 > 1, (ni.ox1 != 0 && ni.ox2 != 0),
               /*is_load=*/false);
    break;

  case Sampling::VC: {
    // VC includes shared vertex in set-from-finer (additive unpack done by caller)
    int hs = pmb->block_size.nx1/2;

    SetFromFinerRangeVC(ni.ox1, r.si, r.ei,
                        pmb->ivs, pmb->ive, pmb->ims, pmb->ipe,
                        ni.fi1, ni.fi2, hs,
                        true, true);
    hs = pmb->block_size.nx2/2;
    SetFromFinerRangeVC(ni.ox2, r.sj, r.ej,
                        pmb->jvs, pmb->jve, pmb->jms, pmb->jpe,
                        ni.fi1, ni.fi2, hs,
                        pmb->block_size.nx2 > 1, ni.ox1 != 0);
    hs = pmb->block_size.nx3/2;
    SetFromFinerRangeVC(ni.ox3, r.sk, r.ek,
                        pmb->kvs, pmb->kve, pmb->kms, pmb->kpe,
                        ni.fi1, ni.fi2, hs,
                        pmb->block_size.nx3 > 1, (ni.ox1 != 0 && ni.ox2 != 0));
    break;
  }

  case Sampling::CX: {
    // CX has gap at boundary, same pattern as CX set-same-level for ox!=0
    int hs = pmb->block_size.nx1/2;

    SetFromFinerRangeCX(ni.ox1, r.si, r.ei,
                        pmb->cx_is, pmb->cx_ie, pmb->cx_ims, pmb->cx_ipe,
                        ni.fi1, ni.fi2, hs,
                        true, true);
    hs = pmb->block_size.nx2/2;
    SetFromFinerRangeCX(ni.ox2, r.sj, r.ej,
                        pmb->cx_js, pmb->cx_je, pmb->cx_jms, pmb->cx_jpe,
                        ni.fi1, ni.fi2, hs,
                        pmb->block_size.nx2 > 1, ni.ox1 != 0);
    hs = pmb->block_size.nx3/2;
    SetFromFinerRangeCX(ni.ox3, r.sk, r.ek,
                        pmb->cx_ks, pmb->cx_ke, pmb->cx_kms, pmb->cx_kpe,
                        ni.fi1, ni.fi2, hs,
                        pmb->block_size.nx3 > 1, (ni.ox1 != 0 && ni.ox2 != 0));
    break;
  }

  case Sampling::FC: {
    // FC: ghost width = NGHOST; stagger dim skips shared face on ox>0.
    // half_size = nx/2 (SetFromFiner uses fine-level half, not cng-adjusted).
    int ng = nghost;
    int nx1 = pmb->block_size.nx1;
    int nx2 = pmb->block_size.nx2;
    int nx3 = pmb->block_size.nx3;
    // x1
    SetFromFinerRangeFC(ni.ox1, r.si, r.ei, pmb->is, pmb->ie, ng,
                        ni.fi1, ni.fi2, nx1/2,
                        true, true,
                        stagger_axis == 0);
    // Edge/corner overlap for stagger_axis==0: direction opposite from LoadToCoarser
    if (stagger_axis == 0 && ni.type != NeighborConnect::face) {
      if (ni.ox1 > 0) r.si--;
      else if (ni.ox1 < 0) r.ei++;
    }
    // x2
    SetFromFinerRangeFC(ni.ox2, r.sj, r.ej, pmb->js, pmb->je, ng,
                        ni.fi1, ni.fi2, nx2/2,
                        nx2 > 1, ni.ox1 != 0,
                        stagger_axis == 1);
    if (stagger_axis == 1 && ni.type != NeighborConnect::face) {
      if (ni.ox2 > 0) r.sj--;
      else if (ni.ox2 < 0) r.ej++;
    }
    // x3
    SetFromFinerRangeFC(ni.ox3, r.sk, r.ek, pmb->ks, pmb->ke, ng,
                        ni.fi1, ni.fi2, nx3/2,
                        nx3 > 1, (ni.ox1 != 0 && ni.ox2 != 0),
                        stagger_axis == 2);
    if (stagger_axis == 2 && ni.type != NeighborConnect::face) {
      if (ni.ox3 > 0) r.sk--;
      else if (ni.ox3 < 0) r.ek++;
    }
    break;
  }

  default: {
    std::stringstream msg;
    msg << "### FATAL ERROR in SetFromFiner: unsupported Sampling type" << std::endl;
    ATHENA_ERROR(msg);
  }
  }

  return r;
}

//========================================================================================
// ComputeBufferSizeFromRanges: compute the maximum buffer size needed for one neighbor
// by computing actual index ranges for all communication patterns.
//
// This replaces the CC-only analytical formula and works for all samplings.
// For multilevel, takes the max over same-level (fine+coarse), to-coarser, to-finer,
// from-coarser, and from-finer buffer sizes, mirroring the old per-subclass
// NeighborVariableBufferSize() logic.
//========================================================================================

// Count cells in an inclusive 3D range, scaled by nvar components.
static int CountCells(int nvar, const IndexRange3D &r) {
  int ni = r.ei - r.si + 1;
  int nj = r.ej - r.sj + 1;
  int nk = r.ek - r.sk + 1;
  // Guard against degenerate ranges (can happen in 1D/2D trivial dimensions)
  if (ni <= 0 || nj <= 0 || nk <= 0) return 0;
  return nvar * ni * nj * nk;
}

int ComputeBufferSizeFromRanges(const MeshBlock *pmb, const NeighborIndexes &ni,
                                int nvar, int nghost, Sampling sampling) {
  // On a uniform mesh the MeshBlock never populates coarse grid indices
  // (cis/cie/cjs/cje/cks/cke and their CX counterparts are left as zero).
  // Cross-level payloads (coarse same-level, to-coarser, to-finer) are never
  // packed in that case, so skip them to avoid computing bogus sizes from
  // uninitialized indices.
  const bool multilevel = pmb->pmy_mesh->multilevel;

  if (sampling == Sampling::FC) {
    // FC: sum 3 components (x1f, x2f, x3f) for each communication pattern.
    // nvar is ignored - each component contributes 1 variable.

    // Same-level: fine payload (3 components)
    int size = 0;
    for (int sa = 0; sa < 3; ++sa) {
      IndexRange3D rng = LoadSameLevel(pmb, ni, nghost, false, sampling, sa);
      size += CountCells(1, rng);
    }

    if (multilevel) {
      // Same-level: coarse payload (3 components)
      for (int sa = 0; sa < 3; ++sa) {
        IndexRange3D rng = LoadSameLevel(pmb, ni, nghost, true, sampling, sa);
        size += CountCells(1, rng);
      }

      // To-coarser (3 components)
      int sizef2c = 0;
      for (int sa = 0; sa < 3; ++sa) {
        IndexRange3D rng = LoadToCoarser(pmb, ni, nghost, sampling, sa);
        sizef2c += CountCells(1, rng);
      }

      // To-finer (3 components)
      int sizec2f = 0;
      for (int sa = 0; sa < 3; ++sa) {
        IndexRange3D rng = LoadToFiner(pmb, ni, nghost, sampling, sa);
        sizec2f += CountCells(1, rng);
      }

      size = std::max(size, sizef2c);
      size = std::max(size, sizec2f);
    }

    return size;
  }

  // CC/VC/CX: single array, nvar components
  IndexRange3D rng = LoadSameLevel(pmb, ni, nghost, false, sampling);
  int size = CountCells(nvar, rng);

  int sizef2c = 0, sizec2f = 0;
  if (multilevel) {
    // Same-level: coarse payload (sent alongside fine for cross-level neighbors)
    IndexRange3D crng = LoadSameLevel(pmb, ni, nghost, true, sampling);
    size += CountCells(nvar, crng);

    // To-coarser (fine->coarse send)
    IndexRange3D f2crng = LoadToCoarser(pmb, ni, nghost, sampling);
    sizef2c = CountCells(nvar, f2crng);

    // To-finer (coarse->fine send)
    IndexRange3D c2frng = LoadToFiner(pmb, ni, nghost, sampling);
    sizec2f = CountCells(nvar, c2frng);
  }

  // Buffer must accommodate the largest of same-level, to-coarser, to-finer
  size = std::max(size, sizef2c);
  size = std::max(size, sizec2f);

  return size;
}

//========================================================================================
// ComputeMPIBufferSize: compute the MPI message count for one neighbor, potentially
// smaller than the buffer allocation size when the coarse payload is not needed.
//
// When skip_coarse is true and the mesh is multilevel, the same-level size omits the
// coarse payload.  Cross-level sizes (to-coarser, to-finer) are unchanged because they
// are never sent to same-level neighbors.  The final result is still the max over all
// level cases - the persistent MPI request must accommodate the largest message this
// neighbor slot can ever carry (same-level, to-coarser, or to-finer after regrid).
//
// Sender calls with skip_coarse = nb.neighbor_all_same_level (receiver's flag).
// Receiver calls with skip_coarse = pmb->NeighborBlocksSameLevel() (own flag).
//========================================================================================

int ComputeMPIBufferSize(const MeshBlock *pmb, const NeighborIndexes &ni,
                         int nvar, int nghost, Sampling sampling,
                         bool skip_coarse) {
  const bool multilevel = pmb->pmy_mesh->multilevel;

  // If not multilevel, or not skipping coarse, fall back to full allocation size.
  if (!multilevel || !skip_coarse) {
    return ComputeBufferSizeFromRanges(pmb, ni, nvar, nghost, sampling);
  }

  // Multilevel with skip_coarse: same-level size WITHOUT coarse payload,
  // still taking max over to-coarser and to-finer sizes.

  if (sampling == Sampling::FC) {
    // Same-level: fine payload only (3 components)
    int size = 0;
    for (int sa = 0; sa < 3; ++sa) {
      IndexRange3D rng = LoadSameLevel(pmb, ni, nghost, false, sampling, sa);
      size += CountCells(1, rng);
    }
    // NOTE: coarse payload intentionally omitted

    // To-coarser (3 components)
    int sizef2c = 0;
    for (int sa = 0; sa < 3; ++sa) {
      IndexRange3D rng = LoadToCoarser(pmb, ni, nghost, sampling, sa);
      sizef2c += CountCells(1, rng);
    }

    // To-finer (3 components)
    int sizec2f = 0;
    for (int sa = 0; sa < 3; ++sa) {
      IndexRange3D rng = LoadToFiner(pmb, ni, nghost, sampling, sa);
      sizec2f += CountCells(1, rng);
    }

    size = std::max(size, sizef2c);
    size = std::max(size, sizec2f);
    return size;
  }

  // CC/VC/CX: single array, nvar components
  IndexRange3D rng = LoadSameLevel(pmb, ni, nghost, false, sampling);
  int size = CountCells(nvar, rng);
  // NOTE: coarse payload intentionally omitted

  // To-coarser (fine->coarse send)
  IndexRange3D f2crng = LoadToCoarser(pmb, ni, nghost, sampling);
  int sizef2c = CountCells(nvar, f2crng);

  // To-finer (coarse->fine send)
  IndexRange3D c2frng = LoadToFiner(pmb, ni, nghost, sampling);
  int sizec2f = CountCells(nvar, c2frng);

  size = std::max(size, sizef2c);
  size = std::max(size, sizec2f);

  return size;
}

//========================================================================================
// CC flux correction buffer size.
// Each face neighbor exchanges nvar * (nx_perp1/2) * (nx_perp2/2) restricted flux values,
// where the division is integer (fine-to-coarse 2:1 restriction in the transverse plane).
// Non-face neighbors return 0 - CC flux correction is face-only.
//========================================================================================

int ComputeFluxCorrBufferSizeCC(const MeshBlock *pmb, const NeighborIndexes &ni,
                                int nvar) {
  if (ni.type != NeighborConnect::face) return 0;

  int nx1 = pmb->block_size.nx1;
  int nx2 = pmb->block_size.nx2;
  int nx3 = pmb->block_size.nx3;

  // Transverse sizes after 2:1 restriction.  (nx+1)/2 handles the integer division.
  if (ni.ox1 != 0)       // x1-face: transverse = x2 x x3
    return ((nx2 + 1) / 2) * ((nx3 + 1) / 2) * nvar;
  else if (ni.ox2 != 0)  // x2-face: transverse = x1 x x3
    return ((nx1 + 1) / 2) * ((nx3 + 1) / 2) * nvar;
  else                    // x3-face: transverse = x1 x x2
    return ((nx1 + 1) / 2) * ((nx2 + 1) / 2) * nvar;
}

//========================================================================================
// FC EMF flux correction buffer size.
// Face neighbors carry two tangential EMF components; edge neighbors carry one parallel
// EMF component.  Corner neighbors return 0.
//
// For same-level exchange the full fine-grid EMF array is packed (no restriction).
// For to-coarser the stride-2 restricted size is half, but we allocate the maximum,
// which is the same-level (unrestricted) size.
//========================================================================================

int ComputeFluxCorrBufferSizeFC(const MeshBlock *pmb, const NeighborIndexes &ni) {
  int nx1 = pmb->block_size.nx1;
  int nx2 = pmb->block_size.nx2;
  int nx3 = pmb->block_size.nx3;
  int f2 = (nx2 > 1) ? 1 : 0;
  int f3 = (nx3 > 1) ? 1 : 0;

  if (ni.type == NeighborConnect::face) {
    // Two tangential EMF components.  Each component has stagger-offset +1 in its own
    // direction on the face (edges span one more than cells in that direction).
    if (ni.ox1 != 0) {
      // x1-face: tangential = e2 (size nx3 x (nx2+1)) + e3 (size (nx3+1) x nx2)
      // For 2D (nx3==1): e2 -> 1*(nx2+1), e3 -> 1*nx2;  for 1D: 2 (one per component)
      return (nx2 + f2) * (nx3 > 1 ? nx3 : 1)
           + nx2 * (nx3 + f3 > 1 ? nx3 + f3 : 1);
    } else if (ni.ox2 != 0) {
      // x2-face: tangential = e1 (size (nx1+1) x nx3) + e3 (size nx1 x (nx3+1))
      return (nx1 + 1) * (nx3 > 1 ? nx3 : 1)
           + nx1 * (nx3 + f3 > 1 ? nx3 + f3 : 1);
    } else {
      // x3-face: tangential = e1 (size (nx1+1) x nx2) + e2 (size nx1 x (nx2+1))
      return (nx1 + 1) * nx2
           + nx1 * (nx2 + f2);
    }
  } else if (ni.type == NeighborConnect::edge) {
    // One EMF component parallel to the edge.
    // Edge orientation: the zero ox component determines the edge direction.
    if (ni.ox1 == 0) {
      // x2x3 edge: parallel to x1 -> e1 has size nx1
      return nx1;
    } else if (ni.ox2 == 0) {
      // x1x3 edge: parallel to x2 -> e2 has size nx2
      return (nx2 > 1) ? nx2 : 1;
    } else {
      // x1x2 edge: parallel to x3 -> e3 has size nx3
      return (nx3 > 1) ? nx3 : 1;
    }
  }
  // Corner neighbors don't participate in FC flux correction.
  return 0;
}

} // namespace idx
} // namespace comm
