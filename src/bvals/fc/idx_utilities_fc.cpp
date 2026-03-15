// C headers

// C++ headers
#include <algorithm>  // std::max
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../mesh/mesh.hpp"
#include "bvals_fc.hpp"

//=============================================================================
// Index-range helper functions for FC boundary variable pack / unpack
//=============================================================================

//----------------------------------------------------------------------------
// Per-dimension inline helper: Load same-level / Load to-coarser
//
// Non-stagger (is_stagger=false):
//   ox>0  -> [ve - ng + 1, ve]        (high-end interior slab)
//   ox<0  -> [vs, vs + ng - 1]        (low-end interior slab)
//   ox==0 -> [vs, ve]                 (full interior span)
//
// Stagger (is_stagger=true):
//   ox>0  -> [ve - ng + 1, ve]        (face excluded from slab)
//   ox<0  -> [vs + 1, vs + ng]        (face excluded from slab)
//   ox==0 -> [vs, ve + 1]             (includes boundary face; trivial dim: [vs, ve])
inline void SetIndexRangesLoad_FC(
    int ox, int &s, int &e,
    int vs, int ve, int ng,
    bool is_stagger, bool is_dim_nontrivial) {
  if (!is_stagger) {
    // Standard CC-like Load
    s = (ox > 0) ? (ve - ng + 1) : vs;
    e = (ox < 0) ? (vs + ng - 1) : ve;
  } else {
    // Face-staggered dimension
    if (ox == 0) {
      s = vs;
      e = is_dim_nontrivial ? (ve + 1) : ve;
    } else if (ox > 0) {
      s = ve - ng + 1;
      e = ve;             // boundary face excluded
    } else {
      s = vs + 1;         // boundary face excluded
      e = vs + ng;
    }
  }
}

//----------------------------------------------------------------------------
// Per-dimension inline helper: Set same-level (unpack destination)
//
// Non-stagger:
//   ox>0  -> [ve + 1, ve + ng]        (high ghost zone)
//   ox<0  -> [vs - ng, vs - 1]        (low ghost zone)
//   ox==0 -> [vs, ve]                 (interior)
//
// Stagger:
//   ox>0  -> [ve + 2, ve + 1 + ng]    (skips shared boundary face)
//   ox<0  -> [vs - ng, vs - 1]        (low ghost zone)
//   ox==0 -> [vs, ve + 1]             (includes boundary face; trivial dim: [vs, ve])
inline void SetIndexRangesSet_FC(
    int ox, int &s, int &e,
    int vs, int ve, int ng,
    bool is_stagger, bool is_dim_nontrivial) {
  if (!is_stagger) {
    // Standard CC-like Set
    if (ox == 0)     { s = vs;        e = ve; }
    else if (ox > 0) { s = ve + 1;    e = ve + ng; }
    else             { s = vs - ng;   e = vs - 1; }
  } else {
    // Face-staggered dimension
    if (ox == 0) {
      s = vs;
      e = is_dim_nontrivial ? (ve + 1) : ve;
    } else if (ox > 0) {
      s = ve + 2;         // skip shared boundary face
      e = ve + 1 + ng;
    } else {
      s = vs - ng;
      e = vs - 1;
    }
  }
}

//----------------------------------------------------------------------------
// Per-dimension inline helper: SetFromCoarser (unpack into coarse_buf)
//
// Non-stagger: same as CC
//   ox>0  -> [cve + 1, cve + cng]
//   ox<0  -> [cvs - cng, cvs - 1]
//   ox==0 -> [cvs, cve] extended by cng via lx parity (if nontrivial dim)
//
// Stagger:
//   ox>0  -> [cve + 1, cve + 1 + cng]  (includes boundary face, span = cng+1)
//   ox<0  -> [cvs - cng, cvs]           (includes boundary face, span = cng+1)
//   ox==0 -> [cvs, cve + 1] then parity-extend by cng (if nontrivial)
//            trivial dim: [cvs, cve]
inline void SetIndexRangesFromCoarser_FC(
    int ox, int &s, int &e,
    int cvs, int cve, int cng,
    std::int64_t lx, bool is_nontrivial,
    bool is_stagger) {
  if (!is_stagger) {
    // Standard CC-like FromCoarser
    if (ox == 0) {
      s = cvs; e = cve;
      if (is_nontrivial) {
        if ((lx & 1LL) == 0LL) e += cng;
        else                    s -= cng;
      }
    } else if (ox > 0) {
      s = cve + 1;    e = cve + cng;
    } else {
      s = cvs - cng;  e = cvs - 1;
    }
  } else {
    // Face-staggered dimension
    if (ox == 0) {
      s = cvs;
      e = is_nontrivial ? (cve + 1) : cve;
      if (is_nontrivial) {
        if ((lx & 1LL) == 0LL) e += cng;
        else                    s -= cng;
      }
    } else if (ox > 0) {
      s = cve + 1;         // boundary face included
      e = cve + 1 + cng;
    } else {
      s = cvs - cng;       // boundary face included
      e = cvs;
    }
  }
}

//----------------------------------------------------------------------------
// Per-dimension inline helper: LoadToFiner (pack source, interior slab)
//
// Non-stagger: same as CC LoadToFiner
//   ox>0  -> [ve - ng + 1, ve]
//   ox<0  -> [vs, vs + ng - 1]
//   ox==0 -> [vs, ve] then narrow by fi1/fi2
//
// Stagger:
//   ox>0  -> [ve + 1 - ng, ve + 1]     (includes boundary face)
//   ox<0  -> [vs, vs + ng]             (includes boundary face)
//   ox==0 -> [vs, ve + 1] then narrow by fi1/fi2 with half_size
//            trivial dim: [vs, ve] (no +1, no narrowing)
inline void SetIndexRangesLoadToFiner_FC(
    int ox, int &s, int &e,
    int vs, int ve, int ng,
    int fi1, int fi2, int half_size,
    bool is_nontrivial, bool use_fi1,
    bool is_stagger) {
  if (!is_stagger) {
    // Standard CC-like LoadToFiner
    if (ox == 0) {
      s = vs; e = ve;
      if (is_nontrivial) {
        if (use_fi1) {
          if (fi1 == 1) s += half_size;
          else          e -= half_size;
        } else {
          if (fi2 == 1) s += half_size;
          else          e -= half_size;
        }
      }
    } else if (ox > 0) {
      s = ve - ng + 1;  e = ve;
    } else {
      s = vs;           e = vs + ng - 1;
    }
  } else {
    // Face-staggered dimension
    if (ox == 0) {
      s = vs;
      e = is_nontrivial ? (ve + 1) : ve;
      if (is_nontrivial) {
        if (use_fi1) {
          if (fi1 == 1) s += half_size;
          else          e -= half_size;
        } else {
          if (fi2 == 1) s += half_size;
          else          e -= half_size;
        }
      }
    } else if (ox > 0) {
      s = ve + 1 - ng;   // includes boundary face
      e = ve + 1;
    } else {
      s = vs;             // includes boundary face
      e = vs + ng;
    }
  }
}

//----------------------------------------------------------------------------
// Per-dimension inline helper: SetFromFiner (unpack destination)
//
// Non-stagger: same as CC SetFromFiner (ghost-zone destination)
//   ox>0  -> [ve + 1, ve + ng]
//   ox<0  -> [vs - ng, vs - 1]
//   ox==0 -> [vs, ve] then narrow by fi1/fi2
//
// Stagger:
//   ox>0  -> [ve + 2, ve + 1 + ng]     (skips shared face, like Set same-level)
//   ox<0  -> [vs - ng, vs - 1]
//   ox==0 -> [vs, ve + 1] then narrow by fi1/fi2 with half_size
//            trivial dim: [vs, ve] (no +1, no narrowing)
inline void SetIndexRangesSetFromFiner_FC(
    int ox, int &s, int &e,
    int vs, int ve, int ng,
    int fi1, int fi2, int half_size,
    bool is_nontrivial, bool use_fi1,
    bool is_stagger) {
  if (!is_stagger) {
    // Standard CC-like SetFromFiner
    if (ox == 0) {
      s = vs; e = ve;
      if (is_nontrivial) {
        if (use_fi1) {
          if (fi1 == 1) s += half_size;
          else          e -= half_size;
        } else {
          if (fi2 == 1) s += half_size;
          else          e -= half_size;
        }
      }
    } else if (ox > 0) {
      s = ve + 1;    e = ve + ng;
    } else {
      s = vs - ng;   e = vs - 1;
    }
  } else {
    // Face-staggered dimension
    if (ox == 0) {
      s = vs;
      e = is_nontrivial ? (ve + 1) : ve;
      if (is_nontrivial) {
        if (use_fi1) {
          if (fi1 == 1) s += half_size;
          else          e -= half_size;
        } else {
          if (fi2 == 1) s += half_size;
          else          e -= half_size;
        }
      }
    } else if (ox > 0) {
      s = ve + 2;        // skip shared face
      e = ve + 1 + ng;
    } else {
      s = vs - ng;
      e = vs - 1;
    }
  }
}

//=============================================================================
// Class method idx helpers -- each computes (si,ei,sj,ej,sk,ek) for one
// face-field component identified by stagger_axis (0=x1f, 1=x2f, 2=x3f).
//=============================================================================

//----------------------------------------------------------------------------------------
//! \fn void FaceCenteredBoundaryVariable::idxLoadSameLevelRanges_FC(...)
//  \brief Index ranges for LoadBoundaryBufferSameLevel, one face component.
//
//  is_coarse=false: fine grid indices (edge/corner overlap disabled - see below).
//  is_coarse=true:  coarse same-level payload, no edge/corner overlap.
void FaceCenteredBoundaryVariable::idxLoadSameLevelRanges_FC(
    const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
    int stagger_axis, bool is_coarse) {
  MeshBlock *pmb = pmy_block_;

  if (!is_coarse) {
    // --- fine grid ---
    int ng = NGHOST;
    SetIndexRangesLoad_FC(ni.ox1, si, ei, pmb->is, pmb->ie, ng,
                          stagger_axis == 0, true);
    SetIndexRangesLoad_FC(ni.ox2, sj, ej, pmb->js, pmb->je, ng,
                          stagger_axis == 1, pmb->block_size.nx2 > 1);
    SetIndexRangesLoad_FC(ni.ox3, sk, ek, pmb->ks, pmb->ke, ng,
                          stagger_axis == 2, pmb->block_size.nx3 > 1);

    // Same-level edge/corner overlap re-included the shared
    // interior boundary face in edge/corner neighbor packs.
    // This overwrote EMF-corrected face values with neighbor-computed values.
    // The matching set-side and buffer-size overlaps are also disabled and
    // must stay in sync.

    // if (pmy_mesh_->multilevel && ni.type != NeighborConnect::face) {
    //   if (stagger_axis == 0) {
    //     if (ni.ox1 > 0) ei++;
    //     else if (ni.ox1 < 0) si--;
    //   } else if (stagger_axis == 1) {
    //     if (ni.ox2 > 0) ej++;
    //     else if (ni.ox2 < 0) sj--;
    //   } else {
    //     if (ni.ox3 > 0) ek++;
    //     else if (ni.ox3 < 0) sk--;
    //   }
    // }

  } else {
    // --- coarse same-level payload (no edge/corner overlap) ---
    int cng  = pmb->cnghost;
    int cng2 = cng * pmy_mesh_->f2;
    int cng3 = cng * pmy_mesh_->f3;
    SetIndexRangesLoad_FC(ni.ox1, si, ei, pmb->cis, pmb->cie, cng,
                          stagger_axis == 0, true);
    SetIndexRangesLoad_FC(ni.ox2, sj, ej, pmb->cjs, pmb->cje, cng2,
                          stagger_axis == 1, pmb->block_size.nx2 > 1);
    SetIndexRangesLoad_FC(ni.ox3, sk, ek, pmb->cks, pmb->cke, cng3,
                          stagger_axis == 2, pmb->block_size.nx3 > 1);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void FaceCenteredBoundaryVariable::idxLoadToCoarserRanges_FC(...)
//  \brief Index ranges for LoadBoundaryBufferToCoarser, one face component.
//
//  Packs from coarse_buf (pre-restricted by RestrictNonGhost).
//  Uses cng=NGHOST.  Includes edge/corner overlap adjustment.
void FaceCenteredBoundaryVariable::idxLoadToCoarserRanges_FC(
    const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
    int stagger_axis) {
  MeshBlock *pmb = pmy_block_;
  int cng = NGHOST;

  SetIndexRangesLoad_FC(ni.ox1, si, ei, pmb->cis, pmb->cie, cng,
                        stagger_axis == 0, true);
  SetIndexRangesLoad_FC(ni.ox2, sj, ej, pmb->cjs, pmb->cje, cng,
                        stagger_axis == 1, pmb->block_size.nx2 > 1);
  SetIndexRangesLoad_FC(ni.ox3, sk, ek, pmb->cks, pmb->cke, cng,
                        stagger_axis == 2, pmb->block_size.nx3 > 1);

  // Edge/corner overlap: re-include the boundary face for non-face neighbors
  if (ni.type != NeighborConnect::face) {
    if (stagger_axis == 0) {
      if (ni.ox1 > 0) ei++;
      else if (ni.ox1 < 0) si--;
    } else if (stagger_axis == 1) {
      if (ni.ox2 > 0) ej++;
      else if (ni.ox2 < 0) sj--;
    } else {
      if (ni.ox3 > 0) ek++;
      else if (ni.ox3 < 0) sk--;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void FaceCenteredBoundaryVariable::idxLoadToFinerRanges_FC(...)
//  \brief Index ranges for LoadBoundaryBufferToFiner, one face component.
//
//  Packs from var_fc (fine grid) as interior slab for prolongation on receiver.
//  For non-stagger dims ox!=0, uses span = cnghost cells.
//  For stagger dim ox!=0, includes the boundary face (span = cnghost+1).
void FaceCenteredBoundaryVariable::idxLoadToFinerRanges_FC(
    const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
    int stagger_axis) {
  MeshBlock *pmb = pmy_block_;
  int cng = pmb->cnghost;
  int nx1 = pmb->block_size.nx1;
  int nx2 = pmb->block_size.nx2;
  int nx3 = pmb->block_size.nx3;

  // x1 dimension
  SetIndexRangesLoadToFiner_FC(ni.ox1, si, ei, pmb->is, pmb->ie, cng,
      ni.fi1, ni.fi2, nx1/2 - cng,
      true, true,
      stagger_axis == 0);

  // x2 dimension
  SetIndexRangesLoadToFiner_FC(ni.ox2, sj, ej, pmb->js, pmb->je, cng,
      ni.fi1, ni.fi2, nx2/2 - cng,
      nx2 > 1, ni.ox1 != 0,
      stagger_axis == 1);

  // x3 dimension
  SetIndexRangesLoadToFiner_FC(ni.ox3, sk, ek, pmb->ks, pmb->ke, cng,
      ni.fi1, ni.fi2, nx3/2 - cng,
      nx3 > 1, (ni.ox1 != 0 && ni.ox2 != 0),
      stagger_axis == 2);
}

//----------------------------------------------------------------------------------------
//! \fn void FaceCenteredBoundaryVariable::idxSetSameLevelRanges_FC(...)
//  \brief Index ranges for SetBoundarySameLevel, one face component.
//
//  type=1: fine data (edge/corner overlap disabled - see idxLoadSameLevelRanges_FC).
//  type=2: coarse same-level payload (no edge/corner overlap).
void FaceCenteredBoundaryVariable::idxSetSameLevelRanges_FC(
    const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
    int stagger_axis, int type) {
  MeshBlock *pmb = pmy_block_;

  if (type == 1) {
    // --- fine grid ---
    int ng = NGHOST;
    SetIndexRangesSet_FC(ni.ox1, si, ei, pmb->is, pmb->ie, ng,
                         stagger_axis == 0, true);
    SetIndexRangesSet_FC(ni.ox2, sj, ej, pmb->js, pmb->je, ng,
                         stagger_axis == 1, pmb->block_size.nx2 > 1);
    SetIndexRangesSet_FC(ni.ox3, sk, ek, pmb->ks, pmb->ke, ng,
                         stagger_axis == 2, pmb->block_size.nx3 > 1);
    // Same-level edge/corner overlap (Set side).
    // See idxLoadSameLevelRanges_FC for the full explanation.
    // if (pmy_mesh_->multilevel && ni.type != NeighborConnect::face) {
    //   if (stagger_axis == 0) {
    //     if (ni.ox1 > 0) si--;
    //     else if (ni.ox1 < 0) ei++;
    //   } else if (stagger_axis == 1) {
    //     if (ni.ox2 > 0) sj--;
    //     else if (ni.ox2 < 0) ej++;
    //   } else {
    //     if (ni.ox3 > 0) sk--;
    //     else if (ni.ox3 < 0) ek++;
    //   }
    // }
  } else {
    // --- coarse same-level payload (no edge/corner overlap) ---
    int cng  = pmb->cnghost;
    int cng2 = cng * pmy_mesh_->f2;
    int cng3 = cng * pmy_mesh_->f3;
    SetIndexRangesSet_FC(ni.ox1, si, ei, pmb->cis, pmb->cie, cng,
                         stagger_axis == 0, true);
    SetIndexRangesSet_FC(ni.ox2, sj, ej, pmb->cjs, pmb->cje, cng2,
                         stagger_axis == 1, pmb->block_size.nx2 > 1);
    SetIndexRangesSet_FC(ni.ox3, sk, ek, pmb->cks, pmb->cke, cng3,
                         stagger_axis == 2, pmb->block_size.nx3 > 1);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void FaceCenteredBoundaryVariable::idxSetFromCoarserRanges_FC(...)
//  \brief Index ranges for SetBoundaryFromCoarser, one face component.
//
//  Unpacks into coarse_buf with parity-extended ranges for prolongation.
void FaceCenteredBoundaryVariable::idxSetFromCoarserRanges_FC(
    const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
    int stagger_axis) {
  MeshBlock *pmb = pmy_block_;
  int cng = pmb->cnghost;

  SetIndexRangesFromCoarser_FC(ni.ox1, si, ei,
      pmb->cis, pmb->cie, cng, pmb->loc.lx1, true,
      stagger_axis == 0);
  SetIndexRangesFromCoarser_FC(ni.ox2, sj, ej,
      pmb->cjs, pmb->cje, cng, pmb->loc.lx2, pmb->block_size.nx2 > 1,
      stagger_axis == 1);
  SetIndexRangesFromCoarser_FC(ni.ox3, sk, ek,
      pmb->cks, pmb->cke, cng, pmb->loc.lx3, pmb->block_size.nx3 > 1,
      stagger_axis == 2);
}

//----------------------------------------------------------------------------------------
//! \fn void FaceCenteredBoundaryVariable::idxSetFromFinerRanges_FC(...)
//  \brief Index ranges for SetBoundaryFromFiner, one face component.
//
//  Unpacks restricted data from finer neighbor into var_fc ghost zones.
//  Includes edge/corner overlap adjustment (direction opposite from Load).
void FaceCenteredBoundaryVariable::idxSetFromFinerRanges_FC(
    const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
    int stagger_axis) {
  MeshBlock *pmb = pmy_block_;
  int ng = NGHOST;
  int nx1 = pmb->block_size.nx1;
  int nx2 = pmb->block_size.nx2;
  int nx3 = pmb->block_size.nx3;

  // x1 dimension
  SetIndexRangesSetFromFiner_FC(ni.ox1, si, ei, pmb->is, pmb->ie, ng,
      ni.fi1, ni.fi2, nx1/2,
      true, true,
      stagger_axis == 0);

  // Edge/corner overlap for stagger_axis==0 (bx1): applied after x1 indices
  if (stagger_axis == 0 && ni.type != NeighborConnect::face) {
    if (ni.ox1 > 0) si--;
    else if (ni.ox1 < 0) ei++;
  }

  // x2 dimension
  SetIndexRangesSetFromFiner_FC(ni.ox2, sj, ej, pmb->js, pmb->je, ng,
      ni.fi1, ni.fi2, nx2/2,
      nx2 > 1, ni.ox1 != 0,
      stagger_axis == 1);

  // Edge/corner overlap for stagger_axis==1 (bx2)
  if (stagger_axis == 1 && ni.type != NeighborConnect::face) {
    if (ni.ox2 > 0) sj--;
    else if (ni.ox2 < 0) ej++;
  }

  // x3 dimension
  SetIndexRangesSetFromFiner_FC(ni.ox3, sk, ek, pmb->ks, pmb->ke, ng,
      ni.fi1, ni.fi2, nx3/2,
      nx3 > 1, (ni.ox1 != 0 && ni.ox2 != 0),
      stagger_axis == 2);

  // Edge/corner overlap for stagger_axis==2 (bx3)
  if (stagger_axis == 2 && ni.type != NeighborConnect::face) {
    if (ni.ox3 > 0) sk--;
    else if (ni.ox3 < 0) ek++;
  }
}

//=============================================================================
// MPI buffer size helpers
//=============================================================================
#ifdef MPI_PARALLEL

int FaceCenteredBoundaryVariable::MPI_BufferSizeSameLevel_FC(
    const NeighborIndexes& ni, bool skip_coarse) {
  MeshBlock *pmb = pmy_block_;
  int nx1 = pmb->block_size.nx1;
  int nx2 = pmb->block_size.nx2;
  int nx3 = pmb->block_size.nx3;
  int f2 = pmy_mesh_->f2, f3 = pmy_mesh_->f3;

  // Fine payload: 3 components with stagger +1 in their respective directions
  int size1 = ((ni.ox1 == 0) ? (nx1 + 1) : NGHOST)
              *((ni.ox2 == 0) ? (nx2) : NGHOST)
              *((ni.ox3 == 0) ? (nx3) : NGHOST);
  int size2 = ((ni.ox1 == 0) ? (nx1) : NGHOST)
              *((ni.ox2 == 0) ? (nx2 + f2) : NGHOST)
              *((ni.ox3 == 0) ? (nx3) : NGHOST);
  int size3 = ((ni.ox1 == 0) ? (nx1) : NGHOST)
              *((ni.ox2 == 0) ? (nx2) : NGHOST)
              *((ni.ox3 == 0) ? (nx3 + f3) : NGHOST);
  // Same-level edge/corner overlap (buffer size).
  // See idxLoadSameLevelRanges_FC for the full explanation.

  // if (pmy_mesh_->multilevel && ni.type != NeighborConnect::face) {
  //   if (ni.ox1 != 0) size1 = size1/NGHOST*(NGHOST + 1);
  //   if (ni.ox2 != 0) size2 = size2/NGHOST*(NGHOST + 1);
  //   if (ni.ox3 != 0) size3 = size3/NGHOST*(NGHOST + 1);
  // }

  int size = size1 + size2 + size3;

  // Coarse same-level payload
  if (pmy_mesh_->multilevel && !skip_coarse) {
    int cng  = pmb->cnghost;
    int cng2 = cng * f2;
    int cng3 = cng * f3;
    int sc1 = ((ni.ox1 == 0) ? ((nx1 + 1)/2 + 1) : cng)
              *((ni.ox2 == 0) ? ((nx2 + 1)/2) : cng2)
              *((ni.ox3 == 0) ? ((nx3 + 1)/2) : cng3);
    int sc2 = ((ni.ox1 == 0) ? ((nx1 + 1)/2) : cng)
              *((ni.ox2 == 0) ? ((nx2 + 1)/2 + f2) : cng2)
              *((ni.ox3 == 0) ? ((nx3 + 1)/2) : cng3);
    int sc3 = ((ni.ox1 == 0) ? ((nx1 + 1)/2) : cng)
              *((ni.ox2 == 0) ? ((nx2 + 1)/2) : cng2)
              *((ni.ox3 == 0) ? ((nx3 + 1)/2 + f3) : cng3);
    size += sc1 + sc2 + sc3;
  }
  return size;
}

int FaceCenteredBoundaryVariable::MPI_BufferSizeToCoarser_FC(
    const NeighborIndexes& ni) {
  MeshBlock *pmb = pmy_block_;
  int nx1 = pmb->block_size.nx1;
  int nx2 = pmb->block_size.nx2;
  int nx3 = pmb->block_size.nx3;
  int f2 = pmy_mesh_->f2, f3 = pmy_mesh_->f3;

  int f2c1 = ((ni.ox1 == 0) ? ((nx1 + 1)/2 + 1) : NGHOST)
             *((ni.ox2 == 0) ? ((nx2 + 1)/2) : NGHOST)
             *((ni.ox3 == 0) ? ((nx3 + 1)/2) : NGHOST);
  int f2c2 = ((ni.ox1 == 0) ? ((nx1 + 1)/2) : NGHOST)
             *((ni.ox2 == 0) ? ((nx2 + 1)/2 + f2) : NGHOST)
             *((ni.ox3 == 0) ? ((nx3 + 1)/2) : NGHOST);
  int f2c3 = ((ni.ox1 == 0) ? ((nx1 + 1)/2) : NGHOST)
             *((ni.ox2 == 0) ? ((nx2 + 1)/2) : NGHOST)
             *((ni.ox3 == 0) ? ((nx3 + 1)/2 + f3) : NGHOST);
  // Edge/corner overlap adjustment
  if (ni.type != NeighborConnect::face) {
    if (ni.ox1 != 0) f2c1 = f2c1/NGHOST*(NGHOST + 1);
    if (ni.ox2 != 0) f2c2 = f2c2/NGHOST*(NGHOST + 1);
    if (ni.ox3 != 0) f2c3 = f2c3/NGHOST*(NGHOST + 1);
  }
  return f2c1 + f2c2 + f2c3;
}

int FaceCenteredBoundaryVariable::MPI_BufferSizeFromCoarser_FC(
    const NeighborIndexes& ni) {
  MeshBlock *pmb = pmy_block_;
  int nx1 = pmb->block_size.nx1;
  int nx2 = pmb->block_size.nx2;
  int nx3 = pmb->block_size.nx3;
  int f2 = pmy_mesh_->f2, f3 = pmy_mesh_->f3;
  int cng  = pmb->cnghost;
  int cng1 = cng;
  int cng2 = cng * f2;
  int cng3 = cng * f3;

  int c2f1 = ((ni.ox1 == 0) ? ((nx1 + 1)/2 + cng1 + 1) : cng + 1)
             *((ni.ox2 == 0) ? ((nx2 + 1)/2 + cng2) : cng)
             *((ni.ox3 == 0) ? ((nx3 + 1)/2 + cng3) : cng);
  int c2f2 = ((ni.ox1 == 0) ? ((nx1 + 1)/2 + cng1) : cng)
             *((ni.ox2 == 0) ? ((nx2 + 1)/2 + cng2 + f2) : cng + 1)
             *((ni.ox3 == 0) ? ((nx3 + 1)/2 + cng3) : cng);
  int c2f3 = ((ni.ox1 == 0) ? ((nx1 + 1)/2 + cng1) : cng)
             *((ni.ox2 == 0) ? ((nx2 + 1)/2 + cng2) : cng)
             *((ni.ox3 == 0) ? ((nx3 + 1)/2 + cng3 + f3) : cng + 1);
  return c2f1 + c2f2 + c2f3;
}

int FaceCenteredBoundaryVariable::MPI_BufferSizeToFiner_FC(
    const NeighborIndexes& ni) {
  // Same as FromCoarser: parent sends interior+ghost slab
  return MPI_BufferSizeFromCoarser_FC(ni);
}

int FaceCenteredBoundaryVariable::MPI_BufferSizeFromFiner_FC(
    const NeighborIndexes& ni) {
  // Same as ToCoarser: child sends restricted data
  return MPI_BufferSizeToCoarser_FC(ni);
}

#endif // MPI_PARALLEL

//-----------------------------------------------------------------------------
// Prolongation index helpers (existing)
//-----------------------------------------------------------------------------

void FaceCenteredBoundaryVariable::CalculateProlongationIndices(
  NeighborBlock &nb,
  int &si, int &ei,
  int &sj, int &ej,
  int &sk, int &ek)
{
  MeshBlock * pmb = pmy_block_;

  static const int pcng = pmb->cnghost - 1;

  CalculateProlongationIndices(pmb->loc.lx1, nb.ni.ox1, pcng,
                               pmb->cis, pmb->cie,
                               si, ei,
                               true);
  CalculateProlongationIndices(pmb->loc.lx2, nb.ni.ox2, pcng,
                               pmb->cjs, pmb->cje,
                               sj, ej,
                               pmb->block_size.nx2 > 1);
  CalculateProlongationIndices(pmb->loc.lx3, nb.ni.ox3, pcng,
                               pmb->cks, pmb->cke,
                               sk, ek,
                               pmb->block_size.nx3 > 1);

}

void FaceCenteredBoundaryVariable::CalculateProlongationSharedIndices(
  NeighborBlock &nb,
  const int si, const int ei,
  const int sj, const int ej,
  const int sk, const int ek,
  int &il, int &iu, int &jl, int &ju, int &kl, int &ku)
{
  MeshBlock *pmb = pmy_block_;
  BoundaryValues *pbval = pmb->pbval;

  const int &mylevel = pmb->loc.level;

  il = si, iu = ei + 1;
  if ((nb.ni.ox1 >= 0) &&
      (pbval->nblevel[nb.ni.ox3 + 1][nb.ni.ox2 + 1][nb.ni.ox1] >= mylevel))
    il++;

  if ((nb.ni.ox1 <= 0) &&
      (pbval->nblevel[nb.ni.ox3 + 1][nb.ni.ox2 + 1][nb.ni.ox1 + 2] >= mylevel))
    iu--;

  if (pmb->block_size.nx2 > 1) {
    jl = sj, ju = ej + 1;
    if ((nb.ni.ox2 >= 0) &&
        (pbval->nblevel[nb.ni.ox3 + 1][nb.ni.ox2][nb.ni.ox1 + 1] >=
         mylevel))
      jl++;
    if ((nb.ni.ox2 <= 0) &&
        (pbval->nblevel[nb.ni.ox3 + 1][nb.ni.ox2 + 2][nb.ni.ox1 + 1] >=
         mylevel))
      ju--;
  } else {
    jl = sj;
    ju = ej;
  }

  if (pmb->block_size.nx3 > 1) {
    kl = sk, ku = ek + 1;
    if ((nb.ni.ox3 >= 0) &&
        (pbval->nblevel[nb.ni.ox3][nb.ni.ox2 + 1][nb.ni.ox1 + 1] >=
         mylevel))
      kl++;
    if ((nb.ni.ox3 <= 0) &&
        (pbval->nblevel[nb.ni.ox3 + 2][nb.ni.ox2 + 1][nb.ni.ox1 + 1] >=
         mylevel))
      ku--;
  } else {
    kl = sk;
    ku = ek;
  }

}

void FaceCenteredBoundaryVariable::CalculateProlongationIndicesFine(
  NeighborBlock &nb,
  int &fsi, int &fei,
  int &fsj, int &fej,
  int &fsk, int &fek)
{
  MeshBlock * pmb = pmy_block_;

  int si, ei, sj, ej, sk, ek;
  CalculateProlongationIndices(nb, si, ei, sj, ej, sk, ek);

  // ghost-ghost zones are filled and prolongated,
  // calculate the loop limits for the finer grid
  fsi = (si - pmb->cis)*2 + pmb->is;
  fei = (ei - pmb->cis)*2 + pmb->is + 1;
  if (pmb->block_size.nx2 > 1) {
    fsj = (sj - pmb->cjs)*2 + pmb->js;
    fej = (ej - pmb->cjs)*2 + pmb->js + 1;
  } else {
    fsj = pmb->js;
    fej = pmb->je;
  }
  if (pmb->block_size.nx3 > 1) {
    fsk = (sk - pmb->cks)*2 + pmb->ks;
    fek = (ek - pmb->cks)*2 + pmb->ks + 1;
  } else {
    fsk = pmb->ks;
    fek = pmb->ke;
  }
}

//
// :D
//
