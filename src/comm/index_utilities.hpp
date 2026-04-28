#ifndef COMM_INDEX_UTILITIES_HPP_
#define COMM_INDEX_UTILITIES_HPP_
//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file index_utilities.hpp
//  \brief Unified index-range computation for pack/unpack across all sampling types.
//
//  Replaces the four near-identical idx_utilities_{cc,vc,cx,fc}.cpp files with
//  one set of free functions parameterized by Sampling and ghost width.
//
//  All functions compute inclusive [start, end] index ranges for one dimension.
//  The caller loops over dimensions and combines the 3D result.
//
//  Sampling differences:
//    CC  - arithmetic offsets from is/ie + NGHOST, symmetric ghost zones.
//    VC  - pre-computed named indices (ivs/ive, igs/ige, ims/ipe). Shared vertex
//           is included in loads and sets. Unpack is additive (UnpackDataAdd).
//    CX  - pre-computed named indices (cx_is/cx_ie, cx_igs/cx_ige, cx_ims/cx_ipe).
//           Set-same-level has a gap (+1/-1). Coarser-level ghost offset via cx_dng.
//    FC  - 3-component staggered structure. Per-dimension helpers take is_stagger
//           bool; 3D functions called once per stagger_axis (0=x1f, 1=x2f, 2=x3f).
//           Edge/corner overlap applied in LoadToCoarser and SetFromFiner.

#include <cstdint>  // std::int64_t

#include "../athena.hpp"
#include "comm_enums.hpp"

// forward declarations
class MeshBlock;
struct NeighborIndexes;

namespace comm {
namespace idx {

//----------------------------------------------------------------------------------------
// Structs to hold 3D index ranges (inclusive).

struct IndexRange3D {
  int si, ei, sj, ej, sk, ek;
};

//========================================================================================
// Per-dimension helpers for CC (also used by FC same-level).
// These are the building blocks - identical logic for CC, differing only in the
// ghost width (ng) and the start/end indices (vs, ve) passed by the caller.
//========================================================================================

// Load same-level / load to coarser: interior slab that the neighbor needs.
//   ox > 0 -> high-end interior slab [ve-ng+1, ve]
//   ox < 0 -> low-end interior slab  [vs, vs+ng-1]
//   ox == 0 -> full interior         [vs, ve]
inline void LoadRangeCC(int ox, int &s, int &e, int vs, int ve, int ng) {
  s = (ox > 0) ? (ve - ng + 1) : vs;
  e = (ox < 0) ? (vs + ng - 1) : ve;
}

// Set same-level (CC): destination ghost zone on the receiving side.
//   ox > 0 -> high ghost  [ve+1, ve+ng]
//   ox < 0 -> low ghost   [vs-ng, vs-1]
//   ox == 0 -> interior   [vs, ve]
inline void SetRangeCC(int ox, int &s, int &e, int vs, int ve, int ng) {
  if (ox == 0)     { s = vs;      e = ve; }
  else if (ox > 0) { s = ve + 1;  e = ve + ng; }
  else             { s = vs - ng; e = vs - 1; }
}

// Set from coarser neighbor (CC): unpack into coarse buffer.
//   ox > 0 -> [cve+1, cve+cng]
//   ox < 0 -> [cvs-cng, cvs-1]
//   ox == 0 -> [cvs, cve] extended by cng via lx parity
inline void SetFromCoarserRangeCC(int ox, int &s, int &e,
                                int cvs, int cve, int cng,
                                std::int64_t lx, bool is_nontrivial) {
  if (ox == 0) {
    s = cvs; e = cve;
    if (is_nontrivial) {
      if ((lx & 1LL) == 0LL) e += cng;
      else                    s -= cng;
    }
  } else if (ox > 0) {
    s = cve + 1; e = cve + cng;
  } else {
    s = cvs - cng; e = cvs - 1;
  }
}

// Load to finer / Set from finer (CC): needs fi1/fi2 sub-face selection.
//   ox != 0 -> same as LoadRangeCC / SetRangeCC
//   ox == 0 -> full span, then narrow by half_size via fi1/fi2
inline void FinerRangeCC(int ox, int &s, int &e,
                       int vs, int ve, int ng,
                       int fi1, int fi2, int half_size,
                       bool is_nontrivial, bool use_fi1,
                       bool is_load) {
  if (ox == 0) {
    s = vs; e = ve;
    if (is_nontrivial) {
      int fi = use_fi1 ? fi1 : fi2;
      if (fi == 1) s += half_size;
      else         e -= half_size;
    }
  } else if (is_load) {
    // Load: interior slab
    s = (ox > 0) ? (ve - ng + 1) : vs;
    e = (ox < 0) ? (vs + ng - 1) : ve;
  } else {
    // Set: ghost zone
    if (ox > 0) { s = ve + 1; e = ve + ng; }
    else        { s = vs - ng; e = vs - 1; }
  }
}

//========================================================================================
// Per-dimension helpers for VC.
// VC includes the shared vertex in loads, and uses precomputed ghost range indices.
//========================================================================================

// Load same-level (VC): includes shared vertex at boundary.
//   ox > 0 -> [ge, ve] (from ghost-load boundary through interior end)
//   ox < 0 -> [vs, gs] (from interior start through ghost-load boundary)
//   ox == 0 -> [vs, ve] (full interior including both boundary vertices)
inline void LoadRangeVC(int ox, int &s, int &e,
                        int vs, int ve, int gs, int ge) {
  s = (ox > 0) ? ge : vs;
  e = (ox < 0) ? gs : ve;
}

// Set same-level (VC): includes shared vertex (additive unpack).
//   ox > 0 -> [ve, pe] (shared vertex through right ghost end)
//   ox < 0 -> [ms, vs] (left ghost start through shared vertex)
//   ox == 0 -> [vs, ve]
inline void SetRangeVC(int ox, int &s, int &e,
                       int vs, int ve, int ms, int pe) {
  if (ox == 0)     { s = vs; e = ve; }
  else if (ox > 0) { s = ve; e = pe; }
  else             { s = ms; e = vs; }
}

// Set from coarser (VC): uses precomputed coarse ghost ranges with lx parity.
//   ox > 0 -> [cps, cpe] (right coarse ghost)
//   ox < 0 -> [cms, cme] (left coarse ghost)
//   ox == 0 -> lx parity selects: even -> [cvs, cpe], odd -> [cms, cve]
inline void SetFromCoarserRangeVC(int ox, int &s, int &e,
                                  int cvs, int cve,
                                  int cms, int cme, int cps, int cpe,
                                  std::int64_t lx, bool is_nontrivial) {
  if (ox == 0) {
    s = cvs; e = cve;
    if (is_nontrivial) {
      if ((lx & 1LL) == 0LL) { e = cpe; }
      else                    { s = cms; }
    }
  } else if (ox > 0) {
    s = cps; e = cpe;
  } else {
    s = cms; e = cme;
  }
}

// Load to coarser (VC): packs from coarse buffer with fine-level ghost width ng.
// Includes the shared vertex - slab width is ng+1, not ng.
//   ox > 0 -> [ve-ng, ve]
//   ox < 0 -> [vs, vs+ng]
//   ox == 0 -> [vs, ve]
inline void LoadToCoarserRangeVC(int ox, int &s, int &e,
                                  int vs, int ve, int ng) {
  if (ox > 0)      { s = ve - ng; e = ve; }
  else if (ox < 0) { s = vs;      e = vs + ng; }
  else             { s = vs;      e = ve; }
}

// Load to finer (VC): excludes boundary vertex for ox!=0.
//   ox > 0 -> [ve-cng, ve-1]
//   ox < 0 -> [vs+1, vs+cng]
//   ox == 0 -> [vs, ve] narrowed by half_size via fi
inline void LoadToFinerRangeVC(int ox, int &s, int &e,
                               int vs, int ve, int cng,
                               int fi1, int fi2, int half_size,
                               bool is_nontrivial, bool use_fi1) {
  if (ox > 0) {
    s = ve - cng; e = ve - 1;
  } else if (ox < 0) {
    s = vs + 1; e = vs + cng;
  } else {
    s = vs; e = ve;
    if (is_nontrivial) {
      int fi = use_fi1 ? fi1 : fi2;
      if (fi == 1) s += half_size;
      else         e -= half_size;
    }
  }
}

// Set from finer (VC): includes shared vertex (same pattern as set same-level).
//   ox > 0 -> [ve, pe] (shared vertex through ghost end)
//   ox < 0 -> [ms, vs] (ghost start through shared vertex)
//   ox == 0 -> [vs, ve] narrowed by half_size via fi
inline void SetFromFinerRangeVC(int ox, int &s, int &e,
                                int vs, int ve, int ms, int pe,
                                int fi1, int fi2, int half_size,
                                bool is_nontrivial, bool use_fi1) {
  if (ox == 0) {
    s = vs; e = ve;
    if (is_nontrivial) {
      int fi = use_fi1 ? fi1 : fi2;
      if (fi == 1) s += half_size;
      else         e -= half_size;
    }
  } else if (ox > 0) {
    s = ve; e = pe;
  } else {
    s = ms; e = vs;
  }
}

//========================================================================================
// Per-dimension helpers for CX.
// CX has a gap at the boundary (like CC) but uses distinct ghost-load indices
// and a ghost-width offset (cx_dng) for cross-level communication.
//========================================================================================
// Load same-level (CX): uses CX ghost-load indices.
//   ox > 0 -> [ge, ie] (ghost-load boundary through interior end)
//   ox < 0 -> [is, gs] (interior start through ghost-load boundary)
//   ox == 0 -> [is, ie]
inline void LoadRangeCX(int ox, int &s, int &e,
                        int is, int ie, int gs, int ge) {
  s = (ox > 0) ? ge : is;
  e = (ox < 0) ? gs : ie;
}

// Set same-level (CX): has a gap at the boundary.
//   ox > 0 -> [ie+1, pe] (one past interior end through ghost end)
//   ox < 0 -> [ms, is-1] (ghost start through one before interior start)
//   ox == 0 -> [is, ie]
inline void SetRangeCX(int ox, int &s, int &e,
                       int is, int ie, int ms, int pe) {
  if (ox == 0)     { s = is; e = ie; }
  else if (ox > 0) { s = ie + 1; e = pe; }
  else             { s = ms; e = is - 1; }
}

// Load to coarser (CX): adjusts for different ghost width at coarser level.
//   ox > 0 -> [cx_cige + cx_dng, cx_cie]
//   ox < 0 -> [cx_cis, cx_cigs - cx_dng]
//   ox == 0 -> [cx_cis, cx_cie]
inline void LoadToCoarserRangeCX(int ox, int &s, int &e,
                                 int cis, int cie, int cigs, int cige,
                                 int dng) {
  s = (ox > 0) ? (cige + dng) : cis;
  e = (ox < 0) ? (cigs - dng) : cie;
}

// Set from coarser (CX): uses precomputed coarse ghost ranges with lx parity.
// Same structure as VC's SetFromCoarserRangeVC but with CX indices.
inline void SetFromCoarserRangeCX(int ox, int &s, int &e,
                                  int cvs, int cve,
                                  int cms, int cme, int cps, int cpe,
                                  std::int64_t lx, bool is_nontrivial) {
  if (ox == 0) {
    s = cvs; e = cve;
    if (is_nontrivial) {
      if ((lx & 1LL) == 0LL) { e = cpe; }
      else                    { s = cms; }
    }
  } else if (ox > 0) {
    s = cps; e = cpe;
  } else {
    s = cms; e = cme;
  }
}

// Load to finer (CX): adjusts for cx_dng offset.
//   ox > 0 -> [cx_ige - cx_dng, cx_ie]
//   ox < 0 -> [cx_is, cx_igs + cx_dng]
//   ox == 0 -> [cx_is, cx_ie] narrowed by NCGHOST_CX half-size
inline void LoadToFinerRangeCX(int ox, int &s, int &e,
                               int is, int ie, int igs, int ige, int dng,
                               int fi1, int fi2, int half_size,
                               bool is_nontrivial, bool use_fi1) {
  if (ox > 0) {
    s = ige - dng; e = ie;
  } else if (ox < 0) {
    s = is; e = igs + dng;
  } else {
    s = is; e = ie;
    if (is_nontrivial) {
      int fi = use_fi1 ? fi1 : fi2;
      if (fi == 1) s += half_size;
      else         e -= half_size;
    }
  }
}

// Set from finer (CX): gap at boundary (like set same-level).
//   ox > 0 -> [ie+1, pe]
//   ox < 0 -> [ms, is-1]
//   ox == 0 -> [is, ie] narrowed by half_size via fi
inline void SetFromFinerRangeCX(int ox, int &s, int &e,
                                int is, int ie, int ms, int pe,
                                int fi1, int fi2, int half_size,
                                bool is_nontrivial, bool use_fi1) {
  if (ox == 0) {
    s = is; e = ie;
    if (is_nontrivial) {
      int fi = use_fi1 ? fi1 : fi2;
      if (fi == 1) s += half_size;
      else         e -= half_size;
    }
  } else if (ox > 0) {
    s = ie + 1; e = pe;
  } else {
    s = ms; e = is - 1;
  }
}

//========================================================================================
// Per-dimension helpers for FC (face-centered / staggered fields).
// Each helper takes is_stagger (true when this dimension is the stagger direction)
// and is_dim_nontrivial (false for degenerate 1D/2D dims where the +1 face offset
// does not apply).  Non-stagger branches delegate to CC logic.
//========================================================================================

// Load same-level / Load to-coarser (FC).
// Stagger: ox==0 includes boundary face (+1); ox!=0 excludes boundary face.
inline void LoadRangeFC(int ox, int &s, int &e,
                        int vs, int ve, int ng,
                        bool is_stagger, bool is_dim_nontrivial) {
  if (!is_stagger) {
    LoadRangeCC(ox, s, e, vs, ve, ng);
  } else {
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

// Set same-level (FC).
// Stagger: ox>0 skips shared boundary face (ve+2); ox<0 standard ghost.
inline void SetRangeFC(int ox, int &s, int &e,
                       int vs, int ve, int ng,
                       bool is_stagger, bool is_dim_nontrivial) {
  if (!is_stagger) {
    SetRangeCC(ox, s, e, vs, ve, ng);
  } else {
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

// Set from coarser (FC): unpack into coarse buffer.
// Stagger: boundary face included (ox!=0 spans cng+1); ox==0 gets +1 then parity extend.
inline void SetFromCoarserRangeFC(int ox, int &s, int &e,
                                  int cvs, int cve, int cng,
                                  std::int64_t lx, bool is_nontrivial,
                                  bool is_stagger) {
  if (!is_stagger) {
    SetFromCoarserRangeCC(ox, s, e, cvs, cve, cng, lx, is_nontrivial);
  } else {
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

// Load to finer (FC).
// Stagger: ox!=0 includes boundary face (span = ng+1); ox==0 gets +1 then fi narrowing.
inline void LoadToFinerRangeFC(int ox, int &s, int &e,
                               int vs, int ve, int ng,
                               int fi1, int fi2, int half_size,
                               bool is_nontrivial, bool use_fi1,
                               bool is_stagger) {
  if (!is_stagger) {
    FinerRangeCC(ox, s, e, vs, ve, ng, fi1, fi2, half_size,
               is_nontrivial, use_fi1, /*is_load=*/true);
  } else {
    if (ox == 0) {
      s = vs;
      e = is_nontrivial ? (ve + 1) : ve;
      if (is_nontrivial) {
        int fi = use_fi1 ? fi1 : fi2;
        if (fi == 1) s += half_size;
        else         e -= half_size;
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

// Set from finer (FC): ghost-zone destination.
// Stagger: ox>0 skips shared face (ve+2); ox==0 gets +1 then fi narrowing.
inline void SetFromFinerRangeFC(int ox, int &s, int &e,
                                int vs, int ve, int ng,
                                int fi1, int fi2, int half_size,
                                bool is_nontrivial, bool use_fi1,
                                bool is_stagger) {
  if (!is_stagger) {
    FinerRangeCC(ox, s, e, vs, ve, ng, fi1, fi2, half_size,
               is_nontrivial, use_fi1, /*is_load=*/false);
  } else {
    if (ox == 0) {
      s = vs;
      e = is_nontrivial ? (ve + 1) : ve;
      if (is_nontrivial) {
        int fi = use_fi1 ? fi1 : fi2;
        if (fi == 1) s += half_size;
        else         e -= half_size;
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

//----------------------------------------------------------------------------------------
// 3D index range computation: dispatches per-dimension helpers based on context.
// These are the public API for pack/unpack.
// All functions now take a Sampling parameter to select correct base indices.
//
// FC requires a stagger_axis parameter (0=x1f, 1=x2f, 2=x3f) to select which
// face component's index ranges to compute.  Caller invokes each function 3 times.

// Same-level pack: source indices in fine or coarse array.
IndexRange3D LoadSameLevel(const MeshBlock *pmb, const NeighborIndexes &ni,
                           int nghost, bool is_coarse,
                           Sampling sampling = Sampling::CC,
                           int stagger_axis = -1);

// Same-level unpack: destination indices in fine or coarse array.
// type: 1=fine, 2=coarse same-level payload
IndexRange3D SetSameLevel(const MeshBlock *pmb, const NeighborIndexes &ni,
                          int nghost, int type,
                          Sampling sampling = Sampling::CC,
                          int stagger_axis = -1);

// To-coarser pack: source indices in coarse buffer.
// For FC, ni_type is needed for edge/corner overlap adjustment.
IndexRange3D LoadToCoarser(const MeshBlock *pmb, const NeighborIndexes &ni,
                           int nghost,
                           Sampling sampling = Sampling::CC,
                           int stagger_axis = -1);

// From-coarser unpack: destination indices in coarse buffer.
IndexRange3D SetFromCoarser(const MeshBlock *pmb, const NeighborIndexes &ni,
                            int nghost,
                            Sampling sampling = Sampling::CC,
                            int stagger_axis = -1);

// To-finer pack: source indices in fine array.
IndexRange3D LoadToFiner(const MeshBlock *pmb, const NeighborIndexes &ni,
                         int nghost,
                         Sampling sampling = Sampling::CC,
                         int stagger_axis = -1);

// From-finer unpack: destination indices in fine array.
// For FC, ni_type is needed for edge/corner overlap adjustment.
IndexRange3D SetFromFiner(const MeshBlock *pmb, const NeighborIndexes &ni,
                          int nghost,
                          Sampling sampling = Sampling::CC,
                          int stagger_axis = -1);

// Compute buffer size for one neighbor by computing actual index ranges.
// This replaces the CC-only analytical formula and works for all samplings.
// FC sums 3 components internally (one per stagger_axis).
int ComputeBufferSizeFromRanges(const MeshBlock *pmb, const NeighborIndexes &ni,
                                int nvar, int nghost, Sampling sampling);

// Compute the MPI message size for one same-level neighbor, which may be smaller
// than the buffer allocation size when the coarse payload can be skipped.
// When skip_coarse is true and the mesh is multilevel, the same-level size omits
// the coarse payload component.  Cross-level sizes are unaffected.
// This is the persistent-MPI counterpart of ComputeBufferSizeFromRanges.
int ComputeMPIBufferSize(const MeshBlock *pmb, const NeighborIndexes &ni,
                         int nvar, int nghost, Sampling sampling,
                         bool skip_coarse);

//========================================================================================
// Flux correction buffer size computation.
// Flux correction buffers are separate from ghost-exchange buffers and are only
// allocated for face neighbors (CC) or face+edge neighbors (FC EMF).
//========================================================================================

// CC flux correction buffer size: restricted transverse face area.
// (nx_perp1+1)/2 * (nx_perp2+1)/2 * nvar per face neighbor.
// Returns 0 for non-face neighbors.
int ComputeFluxCorrBufferSizeCC(const MeshBlock *pmb, const NeighborIndexes &ni,
                                int nvar);

// FC EMF flux correction buffer size: tangential EMF edges on shared face/edge.
// Face neighbors: two tangential components, each with staggered transverse size.
// Edge neighbors: one parallel component along the edge.
// Returns 0 for corner neighbors.
int ComputeFluxCorrBufferSizeFC(const MeshBlock *pmb, const NeighborIndexes &ni);

} // namespace idx
} // namespace comm

#endif // COMM_INDEX_UTILITIES_HPP_
