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
#include "bvals_cc.hpp"

//=============================================================================
// Index-range helper functions for CC boundary variable pack / unpack
//=============================================================================

//----------------------------------------------------------------------------------------
// Per-dimension inline helper: LoadSameLevel / LoadToCoarser
//   ox>0  -> [ve - ng + 1, ve]        (high-end interior slab)
//   ox<0  -> [vs, vs + ng - 1]        (low-end interior slab)
//   ox==0 -> [vs, ve]                 (full interior span)
inline void SetIndexRangesLoad_CC(
    int ox, int &s, int &e,
    int vs, int ve, int ng) {
  s = (ox > 0) ? (ve - ng + 1) : vs;
  e = (ox < 0) ? (vs + ng - 1) : ve;
}

//----------------------------------------------------------------------------------------
// Per-dimension inline helper: SetSameLevel (unpack destination)
//   ox>0  -> [ve + 1, ve + ng]        (high ghost zone)
//   ox<0  -> [vs - ng, vs - 1]        (low ghost zone)
//   ox==0 -> [vs, ve]                 (interior)
inline void SetIndexRangesSet_CC(
    int ox, int &s, int &e,
    int vs, int ve, int ng) {
  if (ox == 0)     { s = vs;        e = ve; }
  else if (ox > 0) { s = ve + 1;    e = ve + ng; }
  else             { s = vs - ng;   e = vs - 1; }
}

//----------------------------------------------------------------------------------------
// Per-dimension inline helper: SetFromCoarser (unpack into coarse_buf)
//   ox>0  -> [cve + 1, cve + cng]
//   ox<0  -> [cvs - cng, cvs - 1]
//   ox==0 -> [cvs, cve] extended by cng via lx parity (if nontrivial dim)
inline void SetIndexRangesFromCoarser_CC(
    int ox, int &s, int &e,
    int cvs, int cve, int cng,
    std::int64_t lx, bool is_nontrivial) {
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
}

//----------------------------------------------------------------------------------------
// Per-dimension inline helper: SetFromFiner (unpack destination)
//   ox>0  -> [ve + 1, ve + ng]  (high ghost zone)
//   ox<0  -> [vs - ng, vs - 1]  (low ghost zone)
//   ox==0 -> [vs, ve] then narrow by half_size based on fi1/fi2
inline void SetIndexRangesSetFromFiner_CC(
    int ox, int &s, int &e,
    int vs, int ve, int ng,
    int fi1, int fi2, int half_size,
    bool is_nontrivial, bool use_fi1) {
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
}

//----------------------------------------------------------------------------------------
// Per-dimension inline helper: LoadToFiner (pack source - interior slab)
//   ox>0  -> [ve - ng + 1, ve]  (high-end interior slab, ng cells)
//   ox<0  -> [vs, vs + ng - 1]  (low-end interior slab, ng cells)
//   ox==0 -> [vs, ve] then narrow by half_size based on fi1/fi2
inline void SetIndexRangesLoadToFiner_CC(
    int ox, int &s, int &e,
    int vs, int ve, int ng,
    int fi1, int fi2, int half_size,
    bool is_nontrivial, bool use_fi1) {
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
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::idxLoadSameLevelRanges(...)
//  \brief Compute index ranges for LoadBoundaryBufferSameLevel
void CellCenteredBoundaryVariable::idxLoadSameLevelRanges(
    const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
    bool is_coarse) {
  MeshBlock *pmb = pmy_block_;
  if (!is_coarse) {
    SetIndexRangesLoad_CC(ni.ox1, si, ei, pmb->is, pmb->ie, NGHOST);
    SetIndexRangesLoad_CC(ni.ox2, sj, ej, pmb->js, pmb->je, NGHOST);
    SetIndexRangesLoad_CC(ni.ox3, sk, ek, pmb->ks, pmb->ke, NGHOST);
  } else {
    int cng = pmb->cnghost;
    SetIndexRangesLoad_CC(ni.ox1, si, ei, pmb->cis, pmb->cie, cng);
    SetIndexRangesLoad_CC(ni.ox2, sj, ej, pmb->cjs, pmb->cje, cng);
    SetIndexRangesLoad_CC(ni.ox3, sk, ek, pmb->cks, pmb->cke, cng);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::idxLoadToCoarserRanges(...)
//  \brief Compute index ranges for LoadBoundaryBufferToCoarser
void CellCenteredBoundaryVariable::idxLoadToCoarserRanges(
    const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek) {
  MeshBlock *pmb = pmy_block_;
  // Original code used cn = NGHOST - 1 as a max offset (0-based), giving a
  // span of cn+1 = NGHOST cells.  SetIndexRangesLoad_CC treats ng as the
  // cell count, so we pass NGHOST directly.
  SetIndexRangesLoad_CC(ni.ox1, si, ei, pmb->cis, pmb->cie, NGHOST);
  SetIndexRangesLoad_CC(ni.ox2, sj, ej, pmb->cjs, pmb->cje, NGHOST);
  SetIndexRangesLoad_CC(ni.ox3, sk, ek, pmb->cks, pmb->cke, NGHOST);
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::idxLoadToFinerRanges(...)
//  \brief Compute index ranges for LoadBoundaryBufferToFiner
void CellCenteredBoundaryVariable::idxLoadToFinerRanges(
    const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek) {
  MeshBlock *pmb = pmy_block_;
  // For ox!=0: interior slab of cnghost cells (same as Load pattern).
  // For ox==0: full interior narrowed by fi1/fi2 with half_size = nx/2 - cnghost.
  int cng = pmb->cnghost;
  // x1: always nontrivial, always use fi1 directly
  SetIndexRangesLoadToFiner_CC(ni.ox1, si, ei, pmb->is, pmb->ie, cng,
      ni.fi1, ni.fi2, pmb->block_size.nx1/2 - cng,
      true, true);
  // x2: use fi1 when ox1!=0, fi2 otherwise
  SetIndexRangesLoadToFiner_CC(ni.ox2, sj, ej, pmb->js, pmb->je, cng,
      ni.fi1, ni.fi2, pmb->block_size.nx2/2 - cng,
      pmb->block_size.nx2 > 1, ni.ox1 != 0);
  // x3: use fi1 when (ox1!=0 && ox2!=0), fi2 otherwise
  SetIndexRangesLoadToFiner_CC(ni.ox3, sk, ek, pmb->ks, pmb->ke, cng,
      ni.fi1, ni.fi2, pmb->block_size.nx3/2 - cng,
      pmb->block_size.nx3 > 1, (ni.ox1 != 0 && ni.ox2 != 0));
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::idxSetSameLevelRanges(...)
//  \brief Compute index ranges for SetBoundarySameLevel
//  type=1: fine data, type=2: coarse same-level payload
void CellCenteredBoundaryVariable::idxSetSameLevelRanges(
    const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
    int type) {
  MeshBlock *pmb = pmy_block_;
  if (type == 1) {
    SetIndexRangesSet_CC(ni.ox1, si, ei, pmb->is, pmb->ie, NGHOST);
    SetIndexRangesSet_CC(ni.ox2, sj, ej, pmb->js, pmb->je, NGHOST);
    SetIndexRangesSet_CC(ni.ox3, sk, ek, pmb->ks, pmb->ke, NGHOST);
  } else {
    int cng = pmb->cnghost;
    SetIndexRangesSet_CC(ni.ox1, si, ei, pmb->cis, pmb->cie, cng);
    SetIndexRangesSet_CC(ni.ox2, sj, ej, pmb->cjs, pmb->cje, cng);
    SetIndexRangesSet_CC(ni.ox3, sk, ek, pmb->cks, pmb->cke, cng);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::idxSetFromCoarserRanges(...)
//  \brief Compute index ranges for SetBoundaryFromCoarser
void CellCenteredBoundaryVariable::idxSetFromCoarserRanges(
    const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek) {
  MeshBlock *pmb = pmy_block_;
  int cng = pmb->cnghost;
  SetIndexRangesFromCoarser_CC(ni.ox1, si, ei,
      pmb->cis, pmb->cie, cng, pmb->loc.lx1, true);
  SetIndexRangesFromCoarser_CC(ni.ox2, sj, ej,
      pmb->cjs, pmb->cje, cng, pmb->loc.lx2, pmb->block_size.nx2 > 1);
  SetIndexRangesFromCoarser_CC(ni.ox3, sk, ek,
      pmb->cks, pmb->cke, cng, pmb->loc.lx3, pmb->block_size.nx3 > 1);
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::idxSetFromFinerRanges(...)
//  \brief Compute index ranges for SetBoundaryFromFiner
void CellCenteredBoundaryVariable::idxSetFromFinerRanges(
    const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek) {
  MeshBlock *pmb = pmy_block_;
  SetIndexRangesSetFromFiner_CC(ni.ox1, si, ei, pmb->is, pmb->ie, NGHOST,
      ni.fi1, ni.fi2, pmb->block_size.nx1/2,
      true, true);
  SetIndexRangesSetFromFiner_CC(ni.ox2, sj, ej, pmb->js, pmb->je, NGHOST,
      ni.fi1, ni.fi2, pmb->block_size.nx2/2,
      pmb->block_size.nx2 > 1, ni.ox1 != 0);
  SetIndexRangesSetFromFiner_CC(ni.ox3, sk, ek, pmb->ks, pmb->ke, NGHOST,
      ni.fi1, ni.fi2, pmb->block_size.nx3/2,
      pmb->block_size.nx3 > 1, (ni.ox1 != 0 && ni.ox2 != 0));
}

//=============================================================================
// MPI buffer size helpers
//=============================================================================
#ifdef MPI_PARALLEL

int CellCenteredBoundaryVariable::MPI_BufferSizeSameLevel(
    const NeighborIndexes& ni, bool skip_coarse) {
  MeshBlock *pmb = pmy_block_;
  int cng1, cng2, cng3;
  cng1 = pmb->cnghost;
  cng2 = cng1*(pmb->block_size.nx2 > 1 ? 1 : 0);
  cng3 = cng1*(pmb->block_size.nx3 > 1 ? 1 : 0);

  int size = ((ni.ox1 == 0) ? pmb->block_size.nx1 : NGHOST)
           * ((ni.ox2 == 0) ? pmb->block_size.nx2 : NGHOST)
           * ((ni.ox3 == 0) ? pmb->block_size.nx3 : NGHOST);
  if (pmy_mesh_->multilevel && !skip_coarse) {
    size += ((ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1)/2) : cng1)
          * ((ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1)/2) : cng2)
          * ((ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1)/2) : cng3);
  }
  return size * (nu_ + 1);
}

int CellCenteredBoundaryVariable::MPI_BufferSizeToCoarser(
    const NeighborIndexes& ni) {
  MeshBlock *pmb = pmy_block_;
  int size = ((ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1)/2) : NGHOST)
           * ((ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1)/2) : NGHOST)
           * ((ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1)/2) : NGHOST);
  return size * (nu_ + 1);
}

int CellCenteredBoundaryVariable::MPI_BufferSizeFromCoarser(
    const NeighborIndexes& ni) {
  MeshBlock *pmb = pmy_block_;
  int cng1, cng2, cng3;
  cng1 = pmb->cnghost;
  cng2 = cng1*(pmb->block_size.nx2 > 1 ? 1 : 0);
  cng3 = cng1*(pmb->block_size.nx3 > 1 ? 1 : 0);
  int size = ((ni.ox1 == 0) ? ((pmb->block_size.nx1 + 1)/2 + cng1) : cng1)
           * ((ni.ox2 == 0) ? ((pmb->block_size.nx2 + 1)/2 + cng2) : cng2)
           * ((ni.ox3 == 0) ? ((pmb->block_size.nx3 + 1)/2 + cng3) : cng3);
  return size * (nu_ + 1);
}

int CellCenteredBoundaryVariable::MPI_BufferSizeToFiner(
    const NeighborIndexes& ni) {
  // Same as FromCoarser: parent sends interior+ghost slab
  return MPI_BufferSizeFromCoarser(ni);
}

int CellCenteredBoundaryVariable::MPI_BufferSizeFromFiner(
    const NeighborIndexes& ni) {
  // Same as ToCoarser: child sends restricted data
  return MPI_BufferSizeToCoarser(ni);
}

#endif // MPI_PARALLEL

//-----------------------------------------------------------------------------
// Prolongation index helpers (existing)
//-----------------------------------------------------------------------------

void CellCenteredBoundaryVariable::CalculateProlongationIndices(
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

void CellCenteredBoundaryVariable::CalculateProlongationIndicesFine(
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