//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file idx_utilities_vc.cpp
//  \brief Various utitilies for indicies and buffers

// C headers

// C++ headers
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../mesh/mesh.hpp"
#include "bvals_cx.hpp"

//----------------------------------------------------------------------------------------
//! \fn inline void CellCenteredXBoundaryVariable::AccumulateBufferSize(...)
//  \brief Buffer size accumulator
inline void CellCenteredXBoundaryVariable::AccumulateBufferSize(
  int sn, int en,
  int si, int ei, int sj, int ej, int sk, int ek,
  int &offset, int ijk_step=1) {

  for (int n=sn; n<=en; ++n) {
    for (int k=sk; k<=ek; k+=ijk_step) {
      for (int j=sj; j<=ej; j+=ijk_step) {
//#pragma omp parallel for shared(offset, si, ei)
        for (int i=si; i<=ei; i+=ijk_step)
          offset++;
      }
    }
  }

}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredXBoundaryVariable::idxLoadSameLevelRanges(...)
//  \brief Compute indicial ranges for LoadBoundaryBufferSameLevel
void CellCenteredXBoundaryVariable::idxLoadSameLevelRanges(
  const NeighborIndexes& ni,
  int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
  bool is_coarse=false) {

  MeshBlock *pmb = pmy_block_;

  if (!is_coarse) {
    si = (ni.ox1 > 0) ? pmb->cx_ige : pmb->cx_is;
    ei = (ni.ox1 < 0) ? pmb->cx_igs : pmb->cx_ie;

    sj = (ni.ox2 > 0) ? pmb->cx_jge : pmb->cx_js;
    ej = (ni.ox2 < 0) ? pmb->cx_jgs : pmb->cx_je;

    sk = (ni.ox3 > 0) ? pmb->cx_kge : pmb->cx_ks;
    ek = (ni.ox3 < 0) ? pmb->cx_kgs : pmb->cx_ke;

    // shared vertex is packed
    // si = (ni.ox1 > 0) ? pmb->ige : pmb->ivs;
    // ei = (ni.ox1 < 0) ? pmb->igs : pmb->ive;

    // sj = (ni.ox2 > 0) ? pmb->jge : pmb->jvs;
    // ej = (ni.ox2 < 0) ? pmb->jgs : pmb->jve;

    // sk = (ni.ox3 > 0) ? pmb->kge : pmb->kvs;
    // ek = (ni.ox3 < 0) ? pmb->kgs : pmb->kve;

  } else {
    si = (ni.ox1 > 0) ? pmb->cx_cige : pmb->cx_cis;
    ei = (ni.ox1 < 0) ? pmb->cx_cigs : pmb->cx_cie;

    sj = (ni.ox2 > 0) ? pmb->cx_cjge : pmb->cx_cjs;
    ej = (ni.ox2 < 0) ? pmb->cx_cjgs : pmb->cx_cje;

    sk = (ni.ox3 > 0) ? pmb->cx_ckge : pmb->cx_cks;
    ek = (ni.ox3 < 0) ? pmb->cx_ckgs : pmb->cx_cke;

    // // [coarse] shared vertex is packed
    // si = (ni.ox1 > 0) ? pmb->cige : pmb->civs;
    // ei = (ni.ox1 < 0) ? pmb->cigs : pmb->cive;

    // sj = (ni.ox2 > 0) ? pmb->cjge : pmb->cjvs;
    // ej = (ni.ox2 < 0) ? pmb->cjgs : pmb->cjve;

    // sk = (ni.ox3 > 0) ? pmb->ckge : pmb->ckvs;
    // ek = (ni.ox3 < 0) ? pmb->ckgs : pmb->ckve;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredXBoundaryVariable::idxLoadToCoarserRanges(...)
//  \brief Compute indicial ranges for LoadBoundaryBufferToCoarser
void CellCenteredXBoundaryVariable::idxLoadToCoarserRanges(
  const NeighborIndexes& ni,
  int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
  bool is_coarse=false) {

  MeshBlock *pmb = pmy_block_;

  // coarser level may have different sized fund. ghost zone
  si = (ni.ox1 > 0) ? (pmb->cx_cige + pmb->cx_dng) : pmb->cx_cis;
  ei = (ni.ox1 < 0) ? (pmb->cx_cigs - pmb->cx_dng) : pmb->cx_cie;

  sj = (ni.ox2 > 0) ? (pmb->cx_cjge + pmb->cx_dng) : pmb->cx_cjs;
  ej = (ni.ox2 < 0) ? (pmb->cx_cjgs - pmb->cx_dng) : pmb->cx_cje;

  sk = (ni.ox3 > 0) ? (pmb->cx_ckge + pmb->cx_dng) : pmb->cx_cks;
  ek = (ni.ox3 < 0) ? (pmb->cx_ckgs - pmb->cx_dng) : pmb->cx_cke;


  // int cng = 2 * pmb->cng;  // "2 for coarse-coarse"
  // // si = (ni.ox1 > 0) ? (pmb->cive - cng) : pmb->civs;
  // // ei = (ni.ox1 < 0) ? (pmb->civs + cng) : pmb->cive;

  // sj = (ni.ox2 > 0) ? (pmb->cjve - cng) : pmb->cjvs;
  // ej = (ni.ox2 < 0) ? (pmb->cjvs + cng) : pmb->cjve;

  // sk = (ni.ox3 > 0) ? (pmb->ckve - cng) : pmb->ckvs;
  // ek = (ni.ox3 < 0) ? (pmb->ckvs + cng) : pmb->ckve;

  // if (!is_coarse) {
  //   // vertices that are shared with adjacent MeshBlocks are to be copied to
  //   // coarser level
  //   int ng = pmb->ng;

  //   // si = (ni.ox1 > 0) ? (pmb->cive - ng) : pmb->civs;
  //   // ei = (ni.ox1 < 0) ? (pmb->civs + ng) : pmb->cive;

  //   si = (ni.ox1 > 0) ? pmb->cx_cige + pmb->cx_dng : pmb->cx_cis;
  //   ei = (ni.ox1 < 0) ? pmb->cx_cigs - pmb->cx_dng : pmb->cx_cie;

  //   sj = (ni.ox2 > 0) ? (pmb->cjve - ng) : pmb->cjvs;
  //   ej = (ni.ox2 < 0) ? (pmb->cjvs + ng) : pmb->cjve;

  //   sk = (ni.ox3 > 0) ? (pmb->ckve - ng) : pmb->ckvs;
  //   ek = (ni.ox3 < 0) ? (pmb->ckvs + ng) : pmb->ckve;

  // } else {
  //   si = (ni.ox1 > 0) ? (pmb->cive - cng) : pmb->cx_cis;
  //   ei = (ni.ox1 < 0) ? (pmb->cx_cigs - pmb->cx_dng) : pmb->cive;

  //   int cng = 2 * pmb->cng;  // "2 for coarse-coarse"
  //   // si = (ni.ox1 > 0) ? (pmb->cive - cng) : pmb->civs;
  //   // ei = (ni.ox1 < 0) ? (pmb->civs + cng) : pmb->cive;

  //   sj = (ni.ox2 > 0) ? (pmb->cjve - cng) : pmb->cjvs;
  //   ej = (ni.ox2 < 0) ? (pmb->cjvs + cng) : pmb->cjve;

  //   sk = (ni.ox3 > 0) ? (pmb->ckve - cng) : pmb->ckvs;
  //   ek = (ni.ox3 < 0) ? (pmb->ckvs + cng) : pmb->ckve;
  // }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredXBoundaryVariable::idxLoadToFinerRanges(...)
//  \brief Compute indicial ranges for LoadBoundaryBufferToFiner
void CellCenteredXBoundaryVariable::idxLoadToFinerRanges(
  const NeighborIndexes& ni,
  int &si, int &ei, int &sj, int &ej, int &sk, int &ek) {

  MeshBlock *pmb = pmy_block_;

  // int cng = pmb->cng;

  // int tmp = cng;

  // take into acccount that coarse var. rep. may have distinct ghosts
  if (ni.ox1 > 0)
  {
    // si = pmb->ive-cng, ei = pmb->ive-1;
    si = pmb->cx_ige - pmb->cx_dng;
    ei = pmb->cx_ie;
  }
  else if (ni.ox1 < 0)
  {
    si = pmb->cx_is;
    ei = pmb->cx_igs + pmb->cx_dng;
  }
  else
  {
    si = pmb->cx_is;
    ei = pmb->cx_ie;

    if (ni.fi1 == 1)
      si += pmb->block_size.nx1 / 2 - NCGHOST_CX;
    else
      ei -= pmb->block_size.nx1 / 2 - NCGHOST_CX;

    // si = pmb->ivs, ei = pmb->ive;
    // if (ni.fi1 == 1)
    //   si += pmb->block_size.nx1/2 - tmp;
    // else
    //   ei -= pmb->block_size.nx1/2 - tmp;

  }

  if (ni.ox2 > 0)
  {
    // sj = pmb->jve-cng;
    // ej = pmb->jve-1;
    sj = pmb->cx_jge - pmb->cx_dng;
    ej = pmb->cx_je;
  }
  else if (ni.ox2 < 0)
  {
    // sj = pmb->jvs+1;
    // ej = pmb->jvs+cng;
    sj = pmb->cx_js;
    ej = pmb->cx_jgs + pmb->cx_dng;

  }
  else
  {
    // sj = pmb->jvs;
    // ej = pmb->jve;

    sj = pmb->cx_js;
    ej = pmb->cx_je;

    if (pmb->block_size.nx2 > 1) {
      if (ni.ox1 != 0) {
        if (ni.fi1 == 1)
          sj += pmb->block_size.nx2/2 - NCGHOST_CX;
        else
          ej -= pmb->block_size.nx2/2 - NCGHOST_CX;
      } else {
        if (ni.fi2 == 1)
          sj += pmb->block_size.nx2/2 - NCGHOST_CX;
        else
          ej -= pmb->block_size.nx2/2 - NCGHOST_CX;

      }

      // if (ni.ox1 != 0) {
      //   if (ni.fi1 == 1)
      //     sj += pmb->block_size.nx2/2 - tmp;
      //   else
      //     ej -= pmb->block_size.nx2/2 - tmp;
      // } else {
      //   if (ni.fi2 == 1)
      //     sj += pmb->block_size.nx2/2 - tmp;
      //   else
      //     ej -= pmb->block_size.nx2/2 - tmp;

      // }
    }
  }

  if (ni.ox3 > 0)
  {
    // sk = pmb->kve-cng;
    // ek = pmb->kve-1;

    sk = pmb->cx_kge - pmb->cx_dng;
    ek = pmb->cx_ke;
  }
  else if (ni.ox3 < 0)
  {
    // sk = pmb->kvs+1;
    // ek = pmb->kvs+cng;

    sk = pmb->cx_ks;
    ek = pmb->cx_kgs + pmb->cx_dng;
  }
  else
  {
    // sk = pmb->kvs;
    // ek = pmb->kve;

    sk = pmb->cx_ks;
    ek = pmb->cx_ke;

    if (pmb->block_size.nx3 > 1) {
      if ((ni.ox1 != 0) && (ni.ox2 != 0)) {
        if (ni.fi1 == 1)
          sk += pmb->block_size.nx3/2 - NCGHOST_CX;
        else
          ek -= pmb->block_size.nx3/2 - NCGHOST_CX;
      } else {
        if (ni.fi2 == 1)
          sk += pmb->block_size.nx3/2 - NCGHOST_CX;
        else
          ek -= pmb->block_size.nx3/2 - NCGHOST_CX;

      }
    }

    // if (pmb->block_size.nx3 > 1) {
    //   if ((ni.ox1 != 0) && (ni.ox2 != 0)) {
    //     if (ni.fi1 == 1)
    //       sk += pmb->block_size.nx3/2 - tmp;
    //     else
    //       ek -= pmb->block_size.nx3/2 - tmp;
    //   } else {
    //     if (ni.fi2 == 1)
    //       sk += pmb->block_size.nx3/2 - tmp;
    //     else
    //       ek -= pmb->block_size.nx3/2 - tmp;

    //   }
    // }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn inline void MeshBlock::SetIndexRangesSBSL(...)
//  \brief Set index ranges for a given dimension
inline void CellCenteredXBoundaryVariable::SetIndexRangesSBSL(
  int ox, int &ix_s, int &ix_e,
  int ix_vs, int ix_ve,
  int ix_ms, int ix_pe) {

  if (ox == 0) {
    ix_s = ix_vs;
    ix_e = ix_ve;
  } else if (ox > 0) {
    ix_s = ix_ve+1;
    ix_e = ix_pe;
  } else {
    ix_s = ix_ms;
    ix_e = ix_vs-1;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredXBoundaryVariable::idxSetSameLevelRanges(...)
//  \brief Compute indicial ranges for SetBoundarySameLevel
void CellCenteredXBoundaryVariable::idxSetSameLevelRanges(
  const NeighborIndexes& ni,
  int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
  int type) {
  // type = 1 for fundamental, 2 for coarse

  MeshBlock *pmb = pmy_block_;

  if (type == 1)
  {
    // SetIndexRangesSBSL(ni.ox1, si, ei,
    //                    pmb->cx_ime, pmb->cx_ips,
    //                    pmb->cx_ims, pmb->cx_ipe);


    SetIndexRangesSBSL(ni.ox1, si, ei,
                       pmb->cx_is,  pmb->cx_ie,
                       pmb->cx_ims, pmb->cx_ipe);
    SetIndexRangesSBSL(ni.ox2, sj, ej,
                       pmb->cx_js,  pmb->cx_je,
                       pmb->cx_jms, pmb->cx_jpe);
    SetIndexRangesSBSL(ni.ox3, sk, ek,
                       pmb->cx_ks,  pmb->cx_ke,
                       pmb->cx_kms, pmb->cx_kpe);

    // SetIndexRangesSBSL(ni.ox2, sj, ej,
    //                    pmb->cx_jme, pmb->cx_jps,
    //                    pmb->cx_jms, pmb->cx_jpe);
    // SetIndexRangesSBSL(ni.ox3, sk, ek,
    //                    pmb->cx_kme, pmb->cx_kps,
    //                    pmb->cx_kms, pmb->cx_kpe);

    // SetIndexRangesSBSL(ni.ox2, sj, ej,
    //                    pmb->jvs, pmb->jve,
    //                    pmb->jms, pmb->jpe);
    // SetIndexRangesSBSL(ni.ox3, sk, ek,
    //                    pmb->kvs, pmb->kve, pmb->kms, pmb->kpe);
  }
  else if (type == 2)
  {
    SetIndexRangesSBSL(ni.ox1, si, ei,
                       pmb->cx_cis, pmb->cx_cie,
                       pmb->cx_cims, pmb->cx_cipe);
    SetIndexRangesSBSL(ni.ox2, sj, ej,
                       pmb->cx_cjs, pmb->cx_cje,
                       pmb->cx_cjms, pmb->cx_cjpe);
    SetIndexRangesSBSL(ni.ox3, sk, ek,
                       pmb->cx_cks, pmb->cx_cke,
                       pmb->cx_ckms, pmb->cx_ckpe);

    // SetIndexRangesSBSL(ni.ox1, si, ei,
    //                    pmb->civs, pmb->cive, pmb->cims, pmb->cipe);
    // SetIndexRangesSBSL(ni.ox2, sj, ej,
    //                    pmb->cjvs, pmb->cjve, pmb->cjms, pmb->cjpe);
    // SetIndexRangesSBSL(ni.ox3, sk, ek,
    //                    pmb->ckvs, pmb->ckve, pmb->ckms, pmb->ckpe);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn inline void MeshBlock::SetIndexRangesSBFC(...)
//  \brief Set index ranges for a given dimension
inline void CellCenteredXBoundaryVariable::SetIndexRangesSBFC(
  int ox, int &ix_s, int &ix_e,
  int ix_cvs, int ix_cve, int ix_cms, int ix_cme, int ix_cps, int ix_cpe,
  bool level_flag) {

  if (ox == 0)
  {
    ix_s = ix_cvs; ix_e = ix_cve;
    if (level_flag)
    {
      ix_s = ix_cvs; ix_e = ix_cpe;
    }
    else
    {
      ix_s = ix_cms; ix_e = ix_cve;
    }
  }
  else if (ox > 0)
  {
    ix_s = ix_cps; ix_e = ix_cpe;
  }
  else
  {
    ix_s = ix_cms;
    ix_e = ix_cme;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredXBoundaryVariable::idxSetFromCoarserRanges(...)
//  \brief Compute indicial ranges for SetBoundaryFromCoarser
void CellCenteredXBoundaryVariable::idxSetFromCoarserRanges(
  const NeighborIndexes& ni,
  int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
  bool is_node_mult=false) {

  MeshBlock *pmb = pmy_block_;

  SetIndexRangesSBFC(ni.ox1, si, ei,
                     pmb->cx_cis,
                     pmb->cx_cie,
                     pmb->cx_cims,
                     pmb->cx_cime,
                     pmb->cx_cips,
                     pmb->cx_cipe,
                     (pmb->loc.lx1 & 1LL) == 0LL);

  // [2d] modifies sj, ej
  SetIndexRangesSBFC(ni.ox2, sj, ej,
                     pmb->cx_cjs,
                     pmb->cx_cje,
                     pmb->cx_cjms,
                     pmb->cx_cjme,
                     pmb->cx_cjps,
                     pmb->cx_cjpe,
                     (pmb->loc.lx2 & 1LL) == 0LL);

  // [3d] modifies sk, ek
  SetIndexRangesSBFC(ni.ox3, sk, ek,
                     pmb->cx_cks,
                     pmb->cx_cke,
                     pmb->cx_ckms,
                     pmb->cx_ckme,
                     pmb->cx_ckps,
                     pmb->cx_ckpe,
                     (pmb->loc.lx3 & 1LL) == 0LL);
}

//----------------------------------------------------------------------------------------
//! \fn inline void MeshBlock::SetIndexRangesSBFF(...)
//  \brief Set index ranges for a given dimension
inline void CellCenteredXBoundaryVariable::SetIndexRangesSBFF(
  int ox, int &ix_s, int &ix_e, int ix_vs, int ix_ve, int ix_ms, int ix_pe,
  int fi1, int fi2, int axis_half_size, bool size_flag, bool offset_flag) {

  if (ox == 0)
  {
    ix_s = ix_vs;
    ix_e = ix_ve;

    if (size_flag)
    {
      if (offset_flag)
      {
        if (fi1 == 1) {
          ix_s += axis_half_size;
        } else {
          ix_e -= axis_half_size;
        }
      } else {
        if (fi2 == 1) {
          ix_s += axis_half_size;
        } else {
          ix_e -= axis_half_size;
        }
      }
    }
  }
  else if (ox > 0)
  {
    ix_s = ix_ve+1;
    ix_e = ix_pe;
  }
  else
  {
    ix_s = ix_ms;
    ix_e = ix_vs-1;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredXBoundaryVariable::idxSetFromFinerRanges(...)
//  \brief Compute indicial ranges for SetBoundaryFromFiner
void CellCenteredXBoundaryVariable::idxSetFromFinerRanges(
  const NeighborIndexes& ni,
  int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
  int type) {
  // type = 1 for fundamental, 2 for coarse

  MeshBlock *pmb = pmy_block_;

  if (type == 1)
  {
    SetIndexRangesSBFF(ni.ox1, si, ei,
                       pmb->cx_is, pmb->cx_ie, pmb->cx_ims, pmb->cx_ipe,
                       ni.fi1, ni.fi2,
                       pmb->block_size.nx1 / 2,
                       true,
                       true);

    SetIndexRangesSBFF(ni.ox2, sj, ej,
                       pmb->cx_js, pmb->cx_je, pmb->cx_jms, pmb->cx_jpe,
                       ni.fi1, ni.fi2,
                       pmb->block_size.nx2 / 2,
                       (pmb->block_size.nx2 > 1),
                       (ni.ox1 != 0));

    SetIndexRangesSBFF(ni.ox3, sk, ek,
                       pmb->cx_ks, pmb->cx_ke, pmb->cx_kms, pmb->cx_kpe,
                       ni.fi1, ni.fi2,
                       pmb->block_size.nx3 / 2,
                       (pmb->block_size.nx3 > 1),
                       (ni.ox1 != 0 && ni.ox2 != 0));

    // SetIndexRangesSBFF(ni.ox2, sj, ej,
    //                    pmb->jvs, pmb->jve, pmb->jms, pmb->jpe,
    //                    ni.fi1, ni.fi2,
    //                    pmb->block_size.nx2 / 2,
    //                    (pmb->block_size.nx2 > 1),
    //                    (ni.ox1 != 0));

    // SetIndexRangesSBFF(ni.ox3, sk, ek,
    //                    pmb->kvs, pmb->kve, pmb->kms, pmb->kpe,
    //                    ni.fi1, ni.fi2,
    //                    pmb->block_size.nx3 / 2,
    //                    (pmb->block_size.nx3 > 1),
    //                    (ni.ox1 != 0 && ni.ox2 != 0));

  }
  // else if (type == 2)
  // {
  //   SetIndexRangesSBFF(ni.ox1, si, ei,
  //                      pmb->civs, pmb->cive, pmb->cims, pmb->cipe,
  //                      ni.fi1, ni.fi2,
  //                      pmb->block_size.nx1 / 4,
  //                      true,
  //                      true);

  //   SetIndexRangesSBFF(ni.ox2, sj, ej,
  //                      pmb->cjvs, pmb->cjve, pmb->cjms, pmb->cjpe,
  //                      ni.fi1, ni.fi2,
  //                      pmb->block_size.nx2 / 4,
  //                      (pmb->block_size.nx2 > 1),
  //                      (ni.ox1 != 0));

  //   SetIndexRangesSBFF(ni.ox3, sk, ek,
  //                      pmb->ckvs, pmb->ckve, pmb->ckms, pmb->ckpe,
  //                      ni.fi1, ni.fi2,
  //                      pmb->block_size.nx3 / 4,
  //                      (pmb->block_size.nx3 > 1),
  //                      (ni.ox1 != 0 && ni.ox2 != 0));

  // }
  // else if (type == 3) {
  //   SetIndexRangesSBFF(ni.ox1, si, ei,
  //                      c_ivs, c_ive, c_ims, c_ipe,
  //                      ni.fi1, ni.fi2,
  //                      2,
  //                      true,
  //                      true);

  //   SetIndexRangesSBFF(ni.ox2, sj, ej,
  //                      c_jvs, c_jve, c_jms, c_jpe,
  //                      ni.fi1, ni.fi2,
  //                      2,
  //                      (pmb->block_size.nx2 > 1),
  //                      (ni.ox1 != 0));

  //   SetIndexRangesSBFF(ni.ox3, sk, ek,
  //                      c_kvs, c_kve, c_kms, c_kpe,
  //                      ni.fi1, ni.fi2,
  //                      2,
  //                      (pmb->block_size.nx3 > 1),
  //                      (ni.ox1 != 0 && ni.ox2 != 0));

  // }
}

// For calculation of buffer sizes based on neighbor index information
int CellCenteredXBoundaryVariable::NeighborVariableBufferSize(const NeighborIndexes& ni) {
  // int si, sj, sk, ei, ej, ek;
  int si = 0, sj = 0, sk = 0, ei = 0, ej = 0, ek = 0;
  int size = 0;

  idxLoadSameLevelRanges(ni, si, ei, sj, ej, sk, ek, false);
  AccumulateBufferSize(nl_, nu_, si, ei, sj, ej, sk, ek, size);

  if (pmy_mesh_->multilevel) {
    idxLoadSameLevelRanges(ni, si, ei, sj, ej, sk, ek, true);
    AccumulateBufferSize(nl_, nu_, si, ei, sj, ej, sk, ek, size);

    // as we are multi-level we should also maximize over fine to coarse and vice versa ops.
    int sizef2c = 0;
    idxLoadToCoarserRanges(ni, si, ei, sj, ej, sk, ek, false);
    AccumulateBufferSize(nl_, nu_, si, ei, sj, ej, sk, ek, sizef2c);

    // BD: Is this needed? 2:1 ratio enforced...
    /*
    idxLoadToCoarserRanges(ni, si, ei, sj, ej, sk, ek, true);
    // double restrict means spatial indices jump by two per iterate here
    AccumulateBufferSize(nl_, nu_, si, ei, sj, ej, sk, ek, sizef2c, 2);
    */

    int sizec2f = 0;
    idxLoadToFinerRanges(ni, si, ei, sj, ej, sk, ek);
    AccumulateBufferSize(nl_, nu_, si, ei, sj, ej, sk, ek, sizec2f);

    size = std::max(size, sizef2c);
    size = std::max(size, sizec2f);
  }

  return size;
}

//----------------------------------------------------------------------------------------
// MPI buffer sizes
#ifdef MPI_PARALLEL

int CellCenteredXBoundaryVariable::MPI_BufferSizeSameLevel(
  const NeighborIndexes& ni,
  bool is_send=true) {

  int si, sj, sk, ei, ej, ek;
  int size = 0;

  if (is_send) {
    idxLoadSameLevelRanges(ni, si, ei, sj, ej, sk, ek, false);
    AccumulateBufferSize(nl_, nu_, si, ei, sj, ej, sk, ek, size);

    if (pmy_mesh_->multilevel) {
      idxLoadSameLevelRanges(ni, si, ei, sj, ej, sk, ek, true);
      AccumulateBufferSize(nl_, nu_, si, ei, sj, ej, sk, ek, size);
    }
  } else {
    idxSetSameLevelRanges(ni, si, ei, sj, ej, sk, ek, 1);
    AccumulateBufferSize(nl_, nu_, si, ei, sj, ej, sk, ek, size);

    if (pmy_mesh_->multilevel) {
      idxSetSameLevelRanges(ni, si, ei, sj, ej, sk, ek, 2);
      AccumulateBufferSize(nl_, nu_, si, ei, sj, ej, sk, ek, size);
    }
  }

  return size;
}

int CellCenteredXBoundaryVariable::MPI_BufferSizeToCoarser(
  const NeighborIndexes& ni) {

  int si, sj, sk, ei, ej, ek;
  int size = 0;

  idxLoadToCoarserRanges(ni, si, ei, sj, ej, sk, ek, false);
  AccumulateBufferSize(nl_, nu_, si, ei, sj, ej, sk, ek, size);
  // idxLoadToCoarserRanges(ni, si, ei, sj, ej, sk, ek, true);
  // double restrict means spatial indices jump by two per iterate here
  // AccumulateBufferSize(nl_, nu_, si, ei, sj, ej, sk, ek, size, 2);

  return size;
}

int CellCenteredXBoundaryVariable::MPI_BufferSizeFromCoarser(
  const NeighborIndexes& ni) {

  int si, sj, sk, ei, ej, ek;
  int size = 0;

  idxSetFromCoarserRanges(ni, si, ei, sj, ej, sk, ek, false);
  AccumulateBufferSize(nl_, nu_, si, ei, sj, ej, sk, ek, size);

  return size;
}

int CellCenteredXBoundaryVariable::MPI_BufferSizeToFiner(
  const NeighborIndexes& ni) {

  int si, sj, sk, ei, ej, ek;
  int size = 0;

  idxLoadToFinerRanges(ni, si, ei, sj, ej, sk, ek);
  AccumulateBufferSize(nl_, nu_, si, ei, sj, ej, sk, ek, size);

  return size;
}

int CellCenteredXBoundaryVariable::MPI_BufferSizeFromFiner(
  const NeighborIndexes& ni) {

  int si, sj, sk, ei, ej, ek;
  int size = 0;

  idxSetFromFinerRanges(ni, si, ei, sj, ej, sk, ek, 1);
  AccumulateBufferSize(nl_, nu_, si, ei, sj, ej, sk, ek, size);

  // idxSetFromFinerRanges(ni, si, ei, sj, ej, sk, ek, 2);
  // AccumulateBufferSize(nl_, nu_, si, ei, sj, ej, sk, ek, size);

  return size;
}

#endif
