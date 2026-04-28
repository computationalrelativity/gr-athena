//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file node_multiplicity.cpp
//  \brief NodeMultiplicity implementation: precompute vertex sharing counts from topology.
//
//  Mirrors the old VertexCenteredBoundaryVariable consistency_conditions_vc.cpp logic
//  but precomputes the multiplicity array once at Finalize/Reinitialize time (purely
//  topological - no data dependency).  The compact 7-zone representation is identical
//  to the old system.

#include "node_multiplicity.hpp"

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "neighbor_connectivity.hpp"

namespace comm {

//----------------------------------------------------------------------------------------
// Constructor: invalid until Precompute() is called.

NodeMultiplicity::NodeMultiplicity()
    : valid_(false),
      c_ivs_(c_vs), c_ive_(c_ve), c_ims_(c_ms), c_ipe_(c_pe),
      c_jvs_(c_vs), c_jve_(c_ve), c_jms_(c_ms), c_jpe_(c_pe),
      c_kvs_(c_vs), c_kve_(c_ve), c_kms_(c_ms), c_kpe_(c_pe) {}

//========================================================================================
// Compact-index helpers.
// These map neighbor direction + sub-position into the compact 7-zone space,
// matching the old idx_utilities_vc.cpp type=3 / is_node_mult logic exactly.
//========================================================================================

//----------------------------------------------------------------------------------------
// Same-level compact range.
// ox > 0 -> [c_ve, c_pe] (high ghost + shared vertex)
// ox < 0 -> [c_ms, c_vs] (low ghost + shared vertex)
// ox == 0 -> [c_vs, c_ve] (interior)

void NodeMultiplicity::CompactRangeSameLevel(int ox, int &s, int &e,
                                             int cvs, int cve, int cms, int cpe) {
  if (ox == 0)     { s = cvs; e = cve; }
  else if (ox > 0) { s = cve; e = cpe; }
  else             { s = cms; e = cvs; }
}

//----------------------------------------------------------------------------------------
// From-coarser compact range.
// ox > 0 -> [c_pe, c_pe] (single far ghost point)
// ox < 0 -> [c_ms, c_ms] (single near ghost point)
// ox == 0 -> lx parity selects: even -> [c_vs, c_pe], odd -> [c_ms, c_ve]

void NodeMultiplicity::CompactRangeFromCoarser(int ox, int &s, int &e,
                                               int cvs, int cve, int cms, int cpe,
                                               std::int64_t lx, bool is_nontrivial) {
  if (ox == 0) {
    s = cvs; e = cve;
    if (is_nontrivial) {
      // Extend into ghost on one side based on block parity.
      if ((lx & 1LL) == 0LL) e = cpe;
      else                    s = cms;
    }
  } else if (ox > 0) {
    // Coarser high side: just the boundary point in compact space.
    s = cpe; e = cpe;
  } else {
    // Coarser low side: just the boundary point in compact space.
    s = cms; e = cms;
  }
}

//----------------------------------------------------------------------------------------
// From-finer compact range.
// ox > 0 -> [c_ve, c_pe] (shared vertex through ghost end)
// ox < 0 -> [c_ms, c_vs] (ghost start through shared vertex)
// ox == 0 -> [c_vs, c_ve] narrowed by half_size=2 via fi

void NodeMultiplicity::CompactRangeFromFiner(int ox, int &s, int &e,
                                             int cvs, int cve, int cms, int cpe,
                                             int fi1, int fi2, int half_size,
                                             bool is_nontrivial, bool use_fi1) {
  if (ox == 0) {
    s = cvs; e = cve;
    if (is_nontrivial) {
      int fi = use_fi1 ? fi1 : fi2;
      if (fi == 1) s += half_size;
      else         e -= half_size;
    }
  } else if (ox > 0) {
    s = cve; e = cpe;
  } else {
    s = cms; e = cvs;
  }
}

//========================================================================================
// Precompute: iterate all neighbors and accumulate compact multiplicity.
//========================================================================================

void NodeMultiplicity::Precompute(MeshBlock *pmb, const NeighborConnectivity &nc,
                                  int nghost) {
  const int nx2 = pmb->block_size.nx2;
  const int nx3 = pmb->block_size.nx3;
  const int mylevel = pmb->loc.level;

  // Set per-axis compact index limits.  Degenerate axes collapse to 0.
  c_ivs_ = c_vs; c_ive_ = c_ve; c_ims_ = c_ms; c_ipe_ = c_pe;
  c_jvs_ = c_vs; c_jve_ = c_ve; c_jms_ = c_ms; c_jpe_ = c_pe;
  c_kvs_ = c_vs; c_kve_ = c_ve; c_kms_ = c_ms; c_kpe_ = c_pe;

  if (nx2 == 1) { c_jvs_ = 0; c_jve_ = 0; c_jms_ = 0; c_jpe_ = 0; }
  if (nx3 == 1) { c_kvs_ = 0; c_kve_ = 0; c_kms_ = 0; c_kpe_ = 0; }

  // Allocate compact array based on dimensionality.
  int nk = (nx3 > 1) ? 7 : 1;
  int nj = (nx2 > 1) ? 7 : 1;
  int ni = 7;
  mult_.NewAthenaArray(1, nk, nj, ni);

  // Zero the entire array.
  for (int k = 0; k < nk; ++k)
    for (int j = 0; j < nj; ++j)
      for (int i = 0; i < ni; ++i)
        mult_(0, k, j, i) = 0;

  // Initialize interior to 1 (the block itself contributes once).
  for (int k = c_kvs_; k <= c_kve_; ++k)
    for (int j = c_jvs_; j <= c_jve_; ++j)
      for (int i = c_ivs_; i <= c_ive_; ++i)
        mult_(0, k, j, i) = 1;

  // Compact half-size: in the 7-zone representation the interior spans [1..5] = 4,
  // so half = 2.  This is hardcoded (same as the old system).
  const int compact_half = 2;

  // Accumulate contributions from each neighbor.
  for (int n = 0; n < nc.num_neighbors(); ++n) {
    const NeighborBlock &nb = nc.neighbor(n);
    int si, ei, sj, ej, sk, ek;

    if (nb.snb.level == mylevel) {
      // Same-level neighbor: compact range determined by (ox1,ox2,ox3).
      CompactRangeSameLevel(nb.ni.ox1, si, ei, c_ivs_, c_ive_, c_ims_, c_ipe_);
      CompactRangeSameLevel(nb.ni.ox2, sj, ej, c_jvs_, c_jve_, c_jms_, c_jpe_);
      CompactRangeSameLevel(nb.ni.ox3, sk, ek, c_kvs_, c_kve_, c_kms_, c_kpe_);
    } else if (nb.snb.level < mylevel) {
      // Coarser neighbor: use parity-based compact range.
      CompactRangeFromCoarser(nb.ni.ox1, si, ei,
                              c_ivs_, c_ive_, c_ims_, c_ipe_,
                              pmb->loc.lx1, true);
      CompactRangeFromCoarser(nb.ni.ox2, sj, ej,
                              c_jvs_, c_jve_, c_jms_, c_jpe_,
                              pmb->loc.lx2, (nx2 > 1));
      CompactRangeFromCoarser(nb.ni.ox3, sk, ek,
                              c_kvs_, c_kve_, c_kms_, c_kpe_,
                              pmb->loc.lx3, (nx3 > 1));
    } else {
      // Finer neighbor: use fi1/fi2 sub-position to narrow compact range.
      CompactRangeFromFiner(nb.ni.ox1, si, ei,
                            c_ivs_, c_ive_, c_ims_, c_ipe_,
                            nb.ni.fi1, nb.ni.fi2, compact_half,
                            true, true);
      CompactRangeFromFiner(nb.ni.ox2, sj, ej,
                            c_jvs_, c_jve_, c_jms_, c_jpe_,
                            nb.ni.fi1, nb.ni.fi2, compact_half,
                            (nx2 > 1), (nb.ni.ox1 != 0));
      CompactRangeFromFiner(nb.ni.ox3, sk, ek,
                            c_kvs_, c_kve_, c_kms_, c_kpe_,
                            nb.ni.fi1, nb.ni.fi2, compact_half,
                            (nx3 > 1), (nb.ni.ox1 != 0 && nb.ni.ox2 != 0));
    }

    // Accumulate +1 for all compact zones this neighbor covers.
    for (int k = sk; k <= ek; ++k)
      for (int j = sj; j <= ej; ++j)
        for (int i = si; i <= ei; ++i)
          mult_(0, k, j, i) += 1;
  }

  valid_ = true;
}

//========================================================================================
// ZeroGhosts: zero all ghost zones of a VC array before additive unpack.
// This ensures additive unpack starts from zero in ghost regions.
//========================================================================================

void NodeMultiplicity::ZeroGhosts(AthenaArray<Real> &var, MeshBlock *pmb,
                                  int nvar) const {
  // Ghost zone ranges for VC fine level.
  const int ims = pmb->ims, ipe = pmb->ipe;
  const int ivs = pmb->ivs, ive = pmb->ive;

  const int nx2 = pmb->block_size.nx2;
  const int nx3 = pmb->block_size.nx3;

  if (nx3 > 1) {
    // 3D: zero six ghost slabs (non-overlapping).
    const int jms = pmb->jms, jpe = pmb->jpe;
    const int jvs = pmb->jvs, jve = pmb->jve;
    const int kms = pmb->kms, kpe = pmb->kpe;
    const int kvs = pmb->kvs, kve = pmb->kve;

    // Low/high i-bands (full j,k extent).
    for (int nn = 0; nn < nvar; ++nn)
      for (int k = kms; k <= kpe; ++k)
        for (int j = jms; j <= jpe; ++j) {
          for (int i = ims; i < ivs; ++i) var(nn,k,j,i) = 0.0;
          for (int i = ive+1; i <= ipe; ++i) var(nn,k,j,i) = 0.0;
        }

    // Low/high j-bands (interior i only, full k).
    for (int nn = 0; nn < nvar; ++nn)
      for (int k = kms; k <= kpe; ++k) {
        for (int j = jms; j < jvs; ++j)
          for (int i = ivs; i <= ive; ++i) var(nn,k,j,i) = 0.0;
        for (int j = jve+1; j <= jpe; ++j)
          for (int i = ivs; i <= ive; ++i) var(nn,k,j,i) = 0.0;
      }

    // Low/high k-bands (interior i and j only).
    for (int nn = 0; nn < nvar; ++nn) {
      for (int k = kms; k < kvs; ++k)
        for (int j = jvs; j <= jve; ++j)
          for (int i = ivs; i <= ive; ++i) var(nn,k,j,i) = 0.0;
      for (int k = kve+1; k <= kpe; ++k)
        for (int j = jvs; j <= jve; ++j)
          for (int i = ivs; i <= ive; ++i) var(nn,k,j,i) = 0.0;
    }
  } else if (nx2 > 1) {
    // 2D: zero four ghost bands (non-overlapping).
    const int jms = pmb->jms, jpe = pmb->jpe;
    const int jvs = pmb->jvs, jve = pmb->jve;
    const int ks = pmb->ks;

    // Low/high i-bands (full j).
    for (int nn = 0; nn < nvar; ++nn)
      for (int j = jms; j <= jpe; ++j) {
        for (int i = ims; i < ivs; ++i) var(nn,ks,j,i) = 0.0;
        for (int i = ive+1; i <= ipe; ++i) var(nn,ks,j,i) = 0.0;
      }

    // Low/high j-bands (interior i only).
    for (int nn = 0; nn < nvar; ++nn) {
      for (int j = jms; j < jvs; ++j)
        for (int i = ivs; i <= ive; ++i) var(nn,ks,j,i) = 0.0;
      for (int j = jve+1; j <= jpe; ++j)
        for (int i = ivs; i <= ive; ++i) var(nn,ks,j,i) = 0.0;
    }
  } else {
    // 1D: zero low/high i ghost zones.
    const int js = pmb->js, ks = pmb->ks;
    for (int nn = 0; nn < nvar; ++nn) {
      for (int i = ims; i < ivs; ++i) var(nn,ks,js,i) = 0.0;
      for (int i = ive+1; i <= ipe; ++i) var(nn,ks,js,i) = 0.0;
    }
  }
}

//----------------------------------------------------------------------------------------
// ZeroGhostsCoarse: same as ZeroGhosts but for the coarse VC array.

void NodeMultiplicity::ZeroGhostsCoarse(AthenaArray<Real> &cvar, MeshBlock *pmb,
                                        int nvar) const {
  const int cims = pmb->cims, cipe = pmb->cipe;
  const int civs = pmb->civs, cive = pmb->cive;

  const int nx2 = pmb->block_size.nx2;
  const int nx3 = pmb->block_size.nx3;

  if (nx3 > 1) {
    const int cjms = pmb->cjms, cjpe = pmb->cjpe;
    const int cjvs = pmb->cjvs, cjve = pmb->cjve;
    const int ckms = pmb->ckms, ckpe = pmb->ckpe;
    const int ckvs = pmb->ckvs, ckve = pmb->ckve;

    for (int nn = 0; nn < nvar; ++nn)
      for (int k = ckms; k <= ckpe; ++k)
        for (int j = cjms; j <= cjpe; ++j) {
          for (int i = cims; i < civs; ++i) cvar(nn,k,j,i) = 0.0;
          for (int i = cive+1; i <= cipe; ++i) cvar(nn,k,j,i) = 0.0;
        }

    for (int nn = 0; nn < nvar; ++nn)
      for (int k = ckms; k <= ckpe; ++k) {
        for (int j = cjms; j < cjvs; ++j)
          for (int i = civs; i <= cive; ++i) cvar(nn,k,j,i) = 0.0;
        for (int j = cjve+1; j <= cjpe; ++j)
          for (int i = civs; i <= cive; ++i) cvar(nn,k,j,i) = 0.0;
      }

    for (int nn = 0; nn < nvar; ++nn) {
      for (int k = ckms; k < ckvs; ++k)
        for (int j = cjvs; j <= cjve; ++j)
          for (int i = civs; i <= cive; ++i) cvar(nn,k,j,i) = 0.0;
      for (int k = ckve+1; k <= ckpe; ++k)
        for (int j = cjvs; j <= cjve; ++j)
          for (int i = civs; i <= cive; ++i) cvar(nn,k,j,i) = 0.0;
    }
  } else if (nx2 > 1) {
    const int cjms = pmb->cjms, cjpe = pmb->cjpe;
    const int cjvs = pmb->cjvs, cjve = pmb->cjve;
    const int cks = pmb->cks;

    for (int nn = 0; nn < nvar; ++nn)
      for (int j = cjms; j <= cjpe; ++j) {
        for (int i = cims; i < civs; ++i) cvar(nn,cks,j,i) = 0.0;
        for (int i = cive+1; i <= cipe; ++i) cvar(nn,cks,j,i) = 0.0;
      }

    for (int nn = 0; nn < nvar; ++nn) {
      for (int j = cjms; j < cjvs; ++j)
        for (int i = civs; i <= cive; ++i) cvar(nn,cks,j,i) = 0.0;
      for (int j = cjve+1; j <= cjpe; ++j)
        for (int i = civs; i <= cive; ++i) cvar(nn,cks,j,i) = 0.0;
    }
  } else {
    const int cjs = pmb->cjs, cks = pmb->cks;
    for (int nn = 0; nn < nvar; ++nn) {
      for (int i = cims; i < civs; ++i) cvar(nn,cks,cjs,i) = 0.0;
      for (int i = cive+1; i <= cipe; ++i) cvar(nn,cks,cjs,i) = 0.0;
    }
  }
}

//========================================================================================
// ApplyDivision: divide VC variable by precomputed node multiplicity.
// Uses compact-to-physical mapping to iterate only zones where mult > 1.
//========================================================================================

void NodeMultiplicity::ApplyDivision(AthenaArray<Real> &var, MeshBlock *pmb,
                                     int nvar) const {
  ApplyDivisionImpl(var, nvar,
                    pmb->ims, pmb->ivs, pmb->ive, pmb->ipe,
                    pmb->jms, pmb->jvs, pmb->jve, pmb->jpe,
                    pmb->kms, pmb->kvs, pmb->kve, pmb->kpe);
}

void NodeMultiplicity::ApplyDivisionCoarse(AthenaArray<Real> &cvar, MeshBlock *pmb,
                                           int nvar) const {
  ApplyDivisionImpl(cvar, nvar,
                    pmb->cims, pmb->civs, pmb->cive, pmb->cipe,
                    pmb->cjms, pmb->cjvs, pmb->cjve, pmb->cjpe,
                    pmb->ckms, pmb->ckvs, pmb->ckve, pmb->ckpe);
}

//----------------------------------------------------------------------------------------
// Internal: apply division using the compact-to-physical 7-zone mapping.
// For each compact index (K, J, I), determine the corresponding physical range
// and divide if node_mult > 1.
//
// The physical range mapping is:
//   Odd compact indices (1,3,5) -> single physical index (the pivot point)
//   Even compact indices (0,2,4,6) -> range between neighboring pivots
// This is identical to the old ApplyNodeMultiplicitesDim{1,2,3} logic.

void NodeMultiplicity::ApplyDivisionImpl(AthenaArray<Real> &var, int nvar,
                                         int ims, int ivs, int ive, int ipe,
                                         int jms, int jvs, int jve, int jpe,
                                         int kms, int kvs, int kve, int kpe) const {
  const int nx2 = jpe - jms + 1;
  const int nx3 = kpe - kms + 1;

  // Compact array dimensions for iteration bounds.
  const int cnk = (nx3 > 1) ? 7 : 1;
  const int cnj = (nx2 > 1) ? 7 : 1;
  const int cni = 7;

  // Axis half-size: (ive - ivs) / 2 for physical array.
  const int ihalf = (ive - ivs) / 2;
  const int jhalf = (nx2 > 1) ? (jve - jvs) / 2 : 0;
  const int khalf = (nx3 > 1) ? (kve - kvs) / 2 : 0;

  // Build the compact-to-physical pivot arrays.
  // ti_c[c] gives the "pivot" physical index for compact index c.
  int ti_c[7] = {ims, ivs, ivs + 1, ivs + ihalf, ive - 1, ive, ipe};
  int tj_c[7], tk_c[7];

  if (nx2 > 1) {
    tj_c[0] = jms; tj_c[1] = jvs; tj_c[2] = jvs + 1;
    tj_c[3] = jvs + jhalf; tj_c[4] = jve - 1; tj_c[5] = jve; tj_c[6] = jpe;
  } else {
    tj_c[0] = jms;
    for (int c = 1; c < 7; ++c) tj_c[c] = jms;  // degenerate
  }

  if (nx3 > 1) {
    tk_c[0] = kms; tk_c[1] = kvs; tk_c[2] = kvs + 1;
    tk_c[3] = kvs + khalf; tk_c[4] = kve - 1; tk_c[5] = kve; tk_c[6] = kpe;
  } else {
    tk_c[0] = kms;
    for (int c = 1; c < 7; ++c) tk_c[c] = kms;  // degenerate
  }

  // Lambda to compute physical range for a compact index.
  // Odd compact indices -> single point.
  // Even compact indices -> range between adjacent pivots.
  auto PhysRange = [](int C, const int *t_c, int &lo, int &hi) {
    if (C & 1) {
      // Odd: single point.
      lo = t_c[C]; hi = t_c[C];
    } else if (C <= 3) {
      // Even, low half: [t_c[C], t_c[C+1] - 1].
      lo = t_c[C]; hi = t_c[C + 1] - 1;
    } else {
      // Even, high half: [t_c[C-1] + 1, t_c[C]].
      lo = t_c[C - 1] + 1; hi = t_c[C];
    }
  };

  // Iterate compact space and divide physical zones.
  for (int K = 0; K < cnk; ++K) {
    int kl, ku;
    PhysRange(K, tk_c, kl, ku);

    for (int J = 0; J < cnj; ++J) {
      int jl, ju;
      PhysRange(J, tj_c, jl, ju);

      for (int I = 0; I < cni; ++I) {
        unsigned short int m = mult_(0, K, J, I);
        if (m <= 1) continue;  // no division needed

        int il, iu;
        PhysRange(I, ti_c, il, iu);

        const Real inv = 1.0 / static_cast<Real>(m);
        for (int nn = 0; nn < nvar; ++nn)
          for (int k = kl; k <= ku; ++k)
            for (int j = jl; j <= ju; ++j)
              for (int i = il; i <= iu; ++i)
                var(nn, k, j, i) *= inv;
      }
    }
  }
}

} // namespace comm
