#ifndef COMM_NODE_MULTIPLICITY_HPP_
#define COMM_NODE_MULTIPLICITY_HPP_
//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file node_multiplicity.hpp
//  \brief NodeMultiplicity: precomputed vertex sharing counts for VC consistency.
//
//  Vertex-centered variables use additive unpack - each shared vertex accumulates
//  contributions from all neighbors that overlap it.  The multiplicity array counts
//  how many times each region is written, so the caller can divide once to get the
//  correct average.
//
//  Uses a compact 7x7x7 representation (per axis: ghost-low, vertex-start,
//  interior-low, midpoint, interior-high, vertex-end, ghost-high).  Precomputed
//  at Finalize/Reinitialize time from topology alone (no data dependency).

#include <cstdint>  // std::int64_t

#include "../athena.hpp"        // Real
#include "../athena_arrays.hpp"

// forward declarations
class MeshBlock;

namespace comm {

class NeighborConnectivity;

//----------------------------------------------------------------------------------------
// Compact axis zone constants.
// Each axis is divided into 7 regions for the compact multiplicity representation.

static constexpr int c_ms = 0;   // ghost-low start
static constexpr int c_vs = 1;   // shared vertex start (interior boundary)
static constexpr int c_il = 2;   // interior low band
static constexpr int c_mp = 3;   // midpoint
static constexpr int c_ih = 4;   // interior high band
static constexpr int c_ve = 5;   // shared vertex end (interior boundary)
static constexpr int c_pe = 6;   // ghost-high end

//----------------------------------------------------------------------------------------
//! \class NodeMultiplicity
//  \brief Precomputed vertex sharing counts for one MeshBlock.
//
//  Lifecycle:
//    1. Precompute() - iterate all neighbors, accumulate compact multiplicity
//    2. ZeroGhosts() - zero VC ghost zones before additive unpack cycle
//    3. ApplyDivision() - divide VC array by multiplicity after all unpacks complete
//
//  The compact array has shape (1, 7, 7, 7) in 3D, (1, 1, 7, 7) in 2D,
//  (1, 1, 1, 7) in 1D.  Degenerate axes use constant index 0.

class NodeMultiplicity {
 public:
  NodeMultiplicity();

  // Precompute multiplicity from neighbor topology.  Called at Finalize/Reinitialize.
  void Precompute(MeshBlock *pmb, const NeighborConnectivity &nc, int nghost);

  // Zero all ghost zones of the fine VC array before additive unpack.
  void ZeroGhosts(AthenaArray<Real> &var, MeshBlock *pmb, int nvar) const;

  // Zero all ghost zones of the coarse VC array before additive unpack.
  void ZeroGhostsCoarse(AthenaArray<Real> &cvar, MeshBlock *pmb, int nvar) const;

  // Divide fine VC array by accumulated multiplicity.
  void ApplyDivision(AthenaArray<Real> &var, MeshBlock *pmb, int nvar) const;

  // Divide coarse VC array by accumulated multiplicity (uses coarse indices).
  void ApplyDivisionCoarse(AthenaArray<Real> &cvar, MeshBlock *pmb, int nvar) const;

  // True if Precompute() has been called.
  bool is_valid() const { return valid_; }

 private:
  AthenaArray<unsigned short int> mult_;
  bool valid_;

  // Per-axis compact index limits (degenerate axes collapse to 0..0).
  int c_ivs_, c_ive_, c_ims_, c_ipe_;
  int c_jvs_, c_jve_, c_jms_, c_jpe_;
  int c_kvs_, c_kve_, c_kms_, c_kpe_;

  // --- compact index helpers ---

  // Same-level compact range for one dimension.
  static void CompactRangeSameLevel(int ox, int &s, int &e,
                                    int cvs, int cve, int cms, int cpe);

  // From-coarser compact range for one dimension.
  static void CompactRangeFromCoarser(int ox, int &s, int &e,
                                      int cvs, int cve, int cms, int cpe,
                                      std::int64_t lx, bool is_nontrivial);

  // From-finer compact range for one dimension.
  static void CompactRangeFromFiner(int ox, int &s, int &e,
                                    int cvs, int cve, int cms, int cpe,
                                    int fi1, int fi2, int half_size,
                                    bool is_nontrivial, bool use_fi1);

  // Apply division in 3D using compact-to-physical mapping.
  void ApplyDivisionImpl(AthenaArray<Real> &var, int nvar,
                         int ims, int ivs, int ive, int ipe,
                         int jms, int jvs, int jve, int jpe,
                         int kms, int kvs, int kve, int kpe) const;
};

} // namespace comm

#endif // COMM_NODE_MULTIPLICITY_HPP_
