#ifndef COMM_AMR_REGISTRY_HPP_
#define COMM_AMR_REGISTRY_HPP_
//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file amr_registry.hpp
//  \brief AMRRegistry: per-MeshBlock owner of AMR-enrolled variables.
//
//  Lifecycle:
//    1. Physics modules call Register() to enroll variables for AMR
//    redistribution.
//    2. Finalize() sorts specs by group and caches buffer sizes.
//    3. During AMR redistribution, the orchestrator (amr_loadbalance.cpp)
//    calls
//       PackSameLevel / UnpackSameLevel / etc. to transfer block data.
//
//  One AMRRegistry per MeshBlock.  The registry owns no MPI state - that
//  belongs to AMRTransfer (Phase 2).  The registry handles packing, unpacking,
//  restriction, and prolongation for the enrolled variables.

#include <string>
#include <vector>

#include "../athena.hpp"  // Real, AthenaArray
#include "../athena_arrays.hpp"
#include "amr_spec.hpp"
#include "comm_enums.hpp"

// forward declarations
class MeshBlock;
class MeshRefinement;
struct LogicalLocation;

namespace comm
{

//----------------------------------------------------------------------------------------
//! \struct AMRBufferSizes
//  \brief Pre-computed buffer sizes (in Real elements) for a single AMRGroup.
//
//  These depend only on the block dimensions and the enrolled variables, so
//  they can be computed once in Finalize() and reused for every redistribution
//  step.

struct AMRBufferSizes
{
  int same_level;      // size for same-level block transfer (full interior)
  int fine_to_coarse;  // size for fine-to-coarse (restricted octant)
  int coarse_to_fine;  // size for coarse-to-fine (half-block + halo)

  AMRBufferSizes() : same_level(0), fine_to_coarse(0), coarse_to_fine(0)
  {
  }
};

//----------------------------------------------------------------------------------------
//! \class AMRRegistry
//  \brief Per-MeshBlock registry of AMR-enrolled variables.

class AMRRegistry
{
  public:
  explicit AMRRegistry(MeshBlock* pmb);
  ~AMRRegistry() = default;

  // non-copyable, non-movable (one per MeshBlock, stable address)
  AMRRegistry(const AMRRegistry&)            = delete;
  AMRRegistry& operator=(const AMRRegistry&) = delete;

  // --- registration ---

  // Register a variable for AMR redistribution.  Returns the spec_id (index
  // into the internal spec vector for this group).  Must be called in the same
  // order on every MeshBlock so that spec_ids and pack/unpack ordering are
  // consistent.
  int Register(const AMRSpec& spec);

  // --- lifecycle ---

  // Compute buffer sizes and freeze the registration.  Must be called after
  // all Register() calls and before any pack/unpack operations.
  void Finalize();

  // --- buffer sizes ---

  // Get pre-computed buffer sizes for a group (only valid after Finalize()).
  const AMRBufferSizes& GetBufferSizes(AMRGroup group) const
  {
    return buf_sizes_[static_cast<int>(group)];
  }

  // Total buffer size for same-level transfer (all groups combined).
  // Note: the orchestrator must add +1 for deref_count_ if adaptive,
  // since deref_count_ is block metadata not managed by the registry.
  int TotalSameLevelSize() const
  {
    return total_same_level_size_;
  }

  // --- pack / unpack operations ---

  // Same-level: pack all enrolled variables (all groups) into sendbuf.
  // Returns number of Reals written. The orchestrator appends deref_count_
  // separately after this call.
  int PackSameLevel(MeshBlock* pb, Real* sendbuf) const;

  // Same-level: unpack recvbuf into all enrolled variables (all groups).
  // Note: the orchestrator reads deref_count_ separately after this call.
  void UnpackSameLevel(MeshBlock* pb, Real* recvbuf) const;

  // Fine-to-coarse: restrict enrolled variables for a given group, then pack
  // the restricted (coarse) data into sendbuf.  Returns number of Reals
  // written.
  int PackFineToCoarse(MeshBlock* pb, Real* sendbuf, AMRGroup group) const;

  // Coarse-to-fine: pack the relevant half-block plus halo from the source
  // (coarse parent) into sendbuf.  lloc is the target child's LogicalLocation,
  // needed to determine which octant to extract.  Returns number of Reals
  // written.
  int PackCoarseToFine(MeshBlock* pb,
                       Real* sendbuf,
                       LogicalLocation& lloc,
                       AMRGroup group) const;

  // Fine-to-coarse unpack: unpack recvbuf into the correct octant of the
  // destination block.  lloc identifies which octant the data came from.
  void UnpackFineToCoarse(MeshBlock* pb,
                          Real* recvbuf,
                          LogicalLocation& lloc,
                          AMRGroup group) const;

  // Coarse-to-fine unpack: unpack recvbuf into the coarse buffer, then
  // prolongate to fill the fine-level destination block.
  void UnpackCoarseToFine(MeshBlock* pb, Real* recvbuf, AMRGroup group) const;

  // --- same-rank fill operations (no MPI, direct memory copy) ---

  // Fine-to-coarse, same rank: restrict source block's data and copy the
  // restricted data into the correct octant of the destination block.
  // dst_amr is the destination block's AMRRegistry (passed explicitly since
  // MeshBlock::pamr may not exist yet during the transition).
  void FillSameRankFineToCoarse(MeshBlock* src,
                                MeshBlock* dst,
                                LogicalLocation& loc,
                                AMRGroup group,
                                AMRRegistry* dst_amr) const;

  // Coarse-to-fine, same rank: copy relevant portion from source block into
  // destination's coarse buffer, then prolongate.
  // src_amr/dst_amr are the source/destination blocks' AMRRegistries.
  void FillSameRankCoarseToFine(MeshBlock* src,
                                MeshBlock* dst,
                                LogicalLocation& newloc,
                                AMRGroup group,
                                AMRRegistry* src_amr,
                                AMRRegistry* dst_amr) const;

  // --- accessors ---

  int num_specs() const
  {
    return static_cast<int>(all_specs_.size());
  }
  int num_specs_in_group(AMRGroup group) const
  {
    return static_cast<int>(group_specs_[static_cast<int>(group)].size());
  }
  bool is_finalized() const
  {
    return finalized_;
  }

  private:
  MeshBlock* pmy_block_;  // not owned

  // All registered specs, in registration order (across all groups).
  std::vector<AMRSpec> all_specs_;

  // Per-group spec indices: group_specs_[g] holds indices into all_specs_ for
  // group g.
  std::vector<int> group_specs_[static_cast<int>(AMRGroup::NumGroups)];

  // Pre-computed buffer sizes per group.
  AMRBufferSizes buf_sizes_[static_cast<int>(AMRGroup::NumGroups)];

  // Total same-level size across all groups (does NOT include +1 for
  // deref_count_ - the orchestrator adds that separately).
  int total_same_level_size_;

  bool finalized_;

  // --- internal helpers ---

  // Compute buffer sizes for a single group based on enrolled variables and
  // block dims.
  void ComputeBufferSizes(AMRGroup group);

  // Pack/unpack helpers per sampling type (CC, CX, VC, FC).
  // These iterate over specs in a group matching a particular sampling type.

  // Pack CC variables from fine-level interior into buffer.
  int PackCCSameLevel(MeshBlock* pb,
                      Real* buf,
                      int p,
                      const std::vector<int>& spec_ids) const;
  int PackCXSameLevel(MeshBlock* pb,
                      Real* buf,
                      int p,
                      const std::vector<int>& spec_ids) const;
  int PackVCSameLevel(MeshBlock* pb,
                      Real* buf,
                      int p,
                      const std::vector<int>& spec_ids) const;
  int PackFCSameLevel(MeshBlock* pb,
                      Real* buf,
                      int p,
                      const std::vector<int>& spec_ids) const;

  // Unpack counterparts.
  int UnpackCCSameLevel(MeshBlock* pb,
                        Real* buf,
                        int p,
                        const std::vector<int>& spec_ids) const;
  int UnpackCXSameLevel(MeshBlock* pb,
                        Real* buf,
                        int p,
                        const std::vector<int>& spec_ids) const;
  int UnpackVCSameLevel(MeshBlock* pb,
                        Real* buf,
                        int p,
                        const std::vector<int>& spec_ids) const;
  int UnpackFCSameLevel(MeshBlock* pb,
                        Real* buf,
                        int p,
                        const std::vector<int>& spec_ids) const;
};

}  // namespace comm

#endif  // COMM_AMR_REGISTRY_HPP_
