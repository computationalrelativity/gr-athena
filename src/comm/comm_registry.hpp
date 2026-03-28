#ifndef COMM_COMM_REGISTRY_HPP_
#define COMM_COMM_REGISTRY_HPP_
//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file comm_registry.hpp
//  \brief CommRegistry: per-MeshBlock owner of all CommChannels.
//
//  Lifecycle:
//    1. Physics modules call Register() to declare variables.
//    2. Finalize() allocates buffers, creates MPI requests, validates tags.
//    3. Communication cycle: Send / Receive / Set / Clear per group.
//    4. Reinitialize() after regrid rebuilds everything from new topology.
//
//  One CommRegistry per MeshBlock.  Channels in the same CommGroup share a
//  fused MPI message per neighbor (future - currently each channel sends
//  independently).

#include <string>
#include <unordered_map>
#include <vector>

#include "comm_channel.hpp"
#include "comm_enums.hpp"
#include "comm_spec.hpp"
#include "neighbor_connectivity.hpp"

#ifdef DBG_FUSED_COMM
#include <atomic>
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif
#endif

// forward declarations
class MeshBlock;

namespace comm
{

#ifdef DBG_FUSED_COMM
//----------------------------------------------------------------------------------------
//! \struct FusedGroupState
//  \brief Per-group fused communication buffers and MPI state.
//
//  When active, all channels in the group pack sequentially into a single
//  contiguous buffer per neighbor, and a single MPI message is sent/received
//  instead of one per channel.  Same-rank neighbors use zero-copy: the sender
//  packs directly into the target's recv_buf, then sets recv_flag atomically.

struct FusedGroupState
{
  // Per-neighbor fused send/recv buffers.  nullptr for same-rank (zero-copy).
  Real* send_buf[kMaxNeighbor]{};
  Real* recv_buf[kMaxNeighbor]{};

  // Per-neighbor allocation size (max over all possible send sizes).
  int buf_alloc_size[kMaxNeighbor]{};

  // Per-neighbor status flags.  Atomic for same-rank zero-copy concurrency.
  std::atomic<BoundaryStatus> recv_flag[kMaxNeighbor];
  std::atomic<BoundaryStatus> send_flag[kMaxNeighbor];

#ifdef MPI_PARALLEL
  MPI_Request req_send[kMaxNeighbor];
  MPI_Request req_recv[kMaxNeighbor];
#endif

  // Zero-copy target: for same-rank neighbors, points to the target block's
  // FusedGroupState so the sender can pack directly into
  // target->recv_buf[targetid].
  FusedGroupState* target_fused[kMaxNeighbor]{};

  // True if this group has >= 2 channels and fused comm is worthwhile.
  // Groups with < 2 channels fall through to the per-channel path.
  bool active = false;

  FusedGroupState()
  {
    for (int n = 0; n < kMaxNeighbor; n++)
    {
      recv_flag[n].store(BoundaryStatus::waiting, std::memory_order_relaxed);
      send_flag[n].store(BoundaryStatus::waiting, std::memory_order_relaxed);
#ifdef MPI_PARALLEL
      req_send[n] = MPI_REQUEST_NULL;
      req_recv[n] = MPI_REQUEST_NULL;
#endif
    }
  }
};
#endif  // DBG_FUSED_COMM

//----------------------------------------------------------------------------------------
//! \class CommRegistry
//  \brief Per-MeshBlock registry of communication channels.

class CommRegistry
{
  public:
  explicit CommRegistry(MeshBlock* pmb);
  ~CommRegistry();

  // non-copyable, non-movable (one per MeshBlock, stable address)
  CommRegistry(const CommRegistry&)            = delete;
  CommRegistry& operator=(const CommRegistry&) = delete;

  // --- registration ---

  // Register a variable for communication.  Returns the channel_id (index into
  // the internal channel vector).  Must be called in the same order on every
  // MeshBlock so that channel_ids are consistent across ranks.
  int Register(const CommSpec& spec);

  // Add an already-registered channel to an additional communication group.
  // Must be called before Finalize().  The channel retains its primary group;
  // this simply makes it also addressable via the secondary group.
  void AddToGroup(int channel_id, CommGroup group);

  // --- lifecycle ---

  // Allocate buffers and create MPI requests for all channels.
  // Must be called after all Register() calls and after
  // SearchAndSetNeighbors().
  void Finalize();

  // After regrid: rebuild NeighborConnectivity and all channel MPI state.
  void Reinitialize();

  // --- group-level communication ---

  // Post persistent receive requests for all channels in the group.
  void StartReceiving(CommGroup group,
                      CommTarget target_filter = CommTarget::All);

  // Pack and send boundary data for all channels in the group.
  void SendBoundaryBuffers(CommGroup group,
                           CommTarget target_filter = CommTarget::All);

  // Poll for received data.  Returns true when all channels in the group have
  // received all expected messages.
  bool ReceiveBoundaryBuffers(CommGroup group,
                              CommTarget target_filter = CommTarget::All);

  // Unpack received data into state arrays for all channels in the group.
  void SetBoundaries(CommGroup group,
                     CommTarget target_filter = CommTarget::All);

  // Test (or wait on) outstanding sends and reset flags for all channels in
  // the group.  When wait=false, returns false without resetting flags if
  // any off-rank send is still pending.
  bool ClearBoundary(CommGroup group,
                     CommTarget target_filter = CommTarget::All,
                     bool wait                = true);

  // --- flux correction group-level communication ---
  // These methods operate on channels whose flux_group matches the given
  // group. Typically called with CommGroup::FluxCorr.

  // Post persistent receive requests for flux correction.
  void StartReceivingFluxCorr(CommGroup group);

  // Pack and send flux correction data for all channels in the group.
  void SendFluxCorrBuffers(CommGroup group);

  // Pack and send flux correction data for a single channel (by channel_id).
  // Used when different physics variables become ready at different task-DAG
  // stages.
  void SendFluxCorrSingleChannel(int channel_id);

  // Post persistent receive for flux correction on a single channel (by
  // channel_id). Used for per-phase startup so that each phase only activates
  // its own channels, avoiding double MPI_Start on channels belonging to a
  // different phase.
  void StartReceivingFluxCorrSingleChannel(int channel_id);

  // Test (or wait on) flux correction sends and reset flags for a single
  // channel (by channel_id).  When wait=false, returns false without
  // resetting flags if any off-rank send is still pending.
  bool ClearFluxCorrSingleChannel(int channel_id, bool wait = true);

  // Poll for received flux correction data.  Returns true when all channels
  // done. For FC (AccumulateAverage), the two-phase state machine runs inside
  // PollReceive.
  bool ReceiveFluxCorrBuffers(CommGroup group);

  // Unpack flux correction data.  CC channels overwrite coarse fluxes; FC
  // channels are no-ops (unpack happens inside PollReceiveFluxCorrFC).
  void SetFluxCorrBoundaries(CommGroup group);

  // Wait on flux correction sends and reset flags.
  void ClearFluxCorrBoundary(CommGroup group);

  // --- physical boundary conditions (group-level helpers) ---

  // Apply physical BCs for all channels in a group (fine-level).
  // Dispatches CC vs FC per channel automatically.
  void ApplyPhysicalBCs(CommGroup group, Real time, Real dt);

  // Apply coarse-level BCs + prolongation for all channels in a group.
  // Caller passes coarse index bounds and ghost width from the physics module.
  void ProlongateAndApplyPhysicalBCs(CommGroup group,
                                     Real time,
                                     Real dt,
                                     int cis,
                                     int cie,
                                     int cjs,
                                     int cje,
                                     int cks,
                                     int cke,
                                     int cng);

  // --- accessors ---

  int num_channels() const
  {
    return static_cast<int>(channels_.size());
  }

  CommChannel& channel(int channel_id)
  {
    return channels_[channel_id];
  }
  const CommChannel& channel(int channel_id) const
  {
    return channels_[channel_id];
  }

  // Look up a channel by label.  Returns -1 if not found.
  int FindChannelByLabel(const std::string& label) const;

  // Access the current NeighborConnectivity snapshot.
  const NeighborConnectivity& connectivity() const
  {
    return nc_;
  }

  // True after Finalize() has been called (buffers and MPI requests
  // allocated). Used by Mesh::Initialize to distinguish surviving blocks
  // (needs Reinitialize) from newly created blocks (needs first Finalize)
  // during AMR regrid.
  bool is_finalized() const
  {
    return finalized_;
  }

  // Static lookup: find the CommRegistry for a given MeshBlock.
  // Used by same-process buffer copy to reach the target block's channels.
  // Returns nullptr if no registry has been constructed for the block.
  static CommRegistry* FindForBlock(MeshBlock* pmb);

  private:
  MeshBlock* pmy_block_;  // not owned
  NeighborConnectivity nc_;

  // All channels, indexed by channel_id (registration order).
  std::vector<CommChannel> channels_;

  // Group membership: group_channels_[g] holds channel_ids belonging to group
  // g.
  std::vector<int> group_channels_[static_cast<int>(CommGroup::NumGroups)];

  // Label -> channel_id lookup.
  std::unordered_map<std::string, int> label_to_id_;

  bool finalized_;

  // Rebuild NeighborConnectivity from the MeshBlock's topology data.
  void RebuildConnectivity();

#ifdef DBG_FUSED_COMM
  // --- Fused communication state (one per group) ---
  FusedGroupState fused_[static_cast<int>(CommGroup::NumGroups)];

  // Allocate fused buffers and set up persistent MPI for all active groups.
  // Called at the end of Finalize() after per-channel setup is complete.
  void FinalizeFused();

  // Teardown + rebuild fused state after regrid.
  void ReinitializeFused();

  // Cleanup helpers.
  void FreeFusedBuffers(int g);
  void FreeFusedMPIRequests(int g);

  // --- Fused implementations of the 5 group-level communication methods ---

  // Pack all channels in the group sequentially into fused send_buf, then
  // send.
  void SendBoundaryBuffersFused(CommGroup group, CommTarget target_filter);

  // Poll fused recv_flag / MPI_Test.  Returns true when all neighbors done.
  bool ReceiveBoundaryBuffersFused(CommGroup group, CommTarget target_filter);

  // VC zero-ghost, then unpack per-channel from fused recv_buf, then VC
  // divide.
  void SetBoundariesFused(CommGroup group, CommTarget target_filter);

  // MPI_Test (or MPI_Wait) on sends, reset flags.  When wait=false, polls
  // sends first; flags are only reset after all sends are confirmed complete.
  bool ClearBoundaryFused(CommGroup group,
                          CommTarget target_filter,
                          bool wait = true);

  // MPI_Start on persistent recv requests.
  void StartReceivingFused(CommGroup group, CommTarget target_filter);
#endif  // DBG_FUSED_COMM

  // Global registry of all CommRegistry instances, keyed by MeshBlock pointer.
  // Enables same-process buffer copy to find the target block's channels.
  static std::unordered_map<MeshBlock*, CommRegistry*> registry_map_;
};

}  // namespace comm

#endif  // COMM_COMM_REGISTRY_HPP_
