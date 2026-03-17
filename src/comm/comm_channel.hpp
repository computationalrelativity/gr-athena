#ifndef COMM_COMM_CHANNEL_HPP_
#define COMM_COMM_CHANNEL_HPP_
//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file comm_channel.hpp
//  \brief CommChannel: non-polymorphic per-variable communication object.
//
//  One CommChannel per registered variable per MeshBlock.  Owns its send/recv flat
//  buffers, status flags, and persistent MPI requests.  Does NOT own the state array
//  (that lives in the physics module).
//
//  Pack/unpack and index arithmetic are delegated to free functions in
//  index_utilities.hpp.  The channel itself only manages buffer lifecycle, MPI
//  setup, and same-rank copy.

#include <atomic>
#include <vector>

#include "../athena.hpp"        // Real
#include "../athena_arrays.hpp"
#include "../mesh/mesh_topology.hpp"  // BoundaryStatus
#include "comm_enums.hpp"
#include "comm_spec.hpp"
#include "node_multiplicity.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// forward declarations
class MeshBlock;

namespace comm {

class NeighborConnectivity;
class CommRegistry;

//----------------------------------------------------------------------------------------
// Maximum number of neighbors per MeshBlock (3D with refinement).
// Matches BoundaryData<56>::kMaxNeighbor in the old system.

static constexpr int kMaxNeighbor = 56;

//----------------------------------------------------------------------------------------
//! \class CommChannel
//  \brief Per-variable communication state for one MeshBlock.
//
//  Lifecycle:
//    1. Construct (from CommSpec + MeshBlock)
//    2. Finalize() - allocate buffers, create persistent MPI requests
//    3. Send / Receive / Set - called each communication cycle
//    4. Clear - wait on sends, reset flags
//    5. Reinitialize() - after regrid, rebuild MPI requests from new topology
//    6. Destruct - free buffers and MPI requests

class CommChannel {
 public:
  // --- construction / destruction ---
  CommChannel(const CommSpec &spec, MeshBlock *pmb, int channel_id);
  ~CommChannel();

  // non-copyable, movable
  CommChannel(const CommChannel&) = delete;
  CommChannel& operator=(const CommChannel&) = delete;
  CommChannel(CommChannel&&) noexcept;
  CommChannel& operator=(CommChannel&&) noexcept;

  // --- lifecycle ---

  // Allocate buffers and create persistent MPI requests.
  // Called once after construction (or after Reinitialize on regrid).
  // max_channel_id: largest channel_id across all channels on this block -
  //   needed so all channels agree on MPI tag bit layout.
  void Finalize(const NeighborConnectivity &nc, int max_channel_id);

  // After regrid: invalidate and rebuild MPI state from new topology.
  // Equivalent to teardown + Finalize with new neighbor info.
  void Reinitialize(const NeighborConnectivity &nc, int max_channel_id);

  // --- communication cycle ---

  // Pack data from the state array into send buffers for all active neighbors.
  // target_filter narrows the set of neighbors actually communicated (cannot widen).
  void PackAndSend(const NeighborConnectivity &nc, CommTarget target_filter);

  // Poll for received data.  Returns true if all expected receives have arrived.
  bool PollReceive(const NeighborConnectivity &nc, CommTarget target_filter);

  // Unpack received data into the state array.
  void Unpack(const NeighborConnectivity &nc, CommTarget target_filter);

  // Wait on all outstanding sends and reset status flags.
  void Clear(const NeighborConnectivity &nc, CommTarget target_filter);

  // Post persistent receive requests (called before the send/recv cycle).
  void StartReceiving(const NeighborConnectivity &nc, CommTarget target_filter);

  // --- flux correction communication cycle ---
  // These methods operate on flcor_send_buf_/flcor_recv_buf_ and use the flux arrays
  // (spec_.flx_cc or spec_.flx_fc) rather than the variable arrays.
  // Only active when spec_.flcor_mode != FluxCorrMode::None.

  // Pack area-weighted (CC) or edge-length-weighted (FC) restricted fluxes into
  // send buffers, then send to coarser face (CC) or face+edge (FC) neighbors.
  void PackAndSendFluxCorr(const NeighborConnectivity &nc);

  // Poll for received flux correction data.  Returns true when all expected
  // buffers have arrived.  For AccumulateAverage (FC EMF), this implements the
  // two-phase protocol: same-level first, then from-finer.
  bool PollReceiveFluxCorr(const NeighborConnectivity &nc);

  // Unpack received flux correction data into the flux arrays.
  // CC (OverwriteFromFiner): overwrites coarse flux at the shared face.
  // FC (AccumulateAverage): accumulates EMF values, then averages.
  void UnpackFluxCorr(const NeighborConnectivity &nc);

  // Wait on outstanding flux correction sends and reset flags.
  void ClearFluxCorr(const NeighborConnectivity &nc);

  // Post persistent receive requests for flux correction.
  void StartReceivingFluxCorr(const NeighborConnectivity &nc);

  // --- accessors ---
  int channel_id() const { return channel_id_; }
  CommGroup group() const { return spec_.group; }
  CommGroup flux_group() const { return spec_.flux_group; }
  FluxCorrMode flcor_mode() const { return spec_.flcor_mode; }
  const CommSpec& spec() const { return spec_; }

  // VC node multiplicity (precomputed at Finalize; valid only for Sampling::VC).
  NodeMultiplicity& node_multiplicity() { return node_mult_; }
  const NodeMultiplicity& node_multiplicity() const { return node_mult_; }

 private:
  // --- helpers ---
  void AllocateBuffers(const NeighborConnectivity &nc);
  void FreeBuffers();
  void SetupPersistentMPI(const NeighborConnectivity &nc, int max_channel_id);
  void FreeMPIRequests();

  // Compute the buffer size (in Reals) for one neighbor direction.
  int ComputeBufferSize(const NeighborConnectivity &nc, int nb_idx) const;

  // Same-rank direct copy: write directly into the target block's recv buffer.
  // NOTE: CopyBufferSameProcess is commented out - superseded by zero-copy
  // optimization that packs directly into the target's recv buffer.
  // void CopyBufferSameProcess(int nb_idx, int send_size);

  // Zero-copy helpers: resolve target block's recv buffer pointer for same-rank
  // neighbors so the sender can pack directly into it (eliminating the
  // intermediate send_buf copy + memcpy).
  Real* ResolveTargetRecvBuffer(int nb_idx);
  void  SetTargetRecvFlag(int nb_idx);

  Real* ResolveTargetFluxCorrRecvBuffer(int nb_idx);
  void  SetTargetFluxCorrRecvFlag(int nb_idx);

  // --- flux correction helpers ---

  // Allocate flcor_send_buf_ / flcor_recv_buf_ for all active flux-correction neighbors.
  void AllocateFluxCorrBuffers(const NeighborConnectivity &nc);

  // Free flux correction buffers (called from destructor / Reinitialize).
  void FreeFluxCorrBuffers();

  // Create persistent MPI send/recv for flux correction buffers.
  void SetupFluxCorrMPI(const NeighborConnectivity &nc, int max_channel_id);

  // Free flux correction MPI requests (called from destructor / Reinitialize).
  void FreeFluxCorrMPIRequests();

  // Compute flux correction buffer size (in Reals) for one neighbor.
  // Dispatches to ComputeFluxCorrBufferSizeCC or FC based on flcor_mode.
  int ComputeFluxCorrBufferSize(const NeighborConnectivity &nc, int nb_idx) const;

  // Populate edge_flag_[12] and nedge_fine_[12] from the nblevel table.
  // Only meaningful for AccumulateAverage (FC EMF); no-op otherwise.
  void CountFineEdges(const NeighborConnectivity &nc);

  // Same-rank direct copy for flux correction buffers.
  // NOTE: CopyFluxCorrBufferSameProcess is commented out - superseded by
  // zero-copy optimization that packs directly into the target's recv buffer.
  // void CopyFluxCorrBufferSameProcess(int nb_idx, int send_size);

  // CC flux correction pack+send (area-weighted restriction).
  void PackAndSendFluxCorrCC(const NeighborConnectivity &nc);

  // FC EMF flux correction pack+send (raw or edge-length-weighted restriction).
  void PackAndSendFluxCorrFC(const NeighborConnectivity &nc);

  // CC flux correction poll (face neighbors with level > mylevel only).
  bool PollReceiveFluxCorrCC(const NeighborConnectivity &nc);

  // FC EMF two-phase poll+unpack state machine (same-level -> clear -> from-finer -> avg).
  bool PollReceiveFluxCorrFC(const NeighborConnectivity &nc);

  // Pack raw EMF values for a same-level face/edge neighbor.
  int LoadFluxBoundaryBufferSameLevel(
      Real *buf, const NeighborBlock &nb,
      const AthenaArray<Real> &e1, const AthenaArray<Real> &e2,
      const AthenaArray<Real> &e3);

  // Pack edge-length-weighted restricted EMFs for a to-coarser face/edge neighbor.
  int LoadFluxBoundaryBufferToCoarser(
      Real *buf, const NeighborBlock &nb,
      const AthenaArray<Real> &e1, const AthenaArray<Real> &e2,
      const AthenaArray<Real> &e3,
      Coordinates *pco, AthenaArray<Real> &le1, AthenaArray<Real> &le2);

  // FC EMF unpack: accumulate same-level EMFs into the EMF arrays.
  void SetFluxBoundarySameLevel(Real *buf, const NeighborBlock &nb,
                                AthenaArray<Real> &e1, AthenaArray<Real> &e2,
                                AthenaArray<Real> &e3);

  // FC EMF unpack: accumulate from-finer restricted EMFs (quadrant narrowing via fi1/fi2).
  void SetFluxBoundaryFromFiner(Real *buf, const NeighborBlock &nb,
                                AthenaArray<Real> &e1, AthenaArray<Real> &e2,
                                AthenaArray<Real> &e3);

  // CC flux correction unpack: overwrite coarse flux at the shared face.
  void UnpackFluxCorrCC(Real *buf, const NeighborBlock &nb);

  // Zero EMFs at fine/coarse interfaces (called between same-level and from-finer
  // unpack phases in the two-phase FC EMF protocol).
  void ClearCoarseFluxBoundary(const NeighborConnectivity &nc);

  // Divide accumulated EMFs by contributor count after both unpack phases complete.
  void AverageFluxBoundary(const NeighborConnectivity &nc);

  // --- data ---
  CommSpec spec_;        // copy of registration spec (owns label string, etc.)
  MeshBlock *pmy_block_; // not owned
  int channel_id_;       // assigned by CommRegistry, used in MPI tag

  // Per-neighbor send/recv flat buffers (owned).
  Real *send_buf_[kMaxNeighbor];
  Real *recv_buf_[kMaxNeighbor];

  // Per-neighbor status flags.  Atomic because same-rank neighbors may write
  // to recv flags concurrently (via CopyBufferSameProcess).
  std::atomic<BoundaryStatus> recv_flag_[kMaxNeighbor];
  std::atomic<BoundaryStatus> send_flag_[kMaxNeighbor];

#ifdef MPI_PARALLEL
  MPI_Request req_send_[kMaxNeighbor];
  MPI_Request req_recv_[kMaxNeighbor];
#endif

  bool finalized_;  // true after Finalize() has been called

  // Cached polar sign array: computed once at Finalize() from component_groups.
  // Empty if component_groups is empty (no flips needed).
  std::vector<Real> polar_signs_;

  // VC node multiplicity: precomputed at Finalize() for vertex-centered channels.
  // Unused (empty) for CC/CX/FC.
  NodeMultiplicity node_mult_;

  // --- flux correction data ---
  // Separate buffer set from ghost exchange.  Only allocated when
  // spec_.flcor_mode != FluxCorrMode::None.

  // Per-neighbor flux correction flat buffers (owned).
  Real *flcor_send_buf_[kMaxNeighbor];
  Real *flcor_recv_buf_[kMaxNeighbor];

  // Per-neighbor flux correction status flags.  Atomic for same-rank concurrency.
  std::atomic<BoundaryStatus> flcor_recv_flag_[kMaxNeighbor];
  std::atomic<BoundaryStatus> flcor_send_flag_[kMaxNeighbor];

#ifdef MPI_PARALLEL
  MPI_Request req_flcor_send_[kMaxNeighbor];
  MPI_Request req_flcor_recv_[kMaxNeighbor];
#endif

  // Maximum buffer index used by flux correction (may differ from ghost exchange
  // because only face and edge neighbors participate).
  int nbmax_flcor_;

  // FC EMF two-phase state: true while receiving same-level contributions,
  // false during from-finer phase.  Only used when flcor_mode == AccumulateAverage.
  bool recv_flx_same_lvl_;

  // Per-edge flags for FC EMF averaging (matches old bvals edge_flag_[12]/nedge_fine_[12]).
  // edge_flag_[eid]: true if the finest neighbor touching this edge is at the same level
  //   (i.e. no finer neighbor contributes to this edge).
  // nedge_fine_[eid]: number of blocks contributing at the finest level to this edge
  //   (used as the divisor in AverageFluxBoundary).
  bool edge_flag_[12];
  int  nedge_fine_[12];

  // Scratch arrays for area weighting during CC flux restriction (x2/x3 faces).
  // sarea_[0] and sarea_[1] are 1D arrays of size (nx1 + 2*NGHOST), matching
  // BoundaryBase::sarea_[2] in the old system.
  AthenaArray<Real> sarea_[2];

  // CommRegistry needs access to buffers for fusion (future).
  friend class CommRegistry;
};

} // namespace comm

#endif // COMM_COMM_CHANNEL_HPP_
