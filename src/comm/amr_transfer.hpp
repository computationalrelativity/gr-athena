#ifndef COMM_AMR_TRANSFER_HPP_
#define COMM_AMR_TRANSFER_HPP_
//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file amr_transfer.hpp
//  \brief AMRTransfer: transient MPI lifecycle manager for AMR block
//  redistribution.
//
//  AMRTransfer is created once per redistribution cycle by the orchestrator
//  (RedistributeAndRefineMeshBlocks).  It encapsulates:
//    - MPI tag creation (CreateAMRMPITag)
//    - Buffer sizing (from AMRRegistry)
//    - Irecv/Isend posting for cross-rank transfers
//    - Same-rank block data movement (FillSameRank* delegation to AMRRegistry)
//    - Waitall + unpack
//    - Buffer cleanup
//
//  The orchestrator retains responsibility for:
//    - Computing the redistribution mapping (newloc, newtoold, oldtonew, etc.)
//    - Building the new MeshBlock linked list (Step 7)
//    - deref_count_ handling (appended/read outside AMRTransfer)
//    - Updating Mesh-level metadata (loclist, ranklist, costlist)
//
//  Lifecycle:
//    1. Orchestrator creates AMRTransfer with mapping arrays.
//    2. Calls PostReceives() to post all MPI_Irecv.
//    3. Calls PackAndSend() to pack + MPI_Isend all outgoing blocks.
//    4. (Orchestrator builds new MeshBlock linked list - Step 7,
//        calling FillFineToCoarseSameRank / FillCoarseToFineSameRank
//        inline for same-rank cross-level blocks.)
//    5. Calls WaitAndUnpack() to complete MPI + unpack received data.
//    6. Calls WaitSendsAndCleanup() to wait for sends + free buffers.
//    7. Destructor frees any remaining buffers.
//
//  Note: same-rank same-level blocks are moved by pointer (no data copy),
//  which the orchestrator handles directly.  AMRTransfer only handles
//  cross-rank transfers and same-rank cross-level transfers.

#include <vector>

#include "../athena.hpp"  // Real

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// forward declarations
class Mesh;
class MeshBlock;
struct LogicalLocation;

namespace comm
{

class AMRRegistry;

//----------------------------------------------------------------------------------------
//! \struct AMRTransferMap
//  \brief Redistribution mapping arrays computed by the orchestrator.
//
//  These are borrowed pointers - AMRTransfer does not own or free them.
//  The orchestrator must keep them alive for the duration of the transfer.

struct AMRTransferMap
{
  // Global mapping arrays (size = nbtotal).
  LogicalLocation* newloc;  // new LogicalLocation for each new GID
  LogicalLocation*
    loclist;      // old LogicalLocation for each old GID (= Mesh::loclist)
  int* newtoold;  // new GID -> old GID
  int* oldtonew;  // old GID -> new GID
  int* ranklist;  // old GID -> old rank (= Mesh::ranklist)
  int* newrank;   // new GID -> new rank
  int* nslist;    // rank -> starting new GID for that rank (= Mesh::nslist)

  // This rank's block ranges.
  int nbs;   // first new GID on this rank
  int nbe;   // last new GID on this rank
  int onbs;  // first old GID on this rank
  int onbe;  // last old GID on this rank

  // Dimensionality.
  int nleaf;      // 2, 4, or 8 (children per parent in 1D/2D/3D)
  bool adaptive;  // whether derefinement counters are transferred

  // Buffer sizes (computed by the orchestrator from AMRRegistry, plus
  // the +1 for deref_count_ on bssame when adaptive).
  int bssame;  // same-level buffer size per block (in Real elements)
  int bsf2c;   // fine-to-coarse buffer size
  int bsc2f;   // coarse-to-fine buffer size
};

//----------------------------------------------------------------------------------------
//! \class AMRTransfer
//  \brief Transient MPI lifecycle for one AMR redistribution cycle.

class AMRTransfer
{
  public:
  // Construct with a pointer to the parent Mesh and the mapping arrays.
  // Counts nsend/nrecv and allocates request/buffer pointer arrays.
  AMRTransfer(Mesh* pm, const AMRTransferMap& map);

  // Destructor: frees all allocated buffers and MPI request arrays.
  ~AMRTransfer();

  // non-copyable, non-movable
  AMRTransfer(const AMRTransfer&)            = delete;
  AMRTransfer& operator=(const AMRTransfer&) = delete;

  // --- MPI lifecycle steps ---

  // Post MPI_Irecv for every cross-rank incoming block.
  // Must be called before PackAndSend (standard MPI overlap pattern).
  void PostReceives();

  // Pack outgoing block data and post MPI_Isend for every cross-rank
  // outgoing block.  For each block, delegates to AMRRegistry::Pack*.
  // The orchestrator must call this BEFORE building the new block list,
  // because it reads from the OLD blocks which are about to be deleted.
  //
  // deref_count_fn: optional callback to append deref_count_ after
  // AMRRegistry packs same-level blocks.  Signature:
  //   void(MeshBlock* old_block, Real* sendbuf, int offset)
  // where offset is the byte position after AMRRegistry's pack.
  // Pass nullptr to skip (non-adaptive runs).
  using DerefPackFn = void (*)(MeshBlock* pb, Real* buf, int offset);
  void PackAndSend(DerefPackFn deref_pack_fn = nullptr);

  // Same-rank fine-to-coarse: restrict each child's data into the correct
  // octant of the new parent block.  The orchestrator calls this inline
  // during Step 7 for each same-rank child, passing explicit block pointers
  // (since FindMeshBlock cannot locate both old and new blocks
  // simultaneously).
  //
  // old_fine:  the old fine-level child block (still alive during Step 7)
  // new_coarse: the newly constructed parent block
  // old_loc:   LogicalLocation of old_fine (for octant determination)
  void FillFineToCoarseSameRank(MeshBlock* old_fine,
                                MeshBlock* new_coarse,
                                LogicalLocation& old_loc);

  // Same-rank coarse-to-fine: copy relevant portion from old parent into
  // the new child's coarse buffer, then prolongate.  The orchestrator calls
  // this inline during Step 7, passing explicit block pointers.
  //
  // old_coarse: the old coarse-level parent block (still alive during Step
  // 7) new_fine:   the newly constructed child block new_loc:
  // LogicalLocation of new_fine (for sub-block extraction)
  void FillCoarseToFineSameRank(MeshBlock* old_coarse,
                                MeshBlock* new_fine,
                                LogicalLocation& new_loc);

  // Wait for all MPI_Irecv to complete, then unpack received data into
  // the new (already constructed) MeshBlocks.
  //
  // deref_unpack_fn: optional callback to read deref_count_ after
  // AMRRegistry unpacks same-level blocks.  Signature:
  //   void(MeshBlock* new_block, Real* recvbuf, int offset)
  // Pass nullptr to skip.
  using DerefUnpackFn = void (*)(MeshBlock* pb, Real* buf, int offset);
  void WaitAndUnpack(DerefUnpackFn deref_unpack_fn = nullptr);

  // Wait for all sends to complete and free send buffers.
  // Called after WaitAndUnpack to ensure send buffers are not freed before
  // the MPI implementation is done with them.
  void WaitSendsAndCleanup();

  // --- accessors ---
  int nsend() const
  {
    return nsend_;
  }
  int nrecv() const
  {
    return nrecv_;
  }

  // --- static utility ---

  // Create an MPI tag for AMR block transfer.
  // lid: local destination block ID (relative to receiver's rank start)
  // ox1, ox2, ox3: octant position (0 or 1), used for f2c disambiguation.
  // Returns (lid << 3) | (ox1 << 2) | (ox2 << 1) | ox3.
  static int CreateAMRMPITag(int lid, int ox1, int ox2, int ox3);

  private:
  Mesh* pmy_mesh_;      // not owned
  AMRTransferMap map_;  // borrowed pointers - not owned

  int nsend_;
  int nrecv_;

#ifdef MPI_PARALLEL
  MPI_Request* req_send_;
  MPI_Request* req_recv_;
#endif

  // Per-message buffers (individually allocated, matching legacy pattern).
  Real** sendbuf_;
  Real** recvbuf_;

  // --- internal helpers ---

  // Count nsend/nrecv by iterating over the mapping arrays.
  void CountSendRecv();
};

}  // namespace comm

#endif  // COMM_AMR_TRANSFER_HPP_
