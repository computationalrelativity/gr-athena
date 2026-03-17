#ifndef COMM_NEIGHBOR_CONNECTIVITY_HPP_
#define COMM_NEIGHBOR_CONNECTIVITY_HPP_
//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file neighbor_connectivity.hpp
//  \brief NeighborConnectivity: owns a MeshBlock's neighbor topology data.
//
//  Stores neighbor[], nneighbor, nblevel[3][3][3], and block_bcs[6].
//  SearchAndSetNeighbors (friend) populates neighbor/nneighbor/nblevel.
//  MeshBlock constructors call InitBoundaryFlags() to set block_bcs.

#include "../mesh/mesh_topology.hpp"  // NeighborBlock, BoundaryFace, BoundaryFlag,
                                      // kMaxNeighbor

// forward declarations
class MeshBlock;
class MeshBlockTree;

namespace mesh_topology {
void SearchAndSetNeighbors(MeshBlock *pmb, MeshBlockTree &tree,
                           int *ranklist, int *nslist);
} // namespace mesh_topology

namespace comm {

//----------------------------------------------------------------------------------------
//! \class NeighborConnectivity
//  \brief Owns and provides access to a MeshBlock's neighbor topology.
//
//  Lifetime matches the MeshBlock.  SearchAndSetNeighbors() populates the
//  neighbor data after construction; Reinitialize() via the same path after
//  regrid.

class NeighborConnectivity {
  // SearchAndSetNeighbors writes neighbor_, nneighbor_, nblevel_ directly.
  friend void mesh_topology::SearchAndSetNeighbors(MeshBlock*, MeshBlockTree&,
                                                   int*, int*);
 public:
  // Default state: no neighbors, all nblevel = -1, all block_bcs = undef.
  NeighborConnectivity();

  // --- initialization (called by MeshBlock constructors) ---

  // Set block_bcs_[6] from an input array (copied from ParameterInput).
  void InitBoundaryFlags(const BoundaryFlag input_bcs[6]);

  // Reset nneighbor to 0 and nblevel to -1 before SearchAndSetNeighbors.
  void ResetForSearch(int my_level);

  // --- read-only accessors ---

  int num_neighbors() const { return nneighbor_; }

  // Access the i-th neighbor (0 <= i < num_neighbors()).
  const NeighborBlock& neighbor(int i) const { return neighbor_[i]; }

  // Neighbor level relative to this block at offset (ox1,ox2,ox3) in [-1,0,+1].
  // Returns -1 if no neighbor at that offset (physical boundary or absent).
  int neighbor_level(int ox1, int ox2, int ox3) const {
    return nblevel_[ox3 + 1][ox2 + 1][ox1 + 1];
  }

  // Physical boundary condition flag on the given face.
  BoundaryFlag boundary_flag(BoundaryFace face) const {
    return block_bcs_[static_cast<int>(face)];
  }
  // Overload accepting raw int index (avoids casts at call sites).
  BoundaryFlag boundary_flag(int face) const {
    return block_bcs_[face];
  }

  // True if the given face is a physical (non-block, non-periodic) boundary
  // that requires ghost-zone fill by function evaluation.
  bool is_physical_boundary(BoundaryFace face) const {
    BoundaryFlag f = boundary_flag(face);
    return f != BoundaryFlag::block && f != BoundaryFlag::periodic
        && f != BoundaryFlag::polar && f != BoundaryFlag::undef;
  }

 private:
  NeighborBlock neighbor_[kMaxNeighbor];  // sparse, indexed by bufid slot
  int nneighbor_;
  int nblevel_[3][3][3];                  // indexed [k+1][j+1][i+1]
  BoundaryFlag block_bcs_[6];            // one per BoundaryFace
};

} // namespace comm

#endif // COMM_NEIGHBOR_CONNECTIVITY_HPP_
