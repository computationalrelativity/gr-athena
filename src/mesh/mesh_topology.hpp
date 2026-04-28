#ifndef MESH_MESH_TOPOLOGY_HPP_
#define MESH_MESH_TOPOLOGY_HPP_
//========================================================================================
// GR-Athena++
//========================================================================================
//! \file mesh_topology.hpp
//  \brief Enums, structs, and free functions that describe mesh-block topology:
//         boundary flags, neighbor connectivity, and neighbor indexing.
//
//  Extracted from the former src/bvals/bvals_interfaces.hpp and src/bvals/bvals.hpp
//  so that the rest of the code can use these types without depending on the old
//  boundary-variable class hierarchy.

// C++ headers
#include <string>

// Athena++ headers
#include "../athena.hpp"

// forward declarations
class Mesh;
class MeshBlock;
class MeshBlockTree;
struct RegionSize;

//----------------------------------------------------------------------------------------
// Identifiers for all 6 faces of a MeshBlock.
// Unscoped enum so enumerators can be used as raw-array indices.
enum BoundaryFace {undef=-1, inner_x1=0, outer_x1=1, inner_x2=2, outer_x2=3,
                   inner_x3=4, outer_x3=5};

// Identifiers for boundary conditions
enum class BoundaryFlag {block=-1, undef, reflect, outflow, extrapolate_outflow,
                         user, periodic, polar, polar_wedge,
                         gr_sommerfeld};

// Identifiers for types of neighbor connectivity
enum class NeighborConnect {none, face, edge, corner};

// Identifiers for status of MPI boundary communications
enum class BoundaryStatus {waiting, arrived, completed};

//----------------------------------------------------------------------------------------
// Parity flags for polar boundaries (which hydro/field components flip sign)
constexpr const bool flip_across_pole_hydro[] = {false, false, true, true, false};
constexpr const bool flip_across_pole_field[] = {false, true, true};

//----------------------------------------------------------------------------------------
//! \struct SimpleNeighborBlock
//  \brief Minimal info about a neighbor (rank, level, lid, gid).  Used e.g. to describe
//         the set of blocks around a pole at the same radius and polar angle.

struct SimpleNeighborBlock {
  int rank;
  int level;
  int lid;
  int gid;
};

//----------------------------------------------------------------------------------------
//! \struct NeighborIndexes
//  \brief Directional offsets (ox1,ox2,ox3 in {-1,0,+1}) and refinement sub-indices
//         (fi1,fi2 in {0,1}) that identify a specific neighbor slot.

struct NeighborIndexes {
  int ox1, ox2, ox3;
  int fi1, fi2;
  NeighborConnect type;
};

//----------------------------------------------------------------------------------------
//! \struct NeighborBlock
//  \brief Full description of one neighbor block, combining identity, direction, and
//         buffer/target IDs used by the communication layer.

struct NeighborBlock {
  SimpleNeighborBlock snb;
  NeighborIndexes ni;

  int bufid, eid, targetid;
  BoundaryFace fid;
  bool polar;
  // True when the neighbor block itself has all same-level neighbors.
  // Set during SearchAndSetNeighbors from the MeshBlockTree flag.
  bool neighbor_all_same_level;

  void SetNeighbor(int irank, int ilevel, int igid, int ilid, int iox1, int iox2,
                   int iox3, NeighborConnect itype, int ibid, int itargetid,
                   bool ipolar, int ifi1=0, int ifi2=0);
};

//----------------------------------------------------------------------------------------
// Maximum number of neighbor buffer slots (3D refined case).
// 56 = 6 faces * 4 + 12 edges * 2 + 8 corners
constexpr int kMaxNeighbor = 56;

//----------------------------------------------------------------------------------------
// Free functions

// Parse input string to BoundaryFlag enum
BoundaryFlag GetBoundaryFlag(const std::string& input_string);
// Inverse: BoundaryFlag enum to descriptive string
std::string GetBoundaryString(BoundaryFlag input_flag);
// Validate that a BoundaryFlag is legal for the given coordinate direction
void CheckBoundaryFlag(BoundaryFlag block_flag, CoordinateDirection dir);

//----------------------------------------------------------------------------------------
// Static topology tables: neighbor index patterns and buffer IDs.
// Populated once during the first MeshBlock construction, then immutable.
// Declared in mesh_topology.cpp.

namespace mesh_topology {

// 1 pair (neighbor index, buffer ID) per neighbor slot.
// Greedy allocation for worst-case refined 3D (56 entries).
extern NeighborIndexes ni[kMaxNeighbor];
extern int bufid[kMaxNeighbor];
extern int maxneighbor;

// Populate ni[] and bufid[] for given dimensionality and multilevel flag.
// Returns the actual number of neighbor slots used.
int InitBufferID(int dim, bool multilevel);

// Create a buffer ID from directional offsets and refinement sub-indices.
int CreateBufferID(int ox1, int ox2, int ox3, int fi1, int fi2);

// Look up the slot index for a given set of directional offsets.
// Returns -1 if not found.
int FindBufferID(int ox1, int ox2, int ox3, int fi1, int fi2);

// Search the MeshBlockTree and populate neighbor[], nneighbor, nblevel for a block.
void SearchAndSetNeighbors(MeshBlock *pmb, MeshBlockTree &tree,
                           int *ranklist, int *nslist);

} // namespace mesh_topology

#endif // MESH_MESH_TOPOLOGY_HPP_
