#ifndef MESH_MESHBLOCK_TREE_HPP_
#define MESH_MESHBLOCK_TREE_HPP_
//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
// See LICENSE file for full public license information.
//======================================================================================
//! \file meshblock_tree.hpp
//  \brief defines the LogicalLocation structure and MeshBlockTree class
//======================================================================================

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../defs.hpp"

class Mesh;

// Forward declarations for the free function so MeshBlockTree can befriend it
class MeshBlock;
class MeshBlockTree;
namespace mesh_topology {
void SearchAndSetNeighbors(MeshBlock *pmb, MeshBlockTree &tree,
                           int *ranklist, int *nslist);
} // namespace mesh_topology

//--------------------------------------------------------------------------------------
//! \class MeshBlockTree
//  \brief Objects are nodes in an AMR MeshBlock tree structure

class MeshBlockTree {
  friend class Mesh;
  friend class MeshBlock;
  // Allow the new mesh_topology free function to access private tree data
  friend void mesh_topology::SearchAndSetNeighbors(MeshBlock*, MeshBlockTree&,
                                                   int*, int*);
 public:
  explicit MeshBlockTree(Mesh *pmesh);
  MeshBlockTree(MeshBlockTree *parent, int ox1, int ox2, int ox3);
  ~MeshBlockTree();

  // accessor
  MeshBlockTree* GetLeaf(int ox1, int ox2, int ox3)
  { return pleaf_[(ox1 + (ox2<<1) + (ox3<<2))]; }

  // functions
  void CreateRootGrid();
  void AddMeshBlock(LogicalLocation rloc, int &nnew);
  void AddMeshBlockWithoutRefine(LogicalLocation rloc);
  void Refine(int &nnew);
  void Derefine(int &ndel);
  MeshBlockTree* FindMeshBlock(LogicalLocation tloc);
  void CountMeshBlock(int& count);
  void GetMeshBlockList(LogicalLocation *list, int *pglist, int& count);
  MeshBlockTree* FindNeighbor(LogicalLocation myloc, int ox1, int ox2, int ox3,
                              bool amrflag=false);
  // Pre-compute, for each leaf node, whether all of its neighbors are at the
  // same refinement level.  Called once after the tree is finalized (before
  // SearchAndSetNeighbors) so the flag can be propagated to NeighborBlock.
  void ComputeNeighborLevelFlags();

 private:
  // data
  MeshBlockTree** pleaf_;
  int gid_;
  LogicalLocation loc_;
  // True when every neighbor of this leaf is at the same refinement level.
  // Only meaningful for leaf nodes; set by ComputeNeighborLevelFlags().
  bool all_neighbors_same_level_;

  static MeshBlockTree* proot_;
  static int nleaf_;
};

#endif // MESH_MESHBLOCK_TREE_HPP_
