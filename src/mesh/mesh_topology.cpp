//========================================================================================
// GR-Athena++
//========================================================================================
//! \file mesh_topology.cpp
//  \brief Implementations for mesh topology utilities: neighbor search, buffer ID logic,
//         boundary flag parsing, and NeighborBlock::SetNeighbor.
//
//  Extracted from the former src/bvals/bvals_base.cpp and src/bvals/utils/boundary_flag.cpp.

// C++ headers
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "mesh.hpp"
#include "mesh_topology.hpp"
#include "meshblock_tree.hpp"

//----------------------------------------------------------------------------------------
// Static topology tables (zero-initialized by default for static storage duration)
namespace mesh_topology {

NeighborIndexes ni[kMaxNeighbor];
int bufid[kMaxNeighbor];
int maxneighbor;

// Has InitBufferID been called yet?
static bool called_ = false;

} // namespace mesh_topology

//----------------------------------------------------------------------------------------
// NeighborBlock::SetNeighbor -- populate all fields of a NeighborBlock entry

void NeighborBlock::SetNeighbor(int irank, int ilevel, int igid, int ilid,
                                int iox1, int iox2, int iox3,
                                NeighborConnect itype, int ibid, int itargetid,
                                bool ipolar,
                                int ifi1,  // =0
                                int ifi2   // =0
                                ) {
  snb.rank = irank; snb.level = ilevel; snb.gid = igid; snb.lid = ilid;
  ni.ox1 = iox1; ni.ox2 = iox2; ni.ox3 = iox3;
  ni.type = itype; ni.fi1 = ifi1; ni.fi2 = ifi2;
  bufid = ibid; targetid = itargetid; polar = ipolar;
  neighbor_all_same_level = false;  // conservative default; set by SearchAndSetNeighbors
  if (ni.type == NeighborConnect::face) {
    if (ni.ox1 == -1)      fid = BoundaryFace::inner_x1;
    else if (ni.ox1 == 1)  fid = BoundaryFace::outer_x1;
    else if (ni.ox2 == -1) fid = BoundaryFace::inner_x2;
    else if (ni.ox2 == 1)  fid = BoundaryFace::outer_x2;
    else if (ni.ox3 == -1) fid = BoundaryFace::inner_x3;
    else if (ni.ox3 == 1)  fid = BoundaryFace::outer_x3;
  }
  if (ni.type == NeighborConnect::edge) {
    if (ni.ox3 == 0)      eid = (    (((ni.ox1 + 1) >> 1) | ((ni.ox2 + 1) & 2)));
    else if (ni.ox2 == 0) eid = (4 + (((ni.ox1 + 1) >> 1) | ((ni.ox3 + 1) & 2)));
    else if (ni.ox1 == 0) eid = (8 + (((ni.ox2 + 1) >> 1) | ((ni.ox3 + 1) & 2)));
  }
}

//----------------------------------------------------------------------------------------
// Buffer ID utilities

int mesh_topology::CreateBufferID(int ox1, int ox2, int ox3, int fi1, int fi2) {
  int ux1 = (ox1 + 1);
  int ux2 = (ox2 + 1);
  int ux3 = (ox3 + 1);
  return (ux1<<6) | (ux2<<4) | (ux3<<2) | (fi1<<1) | fi2;
}

int mesh_topology::InitBufferID(int dim, bool multilevel) {
  int nf1 = 1, nf2 = 1;
  if (multilevel) {
    if (dim >= 2) nf1 = 2;
    if (dim >= 3) nf2 = 2;
  }
  int b = 0;
  // x1 face
  for (int n=-1; n<=1; n+=2) {
    for (int f2=0; f2<nf2; f2++) {
      for (int f1=0; f1<nf1; f1++) {
        ni[b].ox1 = n; ni[b].ox2 = 0; ni[b].ox3 = 0;
        ni[b].fi1 = f1; ni[b].fi2 = f2;
        ni[b].type = NeighborConnect::face;
        b++;
      }
    }
  }
  // x2 face
  if (dim >= 2) {
    for (int n=-1; n<=1; n+=2) {
      for (int f2=0; f2<nf2; f2++) {
        for (int f1=0; f1<nf1; f1++) {
          ni[b].ox1 = 0; ni[b].ox2 = n; ni[b].ox3 = 0;
          ni[b].fi1 = f1; ni[b].fi2 = f2;
          ni[b].type = NeighborConnect::face;
          b++;
        }
      }
    }
  }
  if (dim == 3) {
    // x3 face
    for (int n=-1; n<=1; n+=2) {
      for (int f2=0; f2<nf2; f2++) {
        for (int f1=0; f1<nf1; f1++) {
          ni[b].ox1 = 0; ni[b].ox2 = 0; ni[b].ox3 = n;
          ni[b].fi1 = f1; ni[b].fi2 = f2;
          ni[b].type = NeighborConnect::face;
          b++;
        }
      }
    }
  }
  // edges: x1x2
  if (dim >= 2) {
    for (int m=-1; m<=1; m+=2) {
      for (int n=-1; n<=1; n+=2) {
        for (int f1=0; f1<nf2; f1++) {
          ni[b].ox1 = n; ni[b].ox2 = m; ni[b].ox3 = 0;
          ni[b].fi1 = f1; ni[b].fi2 = 0;
          ni[b].type = NeighborConnect::edge;
          b++;
        }
      }
    }
  }
  if (dim == 3) {
    // x1x3
    for (int m=-1; m<=1; m+=2) {
      for (int n=-1; n<=1; n+=2) {
        for (int f1=0; f1<nf1; f1++) {
          ni[b].ox1 = n; ni[b].ox2 = 0; ni[b].ox3 = m;
          ni[b].fi1 = f1; ni[b].fi2 = 0;
          ni[b].type = NeighborConnect::edge;
          b++;
        }
      }
    }
    // x2x3
    for (int m=-1; m<=1; m+=2) {
      for (int n=-1; n<=1; n+=2) {
        for (int f1=0; f1<nf1; f1++) {
          ni[b].ox1 = 0; ni[b].ox2 = n; ni[b].ox3 = m;
          ni[b].fi1 = f1; ni[b].fi2 = 0;
          ni[b].type = NeighborConnect::edge;
          b++;
        }
      }
    }
    // corners
    for (int l=-1; l<=1; l+=2) {
      for (int m=-1; m<=1; m+=2) {
        for (int n=-1; n<=1; n+=2) {
          ni[b].ox1 = n; ni[b].ox2 = m; ni[b].ox3 = l;
          ni[b].fi1 = 0; ni[b].fi2 = 0;
          ni[b].type = NeighborConnect::corner;
          b++;
        }
      }
    }
  }

  for (int n=0; n<b; n++)
    bufid[n] = CreateBufferID(ni[n].ox1, ni[n].ox2, ni[n].ox3, ni[n].fi1, ni[n].fi2);

  return b;
}

int mesh_topology::FindBufferID(int ox1, int ox2, int ox3, int fi1, int fi2) {
  int bid = CreateBufferID(ox1, ox2, ox3, fi1, fi2);
  for (int i=0; i<maxneighbor; i++) {
    if (bid == bufid[i]) return i;
  }
  return -1;
}

//----------------------------------------------------------------------------------------
// SearchAndSetNeighbors -- walks the MeshBlockTree to fill neighbor[], nneighbor, nblevel
// for a given MeshBlock.  block_bcs and loc are read from pmb->nc_ and pmb->loc.
// Friend access to NeighborConnectivity private members (neighbor_, nneighbor_, nblevel_).

void mesh_topology::SearchAndSetNeighbors(MeshBlock *pmb, MeshBlockTree &tree,
                                          int *ranklist, int *nslist) {
  // One-time init of the static ni[]/bufid[] tables
  if (!called_) {
    Mesh *pm = pmb->pmy_mesh;
    int dim = 1;
    if (pm->f2) dim = 2;
    if (pm->f3) dim = 3;
    maxneighbor = InitBufferID(dim, pm->multilevel);
    called_ = true;
  }

  LogicalLocation &loc = pmb->loc;
  RegionSize &block_size = pmb->block_size;
  Mesh *pmy_mesh = pmb->pmy_mesh;

  // Aliases into NeighborConnectivity owned data (friend access)
  comm::NeighborConnectivity &nc = pmb->nc_;
  NeighborBlock *neighbor = nc.neighbor_;
  int &nneighbor = nc.nneighbor_;
  int (&nblevel)[3][3][3] = nc.nblevel_;

  MeshBlockTree* neibt;
  int myox1, myox2 = 0, myox3 = 0, myfx1, myfx2, myfx3;
  myfx1 = ((loc.lx1 & 1LL) == 1LL);
  myfx2 = ((loc.lx2 & 1LL) == 1LL);
  myfx3 = ((loc.lx3 & 1LL) == 1LL);
  myox1 = ((loc.lx1 & 1LL) == 1LL)*2 - 1;
  if (block_size.nx2 > 1) myox2 = ((loc.lx2 & 1LL) == 1LL)*2 - 1;
  if (block_size.nx3 > 1) myox3 = ((loc.lx3 & 1LL) == 1LL)*2 - 1;

  int nf1 = 1, nf2 = 1;
  if (pmy_mesh->multilevel) {
    if (block_size.nx2 > 1) nf1 = 2;
    if (block_size.nx3 > 1) nf2 = 2;
  }
  int bid = 0;
  nc.ResetForSearch(loc.level);

  // x1 face
  for (int n=-1; n<=1; n+=2) {
    neibt = tree.FindNeighbor(loc, n, 0, 0);
    if (neibt == nullptr) { bid += nf1*nf2; continue;}
    if (neibt->pleaf_ != nullptr) { // neighbor at finer level
      int fface = 1 - (n + 1)/2;
      nblevel[1][1][n+1] = neibt->loc_.level + 1;
      for (int f2=0; f2<nf2; f2++) {
        for (int f1=0; f1<nf1; f1++) {
          MeshBlockTree* nf = neibt->GetLeaf(fface, f1, f2);
          int fid = nf->gid_;
          int nlevel = nf->loc_.level;
          int tbid = FindBufferID(-n, 0, 0, 0, 0);
          neighbor[nneighbor].SetNeighbor(
              ranklist[fid], nlevel, fid, fid-nslist[ranklist[fid]], n, 0, 0,
              NeighborConnect::face, bid, tbid, false, f1, f2);
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
          neighbor[nneighbor].neighbor_all_same_level =
              nf->all_neighbors_same_level_;
#endif
          bid++; nneighbor++;
        }
      }
    } else { // same or coarser level
      int nlevel = neibt->loc_.level;
      int nid = neibt->gid_;
      nblevel[1][1][n+1] = nlevel;
      int tbid;
      if (nlevel == loc.level) {
        tbid = FindBufferID(-n, 0, 0, 0, 0);
      } else {
        tbid = FindBufferID(-n, 0, 0, myfx2, myfx3);
      }
      neighbor[nneighbor].SetNeighbor(
          ranklist[nid], nlevel, nid, nid-nslist[ranklist[nid]], n, 0, 0,
          NeighborConnect::face, bid, tbid, false);
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
      neighbor[nneighbor].neighbor_all_same_level =
          neibt->all_neighbors_same_level_;
#endif
      bid += nf1*nf2; nneighbor++;
    }
  }
  if (block_size.nx2 == 1) return;

  // x2 face
  for (int n=-1; n<=1; n+=2) {
    neibt = tree.FindNeighbor(loc, 0, n, 0);
    if (neibt == nullptr) { bid += nf1*nf2; continue;}
    if (neibt->pleaf_ != nullptr) { // finer
      int fface = 1 - (n + 1)/2;
      nblevel[1][n+1][1] = neibt->loc_.level+1;
      for (int f2=0; f2<nf2; f2++) {
        for (int f1=0; f1<nf1; f1++) {
          MeshBlockTree* nf = neibt->GetLeaf(f1, fface, f2);
          int fid = nf->gid_;
          int nlevel = nf->loc_.level;
          int tbid = FindBufferID(0, -n, 0, 0, 0);
          neighbor[nneighbor].SetNeighbor(
              ranklist[fid], nlevel, fid, fid-nslist[ranklist[fid]], 0, n, 0,
              NeighborConnect::face, bid, tbid, false, f1, f2);
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
          neighbor[nneighbor].neighbor_all_same_level =
              nf->all_neighbors_same_level_;
#endif
          bid++; nneighbor++;
        }
      }
    } else { // same or coarser
      int nlevel = neibt->loc_.level;
      int nid = neibt->gid_;
      nblevel[1][n+1][1] = nlevel;
      int tbid;
      bool polar = false;
      if (nlevel == loc.level) {
        if ((n == -1 && nc.block_bcs_[BoundaryFace::inner_x2] == BoundaryFlag::polar)
            || (n == 1 && nc.block_bcs_[BoundaryFace::outer_x2] == BoundaryFlag::polar)){
          polar = true;
        }
        tbid = FindBufferID(0, polar ? n : -n, 0, 0, 0);
      } else {
        tbid = FindBufferID(0, -n, 0, myfx1, myfx3);
      }
      neighbor[nneighbor].SetNeighbor(
          ranklist[nid], nlevel, nid, nid-nslist[ranklist[nid]], 0, n, 0,
          NeighborConnect::face, bid, tbid, polar);
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
      neighbor[nneighbor].neighbor_all_same_level =
          neibt->all_neighbors_same_level_;
#endif
      bid += nf1*nf2; nneighbor++;
    }
  }

  // x3 face
  if (block_size.nx3 > 1) {
    for (int n=-1; n<=1; n+=2) {
      neibt = tree.FindNeighbor(loc, 0, 0, n);
      if (neibt == nullptr) { bid += nf1*nf2; continue;}
      if (neibt->pleaf_ != nullptr) { // finer
        int fface = 1 - (n + 1)/2;
        nblevel[n+1][1][1] = neibt->loc_.level+1;
        for (int f2=0; f2<nf2; f2++) {
          for (int f1=0; f1<nf1; f1++) {
            MeshBlockTree* nf = neibt->GetLeaf(f1, f2, fface);
            int fid = nf->gid_;
            int nlevel = nf->loc_.level;
            int tbid = FindBufferID(0, 0, -n, 0, 0);
            neighbor[nneighbor].SetNeighbor(
                ranklist[fid], nlevel, fid, fid-nslist[ranklist[fid]], 0, 0, n,
                NeighborConnect::face, bid, tbid, false, f1, f2);
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
            neighbor[nneighbor].neighbor_all_same_level =
                nf->all_neighbors_same_level_;
#endif
            bid++; nneighbor++;
          }
        }
      } else { // same or coarser
        int nlevel = neibt->loc_.level;
        int nid = neibt->gid_;
        nblevel[n+1][1][1] = nlevel;
        int tbid;
        if (nlevel == loc.level) {
          tbid = FindBufferID(0, 0, -n, 0, 0);
        } else {
          tbid = FindBufferID(0, 0, -n, myfx1, myfx2);
        }
        neighbor[nneighbor].SetNeighbor(
            ranklist[nid], nlevel, nid, nid-nslist[ranklist[nid]], 0, 0, n,
            NeighborConnect::face, bid, tbid, false);
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
        neighbor[nneighbor].neighbor_all_same_level =
            neibt->all_neighbors_same_level_;
#endif
        bid += nf1*nf2; nneighbor++;
      }
    }
  }

  // x1x2 edge
  for (int m=-1; m<=1; m+=2) {
    for (int n=-1; n<=1; n+=2) {
      neibt = tree.FindNeighbor(loc, n, m, 0);
      if (neibt == nullptr) { bid += nf2; continue;}
      bool polar = false;
      if ((m == -1 && nc.block_bcs_[BoundaryFace::inner_x2] == BoundaryFlag::polar)
          || (m == 1 && nc.block_bcs_[BoundaryFace::outer_x2] == BoundaryFlag::polar)) {
        polar = true;
      }
      if (neibt->pleaf_ != nullptr) { // finer
        int ff1 = 1 - (n + 1)/2;
        int ff2 = 1 - (m + 1)/2;
        if (polar) ff2 = 1 - ff2;
        nblevel[1][m+1][n+1] = neibt->loc_.level + 1;
        for (int f1=0; f1<nf2; f1++) {
          MeshBlockTree* nf = neibt->GetLeaf(ff1, ff2, f1);
          int fid = nf->gid_;
          int nlevel = nf->loc_.level;
          int tbid = FindBufferID(-n, polar ? m : -m, 0, 0, 0);
          neighbor[nneighbor].SetNeighbor(
              ranklist[fid], nlevel, fid, fid-nslist[ranklist[fid]], n, m, 0,
              NeighborConnect::edge, bid, tbid, polar, f1, 0);
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
          neighbor[nneighbor].neighbor_all_same_level =
              nf->all_neighbors_same_level_;
#endif
          bid++; nneighbor++;
        }
      } else { // same or coarser
        int nlevel = neibt->loc_.level;
        int nid = neibt->gid_;
        nblevel[1][m+1][n+1] = nlevel;
        int tbid;
        if (nlevel == loc.level) {
          tbid = FindBufferID(-n, polar ? m : -m, 0, 0, 0);
        } else {
          tbid = FindBufferID(-n, polar ? m : -m, 0, myfx3, 0);
        }
        if (nlevel >= loc.level || (myox1 == n && myox2 == m)) {
          neighbor[nneighbor].SetNeighbor(
              ranklist[nid], nlevel, nid, nid-nslist[ranklist[nid]], n, m, 0,
              NeighborConnect::edge, bid, tbid, polar);
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
          neighbor[nneighbor].neighbor_all_same_level =
              neibt->all_neighbors_same_level_;
#endif
          nneighbor++;
        }
        bid += nf2;
      }
    }
  }

  // polar neighbors (north)
  if (nc.block_bcs_[BoundaryFace::inner_x2] == BoundaryFlag::polar
      || nc.block_bcs_[BoundaryFace::inner_x2] == BoundaryFlag::polar_wedge) {
    int level = loc.level - pmy_mesh->GetRootLevel();
    int num_north_polar_blocks = static_cast<int>(pmy_mesh->GetNrbx3() * (1 << level));
    for (int n = 0; n < num_north_polar_blocks; ++n) {
      LogicalLocation neighbor_loc;
      neighbor_loc.lx1 = loc.lx1;
      neighbor_loc.lx2 = loc.lx2;
      neighbor_loc.lx3 = n;
      neighbor_loc.level = loc.level;
      neibt = tree.FindMeshBlock(neighbor_loc);
      if (neibt == nullptr || neibt->gid_ < 0) {
        std::stringstream msg;
        msg << "### FATAL ERROR in SearchAndSetNeighbors" << std::endl
            << "SMR with polar boundary in 3D requires all MeshBlocks around a pole\n"
            << "at the same radius are at the same refinement level.\n"
            << "Current MeshBlock LogicalLocation = ("
            << loc.lx1 << ", " << loc.lx2 << ", " << loc.lx3
            << "), azimuthal neighbor n=" << n << "/" << num_north_polar_blocks
            << std::endl;
        ATHENA_ERROR(msg);
      }
      int nid = neibt->gid_;
      pmb->polar_neighbor_north[neibt->loc_.lx3].rank = ranklist[nid];
      pmb->polar_neighbor_north[neibt->loc_.lx3].level = loc.level;
      pmb->polar_neighbor_north[neibt->loc_.lx3].lid = nid - nslist[ranklist[nid]];
      pmb->polar_neighbor_north[neibt->loc_.lx3].gid = nid;
    }
  }
  // polar neighbors (south)
  if (nc.block_bcs_[BoundaryFace::outer_x2] == BoundaryFlag::polar
      || nc.block_bcs_[BoundaryFace::outer_x2] == BoundaryFlag::polar_wedge) {
    int level = loc.level - pmy_mesh->GetRootLevel();
    int num_south_polar_blocks = static_cast<int>(pmy_mesh->GetNrbx3() * (1 << level));
    for (int n = 0; n < num_south_polar_blocks; ++n) {
      LogicalLocation neighbor_loc;
      neighbor_loc.lx1 = loc.lx1;
      neighbor_loc.lx2 = loc.lx2;
      neighbor_loc.lx3 = n;
      neighbor_loc.level = loc.level;
      neibt = tree.FindMeshBlock(neighbor_loc);
      if (neibt == nullptr || neibt->gid_ < 0) {
        std::stringstream msg;
        msg << "### FATAL ERROR in SearchAndSetNeighbors" << std::endl
            << "SMR with polar boundary in 3D requires all MeshBlocks around a pole\n"
            << "at the same radius are at the same refinement level.\n"
            << "Current MeshBlock LogicalLocation = ("
            << loc.lx1 << ", " << loc.lx2 << ", " << loc.lx3
            << "), azimuthal neighbor n=" << n << "/" << num_south_polar_blocks
            << std::endl;
        ATHENA_ERROR(msg);
      }
      int nid = neibt->gid_;
      pmb->polar_neighbor_south[neibt->loc_.lx3].rank = ranklist[nid];
      pmb->polar_neighbor_south[neibt->loc_.lx3].level = loc.level;
      pmb->polar_neighbor_south[neibt->loc_.lx3].lid = nid - nslist[ranklist[nid]];
      pmb->polar_neighbor_south[neibt->loc_.lx3].gid = nid;
    }
  }
  if (block_size.nx3 == 1) return;

  // x1x3 edge
  for (int m=-1; m<=1; m+=2) {
    for (int n=-1; n<=1; n+=2) {
      neibt = tree.FindNeighbor(loc, n, 0, m);
      if (neibt == nullptr) { bid += nf1; continue;}
      if (neibt->pleaf_ != nullptr) { // finer
        int ff1 = 1 - (n + 1)/2;
        int ff2 = 1 - (m + 1)/2;
        nblevel[m+1][1][n+1] = neibt->loc_.level + 1;
        for (int f1=0; f1<nf1; f1++) {
          MeshBlockTree* nf = neibt->GetLeaf(ff1, f1, ff2);
          int fid = nf->gid_;
          int nlevel = nf->loc_.level;
          int tbid = FindBufferID(-n, 0, -m, 0, 0);
          neighbor[nneighbor].SetNeighbor(
              ranklist[fid], nlevel, fid, fid-nslist[ranklist[fid]], n, 0, m,
              NeighborConnect::edge, bid, tbid, false, f1, 0);
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
          neighbor[nneighbor].neighbor_all_same_level =
              nf->all_neighbors_same_level_;
#endif
          bid++; nneighbor++;
        }
      } else { // same or coarser
        int nlevel = neibt->loc_.level;
        int nid = neibt->gid_;
        nblevel[m+1][1][n+1] = nlevel;
        int tbid;
        if (nlevel == loc.level) {
          tbid = FindBufferID(-n, 0, -m, 0, 0);
        } else {
          tbid = FindBufferID(-n, 0, -m, myfx2, 0);
        }
        if (nlevel >= loc.level || (myox1 == n && myox3 == m)) {
          neighbor[nneighbor].SetNeighbor(
              ranklist[nid], nlevel, nid, nid-nslist[ranklist[nid]], n, 0, m,
              NeighborConnect::edge, bid, tbid, false);
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
          neighbor[nneighbor].neighbor_all_same_level =
              neibt->all_neighbors_same_level_;
#endif
          nneighbor++;
        }
        bid += nf1;
      }
    }
  }

  // x2x3 edge
  for (int m=-1; m<=1; m+=2) {
    for (int n=-1; n<=1; n+=2) {
      neibt = tree.FindNeighbor(loc, 0, n, m);
      if (neibt == nullptr) { bid += nf1; continue;}
      if (neibt->pleaf_ != nullptr) { // finer
        int ff1 = 1 - (n + 1)/2;
        int ff2 = 1 - (m + 1)/2;
        nblevel[m+1][n+1][1] = neibt->loc_.level + 1;
        for (int f1=0; f1<nf1; f1++) {
          MeshBlockTree* nf = neibt->GetLeaf(f1, ff1, ff2);
          int fid = nf->gid_;
          int nlevel = nf->loc_.level;
          int tbid = FindBufferID(0, -n, -m, 0, 0);
          neighbor[nneighbor].SetNeighbor(
              ranklist[fid], nlevel, fid, fid-nslist[ranklist[fid]], 0, n, m,
              NeighborConnect::edge, bid, tbid, false, f1, 0);
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
          neighbor[nneighbor].neighbor_all_same_level =
              nf->all_neighbors_same_level_;
#endif
          bid++; nneighbor++;
        }
      } else { // same or coarser
        int nlevel = neibt->loc_.level;
        int nid = neibt->gid_;
        nblevel[m+1][n+1][1] = nlevel;
        int tbid;
        bool polar = false;
        if (nlevel == loc.level) {
          if ((n == -1 && nc.block_bcs_[BoundaryFace::inner_x2] == BoundaryFlag::polar)
              || (n == 1 && nc.block_bcs_[BoundaryFace::outer_x2]==BoundaryFlag::polar)){
            polar = true;
          }
          tbid = FindBufferID(0, polar ? n : -n, -m, 0, 0);
        } else {
          tbid = FindBufferID(0, -n, -m, myfx1, 0);
        }
        if (nlevel >= loc.level || (myox2 == n && myox3 == m)) {
          neighbor[nneighbor].SetNeighbor(
              ranklist[nid], nlevel, nid, nid-nslist[ranklist[nid]], 0, n, m,
              NeighborConnect::edge, bid, tbid, polar);
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
          neighbor[nneighbor].neighbor_all_same_level =
              neibt->all_neighbors_same_level_;
#endif
          nneighbor++;
        }
        bid += nf1;
      }
    }
  }

  // corners
  for (int l=-1; l<=1; l+=2) {
    for (int m=-1; m<=1; m+=2) {
      for (int n=-1; n<=1; n+=2) {
        neibt = tree.FindNeighbor(loc, n, m, l);
        if (neibt == nullptr) { bid++; continue;}
        bool polar = false;
        if ((m == -1 && nc.block_bcs_[BoundaryFace::inner_x2] == BoundaryFlag::polar)
            || (m == 1 && nc.block_bcs_[BoundaryFace::outer_x2]==BoundaryFlag::polar)) {
          polar = true;
        }
        if (neibt->pleaf_ != nullptr) { // finer
          int ff1 = 1 - (n + 1)/2;
          int ff2 = 1 - (m + 1)/2;
          int ff3 = 1 - (l + 1)/2;
          if (polar) ff2 = 1 - ff2;
          neibt = neibt->GetLeaf(ff1, ff2, ff3);
        }
        int nlevel = neibt->loc_.level;
        nblevel[l+1][m+1][n+1] = nlevel;
        if (nlevel >= loc.level || (myox1 == n && myox2 == m && myox3 == l)) {
          int nid = neibt->gid_;
          int tbid = FindBufferID(-n, polar ? m : -m, -l, 0, 0);
          neighbor[nneighbor].SetNeighbor(
              ranklist[nid], nlevel, nid, nid-nslist[ranklist[nid]], n, m, l,
              NeighborConnect::corner, bid, tbid, polar);
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
          neighbor[nneighbor].neighbor_all_same_level =
              neibt->all_neighbors_same_level_;
#endif
          nneighbor++;
        }
        bid++;
      }
    }
  }
}

//----------------------------------------------------------------------------------------
// Boundary flag string <-> enum conversions

BoundaryFlag GetBoundaryFlag(const std::string& input_string) {
  if (input_string == "reflecting") {
    return BoundaryFlag::reflect;
  } else if (input_string == "outflow") {
    return BoundaryFlag::outflow;
  } else if (input_string == "extrapolate_outflow") {
    return BoundaryFlag::extrapolate_outflow;
  } else if (input_string == "gr_sommerfeld") {
    return BoundaryFlag::gr_sommerfeld;
  } else if (input_string == "user") {
    return BoundaryFlag::user;
  } else if (input_string == "periodic") {
    return BoundaryFlag::periodic;
  } else if (input_string == "polar") {
    return BoundaryFlag::polar;
  } else if (input_string == "polar_wedge") {
    return BoundaryFlag::polar_wedge;
  } else if (input_string == "none") {
    return BoundaryFlag::undef;
  } else if (input_string == "block") {
    return BoundaryFlag::block;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in GetBoundaryFlag" << std::endl
        << "Input string=" << input_string << "\n"
        << "is an invalid boundary type" << std::endl;
    ATHENA_ERROR(msg);
  }
}

std::string GetBoundaryString(BoundaryFlag input_flag) {
  switch (input_flag) {
    case BoundaryFlag::block:
      return "block";
    case BoundaryFlag::undef:
      return "none";
    case BoundaryFlag::reflect:
      return "reflecting";
    case BoundaryFlag::extrapolate_outflow:
      return "extrapolate_outflow";
    case BoundaryFlag::gr_sommerfeld:
      return "gr_sommerfeld";
    case BoundaryFlag::outflow:
      return "outflow";
    case BoundaryFlag::user:
      return "user";
    case BoundaryFlag::periodic:
      return "periodic";
    case BoundaryFlag::polar:
      return "polar";
    case BoundaryFlag::polar_wedge:
      return "polar_wedge";
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in GetBoundaryString" << std::endl
          << "Input enum class BoundaryFlag=" << static_cast<int>(input_flag) << "\n"
          << "is an invalid boundary type" << std::endl;
      ATHENA_ERROR(msg);
      break;
  }
}

void CheckBoundaryFlag(BoundaryFlag block_flag, CoordinateDirection dir) {
  std::stringstream msg;
  msg << "### FATAL ERROR in CheckBoundaryFlag" << std::endl
      << "Attempting to set invalid MeshBlock boundary= " << GetBoundaryString(block_flag)
      << "\nin x" << dir+1 << " direction" << std::endl;
  switch(dir) {
    case CoordinateDirection::X1DIR:
      switch(block_flag) {
        case BoundaryFlag::polar:
        case BoundaryFlag::polar_wedge:
        case BoundaryFlag::undef:
          ATHENA_ERROR(msg);
          break;
        default:
          break;
      }
      break;
    case CoordinateDirection::X2DIR:
      switch(block_flag) {
        case BoundaryFlag::undef:
          ATHENA_ERROR(msg);
          break;
        default:
          break;
      }
      break;
    case CoordinateDirection::X3DIR:
      switch(block_flag) {
        case BoundaryFlag::polar:
        case BoundaryFlag::polar_wedge:
        case BoundaryFlag::undef:
          ATHENA_ERROR(msg);
          break;
        default:
          break;
      }
  }
}
