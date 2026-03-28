#ifndef COMM_RECONCILE_FACES_HPP_
#define COMM_RECONCILE_FACES_HPP_
//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file reconcile_faces.hpp
//  \brief Post-initialization reconciliation of shared face-centered field
//  values.
//
//  After problem generator / restart / regrid initialization, same-level
//  blocks sharing a face may hold slightly different values at the shared
//  boundary face (set independently by each block).  ReconcileSharedFacesFC
//  makes those shared faces bitwise consistent by choosing the lower-gid
//  block's value as canonical.
//
//  This is a one-time fixup - during evolution the constrained-transport EMF
//  update maintains face consistency automatically.

#include <vector>

class Mesh;
class MeshBlock;

namespace comm
{

// Must be called OUTSIDE any OpenMP parallel region (uses MPI internally).
void ReconcileSharedFacesFC(Mesh* pm,
                            const std::vector<MeshBlock*>& pmb_array);

}  // namespace comm

#endif  // COMM_RECONCILE_FACES_HPP_
