//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file reconcile_faces.cpp
//  \brief Post-initialization reconciliation of shared face-centered field
//  values.
//
//  After problem generator / restart / regrid initialization, same-level
//  blocks sharing a face may hold slightly different values at the shared
//  boundary face (set independently by each block).  This function makes those
//  shared faces bitwise consistent by choosing the lower-gid block's value as
//  canonical and overwriting the higher-gid block's value.
//
//  The normal FC ghost exchange deliberately excludes the shared boundary face
//  from both packing and unpacking - during evolution, consistency is
//  maintained by the constrained-transport EMF update.  This reconciliation is
//  therefore a one-time fixup needed only after initialization.

#include "reconcile_faces.hpp"

#include <vector>

#include "../athena.hpp"       // Real, FaceField, MPI_ATHENA_REAL
#include "../field/field.hpp"  // Field::b
#include "../globals.hpp"      // Globals::my_rank
#include "../mesh/mesh.hpp"  // Mesh, MeshBlock, NeighborBlock, NeighborConnect

namespace comm
{

//----------------------------------------------------------------------------------------
//! \fn void ReconcileSharedFacesFC(Mesh *pm, std::vector<MeshBlock*>
//! &pmb_array)
//  \brief Make shared boundary-face B values bitwise identical across
//  same-level
//         neighbors.  Lower-gid block's value is taken as canonical.
//
//  Must be called OUTSIDE any OpenMP parallel region (uses MPI internally).

void ReconcileSharedFacesFC(Mesh* pm, std::vector<MeshBlock*>& pmb_array)
{
  const int nmb = pmb_array.size();

#ifdef MPI_PARALLEL
  // Collect all MPI requests and their associated receive buffers for cleanup
  std::vector<MPI_Request> send_reqs;
  std::vector<Real*> send_bufs;
  std::vector<MPI_Request> recv_reqs;
  // Each recv entry: {buffer_ptr, dest_AthenaArray_ptr, face_dir, i/j/k
  // indices}
  struct RecvInfo
  {
    Real* buf;
    int buf_size;
    AthenaArray<Real>* dest;
    int dir;  // 0=x1, 1=x2, 2=x3
    // index ranges to unpack into
    int il, iu, jl, ju, kl, ku;
  };
  std::vector<RecvInfo> recv_info;
#endif

  for (int n = 0; n < nmb; ++n)
  {
    MeshBlock* pmb = pmb_array[n];
    FaceField& b   = pmb->pfield->b;

    const int is = pmb->is, ie = pmb->ie;
    const int js = pmb->js, je = pmb->je;
    const int ks = pmb->ks, ke = pmb->ke;
    const int my_gid   = pmb->gid;
    const int my_level = pmb->loc.level;

    for (int nb = 0; nb < pmb->nc().num_neighbors(); ++nb)
    {
      const NeighborBlock& nbr = pmb->nc().neighbor(nb);
      if (nbr.ni.type != NeighborConnect::face)
        continue;
      if (nbr.snb.level != my_level)
        continue;

      const int ngid = nbr.snb.gid;
      const int ox1 = nbr.ni.ox1, ox2 = nbr.ni.ox2, ox3 = nbr.ni.ox3;

      // Determine face direction and index
      int face_dir = -1;  // 0=x1, 1=x2, 2=x3
      if (ox1 != 0)
        face_dir = 0;
      else if (ox2 != 0)
        face_dir = 1;
      else if (ox3 != 0)
        face_dir = 2;

      if (my_gid < ngid)
      {
        // --- I am canonical (lower gid). Send my face value to neighbor. ---
        if (nbr.snb.rank == Globals::my_rank)
        {
          // Same rank: directly overwrite neighbor's face data
          MeshBlock* nb_pmb = pm->FindMeshBlock(ngid);
          FaceField& nb_b   = nb_pmb->pfield->b;

          if (face_dir == 0)
          {
            // shared x1-face
            int fi    = (ox1 > 0) ? ie + 1 : is;
            int nb_fi = (ox1 > 0) ? nb_pmb->is : nb_pmb->ie + 1;
            for (int k = ks; k <= ke; ++k)
              for (int j = js; j <= je; ++j)
                nb_b.x1f(k, j, nb_fi) = b.x1f(k, j, fi);
          }
          else if (face_dir == 1)
          {
            // shared x2-face
            int fj    = (ox2 > 0) ? je + 1 : js;
            int nb_fj = (ox2 > 0) ? nb_pmb->js : nb_pmb->je + 1;
            for (int k = ks; k <= ke; ++k)
              for (int i = is; i <= ie; ++i)
                nb_b.x2f(k, nb_fj, i) = b.x2f(k, fj, i);
          }
          else
          {
            // shared x3-face
            int fk    = (ox3 > 0) ? ke + 1 : ks;
            int nb_fk = (ox3 > 0) ? nb_pmb->ks : nb_pmb->ke + 1;
            for (int j = js; j <= je; ++j)
              for (int i = is; i <= ie; ++i)
                nb_b.x3f(nb_fk, j, i) = b.x3f(fk, j, i);
          }
        }
#ifdef MPI_PARALLEL
        else
        {
          // Cross-rank: pack and send
          int count = 0;
          if (face_dir == 0)
            count = (je - js + 1) * (ke - ks + 1);
          else if (face_dir == 1)
            count = (ie - is + 1) * (ke - ks + 1);
          else
            count = (ie - is + 1) * (je - js + 1);

          Real* sendbuf = new Real[count];
          int idx       = 0;

          if (face_dir == 0)
          {
            int fi = (ox1 > 0) ? ie + 1 : is;
            for (int k = ks; k <= ke; ++k)
              for (int j = js; j <= je; ++j)
                sendbuf[idx++] = b.x1f(k, j, fi);
          }
          else if (face_dir == 1)
          {
            int fj = (ox2 > 0) ? je + 1 : js;
            for (int k = ks; k <= ke; ++k)
              for (int i = is; i <= ie; ++i)
                sendbuf[idx++] = b.x2f(k, fj, i);
          }
          else
          {
            int fk = (ox3 > 0) ? ke + 1 : ks;
            for (int j = js; j <= je; ++j)
              for (int i = is; i <= ie; ++i)
                sendbuf[idx++] = b.x3f(fk, j, i);
          }

          // face_id: ox>0 -> even, ox<0 -> odd
          // x1: 0,1  x2: 2,3  x3: 4,5
          int face_id = face_dir * 2 + ((ox1 + ox2 + ox3) > 0 ? 0 : 1);
          int tag     = my_gid * 6 + face_id;

          MPI_Request req;
          MPI_Isend(sendbuf,
                    count,
                    MPI_ATHENA_REAL,
                    nbr.snb.rank,
                    tag,
                    MPI_COMM_WORLD,
                    &req);
          send_reqs.push_back(req);
          send_bufs.push_back(sendbuf);
        }
#endif
      }
      else
      {
        // --- I am non-canonical (higher gid). Receive from neighbor. ---
        if (nbr.snb.rank == Globals::my_rank)
        {
          // Same rank: already handled in the sender's iteration above
          // (the sender directly wrote into our b arrays)
        }
#ifdef MPI_PARALLEL
        else
        {
          // Cross-rank: post receive
          int count = 0;
          if (face_dir == 0)
            count = (je - js + 1) * (ke - ks + 1);
          else if (face_dir == 1)
            count = (ie - is + 1) * (ke - ks + 1);
          else
            count = (ie - is + 1) * (je - js + 1);

          Real* recvbuf = new Real[count];

          // Mirror the sender's face_id computation:
          // Sender has ox in the OPPOSITE direction from us.
          // Sender's face_id: face_dir*2 + (sender_ox > 0 ? 0 : 1)
          // Our ox is opposite to sender's, so sender_ox = -our_ox
          int sender_ox = -(ox1 + ox2 + ox3);
          int face_id   = face_dir * 2 + (sender_ox > 0 ? 0 : 1);
          int tag       = ngid * 6 + face_id;

          MPI_Request req;
          MPI_Irecv(recvbuf,
                    count,
                    MPI_ATHENA_REAL,
                    nbr.snb.rank,
                    tag,
                    MPI_COMM_WORLD,
                    &req);
          recv_reqs.push_back(req);

          // Determine target face index for unpacking
          int il = is, iu = ie, jl = js, ju = je, kl = ks, ku = ke;
          if (face_dir == 0)
          {
            int fi = (ox1 > 0) ? ie + 1 : is;
            il     = fi;
            iu     = fi;
          }
          else if (face_dir == 1)
          {
            int fj = (ox2 > 0) ? je + 1 : js;
            jl     = fj;
            ju     = fj;
          }
          else
          {
            int fk = (ox3 > 0) ? ke + 1 : ks;
            kl     = fk;
            ku     = fk;
          }

          AthenaArray<Real>* dest = nullptr;
          if (face_dir == 0)
            dest = &b.x1f;
          else if (face_dir == 1)
            dest = &b.x2f;
          else
            dest = &b.x3f;

          recv_info.push_back(
            { recvbuf, count, dest, face_dir, il, iu, jl, ju, kl, ku });
        }
#endif
      }
    }  // neighbor loop
  }  // MeshBlock loop

#ifdef MPI_PARALLEL
  // Wait for all receives to complete and unpack
  if (!recv_reqs.empty())
  {
    MPI_Waitall(static_cast<int>(recv_reqs.size()),
                recv_reqs.data(),
                MPI_STATUSES_IGNORE);

    for (auto& ri : recv_info)
    {
      int idx                = 0;
      AthenaArray<Real>& arr = *(ri.dest);
      if (ri.dir == 0)
      {
        // x1f: arr(k, j, fi)
        for (int k = ri.kl; k <= ri.ku; ++k)
          for (int j = ri.jl; j <= ri.ju; ++j)
            arr(k, j, ri.il) = ri.buf[idx++];
      }
      else if (ri.dir == 1)
      {
        // x2f: arr(k, fj, i)
        for (int k = ri.kl; k <= ri.ku; ++k)
          for (int i = ri.il; i <= ri.iu; ++i)
            arr(k, ri.jl, i) = ri.buf[idx++];
      }
      else
      {
        // x3f: arr(fk, j, i)
        for (int j = ri.jl; j <= ri.ju; ++j)
          for (int i = ri.il; i <= ri.iu; ++i)
            arr(ri.kl, j, i) = ri.buf[idx++];
      }
      delete[] ri.buf;
    }
  }

  // Wait for all sends to complete and clean up
  if (!send_reqs.empty())
  {
    MPI_Waitall(static_cast<int>(send_reqs.size()),
                send_reqs.data(),
                MPI_STATUSES_IGNORE);
    for (auto* p : send_bufs)
      delete[] p;
  }
#endif
}

}  // namespace comm
