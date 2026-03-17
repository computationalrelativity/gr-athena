// C headers

// C++ headers
#include <iostream>
#include <limits>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_aliases.hpp"
#include "../globals.hpp"
#include "mesh.hpp"

#include "../hydro/hydro.hpp"
#include "../field/field.hpp"
#include "../m1/m1.hpp"
#include "../scalars/scalars.hpp"
#include "../wave/wave.hpp"
#include "../z4c/z4c.hpp"

#include "../comm/comm_registry.hpp"

void Mesh::FinalizeWave(std::vector<MeshBlock*> & pmb_array)
{
  MeshBlock *pmb;

  const int nmb = pmb_array.size();

  Wave *pw = nullptr;

  #pragma omp for private(pmb, pw)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    pw = pmb->pwave;

    comm::CommRegistry *pcomm = pmb->pcomm;

    // Wave uses module-specific coarse indices from MB_info.
    const int cil = pw->mbi.cil, ciu = pw->mbi.ciu;
    const int cjl = pw->mbi.cjl, cju = pw->mbi.cju;
    const int ckl = pw->mbi.ckl, cku = pw->mbi.cku;
    const int cng = pw->mbi.cng;

    // Prolongation: coarse-level BCs then prolongate each Wave channel.
    if (multilevel) {
      pcomm->ProlongateAndApplyPhysicalBCs(
          comm::CommGroup::Wave, time, 0.0, cil, ciu, cjl, cju, ckl, cku, cng);
    }

    // Fine-level physical BCs for every Wave channel.
    pcomm->ApplyPhysicalBCs(comm::CommGroup::Wave, time, 0.0);
  }
}

void Mesh::FinalizeZ4cADMPhysical(std::vector<MeshBlock*> & pmb_array,
                                  const bool enforce_alg)
{
#if Z4C_ENABLED
  MeshBlock *pmb;

  const int nmb = pmb_array.size();

  Z4c *pz = nullptr;

  #pragma omp for private(pmb, pz)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];

    pz = pmb->pz4c;

    const bool skip_physical = false;

    // Enforce the algebraic constraints
    if (enforce_alg)
    {
      pz->AlgConstr(pz->storage.u,
                    pz->mbi.il, pz->mbi.iu,
                    pz->mbi.jl, pz->mbi.ju,
                    pz->mbi.kl, pz->mbi.ku,
                    skip_physical);
    }

    // Need ADM variables for con2prim
    pz->Z4cToADM(pz->storage.u,
                 pz->storage.adm,
                 pz->mbi.il, pz->mbi.iu,
                 pz->mbi.jl, pz->mbi.ju,
                 pz->mbi.kl, pz->mbi.ku,
                 skip_physical);
  }
#endif // Z4C_ENABLED
}

void Mesh::FinalizeZ4cADMGhosts(std::vector<MeshBlock*> & pmb_array,
                                const bool enforce_alg)
{
#if Z4C_ENABLED
  MeshBlock *pmb;

  const int nmb = pmb_array.size();

  Z4c *pz = nullptr;

  #pragma omp for private(pmb, pz)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    pz = pmb->pz4c;

    comm::CommRegistry *pcomm = pmb->pcomm;

    // Z4c uses module-specific coarse indices from MB_info.
    const int cil = pz->mbi.cil, ciu = pz->mbi.ciu;
    const int cjl = pz->mbi.cjl, cju = pz->mbi.cju;
    const int ckl = pz->mbi.ckl, cku = pz->mbi.cku;
    const int cng = pz->mbi.cng;

    // Prolongation: coarse-level BCs then prolongate each Z4c channel.
    if (multilevel) {
      pcomm->ProlongateAndApplyPhysicalBCs(
          comm::CommGroup::Z4c, time, 0.0, cil, ciu, cjl, cju, ckl, cku, cng);
    }

    // Fine-level physical BCs for every Z4c channel.
    pcomm->ApplyPhysicalBCs(comm::CommGroup::Z4c, time, 0.0);

    const bool skip_physical = true;

    // Enforce the algebraic constraints
    if (enforce_alg)
    {
      pz->AlgConstr(pz->storage.u,
        0, pz->mbi.nn1-1,
        0, pz->mbi.nn2-1,
        0, pz->mbi.nn3-1,
        skip_physical);
    }

    // Need ADM variables for con2prim
    pz->Z4cToADM(pz->storage.u,
      pz->storage.adm,
      0, pz->mbi.nn1-1,
      0, pz->mbi.nn2-1,
      0, pz->mbi.nn3-1,
      skip_physical);

    // Initialize 3D derivative arrays + storage.aux for fresh MeshBlocks
    pz->InitializeZ4cDerivatives(pz->storage.u);
  }
#endif // Z4C_ENABLED
}

void Mesh::FinalizeZ4cADM_Matter(std::vector<MeshBlock*> & pmb_array)
{
#if defined(Z4C_WITH_HYDRO_ENABLED)
  MeshBlock *pmb = nullptr;

  const int nmb = pmb_array.size();

  Field *pf = nullptr;
  Hydro *ph = nullptr;
  PassiveScalars *ps = nullptr;
  Z4c *pz = nullptr;

  #pragma omp for private(pmb, pf, ph, ps, pz)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];

    pf = pmb->pfield;
    ph = pmb->phydro;
    ps = pmb->pscalars;
    pz = pmb->pz4c;

    // Try to smooth temperature with nn avg:
    if (pmb->peos->smooth_temperature)
    {
      const bool exclude_first_extrema = true;

      AA src;
      AA tar;

      int il = 0;
      int iu = pmb->ncells1-1;
      int jl = 0;
      int ju = pmb->ncells2-1;
      int kl = 0;
      int ku = pmb->ncells3-1;

      src.InitWithShallowSlice(ph->derived_ms, IX_T, 1);
      tar.InitWithShallowSlice(ph->w1, 0, 1);

      pmb->peos->NearestNeighborSmooth(tar, src, il, iu, jl, ju, kl, ku,
                                       exclude_first_extrema);

      CC_GLOOP3(k,j,i)
      {
        ph->derived_ms(IX_T,k,j,i) = tar(k,j,i);
      }

      if (pmb->peos->recompute_enthalpy)
      {
        // Hoist loop-invariant EOS query out of the omp simd inner loop
        const Real mb = pmb->peos->GetEOS().GetBaryonMass();

        CC_GLOOP3(k,j,i)
        {
          Real Y[MAX_SPECIES] = {0.0};
          for (int l=0; l<NSCALARS; l++)
          {
            Y[l] = ps->r(l,k,j,i);
          }

          const Real n = ph->w(IDN,k,j,i) / mb;

          ph->derived_ms(IX_ETH,k,j,i) = pmb->peos->GetEOS().GetEnthalpy(
            n, ph->derived_ms(IX_T,k,j,i), Y
          );
        }
      }
    }

    pz->GetMatter(pz->storage.mat, pz->storage.adm, ph->w, ps->r, pf->bcc);

    // pmb->DebugMeshBlock(-15,-15,-15, 2, 20, 3, "@S:Sc\n", "@E:Sc\n");

  }
#endif // Z4C_ENABLED
}

void Mesh::FinalizeM1(std::vector<MeshBlock*> & pmb_array)
{
#if M1_ENABLED
  MeshBlock *pmb;

  const int nmb = pmb_array.size();

  M1::M1 *pm1 = nullptr;

  #pragma omp for private(pmb, pm1)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    pm1 = pmb->pm1;

    comm::CommRegistry *pcomm = pmb->pcomm;

    // M1 is CC-only; uses standard MeshBlock coarse indices.
    const int cis = pmb->cis, cie = pmb->cie;
    const int cjs = pmb->cjs, cje = pmb->cje;
    const int cks = pmb->cks, cke = pmb->cke;

    // Prolongation: coarse-level BCs then prolongate each M1 channel.
    if (multilevel) {
      pcomm->ProlongateAndApplyPhysicalBCs(
          comm::CommGroup::M1, time, 0.0, cis, cie, cjs, cje, cks, cke, NGHOST);
    }

    // Fine-level physical BCs for every M1 channel.
    pcomm->ApplyPhysicalBCs(comm::CommGroup::M1, time, 0.0);

    // Conserved variables are now available globally;
    // Ensure that geometric & hydro terms are available
    pm1->UpdateGeometry(pm1->geom, pm1->scratch);
    pm1->UpdateHydro(pm1->hydro, pm1->geom, pm1->scratch);
    // Need the reference velocity & closure
    pm1->CalcFiducialVelocity();
    pm1->CalcClosure(pm1->storage.u);
  }
#endif // M1_ENABLED
}

void Mesh::FinalizeHydro_pgen(std::vector<MeshBlock*> & pmb_array)
{
#if FLUID_ENABLED
  MeshBlock *pmb;

  const int nmb = pmb_array.size();

  Hydro *ph = nullptr;

  #pragma omp for private(pmb, ph)
  for (int i = 0; i < nmb; ++i)
  {
    pmb = pmb_array[i];
    ph = pmb->phydro;

    const int il = 0, iu = (pmb->ncells1 > 1)? pmb->ncells1 - 1 : 0;
    const int jl = 0, ju = (pmb->ncells2 > 1)? pmb->ncells2 - 1 : 0;
    const int kl = 0, ku = (pmb->ncells3 > 1)? pmb->ncells3 - 1 : 0;

    ph->RetainState(ph->w1, ph->w, il, iu, jl, ju, kl, ku);
  }
#endif // FLUID_ENABLED
}

void Mesh::FinalizeHydroConsRP(std::vector<MeshBlock*> & pmb_array)
{
  MeshBlock *pmb;

  const int nmb = pmb_array.size();

  Field *pf = nullptr;
  Hydro *ph = nullptr;
  PassiveScalars *ps = nullptr;

  #pragma omp for private(pmb, ph, ps, pf)
  for (int i = 0; i < nmb; ++i)
  {
    pmb = pmb_array[i];

    pf = pmb->pfield;
    ph = pmb->phydro;
    ps = pmb->pscalars;

    comm::CommRegistry *pcomm = pmb->pcomm;

    const int cis = pmb->cis, cie = pmb->cie;
    const int cjs = pmb->cjs, cje = pmb->cje;
    const int cks = pmb->cks, cke = pmb->cke;

    // Prolongation: coarse-level BCs then prolongate each MainInt channel.
    if (multilevel) {
      pcomm->ProlongateAndApplyPhysicalBCs(
          comm::CommGroup::MainInt, time, 0.0,
          cis, cie, cjs, cje, cks, cke, NGHOST);

      if (MAGNETIC_FIELDS_ENABLED)
        pmb->CalculateCellCenteredFieldOnProlongedBoundaries();
    }

    // N.B.
    // Results in two-fold application of BC to magnetic fields;
    // but that is harmless
    pcomm->ApplyPhysicalBCs(comm::CommGroup::MainInt, time, 0.0);

    if (MAGNETIC_FIELDS_ENABLED)
    {
      const int il = 0, iu = (pmb->ncells1 > 1)? pmb->ncells1 - 1 : 0;
      const int jl = 0, ju = (pmb->ncells2 > 1)? pmb->ncells2 - 1 : 0;
      const int kl = 0, ku = (pmb->ncells3 > 1)? pmb->ncells3 - 1 : 0;

      pf->CalculateCellCenteredField(pf->b, pf->bcc, pmb->pcoord,
                                     il, iu, jl, ju, kl, ku);
    }
  }
}

void Mesh::PreparePrimitives(std::vector<MeshBlock*> & pmb_array,
                             const bool interior_only,
                             const bool skip_physical)
{
#if FLUID_ENABLED
  MeshBlock *pmb;

  const int nmb = pmb_array.size();

  Field *pf = nullptr;
  Hydro *ph = nullptr;
  PassiveScalars *ps = nullptr;

  #pragma omp for private(pmb, pf, ph, ps)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];

    ph = pmb->phydro;
    pf = pmb->pfield;
    ps = pmb->pscalars;

    int il = pmb->is, iu = pmb->ie,
        jl = pmb->js, ju = pmb->je,
        kl = pmb->ks, ku = pmb->ke;

    if (!interior_only)
    {
      il = 0, iu = (pmb->ncells1 > 1)? pmb->ncells1 - 1 : 0;
      jl = 0, ju = (pmb->ncells2 > 1)? pmb->ncells2 - 1 : 0;
      kl = 0, ku = (pmb->ncells3 > 1)? pmb->ncells3 - 1 : 0;
    }

    static const int coarseflag = 0;
    pmb->peos->ConservedToPrimitive(ph->u, ph->w1, ph->w,
                                    ps->s, ps->r,
                                    pf->bcc, pmb->pcoord,
                                    il, iu, jl, ju, kl, ku,
                                    coarseflag, skip_physical);

    // Update w1 to have the state of w
    ph->RetainState(ph->w1, ph->w, il, iu, jl, ju, kl, ku);
  }

#endif // FLUID_ENABLED
}

void Mesh::CommunicateConserved(std::vector<MeshBlock*> & pmb_array)
{
  MeshBlock *pmb;

  const int nmb = pmb_array.size();

  // Collect all groups that need initial ghost exchange.
  // Order matches old BoundaryCommSubset::mesh_init (all registered bvars).
  std::vector<comm::CommGroup> groups;
  if (FLUID_ENABLED)
    groups.push_back(comm::CommGroup::MainInt);
  if (WAVE_ENABLED)
    groups.push_back(comm::CommGroup::Wave);
  if (Z4C_ENABLED)
    groups.push_back(comm::CommGroup::Z4c);
  if (M1_ENABLED)
    groups.push_back(comm::CommGroup::M1);

  // For each active group: start -> send -> spin-wait receive -> set -> clear.
  for (const auto grp : groups) {
    #pragma omp for private(pmb)
    for (int i = 0; i < nmb; ++i) {
      pmb = pmb_array[i];
      pmb->pcomm->StartReceiving(grp);
    }

    #pragma omp for private(pmb)
    for (int i = 0; i < nmb; ++i) {
      pmb = pmb_array[i];
      pmb->pcomm->SendBoundaryBuffers(grp);
    }

    #pragma omp for private(pmb)
    for (int i = 0; i < nmb; ++i) {
      pmb = pmb_array[i];
      while (!pmb->pcomm->ReceiveBoundaryBuffers(grp)) {}
      pmb->pcomm->SetBoundaries(grp);
      pmb->pcomm->ClearBoundary(grp);
    }
  }
}

//----------------------------------------------------------------------------------------
// After pgen initialization, same-level blocks sharing a face may have slightly
// different values at their shared boundary face.
//
// This function makes those shared faces bitwise consistent by choosing the lower-gid
// block's value as canonical and overwriting the higher-gid block's value.
//
// Must be called OUTSIDE any OpenMP parallel region (uses MPI internally).
//----------------------------------------------------------------------------------------

void Mesh::ReconcileSharedFacesFC(std::vector<MeshBlock*> &pmb_array) {
#if !MAGNETIC_FIELDS_ENABLED
  return;
#endif
  const int nmb = pmb_array.size();

#ifdef MPI_PARALLEL
  // Collect all MPI requests and their associated receive buffers for cleanup
  std::vector<MPI_Request> send_reqs;
  std::vector<Real*> send_bufs;
  std::vector<MPI_Request> recv_reqs;
  // Each recv entry: {buffer_ptr, dest_AthenaArray_ptr, face_dir, i/j/k indices}
  struct RecvInfo {
    Real *buf;
    int buf_size;
    AthenaArray<Real> *dest;
    int dir;  // 0=x1, 1=x2, 2=x3
    // index ranges to unpack into
    int il, iu, jl, ju, kl, ku;
  };
  std::vector<RecvInfo> recv_info;
#endif

  for (int n = 0; n < nmb; ++n) {
    MeshBlock *pmb = pmb_array[n];
    FaceField &b = pmb->pfield->b;

    const int is = pmb->is, ie = pmb->ie;
    const int js = pmb->js, je = pmb->je;
    const int ks = pmb->ks, ke = pmb->ke;
    const int my_gid = pmb->gid;
    const int my_level = pmb->loc.level;

    for (int nb = 0; nb < pmb->nc().num_neighbors(); ++nb) {
      const NeighborBlock &nbr = pmb->nc().neighbor(nb);
      if (nbr.ni.type != NeighborConnect::face) continue;
      if (nbr.snb.level != my_level) continue;

      const int ngid = nbr.snb.gid;
      const int ox1 = nbr.ni.ox1, ox2 = nbr.ni.ox2, ox3 = nbr.ni.ox3;

      // Determine face direction and index
      int face_dir = -1;  // 0=x1, 1=x2, 2=x3
      if (ox1 != 0) face_dir = 0;
      else if (ox2 != 0) face_dir = 1;
      else if (ox3 != 0) face_dir = 2;

      if (my_gid < ngid) {
        // --- I am canonical (lower gid). Send my face value to neighbor. ---
        if (nbr.snb.rank == Globals::my_rank) {
          // Same rank: directly overwrite neighbor's face data
          MeshBlock *nb_pmb = FindMeshBlock(ngid);
          FaceField &nb_b = nb_pmb->pfield->b;

          if (face_dir == 0) {
            // shared x1-face
            int fi = (ox1 > 0) ? ie + 1 : is;
            int nb_fi = (ox1 > 0) ? nb_pmb->is : nb_pmb->ie + 1;
            for (int k = ks; k <= ke; ++k)
              for (int j = js; j <= je; ++j)
                nb_b.x1f(k, j, nb_fi) = b.x1f(k, j, fi);
          } else if (face_dir == 1) {
            // shared x2-face
            int fj = (ox2 > 0) ? je + 1 : js;
            int nb_fj = (ox2 > 0) ? nb_pmb->js : nb_pmb->je + 1;
            for (int k = ks; k <= ke; ++k)
              for (int i = is; i <= ie; ++i)
                nb_b.x2f(k, nb_fj, i) = b.x2f(k, fj, i);
          } else {
            // shared x3-face
            int fk = (ox3 > 0) ? ke + 1 : ks;
            int nb_fk = (ox3 > 0) ? nb_pmb->ks : nb_pmb->ke + 1;
            for (int j = js; j <= je; ++j)
              for (int i = is; i <= ie; ++i)
                nb_b.x3f(nb_fk, j, i) = b.x3f(fk, j, i);
          }
        }
#ifdef MPI_PARALLEL
        else {
          // Cross-rank: pack and send
          int count = 0;
          if (face_dir == 0) count = (je - js + 1) * (ke - ks + 1);
          else if (face_dir == 1) count = (ie - is + 1) * (ke - ks + 1);
          else count = (ie - is + 1) * (je - js + 1);

          Real *sendbuf = new Real[count];
          int idx = 0;

          if (face_dir == 0) {
            int fi = (ox1 > 0) ? ie + 1 : is;
            for (int k = ks; k <= ke; ++k)
              for (int j = js; j <= je; ++j)
                sendbuf[idx++] = b.x1f(k, j, fi);
          } else if (face_dir == 1) {
            int fj = (ox2 > 0) ? je + 1 : js;
            for (int k = ks; k <= ke; ++k)
              for (int i = is; i <= ie; ++i)
                sendbuf[idx++] = b.x2f(k, fj, i);
          } else {
            int fk = (ox3 > 0) ? ke + 1 : ks;
            for (int j = js; j <= je; ++j)
              for (int i = is; i <= ie; ++i)
                sendbuf[idx++] = b.x3f(fk, j, i);
          }

          // face_id: ox>0 -> even, ox<0 -> odd
          // x1: 0,1  x2: 2,3  x3: 4,5
          int face_id = face_dir * 2 + ((ox1 + ox2 + ox3) > 0 ? 0 : 1);
          int tag = my_gid * 6 + face_id;

          MPI_Request req;
          MPI_Isend(sendbuf, count, MPI_ATHENA_REAL,
                    nbr.snb.rank, tag, MPI_COMM_WORLD, &req);
          send_reqs.push_back(req);
          send_bufs.push_back(sendbuf);
        }
#endif
      } else {
        // --- I am non-canonical (higher gid). Receive from neighbor. ---
        if (nbr.snb.rank == Globals::my_rank) {
          // Same rank: already handled in the sender's iteration above
          // (the sender directly wrote into our b arrays)
        }
#ifdef MPI_PARALLEL
        else {
          // Cross-rank: post receive
          int count = 0;
          if (face_dir == 0) count = (je - js + 1) * (ke - ks + 1);
          else if (face_dir == 1) count = (ie - is + 1) * (ke - ks + 1);
          else count = (ie - is + 1) * (je - js + 1);

          Real *recvbuf = new Real[count];

          // Mirror the sender's face_id computation:
          // Sender has ox in the OPPOSITE direction from us.
          // Sender's face_id: face_dir*2 + (sender_ox > 0 ? 0 : 1)
          // Our ox is opposite to sender's, so sender_ox = -our_ox
          int sender_ox = -(ox1 + ox2 + ox3);
          int face_id = face_dir * 2 + (sender_ox > 0 ? 0 : 1);
          int tag = ngid * 6 + face_id;

          MPI_Request req;
          MPI_Irecv(recvbuf, count, MPI_ATHENA_REAL,
                    nbr.snb.rank, tag, MPI_COMM_WORLD, &req);
          recv_reqs.push_back(req);

          // Determine target face index for unpacking
          int il = is, iu = ie, jl = js, ju = je, kl = ks, ku = ke;
          if (face_dir == 0) {
            int fi = (ox1 > 0) ? ie + 1 : is;
            il = fi; iu = fi;
          } else if (face_dir == 1) {
            int fj = (ox2 > 0) ? je + 1 : js;
            jl = fj; ju = fj;
          } else {
            int fk = (ox3 > 0) ? ke + 1 : ks;
            kl = fk; ku = fk;
          }

          AthenaArray<Real> *dest = nullptr;
          if (face_dir == 0) dest = &b.x1f;
          else if (face_dir == 1) dest = &b.x2f;
          else dest = &b.x3f;

          recv_info.push_back({recvbuf, count, dest, face_dir,
                               il, iu, jl, ju, kl, ku});
        }
#endif
      }
    }  // neighbor loop
  }  // MeshBlock loop

#ifdef MPI_PARALLEL
  // Wait for all receives to complete and unpack
  if (!recv_reqs.empty()) {
    MPI_Waitall(static_cast<int>(recv_reqs.size()),
                recv_reqs.data(), MPI_STATUSES_IGNORE);

    for (auto &ri : recv_info) {
      int idx = 0;
      AthenaArray<Real> &arr = *(ri.dest);
      if (ri.dir == 0) {
        // x1f: arr(k, j, fi)
        for (int k = ri.kl; k <= ri.ku; ++k)
          for (int j = ri.jl; j <= ri.ju; ++j)
            arr(k, j, ri.il) = ri.buf[idx++];
      } else if (ri.dir == 1) {
        // x2f: arr(k, fj, i)
        for (int k = ri.kl; k <= ri.ku; ++k)
          for (int i = ri.il; i <= ri.iu; ++i)
            arr(k, ri.jl, i) = ri.buf[idx++];
      } else {
        // x3f: arr(fk, j, i)
        for (int j = ri.jl; j <= ri.ju; ++j)
          for (int i = ri.il; i <= ri.iu; ++i)
            arr(ri.kl, j, i) = ri.buf[idx++];
      }
      delete[] ri.buf;
    }
  }

  // Wait for all sends to complete and clean up
  if (!send_reqs.empty()) {
    MPI_Waitall(static_cast<int>(send_reqs.size()),
                send_reqs.data(), MPI_STATUSES_IGNORE);
    for (auto *p : send_bufs) delete[] p;
  }
#endif
}

void Mesh::CommunicateConservedMatter(std::vector<MeshBlock*> & pmb_array)
{
  MeshBlock *pmb;

  const int nmb = pmb_array.size();

  // MainInt group covers hydro + field + scalars.
  const auto grp = comm::CommGroup::MainInt;

  #pragma omp for private(pmb)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    pmb->pcomm->StartReceiving(grp);
  }

  #pragma omp for private(pmb)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    pmb->pcomm->SendBoundaryBuffers(grp);
  }

  #pragma omp for private(pmb)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    while (!pmb->pcomm->ReceiveBoundaryBuffers(grp)) {}
    pmb->pcomm->SetBoundaries(grp);
    pmb->pcomm->ClearBoundary(grp);
  }
}


void Mesh::CommunicateAuxZ4c()
{
  // short-circuit if this is called without z4c
  if (!Z4C_ENABLED)
  {
    return;
  }

  int nthreads = GetNumMeshThreads();
  (void)nthreads;
  std::vector<MeshBlock*> pmb_array;
  GetMeshBlocksMyRank(pmb_array);
  const int nmb = pmb_array.size();

  const auto grp = comm::CommGroup::Aux;

  #pragma omp parallel num_threads(nthreads)
  {
    MeshBlock *pmb = nullptr;

    #pragma omp for private(pmb)
    for (int i=0; i<nmb; ++i)
    {
      pmb = pmb_array[i];
      pmb->pcomm->StartReceiving(grp);
    }

    #pragma omp for private(pmb)
    for (int i=0; i<nmb; ++i)
    {
      pmb = pmb_array[i];
      pmb->pcomm->SendBoundaryBuffers(grp);
    }

    #pragma omp for private(pmb)
    for (int i=0; i<nmb; ++i)
    {
      pmb = pmb_array[i];
      while (!pmb->pcomm->ReceiveBoundaryBuffers(grp)) {}
      pmb->pcomm->SetBoundaries(grp);
      pmb->pcomm->ClearBoundary(grp);
    }

    #pragma omp for private(pmb)
    for (int i=0; i<nmb; ++i)
    {
      pmb = pmb_array[i];
      Z4c *pz = pmb->pz4c;

      comm::CommRegistry *pcomm = pmb->pcomm;

      // Z4c uses module-specific coarse indices from MB_info.
      const int cil = pz->mbi.cil, ciu = pz->mbi.ciu;
      const int cjl = pz->mbi.cjl, cju = pz->mbi.cju;
      const int ckl = pz->mbi.ckl, cku = pz->mbi.cku;
      const int cng = pz->mbi.cng;

      // Prolongation: coarse-level BCs then prolongate each Aux channel.
      if (multilevel) {
        pcomm->ProlongateAndApplyPhysicalBCs(
            grp, time, 0.0, cil, ciu, cjl, cju, ckl, cku, cng);
      }

      // Fine-level physical BCs for every Aux channel.
      pcomm->ApplyPhysicalBCs(grp, time, 0.0);
    }

  }
}

void Mesh::CommunicateAuxADM()
{
  // short-circuit if this is called without z4c or no com. is required
  if (!Z4C_ENABLED || !pblock->pz4c->opt.communicate_aux_adm)
  {
    return;
  }

  int nthreads = GetNumMeshThreads();
  (void)nthreads;
  std::vector<MeshBlock*> pmb_array;
  GetMeshBlocksMyRank(pmb_array);
  const int nmb = pmb_array.size();

  const auto grp = comm::CommGroup::AuxADM;

  #pragma omp parallel num_threads(nthreads)
  {
    MeshBlock *pmb = nullptr;

    #pragma omp for private(pmb)
    for (int i=0; i<nmb; ++i)
    {
      pmb = pmb_array[i];
      pmb->pcomm->StartReceiving(grp);
    }

    #pragma omp for private(pmb)
    for (int i=0; i<nmb; ++i)
    {
      pmb = pmb_array[i];
      pmb->pcomm->SendBoundaryBuffers(grp);
    }

    #pragma omp for private(pmb)
    for (int i=0; i<nmb; ++i)
    {
      pmb = pmb_array[i];
      while (!pmb->pcomm->ReceiveBoundaryBuffers(grp)) {}
      pmb->pcomm->SetBoundaries(grp);
      pmb->pcomm->ClearBoundary(grp);
    }

    #pragma omp for private(pmb)
    for (int i=0; i<nmb; ++i)
    {
      pmb = pmb_array[i];
      Z4c *pz = pmb->pz4c;

      comm::CommRegistry *pcomm = pmb->pcomm;

      // Z4c uses module-specific coarse indices from MB_info.
      const int cil = pz->mbi.cil, ciu = pz->mbi.ciu;
      const int cjl = pz->mbi.cjl, cju = pz->mbi.cju;
      const int ckl = pz->mbi.ckl, cku = pz->mbi.cku;
      const int cng = pz->mbi.cng;

      // Prolongation: coarse-level BCs then prolongate each AuxADM channel.
      if (multilevel) {
        pcomm->ProlongateAndApplyPhysicalBCs(
            grp, time, 0.0, cil, ciu, cjl, cju, ckl, cku, cng);
      }

      // Fine-level physical BCs for every AuxADM channel.
      pcomm->ApplyPhysicalBCs(grp, time, 0.0);
    }

  }
}

void Mesh::CommunicateIteratedZ4c(const int iterations)
{

#if defined(Z4C_CX_ENABLED)

  if (iterations > 0)
  {
    int nthreads = GetNumMeshThreads();
    (void)nthreads;
    std::vector<MeshBlock*> pmb_array;
    GetMeshBlocksMyRank(pmb_array);
    const int nmb = pmb_array.size();

    // Communication uses the Iterated group (z4c_rbc channel with
    // RestrictOp::LagrangeFull).  Prolongation and physical BCs use the Z4c
    // group channels (z4c_u) because the RBC channel shares storage.u /
    // coarse_u_ with the main Z4c channel.
    const auto comm_grp = comm::CommGroup::Iterated;

    for (int iter=0; iter<iterations; ++iter)
    {
      #pragma omp parallel num_threads(nthreads)
      {
        MeshBlock *pmb = nullptr;

        #pragma omp for private(pmb)
        for (int i=0; i<nmb; ++i)
        {
          pmb = pmb_array[i];
          pmb->pcomm->StartReceiving(comm_grp);
        }

        // SendBoundaryBuffers on Iterated group automatically applies
        // LagrangeFull restriction (embedded in CommRegistry::SendBoundaryBuffers).
        #pragma omp for private(pmb)
        for (int i=0; i<nmb; ++i)
        {
          pmb = pmb_array[i];
          pmb->pcomm->SendBoundaryBuffers(comm_grp);
        }

        #pragma omp for private(pmb)
        for (int i=0; i<nmb; ++i)
        {
          pmb = pmb_array[i];
          while (!pmb->pcomm->ReceiveBoundaryBuffers(comm_grp)) {}
          pmb->pcomm->SetBoundaries(comm_grp);
          pmb->pcomm->ClearBoundary(comm_grp);
        }

        // Prolongation and physical BCs use Z4c group channels.
        #pragma omp for private(pmb)
        for (int i=0; i<nmb; ++i)
        {
          pmb = pmb_array[i];
          Z4c *pz = pmb->pz4c;

          comm::CommRegistry *pcomm = pmb->pcomm;

          const int cil = pz->mbi.cil, ciu = pz->mbi.ciu;
          const int cjl = pz->mbi.cjl, cju = pz->mbi.cju;
          const int ckl = pz->mbi.ckl, cku = pz->mbi.cku;
          const int cng = pz->mbi.cng;

          if (multilevel) {
            pcomm->ProlongateAndApplyPhysicalBCs(
                comm::CommGroup::Z4c, time, 0.0,
                cil, ciu, cjl, cju, ckl, cku, cng);
          }

          pcomm->ApplyPhysicalBCs(comm::CommGroup::Z4c, time, 0.0);

        }
      }
    }
  }
#endif // Z4C_CX_ENABLED
}


// Compute global minima of various quantities
void Mesh::GlobalExtrema()
{
  enum vars_min {
    IX_min_adm_alpha,
    IX_min_hydro_cons_D,
    N_vars_min
  };

  enum vars_max {
    IX_max_adm_alpha=N_vars_min,
    IX_max_hydro_cons_D,
    N_vars
  };

  AA res_V;
  res_V.NewAthenaArray(N_vars);
  res_V.Fill(std::numeric_limits<Real>::infinity());

  int nthreads = GetNumMeshThreads();
  (void)nthreads;

  std::vector<MeshBlock*> pmb_array;
  GetMeshBlocksMyRank(pmb_array);
  const int nmb = pmb_array.size();

  // minima-per-MeshBlock and then reduce
  AA res_V_mb;
  res_V_mb.NewAthenaArray(N_vars, nmb);
  res_V_mb.Fill(std::numeric_limits<Real>::infinity());

  // Per-MeshBlock parallism
  #pragma omp parallel for num_threads(nthreads)
  for (int nix = 0; nix < nmb; ++nix)
  {
    MeshBlock *pmb = pmb_array[nix];

#if Z4C_ENABLED
    Z4c * pz4c = pmb->pz4c;

    AA adm_alpha;
    adm_alpha.InitWithShallowSlice(pz4c->storage.adm, Z4c::I_ADM_alpha, 1);
#endif // Z4C_ENABLED

#if FLUID_ENABLED && Z4C_ENABLED
    Hydro * ph = pmb->phydro;

    AA ms_adm_sqrt_det_gamma;
    ms_adm_sqrt_det_gamma.InitWithShallowSlice(
      pz4c->storage.aux_extended, Z4c::I_AUX_EXTENDED_ms_sqrt_detgamma, 1
    );

    AA hydro_cons_D;
    hydro_cons_D.InitWithShallowSlice(
      ph->u, IDN, 1
    );
#endif // FLUID_ENABLED

    // Thread-local accumulators to avoid false sharing on res_V_mb
    Real loc_min_alpha     = std::numeric_limits<Real>::infinity();
    Real loc_max_neg_alpha = std::numeric_limits<Real>::infinity();
    Real loc_min_D         = std::numeric_limits<Real>::infinity();
    Real loc_max_neg_D     = std::numeric_limits<Real>::infinity();

#if Z4C_ENABLED
    ILOOP2(k, j)
    for (int i=pz4c->mbi.il; i<=pz4c->mbi.iu; ++i)
    {
      loc_min_alpha     = std::min(loc_min_alpha,     adm_alpha(k,j,i));
      loc_max_neg_alpha = std::min(loc_max_neg_alpha, -adm_alpha(k,j,i));
    }
#endif // Z4C_ENABLED

#if FLUID_ENABLED && Z4C_ENABLED
    CC_ILOOP2(k, j)
    for (int i=pmb->is; i<=pmb->ie; ++i)
    {
      const Real oo_sqrt_detgamma = OO(
        ms_adm_sqrt_det_gamma(k,j,i)
      );
      const Real hydro_cons_D__ = hydro_cons_D(k,j,i) * oo_sqrt_detgamma;

      loc_min_D     = std::min(loc_min_D,     hydro_cons_D__);
      loc_max_neg_D = std::min(loc_max_neg_D, -hydro_cons_D__);
    }
#endif // FLUID_ENABLED && Z4C_ENABLED

    // Write accumulated results back once per MeshBlock
    res_V_mb(vars_min::IX_min_adm_alpha,    nix) = loc_min_alpha;
    res_V_mb(vars_max::IX_max_adm_alpha,    nix) = loc_max_neg_alpha;
    res_V_mb(vars_min::IX_min_hydro_cons_D, nix) = loc_min_D;
    res_V_mb(vars_max::IX_max_hydro_cons_D, nix) = loc_max_neg_D;

  }

  // reduce over MeshBlock
  for (int vix = 0; vix < N_vars; ++vix)
  for (int nix = 0; nix < nmb; ++nix)
  {
    res_V(vix) = std::min(
      res_V(vix), res_V_mb(vix, nix)
    );
  }

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, res_V.data(),
    N_vars,
    MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif

  // flip sign to get max for salient variables
  for (int vix=0; vix < N_vars-N_vars_min; ++vix)
  {
    res_V(vix+N_vars_min) = -res_V(vix+N_vars_min);
  }

  global_extrema.min_adm_alpha = res_V(vars_min::IX_min_adm_alpha);
  global_extrema.max_adm_alpha = res_V(vars_max::IX_max_adm_alpha);

  global_extrema.min_hydro_cons_D = res_V(vars_min::IX_min_hydro_cons_D);
  global_extrema.max_hydro_cons_D = res_V(vars_max::IX_max_hydro_cons_D);
}

//
// :D
//
