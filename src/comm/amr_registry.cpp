//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file amr_registry.cpp
//  \brief AMRRegistry implementation: registration, buffer-size computation,
//         pack / unpack / restrict / prolongate for AMR block redistribution.
//
//  This file extracts the data-handling logic that was previously embedded in
//  Mesh::PrepareSendSameLevel, PrepareSendCoarseToFineAMR,
//  PrepareSendFineToCoarseAMR, FillSameRankFineToCoarseAMR,
//  FillSameRankCoarseToFineAMR, FinishRecvSameLevel,
//  FinishRecvFineToCoarseAMR, FinishRecvCoarseToFineAMR in
//  amr_loadbalance.cpp.

// C headers
#include <algorithm>
#include <sstream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../defs.hpp"       // NGHOST, NCGHOST, NCGHOST_CX, f2/f3 via Mesh
#include "../mesh/mesh.hpp"  // MeshBlock, Mesh, LogicalLocation, FaceField
#include "../mesh/mesh_refinement.hpp"
#include "../utils/buffer_utils.hpp"
#include "amr_registry.hpp"
#include "amr_spec.hpp"
#include "comm_enums.hpp"

namespace comm
{

//----------------------------------------------------------------------------------------
// Constructor
//----------------------------------------------------------------------------------------

AMRRegistry::AMRRegistry(MeshBlock* pmb)
    : pmy_block_(pmb), total_same_level_size_(0), finalized_(false)
{
}

//----------------------------------------------------------------------------------------
// Registration
//----------------------------------------------------------------------------------------

int AMRRegistry::Register(const AMRSpec& spec)
{
  int id = static_cast<int>(all_specs_.size());
  all_specs_.push_back(spec);
  group_specs_[static_cast<int>(spec.group)].push_back(id);
  return id;
}

//----------------------------------------------------------------------------------------
// Finalize: compute buffer sizes, freeze registration
//----------------------------------------------------------------------------------------

void AMRRegistry::Finalize()
{
  for (int g = 0; g < static_cast<int>(AMRGroup::NumGroups); ++g)
    ComputeBufferSizes(static_cast<AMRGroup>(g));

  // Total same-level = sum of per-group same-level sizes.
  // Note: same-level packs ALL groups into one buffer (like the legacy code
  // packs all vars_cc_ + vars_cx_ + vars_vc_ + vars_fc_ together).
  total_same_level_size_ = 0;
  for (int g = 0; g < static_cast<int>(AMRGroup::NumGroups); ++g)
    total_same_level_size_ += buf_sizes_[g].same_level;

  // Note: the orchestrator (amr_loadbalance.cpp) is responsible for appending
  // +1 for deref_count_ when adaptive.  That datum is block metadata, not a
  // registered variable, so we do NOT account for it here.

  finalized_ = true;
}

//----------------------------------------------------------------------------------------
// ComputeBufferSizes - per-group buffer size calculation
//
// Mirrors the logic from amr_loadbalance.cpp Step 4 (L454-541), but computes
// sizes per AMRGroup rather than from the monolithic vars_cc_/pvars_cc_
// vectors.
//----------------------------------------------------------------------------------------

void AMRRegistry::ComputeBufferSizes(AMRGroup group)
{
  AMRBufferSizes& bs = buf_sizes_[static_cast<int>(group)];
  bs.same_level      = 0;
  bs.fine_to_coarse  = 0;
  bs.coarse_to_fine  = 0;

  const auto& spec_ids = group_specs_[static_cast<int>(group)];
  if (spec_ids.empty())
    return;

  MeshBlock* pb  = pmy_block_;
  Mesh* pm       = pb->pmy_mesh;
  const bool f2  = pm->f2;
  const bool f3  = pm->f3;
  const int bnx1 = pb->block_size.nx1;
  const int bnx2 = pb->block_size.nx2;
  const int bnx3 = pb->block_size.nx3;

  for (int sid : spec_ids)
  {
    const AMRSpec& s = all_specs_[sid];

    switch (s.sampling)
    {
      case Sampling::CC:
      {
        int nv = s.nvar;
        bs.same_level += bnx1 * bnx2 * bnx3 * nv;
        bs.fine_to_coarse +=
          (bnx1 / 2) * ((bnx2 + 1) / 2) * ((bnx3 + 1) / 2) * nv;
        bs.coarse_to_fine += (bnx1 / 2 + 2) * ((bnx2 + 1) / 2 + 2 * f2) *
                             ((bnx3 + 1) / 2 + 2 * f3) * nv;
        break;
      }
      case Sampling::CX:
      {
        int nv              = s.nvar;
        const int min_cx_ng = std::min(NCGHOST_CX, NGHOST);
        const int cx_hbnx1  = bnx1 / 2;
        const int cx_hbnx2  = (bnx2 > 1) ? bnx2 / 2 : 1;
        const int cx_hbnx3  = (bnx3 > 1) ? bnx3 / 2 : 1;
        const int cx_ndg1   = min_cx_ng;
        const int cx_ndg2   = (f2 > 0) ? min_cx_ng : 0;
        const int cx_ndg3   = (f3 > 0) ? min_cx_ng : 0;

        bs.same_level += bnx1 * bnx2 * bnx3 * nv;
        bs.fine_to_coarse += cx_hbnx1 * cx_hbnx2 * cx_hbnx3 * nv;
        bs.coarse_to_fine += nv * (cx_hbnx1 + 2 * cx_ndg1) *
                             (cx_hbnx2 + 2 * cx_ndg2) *
                             (cx_hbnx3 + 2 * cx_ndg3);
        break;
      }
      case Sampling::VC:
      {
        int nv              = s.nvar;
        const int vbnx1     = bnx1 + 1;
        const int vbnx2     = (bnx2 > 1) ? bnx2 + 1 : 1;
        const int vbnx3     = (bnx3 > 1) ? bnx3 + 1 : 1;
        const int hbnx1     = bnx1 / 2 + 1;
        const int hbnx2     = bnx2 / 2 + 1;
        const int hbnx3     = bnx3 / 2 + 1;
        const int min_vc_ng = std::min(NCGHOST, NGHOST);
        const int ndg1      = min_vc_ng;
        const int ndg2      = (f2 > 0) ? min_vc_ng : 0;
        const int ndg3      = (f3 > 0) ? min_vc_ng : 0;

        bs.same_level += vbnx1 * vbnx2 * vbnx3 * nv;
        bs.fine_to_coarse += hbnx1 * hbnx2 * hbnx3 * nv;
        bs.coarse_to_fine +=
          nv * (hbnx1 + 2 * ndg1) * (hbnx2 + 2 * ndg2) * (hbnx3 + 2 * ndg3);
        break;
      }
      case Sampling::FC:
      {
        // FC: always 3 face directions, nvar ignored
        bs.same_level += (bnx1 + 1) * bnx2 * bnx3 + bnx1 * (bnx2 + f2) * bnx3 +
                         bnx1 * bnx2 * (bnx3 + f3);
        bs.fine_to_coarse +=
          ((bnx1 / 2) + 1) * ((bnx2 + 1) / 2) * ((bnx3 + 1) / 2) +
          (bnx1 / 2) * (((bnx2 + 1) / 2) + f2) * ((bnx3 + 1) / 2) +
          (bnx1 / 2) * ((bnx2 + 1) / 2) * (((bnx3 + 1) / 2) + f3);
        bs.coarse_to_fine += ((bnx1 / 2) + 1 + 2) * ((bnx2 + 1) / 2 + 2 * f2) *
                               ((bnx3 + 1) / 2 + 2 * f3) +
                             (bnx1 / 2 + 2) *
                               (((bnx2 + 1) / 2) + f2 + 2 * f2) *
                               ((bnx3 + 1) / 2 + 2 * f3) +
                             (bnx1 / 2 + 2) * ((bnx2 + 1) / 2 + 2 * f2) *
                               (((bnx3 + 1) / 2) + f3 + 2 * f3);
        break;
      }
    }  // switch
  }
}

//========================================================================================
// Same-Level Pack / Unpack
//========================================================================================
// These pack/unpack ALL groups into one contiguous buffer, matching the legacy
// PrepareSendSameLevel / FinishRecvSameLevel behavior.  The ordering is:
//   for each group { CC specs, CX specs, VC specs, FC specs }
// Note: deref_count_ is NOT included - the orchestrator appends/reads it
// separately.

int AMRRegistry::PackSameLevel(MeshBlock* pb, Real* sendbuf) const
{
  int p = 0;
  for (int g = 0; g < static_cast<int>(AMRGroup::NumGroups); ++g)
  {
    const auto& spec_ids = group_specs_[g];
    if (spec_ids.empty())
      continue;
    // Pack in sampling order: CC, CX, VC, FC (matches legacy ordering)
    p = PackCCSameLevel(pb, sendbuf, p, spec_ids);
    p = PackCXSameLevel(pb, sendbuf, p, spec_ids);
    p = PackVCSameLevel(pb, sendbuf, p, spec_ids);
    p = PackFCSameLevel(pb, sendbuf, p, spec_ids);
  }

  // Note: deref_count_ is NOT packed here - the orchestrator appends it
  // separately after calling PackSameLevel(), because deref_count_ is block
  // metadata (private to MeshRefinement) rather than a registered variable.
  return p;
}

void AMRRegistry::UnpackSameLevel(MeshBlock* pb, Real* recvbuf) const
{
  int p = 0;
  for (int g = 0; g < static_cast<int>(AMRGroup::NumGroups); ++g)
  {
    const auto& spec_ids = group_specs_[g];
    if (spec_ids.empty())
      continue;
    p = UnpackCCSameLevel(pb, recvbuf, p, spec_ids);
    p = UnpackCXSameLevel(pb, recvbuf, p, spec_ids);
    p = UnpackVCSameLevel(pb, recvbuf, p, spec_ids);
    p = UnpackFCSameLevel(pb, recvbuf, p, spec_ids);
  }

  // Note: deref_count_ is NOT unpacked here - the orchestrator reads it
  // separately after calling UnpackSameLevel(), because deref_count_ is block
  // metadata (private to MeshRefinement) rather than a registered variable.
}

//========================================================================================
// Fine-to-Coarse Pack / Unpack
//========================================================================================

int AMRRegistry::PackFineToCoarse(MeshBlock* pb,
                                  Real* sendbuf,
                                  AMRGroup group) const
{
  MeshRefinement* pmr  = pb->pmr;
  const auto& spec_ids = group_specs_[static_cast<int>(group)];
  int p                = 0;

  for (int sid : spec_ids)
  {
    const AMRSpec& s = all_specs_[sid];

    // FC is handled separately (FaceField layout, always AreaWeightedFace)
    if (s.sampling == Sampling::FC)
    {
      const bool f2 = pb->pmy_mesh->f2;
      const bool f3 = pb->pmy_mesh->f3;
      pmr->RestrictFieldX1(s.face_var->x1f,
                           s.coarse_face_var->x1f,
                           pb->cis,
                           pb->cie + 1,
                           pb->cjs,
                           pb->cje,
                           pb->cks,
                           pb->cke);
      BufferUtility::PackData(s.coarse_face_var->x1f,
                              sendbuf,
                              pb->cis,
                              pb->cie + 1,
                              pb->cjs,
                              pb->cje,
                              pb->cks,
                              pb->cke,
                              p);
      pmr->RestrictFieldX2(s.face_var->x2f,
                           s.coarse_face_var->x2f,
                           pb->cis,
                           pb->cie,
                           pb->cjs,
                           pb->cje + f2,
                           pb->cks,
                           pb->cke);
      BufferUtility::PackData(s.coarse_face_var->x2f,
                              sendbuf,
                              pb->cis,
                              pb->cie,
                              pb->cjs,
                              pb->cje + f2,
                              pb->cks,
                              pb->cke,
                              p);
      pmr->RestrictFieldX3(s.face_var->x3f,
                           s.coarse_face_var->x3f,
                           pb->cis,
                           pb->cie,
                           pb->cjs,
                           pb->cje,
                           pb->cks,
                           pb->cke + f3);
      BufferUtility::PackData(s.coarse_face_var->x3f,
                              sendbuf,
                              pb->cis,
                              pb->cie,
                              pb->cjs,
                              pb->cje,
                              pb->cks,
                              pb->cke + f3,
                              p);
      continue;
    }

    // Non-FC: skip if no restriction requested
    if (s.restrict_op == RestrictOp::None)
      continue;

    int nu = s.nvar - 1;

    // Determine coarse index bounds from sampling
    int cil, ciu, cjl, cju, ckl, cku;
    switch (s.sampling)
    {
      case Sampling::CC:
        cil = pb->cis;
        ciu = pb->cie;
        cjl = pb->cjs;
        cju = pb->cje;
        ckl = pb->cks;
        cku = pb->cke;
        break;
      case Sampling::CX:
        cil = pb->cx_cis;
        ciu = pb->cx_cie;
        cjl = pb->cx_cjs;
        cju = pb->cx_cje;
        ckl = pb->cx_cks;
        cku = pb->cx_cke;
        break;
      case Sampling::VC:
        cil = pb->civs;
        ciu = pb->cive;
        cjl = pb->cjvs;
        cju = pb->cjve;
        ckl = pb->ckvs;
        cku = pb->ckve;
        break;
      default:
        continue;  // unreachable (FC handled above)
    }

    // Restrict using the registered operator
    switch (s.restrict_op)
    {
      case RestrictOp::VolumeWeighted:
        pmr->RestrictCellCenteredValues(
          *s.var, *s.coarse_var, 0, nu, cil, ciu, cjl, cju, ckl, cku);
        break;
      case RestrictOp::Injection:
        pmr->RestrictVertexCenteredValues(
          *s.var, *s.coarse_var, 0, nu, cil, ciu, cjl, cju, ckl, cku);
        break;
      case RestrictOp::LagrangeUniform:
      case RestrictOp::Barycentric:
        pmr->RestrictCellCenteredXWithInteriorValues(
          *s.var, *s.coarse_var, 0, nu);
        break;
      case RestrictOp::LagrangeFull:
        pmr->RestrictCellCenteredX<NGHOST>(
          *s.var, *s.coarse_var, 0, nu, cil, ciu, cjl, cju, ckl, cku);
        break;
      default:
        break;
    }

    // Pack the restricted coarse data
    BufferUtility::PackData(
      *s.coarse_var, sendbuf, 0, nu, cil, ciu, cjl, cju, ckl, cku, p);
  }
  return p;
}

void AMRRegistry::UnpackFineToCoarse(MeshBlock* pb,
                                     Real* recvbuf,
                                     LogicalLocation& lloc,
                                     AMRGroup group) const
{
  const auto& spec_ids = group_specs_[static_cast<int>(group)];
  const int b_hsz1     = pb->block_size.nx1 / 2;
  const int b_hsz2     = pb->block_size.nx2 / 2;
  const int b_hsz3     = pb->block_size.nx3 / 2;
  const int ox1        = ((lloc.lx1 & 1LL) == 1LL);
  const int ox2        = ((lloc.lx2 & 1LL) == 1LL);
  const int ox3        = ((lloc.lx3 & 1LL) == 1LL);
  const bool f2        = pb->pmy_mesh->f2;
  const bool f3        = pb->pmy_mesh->f3;

  int p = 0;

  for (int sid : spec_ids)
  {
    const AMRSpec& s = all_specs_[sid];
    switch (s.sampling)
    {
      case Sampling::CC:
      {
        int il, iu, jl, ju, kl, ku;
        il     = (ox1 == 0) ? pb->is : pb->is + b_hsz1;
        iu     = (ox1 == 0) ? pb->is + b_hsz1 - 1 : pb->ie;
        jl     = (ox2 == 0) ? pb->js : pb->js + b_hsz2;
        ju     = (ox2 == 0) ? pb->js + b_hsz2 - f2 : pb->je;
        kl     = (ox3 == 0) ? pb->ks : pb->ks + b_hsz3;
        ku     = (ox3 == 0) ? pb->ks + b_hsz3 - f3 : pb->ke;
        int nu = s.nvar - 1;
        BufferUtility::UnpackData(
          recvbuf, *s.var, 0, nu, il, iu, jl, ju, kl, ku, p);
        break;
      }
      case Sampling::CX:
      {
        int cx_il, cx_iu, cx_jl, cx_ju, cx_kl, cx_ku;
        cx_il  = (ox1 == 0) ? pb->cx_is : pb->cx_is + b_hsz1;
        cx_iu  = (ox1 == 0) ? pb->cx_is + b_hsz1 - 1 : pb->cx_ie;
        cx_jl  = (ox2 == 0) ? pb->cx_js : pb->cx_js + b_hsz2;
        cx_ju  = (ox2 == 0) ? pb->cx_js + b_hsz2 - f2 : pb->cx_je;
        cx_kl  = (ox3 == 0) ? pb->cx_ks : pb->cx_ks + b_hsz3;
        cx_ku  = (ox3 == 0) ? pb->cx_ks + b_hsz3 - f3 : pb->cx_ke;
        int nu = s.nvar - 1;
        BufferUtility::UnpackData(
          recvbuf, *s.var, 0, nu, cx_il, cx_iu, cx_jl, cx_ju, cx_kl, cx_ku, p);
        break;
      }
      case Sampling::VC:
      {
        int vc_il, vc_iu, vc_jl, vc_ju, vc_kl, vc_ku;
        if (ox1 == 0)
        {
          vc_il = pb->ivs;
          vc_iu = pb->ivs + b_hsz1;
        }
        else
        {
          vc_il = pb->ivs + b_hsz1;
          vc_iu = pb->ive;
        }
        if (ox2 == 0)
        {
          vc_jl = pb->jvs;
          vc_ju = pb->jvs + b_hsz2;
        }
        else
        {
          vc_jl = pb->jvs + b_hsz2;
          vc_ju = pb->jve;
        }
        if (ox3 == 0)
        {
          vc_kl = pb->kvs;
          vc_ku = pb->kvs + b_hsz3;
        }
        else
        {
          vc_kl = pb->kvs + b_hsz3;
          vc_ku = pb->kve;
        }
        int nu = s.nvar - 1;
        BufferUtility::UnpackData(
          recvbuf, *s.var, 0, nu, vc_il, vc_iu, vc_jl, vc_ju, vc_kl, vc_ku, p);
        break;
      }
      case Sampling::FC:
      {
        int il, iu, jl, ju, kl, ku;
        il = (ox1 == 0) ? pb->is : pb->is + b_hsz1;
        iu = (ox1 == 0) ? pb->is + b_hsz1 - 1 : pb->ie;
        jl = (ox2 == 0) ? pb->js : pb->js + b_hsz2;
        ju = (ox2 == 0) ? pb->js + b_hsz2 - f2 : pb->je;
        kl = (ox3 == 0) ? pb->ks : pb->ks + b_hsz3;
        ku = (ox3 == 0) ? pb->ks + b_hsz3 - f3 : pb->ke;
        BufferUtility::UnpackData(
          recvbuf, s.face_var->x1f, il, iu + 1, jl, ju, kl, ku, p);
        BufferUtility::UnpackData(
          recvbuf, s.face_var->x2f, il, iu, jl, ju + f2, kl, ku, p);
        BufferUtility::UnpackData(
          recvbuf, s.face_var->x3f, il, iu, jl, ju, kl, ku + f3, p);
        if (pb->block_size.nx2 == 1)
        {
          for (int i = il; i <= iu; i++)
            s.face_var->x2f(pb->ks, pb->js + 1, i) =
              s.face_var->x2f(pb->ks, pb->js, i);
        }
        if (pb->block_size.nx3 == 1)
        {
          for (int j = jl; j <= ju; j++)
            for (int i = il; i <= iu; i++)
              s.face_var->x3f(pb->ks + 1, j, i) =
                s.face_var->x3f(pb->ks, j, i);
        }
        break;
      }
    }  // switch
  }
}

//========================================================================================
// Coarse-to-Fine Pack / Unpack
//========================================================================================

int AMRRegistry::PackCoarseToFine(MeshBlock* pb,
                                  Real* sendbuf,
                                  LogicalLocation& lloc,
                                  AMRGroup group) const
{
  const auto& spec_ids = group_specs_[static_cast<int>(group)];
  const int b_hsz1     = pb->block_size.nx1 / 2;
  const int b_hsz2     = pb->block_size.nx2 / 2;
  const int b_hsz3     = pb->block_size.nx3 / 2;
  const int ox1        = ((lloc.lx1 & 1LL) == 1LL);
  const int ox2        = ((lloc.lx2 & 1LL) == 1LL);
  const int ox3        = ((lloc.lx3 & 1LL) == 1LL);
  const bool f2        = pb->pmy_mesh->f2;
  const bool f3        = pb->pmy_mesh->f3;

  int p = 0;

  for (int sid : spec_ids)
  {
    const AMRSpec& s = all_specs_[sid];
    switch (s.sampling)
    {
      case Sampling::CC:
      {
        int il, iu, jl, ju, kl, ku;
        il     = (ox1 == 0) ? pb->is - 1 : pb->is + b_hsz1 - 1;
        iu     = (ox1 == 0) ? pb->is + b_hsz1 : pb->ie + 1;
        jl     = (ox2 == 0) ? pb->js - f2 : pb->js + b_hsz2 - f2;
        ju     = (ox2 == 0) ? pb->js + b_hsz2 : pb->je + f2;
        kl     = (ox3 == 0) ? pb->ks - f3 : pb->ks + b_hsz3 - f3;
        ku     = (ox3 == 0) ? pb->ks + b_hsz3 : pb->ke + f3;
        int nu = s.nvar - 1;
        BufferUtility::PackData(
          *s.var, sendbuf, 0, nu, il, iu, jl, ju, kl, ku, p);
        break;
      }
      case Sampling::CX:
      {
        const int min_cx_ng = std::min(NCGHOST_CX, NGHOST);
        int cx_il, cx_iu, cx_jl, cx_ju, cx_kl, cx_ku;
        cx_il  = (ox1 == 0) ? pb->cx_is - min_cx_ng
                            : pb->cx_is + b_hsz1 - 1 - (min_cx_ng - 1);
        cx_iu  = (ox1 == 0) ? pb->cx_is + b_hsz1 + (min_cx_ng - 1)
                            : pb->cx_ie + min_cx_ng;
        cx_jl  = (ox2 == 0) ? pb->cx_js - f2 * min_cx_ng
                            : pb->cx_js + b_hsz2 - f2 * min_cx_ng;
        cx_ju  = (ox2 == 0) ? pb->cx_js + b_hsz2 + f2 * (min_cx_ng - 1)
                            : pb->cx_je + f2 * min_cx_ng;
        cx_kl  = (ox3 == 0) ? pb->cx_ks - f3 * min_cx_ng
                            : pb->cx_ks + b_hsz3 - f3 * min_cx_ng;
        cx_ku  = (ox3 == 0) ? pb->cx_ks + b_hsz3 + f3 * (min_cx_ng - 1)
                            : pb->cx_ke + f3 * min_cx_ng;
        int nu = s.nvar - 1;
        BufferUtility::PackData(
          *s.var, sendbuf, 0, nu, cx_il, cx_iu, cx_jl, cx_ju, cx_kl, cx_ku, p);
        break;
      }
      case Sampling::VC:
      {
        const int min_vc_ng = std::min(NCGHOST, NGHOST);
        const int ndg1      = min_vc_ng;
        const int ndg2      = (f2 > 0) ? min_vc_ng : 0;
        const int ndg3      = (f3 > 0) ? min_vc_ng : 0;
        int vc_il, vc_iu, vc_jl, vc_ju, vc_kl, vc_ku;
        if (ox1 == 0)
        {
          vc_il = pb->ivs - ndg1;
          vc_iu = vc_il + (pb->cive + ndg1 - (pb->civs - ndg1));
        }
        else
        {
          vc_il = b_hsz1 + pb->ivs - ndg1;
          vc_iu = vc_il + (pb->cive + ndg1 - (pb->civs - ndg1));
        }
        if (ox2 == 0)
        {
          vc_jl = pb->jvs - ndg2;
          vc_ju = vc_jl + (pb->cjve + ndg2 - (pb->cjvs - ndg2));
        }
        else
        {
          vc_jl = b_hsz2 + pb->jvs - ndg2;
          vc_ju = vc_jl + (pb->cjve + ndg2 - (pb->cjvs - ndg2));
        }
        if (ox3 == 0)
        {
          vc_kl = pb->kvs - ndg3;
          vc_ku = vc_kl + (pb->ckve + ndg3 - (pb->ckvs - ndg3));
        }
        else
        {
          vc_kl = b_hsz3 + pb->kvs - ndg3;
          vc_ku = vc_kl + (pb->ckve + ndg3 - (pb->ckvs - ndg3));
        }
        int nu = s.nvar - 1;
        BufferUtility::PackData(
          *s.var, sendbuf, 0, nu, vc_il, vc_iu, vc_jl, vc_ju, vc_kl, vc_ku, p);
        break;
      }
      case Sampling::FC:
      {
        int il, iu, jl, ju, kl, ku;
        il = (ox1 == 0) ? pb->is - 1 : pb->is + b_hsz1 - 1;
        iu = (ox1 == 0) ? pb->is + b_hsz1 : pb->ie + 1;
        jl = (ox2 == 0) ? pb->js - f2 : pb->js + b_hsz2 - f2;
        ju = (ox2 == 0) ? pb->js + b_hsz2 : pb->je + f2;
        kl = (ox3 == 0) ? pb->ks - f3 : pb->ks + b_hsz3 - f3;
        ku = (ox3 == 0) ? pb->ks + b_hsz3 : pb->ke + f3;
        BufferUtility::PackData(
          s.face_var->x1f, sendbuf, il, iu + 1, jl, ju, kl, ku, p);
        BufferUtility::PackData(
          s.face_var->x2f, sendbuf, il, iu, jl, ju + f2, kl, ku, p);
        BufferUtility::PackData(
          s.face_var->x3f, sendbuf, il, iu, jl, ju, kl, ku + f3, p);
        break;
      }
    }  // switch
  }
  return p;
}

void AMRRegistry::UnpackCoarseToFine(MeshBlock* pb,
                                     Real* recvbuf,
                                     AMRGroup group) const
{
  MeshRefinement* pmr  = pb->pmr;
  const auto& spec_ids = group_specs_[static_cast<int>(group)];
  const bool f2        = pb->pmy_mesh->f2;
  const bool f3        = pb->pmy_mesh->f3;
  int p                = 0;

  // CC coarse bounds with halo
  const int cc_il = pb->cis - 1;
  const int cc_iu = pb->cie + 1;
  const int cc_jl = pb->cjs - f2;
  const int cc_ju = pb->cje + f2;
  const int cc_kl = pb->cks - f3;
  const int cc_ku = pb->cke + f3;

  // CX coarse bounds with halo
  const int min_cx_ng = std::min(NCGHOST_CX, NGHOST);
  const int cx_il     = pb->cx_cis - min_cx_ng;
  const int cx_iu     = pb->cx_cie + min_cx_ng;
  const int cx_jl     = pb->cx_cjs - f2 * min_cx_ng;
  const int cx_ju     = pb->cx_cje + f2 * min_cx_ng;
  const int cx_kl     = pb->cx_cks - f3 * min_cx_ng;
  const int cx_ku     = pb->cx_cke + f3 * min_cx_ng;

  // VC coarse bounds with halo
  const int min_vc_ng = std::min(NCGHOST, NGHOST);
  const int ndg1      = min_vc_ng;
  const int ndg2      = (f2 > 0) ? min_vc_ng : 0;
  const int ndg3      = (f3 > 0) ? min_vc_ng : 0;
  const int vc_il     = pb->civs - ndg1;
  const int vc_iu     = pb->cive + ndg1;
  const int vc_jl     = pb->cjvs - ndg2;
  const int vc_ju     = pb->cjve + ndg2;
  const int vc_kl     = pb->ckvs - ndg3;
  const int vc_ku     = pb->ckve + ndg3;

  for (int sid : spec_ids)
  {
    const AMRSpec& s = all_specs_[sid];

    // FC is handled separately (FaceField layout, always
    // FaceSharedMinmod+DivPreserving)
    if (s.sampling == Sampling::FC)
    {
      BufferUtility::UnpackData(recvbuf,
                                s.coarse_face_var->x1f,
                                cc_il,
                                cc_iu + 1,
                                cc_jl,
                                cc_ju,
                                cc_kl,
                                cc_ku,
                                p);
      BufferUtility::UnpackData(recvbuf,
                                s.coarse_face_var->x2f,
                                cc_il,
                                cc_iu,
                                cc_jl,
                                cc_ju + f2,
                                cc_kl,
                                cc_ku,
                                p);
      BufferUtility::UnpackData(recvbuf,
                                s.coarse_face_var->x3f,
                                cc_il,
                                cc_iu,
                                cc_jl,
                                cc_ju,
                                cc_kl,
                                cc_ku + f3,
                                p);
      pmr->ProlongateSharedFieldX1(s.coarse_face_var->x1f,
                                   s.face_var->x1f,
                                   pb->cis,
                                   pb->cie + 1,
                                   pb->cjs,
                                   pb->cje,
                                   pb->cks,
                                   pb->cke);
      pmr->ProlongateSharedFieldX2(s.coarse_face_var->x2f,
                                   s.face_var->x2f,
                                   pb->cis,
                                   pb->cie,
                                   pb->cjs,
                                   pb->cje + f2,
                                   pb->cks,
                                   pb->cke);
      pmr->ProlongateSharedFieldX3(s.coarse_face_var->x3f,
                                   s.face_var->x3f,
                                   pb->cis,
                                   pb->cie,
                                   pb->cjs,
                                   pb->cje,
                                   pb->cks,
                                   pb->cke + f3);
      pmr->ProlongateInternalField(
        *s.face_var, pb->cis, pb->cie, pb->cjs, pb->cje, pb->cks, pb->cke);
      continue;
    }

    // Non-FC: skip if no prolongation requested
    if (s.prolong_op == ProlongOp::None)
      continue;

    int nu = s.nvar - 1;

    // Determine coarse index bounds (with halo) from sampling, for unpacking
    int uil, uiu, ujl, uju, ukl, uku;
    // Determine coarse interior bounds from sampling, for prolongation
    int pil, piu, pjl, pju, pkl, pku;
    switch (s.sampling)
    {
      case Sampling::CC:
        uil = cc_il;
        uiu = cc_iu;
        ujl = cc_jl;
        uju = cc_ju;
        ukl = cc_kl;
        uku = cc_ku;
        pil = pb->cis;
        piu = pb->cie;
        pjl = pb->cjs;
        pju = pb->cje;
        pkl = pb->cks;
        pku = pb->cke;
        break;
      case Sampling::CX:
        uil = cx_il;
        uiu = cx_iu;
        ujl = cx_jl;
        uju = cx_ju;
        ukl = cx_kl;
        uku = cx_ku;
        pil = pb->cx_cis;
        piu = pb->cx_cie;
        pjl = pb->cx_cjs;
        pju = pb->cx_cje;
        pkl = pb->cx_cks;
        pku = pb->cx_cke;
        break;
      case Sampling::VC:
        uil = vc_il;
        uiu = vc_iu;
        ujl = vc_jl;
        uju = vc_ju;
        ukl = vc_kl;
        uku = vc_ku;
        pil = pb->civs;
        piu = pb->cive;
        pjl = pb->cjvs;
        pju = pb->cjve;
        pkl = pb->ckvs;
        pku = pb->ckve;
        break;
      default:
        continue;  // unreachable (FC handled above)
    }

    // Unpack coarse data into the coarse buffer
    BufferUtility::UnpackData(
      recvbuf, *s.coarse_var, 0, nu, uil, uiu, ujl, uju, ukl, uku, p);

    // Prolongate using the registered operator
    switch (s.prolong_op)
    {
      case ProlongOp::MinmodLinear:
        pmr->ProlongateCellCenteredValues(
          *s.coarse_var, *s.var, 0, nu, pil, piu, pjl, pju, pkl, pku);
        break;
      case ProlongOp::LagrangeUniform:
        pmr->ProlongateVertexCenteredValues(
          *s.coarse_var, *s.var, 0, nu, pil, piu, pjl, pju, pkl, pku);
        break;
      case ProlongOp::LagrangeChildren:
        pmr->ProlongateCellCenteredXValues(
          *s.coarse_var, *s.var, 0, nu, pil, piu, pjl, pju, pkl, pku);
        break;
      case ProlongOp::LagrangeChildrenBC:
        pmr->ProlongateCellCenteredXBCValues(
          *s.coarse_var, *s.var, 0, nu, pil, piu, pjl, pju, pkl, pku);
        break;
      default:
        break;
    }
  }
}

//========================================================================================
// Same-Rank Fill Operations (no MPI)
//========================================================================================

void AMRRegistry::FillSameRankFineToCoarse(MeshBlock* src_block,
                                           MeshBlock* dst_block,
                                           LogicalLocation& loc,
                                           AMRGroup group,
                                           AMRRegistry* dst_amr) const
{
  MeshRefinement* pmr = src_block->pmr;
  const int b_hsz1    = src_block->block_size.nx1 / 2;
  const int b_hsz2    = src_block->block_size.nx2 / 2;
  const int b_hsz3    = src_block->block_size.nx3 / 2;
  const bool llx1     = ((loc.lx1 & 1LL) == 1LL);
  const bool llx2     = ((loc.lx2 & 1LL) == 1LL);
  const bool llx3     = ((loc.lx3 & 1LL) == 1LL);
  const bool f2       = src_block->pmy_mesh->f2;
  const bool f3       = src_block->pmy_mesh->f3;

  // Get the dst_block's AMRRegistry specs for the same group to find target
  // arrays
  const auto& src_spec_ids = group_specs_[static_cast<int>(group)];
  const auto& dst_spec_ids = dst_amr->group_specs_[static_cast<int>(group)];

  for (int idx = 0; idx < static_cast<int>(src_spec_ids.size()); ++idx)
  {
    const AMRSpec& ss = all_specs_[src_spec_ids[idx]];
    const AMRSpec& ds = dst_amr->all_specs_[dst_spec_ids[idx]];

    // FC is handled separately (FaceField layout, always AreaWeightedFace)
    if (ss.sampling == Sampling::FC)
    {
      int il = dst_block->is + llx1 * b_hsz1;
      int jl = dst_block->js + llx2 * b_hsz2;
      int kl = dst_block->ks + llx3 * b_hsz3;
      pmr->RestrictFieldX1(ss.face_var->x1f,
                           ss.coarse_face_var->x1f,
                           src_block->cis,
                           src_block->cie + 1,
                           src_block->cjs,
                           src_block->cje,
                           src_block->cks,
                           src_block->cke);
      pmr->RestrictFieldX2(ss.face_var->x2f,
                           ss.coarse_face_var->x2f,
                           src_block->cis,
                           src_block->cie,
                           src_block->cjs,
                           src_block->cje + f2,
                           src_block->cks,
                           src_block->cke);
      pmr->RestrictFieldX3(ss.face_var->x3f,
                           ss.coarse_face_var->x3f,
                           src_block->cis,
                           src_block->cie,
                           src_block->cjs,
                           src_block->cje,
                           src_block->cks,
                           src_block->cke + f3);
      // Copy x1f
      for (int k = kl, fk = src_block->cks; fk <= src_block->cke; k++, fk++)
        for (int j = jl, fj = src_block->cjs; fj <= src_block->cje; j++, fj++)
          for (int i = il, fi = src_block->cis; fi <= src_block->cie + 1;
               i++, fi++)
            ds.face_var->x1f(k, j, i) = ss.coarse_face_var->x1f(fk, fj, fi);
      // Copy x2f
      for (int k = kl, fk = src_block->cks; fk <= src_block->cke; k++, fk++)
        for (int j = jl, fj = src_block->cjs; fj <= src_block->cje + f2;
             j++, fj++)
          for (int i = il, fi = src_block->cis; fi <= src_block->cie;
               i++, fi++)
            ds.face_var->x2f(k, j, i) = ss.coarse_face_var->x2f(fk, fj, fi);
      if (dst_block->block_size.nx2 == 1)
      {
        int iu = il + b_hsz1 - 1;
        for (int i = il; i <= iu; i++)
          ds.face_var->x2f(dst_block->ks, dst_block->js + 1, i) =
            ds.face_var->x2f(dst_block->ks, dst_block->js, i);
      }
      // Copy x3f
      for (int k = kl, fk = src_block->cks; fk <= src_block->cke + f3;
           k++, fk++)
        for (int j = jl, fj = src_block->cjs; fj <= src_block->cje; j++, fj++)
          for (int i = il, fi = src_block->cis; fi <= src_block->cie;
               i++, fi++)
            ds.face_var->x3f(k, j, i) = ss.coarse_face_var->x3f(fk, fj, fi);
      if (dst_block->block_size.nx3 == 1)
      {
        int iu = il + b_hsz1 - 1;
        int ju = jl + b_hsz2 - 1;
        if (dst_block->block_size.nx2 == 1)
          ju = jl;
        for (int j = jl; j <= ju; j++)
          for (int i = il; i <= iu; i++)
            ds.face_var->x3f(dst_block->ks + 1, j, i) =
              ds.face_var->x3f(dst_block->ks, j, i);
      }
      continue;
    }

    // Non-FC: skip if no restriction requested
    if (ss.restrict_op == RestrictOp::None)
      continue;

    int nu = ss.nvar - 1;

    // Determine coarse bounds on src_block and dst offset from sampling
    int cil, ciu, cjl, cju, ckl, cku;  // src coarse interior (restrict target)
    int dil, djl, dkl;                 // dst fine offset (copy target start)
    switch (ss.sampling)
    {
      case Sampling::CC:
        cil = src_block->cis;
        ciu = src_block->cie;
        cjl = src_block->cjs;
        cju = src_block->cje;
        ckl = src_block->cks;
        cku = src_block->cke;
        dil = dst_block->is + llx1 * b_hsz1;
        djl = dst_block->js + llx2 * b_hsz2;
        dkl = dst_block->ks + llx3 * b_hsz3;
        break;
      case Sampling::CX:
        cil = src_block->cx_cis;
        ciu = src_block->cx_cie;
        cjl = src_block->cx_cjs;
        cju = src_block->cx_cje;
        ckl = src_block->cx_cks;
        cku = src_block->cx_cke;
        dil = dst_block->cx_is + llx1 * b_hsz1;
        djl = dst_block->cx_js + llx2 * b_hsz2;
        dkl = dst_block->cx_ks + llx3 * b_hsz3;
        break;
      case Sampling::VC:
        cil = src_block->civs;
        ciu = src_block->cive;
        cjl = src_block->cjvs;
        cju = src_block->cjve;
        ckl = src_block->ckvs;
        cku = src_block->ckve;
        dil = dst_block->ivs + llx1 * b_hsz1;
        djl = dst_block->jvs + llx2 * b_hsz2;
        dkl = dst_block->kvs + llx3 * b_hsz3;
        break;
      default:
        continue;  // unreachable (FC handled above)
    }

    // CX NAN-fill before restriction (matches legacy behavior)
    if (ss.sampling == Sampling::CX)
      ss.coarse_var->Fill(NAN);

    // Restrict using the registered operator
    switch (ss.restrict_op)
    {
      case RestrictOp::VolumeWeighted:
        pmr->RestrictCellCenteredValues(
          *ss.var, *ss.coarse_var, 0, nu, cil, ciu, cjl, cju, ckl, cku);
        break;
      case RestrictOp::Injection:
        pmr->RestrictVertexCenteredValues(
          *ss.var, *ss.coarse_var, 0, nu, cil, ciu, cjl, cju, ckl, cku);
        break;
      case RestrictOp::LagrangeUniform:
      case RestrictOp::Barycentric:
        pmr->RestrictCellCenteredXWithInteriorValues(
          *ss.var, *ss.coarse_var, 0, nu);
        break;
      case RestrictOp::LagrangeFull:
        pmr->RestrictCellCenteredX<NGHOST>(
          *ss.var, *ss.coarse_var, 0, nu, cil, ciu, cjl, cju, ckl, cku);
        break;
      default:
        break;
    }

    // Copy restricted data from src coarse buffer into dst fine-level array
    AthenaArray<Real>& src_a = *ss.coarse_var;
    AthenaArray<Real>& dst_a = *ds.var;
    for (int nv = 0; nv <= nu; nv++)
      for (int k = dkl, fk = ckl; fk <= cku; k++, fk++)
        for (int j = djl, fj = cjl; fj <= cju; j++, fj++)
          for (int i = dil, fi = cil; fi <= ciu; i++, fi++)
            dst_a(nv, k, j, i) = src_a(nv, fk, fj, fi);
  }
}

void AMRRegistry::FillSameRankCoarseToFine(MeshBlock* src_block,
                                           MeshBlock* dst_block,
                                           LogicalLocation& newloc,
                                           AMRGroup group,
                                           AMRRegistry* src_amr,
                                           AMRRegistry* dst_amr) const
{
  MeshRefinement* pmr = dst_block->pmr;
  const int b_hsz1    = src_block->block_size.nx1 / 2;
  const int b_hsz2    = src_block->block_size.nx2 / 2;
  const int b_hsz3    = src_block->block_size.nx3 / 2;
  const bool nlx1     = ((newloc.lx1 & 1LL) == 1LL);
  const bool nlx2     = ((newloc.lx2 & 1LL) == 1LL);
  const bool nlx3     = ((newloc.lx3 & 1LL) == 1LL);
  const bool f2       = src_block->pmy_mesh->f2;
  const bool f3       = src_block->pmy_mesh->f3;

  const auto& src_spec_ids = src_amr->group_specs_[static_cast<int>(group)];
  const auto& dst_spec_ids = dst_amr->group_specs_[static_cast<int>(group)];

  for (int idx = 0; idx < static_cast<int>(src_spec_ids.size()); ++idx)
  {
    const AMRSpec& ss = src_amr->all_specs_[src_spec_ids[idx]];
    const AMRSpec& ds = dst_amr->all_specs_[dst_spec_ids[idx]];

    // FC is handled separately (FaceField layout, always
    // FaceSharedMinmod+DivPreserving)
    if (ss.sampling == Sampling::FC)
    {
      int il  = src_block->cis - 1;
      int iu  = src_block->cie + 1;
      int jl  = src_block->cjs - f2;
      int ju  = src_block->cje + f2;
      int kl  = src_block->cks - f3;
      int ku  = src_block->cke + f3;
      int cis = nlx1 * b_hsz1 + src_block->is - 1;
      int cjs = nlx2 * b_hsz2 + src_block->js - f2;
      int cks = nlx3 * b_hsz3 + src_block->ks - f3;

      // Copy src fine-level into dst coarse buffer
      for (int k = kl, ck = cks; k <= ku; k++, ck++)
        for (int j = jl, cj = cjs; j <= ju; j++, cj++)
          for (int i = il, ci = cis; i <= iu + 1; i++, ci++)
            ds.coarse_face_var->x1f(k, j, i) = ss.face_var->x1f(ck, cj, ci);
      for (int k = kl, ck = cks; k <= ku; k++, ck++)
        for (int j = jl, cj = cjs; j <= ju + f2; j++, cj++)
          for (int i = il, ci = cis; i <= iu; i++, ci++)
            ds.coarse_face_var->x2f(k, j, i) = ss.face_var->x2f(ck, cj, ci);
      for (int k = kl, ck = cks; k <= ku + f3; k++, ck++)
        for (int j = jl, cj = cjs; j <= ju; j++, cj++)
          for (int i = il, ci = cis; i <= iu; i++, ci++)
            ds.coarse_face_var->x3f(k, j, i) = ss.face_var->x3f(ck, cj, ci);

      // Prolongate on dst
      pmr->ProlongateSharedFieldX1(ds.coarse_face_var->x1f,
                                   ds.face_var->x1f,
                                   src_block->cis,
                                   src_block->cie + 1,
                                   src_block->cjs,
                                   src_block->cje,
                                   src_block->cks,
                                   src_block->cke);
      pmr->ProlongateSharedFieldX2(ds.coarse_face_var->x2f,
                                   ds.face_var->x2f,
                                   src_block->cis,
                                   src_block->cie,
                                   src_block->cjs,
                                   src_block->cje + f2,
                                   src_block->cks,
                                   src_block->cke);
      pmr->ProlongateSharedFieldX3(ds.coarse_face_var->x3f,
                                   ds.face_var->x3f,
                                   src_block->cis,
                                   src_block->cie,
                                   src_block->cjs,
                                   src_block->cje,
                                   src_block->cks,
                                   src_block->cke + f3);
      pmr->ProlongateInternalField(*ds.face_var,
                                   src_block->cis,
                                   src_block->cie,
                                   src_block->cjs,
                                   src_block->cje,
                                   src_block->cks,
                                   src_block->cke);
      continue;
    }

    // Non-FC: skip if no prolongation requested
    if (ds.prolong_op == ProlongOp::None)
      continue;

    int nu = ss.nvar - 1;

    // Determine coarse bounds (with halo) on dst, copy source offsets, and
    // prolong interior bounds - all from sampling
    int cil, ciu, cjl, cju, ckl,
      cku;              // dst coarse buffer bounds (copy target)
    int cis, cjs, cks;  // src fine-level copy start
    int pil, piu, pjl, pju, pkl, pku;  // prolong interior bounds (on dst)
    switch (ss.sampling)
    {
      case Sampling::CC:
        cil = src_block->cis - 1;
        ciu = src_block->cie + 1;
        cjl = src_block->cjs - f2;
        cju = src_block->cje + f2;
        ckl = src_block->cks - f3;
        cku = src_block->cke + f3;
        cis = nlx1 * b_hsz1 + src_block->is - 1;
        cjs = nlx2 * b_hsz2 + src_block->js - f2;
        cks = nlx3 * b_hsz3 + src_block->ks - f3;
        pil = src_block->cis;
        piu = src_block->cie;
        pjl = src_block->cjs;
        pju = src_block->cje;
        pkl = src_block->cks;
        pku = src_block->cke;
        break;
      case Sampling::CX:
      {
        const int min_cx_ng = std::min(NCGHOST_CX, NGHOST);
        cil                 = src_block->cx_cis - min_cx_ng;
        ciu                 = src_block->cx_cie + min_cx_ng;
        cjl                 = src_block->cx_cjs - f2 * min_cx_ng;
        cju                 = src_block->cx_cje + f2 * min_cx_ng;
        ckl                 = src_block->cx_cks - f3 * min_cx_ng;
        cku                 = src_block->cx_cke + f3 * min_cx_ng;
        cis                 = nlx1 * b_hsz1 + src_block->cx_is - 1 * min_cx_ng;
        cjs = nlx2 * b_hsz2 + src_block->cx_js - f2 * min_cx_ng;
        cks = nlx3 * b_hsz3 + src_block->cx_ks - f3 * min_cx_ng;
        pil = src_block->cx_cis;
        piu = src_block->cx_cie;
        pjl = src_block->cx_cjs;
        pju = src_block->cx_cje;
        pkl = src_block->cx_cks;
        pku = src_block->cx_cke;
        break;
      }
      case Sampling::VC:
      {
        const int min_vc_ng = std::min(NCGHOST, NGHOST);
        const int ndg1      = min_vc_ng;
        const int ndg2      = (f2 > 0) ? min_vc_ng : 0;
        const int ndg3      = (f3 > 0) ? min_vc_ng : 0;
        cil                 = src_block->civs - ndg1;
        ciu                 = src_block->cive + ndg1;
        cjl                 = src_block->cjvs - ndg2;
        cju                 = src_block->cjve + ndg2;
        ckl                 = src_block->ckvs - ndg3;
        cku                 = src_block->ckve + ndg3;
        cis                 = nlx1 * b_hsz1 + src_block->ivs - ndg1;
        cjs                 = nlx2 * b_hsz2 + src_block->jvs - ndg2;
        cks                 = nlx3 * b_hsz3 + src_block->kvs - ndg3;
        pil                 = src_block->civs;
        piu                 = src_block->cive;
        pjl                 = src_block->cjvs;
        pju                 = src_block->cjve;
        pkl                 = src_block->ckvs;
        pku                 = src_block->ckve;
        break;
      }
      default:
        continue;  // unreachable (FC handled above)
    }

    // Copy src fine-level data into dst coarse buffer
    AthenaArray<Real>& src_a = *ss.var;
    AthenaArray<Real>& dst_a = *ds.coarse_var;
    if (ss.sampling == Sampling::CX)
      dst_a.Fill(NAN);
    for (int nv = 0; nv <= nu; nv++)
      for (int k = ckl, ck = cks; k <= cku; k++, ck++)
        for (int j = cjl, cj = cjs; j <= cju; j++, cj++)
          for (int i = cil, ci = cis; i <= ciu; i++, ci++)
            dst_a(nv, k, j, i) = src_a(nv, ck, cj, ci);

    // Prolongate using the registered operator
    switch (ds.prolong_op)
    {
      case ProlongOp::MinmodLinear:
        pmr->ProlongateCellCenteredValues(
          dst_a, *ds.var, 0, nu, pil, piu, pjl, pju, pkl, pku);
        break;
      case ProlongOp::LagrangeUniform:
        pmr->ProlongateVertexCenteredValues(
          dst_a, *ds.var, 0, nu, pil, piu, pjl, pju, pkl, pku);
        break;
      case ProlongOp::LagrangeChildren:
        pmr->ProlongateCellCenteredXValues(
          dst_a, *ds.var, 0, nu, pil, piu, pjl, pju, pkl, pku);
        break;
      case ProlongOp::LagrangeChildrenBC:
        pmr->ProlongateCellCenteredXBCValues(
          dst_a, *ds.var, 0, nu, pil, piu, pjl, pju, pkl, pku);
        break;
      default:
        break;
    }
  }
}

//========================================================================================
// Same-Level Pack/Unpack Helpers (per sampling type)
//========================================================================================

int AMRRegistry::PackCCSameLevel(MeshBlock* pb,
                                 Real* buf,
                                 int p,
                                 const std::vector<int>& spec_ids) const
{
  for (int sid : spec_ids)
  {
    const AMRSpec& s = all_specs_[sid];
    if (s.sampling != Sampling::CC)
      continue;
    int nu = s.nvar - 1;
    BufferUtility::PackData(
      *s.var, buf, 0, nu, pb->is, pb->ie, pb->js, pb->je, pb->ks, pb->ke, p);
  }
  return p;
}

int AMRRegistry::PackCXSameLevel(MeshBlock* pb,
                                 Real* buf,
                                 int p,
                                 const std::vector<int>& spec_ids) const
{
  for (int sid : spec_ids)
  {
    const AMRSpec& s = all_specs_[sid];
    if (s.sampling != Sampling::CX)
      continue;
    int nu = s.nvar - 1;
    BufferUtility::PackData(*s.var,
                            buf,
                            0,
                            nu,
                            pb->cx_is,
                            pb->cx_ie,
                            pb->cx_js,
                            pb->cx_je,
                            pb->cx_ks,
                            pb->cx_ke,
                            p);
  }
  return p;
}

int AMRRegistry::PackVCSameLevel(MeshBlock* pb,
                                 Real* buf,
                                 int p,
                                 const std::vector<int>& spec_ids) const
{
  for (int sid : spec_ids)
  {
    const AMRSpec& s = all_specs_[sid];
    if (s.sampling != Sampling::VC)
      continue;
    int nu = s.nvar - 1;
    BufferUtility::PackData(*s.var,
                            buf,
                            0,
                            nu,
                            pb->ivs,
                            pb->ive,
                            pb->jvs,
                            pb->jve,
                            pb->kvs,
                            pb->kve,
                            p);
  }
  return p;
}

int AMRRegistry::PackFCSameLevel(MeshBlock* pb,
                                 Real* buf,
                                 int p,
                                 const std::vector<int>& spec_ids) const
{
  const bool f2 = pb->pmy_mesh->f2;
  const bool f3 = pb->pmy_mesh->f3;
  for (int sid : spec_ids)
  {
    const AMRSpec& s = all_specs_[sid];
    if (s.sampling != Sampling::FC)
      continue;
    BufferUtility::PackData(s.face_var->x1f,
                            buf,
                            pb->is,
                            pb->ie + 1,
                            pb->js,
                            pb->je,
                            pb->ks,
                            pb->ke,
                            p);
    BufferUtility::PackData(s.face_var->x2f,
                            buf,
                            pb->is,
                            pb->ie,
                            pb->js,
                            pb->je + f2,
                            pb->ks,
                            pb->ke,
                            p);
    BufferUtility::PackData(s.face_var->x3f,
                            buf,
                            pb->is,
                            pb->ie,
                            pb->js,
                            pb->je,
                            pb->ks,
                            pb->ke + f3,
                            p);
  }
  return p;
}

int AMRRegistry::UnpackCCSameLevel(MeshBlock* pb,
                                   Real* buf,
                                   int p,
                                   const std::vector<int>& spec_ids) const
{
  for (int sid : spec_ids)
  {
    const AMRSpec& s = all_specs_[sid];
    if (s.sampling != Sampling::CC)
      continue;
    int nu = s.nvar - 1;
    BufferUtility::UnpackData(
      buf, *s.var, 0, nu, pb->is, pb->ie, pb->js, pb->je, pb->ks, pb->ke, p);
  }
  return p;
}

int AMRRegistry::UnpackCXSameLevel(MeshBlock* pb,
                                   Real* buf,
                                   int p,
                                   const std::vector<int>& spec_ids) const
{
  for (int sid : spec_ids)
  {
    const AMRSpec& s = all_specs_[sid];
    if (s.sampling != Sampling::CX)
      continue;
    int nu = s.nvar - 1;
    BufferUtility::UnpackData(buf,
                              *s.var,
                              0,
                              nu,
                              pb->cx_is,
                              pb->cx_ie,
                              pb->cx_js,
                              pb->cx_je,
                              pb->cx_ks,
                              pb->cx_ke,
                              p);
  }
  return p;
}

int AMRRegistry::UnpackVCSameLevel(MeshBlock* pb,
                                   Real* buf,
                                   int p,
                                   const std::vector<int>& spec_ids) const
{
  for (int sid : spec_ids)
  {
    const AMRSpec& s = all_specs_[sid];
    if (s.sampling != Sampling::VC)
      continue;
    int nu = s.nvar - 1;
    BufferUtility::UnpackData(buf,
                              *s.var,
                              0,
                              nu,
                              pb->ivs,
                              pb->ive,
                              pb->jvs,
                              pb->jve,
                              pb->kvs,
                              pb->kve,
                              p);
  }
  return p;
}

int AMRRegistry::UnpackFCSameLevel(MeshBlock* pb,
                                   Real* buf,
                                   int p,
                                   const std::vector<int>& spec_ids) const
{
  const bool f2 = pb->pmy_mesh->f2;
  const bool f3 = pb->pmy_mesh->f3;
  for (int sid : spec_ids)
  {
    const AMRSpec& s = all_specs_[sid];
    if (s.sampling != Sampling::FC)
      continue;
    BufferUtility::UnpackData(buf,
                              s.face_var->x1f,
                              pb->is,
                              pb->ie + 1,
                              pb->js,
                              pb->je,
                              pb->ks,
                              pb->ke,
                              p);
    BufferUtility::UnpackData(buf,
                              s.face_var->x2f,
                              pb->is,
                              pb->ie,
                              pb->js,
                              pb->je + f2,
                              pb->ks,
                              pb->ke,
                              p);
    BufferUtility::UnpackData(buf,
                              s.face_var->x3f,
                              pb->is,
                              pb->ie,
                              pb->js,
                              pb->je,
                              pb->ks,
                              pb->ke + f3,
                              p);
    if (pb->block_size.nx2 == 1)
    {
      for (int i = pb->is; i <= pb->ie; i++)
        s.face_var->x2f(pb->ks, pb->js + 1, i) =
          s.face_var->x2f(pb->ks, pb->js, i);
    }
    if (pb->block_size.nx3 == 1)
    {
      for (int j = pb->js; j <= pb->je; j++)
        for (int i = pb->is; i <= pb->ie; i++)
          s.face_var->x3f(pb->ks + 1, j, i) = s.face_var->x3f(pb->ks, j, i);
    }
  }
  return p;
}

}  // namespace comm
