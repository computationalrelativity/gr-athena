//=======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals.cpp
//  \brief constructor/destructor and utility functions for BoundaryValues class

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>
#include <cstdlib>
#include <cstring>    // std::memcpy
#include <iomanip>
#include <iostream>   // endl
#include <iterator>
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <utility>    // swap()
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/mesh_refinement.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"
#include "../utils/buffer_utils.hpp"
#include "bvals.hpp"

// MPI header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// ----------------------------------------------------------------------------
namespace {

// In order to utilize slightly fewer lines
void AllClearBoundary_(BoundaryCommSubset phase,
                       std::vector<BoundaryVariable *> & bvar)
{
  for (auto bvars_it = bvar.begin();
      bvars_it != bvar.end();
      ++bvars_it)
  {
    (*bvars_it)->ClearBoundary(phase);
  }
}

void AllStartReceiving_(BoundaryCommSubset phase,
                        std::vector<BoundaryVariable *> & bvar)
{
  for (auto bvars_it = bvar.begin();
      bvars_it != bvar.end();
      ++bvars_it)
  {
    (*bvars_it)->StartReceiving(phase);
  }
}

}


// BoundaryValues constructor (the first object constructed inside the MeshBlock()
// constructor): sets functions for the appropriate boundary conditions at each of the 6
// dirs of a MeshBlock
BoundaryValues::BoundaryValues(MeshBlock *pmb, BoundaryFlag *input_bcs,
                               ParameterInput *pin)
    : BoundaryBase(pmb->pmy_mesh, pmb->loc, pmb->block_size, input_bcs), pmy_block_(pmb) {
  // Check BC functions for each of the 6 boundaries in turn ---------------------
  for (int i=0; i<6; i++) {
    switch (block_bcs[i]) {
    case BoundaryFlag::reflect:
    case BoundaryFlag::outflow:
    case BoundaryFlag::extrapolate_outflow:
    case BoundaryFlag::gr_sommerfeld:
    case BoundaryFlag::user:
    case BoundaryFlag::polar_wedge:
      apply_bndry_fn_[i] = true;
      break;
    default: // already initialized to false in class
      break;
    }
  }
  // Inner x1
  nface_ = 2; nedge_ = 0;
  CheckBoundaryFlag(block_bcs[BoundaryFace::inner_x1], CoordinateDirection::X1DIR);
  CheckBoundaryFlag(block_bcs[BoundaryFace::outer_x1], CoordinateDirection::X1DIR);

  if (pmb->block_size.nx2 > 1) {
    nface_ = 4; nedge_ = 4;
    CheckBoundaryFlag(block_bcs[BoundaryFace::inner_x2], CoordinateDirection::X2DIR);
    CheckBoundaryFlag(block_bcs[BoundaryFace::outer_x2], CoordinateDirection::X2DIR);
  }

  if (pmb->block_size.nx3 > 1) {
    nface_ = 6; nedge_ = 12;
    CheckBoundaryFlag(block_bcs[BoundaryFace::inner_x3], CoordinateDirection::X3DIR);
    CheckBoundaryFlag(block_bcs[BoundaryFace::outer_x3], CoordinateDirection::X3DIR);
  }
  // Perform compatibilty checks of user selections of polar vs. polar_wedge boundaries
  if (block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::polar
      || block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::polar
      || block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::polar_wedge
      || block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::polar_wedge) {
    CheckPolarBoundaries();
  }

  // polar boundary edge-case: single MeshBlock spans the entire azimuthal (x3) range
  if ((pmb->loc.level == pmy_mesh_->root_level && pmy_mesh_->nrbx3 == 1)
      && (block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::polar
       || block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::polar
       || block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::polar_wedge
       || block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::polar_wedge))
    azimuthal_shift_.NewAthenaArray(pmb->ke + NGHOST + 2);

  // prevent reallocation of contiguous memory space for each of 4x possible calls to
  // std::vector<BoundaryVariable *>.push_back() in Hydro, Field, Gravity, PassiveScalars
  bvars.reserve(3);
  // TOOD(KGF): rename to "bvars_time_int"? What about a std::vector for bvars_sts?
  bvars_main_int.reserve(2);
  bvars_main_int_vc.reserve(2);
  bvars_main_int_cx.reserve(2);

  bvars_aux.reserve(2);
  bvars_aux_adm.reserve(2);
  bvars_rbc.reserve(2);

  bvars_m1.reserve(1);

  // Matches initial value of Mesh::next_phys_id_
  // reserve phys=0 for former TAG_AMR=8; now hard-coded in Mesh::CreateAMRMPITag()
  bvars_next_phys_id_ = 1;
}

// destructor

BoundaryValues::~BoundaryValues() {
}

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValues::SetupPersistentMPI()
//  \brief Setup persistent MPI requests to be reused throughout the entire simulation

void BoundaryValues::SetupPersistentMPI() {
  for (auto bvars_it = bvars_main_int.begin(); bvars_it != bvars_main_int.end();
       ++bvars_it) {
    (*bvars_it)->SetupPersistentMPI();
  }

  for (auto bvars_it = bvars_main_int_vc.begin(); bvars_it != bvars_main_int_vc.end();
       ++bvars_it) {
    (*bvars_it)->SetupPersistentMPI();
  }

  for (auto bvars_it = bvars_main_int_cx.begin(); bvars_it != bvars_main_int_cx.end();
       ++bvars_it) {
    (*bvars_it)->SetupPersistentMPI();
  }

  for (auto bvars_it = bvars_aux.begin(); bvars_it != bvars_aux.end();
       ++bvars_it) {
    (*bvars_it)->SetupPersistentMPI();
  }

  for (auto bvars_it = bvars_aux_adm.begin(); bvars_it != bvars_aux_adm.end();
       ++bvars_it) {
    (*bvars_it)->SetupPersistentMPI();
  }

  for (auto bvars_it = bvars_rbc.begin(); bvars_it != bvars_rbc.end();
       ++bvars_it) {
    (*bvars_it)->SetupPersistentMPI();
  }

  for (auto bvars_it = bvars_m1.begin(); bvars_it != bvars_m1.end();
       ++bvars_it) {
    (*bvars_it)->SetupPersistentMPI();
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValues::CheckUserBoundaries()
//  \brief checks if the boundary functions are correctly enrolled (this compatibility
//  check is performed at the top of Mesh::Initialize(), after calling ProblemGenerator())

void BoundaryValues::CheckUserBoundaries() {
  for (int i=0; i<nface_; i++) {
    if (block_bcs[i] == BoundaryFlag::user) {
      if (pmy_mesh_->BoundaryFunction_[i] == nullptr) {
        std::stringstream msg;
        msg << "### FATAL ERROR in BoundaryValues::CheckBoundary" << std::endl
            << "A user-defined boundary is specified but the actual boundary function "
            << "is not enrolled in direction " << i  << " (in [0,6])." << std::endl;
        ATHENA_ERROR(msg);
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValues::StartReceiving(BoundaryCommSubset phase)
//  \brief initiate MPI_Irecv()

void BoundaryValues::StartReceiving(BoundaryCommSubset phase)
{
  // explicitly fill out the details for the different types of communication
  switch (phase)
  {
    case BoundaryCommSubset::aux_z4c:
    {
      AllStartReceiving_(phase, bvars_aux);
      break;
    }
    case BoundaryCommSubset::iterated_z4c:
    {
      AllStartReceiving_(phase, bvars_rbc);
      break;
    }
    case BoundaryCommSubset::aux_adm:
    {
      AllStartReceiving_(phase, bvars_aux_adm);
      break;
    }
    case BoundaryCommSubset::mesh_init:
    {
      AllStartReceiving_(phase, bvars_main_int);
      AllStartReceiving_(phase, bvars_main_int_vc);
      AllStartReceiving_(phase, bvars_main_int_cx);
      AllStartReceiving_(phase, bvars_m1);
      break;
    }
    case BoundaryCommSubset::matter:
    case BoundaryCommSubset::matter_primitives:
    case BoundaryCommSubset::matter_flux_corrected:
    {
      AllStartReceiving_(phase, bvars_main_int);
      break;
    }
    case BoundaryCommSubset::m1:
    {
      AllStartReceiving_(phase, bvars_m1);
      break;
    }
    case BoundaryCommSubset::z4c:
    {
      AllStartReceiving_(
        phase,
        FCN_CC_CX_VC(bvars_main_int,
                     bvars_main_int_cx,
                     bvars_main_int_vc)
      );
      break;
    }
    case BoundaryCommSubset::all:
    {
      AllStartReceiving_(phase, bvars_main_int);
      AllStartReceiving_(phase, bvars_main_int_vc);
      AllStartReceiving_(phase, bvars_main_int_cx);
      break;
    }
    default:
    {
      assert(false);
    }
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void BoundaryValues::ClearBoundary(BoundaryCommSubset phase)
//  \brief clean up the boundary flags after each loop

void BoundaryValues::ClearBoundary(BoundaryCommSubset phase) {
  // Note BoundaryCommSubset::mesh_init corresponds to initial exchange of conserved fluid
  // variables and magentic fields, while BoundaryCommSubset::matter_primitives corresponds to fluid
  // primitive variables sent only in the case of GR with refinement
  // Cf. StartReceiving
  switch (phase)
  {
    case BoundaryCommSubset::aux_z4c:
    {
      AllClearBoundary_(phase, bvars_aux);
      break;
    }
    case BoundaryCommSubset::iterated_z4c:
    {
      AllClearBoundary_(phase, bvars_rbc);
      break;
    }
    case BoundaryCommSubset::aux_adm:
    {
      AllClearBoundary_(phase, bvars_aux_adm);
      break;
    }
    case BoundaryCommSubset::mesh_init:
    {
      AllClearBoundary_(phase, bvars_main_int);
      AllClearBoundary_(phase, bvars_main_int_vc);
      AllClearBoundary_(phase, bvars_main_int_cx);
      AllClearBoundary_(phase, bvars_m1);
      break;
    }
    case BoundaryCommSubset::matter:
    case BoundaryCommSubset::matter_primitives:
    case BoundaryCommSubset::matter_flux_corrected:
    {
      AllClearBoundary_(phase, bvars_main_int);
      break;
    }
    case BoundaryCommSubset::m1:
    {
      AllClearBoundary_(phase, bvars_m1);
      break;
    }
    case BoundaryCommSubset::z4c:
    {
      AllClearBoundary_(
        phase,
        FCN_CC_CX_VC(bvars_main_int,
                     bvars_main_int_cx,
                     bvars_main_int_vc)
      );
      break;
    }
    case BoundaryCommSubset::all:
    {
      AllClearBoundary_(phase, bvars_main_int);
      AllClearBoundary_(phase, bvars_main_int_vc);
      AllClearBoundary_(phase, bvars_main_int_cx);
      break;
    }
    default:
    {
      assert(false);
    }
  }

}

// Handler for application of physical boundaries for all variable types
void BoundaryValues::ApplyPhysicalBoundaries(
  const Real time, const Real dt,
  std::vector<BoundaryVariable *> & bvars,
  const int var_is, const int var_ie,
  const int var_js, const int var_je,
  const int var_ks, const int var_ke,
  const int ng)
{
  MeshBlock *pmb = pmy_block_;
  Coordinates *pco = pmb->pcoord;
  int bis = var_is - ng, bie = var_ie + ng,
      bjs = var_js, bje = var_je,
      bks = var_ks, bke = var_ke;

  // Extend the transverse limits that correspond to periodic boundaries as they are
  // updated: x1, then x2, then x3
  if (!apply_bndry_fn_[BoundaryFace::inner_x2] && pmb->block_size.nx2 > 1)
    bjs = var_js - ng;
  if (!apply_bndry_fn_[BoundaryFace::outer_x2] && pmb->block_size.nx2 > 1)
    bje = var_je + ng;
  if (!apply_bndry_fn_[BoundaryFace::inner_x3] && pmb->block_size.nx3 > 1)
    bks = var_ks - ng;
  if (!apply_bndry_fn_[BoundaryFace::outer_x3] && pmb->block_size.nx3 > 1)
    bke = var_ke + ng;

  // Apply boundary function on inner-x1 and update W,bcc (if not periodic)
  if (apply_bndry_fn_[BoundaryFace::inner_x1])
  {
    DispatchBoundaryFunctions(pmb, pco, time, dt,
                              var_is, var_ie,
                              bjs, bje,
                              bks, bke,
                              ng,
                              BoundaryFace::inner_x1,
                              bvars);
  }

  // Apply boundary function on outer-x1 and update W,bcc (if not periodic)
  if (apply_bndry_fn_[BoundaryFace::outer_x1]) {
    DispatchBoundaryFunctions(pmb, pco, time, dt,
                              var_is, var_ie,
                              bjs, bje,
                              bks, bke,
                              ng,
                              BoundaryFace::outer_x1,
                              bvars);
  }

  if (pmb->block_size.nx2 > 1) { // 2D or 3D
    // Apply boundary function on inner-x2 and update W,bcc (if not periodic)
    if (apply_bndry_fn_[BoundaryFace::inner_x2])
    {
      DispatchBoundaryFunctions(pmb, pco, time, dt,
                                bis, bie,
                                var_js, var_je,
                                bks, bke,
                                ng,
                                BoundaryFace::inner_x2,
                                bvars);
    }

    // Apply boundary function on outer-x2 and update W,bcc (if not periodic)
    if (apply_bndry_fn_[BoundaryFace::outer_x2])
    {
      DispatchBoundaryFunctions(pmb, pco, time, dt,
                                bis, bie,
                                var_js, var_je,
                                bks, bke,
                                ng,
                                BoundaryFace::outer_x2,
                                bvars);
    }
  }

  if (pmb->block_size.nx3 > 1)
  { // 3D
    bjs = var_js - ng;
    bje = var_je + ng;

    // Apply boundary function on inner-x3 and update W,bcc (if not periodic)
    if (apply_bndry_fn_[BoundaryFace::inner_x3])
    {
      DispatchBoundaryFunctions(pmb, pco, time, dt,
                                bis, bie,
                                bjs, bje,
                                var_ks, var_ke,
                                ng,
                                BoundaryFace::inner_x3,
                                bvars);
    }

    // Apply boundary function on outer-x3 and update W,bcc (if not periodic)
    if (apply_bndry_fn_[BoundaryFace::outer_x3])
    {
      DispatchBoundaryFunctions(pmb, pco, time, dt,
                                bis, bie,
                                bjs, bje,
                                var_ks, var_ke,
                                ng,
                                BoundaryFace::outer_x3,
                                bvars);
    }
  }
  return;
}

void BoundaryValues::PrimitiveToConservedOnPhysicalBoundaries()
{
#if FLUID_ENABLED
  MeshBlock *pmb = pmy_block_;
  Coordinates *pco = pmb->pcoord;
  EquationOfState *peos = pmb->peos;

  Hydro *ph = pmb->phydro;
  Field *pf = (MAGNETIC_FIELDS_ENABLED) ? pmb->pfield : nullptr;
  PassiveScalars *ps = (NSCALARS > 0) ? pmb->pscalars : nullptr;

  static const int ng = NGHOST;
  const int var_is = pmb->is;
  const int var_ie = pmb->ie;
  const int var_js = pmb->js;
  const int var_je = pmb->je;
  const int var_ks = pmb->ks;
  const int var_ke = pmb->ke;

  int bis = var_is - ng, bie = var_ie + ng,
      bjs = var_js, bje = var_je,
      bks = var_ks, bke = var_ke;

  // Extend the transverse limits that correspond to periodic boundaries as they are
  // updated: x1, then x2, then x3
  if (!apply_bndry_fn_[BoundaryFace::inner_x2] && pmb->block_size.nx2 > 1)
    bjs = var_js - ng;
  if (!apply_bndry_fn_[BoundaryFace::outer_x2] && pmb->block_size.nx2 > 1)
    bje = var_je + ng;
  if (!apply_bndry_fn_[BoundaryFace::inner_x3] && pmb->block_size.nx3 > 1)
    bks = var_ks - ng;
  if (!apply_bndry_fn_[BoundaryFace::outer_x3] && pmb->block_size.nx3 > 1)
    bke = var_ke + ng;

  if (apply_bndry_fn_[BoundaryFace::inner_x1])
  {
    peos->PrimitiveToConserved(ph->w, ps->r, pf->bcc, ph->u, ps->s,
                               pco,
                               var_is - ng, var_is - 1,
                               bjs, bje,
                               bks, bke);
  }

  if (apply_bndry_fn_[BoundaryFace::outer_x1])
  {
    peos->PrimitiveToConserved(ph->w, ps->r, pf->bcc, ph->u, ps->s,
                               pco,
                               var_ie + 1, var_ie + ng,
                               bjs, bje,
                               bks, bke);
  }

  if (pmb->block_size.nx2 > 1)
  { // 2D or 3D
    if (apply_bndry_fn_[BoundaryFace::inner_x2])
    {
      peos->PrimitiveToConserved(ph->w, ps->r, pf->bcc, ph->u, ps->s,
                                 pco,
                                 bis, bie,
                                 var_js - ng, var_js - 1,
                                 bks, bke);
    }

    if (apply_bndry_fn_[BoundaryFace::outer_x2])
    {
      peos->PrimitiveToConserved(ph->w, ps->r, pf->bcc, ph->u, ps->s,
                                 pco,
                                 bis, bie,
                                 var_je + 1, var_je + ng,
                                 bks, bke);
    }
  }

  if (pmb->block_size.nx3 > 1)
  { // 3D
    bjs = var_js - ng;
    bje = var_je + ng;

    if (apply_bndry_fn_[BoundaryFace::inner_x3])
    {
      peos->PrimitiveToConserved(ph->w, ps->r, pf->bcc, ph->u, ps->s,
                                 pco,
                                 bis, bie,
                                 bjs, bje,
                                 var_ks - ng, var_ks - 1);
    }

    if (apply_bndry_fn_[BoundaryFace::outer_x3])
    {
      peos->PrimitiveToConserved(ph->w, ps->r, pf->bcc, ph->u, ps->s,
                                 pco,
                                 bis, bie,
                                 bjs, bje,
                                 var_ke + 1, var_ke + ng);
    }
  }
#endif // FLUID_ENABLED
}

void BoundaryValues::DispatchBoundaryFunctions(
  MeshBlock *pmb, Coordinates *pco, Real time, Real dt,
  int il, int iu,
  int jl, int ju,
  int kl, int ku,
  int ngh,
  BoundaryFace face,
  std::vector<BoundaryVariable *> &bvars)
{
  if (block_bcs[face] ==  BoundaryFlag::user)
  {  // user-enrolled BCs
    pmy_mesh_->BoundaryFunction_[face](pmb, pco, time, dt,
                                       il, iu,
                                       jl, ju,
                                       kl, ku, ngh);
  }
  // For any function in the BoundaryPhysics interface class, iterate over
  // BoundaryVariable pointers "enrolled"
  for (auto bvars_it = bvars.begin(); bvars_it != bvars.end();
       ++bvars_it) {
    switch (block_bcs[face]) {
    case BoundaryFlag::user: // handled above, outside loop over BoundaryVariable objs
      break;
    case BoundaryFlag::reflect:
      switch (face) {
      case BoundaryFace::undef: {
        std::stringstream msg;
        msg << "### FATAL ERROR in DispatchBoundaryFunctions" << std::endl
            << "face = BoundaryFace::undef passed to this function" << std::endl;
        ATHENA_ERROR(msg);
      }
      case BoundaryFace::inner_x1:
        (*bvars_it)->ReflectInnerX1(time, dt, il, jl, ju, kl, ku, ngh);
        break;
      case BoundaryFace::outer_x1:
        (*bvars_it)->ReflectOuterX1(time, dt, iu, jl, ju, kl, ku, ngh);
        break;
      case BoundaryFace::inner_x2:
        (*bvars_it)->ReflectInnerX2(time, dt, il, iu, jl, kl, ku, ngh);
        break;
      case BoundaryFace::outer_x2:
        (*bvars_it)->ReflectOuterX2(time, dt, il, iu, ju, kl, ku, ngh);
        break;
      case BoundaryFace::inner_x3:
        (*bvars_it)->ReflectInnerX3(time, dt, il, iu, jl, ju, kl, ngh);
        break;
      case BoundaryFace::outer_x3:
        (*bvars_it)->ReflectOuterX3(time, dt, il, iu, jl, ju, ku, ngh);
        break;
      }
      break;
    case BoundaryFlag::outflow:
      switch (face) {
      case BoundaryFace::undef: {
        std::stringstream msg;
        msg << "### FATAL ERROR in DispatchBoundaryFunctions" << std::endl
            << "face = BoundaryFace::undef passed to this function" << std::endl;
        ATHENA_ERROR(msg);
      }
      case BoundaryFace::inner_x1:
        (*bvars_it)->OutflowInnerX1(time, dt, il, jl, ju, kl, ku, ngh);
        break;
      case BoundaryFace::outer_x1:
        (*bvars_it)->OutflowOuterX1(time, dt, iu, jl, ju, kl, ku, ngh);
        break;
      case BoundaryFace::inner_x2:
        (*bvars_it)->OutflowInnerX2(time, dt, il, iu, jl, kl, ku, ngh);
        break;
      case BoundaryFace::outer_x2:
        (*bvars_it)->OutflowOuterX2(time, dt, il, iu, ju, kl, ku, ngh);
        break;
      case BoundaryFace::inner_x3:
        (*bvars_it)->OutflowInnerX3(time, dt, il, iu, jl, ju, kl, ngh);
        break;
      case BoundaryFace::outer_x3:
        (*bvars_it)->OutflowOuterX3(time, dt, il, iu, jl, ju, ku, ngh);
        break;
      }
      break;
    case BoundaryFlag::extrapolate_outflow:
      switch (face) {
      case BoundaryFace::undef: {
        std::stringstream msg;
        msg << "### FATAL ERROR in DispatchBoundaryFunctions" << std::endl
            << "face = BoundaryFace::undef passed to this function" << std::endl;
        ATHENA_ERROR(msg);
      }
      case BoundaryFace::inner_x1:
        (*bvars_it)->ExtrapolateOutflowInnerX1(time, dt, il, jl, ju, kl, ku, ngh);
        break;
      case BoundaryFace::outer_x1:
        (*bvars_it)->ExtrapolateOutflowOuterX1(time, dt, iu, jl, ju, kl, ku, ngh);
        break;
      case BoundaryFace::inner_x2:
        (*bvars_it)->ExtrapolateOutflowInnerX2(time, dt, il, iu, jl, kl, ku, ngh);
        break;
      case BoundaryFace::outer_x2:
        (*bvars_it)->ExtrapolateOutflowOuterX2(time, dt, il, iu, ju, kl, ku, ngh);
        break;
      case BoundaryFace::inner_x3:
        (*bvars_it)->ExtrapolateOutflowInnerX3(time, dt, il, iu, jl, ju, kl, ngh);
        break;
      case BoundaryFace::outer_x3:
        (*bvars_it)->ExtrapolateOutflowOuterX3(time, dt, il, iu, jl, ju, ku, ngh);
        break;
      }
      break;
    case BoundaryFlag::gr_sommerfeld:
      switch (face) {
      case BoundaryFace::undef: {
        std::stringstream msg;
        msg << "### FATAL ERROR in DispatchBoundaryFunctions" << std::endl
            << "face = BoundaryFace::undef passed to this function" << std::endl;
        ATHENA_ERROR(msg);
      }
      case BoundaryFace::inner_x1:
        (*bvars_it)->GRSommerfeldInnerX1(time, dt, il, jl, ju, kl, ku, ngh);
        break;
      case BoundaryFace::outer_x1:
        (*bvars_it)->GRSommerfeldOuterX1(time, dt, iu, jl, ju, kl, ku, ngh);
        break;
      case BoundaryFace::inner_x2:
        (*bvars_it)->GRSommerfeldInnerX2(time, dt, il, iu, jl, kl, ku, ngh);
        break;
      case BoundaryFace::outer_x2:
        (*bvars_it)->GRSommerfeldOuterX2(time, dt, il, iu, ju, kl, ku, ngh);
        break;
      case BoundaryFace::inner_x3:
        (*bvars_it)->GRSommerfeldInnerX3(time, dt, il, iu, jl, ju, kl, ngh);
        break;
      case BoundaryFace::outer_x3:
        (*bvars_it)->GRSommerfeldOuterX3(time, dt, il, iu, jl, ju, ku, ngh);
        break;
      }
      break;
    case BoundaryFlag::polar_wedge:
      switch (face) {
      case BoundaryFace::undef: {
        std::stringstream msg;
        msg << "### FATAL ERROR in DispatchBoundaryFunctions" << std::endl
            << "face = BoundaryFace::undef passed to this function" << std::endl;
        ATHENA_ERROR(msg);
      }
      case BoundaryFace::inner_x2:
        (*bvars_it)->PolarWedgeInnerX2(time, dt, il, iu, jl, kl, ku, ngh);
        break;
      case BoundaryFace::outer_x2:
        (*bvars_it)->PolarWedgeOuterX2(time, dt, il, iu, ju, kl, ku, ngh);
        break;
      default:
        std::stringstream msg_polar;
        msg_polar << "### FATAL ERROR in DispatchBoundaryFunctions" << std::endl
                  << "Attempting to call polar wedge boundary function on \n"
                  << "MeshBlock boundary other than inner x2 or outer x2"
                  << std::endl;
        ATHENA_ERROR(msg_polar);
      }
        break;
      default:
        std::stringstream msg_flag;
        msg_flag << "### FATAL ERROR in DispatchBoundaryFunctions" << std::endl
                 << "No BoundaryPhysics function associated with provided\n"
                 << "block_bcs[" << face << "] = BoundaryFlag::"
                 << GetBoundaryString(block_bcs[face]) << std::endl;
        ATHENA_ERROR(msg_flag);
        break;
    } // end switch (block_bcs[face])
  } // end loop over BoundaryVariable *
}



// Public function, to be called in MeshBlock ctor for keeping MPI tag bitfields
// consistent across MeshBlocks, even if certain MeshBlocks only construct a subset of
// physical variable classes

int BoundaryValues::AdvanceCounterPhysID(int num_phys) {
#ifdef MPI_PARALLEL
  // TODO(felker): add safety checks? input, output are positive, obey <= 31= MAX_NUM_PHYS
  int start_id = bvars_next_phys_id_;
  bvars_next_phys_id_ += num_phys;
  return start_id;
#else
  return 0;
#endif
}
