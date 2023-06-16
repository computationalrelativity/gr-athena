//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================

#include <cassert> // assert
#include <iostream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../z4c/z4c.hpp"
#include "../hydro/hydro.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"


// twopuncturesc: Stand-alone library ripped from Cactus
#include "RNS.h"

using namespace std;

int RefinementCondition(MeshBlock *pmb);
//namespace {

static ini_data *data;
Real Maxrho(MeshBlock *pmb, int iout);

//}

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin, int res_flag) {
  AllocateUserHistoryOutput(1);
  EnrollUserHistoryOutput(0, Maxrho, "max-rho", UserHistoryOperation::max);
  
  if (!res_flag) {
    string set_name = "problem";
    RNS_params_set_default();

    string inputfile = pin->GetOrAddString("problem", "filename", "tovgamma2.par");
    Real dfloor = pin->GetReal("hydro", "dfloor");

    RNS_params_set_Real("atm_level_rho",dfloor);
    RNS_params_set_inputfile((char *) inputfile.c_str());

    data = RNS_make_initial_data();
  }

  return;
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin, int res_flag) {
  if (!res_flag) {
    RNS_finalise(data);
  }
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Sets the initial conditions.
//========================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  #ifdef Z4C_ASSERT_FINITE
  // sanity check (these should be overwritten)
  pz4c->adm.psi4.Fill(NAN);
  pz4c->adm.g_dd.Fill(NAN);
  pz4c->adm.K_dd.Fill(NAN);

  pz4c->z4c.chi.Fill(NAN);
  pz4c->z4c.Khat.Fill(NAN);
  pz4c->z4c.Theta.Fill(NAN);
  pz4c->z4c.alpha.Fill(NAN);
  pz4c->z4c.Gam_u.Fill(NAN);
  pz4c->z4c.beta_u.Fill(NAN);
  pz4c->z4c.g_dd.Fill(NAN);
  pz4c->z4c.A_dd.Fill(NAN);
  #endif // Z4C_ASSERT_FINITE

  // Interpolate metric quantities to VC
  phydro->RNS_Metric(pin, pz4c->storage.adm, pz4c->storage.u, pz4c->storage.u1, data);

  // Interpolate primitives to CC
  phydro->RNS_Hydro(pin, phydro->w, phydro->w1, phydro->w_init, data);

  // Superimpose a boosted puncture.
  pz4c->ADMAddBoostedPuncture(pin, pz4c->storage.adm, pz4c->storage.u, 0);

  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u1);

  pcoord->UpdateMetric();

  if (pmy_mesh->multilevel) {
    pmr->pcoarsec->UpdateMetric();
  }
  // Prepare index bounds
  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;
  if (block_size.nx2 > 1) {
    jl -= NGHOST;
    ju += NGHOST;
  }
  int kl = ks;
  int ku = ke;
  if (block_size.nx3 > 1) {
    kl -= NGHOST;
    ku += NGHOST;
  }

  // Initialize conserved variables
  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, il, iu, jl, ju, kl, ku);

  // Initialize VC matter
  pz4c->GetMatter(pz4c->storage.mat, pz4c->storage.adm, phydro->w, pfield->bcc);
  pz4c->ADMConstraints(pz4c->storage.con, pz4c->storage.adm, pz4c->storage.mat, pz4c->storage.u);

  #ifdef Z4C_ASSERT_FINITE
  pz4c->assert_is_finite_adm();
  pz4c->assert_is_finite_con();
  pz4c->assert_is_finite_adm();
  pz4c->assert_is_finite_z4c();
  #endif
}

Real Maxrho(MeshBlock *pmb, int iout) {
  Real max_rho = 0.0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &w = pmb->phydro->w;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        max_rho = std::max(std::abs(w(IDN,k,j,i)), max_rho);
      }
    }
  }
  return max_rho;
}
