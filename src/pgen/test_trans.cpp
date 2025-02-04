//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gr_tov.cpp
//  \brief Problem generator for single TOV star in Cowling approximation

// C headers

// C++ headers
#include <cmath>   // abs(), NAN, pow(), sqrt()
#include <cstring> // strcmp()
#include <stdexcept>  // runtime_error
#include <string>     // string
#include <cfloat>

// Athena++ headers
#include "../athena_aliases.hpp"
#include "../bvals/bvals.hpp"              // BoundaryValues
#include "../coordinates/coordinates.hpp"  // Coordinates
#include "../eos/eos.hpp"                  // EquationOfState
#include "../z4c/z4c.hpp"
#include "../trackers/extrema_tracker.hpp"
#include "../field/field.hpp"              // Field
#include "../hydro/hydro.hpp"              // Hydro
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"          // ParameterInput
#include "../scalars/scalars.hpp"

//----------------------------------------------------------------------------------------
using namespace gra::aliases;
//----------------------------------------------------------------------------------------


namespace {
  Real rho_min, rho_max;
  Real temp_min, temp_max;
  Real ye_min, ye_max;
  bool do_c2p;
}

//----------------------------------------------------------------------------------------
//! \fn
// \brief  Function for initializing global mesh properties
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Read problem parameters

  rho_min = pin->GetReal("problem", "rho_min");
  rho_max = pin->GetReal("problem", "rho_max");
  temp_min = pin->GetReal("problem", "temp_max");
  temp_max = pin->GetReal("problem", "temp_max");
  ye_min = pin->GetReal("problem", "ye_max");
  ye_max = pin->GetReal("problem", "ye_max");
  do_c2p = pin->GetOrAddBoolean("problem", "do_c2p", false);
  if (do_c2p) {
    // TODO: implement
    throw std::runtime_error("do_c2p not implemented");
  }
}

//----------------------------------------------------------------------------------------
//! \fn
// \brief Setup User work

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  return;
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  return;
}

void MeshBlock::UserWorkAfterOutput(ParameterInput *pin) {
  return;
}

void Mesh::DeleteTemporaryUserMeshData() {
  return;
}


//----------------------------------------------------------------------------------------
//! \fn
// \brief Function for setting initial conditions
// Inputs:
//   pin: parameters
// Outputs: (none)
// Notes:
//   sets primitive and conserved variables according to input primitives
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  // Parameters - prefilled as TOV is added to these quantities
  phydro->w.Fill(        0);
  phydro->w1.Fill(       0);
  pscalars->r.Fill(      0);
  pz4c->storage.u.Fill(  0);
  pz4c->storage.u1.Fill( 0);
  pz4c->storage.adm.Fill(0);
  pz4c->storage.mat.Fill(0);

  Real mb = peos->GetEOS().GetBaryonMass();
  Real T_trans = peos->GetEOS().GetTempTransStart();
  Real n_trans = peos->GetEOS().GetDensTransStart();

  for (int k = 0; k < ncells3; ++k)
  for (int j = 0; j < ncells2; ++j)
  for (int i = 0; i < ncells1; ++i) {
      Real rho = rho_min + (rho_max - rho_min) * (Real)i / (Real)ncells1;
      Real temp = temp_min + (temp_max - temp_min) * (Real)j / (Real)ncells2;
      Real ye = ye_min + (ye_max - ye_min) * (Real)k / (Real)ncells3;

      Real nb = rho/mb;
      bool transition = (nb < n_trans) or (temp < T_trans);
      Real Y[MAX_SPECIES] = {ye, 1.0};
      Real abar;
      if (transition) {
         abar = peos->GetEOS().GetAbar(n_trans, T_trans, Y);
      } else {
         abar = peos->GetEOS().GetAbar(nb, temp, Y);
      }
      Y[1] = abar;


      phydro->w(IDN, k, j, i) = rho;
      phydro->w(IPR, k, j, i) = peos->GetEOS().GetPressure(rho/mb, temp, Y);
      pscalars->r(0, k, j, i) = ye;
      pscalars->r(1, k, j, i) = abar;
      // peos->ApplyPrimitiveFloors(phydro->w, pscalars->r, k, j, i);

  }
  // --------------------------------------------------------------------------

  // Initialise conserved variables
  peos->PrimitiveToConserved(phydro->w,
                             pscalars->r,
                             pfield->bcc,
                             phydro->u,
                             pscalars->s,
                             pcoord,
                             0, ncells1-1,
                             0, ncells2-1,
                             0, ncells3-1);

  // Initialise matter (also taken care of in task-list)
  pz4c->GetMatter(pz4c->storage.mat,
                  pz4c->storage.adm,
                  phydro->w,
                  pscalars->r,
                  pfield->bcc);

  return;
}
