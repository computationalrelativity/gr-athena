//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_shock_tube.cpp
//  \brief Problem generator for shock tubes in special and general relativity.

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <cstring>    // strcmp()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"                   // macros, enums
#include "../athena_arrays.hpp"            // AthenaArray
#include "../coordinates/coordinates.hpp"  // Coordinates
#include "../eos/eos.hpp"                  // EquationOfState
#include "../field/field.hpp"              // Field
#include "../hydro/hydro.hpp"              // Hydro
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"          // ParameterInput

// Configuration checking
#if not Z4C_ENABLED
#error "This problem generator must be used with z4c"
#endif


//----------------------------------------------------------------------------------------
// Function for setting initial conditions
// Inputs:
//   pin: parameters
// Outputs: (none)
// Notes:
//   sets conserved variables according to input primitives
//   assigns fields based on cell-center positions, rather than interface positions
//     this helps shock tube 2 from Mignone, Ugliano, & Bodo 2009, MNRAS 393 1141
//     otherwise the middle interface would go to left variables, creating a
//         particularly troublesome jump leading to NaN's

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Read and set ratio of specific heats
  Real gamma_adi = peos->GetGamma();
  Real gamma_adi_red = gamma_adi / (gamma_adi - 1.0);

  // Read and check shock direction and position
  int shock_dir = pin->GetInteger("problem", "shock_dir");
  Real shock_pos = pin->GetReal("problem", "xshock");
  Real min_bound, max_bound;
  std::stringstream msg;
  switch (shock_dir) {
    case 1:
      min_bound = pmy_mesh->mesh_size.x1min;
      max_bound = pmy_mesh->mesh_size.x1max;
      break;
    case 2:
      min_bound = pmy_mesh->mesh_size.x2min;
      max_bound = pmy_mesh->mesh_size.x2max;
      break;
    case 3:
      min_bound = pmy_mesh->mesh_size.x3min;
      max_bound = pmy_mesh->mesh_size.x3max;
      break;
    default:
      msg << "### FATAL ERROR in Problem Generator\n"
          << "shock_dir=" << shock_dir << " must be either 1, 2, or 3" << std::endl;
      ATHENA_ERROR(msg);
  }
  if (shock_pos < min_bound || shock_pos > max_bound) {
    msg << "### FATAL ERROR in Problem Generator\n"
        << "xshock=" << shock_pos << " lies outside x" << shock_dir
        << " domain for shkdir=" << shock_dir << std::endl;
    ATHENA_ERROR(msg);
  }
  // NB input velocities are assumed to be 3 velocities v^i
  // NB input B fields are assumed to be the evolved variable \sqrt(\detg) B^i
  // Read left state
  Real rho_left = pin->GetReal("problem", "dl");
  Real pgas_left = pin->GetReal("problem", "pl");
  Real vx_left = pin->GetReal("problem", "ul");
  Real vy_left = pin->GetReal("problem", "vl");
  Real vz_left = pin->GetReal("problem", "wl");
  Real bbx_left = 0.0, bby_left = 0.0, bbz_left = 0.0;
  if (MAGNETIC_FIELDS_ENABLED) {
    bbx_left = pin->GetReal("problem", "bxl");
    bby_left = pin->GetReal("problem", "byl");
    bbz_left = pin->GetReal("problem", "bzl");
  }

  // Read right state
  Real rho_right = pin->GetReal("problem", "dr");
  Real pgas_right = pin->GetReal("problem", "pr");
  Real vx_right = pin->GetReal("problem", "ur");
  Real vy_right = pin->GetReal("problem", "vr");
  Real vz_right = pin->GetReal("problem", "wr");
  Real bbx_right = 0.0, bby_right = 0.0, bbz_right = 0.0;
  if (MAGNETIC_FIELDS_ENABLED) {
    bbx_right = pin->GetReal("problem", "bxr");
    bby_right = pin->GetReal("problem", "byr");
    bbz_right = pin->GetReal("problem", "bzr");
  }

  // Prepare auxiliary arrays
  AthenaArray<Real> bb;
  bb.NewAthenaArray(3, ke+1, je+1, ie+1);


  // Initialise ADM variables to Minkowski
  pz4c->ADMMinkowski(pz4c->storage.adm);
  // Initialise lapse to 1, shift 0
  pz4c->GaugeGeodesic(pz4c->storage.u);
  // populate remaining z4c vars
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);



  // Initialize hydro variables
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        // Determine which variables to use
        Real rho = rho_right;
        Real pgas = pgas_right;
        Real vx = vx_right;
        Real vy = vy_right;
        Real vz = vz_right;
        Real bbx = bbx_right;
        Real bby = bby_right;
        Real bbz = bbz_right;
        bool left_side = false;
        switch(shock_dir) {
          case 1:
            left_side = pcoord->x1v(i) < shock_pos;
            break;
          case 2:
            left_side = pcoord->x2v(j) < shock_pos;
            break;
          case 3:
            left_side = pcoord->x3v(k) < shock_pos;
            break;
        }
        if (left_side) {
          rho = rho_left;
          pgas = pgas_left;
          vx = vx_left;
          vy = vy_left;
          vz = vz_left;
          bbx = bbx_left;
          bby = bby_left;
          bbz = bbz_left;
        }

        // Construct primitive velocity
        Real W = std::sqrt(1.0 / (1.0 - (SQR(vx)+SQR(vy)+SQR(vz))));
        Real utilde_x = W * vx;
        Real utilde_y = W * vy;
        Real utilde_z = W * vz;


        // Set primitives
        phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = rho;
        phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = pgas;
        phydro->w(IVX,k,j,i) = phydro->w1(IVX,k,j,i) = utilde_x; 
        phydro->w(IVY,k,j,i) = phydro->w1(IVY,k,j,i) = utilde_y; 
        phydro->w(IVZ,k,j,i) = phydro->w1(IVZ,k,j,i) = utilde_z; 
        // Set magnetic fields
        bb(IB1,k,j,i) = bbx;
        bb(IB2,k,j,i) = bby;
        bb(IB3,k,j,i) = bbz;
      }
    }
  }


  peos->PrimitiveToConserved(phydro->w, 
#if USETM
  pscalars->r, 
#endif
  bb, phydro->u,
#if USETM
  pscalars->s, 
#endif
  pcoord, is, ie, js, je, ks, ke);

  // Initialize magnetic field - untested
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je+1; ++j) {
        for (int i=is; i<=ie+1; ++i) {
          // Determine which variables to use
          Real bbx = bbx_right;
          Real bby = bby_right;
          Real bbz = bbz_right;
          bool left_side = false;
          switch(shock_dir) {
            case 1:
              left_side = pcoord->x1v(i) < shock_pos;
              break;
            case 2:
              left_side = pcoord->x2v(j) < shock_pos;
              break;
            case 3:
              left_side = pcoord->x3v(k) < shock_pos;
              break;
          }
          if (left_side) {
            bbx = bbx_left;
            bby = bby_left;
            bbz = bbz_left;
          }

          // Set magnetic fields
          if (j != je+1 && k != ke+1) {
            pfield->b.x1f(k,j,i) = bbx;
          }
          if (i != ie+1 && k != ke+1) {
            pfield->b.x2f(k,j,i) = bby;
          }
          if (i != ie+1 && j != je+1) {
            pfield->b.x3f(k,j,i) = bbz;
          }
        }
      }
    }
  }
  return;
}

