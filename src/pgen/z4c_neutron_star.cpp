//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//! \file z4c_neutron_star.cpp
//  \brief Initial conditions for an axisymmetric neutron star with a magnetic field.

#include <cassert>
#include <iostream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../z4c/z4c.hpp"

// Lorene
#include <mag_ns.h>

using namespace std;


//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin){
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin){
    
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Sets the initial conditions.
//========================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin){
#ifdef Z4C_ASSERT_FINITE
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

	pz4c->mat.rho.Fill(NAN);
	pz4c->mat.S_d.Fill(NAN);
	pz4c->mat.S_dd.Fill(NAN);
#endif

	// Interpolate Lorene data onto the grid.
	pz4c->ADMNeutronStar(pin, pz4c->storage.adm, pz4c->storage.mat);

	// Precollapse

	// Construct Z4c vars from ADM vars.
    
}

int RefinementCondition(MeshBlock *pmb){
    
}
