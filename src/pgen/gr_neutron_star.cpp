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
#include "../athena_tensor.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../z4c/z4c.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../bvals/bvals.hpp"
#include "../mesh/mesh.hpp"

// Lorene
#include <mag_ns.h>

using namespace std;


//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin, int res_flag){
}

/*void Mesh::UserWorkAfterLoop(ParameterInput *pin){
    
}*/

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
	//pz4c->ADMNeutronStar(pin, pz4c->storage.u, pz4c->storage.adm, phydro->w);

	// Some conversion factors to go between Lorene data and Athena data. Shamelessly stolen from
	// the Einstein Toolkit's Mag_NS.cc.
	Real const c_light = 299792458.0; // Speed of light [m/s]
	Real const mu0 = 4.0 * M_PI * 1.0e-7; // Vacuum permeability [N/A^2]
	Real const eps0 = 1.0 / (mu0 * pow(c_light, 2));

	// Constants of nature (IAU, CODATA):
	Real const G_grav = 6.67428e-11; // Gravitational constant [m^3/kg/s^2]
	Real const M_sun = 1.98892e+30; // Solar mass [kg]

	// Athena units for conversion.
	Real const coord_unit = 1.0; // Placeholder -- figure out what this actually is.
	Real const rho_unit = 1.0; // Placeholder -- figure out what this actually is.
	Real const ener_unit = 1.0; // Placeholder -- figure out what this actually is.
	Real const vel_unit = 1.0; // Placeholder -- figure out what this actually is.
	Real const B_unit = 1.0; // Placeholder -- figure out what this actually is.

	// Set some aliases for the variables.
	// FIXME: This needs to be generalized to be independent of Z4c.
	AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> alpha;
	AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> beta_u;
	AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> g_dd;
	AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> K_dd;
	alpha.InitWithShallowSlice(pz4c->storage.u, Z4c::I_Z4c_alpha);
	beta_u.InitWithShallowSlice(pz4c->storage.u, Z4c::I_Z4c_betax);
	g_dd.InitWithShallowSlice(pz4c->storage.adm, Z4c::I_ADM_gxx);
	K_dd.InitWithShallowSlice(pz4c->storage.adm, Z4c::I_ADM_Kxx);

	// Index bounds
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
	
	// Coordinates for Lorene.
	int const nx = iu - il + 1;
	int const ny = ju - jl + 1;
	int const nz = ku - kl + 1;
	int const npoints = nx * ny * nz;
	std::vector<double> xx(npoints), yy(npoints), zz(npoints);

	for (int k = kl; k < ku; k++) {
		for (int j = jl; j < ju; j++) {
			for (int i = il; i < iu; i++) {
				int index = (i - kl) + nx * ((j - jl) + ny * (k - kl));
				xx[index] = pcoord->x1v(i) * coord_unit;
				yy[index] = pcoord->x2v(j) * coord_unit;
				zz[index] = pcoord->x3v(k) * coord_unit;
			}
		}
	}

	// Read in the file from Lorene.
	std::string filename = pin->GetOrAddString("problem", "filename", "resu.d");
	Lorene::Mag_NS mag_ns(npoints, &xx[0], &yy[0], &zz[0], filename.c_str());
	assert(mag_ns.np == npoints);

	for (int k = kl; k < ku; k++) {
		for (int j = jl; j < ju; j++) {
			for (int i = il; i < iu; i++) {
				int index = (i - kl) + nx * ((j - jl) + ny * (k - kl));

				// Copy over Lorene's gauge variables.
				alpha(k, j, i) = mag_ns.nnn[index];
				beta_u(0, k, j, i) = mag_ns.beta_x[index];
				beta_u(1, k, j, i) = mag_ns.beta_y[index];
				beta_u(2, k, j, i) = mag_ns.beta_z[index];

				// Copy the 3-metric into a temporary array.
				Real g[3][3];
				g[0][0] = mag_ns.g_xx[index];
				g[0][1] = mag_ns.g_xy[index];
				g[0][2] = mag_ns.g_xz[index];
				g[1][1] = mag_ns.g_yy[index];
				g[1][2] = mag_ns.g_yz[index];
				g[2][2] = mag_ns.g_zz[index];
				g[1][0] = g[0][1];
				g[2][0] = g[0][2];
				g[2][1] = g[1][2];

				// Copy the curvature into a temporary array.
				Real ku[3][3];
				ku[0][0] = mag_ns.k_xx[index];
				ku[0][1] = mag_ns.k_xy[index];
				ku[0][2] = mag_ns.k_xz[index];
				ku[1][1] = mag_ns.k_yy[index];
				ku[1][2] = mag_ns.k_yz[index];
				ku[2][2] = mag_ns.k_zz[index];
				ku[1][0] = ku[0][1];
				ku[2][0] = ku[0][2];
				ku[2][1] = ku[1][2];

				Real K[3][3];
				// Lower the curvature indices: K_ab = g_{ac} g_{bd} k^{cd}.
				for (int a = 0; a < 3; a++) {
					for (int b = 0; b < 3; b++) {
						K[a][b] = 0.0;
						for (int c = 0; c < 3; c++) {
							for (int d = 0; d < 3; d++) {
								K[a][b] = g[a][c] * g[b][d] * ku[c][d];
							}
						}
					}
				}

				// Copy the temporary 3-metric and curvature tensors into the ADM variables.
				for (unsigned int a = 0; a < 3; a++) {
					for (unsigned int b = 0; b < 3; b++) {
						g_dd(a, b, k, j, i) = g[a][b];
						K_dd(a, b, k, j, i) = K[a][b];
					}
				}

				// Get the matter variables from Lorene.
				Real rho = mag_ns.nbar[index] / rho_unit; // Rest-mass density?
				Real eps = rho * mag_ns.ener_spec[index] / ener_unit; // Energy density
				// 3-velocity of the fluid.
				Real vu[3];
				vu[0] = mag_ns.u_euler_x[index] / vel_unit;
				vu[1] = mag_ns.u_euler_y[index] / vel_unit;
				vu[2] = mag_ns.u_euler_z[index] / vel_unit;

				// Calculate the pressure using the EOS.
				// FIXME: Is it possible not good practice to use this here? I had to modify
				// configure.py to set GENERAL_EOS_FILE to ideal.cpp for an adiabatic system
				// to get this to work.
				Real p = peos->PresFromRhoEg(rho, eps);

				// Find the four-velocity stored in the primitive variables.
				Real vsq = 2.0*(vu[0]*vu[1]*g[0][1] + vu[0]*vu[2]*g[0][2] + vu[1]*vu[2]*g[1][2])
						 + vu[0]*vu[0]*g[0][0] + vu[1]*vu[1]*g[1][1] + vu[2]*vu[2]*g[2][2];
				// Make sure that the velocity is physical. If not, scream that something is wrong
				// and quit.
				assert(vsq < 1.0);
				Real W = 1.0/sqrt(1.0 - vsq);

				// Magnetic field
				// FIXME: We don't currently do anything with the magnetic field other
				// than pull it in.
				Real Bu[3];
				Bu[0] = mag_ns.bb_x[i] / B_unit;
				Bu[1] = mag_ns.bb_y[i] / B_unit;
				Bu[2] = mag_ns.bb_z[i] / B_unit;

				// Stuff everything into the hydro variables.
				phydro->w(IDN, k, j, i) = phydro->w1(IDN, k, j, i) = rho;
				phydro->w(IPR, k, j, i) = phydro->w1(IPR, k, j, i) = p;
				phydro->w(IVX, k, j, i) = phydro->w1(IVX, k, j, i) = W*vu[0];
				phydro->w(IVY, k, j, i) = phydro->w1(IVY, k, j, i) = W*vu[1];
				phydro->w(IVZ, k, j, i) = phydro->w1(IVY, k, j, i) = W*vu[2];
			}
		}
	}

	// Construct Z4c vars from ADM vars.
	// FIXME: This needs to be made agnostic of the formalism.
	pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);
	pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u1);
	// Initialize the coordinates
	pcoord->UpdateMetric();
	if (pmy_mesh->multilevel) {
		pmr->pcoarsec->UpdateMetric();
	}
	// We've only set up the primitive variables; go ahead and initialize
	// the conserved variables.
	peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, il, iu, jl, ju, kl, ku);
	// Set up the matter tensor in the Z4c variables.
	pz4c->GetMatter(pz4c->storage.mat, pz4c->storage.adm, phydro->w);
    
	return;
}

/*int RefinementCondition(MeshBlock *pmb){
    
}*/
