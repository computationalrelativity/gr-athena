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

inline Real VertexToCell(const Real *vec,
    int lbb, int rbb, int ltb, int rtb, int lbf, int rbf, int ltf, int rtf) {
  return 0.125*(vec[lbb] + vec[rbb] + vec[ltb] + vec[rtb] 
              + vec[lbf] + vec[rbf] + vec[ltf] + vec[rtf]);
}

inline void CalculateWv(Real uu[3], AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> g_dd,
    Real vu[3], int k, int j, int i) {
  Real vsq = 2.0*(vu[0]*vu[1]*g_dd(0,1,k,j,i) + vu[0]*vu[2]*g_dd(0,2,k,j,i) + 
                  vu[1]*vu[2]*g_dd(1,2,k,j,i))
           + vu[0]*vu[0]*g_dd(0,0,k,j,i) + vu[1]*vu[1]*g_dd(1,1,k,j,i) + vu[2]*vu[2]*g_dd(2,2,k,j,i);
  Real W = 1.0/std::sqrt(1.0 - vsq);
  uu[0] = vu[0]*W;
  uu[1] = vu[1]*W;
  uu[2] = vu[2]*W;
}

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

void MeshBlock::InitUserMeshBlockData(ParameterInput* pin) {
	AllocateUserOutputVariables(1);
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput* pin) {
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
	for (int k = kl; k <= ku; ++k) {
		for (int j = jl; j <= ju; ++j) {
			for (int i = il; i <= iu; ++i) {
				Real rho = phydro->w(IDN, k, j, i);
				Real p = phydro->w(IPR, k, j, i);
				// Kappa for P = kappa*rho^gamma. Here gamma = 2
				// is assumed.
				user_out_var(0, k, j, i) = p / (rho * rho);
			}
		}
	}
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Sets the initial conditions.
//========================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin){
	// Interpolate Lorene data onto the grid.

	// Some conversion factors to go between Lorene data and Athena data. Shamelessly stolen from
	// the Einstein Toolkit's Mag_NS.cc.
	Real const c_light = 299792458.0; // Speed of light [m/s]
	Real const mu0 = 4.0 * M_PI * 1.0e-7; // Vacuum permeability [N/A^2]
	Real const eps0 = 1.0 / (mu0 * pow(c_light, 2));

	// Constants of nature (IAU, CODATA):
	Real const G_grav = 6.67428e-11; // Gravitational constant [m^3/kg/s^2]
	Real const M_sun = 1.98892e+30; // Solar mass [kg]

	// Athena units in SI
	// Athena code units: c = G = 1, M = M_sun
	Real const athenaM = M_sun;
	Real const athenaL = athenaM * G_grav / (c_light * c_light);
	Real const athenaT = athenaL / c_light;
	// This is just a guess based on what ET uses.
	Real const athenaB = 1.0 / athenaL / sqrt(eps0 * G_grav / (c_light * c_light));

	// Athena units for conversion.
	Real const coord_unit = athenaL/1.0e3; // Convert to km for Lorene.
	Real const rho_unit = athenaM/(athenaL*athenaL*athenaL); // kg/m^3.
	Real const ener_unit = 1.0; // c^2
	Real const vel_unit = athenaL / athenaT / c_light; // c
	Real const B_unit = athenaB / 1.0e+9; // 10^9 T

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
	int const npoints = (nx + 1) * (ny + 1) * (nz + 1);
	std::vector<double> xx(npoints), yy(npoints), zz(npoints);

  // Calculate the vertex-centered coordinates.
	for (int k = kl; k <= ku + 1; k++) {
		for (int j = jl; j <= ju + 1; j++) {
			for (int i = il; i <= iu + 1; i++) {
				int index = (i - kl) + (nx + 1) * ((j - jl) + (ny + 1) * (k - kl));
				xx[index] = pcoord->x1f(i) * coord_unit;
				yy[index] = pcoord->x2f(j) * coord_unit;
				zz[index] = pcoord->x3f(k) * coord_unit;
			}
		}
	}

	// Read in the file from Lorene.
	std::string filename = pin->GetOrAddString("problem", "filename", "resu.d");
	Lorene::Mag_NS mag_ns(npoints, &xx[0], &yy[0], &zz[0], filename.c_str());
	assert(mag_ns.np == npoints);

  // Read in the metric.
	for (int k = kl; k <= ku + 1; k++) {
		for (int j = jl; j <= ju + 1; j++) {
			for (int i = il; i <= iu + 1; i++) {
				int index = (i - kl) + (nx + 1) * ((j - jl) + (ny + 1) * (k - kl));

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

				// Make sure that the metric isn't singular.
				Real det = g[0][0]*(g[1][1]*g[2][2] - g[1][2]*g[1][2])
					     - g[0][1]*(g[0][1]*g[2][2] - g[0][2]*g[1][2])
					     + g[0][2]*(g[0][1]*g[1][2] - g[1][1]*g[0][2]);
				assert(fabs(det) > 1e-10);

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
			}
		}
	}

  // Calculate the cell-centered fluid variables.
  for (int k = kl; k <= ku; k++) {
    for (int j = jl; j <= ju; j++) {
      for (int i = il; i <= iu; i++) {
        // Calculate indices for the points we're averaging.
        // Left-bottom-back
				int lbb = (i - kl) + (nx + 1) * ((j - jl) + (ny + 1) * (k - kl));
        // Right-bottom-back
        int rbb = (i - kl + 1) + (nx + 1) * ((j - jl) + (ny + 1) * (k - kl));
        // Left-top-back
        int ltb = (i - kl) + (nx + 1) * ((j - jl + 1) + (ny + 1) * (k - kl));
        // Right-top-back
        int rtb = (i - kl + 1) + (nx + 1) * ((j - jl + 1) + (ny + 1) * (k - kl));
        // Left-bottom-front
				int lbf = (i - kl) + (nx + 1) * ((j - jl) + (ny + 1) * (k - kl + 1));
        // Right-bottom-front
				int rbf = (i - kl + 1) + (nx + 1) * ((j - jl) + (ny + 1) * (k - kl + 1));
        // Left-top-front
				int ltf = (i - kl) + (nx + 1) * ((j - jl + 1) + (ny + 1) * (k - kl + 1));
        // Right-top-front
				int rtf = (i - kl + 1) + (nx + 1) * ((j - jl + 1) + (ny + 1) * (k - kl + 1));

        // Density and specific energy.
        Real rho = VertexToCell(mag_ns.nbar, lbb, rbb, ltb, rtb, lbf, rbf, ltf, rtf) / rho_unit;
        Real eps = VertexToCell(mag_ns.ener_spec, lbb, rbb, ltb, rtb, lbf, rbf, ltf, rtf) / ener_unit;
        Real egas = rho*(1.0 + eps);
        Real pgas = peos->PresFromRhoEg(rho, egas);
        // Kluge to make the pressure work with the EOS framework.
        if (!std::isfinite(pgas) && (egas == 0. || rho == 0.)) {
          pgas = 0.;
        }
        // 3-velocity of the fluid
        Real vu_lbb[3] = {mag_ns.u_euler_x[lbb]/vel_unit, mag_ns.u_euler_y[lbb]/vel_unit,
                          mag_ns.u_euler_z[lbb]/vel_unit};
        Real vu_rbb[3] = {mag_ns.u_euler_x[rbb]/vel_unit, mag_ns.u_euler_y[rbb]/vel_unit,
                          mag_ns.u_euler_z[rbb]/vel_unit};
        Real vu_ltb[3] = {mag_ns.u_euler_x[ltb]/vel_unit, mag_ns.u_euler_y[ltb]/vel_unit,
                          mag_ns.u_euler_z[ltb]/vel_unit};
        Real vu_rtb[3] = {mag_ns.u_euler_x[rtb]/vel_unit, mag_ns.u_euler_y[rtb]/vel_unit,
                          mag_ns.u_euler_z[rtb]/vel_unit};
        Real vu_lbf[3] = {mag_ns.u_euler_x[lbf]/vel_unit, mag_ns.u_euler_y[lbf]/vel_unit,
                          mag_ns.u_euler_z[lbf]/vel_unit};
        Real vu_rbf[3] = {mag_ns.u_euler_x[rbf]/vel_unit, mag_ns.u_euler_y[rbf]/vel_unit,
                          mag_ns.u_euler_z[rbf]/vel_unit};
        Real vu_ltf[3] = {mag_ns.u_euler_x[ltf]/vel_unit, mag_ns.u_euler_y[ltf]/vel_unit,
                          mag_ns.u_euler_z[ltf]/vel_unit};
        Real vu_rtf[3] = {mag_ns.u_euler_x[rtf]/vel_unit, mag_ns.u_euler_y[rtf]/vel_unit,
                          mag_ns.u_euler_z[rtf]/vel_unit};
        
        // 4-velocity of the fluid. We have to calculate the 4-velocities before averaging
        // because averaging 3-velocities is always a bad idea.
        Real uu_lbb[3], uu_rbb[3], uu_ltb[3], uu_rtb[3], 
             uu_lbf[3], uu_rbf[3], uu_ltf[3], uu_rtf[3];
        CalculateWv(uu_lbb, g_dd, vu_lbb, k, j, i);
        CalculateWv(uu_rbb, g_dd, vu_rbb, k, j, i);
        CalculateWv(uu_ltb, g_dd, vu_ltb, k, j, i);
        CalculateWv(uu_rtb, g_dd, vu_rtb, k, j, i);
        CalculateWv(uu_lbf, g_dd, vu_lbf, k, j, i);
        CalculateWv(uu_rbf, g_dd, vu_rbf, k, j, i);
        CalculateWv(uu_ltf, g_dd, vu_ltf, k, j, i);
        CalculateWv(uu_rtf, g_dd, vu_rtf, k, j, i);

        // Average the 4-velocities together.
        Real uu[3];
        uu[0] = 0.125*(uu_lbb[0] + uu_rbb[0] + uu_ltb[0] + uu_rtb[0]
                     + uu_lbf[0] + uu_rbf[0] + uu_ltf[0] + uu_rtf[0]);
        uu[1] = 0.125*(uu_lbb[1] + uu_rbb[1] + uu_ltb[1] + uu_rtb[1]
                     + uu_lbf[1] + uu_rbf[1] + uu_ltf[1] + uu_rtf[1]);
        uu[2] = 0.125*(uu_lbb[2] + uu_rbb[2] + uu_ltb[2] + uu_rtb[2]
                     + uu_lbf[2] + uu_rbf[2] + uu_ltf[2] + uu_rtf[2]);

        // FIXME: Need to load magnetic field at some point.

        // Copy all the variables over to Athena.
        phydro->w(IDN, k, j, i) = rho;
        phydro->w(IVX, k, j, i) = uu[0];
        phydro->w(IVY, k, j, i) = uu[1];
        phydro->w(IVZ, k, j, i) = uu[2];
        phydro->w(IPR, k, j, i) = pgas;

        // FIXME: There needs to be a more consistent way to do this.
        peos->ApplyPrimitiveFloors(phydro->w, k, j, i);
        phydro->w1(IDN, k, j, i) = phydro->w(IDN, k, j, i);
        phydro->w1(IVX, k, j, i) = phydro->w(IVX, k, j, i);
        phydro->w1(IVY, k, j, i) = phydro->w(IVY, k, j, i);
        phydro->w1(IVZ, k, j, i) = phydro->w(IVZ, k, j, i);
        phydro->w1(IPR, k, j, i) = phydro->w(IPR, k, j, i);
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
	// Check if the momentum and velocity are finite.
	// Set up the matter tensor in the Z4c variables.
	pz4c->GetMatter(pz4c->storage.mat, pz4c->storage.adm, phydro->w);
    
	return;
}

/*int RefinementCondition(MeshBlock *pmb){
    
}*/
