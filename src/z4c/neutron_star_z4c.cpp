//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file neutron_star_z4c.cpp
//  \brief implementation of functions in the Z4c class for initializing neutron star evolution

// C++ standard headers
#include <cmath> // pow
#include <assert.h>

// Athena++ headers
#include "z4c.hpp"
#include "z4c_macro.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"

// Lorene
#include <mag_ns.h>

//----------------------------------------------------------------------------------------
// \!fn void Z4c::ADMNeutronStar(ParameterInput* pin, AthenaArray<Real> & u_adm, AthenaArray<Real>& u_mat)
// \brief Initialize ADM vars to a single neutron star
void Z4c::ADMNeutronStar(ParameterInput* pin, AthenaArray<Real>& u_z4c, AthenaArray<Real>& u_adm, AthenaArray<Real>& w) {
	// Some conversion factors to go between Lorene data and Athena data. Shamelessly stolen from
	// the Einstein Toolkit's Mag_NS.cc.
	Real const c_light = 299792458.0; // Speed of light [m/s]
	Real const mu0     = 4.0 * M_PI * 1.0e-7; // Vacuum permeability [N/A^2]
	Real const eps0	   = 1.0 / (mu0 * pow(c_light, 2));

	// Constants of nature (IAU, CODATA);
	Real const G_grav = 6.67428e-11; // Gravitational constant [m^3/kg/s^2]
	Real const M_sun  = 1.98892e+30; // Solar mass [kg]

	// Maybe some stuff in here about Athena units...?
	Real const coord_unit = 1.0; // Placeholder -- figure out what this actually is.
	Real const rho_unit = 1.0; // Placeholder -- figure out what this actually is.
	Real const ener_unit = 1.0; // Placeholder -- figure out what this actually is.
	Real const vel_unit = 1.0; // Placeholder -- figure out what this actually is.
	Real const B_unit = 1.0; // Placeholder -- figure out what this actually is.

	Z4c_vars z4c;
	SetZ4cAliases(u_z4c, z4c);

	ADM_vars adm;
	SetADMAliases(u_adm, adm);
	//Real ADM_mass = pin->GetOrAddReal("problem", "punc_ADM_mass", 1.);

	// Flat vacuum spacetime
	ADMMinkowski(u_adm);

	// Coordinates for Lorene.
	unsigned int const nx = mbi.iu - mbi.il + 2 * GSIZEI + 1;
	unsigned int const ny = mbi.ju - mbi.jl + 2 * GSIZEJ + 1;
	unsigned int const nz = mbi.ku - mbi.kl + 2 * GSIZEK + 1;
	unsigned int const npoints = nx * ny * nz;
	std::vector<double> xx(npoints), yy(npoints), zz(npoints);

	unsigned int const offx = mbi.il - GSIZEI;
	unsigned int const offy = mbi.jl - GSIZEJ;
	unsigned int const offz = mbi.kl - GSIZEK;

	GLOOP2(k, j) {
		GLOOP1(i) {
			// Convert all the coordinates for Lorene. Lorene stores them in a flat array,
			// so we need to convert the three-index form to a single flat index.
			// The assumption is made that the index is stored in the form
			// index = i + nx*(j + ny*k).
			// FIXME: This should be verified.
			unsigned int index = (i - offx) + nx * ((j - offy) + ny * (k - offz));
			xx[index] = mbi.x1(i) * coord_unit;
			yy[index] = mbi.x2(j) * coord_unit;
			zz[index] = mbi.x3(k) * coord_unit;
		}
	}

	// Read in the file from Lorene.
	std::string filename = pin->GetOrAddString("problem", "filename", "resu.d");
	Lorene::Mag_NS mag_ns(npoints, &xx[0], &yy[0], &zz[0], filename.c_str());
	assert(mag_ns.np == npoints);

	GLOOP2(k, j) {
		GLOOP1(i) {
			unsigned int index = (i - offx) + nx * ((j - offy) + ny * (k - offz));

			// Copy over Lorene's gauge variables into the Z4c variables.
			// FIXME: Is there something else we should do here?
			z4c.alpha(k,j,i) = mag_ns.nnn[index];

			z4c.beta_u(0, k, j, i) = mag_ns.beta_x[index];
			z4c.beta_u(1, k, j, i) = mag_ns.beta_y[index];
			z4c.beta_u(2, k, j, i) = mag_ns.beta_z[index];

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
			// Lower the curvature indices: k_ab = g_{ac} g_{bd} k^{cd}.
			// FIXME: Is there a more concise to do this using built-in Athena functionality?
			for (unsigned int a = 0; a < 3; ++a) {
				for (unsigned int b = 0; b < 3; ++b) {
					K[a][b] = 0.0;
					for (unsigned int c = 0; c < 3; ++c) {
						for (unsigned int d = 0; d < 3; ++d) {
							K[a][b] += g[a][c] * g[b][d] * ku[c][d];
						}
					}
				}
			}

			// Copy the temporary 3-metric and curvature tensors into the ADM variables.
			for (unsigned int a = 0; a < 3; ++a) {
				for (unsigned int b = 0; b < 3; ++b) {
					adm.g_dd(a, b, k, j, i) = g[a][b];
					adm.K_dd(a, b, k, j, i) = K[a][b];
				}
			}

			// Get the matter variables from Lorene.
			Real rho = mag_ns.nbar[index] / rho_unit; // Rest-mass density?
			Real eps = rho * mag_ns.ener_spec[i] / ener_unit; // Energy density
			// 3-velocity of the fluid.
			Real vu[3];
			vu[0] = mag_ns.u_euler_x[i] / vel_unit;
			vu[1] = mag_ns.u_euler_y[i] / vel_unit;
			vu[2] = mag_ns.u_euler_z[i] / vel_unit;
			// Magnetic field
			// FIXME: We don't currently do anything with the magnetic field other
			// than pull it in.
			Real Bu[3];
			Bu[0] = mag_ns.bb_x[i] / B_unit;
			Bu[1] = mag_ns.bb_y[i] / B_unit;
			Bu[2] = mag_ns.bb_z[i] / B_unit;

		}
	}
}
