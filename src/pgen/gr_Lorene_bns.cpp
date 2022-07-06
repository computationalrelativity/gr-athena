//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//! \file z4c_bns.cpp
//  \brief Initial conditions for binary neutron stars. Interpolation of Lorene
//         initial data.
//         Note: This is templated based on `gr_neutron_star.cpp`.

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


//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Sets the initial conditions.
//========================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin){
	// Interpolate Lorene data onto the grid.

  // constants ----------------------------------------------------------------

  // TODO: BD - These quantities would be good to collect to a single physical
  //            constants .hpp or so.

	// Some conversion factors to go between Lorene data and Athena data.
  // Shamelessly stolen from the Einstein Toolkit's Mag_NS.cc.
	Real const c_light = 299792458.0;              // Speed of light [m/s]
	Real const mu0 = 4.0 * M_PI * 1.0e-7;          // Vacuum permeability [N/A^2]
	Real const eps0 = 1.0 / (mu0 * std::pow(c_light, 2));

	// Constants of nature (IAU, CODATA):
	Real const G_grav = 6.67428e-11;       // Gravitational constant [m^3/kg/s^2]
	Real const M_sun = 1.98892e+30;        // Solar mass [kg]

	// Athena units in SI
	// Athena code units: c = G = 1, M = M_sun
	Real const athenaM = M_sun;
	Real const athenaL = athenaM * G_grav / (c_light * c_light);
	Real const athenaT = athenaL / c_light;
	// This is just a guess based on what ET uses.
	Real const athenaB = (1.0 / athenaL /
    std::sqrt(eps0 * G_grav / (c_light * c_light)));

	// Athena units for conversion.
	Real const coord_unit = athenaL/1.0e3; // Convert to km for Lorene.
	Real const rho_unit = athenaM/(athenaL*athenaL*athenaL); // kg/m^3.
	Real const ener_unit = 1.0; // c^2
	Real const vel_unit = athenaL / athenaT / c_light; // c
	Real const B_unit = athenaB / 1.0e+9; // 10^9 T

  // --------------------------------------------------------------------------

  // settings -----------------------------------------------------------------
  std::string fn_ini_data = pin->GetOrAddString(
    "problem", "filename", "resu.d");

  const double tol_det_zero = 1e-10;  // TODO: BD - this should go in .inp
  // --------------------------------------------------------------------------

	// Set some aliases for the variables.
	AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> alpha;
	AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> beta_u;
	AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> g_dd;
	AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> K_dd;

	alpha.InitWithShallowSlice( pz4c->storage.u,   Z4c::I_Z4c_alpha);
	beta_u.InitWithShallowSlice(pz4c->storage.u,   Z4c::I_Z4c_betax);
	g_dd.InitWithShallowSlice(  pz4c->storage.adm, Z4c::I_ADM_gxx);
	K_dd.InitWithShallowSlice(  pz4c->storage.adm, Z4c::I_ADM_Kxx);

  // use accumulators to avoid potential errors -------------------------------

  int npoints_vc = 0;

  for (int k = kms; k <= kpe; ++k)
  for (int j = jms; j <= jpe; ++j)
  for (int i = ims; i <= ipe; ++i)
  {
    ++npoints_vc;
  }

  int npoints_cc = 0;

  const int il = (block_size.nx1 > 1) ? is - NGHOST : is;
  const int iu = (block_size.nx1 > 1) ? ie + NGHOST : ie;

  const int jl = (block_size.nx2 > 1) ? js - NGHOST : js;
  const int ju = (block_size.nx2 > 1) ? je + NGHOST : je;

  const int kl = (block_size.nx3 > 1) ? ks - NGHOST : ks;
  const int ku = (block_size.nx3 > 1) ? ke + NGHOST : ke;


  for (int k = kl; k <= ku; ++k)
  for (int j = jl; j <= ju; ++j)
  for (int i = il; i <= iu; ++i)
  {
    ++npoints_cc;
  }


  #pragma omp critical
  {
    // prepare vc grid --------------------------------------------------------
    double * xx_vc = new double[npoints_vc];
    double * yy_vc = new double[npoints_vc];
    double * zz_vc = new double[npoints_vc];

    int I = 0;  // collapsed ijk index

    for (int k = kms; k <= kpe; ++k)
    for (int j = jms; j <= jpe; ++j)
    for (int i = ims; i <= ipe; ++i)
    {
      zz_vc[I] = coord_unit * pcoord->x3f(k);
      yy_vc[I] = coord_unit * pcoord->x2f(j);
      xx_vc[I] = coord_unit * pcoord->x1f(i);

      ++I;
    }
    // ------------------------------------------------------------------------

    // prepare Lorene interpolator (VC) ---------------------------------------
    pmy_mesh->bns = new Lorene::Bin_NS(npoints_vc, xx_vc, yy_vc, zz_vc,
                                       fn_ini_data.c_str());

    Lorene::Bin_NS * bns = pmy_mesh->bns;
    assert(bns->np == npoints_vc);

    I = 0;      // reset

    for (int k = kms; k <= kpe; ++k)
    for (int j = jms; j <= jpe; ++j)
    for (int i = ims; i <= ipe; ++i)
    {
      // Gauge from Lorene
      alpha(k, j, i) = bns->nnn[I];
      beta_u(0, k, j, i) = bns->beta_x[I];
      beta_u(1, k, j, i) = bns->beta_y[I];
      beta_u(2, k, j, i) = bns->beta_z[I];

      const double g_xx = bns->g_xx[I];
      const double g_xy = bns->g_xy[I];
      const double g_xz = bns->g_xz[I];
      const double g_yy = bns->g_yy[I];
      const double g_yz = bns->g_yz[I];
      const double g_zz = bns->g_zz[I];

      const double det = (
        -(SQR(g_xz) * g_yy) + 2 * g_xy * g_xz * g_yz
        - g_xx * SQR(g_yz) - SQR(g_xy) * g_zz
        + g_xx * g_yy * g_zz
      );

      assert(std::fabs(det) > tol_det_zero);

      // TODO: BD - Lorene header indicates that K_xx is covariant?
      // cf. gr_neutron_star
      g_dd(0, 0, k, j, i) = g_xx;
      K_dd(0, 0, k, j, i) = bns->k_xx[I];

      g_dd(0, 1, k, j, i) = g_xy;
      K_dd(0, 1, k, j, i) = bns->k_xy[I];

      g_dd(0, 2, k, j, i) = g_xz;
      K_dd(0, 2, k, j, i) = bns->k_xz[I];

      g_dd(1, 1, k, j, i) = g_yy;
      K_dd(1, 1, k, j, i) = bns->k_yy[I];

      g_dd(1, 2, k, j, i) = g_yz;
      K_dd(1, 2, k, j, i) = bns->k_yz[I];

      g_dd(2, 2, k, j, i) = g_zz;
      K_dd(2, 2, k, j, i) = bns->k_zz[I];

      ++I;
    }
    // ------------------------------------------------------------------------

    // clean up
    delete[] xx_vc;
    delete[] yy_vc;
    delete[] zz_vc;

    delete pmy_mesh->bns;


    // prepare cc grid --------------------------------------------------------
    double * xx_cc = new double[npoints_cc];
    double * yy_cc = new double[npoints_cc];
    double * zz_cc = new double[npoints_cc];

    I = 0;      // reset

    for (int k = kl; k <= ku; ++k)
    for (int j = jl; j <= ju; ++j)
    for (int i = il; i <= iu; ++i)
    {
      zz_cc[I] = coord_unit * pcoord->x3v(k);
      yy_cc[I] = coord_unit * pcoord->x2v(j);
      xx_cc[I] = coord_unit * pcoord->x1v(i);

      ++I;
    }
    // ------------------------------------------------------------------------


    // prepare Lorene interpolator (CC) ---------------------------------------
    pmy_mesh->bns = new Lorene::Bin_NS(npoints_cc, xx_cc, yy_cc, zz_cc,
                                       fn_ini_data.c_str());

    bns = pmy_mesh->bns;
    assert(bns->np == npoints_cc);

    I = 0;      // reset

    for (int k = kl; k <= ku; ++k)
    for (int j = jl; j <= ju; ++j)
    for (int i = il; i <= iu; ++i)
    {

      const Real rho = bns->nbar[I] / rho_unit;
      const Real eps = bns->ener_spec[I] / ener_unit;

      Real egas = rho * (1.0 + eps);
      Real pgas = peos->PresFromRhoEg(rho, egas);

      // Kludge to make the pressure work with the EOS framework.
      if (!std::isfinite(pgas) && (egas == 0. || rho == 0.)) {
        pgas = 0.;
      }

      // TODO: BD - Can we just get CC data directly from the additional Lorene
      //            interpolator?
      const Real u_E_x = bns->u_euler_x[I] / vel_unit;
      const Real u_E_y = bns->u_euler_y[I] / vel_unit;
      const Real u_E_z = bns->u_euler_z[I] / vel_unit;

      const Real vsq = (
        2.0*(u_E_x * u_E_y * bns->g_xy[I] +
             u_E_x * u_E_z * bns->g_xz[I] +
             u_E_y * u_E_z * bns->g_yz[I]) +
        u_E_x * u_E_x * bns->g_xx[I] +
        u_E_y * u_E_y * bns->g_yy[I] +
        u_E_z * u_E_z * bns->g_zz[I]
      );

      const Real W = 1.0 / std::sqrt(1.0 - vsq);

      // Copy all the variables over to Athena.
      phydro->w(IDN, k, j, i) = rho;
      phydro->w(IVX, k, j, i) = W * u_E_x;
      phydro->w(IVY, k, j, i) = W * u_E_y;
      phydro->w(IVZ, k, j, i) = W * u_E_z;
      phydro->w(IPR, k, j, i) = pgas;

      // FIXME: There needs to be a more consistent way to do this.
      peos->ApplyPrimitiveFloors(phydro->w, k, j, i);
      phydro->w1(IDN, k, j, i) = phydro->w(IDN, k, j, i);
      phydro->w1(IVX, k, j, i) = phydro->w(IVX, k, j, i);
      phydro->w1(IVY, k, j, i) = phydro->w(IVY, k, j, i);
      phydro->w1(IVZ, k, j, i) = phydro->w(IVZ, k, j, i);
      phydro->w1(IPR, k, j, i) = phydro->w(IPR, k, j, i);

      // TODO: BD - Magnetic fields...
      // ...

      ++I;
    }


    // clean up
    delete[] xx_cc;
    delete[] yy_cc;
    delete[] zz_cc;

    delete pmy_mesh->bns;
  }

  // --------------------------------------------------------------------------


  // Construct Z4c vars from ADM vars. ----------------------------------------
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u1);

  // Initialize the coordinates
  pcoord->UpdateMetric();
  if (pmy_mesh->multilevel)
  {
    pmr->pcoarsec->UpdateMetric();
  }

  // We've only set up the primitive variables; go ahead and initialize
  // the conserved variables.
  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord,
                             il, iu, jl, ju, kl, ku);

  // Check if the momentum and velocity are finite.
  // Set up the matter tensor in the Z4c variables.

  // TODO: BD - this needs to be fixed properly
  // No magnetic field, pass dummy or fix with overload
  AthenaArray<Real> null_bb_cc;

  pz4c->GetMatter(pz4c->storage.mat, pz4c->storage.adm, phydro->w, null_bb_cc);

  // --------------------------------------------------------------------------

	return;
}


int RefinementCondition(MeshBlock *pmb)
{
  return 0;
}