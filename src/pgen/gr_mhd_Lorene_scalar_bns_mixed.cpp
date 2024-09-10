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
#include <fstream>

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
#include "../mesh/mesh_refinement.hpp"
#include "../scalars/scalars.hpp"
#include "../utils/read_lorene.hpp"

#ifdef TRACKER_EXTREMA
#include "../trackers/tracker_extrema.hpp"
#endif // TRACKER_EXTREMA

#ifndef LORENE_EOS
#define LORENE_EOS (1)
#endif

int RefinementCondition(MeshBlock *pmb);
  Real Det3Metric(AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const & gamma,
                  int const i);
Real DivBface(MeshBlock *pmb, int iout);
Real DivBface_abs(MeshBlock *pmb, int iout);
Real DivBface_norm(MeshBlock *pmb, int iout);
Real Emag(MeshBlock *pmb, int iout);
Real MaxRho(MeshBlock *pmb, int iout);
Real MaxB(MeshBlock *pmb, int iout);
Real MinAlp(MeshBlock *pmb, int iout);
Real AtmosphereLoss(MeshBlock *pmb, int iout);

Real b_energy_bsq(MeshBlock *pmb, int iout);
  Real b_energy_bsq_star1(MeshBlock *pmb, int iout);
  Real b_energy_bsq_star2(MeshBlock *pmb, int iout);
  Real Maxbeta(MeshBlock *pmb, int iout);
  Real Minbeta(MeshBlock *pmb, int iout);
  Real Maxmag(MeshBlock *pmb, int iout);
  Real Minmag(MeshBlock *pmb, int iout);
  Real Maxbsq(MeshBlock *pmb, int iout);
  Real MaxB(MeshBlock *pmb, int iout);
  Real totalvol(MeshBlock *pmb, int iout);
  Real b_energy_bsq_pol(MeshBlock *pmb, int iout);
  Real b_energy_bsq_pol_star1(MeshBlock *pmb, int iout);
  Real b_energy_bsq_pol_star2(MeshBlock *pmb, int iout);
  Real B2_pol(MeshBlock *pmb, int iout);
  Real int_energy(MeshBlock *pmb, int iout);
  
double linear_interp(double *f, double *x, int n, double xv);
int interp_locate(Real *x, int Nx, Real xval);

namespace {
  // Global variables
  std::string table_fname;
  LoreneTable * Table = NULL;
  std::string filename, filename_Y;
  Real nsrad_m, nsrad_p;
  Real integral_rad_m, integral_rad_p;
}

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin, int res_flag) {

#ifdef LORENE_EOS
  filename = pin->GetString("hydro", "lorene");
  filename_Y = pin->GetString("hydro", "lorene_Y");
  Table = new LoreneTable;
  ReadLoreneTable(filename, Table);
  ReadLoreneFractions(filename_Y, Table);
  ConvertLoreneTable(Table);
  #if USETM
    Table->rho_atm = pin->GetReal("hydro", "dfloor");
  #endif
#endif
    integral_rad_m = pin->GetReal("problem","integral_rad_m");
    integral_rad_p = pin->GetReal("problem","integral_rad_p");
//  Real nsrad_m, nsrad_p;
//  nsrad_m = 0.0;
//  nsrad_p = 0.0;
    //  AllocateUserHistoryOutput(18);
  AllocateUserHistoryOutput(22);
  EnrollUserHistoryOutput(0, DivBface, "divBface");
  EnrollUserHistoryOutput(1, DivBface_abs, "divBface_abs");
  EnrollUserHistoryOutput(2, DivBface_norm, "divBface_norm");
  EnrollUserHistoryOutput(3, Emag, "Emag");
  EnrollUserHistoryOutput(4, AtmosphereLoss, "atm-loss");
  EnrollUserHistoryOutput(5, MaxRho, "MaxRho",UserHistoryOperation::max);
  EnrollUserHistoryOutput(6, MinAlp, "MinAlp",UserHistoryOperation::min);
  EnrollUserHistoryOutput(7, MaxB, "MaxB",UserHistoryOperation::max);

  EnrollUserHistoryOutput(8, b_energy_bsq, "b_energy_bsq"); // b^2*wlor
  EnrollUserHistoryOutput(9, Maxbeta, "max_beta", UserHistoryOperation::max); //average and max 2p/b^2

  EnrollUserHistoryOutput(10, Minbeta, "min_beta", UserHistoryOperation::min); //average and max
  EnrollUserHistoryOutput(11, Maxmag, "max_mganetisation" , UserHistoryOperation::max); //average and max 2p/b^2

  EnrollUserHistoryOutput(12, Minmag, "min_magnetisation", UserHistoryOperation::min); //average and max
  EnrollUserHistoryOutput(13, B2_pol, "B2_pol");

  EnrollUserHistoryOutput(14, b_energy_bsq_pol, "b_energy_bsq_pol");
  EnrollUserHistoryOutput(15, int_energy, "internal_energy"); //int D * eps d^3x (D vol weighted) = int W p sqrtdetgamma d^3x

  EnrollUserHistoryOutput(16, Maxbsq, "max_bsq", UserHistoryOperation::max); //max sqrt(b^mu b_mu)
  EnrollUserHistoryOutput(17, totalvol, "totalvol");

    EnrollUserHistoryOutput(18, b_energy_bsq_star1, "b_energy_bsq_star1"); // b^2*wlor
  EnrollUserHistoryOutput(19, b_energy_bsq_pol_star1, "b_energy_bsq_pol_star1");
  EnrollUserHistoryOutput(20, b_energy_bsq_star2, "b_energy_bsq_star2"); // b^2*wlor
  EnrollUserHistoryOutput(21, b_energy_bsq_pol_star2, "b_energy_bsq_pol_star2");

  if(adaptive==true)
    EnrollUserRefinementCondition(RefinementCondition);
  return;
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  // Allocate 3D output arrays
  AllocateUserOutputVariables(17);
  return;
}
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

  Real pgasmax_m, pgasmax_p,sep, mass_m, mass_p,centrex_m, centrex_p;
//  Real pgasmax, sep;
  //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //std::string filename_mpi = "file_" + std::to_string(rank) + ".out";
  //std::ofstream out(filename_mpi);
  // --------------------------------------------------------------------------

  // settings -----------------------------------------------------------------
  std::string fn_ini_data = pin->GetOrAddString("problem", "filename", "resu.d");

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
      beta_u(0, k, j, i) = -bns->beta_x[I];  // BAM inserts -1
      beta_u(1, k, j, i) = -bns->beta_y[I];
      beta_u(2, k, j, i) = -bns->beta_z[I];

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

      //assert(std::fabs(det) > tol_det_zero);

      // TODO: BD - Lorene header indicates that K_xx is covariant?
      // cf. gr_neutron_star
      g_dd(0, 0, k, j, i) = g_xx;
      K_dd(0, 0, k, j, i) = coord_unit * bns->k_xx[I];  // BAM has coord_unit

      g_dd(0, 1, k, j, i) = g_xy;
      K_dd(0, 1, k, j, i) = coord_unit * bns->k_xy[I];

      g_dd(0, 2, k, j, i) = g_xz;
      K_dd(0, 2, k, j, i) = coord_unit * bns->k_xz[I];

      g_dd(1, 1, k, j, i) = g_yy;
      K_dd(1, 1, k, j, i) = coord_unit * bns->k_yy[I];

      g_dd(1, 2, k, j, i) = g_yz;
      K_dd(1, 2, k, j, i) = coord_unit * bns->k_yz[I];

      g_dd(2, 2, k, j, i) = g_zz;
      K_dd(2, 2, k, j, i) = coord_unit * bns->k_zz[I];

      // debug: does it get interp prop.?
      // if (std::abs(K_dd(0, 0, k, j, i)) > 1e-5)
      // if ((-1e-10 < pcoord->x3f(k)) && (pcoord->x3f(k) < +1e-10))
      // if ((-1e-10 < pcoord->x2f(j)) && (pcoord->x2f(j) < +1e-10))
      // {
      //   std::cout << "is_larger" << std::endl;
      //   std::cout << K_dd(0, 0, k, j, i)  << std::endl;
      //   std::cout << pcoord->x3f(k) << ","  << pcoord->x2f(j) << ",";
      //   std::cout << pcoord->x1f(i)  << std::endl;
      //   Q();
      // }

      // // debug: does it get interp prop.?
      // if (std::abs(K_dd(0, 1, k, j, i)) > 1e-5)
      // if ((-1e-10 < pcoord->x3f(k)) && (pcoord->x3f(k) < +1e-10))
      // if ((-1e-10 < pcoord->x2f(j)) && (pcoord->x2f(j) < +1e-10))
      // {
      //   std::cout << "k_xy is_larger" << std::endl;
      //   std::cout << K_dd(0, 1, k, j, i)  << std::endl;
      //   std::cout << pcoord->x3f(k) << ","  << pcoord->x2f(j) << ",";
      //   std::cout << pcoord->x1f(i)  << std::endl;
      //   Q();
      // }

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
    sep = bns->dist/coord_unit;
    mass_m = bns->mass1_b;
    mass_p = bns->mass2_b;
    centrex_p =  sep*mass_m/((mass_m+mass_p));
    centrex_m =  -sep*mass_p/((mass_m+mass_p));
    assert(bns->np == npoints_cc);

    I = 0;      // reset

    for (int k = kl; k <= ku; ++k)
    for (int j = jl; j <= ju; ++j)
    for (int i = il; i <= iu; ++i)
    {
  
#if USETM
      if (NSCALARS == 1) {
        pscalars->s(0,k,j,i) = 0.;
        pscalars->r(0,k,j,i) = 0.;
      }
#endif

       Real u_E_x = 0.0;
       Real u_E_y = 0.0;
       Real u_E_z = 0.0;
       Real vsq = 0.0;
       Real W = 1.0;

       const Real x_star1 = bns->dist/coord_unit * 0.5;
       const Real r1 = std::sqrt(std::pow(pcoord->x1v(i) - x_star1, 2) +
                                 std::pow(pcoord->x2v(j), 2) +
                                 std::pow(pcoord->x3v(k), 2));
       const Real r2 = std::sqrt(std::pow(pcoord->x1v(i) + x_star1, 2) +
                                 std::pow(pcoord->x2v(j), 2) +
                                 std::pow(pcoord->x3v(k), 2));
       Real rho_atm = pin->GetReal("hydro", "dfloor");
       Real T_atm = pin->GetReal("hydro", "tfloor");
       Real Y_atm = pin->GetReal("hydro", "y0_atmosphere");
       Real mb = peos->GetEOS().GetBaryonMass();
       
       Real rho_kji = bns->nbar[I] / rho_unit;
       rho_kji = (rho_kji > rho_atm ? rho_kji : 0.0);
       Real n_kji = (rho_kji > rho_atm ? rho_kji/mb : rho_atm/mb);
       //Real eps_kji = bns->ener_spec[I] / ener_unit;
       //eps_kji = (eps_kji > 0. ? eps_kji : 0.);
       //egas_kji = rho_kji * eps_kji;
#if USETM
       Real Y_kji = (rho_kji > rho_atm ? linear_interp(Table->Y[0], Table->data[tab_logrho], Table->size, log(rho_kji)) : Y_atm);
       Real pgas_kji = peos->GetEOS().GetPressure(n_kji, T_atm, &Y_kji);
       pgas_kji = (rho_kji > rho_atm ? pgas_kji : 0.0);

#else
      // Real egas = rho * (1.0 + eps);  <-------- ?
        Real egas = rho * eps;
        Real pgas = peos->PresFromRhoEg(rho, egas);
#endif

       // Kludge to make the pressure work with the EOS framework.
       //if (!std::isfinite(pgas_kji) && rho_kji == 0.) {
       //  pgas_kji = 0.;
       //}

       // TODO: BD - Can we just get CC data directly from the additional Lorene
       //            interpolator?
       if (rho_kji > rho_atm) {
        u_E_x = bns->u_euler_x[I] / vel_unit;
        u_E_y = bns->u_euler_y[I] / vel_unit;
        u_E_z = bns->u_euler_z[I] / vel_unit;
       } else {
        u_E_x = 0.0;
        u_E_y = 0.0;
        u_E_z = 0.0;
       }
       vsq = (
               2.0*(u_E_x * u_E_y * bns->g_xy[I] +
               u_E_x * u_E_z * bns->g_xz[I] +
               u_E_y * u_E_z * bns->g_yz[I]) +
               u_E_x * u_E_x * bns->g_xx[I] +
               u_E_y * u_E_y * bns->g_yy[I] +
               u_E_z * u_E_z * bns->g_zz[I]
          );

      W = 1.0 / std::sqrt(1.0 - vsq);
      
//      out << "(ux, uy, uz) = " << u_E_x << ", " << u_E_y << ", " << u_E_z << std::endl;
      // Copy all the variables over to Athena.
      phydro->w(IDN, k, j, i) = rho_kji;
      phydro->w(IVX, k, j, i) = W * u_E_x;
      phydro->w(IVY, k, j, i) = W * u_E_y;
      phydro->w(IVZ, k, j, i) = W * u_E_z;
      phydro->w(IPR, k, j, i) = pgas_kji;
      if (NSCALARS == 1) {
        pscalars->r(0,k,j,i) = Y_kji;
      }

      // FIXME: There needs to be a more consistent way to do this.
#if USETM
  //    peos->ApplyPrimitiveFloors(phydro->w, pscalars->r, k, j, i);
#else
      peos->ApplyPrimitiveFloors(phydro->w, k, j, i);
#endif
      phydro->w1(IDN, k, j, i) = phydro->w(IDN, k, j, i);
      phydro->w1(IVX, k, j, i) = phydro->w(IVX, k, j, i);
      phydro->w1(IVY, k, j, i) = phydro->w(IVY, k, j, i);
      phydro->w1(IVZ, k, j, i) = phydro->w(IVZ, k, j, i);
      phydro->w1(IPR, k, j, i) = phydro->w(IPR, k, j, i);
      phydro->temperature(k,j,i) = T_atm;

      // TODO: BD - Magnetic fields...
      // ...
      //
      ++I;
    }

    nsrad_m = std::sqrt(SQR(bns->rad1_x_comp/coord_unit) + SQR(bns->rad1_y/coord_unit) + SQR(bns->rad1_z/coord_unit));
    nsrad_p = std::sqrt(SQR(bns->rad2_x_comp/coord_unit) + SQR(bns->rad2_y/coord_unit) + SQR(bns->rad2_z/coord_unit)); //TODO double check that 1 and 2 refer to m and p correctly. NB!!!! 1 always refers to less massive star

//    printf("centrex_p = %g, centrex_m = %g, nsrad_m = %g, nsrad_p = %g \n",centrex_p,centrex_m,nsrad_m,nsrad_p);
    // clean up
    delete[] xx_cc;
    delete[] yy_cc;
    delete[] zz_cc;

    delete pmy_mesh->bns;
  }
  // m refers to star at negative x coordinate, p to star at positive x coordinate

  pgasmax_m = pin->GetReal("problem","pmax_m");
  pgasmax_p = pin->GetReal("problem","pmax_p");
  Real pcut_m = pin->GetReal("problem","pcut_m")*pgasmax_m;
  Real pcut_p = pin->GetReal("problem","pcut_p")*pgasmax_p;
  int magindex_m = pin->GetInteger("problem","magindex_m");
  int magindex_p = pin->GetInteger("problem","magindex_p");
  Real ns_m = pin->GetReal("problem","ns_m");
  Real ns_p = pin->GetReal("problem","ns_p");
  Real tor_radcut_m = pin->GetReal("problem","tor_radcut_m");
  Real tor_radcut_p = pin->GetReal("problem","tor_radcut_p");
  Real b_polrat_m = 1.0 - std::sqrt( 1.0 - pin->GetReal("problem","b_polrat_m") );
  Real b_polrat_p = 1.0 - std::sqrt( 1.0 - pin->GetReal("problem","b_polrat_p") );  
  Real bpol_amp_m = pin->GetReal("problem","b_amp_m")*0.5*b_polrat_m/std::pow(pgasmax_m-pcut_m, ns_m)/8.351416e19;
  Real bpol_amp_p = pin->GetReal("problem","b_amp_p")*0.5*b_polrat_p/std::pow(pgasmax_p-pcut_p, ns_p)/8.351416e19;
  Real btor_amp_m = pin->GetReal("problem","b_amp_m")*0.5*(1.0-b_polrat_m)/std::pow(pgasmax_m-pcut_m, ns_m)/std::pow(nsrad_m,2)/8.351416e19;
  Real btor_amp_p = pin->GetReal("problem","b_amp_p")*0.5*(1.0-b_polrat_p)/std::pow(pgasmax_p-pcut_p, ns_p)/std::pow(nsrad_p,2)/8.351416e19;
  std::cout << "bpol_amp_m = " << bpol_amp_m << std::endl;
  std::cout << "bpol_amp_p = " << bpol_amp_p << std::endl;
  std::cout << "btor_amp_m = " << btor_amp_m << std::endl;
  std::cout << "btor_amp_p = " << btor_amp_p << std::endl;

  AthenaArray<Real> bxcc,bycc,bzcc;
  int nx1 = (ie-is)+1 + 2*(NGHOST);
  int nx2 = (je-js)+1 + 2*(NGHOST);
  int nx3 = (ke-ks)+1 + 2*(NGHOST);
  pfield->b.x1f.ZeroClear();
  pfield->b.x2f.ZeroClear();
  pfield->b.x3f.ZeroClear();
  pfield->bcc.ZeroClear();
  bxcc.NewAthenaArray(nx3,nx2,nx1);
  bycc.NewAthenaArray(nx3,nx2,nx1);
  bzcc.NewAthenaArray(nx3,nx2,nx1);
  AthenaArray<Real> vcgamma_xx,vcgamma_xy,vcgamma_xz,vcgamma_yy;
  AthenaArray<Real> vcgamma_yz,vcgamma_zz;
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> gamma_dd;
  gamma_dd.NewAthenaTensor(iu+1);
  vcgamma_xx.InitWithShallowSlice(pz4c->storage.adm,Z4c::I_ADM_gxx,1);
  vcgamma_xy.InitWithShallowSlice(pz4c->storage.adm,Z4c::I_ADM_gxy,1);
  vcgamma_xz.InitWithShallowSlice(pz4c->storage.adm,Z4c::I_ADM_gxz,1);
  vcgamma_yy.InitWithShallowSlice(pz4c->storage.adm,Z4c::I_ADM_gyy,1);
  vcgamma_yz.InitWithShallowSlice(pz4c->storage.adm,Z4c::I_ADM_gyz,1);
  vcgamma_zz.InitWithShallowSlice(pz4c->storage.adm,Z4c::I_ADM_gzz,1);

  AthenaArray<Real> Atot;
  Atot.NewAthenaArray(3,nx3,nx2,nx1);

  Atot.ZeroClear();

  Real bpol_amp,pcut,ns,tor_radcut,nsrad,offset,btor_amp,r;

  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        if (pcoord->x1v(i) > 0) {
           bpol_amp = bpol_amp_p;
	   pcut = pcut_p;
	   ns = ns_p;
	   tor_radcut = tor_radcut_p;
	   nsrad = nsrad_p;
	   offset = centrex_p;
           btor_amp = btor_amp_p;
	} else {
           bpol_amp = bpol_amp_m;
	   pcut = pcut_m;
	   ns = ns_m;
	   tor_radcut = tor_radcut_m;
	   nsrad = nsrad_m;
	   offset = centrex_m;
           btor_amp = btor_amp_m;
	}
        Atot(0,k,j,i) = -pcoord->x2v(j) * bpol_amp * std::pow(std::max(phydro->w(IPR,k,j,i) - pcut, 0.0), ns);
        Atot(1,k,j,i) = (pcoord->x1v(i) - offset) * bpol_amp * std::pow(std::max(phydro->w(IPR,k,j,i) - pcut, 0.0), ns); 
        Atot(2,k,j,i) = 0.0;
        r = std::sqrt(SQR(pcoord->x1v(i) - offset) + SQR(pcoord->x2v(j)) + SQR(pcoord->x3v(k)));
            Atot(0,k,j,i) += (pcoord->x1v(i) - offset)* (SQR(pcoord->x3v(k)) - SQR(nsrad)) * btor_amp * std::pow(std::max(phydro->w(IPR,k,j,i) - pcut, 0.0), ns);
            Atot(1,k,j,i) += (pcoord->x2v(j))* (SQR(pcoord->x3v(k)) - SQR(nsrad)) * btor_amp * std::pow(std::max(phydro->w(IPR,k,j,i) - pcut, 0.0), ns);
            Atot(2,k,j,i) += -(pcoord->x3v(k))* (SQR(pcoord->x1v(i) - offset  )  + SQR(pcoord->x2v(j)) - SQR(nsrad)) * btor_amp * std::pow(std::max(phydro->w(IPR,k,j,i) - pcut, 0.0), ns);

      }
    }
  }
 



  for (int k=ks-1; k<=ke+1; k++) {
    for (int j=js-1; j<=je+1; j++) {
      for (int i=is-1; i<=ie+1; i++) {
        bxcc(k,j,i) =  (Atot(2,k,j+1,i) - Atot(2,k,j-1,i))/(2.0*pcoord->dx2v(j))  - (Atot(1,k+1,j,i) - Atot(1,k-1,j,i))/(2.0*pcoord->dx3v(k));
        bycc(k,j,i) =  (Atot(0,k+1,j,i) - Atot(0,k-1,j,i))/(2.0*pcoord->dx3v(k))   - (Atot(2,k,j,i+1) - Atot(2,k,j,i-1))/(2.0*pcoord->dx1v(i))  ;
        bzcc(k,j,i) =  (Atot(1,k,j,i+1) - Atot(1,k,j,i-1))/(2.0*pcoord->dx1v(i))
                   - (Atot(0,k,j+1,i) - Atot(0,k,j-1,i))/(2.0*pcoord->dx2v(j));
      }
    }
  }
  for(int k = ks; k<=ke; k++){
    for(int j = js; j<=je; j++){
      for(int i = is; i<=ie+1; i++){
        pfield->b.x1f(k,j,i) = 0.5*(bxcc(k,j,i-1) + bxcc(k,j,i));
      }
    }
  }
  for(int k = ks; k<=ke; k++){
    for(int j = js; j<=je+1; j++){
      for(int i = is; i<=ie; i++){
        pfield->b.x2f(k,j,i) = 0.5*(bycc(k,j-1,i) + bycc(k,j,i));
      }
    }
  }
  for(int k = ks; k<=ke+1; k++){
    for(int j = js; j<=je; j++){
      for(int i = is; i<=ie; i++){
        pfield->b.x3f(k,j,i) = 0.5*(bzcc(k-1,j,i) + bzcc(k,j,i));
      }
    }
  }

  pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, il,iu,jl,ju,kl,ku);



  // --------------------------------------------------------------------------


  // Construct Z4c vars from ADM vars. ----------------------------------------
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u1);
//  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u_init);

  // Initialize the coordinates
  pcoord->UpdateMetric();
  if (pmy_mesh->multilevel)
  {
    pmr->pcoarsec->UpdateMetric();
  }

  // We've only set up the primitive variables; go ahead and initialize
  // the conserved variables.
#if USETM
  peos->PrimitiveToConserved(phydro->w, pscalars->r, pfield->bcc, phydro->u, pscalars->s, pcoord,
                             il, iu, jl, ju, kl, ku);
#else
  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord,
                             il, iu, jl, ju, kl, ku);
#endif

  // Check if the momentum and velocity are finite.
  // TODO: BD - this needs to be fixed properly
  // No magnetic field, pass dummy or fix with overload

#if USETM
  pz4c->GetMatter(pz4c->storage.mat, pz4c->storage.adm, phydro->w, pscalars->r, pfield->bcc);
#else
  pz4c->GetMatter(pz4c->storage.mat, pz4c->storage.adm, phydro->w, null_bb_cc);
#endif

//  out << "After GetMatter" << std::endl;

//  out.close();
  // --------------------------------------------------------------------------

	return;
}

void MeshBlock::UserWorkInLoop() {

  int nn1 = ie+1;
  int a,b;
  AthenaArray<Real> vcgamma_xx,vcgamma_xy,vcgamma_xz,vcgamma_yy;
  AthenaArray<Real> vcgamma_yz,vcgamma_zz,vcbeta_x,vcbeta_y;
  AthenaArray<Real> vcbeta_z, vcalpha;

  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> alpha, rho, pgas, wgas, detgamma, sqrtdetgamma, Wlor, u0, v2, b0_u, bsq, u1, u2, u3, Bmod; //lapse
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> beta_u, utilde_u, utilde_d, v_u, v_d, bb_u, bi_u, bi_d, beta_d; //lapse
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> gamma_dd, gamma_uu; //lapse

  alpha.NewAthenaTensor(nn1);
  beta_u.NewAthenaTensor(nn1);
  beta_d.NewAthenaTensor(nn1);
  gamma_dd.NewAthenaTensor(nn1);

  detgamma.NewAthenaTensor(nn1);
  sqrtdetgamma.NewAthenaTensor(nn1);
  rho.NewAthenaTensor(nn1);
  pgas.NewAthenaTensor(nn1);
  wgas.NewAthenaTensor(nn1);
  Wlor.NewAthenaTensor(nn1);
  u0.NewAthenaTensor(nn1);
  u1.NewAthenaTensor(nn1);
  u2.NewAthenaTensor(nn1);
  u3.NewAthenaTensor(nn1);
  v2.NewAthenaTensor(nn1);
  utilde_u.NewAthenaTensor(nn1);
  v_u.NewAthenaTensor(nn1);
  v_d.NewAthenaTensor(nn1);
  utilde_d.NewAthenaTensor(nn1);
  gamma_uu.NewAthenaTensor(nn1);
  b0_u.NewAthenaTensor(nn1);
  bsq.NewAthenaTensor(nn1);
  Bmod.NewAthenaTensor(nn1);
  bb_u.NewAthenaTensor(nn1);
  bi_u.NewAthenaTensor(nn1);
  bi_d.NewAthenaTensor(nn1);



  vcgamma_xx.InitWithShallowSlice(pz4c->storage.adm,Z4c::I_ADM_gxx,1);
  vcgamma_xy.InitWithShallowSlice(pz4c->storage.adm,Z4c::I_ADM_gxy,1);
  vcgamma_xz.InitWithShallowSlice(pz4c->storage.adm,Z4c::I_ADM_gxz,1);
  vcgamma_yy.InitWithShallowSlice(pz4c->storage.adm,Z4c::I_ADM_gyy,1);
  vcgamma_yz.InitWithShallowSlice(pz4c->storage.adm,Z4c::I_ADM_gyz,1);
  vcgamma_zz.InitWithShallowSlice(pz4c->storage.adm,Z4c::I_ADM_gzz,1);
  vcbeta_x.InitWithShallowSlice(pz4c->storage.u,Z4c::I_Z4c_betax,1);
  vcbeta_y.InitWithShallowSlice(pz4c->storage.u,Z4c::I_Z4c_betay,1);
  vcbeta_z.InitWithShallowSlice(pz4c->storage.u,Z4c::I_Z4c_betaz,1);
  vcalpha.InitWithShallowSlice(pz4c->storage.u,Z4c::I_Z4c_alpha,1);




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
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {      
      for (int i=is; i<=ie; ++i) {
        gamma_dd(0,0,i) = VCInterpolation(vcgamma_xx, k, j, i);
        gamma_dd(0,1,i) = VCInterpolation(vcgamma_xy, k, j, i);
        gamma_dd(0,2,i) = VCInterpolation(vcgamma_xz, k, j, i);
        gamma_dd(1,1,i) = VCInterpolation(vcgamma_yy, k, j, i);
        gamma_dd(1,2,i) = VCInterpolation(vcgamma_yz, k, j, i);
        gamma_dd(2,2,i) = VCInterpolation(vcgamma_zz, k, j, i);
        alpha(i) = VCInterpolation(vcalpha, k, j, i);
        beta_u(0,i) = VCInterpolation(vcbeta_x, k, j, i);
        beta_u(1,i) = VCInterpolation(vcbeta_y, k, j, i);
        beta_u(2,i) = VCInterpolation(vcbeta_z, k, j, i);
  
        detgamma(i) = Det3Metric(gamma_dd, i);
        sqrtdetgamma(i) = std::sqrt(detgamma(i));
        detgamma(i) = Det3Metric(gamma_dd, i);
        sqrtdetgamma(i) = std::sqrt(detgamma(i));
      }

      for (a=0;a<NDIM;++a) {
        for (int i=is; i<=ie; ++i) {
          utilde_u(a,i) = phydro->w(a+IVX,k,j,i);
         }
      }
      for (int i=is; i<=ie; ++i) {
        pgas(i) = phydro->w(IPR,k,j,i);
        rho(i)  = phydro->w(IDN,k,j,i);
      }


     for (int i=is; i<=ie; ++i) {
       bb_u(0,i) = pfield->bcc(IB1,k,j,i)/sqrtdetgamma(i);
       bb_u(1,i) = pfield->bcc(IB2,k,j,i)/sqrtdetgamma(i);
       bb_u(2,i) = pfield->bcc(IB3,k,j,i)/sqrtdetgamma(i);
     }

     Wlor.ZeroClear();
     for(a=0;a<NDIM;++a){
       for(b=0;b<NDIM;++b){
         for (int i=is; i<=ie; ++i) {
           Wlor(i) += utilde_u(a,i)*utilde_u(b,i)*gamma_dd(a,b,i);
         }
       }
     }
     for (int i=is; i<=ie; ++i) {
       Wlor(i) = std::sqrt(1.0+Wlor(i));
     }


     beta_d.ZeroClear();
     for(a=0;a<NDIM;++a){
       for(b=0;b<NDIM;++b){
         for (int i=is; i<=ie; ++i) {
           beta_d(a,i) += gamma_dd(a,b,i)*beta_u(b,i);
         }
       }
     }


     for(a=0;a<NDIM;++a){
       for (int i=is; i<=ie; ++i) {
         v_u(a,i) = utilde_u(a,i)/Wlor(i);
       }
     }

     v_d.ZeroClear();
     for(a=0;a<NDIM;++a){
       for(b=0;b<NDIM;++b){
         for (int i=is; i<=ie; ++i) {
            v_d(a,i) += v_u(b,i)*gamma_dd(a,b,i);
         }
       }
     }

      b0_u.ZeroClear();
      for(a=0;a<NDIM;++a){
        for (int i=is; i<=ie; ++i) {
          b0_u(i) += Wlor(i)*bb_u(a,i)*v_d(a,i)/alpha(i);
        }
      }
      for(a=0;a<NDIM;++a){
        for (int i=is; i<=ie; ++i) {
          bi_u(a,i) = (bb_u(a,i) + alpha(i)*b0_u(i)*Wlor(i)*(v_u(a,i) - beta_u(a,i)/alpha(i)))/Wlor(i);
        }
      }

      for(a=0;a<NDIM;++a){
        for (int i=is; i<=ie; ++i) {
          bi_d(a,i) = beta_d(a,i) * b0_u(i);
        }
        for(b=0;b<NDIM;++b){
          for (int i=is; i<=ie; ++i) {
            bi_d(a,i) += gamma_dd(a,b,i)*bi_u(b,i);
          }
        }
      }
      for (int i=is; i<=ie; ++i) {
        bsq(i) = alpha(i)*alpha(i)*b0_u(i)*b0_u(i)/(Wlor(i)*Wlor(i));
      }
      for(a=0;a<NDIM;++a){
        for(b=0;b<NDIM;++b){
          for (int i=is; i<=ie; ++i) {
            bsq(i) += bb_u(a,i)*bb_u(b,i)*gamma_dd(a,b,i)/(Wlor(i)*Wlor(i));
          } 
	}
      }
      for (int i=is; i<=ie; ++i) {
        u0(i) = Wlor(i)/alpha(i);
      }

      Bmod.ZeroClear();
      for(a=0;a<NDIM;++a){
        for(b=0;b<NDIM;++b){
          for (int i=is; i<=ie; ++i) {
            Bmod(i) += bb_u(a,i)*bb_u(b,i)*gamma_dd(a,b,i);
          }
        }
      }
      for (int i=is; i<=ie; ++i) {
        Bmod(i) = std::sqrt(Bmod(i));
      }
      for (int i=is; i<=ie; ++i) {
        Real mb = peos->GetEOS().GetBaryonMass();
	Real npt =  phydro->w(IDN,k,j,i) / mb;
	Real T = phydro->temperature(k,j,i);
	Real Ypt[NSCALARS];
        for (int n=0; n<NSCALARS; ++n){
          Ypt[n] = pscalars->r(n,k,j,i);
        }
	user_out_var(0,k,j,i) = b0_u(i);
        user_out_var(1,k,j,i) = bi_u(0,i);
        user_out_var(2,k,j,i) = bi_u(1,i);
        user_out_var(3,k,j,i) = bi_u(2,i);
        user_out_var(4,k,j,i) = bsq(i);
        user_out_var(5,k,j,i) = Wlor(i);
        user_out_var(6,k,j,i) = (- bi_u(0,i) * pcoord->x2v(j) + bi_u(1,i) * pcoord->x1v(i) ) / ( std::sqrt(SQR(pcoord->x1v(i)) + SQR(pcoord->x2v(j)))  );
        user_out_var(7,k,j,i) = (- bi_d(0,i) * pcoord->x2v(j) + bi_d(1,i) * pcoord->x1v(i) ) / ( std::sqrt(SQR(pcoord->x1v(i)) + SQR(pcoord->x2v(j)))  );
        user_out_var(8,k,j,i) = (- pfield->bcc(IB1,k,j,i) * pcoord->x2v(j) + pfield->bcc(IB2,k,j,i) * pcoord->x1v(i) ) / ( std::sqrt(SQR(pcoord->x1v(i)) + SQR(pcoord->x2v(j)))  );
        user_out_var(9,k,j,i) = sqrtdetgamma(i);
        user_out_var(10,k,j,i) = Bmod(i);
        user_out_var(11,k,j,i) = bi_d(0,i);
        user_out_var(12,k,j,i) = bi_d(1,i);
        user_out_var(13,k,j,i) = bi_d(2,i);
        user_out_var(14,k,j,i) = peos->GetEOS().GetSpecificInternalEnergy(npt, T, Ypt);
        user_out_var(15,k,j,i) = T;
        user_out_var(16,k,j,i) = peos->GetEOS().GetEntropy(npt, T, Ypt);
        }
      }
    }






    alpha.DeleteAthenaTensor();
    beta_u.DeleteAthenaTensor();
    beta_d.DeleteAthenaTensor();
    gamma_dd.DeleteAthenaTensor();
    detgamma.DeleteAthenaTensor();
    sqrtdetgamma.DeleteAthenaTensor();
    rho.DeleteAthenaTensor();
    pgas.DeleteAthenaTensor();
    wgas.DeleteAthenaTensor();
    Wlor.DeleteAthenaTensor();
    u0.DeleteAthenaTensor();
    u1.DeleteAthenaTensor();
    u2.DeleteAthenaTensor();
    u3.DeleteAthenaTensor();
    v2.DeleteAthenaTensor();
    utilde_u.DeleteAthenaTensor();
    v_u.DeleteAthenaTensor();
    v_d.DeleteAthenaTensor();
    utilde_d.DeleteAthenaTensor();
    gamma_uu.DeleteAthenaTensor();
    b0_u.DeleteAthenaTensor();
    bsq.DeleteAthenaTensor();
    Bmod.DeleteAthenaTensor();
    bb_u.DeleteAthenaTensor();
    bi_u.DeleteAthenaTensor();
    bi_d.DeleteAthenaTensor();

    return;

}
int RefinementCondition(MeshBlock *pmb)
{

#ifdef TRACKER_EXTREMA

  Mesh * pmesh = pmb->pmy_mesh;
  TrackerExtrema * ptracker_extrema = pmesh->ptracker_extrema;

  int root_level = ptracker_extrema->root_level;
  int mb_physical_level = pmb->loc.level - root_level;

  // to get behaviour correct for when multiple centres occur in a single
  // MeshBlock we need to carry information
  bool centres_contained = false;

  for (int n=1; n<=ptracker_extrema->N_tracker; ++n)
  {
    bool is_contained = false;

    if (ptracker_extrema->ref_type(n-1) == 0)
    {
      is_contained = pmb->PointContained(
        ptracker_extrema->c_x1(n-1),
        ptracker_extrema->c_x2(n-1),
        ptracker_extrema->c_x3(n-1)
      );
    }
    else if (ptracker_extrema->ref_type(n-1) == 1)
    {
      is_contained = pmb->SphereIntersects(
        ptracker_extrema->c_x1(n-1),
        ptracker_extrema->c_x2(n-1),
        ptracker_extrema->c_x3(n-1),
        ptracker_extrema->ref_zone_radius(n-1)
      );

      // is_contained = pmb->PointContained(
      //   ptracker_extrema->c_x1(n-1),
      //   ptracker_extrema->c_x2(n-1),
      //   ptracker_extrema->c_x3(n-1)
      // ) or pmb->PointCentralDistanceSquared(
      //   ptracker_extrema->c_x1(n-1),
      //   ptracker_extrema->c_x2(n-1),
      //   ptracker_extrema->c_x3(n-1)
      // ) < SQR(ptracker_extrema->ref_zone_radius(n-1));

    }

    if (is_contained)
    {
      centres_contained = true;

      // a point in current MeshBlock, now check whether level sufficient
      if (mb_physical_level < ptracker_extrema->ref_level(n-1))
      {
        return 1;
      }

    }
  }

  // Here one could put composite criteria (such as spherical patch cond.)
  // ...

  if (centres_contained)
  {
    // all contained centres are at a sufficient level of refinement
    return 0;
  }

  // Nothing satisfied - flag for de-refinement
  return -1;

#endif // TRACKER_EXTREMA
 return 0;
}

  //--------------------------------------------------------------------------------------
  //! \fn double linear_interp(double *f, double *x, int n, double xv)
  // \brief linearly interpolate f(x), compute f(xv)
  double linear_interp(double *f, double *x, int n, double xv)
  {
    int i = interp_locate(x,n,xv);
    if (i < 0)  i=1;
    if (i == n) i=n-1;
    int j;
    if(xv < x[i]) j = i-1;
    else j = i+1;
    double xj = x[j]; double xi = x[i];
    double fj = f[j]; double fi = f[i];
    double m = (fj-fi)/(xj-xi);
    double df = m*(xv-xj)+fj;
    return df;
  }

  //-----------------------------------------------------------------------------------------
  //! \fn int interp_locate(Real *x, int Nx, Real xval)
  // \brief Bisection to find closest point in interpolating table
  // 
  int interp_locate(Real *x, int Nx, Real xval) {
    int ju,jm,jl;
    int ascnd;
    jl=-1;
    ju=Nx;
    if (xval <= x[0]) {
      return 0;
    } else if (xval >= x[Nx-1]) {
      return Nx-1;
    }
    ascnd = (x[Nx-1] >= x[0]);
    while (ju-jl > 1) {
      jm = (ju+jl) >> 1;
      if (xval >= x[jm] == ascnd)
  jl=jm;
      else
  ju=jm;
    }
    return jl;
  }


Real DivBface(MeshBlock *pmb, int iout) {
  Real divB = 0.0;
  Real vol,dx,dy,dz;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        dx = pmb->pcoord->dx1v(i);
        dy = pmb->pcoord->dx2v(j);
        dz = pmb->pcoord->dx3v(k);
        vol = dx*dy*dz;
        divB += ((pmb->pfield->b.x1f(k,j,i+1) - pmb->pfield->b.x1f(k,j,i))/dx + (pmb->pfield->b.x2f(k,j+1,i) - pmb->pfield->b.x2f(k,j,i))/(dy) + (pmb->pfield->b.x3f(k+1,j,i) - pmb->pfield->b.x3f(k,j,i))/(dz))*vol;
      }
    }
  }
  return divB;
}

Real Det3Metric(AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const & gamma,
                  int const i)
{
  return - SQR(gamma(0,2,i))*gamma(1,1,i) +
          2*gamma(0,1,i)*gamma(0,2,i)*gamma(1,2,i) -
          gamma(0,0,i)*SQR(gamma(1,2,i)) - SQR(gamma(0,1,i))*gamma(2,2,i) +
          gamma(0,0,i)*gamma(1,1,i)*gamma(2,2,i);
}

Real SpatialDet(Real gxx, Real gxy, Real gxz, Real gyy, Real gyz, Real gzz)
{
  return - SQR(gxz)*gyy+
          2*gxy*gxz*gyz -
          gxx*SQR(gyz) - SQR(gxy)*gzz +
          gxx*gyy*gzz;
}

Real DivBface_abs(MeshBlock *pmb, int iout) {
  Real divB = 0.0;
  Real vol,dx,dy,dz;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        dx = pmb->pcoord->dx1v(i);
        dy = pmb->pcoord->dx2v(j);
        dz = pmb->pcoord->dx3v(k);
        vol = dx*dy*dz;
        divB += std::abs((pmb->pfield->b.x1f(k,j,i+1) - pmb->pfield->b.x1f(k,j,i))/dx + (pmb->pfield->b.x2f(k,j+1,i) - pmb->pfield->b.x2f(k,j,i))/(dy) + (pmb->pfield->b.x3f(k+1,j,i) - pmb->pfield->b.x3f(k,j,i))/(dz))*vol;
      }
    }
  }
  return divB;
}


Real DivBface_norm(MeshBlock *pmb, int iout) {
  Real divB = 0.0;
  Real B2 = 0.0;
  Real vol,dx,dy,dz;
  AthenaArray<Real> vcgamma_xx, vcgamma_xy, vcgamma_xz,
                    vcgamma_yy, vcgamma_yz, vcgamma_zz;
  Real gxx, gyy, gzz, gxy, gxz, gyz;
  Real Bx, By, Bz;
  vcgamma_xx.InitWithShallowSlice(pmb->pz4c->storage.adm,Z4c::I_ADM_gxx,1);
  vcgamma_xy.InitWithShallowSlice(pmb->pz4c->storage.adm,Z4c::I_ADM_gxy,1);
  vcgamma_xz.InitWithShallowSlice(pmb->pz4c->storage.adm,Z4c::I_ADM_gxz,1);
  vcgamma_yy.InitWithShallowSlice(pmb->pz4c->storage.adm,Z4c::I_ADM_gyy,1);
  vcgamma_yz.InitWithShallowSlice(pmb->pz4c->storage.adm,Z4c::I_ADM_gyz,1);
  vcgamma_zz.InitWithShallowSlice(pmb->pz4c->storage.adm,Z4c::I_ADM_gzz,1);
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        gxx = VCInterpolation(vcgamma_xx, k, j, i);
        gyy = VCInterpolation(vcgamma_yy, k, j, i);
        gzz = VCInterpolation(vcgamma_zz, k, j, i);
        gxy = VCInterpolation(vcgamma_xy, k, j, i);
        gxz = VCInterpolation(vcgamma_xz, k, j, i);
        gyz = VCInterpolation(vcgamma_yz, k, j, i);

        Bx = 0.5*(pmb->pfield->b.x1f(k,j,i+1) + pmb->pfield->b.x1f(k,j,i));
        By = 0.5*(pmb->pfield->b.x2f(k,j+1,i) + pmb->pfield->b.x2f(k,j,i));
        Bz = 0.5*(pmb->pfield->b.x3f(k+1,j,i) + pmb->pfield->b.x3f(k,j,i));
        B2 = gxx*Bx*Bx + gyy*By*By + gzz*Bz*Bz + 2.0*(gxy*Bx*By + gxz*Bx*Bz + gyz*By*Bz);
        B2 += 1e-30;

        dx = pmb->pcoord->dx1v(i);
        dy = pmb->pcoord->dx2v(j);
        dz = pmb->pcoord->dx3v(k);
        vol = dx*dy*dz;
        divB += std::abs((pmb->pfield->b.x1f(k,j,i+1) - pmb->pfield->b.x1f(k,j,i))/dx + (pmb->pfield->b.x2f(k,j+1,i) - pmb->pfield->b.x2f(k,j,i))/(dy) + (pmb->pfield->b.x3f(k+1,j,i) - pmb->pfield->b.x3f(k,j,i))/(dz))/std::sqrt(B2) * vol;
      }
    }
  }
  return divB;
}

Real b_energy_bsq(MeshBlock *pmb, int iout) {
  Real b_energy_bsq = 0.0;
  Real vol,dx,dy,dz,bsq,Wlor,sqrtdetgamma;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        bsq = pmb->user_out_var(4,k,j,i);
        Wlor = pmb->user_out_var(5,k,j,i);
        sqrtdetgamma = pmb->user_out_var(9,k,j,i);
        dx = pmb->pcoord->dx1v(i);
        dy = pmb->pcoord->dx2v(j);
        dz = pmb->pcoord->dx3v(k);
        vol = dx*dy*dz;
        b_energy_bsq += bsq*Wlor*sqrtdetgamma*vol;
      }
    }
  }
  return b_energy_bsq;
}

Real b_energy_bsq_pol(MeshBlock *pmb, int iout) {
  Real b_energy_bsq_pol = 0.0;
  Real vol,dx,dy,dz,bsq_phi,Wlor,sqrtdetgamma;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        bsq_phi = pmb->user_out_var(6,k,j,i) * pmb->user_out_var(7,k,j,i);
        Wlor = pmb->user_out_var(5,k,j,i);
        sqrtdetgamma = pmb->user_out_var(9,k,j,i);
        dx = pmb->pcoord->dx1v(i);
        dy = pmb->pcoord->dx2v(j);
        dz = pmb->pcoord->dx3v(k);
        vol = dx*dy*dz;
        b_energy_bsq_pol += bsq_phi*Wlor*sqrtdetgamma*vol;
      }
    }
  }
  return b_energy_bsq_pol;
}

Real b_energy_bsq_star1(MeshBlock *pmb, int iout) {
  Real b_energy_bsq = 0.0;
  Mesh * pmesh = pmb->pmy_mesh;
  TrackerExtrema * ptracker_extrema = pmesh->ptracker_extrema;
  Real vol,dx,dy,dz,bsq,Wlor,sqrtdetgamma, x, y, z, x_centre, y_centre, z_centre;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
	x = pmb->pcoord->x1v(i);
	y = pmb->pcoord->x2v(j);
	z = pmb->pcoord->x3v(k);
        x_centre = ptracker_extrema->c_x1(0);
        y_centre = ptracker_extrema->c_x2(0);
        z_centre = ptracker_extrema->c_x3(0);
	if(std::sqrt(SQR(x -x_centre) + SQR(y - y_centre) + SQR(z - z_centre)) < integral_rad_m    ){
          bsq = pmb->user_out_var(4,k,j,i);
          Wlor = pmb->user_out_var(5,k,j,i);
          sqrtdetgamma = pmb->user_out_var(9,k,j,i);
          dx = pmb->pcoord->dx1v(i);
          dy = pmb->pcoord->dx2v(j);
          dz = pmb->pcoord->dx3v(k);
          vol = dx*dy*dz;
          b_energy_bsq += bsq*Wlor*sqrtdetgamma*vol;
	}
      }
    }
  }
  return b_energy_bsq;
}

Real b_energy_bsq_pol_star1(MeshBlock *pmb, int iout) {
  Real b_energy_bsq_pol = 0.0;
  Mesh * pmesh = pmb->pmy_mesh;
  TrackerExtrema * ptracker_extrema = pmesh->ptracker_extrema;
  Real vol,dx,dy,dz,bsq_phi,Wlor,sqrtdetgamma, x, y, z, x_centre, y_centre, z_centre, bphi_u, bphi_d;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
	x = pmb->pcoord->x1v(i);
	y = pmb->pcoord->x2v(j);
	z = pmb->pcoord->x3v(k);
        x_centre = ptracker_extrema->c_x1(0);
        y_centre = ptracker_extrema->c_x2(0);
        z_centre = ptracker_extrema->c_x3(0);
	if(std::sqrt(SQR(x -x_centre) + SQR(y - y_centre) + SQR(z - z_centre)) < integral_rad_m    ){
          bphi_u = (- (y - y_centre) * pmb->user_out_var(1,k,j,i) + (x - x_centre) * pmb->user_out_var(2,k,j,i)) / (std::sqrt(SQR(x - x_centre) + SQR(y - y_centre)));
          bphi_d = (- (y - y_centre) * pmb->user_out_var(11,k,j,i) + (x - x_centre) * pmb->user_out_var(12,k,j,i)) / (std::sqrt(SQR(x - x_centre) + SQR(y - y_centre)));
          bsq_phi = bphi_u * bphi_d;
          Wlor = pmb->user_out_var(5,k,j,i);
          sqrtdetgamma = pmb->user_out_var(9,k,j,i);
          dx = pmb->pcoord->dx1v(i);
          dy = pmb->pcoord->dx2v(j);
          dz = pmb->pcoord->dx3v(k);
          vol = dx*dy*dz;
          b_energy_bsq_pol += bsq_phi*Wlor*sqrtdetgamma*vol;
	}
      }
    }
  }
  return b_energy_bsq_pol;
}

Real b_energy_bsq_star2(MeshBlock *pmb, int iout) {
  Real b_energy_bsq = 0.0;
  Mesh * pmesh = pmb->pmy_mesh;
  TrackerExtrema * ptracker_extrema = pmesh->ptracker_extrema;
  Real vol,dx,dy,dz,bsq,Wlor,sqrtdetgamma, x, y, z, x_centre, y_centre, z_centre;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
	x = pmb->pcoord->x1v(i);
	y = pmb->pcoord->x2v(j);
	z = pmb->pcoord->x3v(k);
        x_centre = ptracker_extrema->c_x1(1);
        y_centre = ptracker_extrema->c_x2(1);
        z_centre = ptracker_extrema->c_x3(1);
	if(std::sqrt(SQR(x -x_centre) + SQR(y - y_centre) + SQR(z - z_centre)) < integral_rad_p    ){//nsrad fix
          bsq = pmb->user_out_var(4,k,j,i);
          Wlor = pmb->user_out_var(5,k,j,i);
          sqrtdetgamma = pmb->user_out_var(9,k,j,i);
          dx = pmb->pcoord->dx1v(i);
          dy = pmb->pcoord->dx2v(j);
          dz = pmb->pcoord->dx3v(k);
          vol = dx*dy*dz;
          b_energy_bsq += bsq*Wlor*sqrtdetgamma*vol;
	}
      }
    }
  }
  return b_energy_bsq;
}

Real b_energy_bsq_pol_star2(MeshBlock *pmb, int iout) {
  Real b_energy_bsq_pol = 0.0;
  Mesh * pmesh = pmb->pmy_mesh;
  TrackerExtrema * ptracker_extrema = pmesh->ptracker_extrema;
  Real vol,dx,dy,dz,bsq_phi,Wlor,sqrtdetgamma, x, y, z, x_centre, y_centre, z_centre, bphi_u, bphi_d;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
	x = pmb->pcoord->x1v(i);
	y = pmb->pcoord->x2v(j);
	z = pmb->pcoord->x3v(k);
        x_centre = ptracker_extrema->c_x1(1);
        y_centre = ptracker_extrema->c_x2(1);
        z_centre = ptracker_extrema->c_x3(1);
	if(std::sqrt(SQR(x -x_centre) + SQR(y - y_centre) + SQR(z - z_centre)) < integral_rad_p   ){//nsrad
          bphi_u = (- (y - y_centre) * pmb->user_out_var(1,k,j,i) + (x - x_centre) * pmb->user_out_var(2,k,j,i)) / (std::sqrt(SQR(x - x_centre) + SQR(y - y_centre)));
          bphi_d = (- (y - y_centre) * pmb->user_out_var(11,k,j,i) + (x - x_centre) * pmb->user_out_var(12,k,j,i)) / (std::sqrt(SQR(x - x_centre) + SQR(y - y_centre)));
          bsq_phi = bphi_u * bphi_d;
          Wlor = pmb->user_out_var(5,k,j,i);
          sqrtdetgamma = pmb->user_out_var(9,k,j,i);
          dx = pmb->pcoord->dx1v(i);
          dy = pmb->pcoord->dx2v(j);
          dz = pmb->pcoord->dx3v(k);
          vol = dx*dy*dz;
          b_energy_bsq_pol += bsq_phi*Wlor*sqrtdetgamma*vol;
	}
      }
    }
  }
  return b_energy_bsq_pol;
}

Real B2_pol(MeshBlock *pmb, int iout) {
  Real b_energy_bsq_pol = 0.0;
  Real vol,dx,dy,dz,B2_pol;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        B2_pol = pmb->user_out_var(8,k,j,i) * pmb->user_out_var(8,k,j,i)   ;
        dx = pmb->pcoord->dx1v(i);
        dy = pmb->pcoord->dx2v(j);
        dz = pmb->pcoord->dx3v(k);
        vol = dx*dy*dz;
        b_energy_bsq_pol += B2_pol*vol; //NB pmb->user_out_var 8 constructed  from pfield->bcc, this is already vol weighted
      }
    }
  }
  return b_energy_bsq_pol;
}

Real Emag(MeshBlock *pmb, int iout) {
  Real Eb = 0.0;
  Real vol, dx, dy, dz, detg;
  AthenaArray<Real> vcgamma_xx, vcgamma_xy, vcgamma_xz,
                    vcgamma_yy, vcgamma_yz, vcgamma_zz;
  Real gxx, gyy, gzz, gxy, gxz, gyz;
  Real Bx, By, Bz, B2; 
  vcgamma_xx.InitWithShallowSlice(pmb->pz4c->storage.adm,Z4c::I_ADM_gxx,1);
  vcgamma_xy.InitWithShallowSlice(pmb->pz4c->storage.adm,Z4c::I_ADM_gxy,1);
  vcgamma_xz.InitWithShallowSlice(pmb->pz4c->storage.adm,Z4c::I_ADM_gxz,1);
  vcgamma_yy.InitWithShallowSlice(pmb->pz4c->storage.adm,Z4c::I_ADM_gyy,1);
  vcgamma_yz.InitWithShallowSlice(pmb->pz4c->storage.adm,Z4c::I_ADM_gyz,1);
  vcgamma_zz.InitWithShallowSlice(pmb->pz4c->storage.adm,Z4c::I_ADM_gzz,1);
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {

        gxx = VCInterpolation(vcgamma_xx, k, j, i);
        gyy = VCInterpolation(vcgamma_yy, k, j, i);
        gzz = VCInterpolation(vcgamma_zz, k, j, i);
        gxy = VCInterpolation(vcgamma_xy, k, j, i);
        gxz = VCInterpolation(vcgamma_xz, k, j, i);
        gyz = VCInterpolation(vcgamma_yz, k, j, i);

        detg = - gxz*gxz*gyy + 2*gxy*gxz*gyz - gxx*gyz*gyz - gxy*gxy*gzz + gxx*gyy*gzz;
        dx = pmb->pcoord->dx1v(i);
        dy = pmb->pcoord->dx2v(j);
        dz = pmb->pcoord->dx3v(k);
        vol = dx*dy*dz/std::sqrt(detg);
        Bx = pmb->pfield->b.x1f(k,j,i);
        By = pmb->pfield->b.x2f(k,j,i);
        Bz = pmb->pfield->b.x3f(k,j,i);
        B2 = gxx*Bx*Bx + gyy*By*By + gzz*Bz*Bz + 2.0*(gxy*Bx*By + gxz*Bx*Bz + gyz*By*Bz);
        Eb += B2*vol;
      }
    }
  }
  return 0.5*Eb;
}

Real AtmosphereLoss(MeshBlock *pmb, int iout) {
  return pmb->mass_loss;
}

Real Maxbsq(MeshBlock *pmb, int iout) {
  Real max_bsq = 0.0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        max_bsq = std::max(std::sqrt(pmb->user_out_var(4,k,j,i)) , max_bsq);
      }
    }
  }
  return max_bsq;
}

Real Maxbeta(MeshBlock *pmb, int iout) {
  Real max_beta = 0.0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        max_beta = std::max(2.0*pmb->phydro->w(IPR,k,j,i)/pmb->user_out_var(4,k,j,i) , max_beta);
      }
    }
  }
  return max_beta;
}

Real Minbeta(MeshBlock *pmb, int iout) {
  Real min_beta = 1.0e300;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        min_beta = std::min(2.0*pmb->phydro->w(IPR,k,j,i)/pmb->user_out_var(4,k,j,i) , min_beta);
      }
    }
  }
  return min_beta;
}


Real Maxmag(MeshBlock *pmb, int iout) {
  Real max_mag = 0.0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        max_mag = std::max(pmb->user_out_var(4,k,j,i)/pmb->phydro->w(IDN,k,j,i) , max_mag);
      }
    }
  }
  return max_mag;
}

Real Minmag(MeshBlock *pmb, int iout) {
  Real min_mag = 1.0e300;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        min_mag = std::min(pmb->user_out_var(4,k,j,i)/pmb->phydro->w(IDN,k,j,i) , min_mag);
      }
    }
  }
  return min_mag;
}


Real int_energy(MeshBlock *pmb, int iout) {
  Real int_energy = 0.0;
  Real vol,dx,dy,dz,bsq,Wlor,sqrtdetgamma,eps;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Wlor = pmb->user_out_var(5,k,j,i);
        sqrtdetgamma = pmb->user_out_var(9,k,j,i);
        eps = pmb->user_out_var(14,k,j,i);
        dx = pmb->pcoord->dx1v(i);
        dy = pmb->pcoord->dx2v(j);
        dz = pmb->pcoord->dx3v(k);
        vol = dx*dy*dz;
        int_energy += pmb->phydro->w(IDN,k,j,i)*Wlor*sqrtdetgamma*vol*eps; //fix me
      }
    }
  }
  return int_energy;
}

Real MaxRho(MeshBlock* pmb, int iout) {
  Real max_rho = 0;
  for (int k = pmb->ks; k <= pmb->ke; k++) {
    for (int j = pmb->js; j <= pmb->je; j++) {
      for (int i = pmb->is; i <= pmb->ie; i++) {
        max_rho = std::fmax(max_rho, pmb->phydro->w(IDN, k, j, i));
      }
    }
  }
  return max_rho;
}

Real MaxB(MeshBlock* pmb, int iout) {
  Real max_B = 0;
  for (int k = pmb->ks; k <= pmb->ke; k++) {
    for (int j = pmb->js; j <= pmb->je; j++) {
      for (int i = pmb->is; i <= pmb->ie; i++) {
        max_B = std::fmax(max_B, std::sqrt(SQR(pmb->pfield->bcc(0, k, j, i)) + SQR(pmb->pfield->bcc(1, k, j, i)) + SQR(pmb->pfield->bcc(2, k, j, i))));
      }
    }
  }
  return max_B;
}

Real MinAlp(MeshBlock* pmb, int iout) {
  Real min_alp = 1.0e300;
  for (int k = pmb->ks; k <= pmb->ke+1; k++) {
    for (int j = pmb->js; j <= pmb->je+1; j++) {
      for (int i = pmb->is; i <= pmb->ie+1; i++) {
        min_alp = std::fmin(min_alp, pmb->pz4c->storage.u(Z4c::I_Z4c_alpha, k, j, i));
      }
    }
  }
  return min_alp;
}

Real totalvol(MeshBlock *pmb, int iout) {
  Real totalvol = 0.0;
  Real vol,dx,dy,dz,bsq,Wlor,sqrtdetgamma;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        sqrtdetgamma = pmb->user_out_var(9,k,j,i);
        dx = pmb->pcoord->dx1v(i);
        dy = pmb->pcoord->dx2v(j);
        dz = pmb->pcoord->dx3v(k);
        vol = dx*dy*dz;
        totalvol += sqrtdetgamma*vol;
      }
    }
  }
  return totalvol;
}

