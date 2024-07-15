//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//! \file gr_Lorene_BinNSs.cpp
//  \brief Initial conditions for binary neutron stars.
//         Interpolation of Lorene initial data.
//         Requires the library:
//         https://lorene.obspm.fr/

#include <algorithm>
#include <cassert>
#include <iostream>

// https://lorene.obspm.fr/
#include <bin_ns.h>
// #include <unites.h>

// Athena++ headers
#include "../athena_aliases.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../z4c/z4c.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../bvals/bvals.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/mesh_refinement.hpp"
#include "../trackers/extrema_tracker.hpp"
#include "../utils/linear_algebra.hpp"
#include "../utils/utils.hpp"

//----------------------------------------------------------------------------------------
using namespace gra::aliases;
//----------------------------------------------------------------------------------------

// For reading composition tables
#if EOS_POLICY_CODE == 2 || EOS_POLICY_CODE == 3
#include "../utils/lorene_table.hpp"
#define LORENE_EOS (1)
#endif

namespace {
  int RefinementCondition(MeshBlock *pmb);

  Real linear_interp(Real *f, Real *x, int n, Real xv);
  int interp_locate(Real *x, int Nx, Real xval);

#if MAGNETIC_FIELDS_ENABLED
  Real DivBface(MeshBlock *pmb, int iout);
#endif

  Real max_rho(      MeshBlock *pmb, int iout);
  Real min_alpha(    MeshBlock *pmb, int iout);
  Real max_abs_con_H(MeshBlock *pmb, int iout);

  // Global variables
#ifdef LORENE_EOS
  LoreneTable * LORENE_EoS_Table = NULL; // Lorene table object
  std::string LORENE_EoS_fname; // Barotropic table
#if EOS_POLICY_CODE == 2
  std::string LORENE_EoS_fname_Y; // Conposition table
#endif
#endif
}


//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin)
{
  if(adaptive==true)
    EnrollUserRefinementCondition(RefinementCondition);


  AllocateUserHistoryOutput(4);
  EnrollUserHistoryOutput(0, max_rho,   "max-rho",
    UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, min_alpha, "min-alpha",
    UserHistoryOperation::min);
  EnrollUserHistoryOutput(2, max_abs_con_H, "max-abs-con.H",
    UserHistoryOperation::max);

#if MAGNETIC_FIELDS_ENABLED
  // AllocateUserHistoryOutput(1);
  EnrollUserHistoryOutput(3, DivBface, "divBface");
#endif

#ifdef LORENE_EOS
  LORENE_EoS_fname   = pin->GetString("hydro", "lorene");
#if EOS_POLICY_CODE == 2
  LORENE_EoS_fname_Y = pin->GetString("hydro", "lorene_Y");
#endif
  LORENE_EoS_Table = new LoreneTable;
  ReadLoreneTable(LORENE_EoS_fname, LORENE_EoS_Table);
#if EOS_POLICY_CODE == 2
  ReadLoreneFractions(LORENE_EoS_fname_Y, LORENE_EoS_Table);
#endif
  ConvertLoreneTable(LORENE_EoS_Table);
  LORENE_EoS_Table->rho_atm = pin->GetReal("hydro", "dfloor");
#endif

  // BD: TODO - remove once "GRAND INTERFACE" is complete.
  const bool have_fn_eos = pin->DoesParameterExist("problem", "filename_eos");
  if (have_fn_eos)
  {

    if (Globals::my_rank == 0)
    {
      std::string filename_eos = pin->GetString("problem", "filename_eos");
      std::string run_dir;

      if (file_exists(filename_eos.c_str()))
      {
        GetRunDir(run_dir);
        file_copy(filename_eos, run_dir);
      }
      else
      {
        std::stringstream msg;
        msg << "### FATAL ERROR problem/filename_eos: "
            << filename_eos << " "
            << " could not be accessed.";
        ATHENA_ERROR(msg);
        std::exit(0);
      }
    }
  }

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Sets the initial conditions.
//========================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  using namespace LinearAlgebra;

  // Interpolate Lorene data onto the grid.

  // constants ----------------------------------------------------------------
#if (1)
  // Some conversion factors to go between Lorene data and Athena data.
  // Shamelessly stolen from the Einstein Toolkit's Mag_NS.cc.
  //
  //SB Constants should be taken from Lorene's "unites" so to ensure consistency.
  //   Note the Lorene library has chandged some constants recently (2022),
  //   This is temporarily kept here for testing purposes.
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
  // This is just a guess based on what Cactus uses.
  Real const athenaB = (1.0 / athenaL /
			std::sqrt(eps0 * G_grav / (c_light * c_light)));
#else
  Real const c_light  = Unites::c_si;      // speed of light [m/s]
  Real const nuc_dens = Unites::rhonuc_si; // Nuclear density as used in Lorene units [kg/m^3]
  Real const G_grav   = Unites::g_si;      // gravitational constant [m^3/kg/s^2]
  Real const M_sun    = Unites::msol_si;   // solar mass [kg]

  // Units in terms of SI units:
  // (These are derived from M = M_sun, c = G = 1,
  //  and using 1/M_sun for the magnetic field)
  Real const athenaM = M_sun;
  Real const athenaL = athenaM * G_grav / pow(c_light,2);
  Real const athenaT = athenaL / c_light;
  // This is just a guess based on what Cactus uses:
  Real const athenaB = (1.0 / athenaL /
			std::sqrt(eps0 * G_grav / (c_light * c_light)));

  // Other quantities in terms of Athena units
  Real const coord_unit = athenaL / 1.0e+3;         // from km (~1.477)
  Real const rho_unit   = athenaM / pow(athenaL,3); // from kg/m^3
#endif

  // Athena units for conversion.
  Real const coord_unit = athenaL/1.0e3; // Convert to km for Lorene.
  Real const rho_unit = athenaM/(athenaL*athenaL*athenaL); // kg/m^3.
  Real const ener_unit = 1.0; // c^2
  Real const vel_unit = athenaL / athenaT / c_light; // c
  Real const B_unit = athenaB / 1.0e+9; // 10^9 T
  // --------------------------------------------------------------------------

  // settings -----------------------------------------------------------------
  std::string fn_ini_data = pin->GetOrAddString("problem", "filename", "resu.d");
  Real const tol_det_zero =  pin->GetOrAddReal("problem","tolerance_det_zero",1e-10);
  bool verbose = pin->GetOrAddBoolean("problem", "verbose", 0);

  // check ID is accessible
  if (!file_exists(fn_ini_data.c_str()))
  {
    std::stringstream msg;
    msg << "### FATAL ERROR problem/filename: " << fn_ini_data << " "
        << " could not be accessed.";
    ATHENA_ERROR(msg);
  }

  // container with idx / grids pertaining z4c
  MB_info* mbi = &(pz4c->mbi);

  // --------------------------------------------------------------------------
  // Set some aliases for the variables.
  AT_N_sca alpha( pz4c->storage.adm, Z4c::I_ADM_alpha);
  AT_N_vec beta_u(pz4c->storage.adm, Z4c::I_ADM_betax);
  AT_N_sym g_dd(  pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sym K_dd(  pz4c->storage.adm, Z4c::I_ADM_Kxx);


  // prepare matter grid ----------------------------------------------------
  const int il = 0;
  const int iu = ncells1-1;

  const int jl = 0;
  const int ju = ncells2-1;

  const int kl = 0;
  const int ku = ncells3-1;

  Real sep;

  // --------------------------------------------------------------------------
  #pragma omp critical
  {
    Lorene::Bin_NS * bns;

    // prepare geometry grid --------------------------------------------------
    int npoints_gs = 0;

    for (int k=0; k<mbi->nn3; ++k)
    for (int j=0; j<mbi->nn2; ++j)
    for (int i=0; i<mbi->nn1; ++i)
    {
    	++npoints_gs;
    }

    Real * xx_gs = new Real[npoints_gs];
    Real * yy_gs = new Real[npoints_gs];
    Real * zz_gs = new Real[npoints_gs];

    int I = 0;  // collapsed ijk index

    for (int k=0; k<mbi->nn3; ++k)
    for (int j=0; j<mbi->nn2; ++j)
    for (int i=0; i<mbi->nn1; ++i)
    {
      zz_gs[I] = coord_unit * mbi->x3(k);
      yy_gs[I] = coord_unit * mbi->x2(j);
      xx_gs[I] = coord_unit * mbi->x1(i);

      ++I;
    }
    // ------------------------------------------------------------------------

    // prepare Lorene interpolator for geometry -------------------------------
    bns = new Lorene::Bin_NS(npoints_gs,
                             xx_gs, yy_gs, zz_gs,
                             fn_ini_data.c_str());

    // std::cout << pmy_mesh->bns->gamma_poly1 << std::endl;
    // std::cout << pmy_mesh->bns->kappa_poly1 / 2.6875380639256204e-4 << std::endl;

    // std::cout << pmy_mesh->bns->gamma_poly2 << std::endl;
    // std::cout << pmy_mesh->bns->kappa_poly2 / 2.6875380639256204e-4 << std::endl;

    // std::exit(0);

    assert(bns->np == npoints_gs);

    I = 0;      // reset

    for (int k=0; k<mbi->nn3; ++k)
    for (int j=0; j<mbi->nn2; ++j)
    for (int i=0; i<mbi->nn1; ++i)
    {
      // Gauge from Lorene
      //TODO Option to reset?
      alpha(k, j, i)     =  bns->nnn[I];
      beta_u(0, k, j, i) = -bns->beta_x[I];
      beta_u(1, k, j, i) = -bns->beta_y[I];
      beta_u(2, k, j, i) = -bns->beta_z[I];

      g_dd(0, 0, k, j, i) = bns->g_xx[I];
      K_dd(0, 0, k, j, i) = coord_unit * bns->k_xx[I];

      g_dd(0, 1, k, j, i) = bns->g_xy[I];
      K_dd(0, 1, k, j, i) = coord_unit * bns->k_xy[I];

      g_dd(0, 2, k, j, i) = bns->g_xz[I];
      K_dd(0, 2, k, j, i) = coord_unit * bns->k_xz[I];

      g_dd(1, 1, k, j, i) = bns->g_yy[I];
      K_dd(1, 1, k, j, i) = coord_unit * bns->k_yy[I];

      g_dd(1, 2, k, j, i) = bns->g_yz[I];
      K_dd(1, 2, k, j, i) = coord_unit * bns->k_yz[I];

      g_dd(2, 2, k, j, i) = bns->g_zz[I];
      K_dd(2, 2, k, j, i) = coord_unit * bns->k_zz[I];

      const Real det = Det3Metric(g_dd, k, j, i);
      assert(std::fabs(det) > tol_det_zero);

      ++I;
    }
    // ------------------------------------------------------------------------

    // show some info ---------------------------------------------------------
    if(verbose)
    {
      std::cout << "Lorene data on current MeshBlock." << std::endl;
      std::cout <<" omega [rad/s]:       " << bns->omega << std::endl;
      std::cout <<" dist [km]:           " << bns->dist<< std::endl;
      std::cout <<" dist_mass [km]:      " << bns->dist_mass << std::endl;
      std::cout <<" mass1_b [M_sun]:     " << bns->mass1_b << std::endl;
      std::cout <<" mass2_b [M_sun]:     " << bns->mass2_b << std::endl;
      std::cout <<" mass_ADM [M_sun]:    " << bns->mass_adm << std::endl;
      std::cout <<" L_tot [G M_sun^2/c]: " << bns->angu_mom << std::endl;
      std::cout <<" rad1_x_comp [km]:    " << bns->rad1_x_comp << std::endl;
      std::cout <<" rad1_y [km]:         " << bns->rad1_y << std::endl;
      std::cout <<" rad1_z [km]:         " << bns->rad1_z << std::endl;
      std::cout <<" rad1_x_opp [km]:     " << bns->rad1_x_opp << std::endl;
      std::cout <<" rad2_x_comp [km]:    " << bns->rad2_x_comp << std::endl;
      std::cout <<" rad2_y [km]:         " << bns->rad2_y << std::endl;
      std::cout <<" rad2_z [km]:         " << bns->rad2_z << std::endl;
      std::cout <<" rad2_x_opp [km]:     " << bns->rad2_x_opp << std::endl;
      // LORENE's EOS is in terms on number density n = rho/m_nucleon:
      // P = K n^Gamma
      // to convert to SI units:
      // K_SI(n) = K_LORENE rho_nuc c^2 / n_nuc^gamma
      // Converting this to be in terms of the mass density rho = n m_nucleon gets
      // changes n_nuc to rho_nuc:
      // K_SI(rho) = K_LORENE c^2 / rho_nuc^(gamma-1)
      // In SI units P has units of M / (L T^2) and rho has units of M/L^3 thus
      // K_SI has units of (L^3/M)^Gamma M/(L T^2).
      // In Cactus units P and rho have the same units thus K_Cactus is unitless.
      // Conversion between K_SI and K_Cactus thus amounts to dividing out the
      // units of the SI quantity.
      // Real K = bns->kappa_poly1 * pow((pow(c_light, 6.0) /
			// 		 ( pow(G_grav, 3.0) * M_sun * M_sun *
			// 		   nuc_dens )),bns->gamma_poly1-1.);
      // std::cout << "EOS K ]:              " << K<< std::endl;

      std::cout << "EOS K ]: (fix units for this)" << std::endl;
    }

    // clean up
    delete[] xx_gs;
    delete[] yy_gs;
    delete[] zz_gs;

    delete bns;
    // ------------------------------------------------------------------------

    // prepare matter grid ----------------------------------------------------
    int npoints_cc = 0;


    for (int k = kl; k <= ku; ++k)
    for (int j = jl; j <= ju; ++j)
    for (int i = il; i <= iu; ++i)
    {
      ++npoints_cc;
    }

    Real * xx_cc = new Real[npoints_cc];
    Real * yy_cc = new Real[npoints_cc];
    Real * zz_cc = new Real[npoints_cc];

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
    // prepare Lorene interpolator for matter ---------------------------------
    bns = new Lorene::Bin_NS(npoints_cc, xx_cc, yy_cc, zz_cc,
                             fn_ini_data.c_str());

    assert(bns->np == npoints_cc);

    sep = bns->dist / coord_unit;
    // Real w_p_max = 0.0; //0.00013; ?? // compute below

    I = 0;      // reset

// Set up EoS.
#if USETM
    Real rho_atm = pin->GetReal("hydro", "dfloor");
    Real T_atm = pin->GetReal("hydro", "tfloor");
    Real mb = peos->GetEOS().GetBaryonMass();
    Real Y_atm[MAX_SPECIES] = {0.0};
#if EOS_POLICY_CODE == 2
    Y_atm[0] = pin->GetReal("hydro", "y0_atmosphere");
#endif
#else
    Real k_adi = pin->GetReal("hydro", "k_adi");
    Real gamma_adi = pin->GetReal("hydro","gamma");
#endif

    for (int k = kl; k <= ku; ++k)
    for (int j = jl; j <= ju; ++j)
    for (int i = il; i <= iu; ++i)
    {
      /*
      const Real w_rho = bns->nbar[I] / rho_unit;
      const Real eps = bns->ener_spec[I] / ener_unit;

      // Real egas = rho * (1.0 + eps);  <-------- ?
      Real egas = w_rho * eps;
      Real w_p = peos->PresFromRhoEg(w_rho, egas);

      // Real w_p = k_adi*pow(w_rho,gamma_adi);

      // Kludge to make the pressure work with the EOS framework.
      if (!std::isfinite(w_p) && (egas == 0. || w_rho == 0.)) {
        w_p = 0.;
      }

      // w_p_max = std::max(w_p_max, w_p);

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

      */

// If scalars are on we should initialise to zero
      Real Y[MAX_SPECIES];
      std::fill(std::begin(Y), std::end(Y), 0.0);  // or Y_atm?
#if NSCALARS > 0
      for (int r=0;r<NSCALARS;r++) {
        pscalars->r(r,k,j,i) = 0.;
        pscalars->s(r,k,j,i) = 0.;
      }
#endif

#if USETM
      Real w_rho = (bns->nbar[I] / rho_unit > rho_atm ? bns->nbar[I] / rho_unit : 0.0);
      Real nb = w_rho/mb;
#if NSCALARS > 0
      for (int l=0; l<NSCALARS; ++l) {
        Y[l] = (w_rho > rho_atm ? linear_interp(LORENE_EoS_Table->Y[l], LORENE_EoS_Table->data[tab_logrho], LORENE_EoS_Table->size, log(w_rho)) : Y_atm[l]);
      }
#endif
      Real w_p = (w_rho > rho_atm ? peos->GetEOS().GetPressure(nb, T_atm, Y) : 0.0);
      phydro->temperature(k,j,i) = T_atm;
#else
      Real w_rho = bns->nbar[I] / rho_unit;
      Real w_p = k_adi*pow(w_rho,gamma_adi);
#endif

#ifdef USE_IDEAL_GAS
      // BD: TODO - stupid work-around to get barotropic init. with
      //            PrimitiveSolver-
      //            w_p = K w_rho ^ gamma_adi & fake T
      {
        const Real gamma_adi = pin->GetReal("hydro", "gamma");
        const Real k_adi     = pin->GetReal("hydro", "k_adi");
        w_rho = bns->nbar[I] / rho_unit;
        w_p = k_adi*pow(w_rho,gamma_adi);
        phydro->temperature(k,j,i) = peos->GetEOS().GetTemperatureFromP(nb, w_p, Y);
      }
#endif

      const Real v_u_x = bns->u_euler_x[I] / vel_unit;
      const Real v_u_y = bns->u_euler_y[I] / vel_unit;
      const Real v_u_z = bns->u_euler_z[I] / vel_unit;

      const Real vsq = (
        2.0*(v_u_x * v_u_y * bns->g_xy[I] +
             v_u_x * v_u_z * bns->g_xz[I] +
             v_u_y * v_u_z * bns->g_yz[I]) +
        v_u_x * v_u_x * bns->g_xx[I] +
        v_u_y * v_u_y * bns->g_yy[I] +
        v_u_z * v_u_z * bns->g_zz[I]
      );

      const Real W = 1.0 / std::sqrt(1.0 - vsq);

      phydro->w(IDN, k, j, i) = w_rho;
      phydro->w(IVX, k, j, i) = W * v_u_x;
      phydro->w(IVY, k, j, i) = W * v_u_y;
      phydro->w(IVZ, k, j, i) = W * v_u_z;
      phydro->w(IPR, k, j, i) = w_p;
#if NSCALARS > 0
      for (int r=0;r<NSCALARS;r++) {
        pscalars->r(r,k,j,i) = Y[r];
      }
#endif
      ++I;
    }

    phydro->w1 = phydro->w;

    // clean up
    delete[] xx_cc;
    delete[] yy_cc;
    delete[] zz_cc;

    delete bns;

  } // OMP Critical

  // --------------------------------------------------------------------------

  if (MAGNETIC_FIELDS_ENABLED)
  {
    // B field ------------------------------------------------------------------
    // Assume stars are located on x axis

    Real pgasmax = pin->GetReal("problem","pmax");
    Real pcut = pin->GetReal("problem","pcut") * pgasmax;

    // Real b_amp = pin->GetReal("problem","b_amp");
    // Scaling taken from project_bnsmhd
    Real ns = pin->GetReal("problem","ns");
    Real b_amp = pin->GetReal("problem","b_amp") *
                 0.5/std::pow(pgasmax-pcut, ns)/8.351416e19;
    int magindex = pin->GetInteger("problem","magindex");

    pfield->b.x1f.ZeroClear();
    pfield->b.x2f.ZeroClear();
    pfield->b.x3f.ZeroClear();
    pfield->bcc.ZeroClear();

    AthenaArray<Real> bxcc,bycc,bzcc;
    bxcc.NewAthenaArray(ncells3,ncells2,ncells1);
    bycc.NewAthenaArray(ncells3,ncells2,ncells1);
    bzcc.NewAthenaArray(ncells3,ncells2,ncells1);

    AthenaArray<Real> Atot;
    Atot.NewAthenaArray(3,ncells3,ncells2,ncells1);

    for (int k = kl; k <= ku; ++k)
    for (int j = jl; j <= ju; ++j)
    for (int i = il; i <= iu; ++i)
      {
        if(pcoord->x1v(i) > 0){
    Atot(0,k,j,i) = -pcoord->x2v(j) * b_amp * std::max(phydro->w(IPR,k,j,i) - pcut, 0.0);
    Atot(1,k,j,i) = (pcoord->x1v(i) - 0.5*sep) * b_amp * std::max(phydro->w(IPR,k,j,i) - pcut, 0.0);
    Atot(2,k,j,i) = 0.0;
        } else {
    Atot(0,k,j,i) = -pcoord->x2v(j) * b_amp * std::max(phydro->w(IPR,k,j,i) - pcut, 0.0);
    Atot(1,k,j,i) = (pcoord->x1v(i) + 0.5*sep) * b_amp * std::max(phydro->w(IPR,k,j,i) - pcut, 0.0);
    Atot(2,k,j,i) = 0.0;
        }
      }

    for(int k = ks-1; k<=ke+1; k++)
    for(int j = js-1; j<=je+1; j++)
    for(int i = is-1; i<=ie+1; i++)
      {

      bxcc(k,j,i) = - ((Atot(1,k+1,j,i) - Atot(1,k-1,j,i))/(2.0*pcoord->dx3v(k)));
      bycc(k,j,i) =  ((Atot(0,k+1,j,i) - Atot(0,k-1,j,i))/(2.0*pcoord->dx3v(k)));
      bzcc(k,j,i) = ( (Atot(1,k,j,i+1) - Atot(1,k,j,i-1))/(2.0*pcoord->dx1v(i))
                    - (Atot(0,k,j+1,i) - Atot(0,k,j-1,i))/(2.0*pcoord->dx2v(j)));
      }

    for(int k = ks; k<=ke; k++)
    for(int j = js; j<=je; j++)
    for(int i = is; i<=ie+1; i++)
      {

      pfield->b.x1f(k,j,i) = 0.5*(bxcc(k,j,i-1) + bxcc(k,j,i));
      }

    for(int k = ks; k<=ke; k++)
    for(int j = js; j<=je+1; j++)
    for(int i = is; i<=ie; i++)
      {
      pfield->b.x2f(k,j,i) = 0.5*(bycc(k,j-1,i) + bycc(k,j,i));
      }

    for(int k = ks; k<=ke+1; k++)
    for(int j = js; j<=je; j++)
    for(int i = is; i<=ie; i++)
      {
        pfield->b.x3f(k,j,i) = 0.5*(bzcc(k-1,j,i) + bzcc(k,j,i));
      }

    pfield->CalculateCellCenteredField(pfield->b,
                                       pfield->bcc,
                                       pcoord,
                                       il,iu,jl,ju,kl,ku);


  } // MAGNETIC_FIELDS_ENABLED
  //  -------------------------------------------------------------------------

  // Construct Z4c vars from ADM vars ------------------------------------------
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);
  // pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u1);

  // Allow override of Lorene gauge -------------------------------------------
  bool fix_gauge_precollapsed = pin->GetOrAddBoolean(
    "problem", "fix_gauge_precollapsed", false);

  if (fix_gauge_precollapsed)
  {
    // to construct psi4
    pz4c->Z4cToADM(pz4c->storage.u, pz4c->storage.adm);
    pz4c->GaugePreCollapsedLapse(pz4c->storage.adm, pz4c->storage.u);
  }
  // --------------------------------------------------------------------------

  // consistent pressure atmosphere -------------------------------------------
  bool id_floor_primitives = pin->GetOrAddBoolean(
    "problem", "id_floor_primitives", false);

  if (id_floor_primitives)
  {

    for (int k = 0; k <= ncells3-1; ++k)
    for (int j = 0; j <= ncells2-1; ++j)
    for (int i = 0; i <= ncells1-1; ++i)
    {
#if USETM
      peos->ApplyPrimitiveFloors(phydro->w, pscalars->r, k, j, i);
#else
      peos->ApplyPrimitiveFloors(phydro->w, k, j, i);
#endif
    }

  }
  // --------------------------------------------------------------------------


  // Initialise conserved variables
#if USETM
  peos->PrimitiveToConserved(phydro->w,
                             pscalars->r,
                             pfield->bcc,
                             phydro->u,
                             pscalars->s,
                             pcoord,
                             0, ncells1-1,
                             0, ncells2-1,
                             0, ncells3-1);
#else
  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord,
                             0, ncells1-1,
                             0, ncells2-1,
                             0, ncells3-1);
#endif
  //TODO Check if the momentum and velocity are finite.

  // Set up the matter tensor in the Z4c variables.
  // TODO: BD - this needs to be fixed properly
  // No magnetic field, pass dummy or fix with overload
  //  AthenaArray<Real> null_bb_cc;
#if USETM
  pz4c->GetMatter(pz4c->storage.mat,
                  pz4c->storage.adm,
                  phydro->w,
                  pscalars->r,
                  pfield->bcc);
#else
  pz4c->GetMatter(pz4c->storage.mat,
                  pz4c->storage.adm,
                  phydro->w,
                  pfield->bcc);
#endif

  pz4c->ADMConstraints(pz4c->storage.con,
                       pz4c->storage.adm,
                       pz4c->storage.mat,
                       pz4c->storage.u);

  // --------------------------------------------------------------------------
  return;
}

namespace {

//----------------------------------------------------------------------------------------
//! \fn
//  \brief refinement condition: extrema based
// 1: refines, -1: de-refines, 0: does nothing
int RefinementCondition(MeshBlock *pmb)
{
  /*
  // BD: TODO in principle this should be possible
  Z4c_AMR *const pz4c_amr = pmb->pz4c->pz4c_amr;

  // ensure we actually have a tracker
  if (pmb->pmy_mesh->ptracker_extrema->N_tracker > 0)
  {
    return 0;
  }

  return pz4c_amr->ShouldIRefine(pmb);
  */

  Mesh * pmesh = pmb->pmy_mesh;
  ExtremaTracker * ptracker_extrema = pmesh->ptracker_extrema;

  int root_level = ptracker_extrema->root_level;
  int mb_physical_level = pmb->loc.level - root_level;


  // Iterate over refinement levels offered by trackers.
  //
  // By default if a point is not in any sphere, completely de-refine.
  int req_level = 0;

  for (int n=1; n<=ptracker_extrema->N_tracker; ++n)
  {
    bool is_contained = false;
    int cur_req_level = ptracker_extrema->ref_level(n-1);

    {
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
      }
    }

    if (is_contained)
    {
      req_level = std::max(cur_req_level, req_level);
    }

  }

  if (req_level > mb_physical_level)
  {
    return 1;  // currently too coarse, refine
  }
  else if (req_level == mb_physical_level)
  {
    return 0;  // level satisfied, do nothing
  }

  // otherwise de-refine
  return -1;

}

#if MAGNETIC_FIELDS_ENABLED
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
        divB += ((pmb->pfield->b.x1f(k,j,i+1) - pmb->pfield->b.x1f(k,j,i))/ dx +
                 (pmb->pfield->b.x2f(k,j+1,i) - pmb->pfield->b.x2f(k,j,i))/ dy +
                 (pmb->pfield->b.x3f(k+1,j,i) - pmb->pfield->b.x3f(k,j,i))/ dz) * vol;
      }
    }
  }
  return divB;
}
#endif

Real max_rho(MeshBlock *pmb, int iout)
{
  Real max_rho = -std::numeric_limits<Real>::infinity();
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;

  AthenaArray<Real> &w = pmb->phydro->w;
  for (int k=ks; k<=ke; k++)
  for (int j=js; j<=je; j++)
  for (int i=is; i<=ie; i++)
  {
    max_rho = std::max(std::abs(w(IDN,k,j,i)), max_rho);
  }

  return max_rho;
}

Real min_alpha(MeshBlock *pmb, int iout)
{

  // --------------------------------------------------------------------------
  // Set some aliases for the variables.
  AT_N_sca alpha(pmb->pz4c->storage.u, Z4c::I_Z4c_alpha);

  // container with idx / grids pertaining z4c
  MB_info* mbi = &(pmb->pz4c->mbi);

  Real m_alpha = std::numeric_limits<Real>::infinity();

  for (int k=mbi->kl; k<=mbi->ku; k++)
  for (int j=mbi->jl; j<=mbi->ju; j++)
  for (int i=mbi->il; i<=mbi->iu; i++)
  {
    m_alpha = std::min(alpha(k,j,i), m_alpha);
  }

  return m_alpha;
}

Real max_abs_con_H(MeshBlock *pmb, int iout)
{
  // --------------------------------------------------------------------------
  // Set some aliases for the variables.
  AT_N_sca con_H(pmb->pz4c->storage.con, Z4c::I_CON_H);

  // container with idx / grids pertaining z4c
  MB_info* mbi = &(pmb->pz4c->mbi);

  Real m_abs_con_H = -std::numeric_limits<Real>::infinity();

  for (int k=mbi->kl; k<=mbi->ku; k++)
  for (int j=mbi->jl; j<=mbi->ju; j++)
  for (int i=mbi->il; i<=mbi->iu; i++)
  {
    m_abs_con_H = std::max(std::abs(con_H(k,j,i)), m_abs_con_H);
  }

  return m_abs_con_H;
}

#ifdef LORENE_EOS
  //--------------------------------------------------------------------------------------
  //! \fn double linear_interp(double *f, double *x, int n, double xv)
  // \brief linearly interpolate f(x), compute f(xv)
  Real linear_interp(Real *f, Real *x, int n, Real xv)
  {
    int i = interp_locate(x,n,xv);
    if (i < 0)  i=1;
    if (i == n) i=n-1;
    int j;
    if(xv < x[i]) j = i-1;
    else j = i+1;
    Real xj = x[j]; Real xi = x[i];
    Real fj = f[j]; Real fi = f[i];
    Real m = (fj-fi)/(xj-xi);
    Real df = m*(xv-xj)+fj;
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
  #endif

}
