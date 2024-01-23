//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//! \file gr_Lorene_BinNSs.cpp
//  \brief Initial conditions for binary neutron stars.
//         Interpolation of Lorene initial data.
//         Requires the library:
//         https://lorene.obspm.fr/

#include <cassert>
#include <iostream>

// https://lorene.obspm.fr/
#include <bin_ns.h>
// #include <unites.h>

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
#include "../trackers/extrema_tracker.hpp"
#include "../utils/linear_algebra.hpp"
#include "../utils/utils.hpp"


namespace {
  int RefinementCondition(MeshBlock *pmb);

#if MAGNETIC_FIELDS_ENABLED
  Real DivBface(MeshBlock *pmb, int iout);
#endif

  Real max_rho(      MeshBlock *pmb, int iout);
  Real min_alpha(    MeshBlock *pmb, int iout);
  Real max_abs_con_H(MeshBlock *pmb, int iout);

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

  const int N = NDIM;

  typedef AthenaTensor<Real, TensorSymm::NONE, N, 0> AT_N_sca;
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 1> AT_N_vec;
  typedef AthenaTensor<Real, TensorSymm::SYM2, N, 2> AT_N_sym;

  // --------------------------------------------------------------------------
  // Set some aliases for the variables.
  AT_N_sca alpha( pz4c->storage.u,   Z4c::I_Z4c_alpha);
  AT_N_vec beta_u(pz4c->storage.u,   Z4c::I_Z4c_betax);
  AT_N_sym g_dd(  pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sym K_dd(  pz4c->storage.adm, Z4c::I_ADM_Kxx);

  // --------------------------------------------------------------------------
  #pragma omp critical
  {

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
    pmy_mesh->bns = new Lorene::Bin_NS(npoints_gs,
                                       xx_gs, yy_gs, zz_gs,
                                       fn_ini_data.c_str());

    // std::cout << pmy_mesh->bns->gamma_poly1 << std::endl;
    // std::cout << pmy_mesh->bns->kappa_poly1 / 2.6875380639256204e-4 << std::endl;

    // std::cout << pmy_mesh->bns->gamma_poly2 << std::endl;
    // std::cout << pmy_mesh->bns->kappa_poly2 / 2.6875380639256204e-4 << std::endl;

    // std::exit(0);

    Lorene::Bin_NS * bns = pmy_mesh->bns;
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

    delete pmy_mesh->bns;
    // ------------------------------------------------------------------------

    // prepare matter grid ----------------------------------------------------
    int npoints_cc = 0;

    const int il = 0;
    const int iu = ncells1-1;

    const int jl = 0;
    const int ju = ncells2-1;

    const int kl = 0;
    const int ku = ncells3-1;

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
    pmy_mesh->bns = new Lorene::Bin_NS(npoints_cc, xx_cc, yy_cc, zz_cc,
                                       fn_ini_data.c_str());

    bns = pmy_mesh->bns;
    assert(bns->np == npoints_cc);

    Real sep = bns->dist / coord_unit;
    // Real w_p_max = 0.0; //0.00013; ?? // compute below

    I = 0;      // reset

    Real k_adi = pin->GetReal("hydro", "k_adi");
    Real gamma_adi = pin->GetReal("hydro","gamma");


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
      const Real w_rho = bns->nbar[I] / rho_unit;
      const Real w_p = k_adi*pow(w_rho,gamma_adi);

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

      ++I;
    }

    phydro->w1 = phydro->w;

    // clean up
    delete[] xx_cc;
    delete[] yy_cc;
    delete[] zz_cc;

    delete pmy_mesh->bns;

  } // OMP Critical

  // --------------------------------------------------------------------------

  /*
  if (MAGNETIC_FIELDS_ENABLED)
  {
    // B field ------------------------------------------------------------------
    // Assume stars are located on x axis

    Real pcut = pin->GetReal("problem","pcut") * pgasmax;
    Real b_amp = pin->GetReal("problem","b_amp");
    int magindex = pin->GetInteger("problem","magindex");

    int nx1 = (ie-is)+1 + 2*(NGHOST); //TODO Shouldn't this be ncell[123]?
    int nx2 = (je-js)+1 + 2*(NGHOST);
    int nx3 = (ke-ks)+1 + 2*(NGHOST);

    pfield->b.x1f.ZeroClear();
    pfield->b.x2f.ZeroClear();
    pfield->b.x3f.ZeroClear();
    pfield->bcc.ZeroClear();

    AthenaArray<Real> bxcc,bycc,bzcc;
    bxcc.NewAthenaArray(nx3,nx2,nx1);
    bycc.NewAthenaArray(nx3,nx2,nx1);
    bzcc.NewAthenaArray(nx3,nx2,nx1);

    AthenaArray<Real> Atot;
    Atot.NewAthenaArray(3,nx3,nx2,nx1);

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

    pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, il,iu,jl,ju,kl,ku);
  } // MAGNETIC_FIELDS_ENABLED
  */
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
      peos->ApplyPrimitiveFloors(phydro->w, k, j, i);
    }

  }
  // --------------------------------------------------------------------------


  // Initialise conserved variables
  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord,
                             0, ncells1,
                             0, ncells2,
                             0, ncells3);

  //TODO Check if the momentum and velocity are finite.

  // Set up the matter tensor in the Z4c variables.
  // TODO: BD - this needs to be fixed properly
  // No magnetic field, pass dummy or fix with overload
  //  AthenaArray<Real> null_bb_cc;
  pz4c->GetMatter(pz4c->storage.mat, pz4c->storage.adm, phydro->w, pfield->bcc);

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

  const int N = NDIM;

  typedef AthenaTensor<Real, TensorSymm::NONE, N, 0> AT_N_sca;
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 1> AT_N_vec;
  typedef AthenaTensor<Real, TensorSymm::SYM2, N, 2> AT_N_sym;

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

  const int N = NDIM;

  typedef AthenaTensor<Real, TensorSymm::NONE, N, 0> AT_N_sca;
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 1> AT_N_vec;
  typedef AthenaTensor<Real, TensorSymm::SYM2, N, 2> AT_N_sym;

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

}