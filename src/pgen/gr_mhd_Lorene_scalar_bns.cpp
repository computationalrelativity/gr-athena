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
Real DivBface(MeshBlock *pmb, int iout);
Real DivBface_abs(MeshBlock *pmb, int iout);
Real DivBface_norm(MeshBlock *pmb, int iout);
Real Emag(MeshBlock *pmb, int iout);
Real AtmosphereLoss(MeshBlock *pmb, int iout);
double linear_interp(double *f, double *x, int n, double xv);
int interp_locate(Real *x, int Nx, Real xval);

namespace {
  // Global variables
  std::string table_fname;
  LoreneTable * Table = NULL;
  std::string filename, filename_Y;
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
  AllocateUserHistoryOutput(5);
  EnrollUserHistoryOutput(0, DivBface, "divBface");
  EnrollUserHistoryOutput(1, DivBface_abs, "divBface_abs");
  EnrollUserHistoryOutput(2, DivBface_norm, "divBface_norm");
  EnrollUserHistoryOutput(3, Emag, "Emag");
  EnrollUserHistoryOutput(4, AtmosphereLoss, "atm-loss");
  if(adaptive==true)
    EnrollUserRefinementCondition(RefinementCondition);
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

  Real pgasmax, sep;
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


    // clean up
    delete[] xx_cc;
    delete[] yy_cc;
    delete[] zz_cc;

    delete pmy_mesh->bns;
  }

  pgasmax = pin->GetReal("problem","pmax");
  Real pcut = pin->GetReal("problem","pcut")*pgasmax;
  int magindex = pin->GetInteger("problem","magindex");
  Real ns = pin->GetReal("problem","ns");
  Real b_amp = pin->GetReal("problem","b_amp")*0.5/std::pow(pgasmax-pcut, ns)/8.351416e19;
  std::cout << "b_amp = " << b_amp << std::endl;
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

  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        if (pcoord->x1v(i) > 0) {
          Atot(0,k,j,i) = -pcoord->x2v(j) * b_amp * std::pow(std::max(phydro->w(IPR,k,j,i) - pcut, 0.0), ns);
          Atot(1,k,j,i) = (pcoord->x1v(i) - 0.5*sep) * b_amp * std::pow(std::max(phydro->w(IPR,k,j,i) - pcut, 0.0), ns);
          Atot(2,k,j,i) = 0.0;
        } else {
          Atot(0,k,j,i) = -pcoord->x2v(j) * b_amp * std::pow(std::max(phydro->w(IPR,k,j,i) - pcut, 0.0), ns);
          Atot(1,k,j,i) = (pcoord->x1v(i) + 0.5*sep) * b_amp * std::pow(std::max(phydro->w(IPR,k,j,i) - pcut, 0.0), ns);
          Atot(2,k,j,i) = 0.0;
        }
      }
    }
  }



  for (int k=ks-1; k<=ke+1; k++) {
    for (int j=js-1; j<=je+1; j++) {
      for (int i=is-1; i<=ie+1; i++) {
        bxcc(k,j,i) = - ((Atot(1,k+1,j,i) - Atot(1,k-1,j,i))/(2.0*pcoord->dx3v(k)));
        bycc(k,j,i) =  ((Atot(0,k+1,j,i) - Atot(0,k-1,j,i))/(2.0*pcoord->dx3v(k)));
        bzcc(k,j,i) = ( (Atot(1,k,j,i+1) - Atot(1,k,j,i-1))/(2.0*pcoord->dx1v(i))
                   - (Atot(0,k,j+1,i) - Atot(0,k,j-1,i))/(2.0*pcoord->dx2v(j)));
      }
    }
  }


  for(int k = ks; k<=ke; k++){
  for(int j = js; j<=je; j++){
  for(int i = is; i<=ie+1; i++){

  pfield->b.x1f(k,j,i) = 0.5*(bxcc(k,j,i-1) + bxcc(k,j,i));
}}}
  for(int k = ks; k<=ke; k++){
  for(int j = js; j<=je+1; j++){
  for(int i = is; i<=ie; i++){
  pfield->b.x2f(k,j,i) = 0.5*(bycc(k,j-1,i) + bycc(k,j,i));
}}}
  for(int k = ks; k<=ke+1; k++){
  for(int j = js; j<=je; j++){
  for(int i = is; i<=ie; i++){

  pfield->b.x3f(k,j,i) = 0.5*(bzcc(k-1,j,i) + bzcc(k,j,i));
}}}

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
