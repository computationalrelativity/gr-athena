//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gr_tov.cpp
//  \brief Problem generator for single TOV star in Cowling approximation

// C headers

// C++ headers
#include <cmath>   // abs(), NAN, pow(), sqrt()
#include <cstring> // strcmp()
#include <fstream>    // ifstream
#include <iomanip>
#include <iostream>   // endl, ostream
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // string

// Athena++ headers
#include "../athena.hpp"                   // macros, enums, FaceField
#include "../athena_arrays.hpp"            // AthenaArray
#include "../bvals/bvals.hpp"              // BoundaryValues
#include "../coordinates/coordinates.hpp"  // Coordinates
#include "../eos/eos.hpp"                  // EquationOfState
#include "../z4c/z4c.hpp"
// #include "../z4c/z4c_amr.hpp"
#include "../trackers/extrema_tracker.hpp"
#include "../field/field.hpp"              // Field
#include "../hydro/hydro.hpp"              // Hydro
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"          // ParameterInput
#include "../utils/linear_algebra.hpp"

// Configuration checking
#if not GENERAL_RELATIVITY
#error "This problem generator must be used with general relativity"
#endif

namespace {
  int TOV_rhs(Real dr, Real *u, Real *k);
  int TOV_solve(Real rhoc, Real rmin, Real dr, int *npts);
  int interp_locate(Real *x, int Nx, Real xval);
  void interp_lag4(Real *f, Real *x, int Nx, Real xv,
		   Real *fv_p, Real *dfv_p, Real *ddfv_p );
  /*
  void TOV_background(Real x1, Real x2, Real x3, ParameterInput *pin,
		      AthenaArray<Real> &g, AthenaArray<Real> &g_inv,
		      AthenaArray<Real> &dg_dx1, AthenaArray<Real> &dg_dx2,
		      AthenaArray<Real> &dg_dx3);//TOV_ID
  */

  // Insert interpolated TOV soln. to extant variables
  void TOV_populate(MeshBlock *pmb,
                    ParameterInput *pin);

  int RefinementCondition(MeshBlock *pmb);

  // Global variables
  Real gamma_adi, k_adi;  // hydro EOS parameters
  Real v_amp; // velocity amplitude for linear perturbations

  // TOV var indexes for ODE integration
  enum{TOV_IRHO,TOV_IMASS,TOV_IPHI,TOV_IINT,TOV_NVAR};

  // TOV 1D data
  enum{itov_rsch,itov_riso,itov_rho,itov_mass,itov_phi,itov_pre,itov_psi4,itov_lapse,itov_nv};
  struct TOVData
  {
    int npts;

    int interp_npts;
    Real interp_dr;
    Real surf_dr;

    Real lapse_0,psi4_0; // Regularized values at r=0
    Real * data[itov_nv];
    Real R, Riso, M;
  };
  TOVData * tov = NULL;

  Real Maxrho(MeshBlock *pmb, int iout);
#if MAGNETIC_FIELDS_ENABLED
  Real DivB(MeshBlock *pmb, int iout);
#endif
} // namespace

namespace TOV_geom {

// BD: TODO this is all generic technology and could be shifted elsewhere ...

// Lorentz transformation matrix.
// Notes:
// - Boost vector is independent of position.
// - If boost non-trivial (i.e. non-zero) return true else false
// - Represents passive transformation by default
//   x^mu' = L^mu'_nu x^nu
template<typename T>
inline bool Lorentz4Boost(
  AthenaArray<T> & lam,
  AthenaArray<T> & xi,
  const bool is_passive=true
);

}

//----------------------------------------------------------------------------------------
//! \fn 
// \brief  Function for initializing global mesh properties
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Read problem parameters

  // Central value of energy density
  Real rhoc = pin->GetReal("problem", "rhoc");
  // minimum radius to start TOV integration
  Real rmin = pin->GetOrAddReal("problem", "rmin", 0);
  // radial step for TOV integration
  Real dr = pin->GetReal("problem", "dr");
  // number of max radial pts for TOV solver
  int npts = pin->GetInteger("problem", "npts");

  k_adi = pin->GetReal("hydro", "k_adi");
  gamma_adi = pin->GetReal("hydro","gamma");
  v_amp = pin->GetOrAddReal("problem", "v_amp", 0.0);

  // Alloc 1D buffer
  tov = new TOVData;
  tov->npts = npts;

  // spacing & number of points to retain for interpolation
  tov->interp_npts = pin->GetOrAddInteger("problem", "interp_npts", npts);
  tov->interp_dr   = pin->GetOrAddReal(   "problem", "interp_dr",   dr);

  // surface identification
  tov->surf_dr   = pin->GetOrAddReal(     "problem", "surf_dr",   dr / 1.0e3);

  for (int v = 0; v < itov_nv; v++)
    tov->data[v] = (Real*) malloc((tov->interp_npts)*sizeof(Real));

  // Solve TOV equations, setting 1D inital data in tov->data
  TOV_solve(rhoc, rmin, dr, &npts);

  if(adaptive==true)
    EnrollUserRefinementCondition(RefinementCondition);

  AllocateUserHistoryOutput(2);
  EnrollUserHistoryOutput(0, Maxrho, "max-rho", UserHistoryOperation::max);
#if MAGNETIC_FIELDS_ENABLED
  EnrollUserHistoryOutput(1, DivB, "divB");
#endif
}


//----------------------------------------------------------------------------------------
//! \fn 
// \brief Setup User work

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  // Allocate output arrays for fluxes
  AllocateUserOutputVariables(15);
  return;
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  AthenaArray<Real> &x1flux = phydro->flux[X1DIR];
  AthenaArray<Real> &x2flux = phydro->flux[X2DIR];
  AthenaArray<Real> &x3flux = phydro->flux[X3DIR];

  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;
  if (block_size.nx2 > 1)
  {
    jl -= NGHOST;
    ju += NGHOST;
  }
  int kl = ks;
  int ku = ke;
  if (block_size.nx3 > 1)
  {
    kl -= NGHOST;
    ku += NGHOST;
  }

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  for (int i=il; i<=iu; ++i)
  {
    user_out_var(0,k,j,i) = x1flux(0,k,j,i);
    user_out_var(1,k,j,i) = x1flux(1,k,j,i);
    user_out_var(2,k,j,i) = x1flux(2,k,j,i);
    user_out_var(3,k,j,i) = x1flux(3,k,j,i);
    user_out_var(4,k,j,i) = x1flux(4,k,j,i);
    user_out_var(5,k,j,i) = x2flux(0,k,j,i);
    user_out_var(6,k,j,i) = x2flux(1,k,j,i);
    user_out_var(7,k,j,i) = x2flux(2,k,j,i);
    user_out_var(8,k,j,i) = x2flux(3,k,j,i);
    user_out_var(9,k,j,i) = x2flux(4,k,j,i);
    user_out_var(10,k,j,i) = x3flux(0,k,j,i);
    user_out_var(11,k,j,i) = x3flux(1,k,j,i);
    user_out_var(12,k,j,i) = x3flux(2,k,j,i);
    user_out_var(13,k,j,i) = x3flux(3,k,j,i);
    user_out_var(14,k,j,i) = x3flux(4,k,j,i);
  }

  return;
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin)
{
  // Free TOV data
  if (NULL != tov )
  {
    for (int v = 0; v < itov_nv; v++)
    {
      if (NULL != tov->data[v])
      {
        free(tov->data[v]);
        tov->data[v] = NULL;
      }
    }

    delete tov;
    tov = NULL;
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn
// \brief Function for setting initial conditions
// Inputs:
//   pin: parameters
// Outputs: (none)
// Notes:
//   sets primitive and conserved variables according to input primitives
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)

// Dictionary between Noble et al 06 and BAM
//
// Noble: T^{mu nu} = w u^mu u^nu + p g^{mu nu}
//        w = rho0 + p + u
//        where u is internal energy/proper vol
//
// BAM:   T^{mu nu} = (e + p) u^mu u^nu + p g^{mu nu}
//        e = rho(1+epsl)
//
// So conversion is rho0=rho and u = rho*epsl

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  // Parameters - prefilled as TOV is added to these quantities
  phydro->w_init.Fill(   0);
  phydro->w.Fill(        0);
  phydro->w1.Fill(       0);
  pz4c->storage.u.Fill(  0);
  pz4c->storage.u1.Fill( 0);
  pz4c->storage.adm.Fill(0);
  pz4c->storage.mat.Fill(0);

  // Populate hydro/gauge/ADM fields based on TOV soln.
  TOV_populate(this, pin);

  // Initialize remaining z4c variables
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);
  // pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u1);

  // // Impose algebraic constraints
  // pz4c->AlgConstr(pz4c->storage.u);
  // pz4c->Z4cToADM(pz4c->storage.u, pz4c->storage.adm);

  AthenaTensor<Real, TensorSymm::SYM2, 4, 2> g_dd;
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> h_dd;

  // phydro->w.dump("w.txt");
  // std::exit(0);
  // phydro->w_init.Fill(0);
  // phydro->w.Fill(0);
  // phydro->w1.Fill(0);
  // pz4c->storage.mat.Fill(0);

  // Initialise conserved variables
  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord,
                             0, ncells1,
                             0, ncells2,
                             0, ncells3);

  // Initialise matter (also taken care of in task-list)
  pz4c->GetMatter(pz4c->storage.mat, pz4c->storage.adm,
                  phydro->w, pfield->bcc);

  pz4c->ADMConstraints(pz4c->storage.con,
                       pz4c->storage.adm,
                       pz4c->storage.mat,
                       pz4c->storage.u);

  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> H(pz4c->storage.con, Z4c::I_CON_H);

  Real s_H {0.};
  for (int k=pz4c->mbi.kl; k<=pz4c->mbi.ku; ++k)
  for (int j=pz4c->mbi.jl; j<=pz4c->mbi.ju; ++j)
  for (int i=pz4c->mbi.il; i<=pz4c->mbi.iu; ++i)
  {
    const Real vol = pz4c->mbi.dx1(i) * pz4c->mbi.dx2(j) * pz4c->mbi.dx3(k);
    s_H += vol * SQR(H(k,j,i));
  }
  s_H = std::sqrt(s_H);

  std::cout << "TOV_solve (ProblemGenerator,MB): ||H||_2=";
  std::cout << s_H << std::endl;


  // if (s_H > 1e-2)
  // {
  //   pz4c->mbi.x1.print_all();
  //   pz4c->mbi.x2.print_all();
  //   pz4c->mbi.x3.print_all();
  //   // H.array().print_all("%.1e");
  //   // rho.array().print_all("%.1e");
  //   // std::exit(0);
  // }

  /*
  // reinject
  MB_info* mbi = &(pz4c->mbi);

  if (true)
  {
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> rho(pz4c->storage.mat, Z4c::I_MAT_rho);

    // adjust ADM density by initial Hamiltonian error
    for (int k=0; k<=pz4c->mbi.nn3; ++k)
    for (int j=0; j<=pz4c->mbi.nn2; ++j)
    for (int i=0; i<=pz4c->mbi.nn1; ++i)
    {
      rho(k,j,i) += H(k,j,i) / (16.0 * M_PI);
    }

    // update the hydro state-vec
    const Real gamma_adi { pin->GetReal("hydro", "gamma") };
    const Real K     { pin->GetReal("hydro", "k_adi") };
    const Real W { 1.0 };  // Lorentz factor in absence of util_u


    for (int k=mbi->kl; k<=mbi->ku; ++k)
    for (int j=mbi->jl; j<=mbi->ju; ++j)
    {
      for (int i=mbi->il; i<=mbi->iu; ++i)
      {
        const Real rh_adm = rho(k,j,i);
        const Real W = 1.;

        phydro->w(IDN,k,j,i) = (-1.+std::sqrt(1.+4.*K*rh_adm)) / (2. * K);
        phydro->w(IPR,k,j,i) = K*pow(phydro->w(IDN,k,j,i),gamma_adi);
        // phydro->w(IPR,k,j,i) = 0.5*(1./K+2*rh_adm-std::sqrt(1.+4.*K*rh_adm)/K);

        // const Real w_p = phydro->w(IPR,k,j,i);
        // phydro->w(IDN,k,j,i) = (
        //   w_p * (-1.0 + gamma_adi - SQR(W) * gamma_adi) /
        //   (-1.0 + gamma_adi) + rho(k,j,i)
        // ) / SQR(W);

      }
    }


    // Recompute conserved variables
    peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord,
                               0, ncells1,
                               0, ncells2,
                               0, ncells3);

    pz4c->GetMatter(pz4c->storage.mat, pz4c->storage.adm,
                    phydro->w, pfield->bcc);

    // recompute constraints
    pz4c->ADMConstraints(pz4c->storage.con,
                         pz4c->storage.adm,
                         pz4c->storage.mat,
                         pz4c->storage.u);


  }
  */


  // dump files for ini pointwise cmp.
#if defined(DBG_HYDRO_DUMPS)
    pz4c->storage.u.dump("data.ini.u.tov");
    pz4c->storage.con.dump("data.ini.con.tov");
    pz4c->storage.adm.dump("data.ini.adm.tov");
    pz4c->storage.mat.dump("data.ini.mat.tov");
    phydro->w.dump("data.ini.w.tov");

    mbi->x1.dump("vc_x1");
    mbi->x2.dump("vc_x2");
    mbi->x3.dump("vc_x3");
#endif // DBG_HYDRO_DUMPS

  return;
}


namespace {

//-----------------------------------------------------------------------------------
//! \fn int TOV_rhs(Real dr, Real *u, Real *k)
// \brief Calculate right hand sides for TOV equations
//

int TOV_rhs(Real r, Real *u, Real *k)
{
  Real rho = u[TOV_IRHO];
  Real m   = u[TOV_IMASS];
  Real phi = u[TOV_IPHI];
  Real I   = u[TOV_IINT]; // Integral for the isotropic radius

  //  Set pressure and internal energy using equation of state
  //TODO(SB) general EOS call
  Real p = k_adi * std::pow(rho,gamma_adi);
  Real eps = p / (rho*(gamma_adi-1.));
  Real dpdrho = gamma_adi*k_adi*std::pow(rho,gamma_adi-1.0);

  // Total energy density
  Real e = rho*(1. + eps);

  Real num   = m + 4.*PI*r*r*r*p;
  Real den   = r*(r-2.*m);
  Real dphidr = (r==0.) ? 0. : num/den;

  Real drhodr = -(e+p) * dphidr / dpdrho;

  Real dmdr   = 4.*PI*r*r*e;

  // limiting behaviours (note m scales with r)
  Real f      = (r > 0) ? std::sqrt(1.-2.*m/r) : 1.;
  Real dIdr   = (r > 0) ? ( 1.-f )/( r*f )     : 0.;

  k[TOV_IRHO] = drhodr;
  k[TOV_IMASS] = dmdr;
  k[TOV_IPHI] = dphidr;
  k[TOV_IINT] = dIdr;

  int knotfinite = 0;
  for (int v = 0; v < TOV_NVAR; v++) {
    if (!std::isfinite(k[v])) knotfinite++;
  }
  return knotfinite;
}


//------------------------------------------------------------------------------------
//! \fn int TOV_solve(Real rhoc, Real rmin, Real dr, int *npts)
// \brief Calculate right hand sides for TOV equations
//

int TOV_solve(Real rhoc, Real rmin, Real dr, int *npts)
{

  std::stringstream msg;

  // Alloc buffers for ODE solve
  const int maxsize = *npts - 1;
  Real u[TOV_NVAR];

  // Set central values of pressure internal energy using EOS
  //TODO(SB) general EOS call
  const Real pc = k_adi*std::pow(rhoc,gamma_adi);
  const Real epslc = pc/(rhoc*(gamma_adi-1.));

  const Real ec = rhoc*(1.+epslc);

  // Data at r = 0^+
  Real r = rmin;
  u[TOV_IRHO] = rhoc;
  u[TOV_IMASS] = 4./3.* PI * ec * rmin*rmin*rmin;
  u[TOV_IPHI] = 0.;
  u[TOV_IINT] = 0.;

  printf("TOV_solve: solve TOV star (only once)\n");
  printf("TOV_solve: dr   = %.16e\n",dr);
  printf("TOV_solve: npts_max = %d\n",maxsize);
  printf("TOV_solve: rhoc = %.16e\n",rhoc);
  printf("TOV_solve: ec   = %.16e\n",ec);
  printf("TOV_solve: pc   = %.16e\n",pc);

  // Integrate from rmin to R : rho(R) ~ 0
  Real rhoo = rhoc;
  int stop = 0;
  int n = 0;

  int interp_n = 1;
  const int interp_maxsize = tov->interp_npts - 1;

  // store IC
  tov->data[itov_rsch][0] = r;
  tov->data[itov_rho][0]  = u[TOV_IRHO];
  tov->data[itov_mass][0] = u[TOV_IMASS];
  tov->data[itov_phi][0]  = u[TOV_IPHI];
  // Mul. by C later
  tov->data[itov_riso][0] = (r) * std::exp(u[TOV_IINT]);


  const Real rho_zero = 0.; //TODO(SB) use atmosphere level

  if (0)
  {
    // To use: needs tov->interp_npts = interp_n = n; added after integ below

    Real u1[TOV_NVAR],u2[TOV_NVAR],u3[TOV_NVAR],k[TOV_NVAR];
    const Real oosix = 1./6.;
    while (n < maxsize)
    {

      // u_1 = u + dt/2 rhs(u)
      stop += TOV_rhs(r, u, k);
      for (int v = 0; v < TOV_NVAR; v++)
        u1[v] = u[v] + 0.5*dr*k[v];
      // u_2 = u + dt/2 rhs(u_1)
      stop += TOV_rhs(r, u1, k);
      for (int v = 0; v < TOV_NVAR; v++)
        u2[v] = u[v] + 0.5*dr*k[v];
      // u_3 = u + dt rhs(u_2)
      stop += TOV_rhs(r, u2, k);
      for (int v = 0; v < TOV_NVAR; v++)
        u3[v] = u[v] + dr*k[v];
      // u = 1/6 ( -2 u + 2 u_1 + 4 u_2 + 2 u_3 + dt rhs(u_3) )
      stop += TOV_rhs(r, u3, k);
      for (int v = 0; v < TOV_NVAR; v++)
      {
        u[v] = oosix*( 2.*( - u[v] + u1[v] + u3[v] ) + 4.*u2[v] + dr*k[v] );
      }

      if (stop)
      {
        msg << "### FATAL ERROR in function [TOV_solve]"
            << std::endl << "TOV r.h.s. not finite";
        ATHENA_ERROR(msg);
      }

      // Stop if radius reached
      rhoo = u[TOV_IRHO];
      if (rhoo < rho_zero)
      {
        break;
      }

      // Store data
      tov->data[itov_rsch][n] = r;
      tov->data[itov_rho][n] =  u[TOV_IRHO];
      tov->data[itov_mass][n] = u[TOV_IMASS];
      tov->data[itov_phi][n] =  u[TOV_IPHI];
      tov->data[itov_riso][n] = r * std::exp(u[TOV_IINT]); // Multiply by C later

      // Prepare next step
      n++;
      r = n * dr;
    }
  }

  int n_halving = 0;
  bool tol_surf_achieved = false;
  // RK4 ----------------------------------------------------------------------
  if (1)
  {

    Real ut[TOV_NVAR];  // next argument for rhs
    Real k1[TOV_NVAR], k2[TOV_NVAR], k3[TOV_NVAR], k4[TOV_NVAR];  // stages

    const Real c2 = 0.5, a21 = 0.5;
    const Real c3 = 0.5, a31 = 0.0, a32 = 0.5;
    const Real c4 = 1.0, a41 = 0.0, a42 = 0.0, a43 = 1.0;

    const Real b1 = 1./6., b2 = 1./3., b3 = 1./3., b4 = 1./6.;

    while (n < maxsize)
    {
      // stage 1
      for (int v = 0; v < TOV_NVAR; v++)
        ut[v] = u[v];

      stop += TOV_rhs(r, ut, k1);

      // stage 2
      for (int v = 0; v < TOV_NVAR; v++)
        ut[v] = u[v] + (a21 * k1[v]) * dr;

      stop += TOV_rhs(r+c2*dr, ut, k2);

      // stage 3
      for (int v = 0; v < TOV_NVAR; v++)
        ut[v] = u[v] + (a31 * k1[v] +
                        a32 * k2[v]) * dr;

      stop += TOV_rhs(r+c3*dr, ut, k3);

      // stage 4
      for (int v = 0; v < TOV_NVAR; v++)
        ut[v] = u[v] + (a41 * k1[v] +
                        a42 * k2[v] +
                        a43 * k3[v]) * dr;

      stop += TOV_rhs(r+c4*dr, ut, k4);

      // assemble (u now stores @ r + dr)
      for (int v = 0; v < TOV_NVAR; v++)
      {
        u[v] = u[v] + dr * (
          b1 * k1[v] + b2 * k2[v] + b3 * k3[v] + b4 * k4[v]
        );
      }

      if (stop)
      {
        msg << "### FATAL ERROR in function [TOV_solve]"
            << std::endl << "TOV r.h.s. not finite";
        ATHENA_ERROR(msg);
      }

      // Stop if radius reached
      rhoo = u[TOV_IRHO];
      bool exceeded_radius = rhoo < rho_zero;

      if (exceeded_radius)
      {
        if (dr < tov->surf_dr)
        {
          tol_surf_achieved = true;

          // revert last step
          for (int v = 0; v < TOV_NVAR; v++)
          {
            u[v] = u[v] - dr * (
              b1 * k1[v] + b2 * k2[v] + b3 * k3[v] + b4 * k4[v]
            );
          }
        }
        else
        {
          // printf("TOV_solve: halving dr (near surface) r=%.16e\n", r);

          // revert last step
          for (int v = 0; v < TOV_NVAR; v++)
          {
            u[v] = u[v] - dr * (
              b1 * k1[v] + b2 * k2[v] + b3 * k3[v] + b4 * k4[v]
            );
          }

          dr = dr / 2;
          n_halving++;
        }
      }
      else
      {
        // Prepare next step
        n++;
        r += dr;
      }

      if ((r >= (interp_n * (tov->interp_dr))) or tol_surf_achieved)
      {
        // Need to store data
        tov->data[itov_rsch][interp_n] = r;
        tov->data[itov_rho][interp_n]  = u[TOV_IRHO];
        tov->data[itov_mass][interp_n] = u[TOV_IMASS];
        tov->data[itov_phi][interp_n]  = u[TOV_IPHI];
        // Mul. by C later
        tov->data[itov_riso][interp_n] = (r) * std::exp(u[TOV_IINT]);
        interp_n++;

        if (interp_n > interp_maxsize-1)
        {
          msg << "### FATAL ERROR in function [TOV_solve]"
              << std::endl << "interp_n - increase";
          ATHENA_ERROR(msg);
        }
      }

      if (tol_surf_achieved)
      {
        break;
      }

    }

  }

  printf("TOV_solve: integ. done\n");
  // --------------------------------------------------------------------------

  if (n >= maxsize)
  {
    msg << "### FATAL ERROR in function [TOV_solve]"
        << std::endl << "Star radius not reached. (Try increasing 'npts')";
    ATHENA_ERROR(msg);
  }

  *npts = n;
  tov->npts = n;
  tov->interp_npts = interp_n;

  tov->R = tov->data[itov_rsch][tov->interp_npts-1]; // r;
  tov->M = tov->data[itov_mass][tov->interp_npts-1]; // u[TOV_IMASS];


  // Re-Alloc 1D data
  for (int v = 0; v < itov_nv; v++)
    tov->data[v] = (Real*) realloc(tov->data[v], (tov->interp_npts)*sizeof(Real));

  // dump ---------------------------------------------------------------------
  /*
  AthenaArray<Real> array_TOV(itov_nv, interp_n);
  for (int v = 0; v < itov_nv; v++)
  for (int ix = 0; ix < interp_n; ++ix)
  {
    array_TOV(v,ix) = tov->data[v][ix];
  }
  array_TOV.dump("tov.pre.ini");
  */
  // --------------------------------------------------------------------------


  // Match to exterior
  const Real phiR  = tov->data[itov_phi][tov->interp_npts-1]; // u[TOV_IPHI];
  // const Real IR    = u[TOV_IINT];
  const Real IR    = std::log(tov->data[itov_riso][tov->interp_npts-1] / tov->R);
  const Real phiRa = 0.5*std::log(1.-2.*tov->M/tov->R);
  const Real C     = 1./(2*tov->R) * (std::sqrt(tov->R*tov->R-2*tov->M*tov->R)+tov->R-tov->M) * std::exp(-IR);

  for (int n = 0; n < tov->interp_npts; n++) {
    tov->data[itov_phi][n] += - phiR + phiRa;
    tov->data[itov_riso][n] *= C; // riso = rsch * C * exp(IINT)
  }

  tov->Riso = tov->data[itov_riso][tov->interp_npts-1];

  // Pressure
  //TODO(SB) general EOS call
  for (int n = 0; n < tov->interp_npts; n++) {
    tov->data[itov_pre][n] = std::pow(tov->data[itov_rho][n],gamma_adi) * k_adi;
  }

  // Other metric fields
  for (int n = 0; n < tov->interp_npts; n++) {
    tov->data[itov_psi4][n] = std::pow(tov->data[itov_rsch][n]/tov->data[itov_riso][n], 2);
    tov->data[itov_lapse][n] = std::exp(tov->data[itov_phi][n]);
  }

  // Metric field (regular origin)
  tov->lapse_0 = std::exp(- phiR + phiRa);
  tov->psi4_0 = 1/(C*C);

  // Done!
  printf("TOV_solve: npts = %d\n",tov->npts);
  printf("TOV_solve: inter_npts = %d\n",tov->interp_npts);
  printf("TOV_solve: R(sch) = %.16e\n",tov->R);
  printf("TOV_solve: R(iso) = %.16e\n",tov->Riso);
  printf("TOV_solve: M = %.16e\n",tov->M);
  printf("TOV_solve: lapse(0) = %.16e\n",tov->lapse_0);
  printf("TOV_solve: psi4(0) = %.16e\n",tov->psi4_0);

  printf("TOV_solve: lapse(R(iso)) = %.16e\n",tov->data[itov_lapse][tov->interp_npts-1]);
  printf("TOV_solve: psi4(R(iso)) = %.16e\n", tov->data[itov_psi4][tov->interp_npts-1]);

  // dump ---------------------------------------------------------------------
  /*
  for (int v = 0; v < itov_nv; v++)
  for (int ix = 0; ix < interp_n; ++ix)
  {
    array_TOV(v,ix) = tov->data[v][ix];
  }
  array_TOV.dump("tov.post.ini");
  */
  // --------------------------------------------------------------------------


  return 0;
}

//-----------------------------------------------------------------------------------------
//! \fn int interp_locate(Real *x, int Nx, Real xval)
// \brief Bisection to find closest point in interpolating table
//
int interp_locate(Real *x, int Nx, Real xval)
{
  int ju,jm,jl;
  int ascnd;
  jl=-1;
  ju=Nx;
  if (xval <= x[0])
  {
    return 0;
  }
  else if (xval >= x[Nx-1])
  {
    return Nx-1;
  }
  ascnd = (x[Nx-1] >= x[0]);

  while (ju-jl > 1)
  {
    jm = (ju+jl) >> 1;
    if (xval >= x[jm] == ascnd)
      jl=jm;
    else
      ju=jm;
  }
  return jl;
}


//--------------------------------------------------------------------------------------
//! \fn void interp_lag4(Real *f, Real *x, int Nx, Real xv,
//                       Real *fv_p, Real *dfv_p, Real *ddfv_p)
// \brief 4th order lagrangian interpolation with derivatives
// Returns the interpolated values fv at xv of the fuction f(x)
// together with 1st and 2nd derivatives dfv, ddfv
void interp_lag4(Real *f, Real *x, int Nx, Real xv,
      Real *fv_p, Real *dfv_p, Real *ddfv_p)
{
  int i = interp_locate(x,Nx,xv);
  if( i < 1 ){
    i = 1;
  }
  if( i > (Nx-3) ){
    i = Nx-3;
  }
  const Real ximo =  x[i-1];
  const Real xi   =  x[i];
  const Real xipo =  x[i+1];
  const Real xipt =  x[i+2];
  const Real C1   = (f[i] - f[i-1])/(xi - ximo);
  const Real C2   = (-f[i] + f[i+1])/(-xi + xipo);
  const Real C3   = (-f[i+1] + f[i+2])/(-xipo + xipt);
  const Real CC1  = (-C1 + C2)/(-ximo + xipo);
  const Real CC2  = (-C2 + C3)/(-xi + xipt);
  const Real CCC1 = (-CC1 + CC2)/(-ximo + xipt);
  *fv_p   = f[i-1] + (-ximo + xv)*(C1 + (-xi + xv)*(CC1 + CCC1*(-xipo + xv)));
  *dfv_p  = C1 - (CC1 - CCC1*(xi + xipo - 2.*xv))*(ximo - xv) + (-xi + xv)*(CC1 + CCC1*(-xipo + xv));
  *ddfv_p = 2.*(CC1 - CCC1*(xi + ximo + xipo - 3.*xv));

  // LO: debug
  // if (i > (Nx-1))
  // {
  //   i = Nx-1;
  // }

  // const Real d1x_0 = x[i  ] - x[i-1];
  // const Real d2x_1 = x[i+1] - x[i-1];

  // const Real d1f_0 = f[i  ] - f[i-1];
  // const Real d1f_1 = f[i+1] - f[i  ];

  // const Real d1fx_0 = d1f_0 / d1x_0;
  // const Real dd1fx_1 = d1f_1 - d1f_0;


  // // *fv_p = (
  // //   f[i-1] + (xv - x[i-1]) * (
  // //     d1fx_0 + dd1fx_1 * (xv-x[i]) / d2x_1
  // //   )
  // // );

  // *fv_p = f[i-1] + (xv - x[i-1]) * d1fx_0;

}

void TOV_populate(MeshBlock *pmb,
                  ParameterInput *pin)
{
  using namespace LinearAlgebra;

  Hydro       * phydro { pmb->phydro };
  Z4c         * pz4c   { pmb->pz4c   };
  GRDynamical * pcoord { static_cast<GRDynamical*>(pmb->pcoord) };

  // container with idx / grids pertaining z4c
  MB_info* mbi = &(pz4c->mbi);

  // for readability
  const int D = NDIM + 1;
  const int N = NDIM;

  typedef AthenaArray< Real>                         AA;
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 0> AT_N_sca;
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 1> AT_N_vec;
  typedef AthenaTensor<Real, TensorSymm::SYM2, N, 2> AT_N_sym;
  typedef AthenaTensor<Real, TensorSymm::SYM2, D, 2> AT_D_sym;
  typedef AthenaTensor<Real, TensorSymm::NONE, D, 1> AT_D_vec;

  // ambient metric derivative
  typedef AthenaTensor<Real, TensorSymm::SYM2, D, 3> AT_D_Dsym;

  // set up center / boost
  AA x_0(3), v_b(3);

  x_0(0) = pin->GetOrAddReal("problem","x_0_x1", 0.0);
  x_0(1) = pin->GetOrAddReal("problem","x_0_x2", 0.0);
  x_0(2) = pin->GetOrAddReal("problem","x_0_x3", 0.0);

  v_b(0) = pin->GetOrAddReal("problem","v_b_x1", 0.0);;
  v_b(1) = pin->GetOrAddReal("problem","v_b_x2", 0.0);;
  v_b(2) = pin->GetOrAddReal("problem","v_b_x3", 0.0);;

  const bool boost { SQR(v_b(0)) + SQR(v_b(1)) + SQR(v_b(2)) > 0 };

  // Active Lorentz:
  // u^mu' = (L_a )^mu'_nu u^nu
  // v_nu' = (iL_a)^mu_nu' v_mu
  AA L_a(4,4), iL_a(4,4);
  TOV_geom::Lorentz4Boost(L_a,  v_b, false);
  // inv is -v_b which is just passive xform
  TOV_geom::Lorentz4Boost(iL_a, v_b, true);

  AA X(4), Xp(4);  // space-time coordinate and primed (xformed)
  X.Fill(0);

  // scratch ------------------------------------------------------------------
  // reuse the same scratch, take largest Nx
  const int Nx1 { std::max(pmb->nverts1, mbi->nn1) };

  // coordinate
  AA sp_x_(N, Nx1);

  AA x_(Nx1);
  AA y_(Nx1);
  AA z_(Nx1);
  AA r_(Nx1);

  // geometric
  AT_D_sym st_g_dd_(Nx1);   // (s)pace-(t)ime
  AT_D_sym st_g_uu_(Nx1);
  AT_N_sym sp_g_dd_(Nx1);   // (sp)atial
  AT_N_sym sp_g_uu_(Nx1);
  AT_N_sym sp_K_dd_(Nx1);

  AT_N_sca alpha_(  Nx1);
  AT_N_sca dralpha_(Nx1);   // radial cpt.
  AT_N_vec d1alpha_(Nx1);   // Cart. cpts.

  AT_N_vec sp_beta_u_(    Nx1);
  AT_N_vec sp_beta_d_(    Nx1);
  AT_N_sca sp_norm2_beta_(Nx1);

  AT_N_sca psi4_(  Nx1);
  AT_N_sca drpsi4_(Nx1);    // radial cpt.
  AT_N_vec d1psi4_(Nx1);    // Cart. cpts.

  AT_D_Dsym st_dg_ddd_(   Nx1);  // metric deriv.
  AT_D_Dsym st_Gamma_ddd_(Nx1);  // Christoffel
  AT_D_Dsym st_Gamma_udd_(Nx1);

  // geometric: transformed to (p)rimed
  // AT_N_sca alphap_(    Nx1);
  AT_D_sym st_gp_dd_(  Nx1);
  AT_D_sym st_gp_uu_(  Nx1);
  AT_N_sym sp_gp_dd_(  Nx1);
  AT_N_sym sp_gp_uu_(  Nx1);
  AT_N_sym sp_Kp_dd_(  Nx1);
  AT_N_vec sp_betap_u_(Nx1);
  AT_N_vec sp_betap_d_(Nx1);

  AT_D_Dsym st_Gammap_udd_(Nx1);  // transformed Christoffel

  // matter
  AT_N_sca w_rho_(   Nx1);
  AT_N_sca w_p_(     Nx1);
  AT_N_vec w_util_u_(Nx1);
  AT_D_vec st_u_u_(  Nx1);
  AT_D_vec st_up_u_( Nx1);

  // Lorentz factor & utilde norm squared
  AT_N_sca W_(              Nx1);
  AT_N_sca sp_utilde_norm2_(Nx1);


  // slicings -----------------------------------------------------------------
  // geometric
  AT_N_sca alpha_init( pz4c->storage.u,   Z4c::I_Z4c_alpha);
  AT_N_vec beta_u_init(pz4c->storage.u,   Z4c::I_Z4c_betax);
  AT_N_sym g_dd_init(  pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sym K_dd_init(  pz4c->storage.adm, Z4c::I_ADM_Kxx);
  AT_N_sca psi4_init(  pz4c->storage.adm, Z4c::I_ADM_psi4);

  // matter
  AT_N_sca sl_w_rho_init(   phydro->w_init, IDN);
  AT_N_sca sl_w_p_init(     phydro->w_init, IPR);
  AT_N_vec sl_w_util_u_init(phydro->w_init, IVX);

  // for debugging
  AT_N_sca sl_rho( pz4c->storage.mat, Z4c::I_MAT_rho);
  AT_N_vec sl_S_d( pz4c->storage.mat, Z4c::I_MAT_Sx);
  AT_N_sym sl_S_dd(pz4c->storage.mat, Z4c::I_MAT_Sxx);


  // Star mass & radius
  const Real M = tov->M;     // Mass of TOV star
  const Real R = tov->Riso;  // Isotropic Radius of TOV star

  // Interpolation dummy
  Real dummy;

  // Debug dump ---------------------------------------------------------------
  if (0)
  {
    const Real R_max = 50;
    const int N_R   = 5000;
    const Real dR = R_max / (N_R - 1);

    Real interp_val;

    AthenaArray<Real> data(itov_nv, N_R);
    data.Fill(0.);

    for (int ix=0; ix<N_R; ++ix)
    {
      const Real r = ix * dR;
      data(0, ix) = r;

      if (r < R)
      {
        interp_lag4(
          tov->data[itov_rho],
          tov->data[itov_riso],
          tov->interp_npts,
          r,
          &data(itov_rho, ix),
          &dummy,
          &dummy);

        interp_lag4(
          tov->data[itov_lapse],
          tov->data[itov_riso],
          tov->interp_npts,
          r,
          &data(itov_lapse, ix),
          &dummy,
          &dummy);

        interp_lag4(
          tov->data[itov_psi4],
          tov->data[itov_riso],
          tov->interp_npts,
          r,
          &data(itov_psi4, ix),
          &dummy,
          &dummy);

        interp_lag4(
          tov->data[itov_mass],
          tov->data[itov_riso],
          tov->interp_npts,
          r,
          &data(itov_mass, ix),
          &dummy,
          &dummy);

      }
      else
      {
        data(itov_rho, ix)   = 0;
        data(itov_mass, ix)  = M;
        // 4, 5
        data(itov_psi4, ix)  = std::pow((1.+0.5*M/r),4);
        data(itov_lapse, ix) = ((r-M/2.)/(r+M/2.));
      }

      data(1, ix) = k_adi*pow(data(itov_rho, ix),gamma_adi);
      data(5, ix) = R;
    }
    data.dump("tov_r.ini");
    std::exit(0);

  }
  // --------------------------------------------------------------------------

  // Initialize primitive values on CC grid -----------------------------------
  Real up_r { 0. };  // for perturbations

  for (int k=0; k<pmb->ncells3; ++k)
  {
    X(3) = pcoord->x3v(k)-x_0(2);

    for (int j=0; j<pmb->ncells2; ++j)
    {
      X(2) = pcoord->x2v(j)-x_0(1);

      const int il = 0;
      const int iu = pmb->ncells1-1;

      for (int i=il; i<=iu; ++i)
      {
        X(1) = pcoord->x1v(i)-x_0(0);

        // Active Lorentz transform on translated coordinates if boosted
        if (boost)
        {
          ApplyLinearTransform(L_a, Xp, X);
        }

        sp_x_(0,i) = x_(i) = (boost) ? Xp(1) : X(1);
        sp_x_(1,i) = y_(i) = (boost) ? Xp(2) : X(2);
        sp_x_(2,i) = z_(i) = (boost) ? Xp(3) : X(3);

        // Isotropic radius
        r_(i) = std::sqrt(SQR(x_(i))+SQR(y_(i))+SQR(z_(i)));
        const Real rho_pol = std::sqrt(SQR(x_(i))+SQR(y_(i)));

        const Real costh = z_(i)/r_(i);
        const Real sinth = rho_pol/r_(i);
        const Real cosphi = x_(i)/rho_pol;
        const Real sinphi = y_(i)/rho_pol;

        if (r_(i)<R)
        {
          // Interpolate rho to star interior
          interp_lag4(
            tov->data[itov_rho],
            tov->data[itov_riso],
            tov->interp_npts,
            r_(i),
            &w_rho_(i),
            &dummy,
            &dummy);

          // Pressure from EOS
          // TODO (SB) general EOS call
          w_p_(i) = k_adi*pow(w_rho_(i),gamma_adi);

          // Add perturbation
          const Real x_kji = r_(i) / R;
          up_r = (v_amp>0) ? (0.5*v_amp*(3.0*x_kji - x_kji*x_kji*x_kji)) : 0.0;
        }
        else
        {
          // Let the EOS decide how to set the atmosphere on the exterior
          w_rho_(i) = 0.0;
          w_p_(i)   = 0.0;
          up_r      = 0.0;
        }

        w_util_u_(0,i) = up_r*sinth*cosphi;
        w_util_u_(1,i) = up_r*sinth*sinphi;
        w_util_u_(2,i) = up_r*costh;
      }

      // Transform if needed (assemble ambient M quantities here)
      // Need to construct CC: lapse, shift, spatial metric, space-time metric
      if (boost)
      {
        // Untransformed shift is taken as 0 initially
        sp_beta_u_.Fill(0);

        // Lapse, conformal factor & Cart. derivative assembly ----------------
        for (int i=il; i<=iu; ++i)
        {
          if (r_(i)<R)
          {
            // Star interior
            interp_lag4(
              tov->data[itov_lapse],
              tov->data[itov_riso],
              tov->interp_npts,
              r_(i),
              &alpha_(i),
              &dummy, // &dralpha_(i),
              &dummy);

            interp_lag4(
              tov->data[itov_psi4],
              tov->data[itov_riso],
              tov->interp_npts,
              r_(i),
              &psi4_(i),
              &dummy, // &drpsi4_(i),
              &dummy);
          }
          else
          {
            // Exterior
            alpha_(i)   = ((r_(i)-M/2.)/(r_(i)+M/2.));
            psi4_(i)   = std::pow((1.+0.5*M/r_(i)),4);
          }
        }
        // --------------------------------------------------------------------

        // metric assembly ----------------------------------------------------
        sp_g_dd_.Fill(0);  // off-diagonals are zero
        for (int a=0; a<N; ++a)
        for (int i=il; i<=iu; ++i)
        {
          sp_g_dd_(a,a,i) = psi4_(i);
        }

        Inv3Metric(sp_g_dd_, sp_g_uu_, il, iu);

        Assemble_ST_Metric_uu(
          st_g_uu_, sp_g_uu_, alpha_, sp_beta_u_,
          il, iu
        );

        // u = W (n + U); U is taken as utilde
        // W = (1 - ||v,v||_g_sp^2)^{-1/2}; between fluid fr. and Eul. obs.
        // u^0 = W / alpha
        // u^i = U^i
        MetricNorm2Vector(sp_utilde_norm2_, w_util_u_, sp_g_dd_, il, iu);

        for (int i=il; i<=iu; ++i)
        {
          W_(i) = std::sqrt(1. + sp_utilde_norm2_(i));
          st_u_u_(0,i) = W_(i) / alpha_(i);
          st_u_u_(1,i) = w_util_u_(0,i);  // BD: is this what the pert.
          st_u_u_(2,i) = w_util_u_(1,i);  //     was meant to represent?
          st_u_u_(3,i) = w_util_u_(2,i);
        }

        // Lorentz transform quantities
        // TOV_geom::ApplyLinearTransform(iL_a, iL_a, st_gp_dd_, st_g_dd_,
        //                                il, iu);
        ApplyLinearTransform(L_a, L_a, st_gp_uu_, st_g_uu_, il, iu);
        ApplyLinearTransform(L_a, st_up_u_, st_u_u_, il, iu);

        ExtractFrom_ST_Metric_uu(
          st_gp_uu_, sp_gp_uu_, alpha_, sp_betap_u_,
          il, iu
        );

        // ExtractFrom_ST_Metric_dd(
        //   st_gp_dd_, sp_gp_dd_, alpha_, sp_betap_d_,
        //   il, iu
        // );


        for (int i=il; i<=iu; ++i)
        {
          W_(i) = st_up_u_(0,i)*alpha_(i);

          // st has (s)pace-(time) ranges so inc. +1
          w_util_u_(0,i) = W_(i) * (st_up_u_(1,i)/W_(i) +
                                    sp_betap_u_(0,i)/alpha_(i));
          w_util_u_(1,i) = W_(i) * (st_up_u_(2,i)/W_(i) +
                                    sp_betap_u_(1,i)/alpha_(i));
          w_util_u_(2,i) = W_(i) * (st_up_u_(3,i)/W_(i) +
                                    sp_betap_u_(2,i)/alpha_(i));
        }

      }

      // Populate hydro quantities additively ---------------------------------
      for (int i=il; i<=iu; ++i)
      {
        sl_w_rho_init( k,j,i) += w_rho_(i);
        sl_w_p_init(k,j,i)    += w_p_(i);
      }
      for (int a=0; a<N; ++a)
      for (int i=il; i<=iu; ++i)
      {
        sl_w_util_u_init(a,k,j,i) += w_util_u_(a,i);
      }
    }


  }

  // Populate hydro registers with the initial data
  phydro->w = phydro->w1 = phydro->w_init;

  // Initialise metric variables on geometric grid ----------------------------
  // Sets alpha, beta, g_ij, K_ij

  for (int k=0; k<mbi->nn3; ++k)
  {
    X(3) = mbi->x3(k)-x_0(2);

    for (int j=0; j<mbi->nn2; ++j)
    {
      X(2) = mbi->x2(j)-x_0(1);

      const int il = 0;
      const int iu = mbi->nn1-1;

      for (int i=il; i<=iu; ++i)
      {
        X(1) = mbi->x1(i)-x_0(0);

        // Active Lorentz transform on translated coordinates if boosted
        if (boost)
        {
          ApplyLinearTransform(L_a, Xp, X);
        }

        sp_x_(0,i) = x_(i) = (boost) ? Xp(1) : X(1);
        sp_x_(1,i) = y_(i) = (boost) ? Xp(2) : X(2);
        sp_x_(2,i) = z_(i) = (boost) ? Xp(3) : X(3);

        // Isotropic radius
        r_(i) = std::sqrt(SQR(x_(i))+SQR(y_(i))+SQR(z_(i)));

        if (r_(i)<R)
        {
          // Interior metric, lapse and conf.fact.
          if (r_(i) > 0.)
          {
            interp_lag4(
              tov->data[itov_lapse],
              tov->data[itov_riso],
              tov->interp_npts,
              r_(i),
              &alpha_(i),
              &dralpha_(i),
              &dummy);

            interp_lag4(
              tov->data[itov_psi4],
              tov->data[itov_riso],
              tov->interp_npts,
              r_(i),
              &psi4_(i),
              &drpsi4_(i),
              &dummy);
          }
          else
          {
            alpha_(i)   = tov->lapse_0;
            dralpha_(i) = 0.0;
            psi4_(i)    = tov->psi4_0;
            drpsi4_(i)  = 0.0;
          }
        }
        else
        {
          // Exterior schw. metric, lapse and conf.fact.
          alpha_(i)   = ((r_(i)-M/2.)/(r_(i)+M/2.));
          dralpha_(i) = M/((0.5*M+r_(i))*(0.5*M+r_(i)));
          psi4_(i)    = std::pow((1.+0.5*M/r_(i)),4.);
          drpsi4_(i)  = -2.*M*std::pow(1.+0.5*M/r_(i),3)/(r_(i)*r_(i));
        }
      }

      // Untransformed shift is taken as 0 initially
      sp_beta_u_.Fill(0);
      sp_beta_d_.Fill(0);
      sp_K_dd_.Fill(0);    // this will only change under boost

      sp_g_dd_.Fill(0);    // off-diagonals are zero
      for (int a=0; a<N; ++a)
      for (int i=il; i<=iu; ++i)
      {
        sp_g_dd_(a,a,i) = psi4_(i);
      }

      // Transform if needed (assemble ambient M quantities here)
      // Need to construct: lapse, shift, spatial metric, space-time metric
      if (boost)
      {
        // metric assembly ----------------------------------------------------
        Inv3Metric(sp_g_dd_, sp_g_uu_, il, iu);
        // raise idx
        SlicedVecMet3Contraction(sp_beta_u_, sp_beta_d_, sp_g_uu_, il, iu);

        Assemble_ST_Metric_uu(st_g_uu_, sp_g_uu_, alpha_, sp_beta_u_, il, iu);
        Assemble_ST_Metric_dd(st_g_dd_, sp_g_dd_, alpha_, sp_beta_d_, il, iu);

        // prepare metric derivativess
        d1alpha_.ZeroClear();
        d1psi4_.ZeroClear();

        for (int a=0; a<N; ++a)
        for (int i=il; i<=iu; ++i)
        {
          // if r == 0 then drX == 0
          if (r_(i) > 0.)
          {
            d1alpha_(a,i) += dralpha_(i) * sp_x_(a,i) / r_(i);
            d1psi4_( a,i) += drpsi4_( i) * sp_x_(a,i) / r_(i);
          }
        }


        // derivatives of (s)pace-(t)ime metric
        st_dg_ddd_.ZeroClear();

        // only iterate on non-zero spatial derivatives of ambient metric
        for (int c=0; c<N; ++c)      // SP idx ranges
        for (int i=il; i<=iu; ++i)
        {
          // lapse (sp. derivs.)
          st_dg_ddd_(c+1,0,0,i) = -2 * alpha_(i) * d1alpha_(c,i);
        }

        // spatial metric only has non-trivial elements on diag.
        for (int c=0; c<N; ++c)      // SP idx ranges
        for (int a=0; a<N; ++a)
        for (int i=il; i<=iu; ++i)
        {
          st_dg_ddd_(c+1,a+1,a+1,i) = d1psi4_(c,i);
        }

        // Form connection coefficients
        for (int c = 0; c < D; ++c)  // ST idx ranges
        for (int a = 0; a < D; ++a)
        for (int b = a; b < D; ++b)
        for (int i=il; i<=iu; ++i)
        {
          st_Gamma_ddd_(c,a,b,i) = 0.5*(st_dg_ddd_(a,b,c,i) +
                                        st_dg_ddd_(b,a,c,i) -
                                        st_dg_ddd_(c,a,b,i));
        }

        st_Gamma_udd_.ZeroClear();
        for (int c = 0; c < D; ++c)  // ST idx ranges
        for (int a = 0; a < D; ++a)
        for (int b = a; b < D; ++b)
        for (int d = 0; d < D; ++d)
        for (int i=il; i<=iu; ++i)
        {
          st_Gamma_udd_(c,a,b,i) += st_g_uu_(c,d,i)*st_Gamma_ddd_(d,a,b,i);
        }

        // Lorentz transform quantities
        ApplyLinearTransform(L_a, L_a, st_gp_uu_, st_g_uu_, il, iu);
        ApplyLinearTransform(iL_a, iL_a, st_gp_dd_, st_g_dd_, il, iu);


        // extract alpha, beta_u, gamma_dd
        ExtractFrom_ST_Metric_uu(
          st_gp_uu_, sp_gp_uu_, alpha_, sp_betap_u_,
          il, iu
        );

        ExtractFrom_ST_Metric_dd(
          st_gp_dd_, sp_gp_dd_, alpha_, sp_betap_d_,
          il, iu
        );

        // linear transform on st_Gamma_udd_ -
        // due to linearity can use tensorial form (2nd deg. is 0)
        ApplyLinearTransform(
          L_a, iL_a, iL_a,
          st_Gammap_udd_, st_Gamma_udd_,
          il, iu
        );

        // extract extrinsic curvature from st_Gammap_udd_ component
        for (int b=0; b<N; ++b)     // SP idx ranges
        for (int a=0; a<=b; ++a)
        for (int i=il; i<=iu; ++i)
        {
          sp_Kp_dd_(a,b,i) = -alpha_(i) * st_Gammap_udd_(0,a+1,b+1,i);
        }

        // swap roles of (un)primed (no need to do anything special below)
        sp_beta_u_.SwapAthenaTensor(sp_betap_u_);
        sp_g_dd_.SwapAthenaTensor(  sp_gp_dd_);
        sp_K_dd_.SwapAthenaTensor(  sp_Kp_dd_);

      }

      // Populate metric quantities additively --------------------------------

      // scalars
      for (int i=il; i<=iu; ++i)
      {
        alpha_init(k,j,i) += alpha_(i);
        psi4_init( k,j,i) += psi4_( i);
      }

      // vectors
      for (int a=0; a<N; ++a)
      for (int i=il; i<=iu; ++i)
      {
        beta_u_init(a,k,j,i) += sp_beta_u_(a,i);
      }

      // symmetric 2-tensors
      for (int b=0; b<N; ++b)
      for (int a=0; a<=b; ++a)
      for (int i=il; i<=iu; ++i)
      {
        g_dd_init(a,b,k,j,i) += sp_g_dd_(a,b,i);
        K_dd_init(a,b,k,j,i) += sp_K_dd_(a,b,i);
      }


      // Check regular --------------------------------------------------------
      const bool geom_fin = (alpha_.is_finite() and
                             psi4_.is_finite()  and
                             sp_beta_u_.is_finite() and
                             sp_g_dd_.is_finite() and
                             sp_K_dd_.is_finite());
      if (!geom_fin)
      {
        alpha_.array().print_all("%.1e");
        psi4_.array().print_all("%.1e");
        sp_beta_u_.array().print_all("%.1e");
        sp_g_dd_.array().print_all("%.1e");
        sp_K_dd_.array().print_all("%.1e");

        L_a.print_all("%.1e");
        iL_a.print_all("%.1e");

        d1alpha_.array().print_all("%.1e");
        d1psi4_.array().print_all("%.1e");
        r_.print_all("%.1e");

        std::exit(0);
      }


    }
  }

  // const bool geom_fin = (g_dd_init.is_finite() and
  //                        K_dd_init.is_finite());
  // if (!geom_fin)
  // {
  //   std::cout << alpha_init.is_finite() << std::endl;
  //   std::cout << beta_u_init.is_finite() << std::endl;
  //   std::cout << psi4_init.is_finite() << std::endl;

  //   std::cout << g_dd_init.is_finite() << std::endl;
  //   std::cout << K_dd_init.is_finite() << std::endl;

  //   std::exit(0);
  // }


  // register u has been populated directly, u1 does not need to be populated

#if MAGNETIC_FIELDS_ENABLED
  // Prepare CC index bounds (BD: TODO why not just use [0, ncellsN] ?)
  int ilcc = is - NGHOST;
  int iucc = ie + NGHOST;
  int jlcc = js;
  int jucc = je;
  if (block_size.nx2 > 1)
  {
    jlcc -= NGHOST;
    jucc += NGHOST;
  }

  int klcc = ks;
  int kucc = ke;
  if (block_size.nx3 > 1)
  {
    klcc -= NGHOST;
    kucc += NGHOST;
  }

  // Initialize magnetic field
  // No metric weighting here
  Real rhomax = tov->data[itov_rho][0];
  Real pgasmax = k_adi*pow(rhomax,gamma_adi);
  Real pcut = pin->GetReal("problem","pcut") * pgasmax;
  Real amp = pin->GetOrAddReal("problem","b_amp", 0.0);
  int magindex=pin->GetInteger("problem","magindex");

  int nx1 = (ie-is)+1 + 2*(NGHOST);
  int nx2 = (je-js)+1 + 2*(NGHOST);
  int nx3 = (ke-ks)+1 + 2*(NGHOST);

  pfield->b.x1f.ZeroClear();
  pfield->b.x2f.ZeroClear();
  pfield->b.x3f.ZeroClear();
  pfield->bcc.ZeroClear();

  AthenaArray<Real> ax,ay,az,bxcc,bycc,bzcc;
  // should be athena tensors if we merge w/ dynamical metric
  ax.NewAthenaArray(nx3,nx2,nx1);
  ay.NewAthenaArray(nx3,nx2,nx1);
  // Don't use az for poloidal field
  //az.NewAthenaArray(nx3,nx2,nx1);
  bxcc.NewAthenaArray(nx3,nx2,nx1);
  bycc.NewAthenaArray(nx3,nx2,nx1);
  bzcc.NewAthenaArray(nx3,nx2,nx1);

  // Initialize construct cell centred potential
  for (int k=klcc; k<=kucc; k++) {
    for (int j=jlcc; j<=jucc; j++) {
      for (int i=ilcc; i<=iucc; i++) {

	ax(k,j,i) = -pcoord->x2v(j)*amp*std::max(phydro->w(IPR,k,j,i) - pcut,0.0)*pow((1.0 - phydro->w(IDN,k,j,i)/rhomax),magindex);
	ay(k,j,i) = pcoord->x1v(i)*amp*std::max(phydro->w(IPR,k,j,i) - pcut,0.0)*pow((1.0 - phydro->w(IDN,k,j,i)/rhomax),magindex);
	
      }
    }
  }
  // Construct cell centred B field from cell centred potential
  //TODO should use pfield->bcc storage here.
  for(int k = klcc+1; k<=kucc-1; k++){
    for(int j = jlcc+1; j<=jucc-1; j++){
      for(int i = ilcc+1; i<=iucc-1; i++){
	
	bxcc(k,j,i) = - ((ay(k+1,j,i) - ay(k-1,j,i))/(2.0*pcoord->dx3v(k)));
	bycc(k,j,i) =  ((ax(k+1,j,i) - ax(k-1,j,i))/(2.0*pcoord->dx3v(k)));
	bzcc(k,j,i) = ( (ay(k,j,i+1) - ay(k,j,i-1))/(2.0*pcoord->dx1v(i))
			- (ax(k,j+1,i) - ax(k,j-1,i))/(2.0*pcoord->dx2v(j)));
      }
    }
  } 
  // Initialise face centred field by averaging cc field
  for(int k = klcc+1; k<=kucc-1; k++){
    for(int j = jlcc+1; j<=jucc-1; j++){
      for(int i = ilcc+2; i<=iucc-1; i++){
	pfield->b.x1f(k,j,i) = 0.5*(bxcc(k,j,i-1) + bxcc(k,j,i));
      }
    }
  }
  for(int k = klcc+1; k<=kucc-1; k++){
    for(int j = jlcc+2; j<=jucc-1; j++){
      for(int i = ilcc+1; i<=iucc-1; i++){
	pfield->b.x2f(k,j,i) = 0.5*(bycc(k,j-1,i) + bycc(k,j,i));
      }
    }
  }
  for(int k = klcc+2; k<=kucc-1; k++){
    for(int j = jlcc+1; j<=jucc-1; j++){
      for(int i = ilcc+1; i<=iucc-1; i++){
	pfield->b.x3f(k,j,i) = 0.5*(bzcc(k-1,j,i) + bzcc(k,j,i));
      }
    }
  }
  
  pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, ilcc,iucc,jlcc,jucc,klcc,kucc);

#endif

}

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

Real Maxrho(MeshBlock *pmb, int iout)
{
  Real max_rho = 0.0;
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

#if MAGNETIC_FIELDS_ENABLED
//TODO make consistent with CT divB
Real DivB(MeshBlock *pmb, int iout)
{
  Real divB = 0.0;
  Real vol,dx,dy,dz;
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;

  AthenaArray<Real> bcc1, bcc2,bcc3;
  bcc1.InitWithShallowSlice(pmb->pfield->bcc,IB1,1);
  bcc2.InitWithShallowSlice(pmb->pfield->bcc,IB2,1);
  bcc3.InitWithShallowSlice(pmb->pfield->bcc,IB3,1);
  for (int k=ks; k<=ke; k++)
  for (int j=js; j<=je; j++)
  for (int i=is; i<=ie; i++)
  {
    dx = pmb->pcoord->dx1v(i);
    dy = pmb->pcoord->dx2v(j);
    dz = pmb->pcoord->dx3v(k);
    vol = dx*dy*dz;
    divB += ((bcc1(k,j,i+1) - bcc1(k,j,i-1))/(2.*dx) + (bcc2(k,j+1,i) - bcc2(k,j-1,i))/(2.* dy) + (bcc3(k+1,j,i) - bcc3(k-1,j,i))/(2. *dz))*vol;
  }

  return divB;
}
#endif

} // namespace


namespace TOV_geom {

template<typename T>
inline bool Lorentz4Boost(
  AthenaArray<T> & lam,
  AthenaArray<T> & xi,
  const bool is_passive)
{
  const int D = 4;
  const int sgn = (is_passive) ? -1.0 : 1.0;

  T xi_2 {0.};
  for (int ix=0; ix<D-1; ++ix)
  {
    xi_2 += SQR(xi(ix));
  }

  if (xi_2 > 0.)
  {
    lam.Fill(0);

    const T gam = 1./std::sqrt(1 - xi_2);
    const T ooxi_2 = 1. / xi_2;

    lam(0,0) = gam;

    for(size_t ix=1; ix<D; ++ix)
    {
      lam(ix,ix) += 1;
      lam(ix,0)  += sgn * gam * xi(ix-1);
      for(size_t jx=ix; jx<D; ++jx)
      {
        lam(jx,ix) += ooxi_2*(gam-1)*xi(ix-1)*xi(jx-1);
      }
    }

    // use symmetry to populate rest
    for(size_t ix=0; ix<D; ++ix)
    for(size_t jx=ix+1; jx<D; ++jx)
    {
      lam(ix,jx) = lam(jx,ix);
    }

    return true;
  }
  else
  {
    // fill identity
    lam.Fill(0.);
    for (size_t ix=0; ix<D; ++ix)
    {
      lam(ix,ix) = 1.;
    }
    return false;
  }
}

}