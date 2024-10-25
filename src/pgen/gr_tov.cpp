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
#include <ostream>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // string
#include <cfloat>

// Athena++ headers
#include "../athena_aliases.hpp"
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
#include "../utils/spherical_harmonics.hpp"
#include "../scalars/scalars.hpp"
#if M1_ENABLED
#include "../m1/m1.hpp"
#endif  // M1_ENABLED

//----------------------------------------------------------------------------------------
using namespace gra::aliases;
//----------------------------------------------------------------------------------------

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
  Real linear_interp(Real *f, Real *x, int n, Real xv);
  /*
  void TOV_background(Real x1, Real x2, Real x3, ParameterInput *pin,
		      AthenaArray<Real> &g, AthenaArray<Real> &g_inv,
		      AthenaArray<Real> &dg_dx1, AthenaArray<Real> &dg_dx2,
		      AthenaArray<Real> &dg_dx3);//TOV_ID
  */

  // Insert interpolated TOV soln. to extant variables
  void TOV_populate(MeshBlock *pmb, ParameterInput *pin);
  void SeedMagneticFields(MeshBlock *pmb, ParameterInput *pin);

  int RefinementCondition(MeshBlock *pmb);

  // Global variables
  Real v_amp; // velocity amplitude for linear perturbations
  Real lambda; // amplitude for pressure perturbations
  Real n_nodes; // number of nodes for the sinusoidal preassure perturbations

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

#if USETM
  Primitive::ColdEOS<Primitive::COLDEOS_POLICY> * ceos = NULL;
#else
  Real gamma_adi;
  Real k_adi;
#endif
  Real rho_zero; // TOV surface density


  Real Maxrho(MeshBlock *pmb, int iout);
#if MAGNETIC_FIELDS_ENABLED
  Real DivBface(MeshBlock *pmb, int iout);
#endif

  Real num_c2p_fail(MeshBlock *pmb, int iout);

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

  // Alloc 1D buffer
  tov = new TOVData;
  tov->npts = npts;

  // spacing & number of points to retain for interpolation
  tov->interp_npts = pin->GetOrAddInteger("problem", "interp_npts", npts);
  tov->interp_dr   = pin->GetOrAddReal(   "problem", "interp_dr",   dr);

  // surface identification
  tov->surf_dr   = pin->GetOrAddReal(     "problem", "surf_dr",   dr / 1.0e3);

  // velocity perturbation
  v_amp = pin->GetOrAddReal("problem", "v_amp", 0.0);
  
  // pressure perturbation
  lambda = pin->GetOrAddReal("problem", "lambda", 0.0);
  n_nodes = pin->GetOrAddInteger("problem", "n_nodes", 1);
  
#if USETM
  // Initialize cold EOS
  ceos = new Primitive::ColdEOS<Primitive::COLDEOS_POLICY>;
  InitColdEOS(ceos, pin);
  rho_zero = ceos->GetDensityFloor();
#else
   gamma_adi = pin->GetOrAddReal("problem", "gamma", 2.0);
   k_adi = pin->GetOrAddReal("problem", "k_adi", 100.0);
   rho_zero = pin->GetOrAddReal("hydro", "dfloor", std::sqrt(1024*(FLT_MIN)));
#endif

  for (int v = 0; v < itov_nv; v++)
    tov->data[v] = (Real*) malloc((tov->interp_npts)*sizeof(Real));

  // Solve TOV equations, setting 1D inital data in tov->data
  TOV_solve(rhoc, rmin, dr, &npts);

  if(adaptive==true)
    EnrollUserRefinementCondition(RefinementCondition);

  AllocateUserHistoryOutput(2+MAGNETIC_FIELDS_ENABLED);
  EnrollUserHistoryOutput(0, Maxrho, "max-rho", UserHistoryOperation::max);
#if MAGNETIC_FIELDS_ENABLED
  EnrollUserHistoryOutput(1, DivBface, "divB", UserHistoryOperation::max);
#endif
  EnrollUserHistoryOutput(1 + MAGNETIC_FIELDS_ENABLED, num_c2p_fail,
                          "num_c2p_fail", UserHistoryOperation::sum);
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

void MeshBlock::UserWorkAfterOutput(ParameterInput *pin) {
  // Reset the status
  phydro->c2p_status.Fill(0);
  return;
}

void Mesh::DeleteTemporaryUserMeshData()
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

#if USETM
  // Free cold EOS data
  delete ceos;
#endif

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
  phydro->w.Fill(        0);
  phydro->w1.Fill(       0);
#if NSCALARS > 0
  pscalars->r.Fill(      0);
#endif
  pz4c->storage.u.Fill(  0);
  pz4c->storage.u1.Fill( 0);
  pz4c->storage.adm.Fill(0);
  pz4c->storage.mat.Fill(0);

  // Populate hydro/gauge/ADM fields based on TOV soln.
  TOV_populate(this, pin);

#if MAGNETIC_FIELDS_ENABLED
  // Regularize prims (needed for some boosted data)
  for (int k=0; k<ncells3; k++)
  for (int j=0; j<ncells2; j++)
  for (int i=0; i<ncells1; i++)
  {
    for (int n=0; n<NHYDRO; ++n)
    if (!std::isfinite(phydro->w(n,k,j,i)))
    {
#if USETM
      peos->ApplyPrimitiveFloors(phydro->w, pscalars->r, k, j, i);
#else
      peos->ApplyPrimitiveFloors(phydro->w, k, j, i);
#endif
      continue;
    }
  }

  SeedMagneticFields(this, pin);
#endif

  // Initialize remaining z4c variables
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);
  // pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u1);

  // Have geom & primitive hydro
#if M1_ENABLED
  pm1->UpdateGeometry(pm1->geom, pm1->scratch);
  pm1->UpdateHydro(pm1->hydro, pm1->geom, pm1->scratch);
  pm1->CalcFiducialVelocity();
#endif  // M1_ENABLED

  // // Impose algebraic constraints
  // pz4c->AlgConstr(pz4c->storage.u);
  // pz4c->Z4cToADM(pz4c->storage.u, pz4c->storage.adm);


  // consistent pressure atmosphere -------------------------------------------
  bool id_floor_primitives = pin->GetOrAddBoolean(
    "problem", "id_floor_primitives", false);

  if (id_floor_primitives)
  {
    for (int k = 0; k < ncells3; ++k)
    for (int j = 0; j < ncells2; ++j)
    for (int i = 0; i < ncells1; ++i)
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
  peos->PrimitiveToConserved(phydro->w,
                             pscalars->r,
                             pfield->bcc,
                             phydro->u,
                             pscalars->s,
                             pcoord,
                             0, ncells1-1,
                             0, ncells2-1,
                             0, ncells3-1);

  // Initialise matter (also taken care of in task-list)
  pz4c->GetMatter(pz4c->storage.mat,
                  pz4c->storage.adm,
                  phydro->w,
                  pscalars->r,
                  pfield->bcc);

  /*
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
  */
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

  //  Set pressure and energy using equation of state
#if USETM
  if (rho < 0.0) rho = ceos->GetDensityFloor();
  Real p = ceos->GetPressure(rho);
  Real e= ceos->GetEnergy(rho);
  Real dpdrho = ceos->GetdPdrho(rho);
#else
  Real p = k_adi * pow(rho, gamma_adi);
  Real e = rho + p / (gamma_adi - 1.0);
  Real dpdrho = gamma_adi * p / rho;
#endif

  Real num   = m + 4.*PI*r*r*r*p;
  Real den   = r*(r-2.*m);
  Real dphidr = (r==0.) ? 0. : num/den;

  Real drhodr = -(e+p) * dphidr / dpdrho;

  Real dmdr   = 4.*PI*r*r*e;
  // Real dmdr   = 4.*PI*r*r*rho;

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
#if USETM
  const Real logrhoc = log(rhoc);
  const Real pc = ceos->GetPressure(rhoc);
  const Real logpc = log(pc);
  const Real ec = ceos->GetEnergy(rhoc);
#else
  const Real pc = k_adi * pow(rhoc, gamma_adi);
  const Real ec = rhoc + pc / (gamma_adi - 1.0);
#endif

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

    // const Real c2 = 1./3., a21 = 1./3.;
    // const Real c3 = 2./3., a31 =-1./3., a32 = 1.0;
    // const Real c4 = 1.0,   a41 = 1.0,   a42 =-1.0, a43 = 1.0;

    // const Real b1 = 1./8., b2 = 3./8., b3 = 3./8., b4 = 1./8.;


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
              << std::endl << "interp_n - increase. r = " << r;
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
  for (int n = 0; n < tov->interp_npts; n++) {
#if USETM
    tov->data[itov_pre][n] = ceos->GetPressure(tov->data[itov_rho][n]);
#else
    tov->data[itov_pre][n] = k_adi * pow(tov->data[itov_rho][n], gamma_adi);
#endif
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

//--------------------------------------------------------------------------------------
//! \fn Real linear_interp(Real *f, Real *x, int n, Real xv)
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

void TOV_populate(MeshBlock *pmb, ParameterInput *pin)
{
  using namespace LinearAlgebra;

  Hydro       * phydro { pmb->phydro };
  PassiveScalars * pscalars { pmb->pscalars };
  Z4c         * pz4c   { pmb->pz4c   };
  GRDynamical * pcoord { static_cast<GRDynamical*>(pmb->pcoord) };

  // container with idx / grids pertaining z4c
  MB_info* mbi = &(pz4c->mbi);

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

  AT_D_VS2 st_dg_ddd_(   Nx1);  // metric deriv.
  AT_D_VS2 st_Gamma_ddd_(Nx1);  // Christoffel
  AT_D_VS2 st_Gamma_udd_(Nx1);

  // geometric: transformed to (p)rimed
  // AT_N_sca alphap_(    Nx1);
  AT_D_sym st_gp_dd_(  Nx1);
  AT_D_sym st_gp_uu_(  Nx1);
  AT_N_sym sp_gp_dd_(  Nx1);
  AT_N_sym sp_gp_uu_(  Nx1);
  AT_N_sym sp_Kp_dd_(  Nx1);
  AT_N_vec sp_betap_u_(Nx1);
  AT_N_vec sp_betap_d_(Nx1);

  AT_D_VS2 st_Gammap_udd_(Nx1);  // transformed Christoffel

  // matter
  AT_N_sca w_rho_(   Nx1);
  AT_N_sca w_p_(     Nx1);
  AT_N_vec w_util_u_(Nx1);
#if NSCALARS>0
  AT_N_vec prim_scalar(Nx1);
#endif
  AT_D_vec st_u_u_(  Nx1);
  AT_D_vec st_up_u_( Nx1);

  // Lorentz factor & utilde norm squared
  AT_N_sca W_(              Nx1);
  AT_N_sca sp_utilde_norm2_(Nx1);


  // slicings -----------------------------------------------------------------
  // geometric
  AT_N_sca alpha_init( pz4c->storage.adm, Z4c::I_ADM_alpha);
  AT_N_vec beta_u_init(pz4c->storage.adm, Z4c::I_ADM_betax);
  AT_N_sym g_dd_init(  pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sym K_dd_init(  pz4c->storage.adm, Z4c::I_ADM_Kxx);
  AT_N_sca psi4_init(  pz4c->storage.adm, Z4c::I_ADM_psi4);

  // matter
  AT_N_sca sl_w_rho_init(   phydro->w, IDN);
  AT_N_sca sl_w_p_init(     phydro->w, IPR);
  AT_N_vec sl_w_util_u_init(phydro->w, IVX);

#if NSCALARS>0
  AT_N_vec sl_prim_scalar(pscalars->r, 0);
#endif

  // for debugging
  AT_N_sca sl_rho( pz4c->storage.mat, Z4c::I_MAT_rho);
  AT_N_vec sl_S_d( pz4c->storage.mat, Z4c::I_MAT_Sx);
  AT_N_sym sl_S_dd(pz4c->storage.mat, Z4c::I_MAT_Sxx);

#if USETM
  Real T_initial = pin->GetReal("hydro","tfloor");
  Real Y_atm[MAX_SPECIES] = {0.0};
#if EOS_POLICY_CODE == 2
  Y_atm[0] = pin->GetReal("hydro", "y0_atmosphere");
#endif
#endif

  // Star mass & radius
  const Real M = tov->M;     // Mass of TOV star
  const Real R = tov->Riso;  // Isotropic Radius of TOV star

  // Interpolation dummy
  Real dummy;

  // Debug dump ---------------------------------------------------------------
#if 0
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
#endif
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
        const Real phi = std::asin(sinphi);
        const Real theta = std::asin(sinth);

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
#if USETM
          w_p_(i) = ceos->GetPressure(w_rho_(i));
#if NSCALARS > 0
          for (int l=0; l<NSCALARS; ++l)
            prim_scalar(l,i) = ceos->GetY(w_rho_(i), l);
#endif
#else
          w_p_(i) = k_adi*pow(w_rho_(i),gamma_adi);
#endif


          // Add velocity perturbation
          const Real x_kji = r_(i) / R;
          up_r = (v_amp>0) ? (0.5*v_amp*(3.0*x_kji - x_kji*x_kji*x_kji)) : 0.0;
         
          // Add pressure perturbation
          if (lambda > 0){
            Real Yr, Yi;
            Real eps = ceos->GetSpecificInternalEnergy(w_rho_(i));
            SphHarm_Ylm(2, 0, theta, phi, &Yr, &Yi);
            Real Ylm = Yr;
            Real H0l = lambda * std::sin((n_nodes+1.)*PI*x_kji/2.);
            Real dp  = (w_p_(i) + w_rho_(i) * (1 + eps)) * H0l * Ylm;
            w_p_(i)  += dp;

#if USETM
            w_rho_(i) = ceos->GetDensityFromPressure(w_p_(i));
#if NSCALARS > 0
            for (int l=0; l<NSCALARS; ++l)
              prim_scalar(l,i) = ceos->GetY(w_rho_(i), l);
#endif
#else
            w_rho_(i) = pow(w_p_(i)/k_adi,1./gamma_adi);
#endif
          }
        }
        else
        {
          // Let the EOS decide how to set the atmosphere on the exterior
          w_rho_(i) = 0.0;
          w_p_(i)   = 0.0;
          up_r      = 0.0;
#if NSCALARS > 0
          for (int l=0; l<NSCALARS; ++l) {
            prim_scalar(l,i) = Y_atm[l];
          }
#endif
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
        InnerProductSlicedVec3Metric(sp_utilde_norm2_, w_util_u_, sp_g_dd_,
                                     il, iu);

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
#if NSCALARS > 0
      for (int l=0; l<NSCALARS; ++l) {
      for (int i=il; i<=iu; ++i) {
        sl_prim_scalar(l,k,j,i) += prim_scalar(l,i);
      }
      }
#endif
    }


  }

  // Populate hydro registers with the initial data
  phydro->w1 = phydro->w;

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
        std::cout << "TOV geometry not finite!" << std::endl;
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
}

void SeedMagneticFields(MeshBlock *pmb, ParameterInput *pin)
{
  GRDynamical * pcoord { static_cast<GRDynamical*>(pmb->pcoord) };
  Field * pfield { pmb->pfield };
  Hydro * phydro { pmb->phydro };

  // Prepare CC index bounds
  const int il = 0;
  const int iu = (pmb->ncells1>1)? pmb->ncells1-1: 0;

  const int jl = 0;
  const int ju = (pmb->ncells2>1)? pmb->ncells2-1: 0;

  const int kl = 0;
  const int ku = (pmb->ncells3>1)? pmb->ncells3-1: 0;

  // Initialize magnetic field
  // No metric weighting here
  Real rhomax = tov->data[itov_rho][0];
#if USETM
  Real pgasmax = ceos->GetPressure(rhomax);
#else
  Real pgasmax = k_adi * pow(rhomax, gamma_adi);
#endif

  Real pcut = pin->GetReal("problem","pcut") * pgasmax;
  int magindex=pin->GetInteger("problem","magindex");

  // Real ns = pin->GetReal("problem","ns");
  // Real b_amp = pin->GetReal("problem","b_amp") *
  //               0.5/std::pow(pgasmax-pcut, ns)/8.351416e19;

  Real b_amp = pin->GetReal("problem","b_amp");

  pfield->b.x1f.ZeroClear();
  pfield->b.x2f.ZeroClear();
  pfield->b.x3f.ZeroClear();
  pfield->bcc.ZeroClear();

  AthenaArray<Real> Acc(NFIELD,pmb->ncells3,pmb->ncells2,pmb->ncells1);

  // Initialize cell centred potential
  for (int k=0; k<pmb->ncells3; k++)
  for (int j=0; j<pmb->ncells2; j++)
  for (int i=0; i<pmb->ncells1; i++)
  {
    const Real x1 = pcoord->x1v(i);
    const Real x2 = pcoord->x2v(j);

    const Real w_p   = phydro->w(IPR,k,j,i);
    const Real w_rho = phydro->w(IDN,k,j,i);

    Acc(0,k,j,i) = -x2 * b_amp * std::max(w_p-pcut, 0.0) *
                    std::pow((1.0 - w_rho/rhomax), magindex);
    Acc(1,k,j,i) =  x1 * b_amp * std::max(w_p-pcut, 0.0) *
                    std::pow((1.0 - w_rho/rhomax), magindex);
    Acc(2,k,j,i) =  0.0;
  }

  // Construct cell centred B field from cell centred potential
  for(int k=pmb->ks-1; k<=pmb->ke+1; k++)
  for(int j=pmb->js-1; j<=pmb->je+1; j++)
  for(int i=pmb->is-1; i<=pmb->ie+1; i++)
  {
    const Real dx1 = pcoord->dx1v(i);
    const Real dx2 = pcoord->dx2v(j);
    const Real dx3 = pcoord->dx3v(k);

    pfield->bcc(0,k,j,i) = -((Acc(1,k+1,j,i) - Acc(1,k-1,j,i))/(2.0*dx3));
    pfield->bcc(1,k,j,i) =  ((Acc(0,k+1,j,i) - Acc(0,k-1,j,i))/(2.0*dx3));
    pfield->bcc(2,k,j,i) =  ((Acc(1,k,j,i+1) - Acc(1,k,j,i-1))/(2.0*dx1) -
                             (Acc(0,k,j+1,i) - Acc(0,k,j-1,i))/(2.0*dx2));

  }

  // Initialise face centred field by averaging cc field
  for(int k=pmb->ks; k<=pmb->ke;   k++)
  for(int j=pmb->js; j<=pmb->je;   j++)
  for(int i=pmb->is; i<=pmb->ie+1; i++)
  {
  	pfield->b.x1f(k,j,i) = 0.5*(pfield->bcc(0,k,j,i-1) +
                                pfield->bcc(0,k,j,i));
  }

  for(int k=pmb->ks; k<=pmb->ke;   k++)
  for(int j=pmb->js; j<=pmb->je+1; j++)
  for(int i=pmb->is; i<=pmb->ie;   i++)
  {
  	pfield->b.x2f(k,j,i) = 0.5*(pfield->bcc(1,k,j-1,i) +
                                pfield->bcc(1,k,j,i));
  }

  for(int k=pmb->ks; k<=pmb->ke+1; k++)
  for(int j=pmb->js; j<=pmb->je;   j++)
  for(int i=pmb->is; i<=pmb->ie;   i++)
  {
	  pfield->b.x3f(k,j,i) = 0.5*(pfield->bcc(2,k-1,j,i) +
                                pfield->bcc(2,k,j,i));
  }
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

Real num_c2p_fail(MeshBlock *pmb, int iout)
{
  Real sum_ = 0;
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;

  AthenaArray<Real> &cstat = pmb->phydro->c2p_status;

  for (int k=ks; k<=ke; k++)
  for (int j=js; j<=je; j++)
  for (int i=is; i<=ie; i++)
  {
    if (pmb->phydro->c2p_status(k,j,i) > 0)
      sum_++;
  }

  return sum_;
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
