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
#include <iostream>   // endl, ostream
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // string

// Athena++ headers
#include "../athena.hpp"                   // macros, enums, FaceField
#include "../athena_arrays.hpp"            // AthenaArray
#include "../bvals/bvals.hpp"              // BoundaryValues
#include "../coordinates/coordinates.hpp"  // Coordinates
#include "../eos/eos.hpp"                  // EquationOfState
#include "../z4c/z4c.hpp"                  // EquationOfState
#include "../field/field.hpp"              // Field
#include "../hydro/hydro.hpp"              // Hydro
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"          // ParameterInput

// Configuration checking
#if not GENERAL_RELATIVITY
#error "This problem generator must be used with general relativity"
#endif

// Declarations
void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int il, int iu, int jl, int ju, int kl, int ku, int ngh);

namespace {
  int TOV_rhs(Real dr, Real *u, Real *k);
  int TOV_solve(Real rhoc, Real rmin, Real dr, int *npts);
  int interp_locate(Real *x, int Nx, Real xval);
  void interp_lag4(Real *f, Real *x, int Nx, Real xv,
		   Real *fv_p, Real *dfv_p, Real *ddfv_p );
  void TOV_background(Real x1, Real x2, Real x3, ParameterInput *pin, 
		      AthenaArray<Real> &g, AthenaArray<Real> &g_inv, 
		      AthenaArray<Real> &dg_dx1, AthenaArray<Real> &dg_dx2,
		      AthenaArray<Real> &dg_dx3);//TOV_ID
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
  Real rmin = pin->GetReal("problem", "rmin");
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
  for (int v = 0; v < itov_nv; v++)
    tov->data[v] = (Real*) malloc(npts*sizeof(Real));

  // Solve TOV equations, setting 1D inital data in tov->data
  TOV_solve(rhoc, rmin, dr, &npts);

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

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
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
// So conversion is rho0=rho and u = rho*epsl

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // container with idx / grids pertaining z4c
  Z4c::MB_info* mbi = &(pz4c->mbi);

  // Parameters
  phydro->w.Fill(NAN);
  phydro->w1.Fill(NAN);
  pz4c->storage.u.Fill(NAN);
  pz4c->storage.adm.Fill(NAN);

  // Prepare scratch arrays
  // AthenaArray<Real> g, gi;
  // g.NewAthenaArray(NMETRIC, iu+1);
  // gi.NewAthenaArray(NMETRIC, iu+1);

  // Star mass & radius
  const Real M = tov->M;     // Mass of TOV star
  const Real R = tov->Riso;  // Isotropic Radius of TOV star

  // Atmosphere
  Real rhomax = tov->data[itov_rho][0];
  Real pgasmax = k_adi*pow(rhomax,gamma_adi);
  Real fatm = pin->GetReal("problem","fatm");
  const Real rho_atm = rhomax * fatm;

  //TODO (SB) general EOS call
  const Real pre_atm = k_adi*std::pow(rhomax*fatm,gamma_adi);

  // Pontwise aux vars
  Real rho_kji, pgas_kji, v_kji, x_kji;
  Real lapse_kji, d_lapse_dr_kji, psi4_kji,d_psi4_dr_kji,dummy;

  // Initialize primitive values on CC grid
  for (int k=0; k<ncells3; ++k)
  for (int j=0; j<ncells2; ++j)
  for (int i=0; i<ncells1; ++i)
  {
    // Isotropic radius
    Real r = (std::sqrt(pow(pcoord->x1v(i),2.) +
                        pow(pcoord->x2v(j),2.) +
                        pow(pcoord->x3v(k),2.)));
    Real rho_pol = std::sqrt(pow(pcoord->x1v(i),2.) +
                             pow(pcoord->x2v(j),2.));

    Real costh = pcoord->x3v(k)/r;
    Real sinth = rho_pol/r;
    Real cosphi = pcoord->x1v(i)/rho_pol;
    Real sinphi = pcoord->x2v(j)/rho_pol;

    if (r<R)
    {
      // Interpolate rho to star interior
      interp_lag4(tov->data[itov_rho], tov->data[itov_riso], tov->npts, r,
            &rho_kji, &dummy,&dummy);
      // Pressure from EOS
      //TODO (SB) general EOS call
      pgas_kji = k_adi*pow(rho_kji,gamma_adi);
      // Add perturbation
      x_kji = r / R;
      v_kji = (v_amp>0) ? (0.5*v_amp*(3.0*x_kji - x_kji*x_kji*x_kji)) : 0.0;
    }
    else
    {
      // Set exterior to atmos
      rho_kji  = rho_atm;
      pgas_kji = pre_atm;
      v_kji    = 0.0;
    }

    phydro->w_init(IDN,k,j,i) = rho_kji;
    phydro->w_init(IPR,k,j,i) = pgas_kji;
    phydro->w_init(IVX, k, j, i) = v_kji*sinth*cosphi;
    phydro->w_init(IVY, k, j, i) = v_kji*sinth*sinphi;
    phydro->w_init(IVZ, k, j, i) = v_kji*costh;

    phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = phydro->w_init(IDN,k,j,i);
    phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = phydro->w_init(IPR,k,j,i);
    phydro->w(IVX,k,j,i) = phydro->w1(IVX,k,j,i) = phydro->w_init(IVX,k,j,i);
    phydro->w(IVY,k,j,i) = phydro->w1(IVY,k,j,i) = phydro->w_init(IVY,k,j,i);
    phydro->w(IVZ,k,j,i) = phydro->w1(IVZ,k,j,i) = phydro->w_init(IVZ,k,j,i);
  }

  // Initialise metric variables on geometric grid
  // Sets alpha, beta, g_ij, K_ij
  for (int k=0; k<mbi->nn3; ++k)
  for (int j=0; j<mbi->nn2; ++j)
  for (int i=0; i<mbi->nn1; ++i)
  {
    // Isotropic radius
    Real const r = std::sqrt(
      pow(mbi->x1(i), 2.) + pow(mbi->x2(j), 2.) + pow(mbi->x3(k), 2.)
    );

    if (r<R){
      // Interior metric, lapse and conf.fact.
      if (r == 0.)
      {
        lapse_kji = tov->lapse_0;
        d_lapse_dr_kji = 0.0;
        psi4_kji = tov->psi4_0;
        d_psi4_dr_kji = 0.0;
      }
      else
      {
        interp_lag4(tov->data[itov_lapse], tov->data[itov_riso], tov->npts, r,
        &lapse_kji, &d_lapse_dr_kji,  &dummy);
        interp_lag4(tov->data[itov_psi4], tov->data[itov_riso], tov->npts, r,
        &psi4_kji, &d_psi4_dr_kji,  &dummy);
      }
    }
    else
    {
      // Exterior schw. metric, lapse and conf.fact.
      lapse_kji = ((r-M/2.)/(r+M/2.));
      psi4_kji = std::pow((1.+0.5*M/r),4.);
    }

    // Set lapse, shift, ADM metric, and extr. curvature
    pz4c->storage.u(Z4c::I_Z4c_alpha,k,j,i) = lapse_kji;
    pz4c->storage.u(Z4c::I_Z4c_betax,k,j,i) = 0.0;
    pz4c->storage.u(Z4c::I_Z4c_betay,k,j,i) = 0.0;
    pz4c->storage.u(Z4c::I_Z4c_betaz,k,j,i) = 0.0;
    pz4c->storage.adm(Z4c::I_ADM_gxx,k,j,i) = psi4_kji;
    pz4c->storage.adm(Z4c::I_ADM_gyy,k,j,i) = psi4_kji;
    pz4c->storage.adm(Z4c::I_ADM_gzz,k,j,i) = psi4_kji;
    pz4c->storage.adm(Z4c::I_ADM_gxy,k,j,i) = 0.0;
    pz4c->storage.adm(Z4c::I_ADM_gxz,k,j,i) = 0.0;
    pz4c->storage.adm(Z4c::I_ADM_gyz,k,j,i) = 0.0;
    pz4c->storage.adm(Z4c::I_ADM_Kxx,k,j,i) = 0.0;
    pz4c->storage.adm(Z4c::I_ADM_Kyy,k,j,i) = 0.0;
    pz4c->storage.adm(Z4c::I_ADM_Kzz,k,j,i) = 0.0;
    pz4c->storage.adm(Z4c::I_ADM_Kxy,k,j,i) = 0.0;
    pz4c->storage.adm(Z4c::I_ADM_Kxz,k,j,i) = 0.0;
    pz4c->storage.adm(Z4c::I_ADM_Kyz,k,j,i) = 0.0;
    pz4c->storage.adm(Z4c::I_ADM_psi4,k,j,i) = psi4_kji;

  }

  // Initialize remaining z4c variables
  pz4c->ADMToZ4c(pz4c->storage.adm,pz4c->storage.u);
  pz4c->ADMToZ4c(pz4c->storage.adm,pz4c->storage.u1);

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
  az.NewAthenaArray(nx3,nx2,nx1);
  bxcc.NewAthenaArray(nx3,nx2,nx1);
  bycc.NewAthenaArray(nx3,nx2,nx1);
  bzcc.NewAthenaArray(nx3,nx2,nx1);

  //SB TODO cleanup, should use directly pz4c->storage.adm
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
  
  // Initialize field at xfaces
  for (int k=klcc; k<=kucc; k++) {
    for (int j=jlcc; j<=jucc; j++) {
      for (int i=ilcc; i<=iucc; i++) {

	// ay at x faces - need w at xface	
	ax(k,j,i) = -pcoord->x2v(j)*amp*std::max(phydro->w(IPR,k,j,i) - pcut,0.0)*pow((1.0 - phydro->w(IDN,k,j,i)/rhomax),magindex);
	ay(k,j,i) = pcoord->x1v(i)*amp*std::max(phydro->w(IPR,k,j,i) - pcut,0.0)*pow((1.0 - phydro->w(IDN,k,j,i)/rhomax),magindex);
	
      }
    }
  }
  
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
// These loop lims are wrong TODO 
  for(int k = klcc; k<=kucc; k++){
    for(int j = jlcc; j<=jucc; j++){
      for(int i = ilcc; i<=iucc+1; i++){
	pfield->b.x1f(k,j,i) = 0.5*(bxcc(k,j,i-1) + bxcc(k,j,i));
      }
    }
  }
  for(int k = klcc; k<=kucc; k++){
    for(int j = jlcc; j<=jucc+1; j++){
      for(int i = ilcc; i<=iucc; i++){
	pfield->b.x2f(k,j,i) = 0.5*(bycc(k,j-1,i) + bycc(k,j,i));
      }
    }
  }
  for(int k = klcc; k<=kucc+1; k++){
    for(int j = jlcc; j<=jucc; j++){
      for(int i = ilcc; i<=iucc; i++){
	pfield->b.x3f(k,j,i) = 0.5*(bzcc(k-1,j,i) + bzcc(k,j,i));
      }
    }
  }
  
  pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, il,iu,jl,ju,kl,ku);

#endif
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

  return;
}


//----------------------------------------------------------------------------------------
//! \fn
// \brief Fixed boundary condition
// Inputs:
//   pmb: pointer to MeshBlock
//   pcoord: pointer to Coordinates
//   time,dt: current time and timestep of simulation
//   is,ie,js,je,ks,ke: indices demarkating active region
// Outputs:
//   prim: primitives set in ghost zones
//   bb: face-centered magnetic field set in ghost zones
// Notes:
//   does nothing
//relic - not needed
void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  return;
}


namespace {

  //-----------------------------------------------------------------------------------
  //! \fn int TOV_rhs(Real dr, Real *u, Real *k)
  // \brief Calculate right hand sides for TOV equations
  //

  int TOV_rhs(Real r, Real *u, Real *k) {

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
    Real den   = r*r*(1.-2.*m/r);
    Real dphidr = (r==0.) ? 0. : num/den;
    
    Real drhodr = -(e+p) * dphidr / dpdrho;

    Real dmdr   = 4.*PI*r*r*e; 
 
    Real f      = std::sqrt(1.-2.*m/r); 
    Real dIdr   = ( 1.-f )/( r*f ); //TODO(SB) sqrt(1 - m(0)/0 ) FIX
    
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

  int TOV_solve(Real rhoc, Real rmin, Real dr, int *npts)  {

    std::stringstream msg;
    
    // Alloc buffers for ODE solve
    const int maxsize = *npts - 1;
    Real u[TOV_NVAR],u1[TOV_NVAR],u2[TOV_NVAR],u3[TOV_NVAR],k[TOV_NVAR];
        
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
    const Real rho_zero = 0.; //TODO(SB) use atmosphere level
    const Real oosix = 1./6.;
    while (n < maxsize) {

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
      for (int v = 0; v < TOV_NVAR; v++) {
	u[v] = oosix*( 2.*( - u[v] + u1[v] + u3[v] ) + 4.*u2[v] + dr*k[v] );
      }
	
      if (stop) {
        msg << "### FATAL ERROR in function [TOV_solve]"
            << std::endl << "TOV r.h.s. not finite";
        ATHENA_ERROR(msg);
      }      
      
      // Stop if radius reached
      rhoo = u[TOV_IRHO];
      if (rhoo < rho_zero) {
	break;
      }      
      
      // Store data
      tov->data[itov_rsch][n] = r;
      tov->data[itov_rho][n] = u[TOV_IRHO];
      tov->data[itov_mass][n] = u[TOV_IMASS];
      tov->data[itov_phi][n] = u[TOV_IPHI];
      tov->data[itov_riso][n] = r * std::exp(u[TOV_IINT]); // Multiply by C later 

      // Prepare next step
      r += dr;      
      n++; 
    }
    
    if (n >= maxsize) {
      msg << "### FATAL ERROR in function [TOV_solve]"
	  << std::endl << "Star radius not reached. (Try increasing 'npts')";
      ATHENA_ERROR(msg);
    }
        
    *npts = n;
    tov->npts = n;
    tov->R = r;
    tov->M = u[TOV_IMASS];
    
    // Re-Alloc 1D data 
    for (int v = 0; v < itov_nv; v++)
      tov->data[v] = (Real*) realloc(tov->data[v], tov->npts*sizeof(Real));
    
    // Match to exterior
    const Real phiR  = u[TOV_IPHI];
    const Real IR    = u[TOV_IINT];
    const Real phiRa = 0.5*std::log(1.-2.*tov->M/tov->R);
    const Real C     = 1./(2*tov->R) * (std::sqrt(tov->R*tov->R-2*tov->M*tov->R)+tov->R-tov->M) * std::exp(-IR);

    for (int n = 0; n < tov->npts; n++) {
      tov->data[itov_phi][n] += - phiR + phiRa;
      tov->data[itov_riso][n] *= C; // riso = rsch * C * exp(IINT) 
    }

    tov->Riso = tov->data[itov_riso][n-1];
    
    // Pressure 
    //TODO(SB) general EOS call
    for (int n = 0; n < tov->npts; n++) {
      tov->data[itov_pre][n] = std::pow(tov->data[itov_rho][n],gamma_adi) * k_adi;
    }

    // Other metric fields
    for (int n = 0; n < tov->npts; n++) {
      tov->data[itov_psi4][n] = std::pow(tov->data[itov_rsch][n]/tov->data[itov_riso][n], 2);
      tov->data[itov_lapse][n] = std::exp(tov->data[itov_phi][n]);
    }

    // Metric field (regular origin)
    tov->lapse_0 = std::exp(- phiR + phiRa);
    tov->psi4_0 = 1/(C*C);
    
    // Done!
    printf("TOV_solve: npts = %d\n",tov->npts);
    printf("TOV_solve: R(sch) = %.16e\n",tov->R);
    printf("TOV_solve: R(iso) = %.16e\n",tov->Riso);
    printf("TOV_solve: M = %.16e\n",tov->M);
    printf("TOV_solve: lapse(0) = %.16e\n",tov->lapse_0);
    printf("TOV_solve: psi4(0) = %.16e\n",tov->psi4_0);
    
    return 0;
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

  
  //--------------------------------------------------------------------------------------
  //! \fn void interp_lag4(Real *f, Real *x, int Nx, Real xv,
  //                       Real *fv_p, Real *dfv_p, Real *ddfv_p)
  // \brief 4th order lagrangian interpolation with derivatives
  // Returns the interpolated values fv at xv of the fuction f(x) 
  // together with 1st and 2nd derivatives dfv, ddfv 
  void interp_lag4(Real *f, Real *x, int Nx, Real xv,
		   Real *fv_p, Real *dfv_p, Real *ddfv_p) {
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
  }


  //----------------------------------------------------------------------------------------
  //! \fn 
  // \brief Function for gr_user coordinate choice for cowling approx - not needed here
  // Function that interpolates calculated TOV initial data to specified isotropic radius 
  // for gr_user coordinate choice
  // Origin is not explicitely fixed here.
  
  void TOV_background(Real x1, Real x2, Real x3, ParameterInput *pin,
		      AthenaArray<Real> &g, AthenaArray<Real> &g_inv,
		      AthenaArray<Real> &dg_dx1, AthenaArray<Real> &dg_dx2,
		      AthenaArray<Real> &dg_dx3) {    
    Real r = std::sqrt(x1*x1+x2*x2+x3*x3); // Athena Isotropic radius
    const Real M = tov->M;     // Mass of TOV star 
    const Real R = tov->Riso;  // Isotropic Radius of TOV star

    Real rsch,drsch;
    Real phi,dphi;
    Real dummy;
    
    if (r<R) {

      // In star interior use numerically found values of metric components
      // Interpolate initial data from function of schwarzschild radius to isotropic radius of given coordinate

      interp_lag4(tov->data[itov_rsch], tov->data[itov_riso], tov->npts, r,
                  &rsch, &drsch,  &dummy);
      interp_lag4(tov->data[itov_phi], tov->data[itov_riso], tov->npts, r,
                  &phi,& dphi,  &dummy);
      
      g(I00) = -exp(2. * phi);
      g(I11) = pow(rsch/r,2.);
      g(I22) = g(I11); 
      g(I33) = g(I11); 
      g(I01) = 0.;
      g(I02) = 0.;
      g(I03) = 0.;
      g(I12) = 0.;
      g(I13) = 0.;
      g(I23) = 0.;

      g_inv(I00) = -exp(-2.*phi);
      g_inv(I11) = pow(r/rsch,2.);
      g_inv(I22) = g_inv(I11);
      g_inv(I33) = g_inv(I11);
      g_inv(I01) = 0.;
      g_inv(I02) = 0.;
      g_inv(I03) = 0.;
      g_inv(I12) = 0.;
      g_inv(I13) = 0.;
      g_inv(I23) = 0.;

      dg_dx1(I00) = -2. * exp(2.*phi) * dphi * x1/r;
      dg_dx1(I11) = 2. * (rsch/r) * (x1/r) * (drsch/r - rsch/(r*r));
      dg_dx1(I22) = dg_dx1(I11);
      dg_dx1(I33) = dg_dx1(I11);
      dg_dx1(I01) = 0.;
      dg_dx1(I02) = 0.;
      dg_dx1(I03) = 0.;
      dg_dx1(I12) = 0.;
      dg_dx1(I13) = 0.;
      dg_dx1(I23) = 0.;

      dg_dx2(I00) = -2. * exp(2.*phi) * dphi * x2/r;
      dg_dx2(I11) = 2. * (rsch/r) * (x2/r) * (drsch/r - rsch/(r*r));
      dg_dx2(I22) = dg_dx2(I11);
      dg_dx2(I33) = dg_dx2(I11);
      dg_dx2(I01) = 0.;
      dg_dx2(I02) = 0.;
      dg_dx2(I03) = 0.;
      dg_dx2(I12) = 0.;
      dg_dx2(I13) = 0.;
      dg_dx2(I23) = 0.;

      dg_dx3(I00) = -2. * exp(2.*phi) * dphi * x3/r;
      dg_dx3(I11) = 2. * (rsch/r) * (x3/r) * (drsch/r - rsch/(r*r));
      dg_dx3(I22) = dg_dx3(I11);
      dg_dx3(I33) = dg_dx3(I11);
      dg_dx3(I01) = 0.;
      dg_dx3(I02) = 0.;
      dg_dx3(I03) = 0.;
      dg_dx3(I12) = 0.;
      dg_dx3(I13) = 0.;
      dg_dx3(I23) = 0.;

    } else {

      // In star exterior use exterior Schwarzschild solution.	    
      
      g(I00) = -pow(((r-M/2.)/(r+M/2.)),2.);
      g(I11) = pow((1.+0.5*M/r),4.);
      g(I22) = g(I11); 
      g(I33) = g(I11); 
      g(I01) = 0.;
      g(I02) = 0.;
      g(I03) = 0.;
      g(I12) = 0.;
      g(I13) = 0.;
      g(I23) = 0.;

      g_inv(I00) = -pow(((r+M/2.)/(r-M/2.)),2.);
      g_inv(I11) =  pow((1.+0.5*M/r),-4.);
      g_inv(I22) = g_inv(I11);
      g_inv(I33) = g_inv(I11);
      g_inv(I01) = 0.;
      g_inv(I02) = 0.;
      g_inv(I03) = 0.;
      g_inv(I12) = 0.;
      g_inv(I13) = 0.;
      g_inv(I23) = 0.;

      dg_dx1(I00) = -2. * x1/r * M*(r-M/2.)/(pow(r+M/2.,3.));
      dg_dx1(I11) = -2. * (x1/r) *pow(1.+0.5*M/r,3)*M/(r*r) ;
      dg_dx1(I22) = dg_dx1(I11);
      dg_dx1(I33) = dg_dx1(I11);
      dg_dx1(I01) = 0.;
      dg_dx1(I02) = 0.;
      dg_dx1(I03) = 0.;
      dg_dx1(I12) = 0.;
      dg_dx1(I13) = 0.;
      dg_dx1(I23) = 0.;

      dg_dx2(I00) =  -2. * x2/r * M*(r-M/2.)/(pow(r+M/2.,3.));
      dg_dx2(I11) = -2. * (x2/r) *pow(1.+0.5*M/r,3)*M/(r*r) ;
      dg_dx2(I22) = dg_dx2(I11);
      dg_dx2(I33) = dg_dx2(I11);
      dg_dx2(I01) = 0.;
      dg_dx2(I02) = 0.;
      dg_dx2(I03) = 0.;
      dg_dx2(I12) = 0.;
      dg_dx2(I13) = 0.;
      dg_dx2(I23) = 0.;
      
      dg_dx3(I00) =  -2. * x3/r * M*(r-M/2.)/(pow(r+M/2.,3.));
      dg_dx3(I11) = -2. * (x3/r) *pow(1.+0.5*M/r,3)*M/(r*r) ;
      dg_dx3(I22) = dg_dx3(I11);
      dg_dx3(I33) = dg_dx3(I11);
      dg_dx3(I01) = 0.;
      dg_dx3(I02) = 0.;
      dg_dx3(I03) = 0.;
      dg_dx3(I12) = 0.;
      dg_dx3(I13) = 0.;
      dg_dx3(I23) = 0.;
      
    }
    
  }

  
  //----------------------------------------------------------------------------------------
  ////! \fn
  ////  \brief refinement condition: refine at large gradients of velocity
  //relic from cowling approx
  int RefinementCondition(MeshBlock *pmb) {
    AthenaArray<Real> &w = pmb->phydro->w;
    Real maxeps=0.0;
    for (int k=pmb->ks; k<=pmb->ke; k++) {
      for (int j=pmb->js; j<=pmb->je; j++) {
	for (int i=pmb->is; i<=pmb->ie; i++) {
	  
	  Real eps = fabs((w(IVX,k,j,i+1) - w(IVX,k,j,i-1))/pmb->pcoord->dx1v(i));
	  maxeps = std::max(maxeps,eps);
	  eps = fabs((w(IVY,k,j,i+1) - w(IVY,k,j,i-1))/pmb->pcoord->dx1v(i));
	  maxeps = std::max(maxeps,eps);
	  eps  = fabs((w(IVZ,k,j,i+1) - w(IVZ,k,j,i-1))/pmb->pcoord->dx1v(i));
	  maxeps = std::max(maxeps,eps);
	  eps  = fabs((w(IVX,k,j+1,i) - w(IVX,k,j-1,i))/pmb->pcoord->dx2v(i));
	  maxeps = std::max(maxeps,eps);
	  eps  = fabs((w(IVY,k,j+1,i) - w(IVY,k,j-1,i))/pmb->pcoord->dx2v(i));
	  maxeps = std::max(maxeps,eps);
	  eps  = fabs((w(IVZ,k,j+1,i) - w(IVZ,k,j-1,i))/pmb->pcoord->dx2v(i));
	  maxeps = std::max(maxeps,eps);
	  eps  = fabs((w(IVX,k+1,j,i) - w(IVX,k-1,j,i))/pmb->pcoord->dx3v(i));
	  maxeps = std::max(maxeps,eps);
	  eps  = fabs((w(IVY,k+1,j,i) - w(IVY,k-1,j,i))/pmb->pcoord->dx3v(i));
	  maxeps = std::max(maxeps,eps);
	  eps  = fabs((w(IVZ,k+1,j,i) - w(IVZ,k-1,j,i))/pmb->pcoord->dx3v(i));
	  maxeps = std::max(maxeps,eps);
	}
      }
    }
    // refine : curvature > 0.01
    if (maxeps > 0.02) return 1;
    // derefinement: curvature < 0.005
    if (maxeps < 0.005) return -1;
    // otherwise, stay
    return 0;
  }

  Real Maxrho(MeshBlock *pmb, int iout) {
    Real max_rho = 0.0;
    int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
    AthenaArray<Real> &w = pmb->phydro->w;
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
	for (int i=is; i<=ie; i++) {
	  max_rho = std::max(std::abs(w(IDN,k,j,i)), max_rho);
	}
      }
    }
    return max_rho;
  }
  
#if MAGNETIC_FIELDS_ENABLED
  //TODO make consistent with CT divB
  Real DivB(MeshBlock *pmb, int iout) {
    Real divB = 0.0;
    Real vol,dx,dy,dz;
    int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
    AthenaArray<Real> bcc1, bcc2,bcc3;
    bcc1.InitWithShallowSlice(pmb->pfield->bcc,IB1,1);
    bcc2.InitWithShallowSlice(pmb->pfield->bcc,IB2,1);
    bcc3.InitWithShallowSlice(pmb->pfield->bcc,IB3,1);
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
	for (int i=is; i<=ie; i++) {
	  dx = pmb->pcoord->dx1v(i);
	  dy = pmb->pcoord->dx2v(j);
	  dz = pmb->pcoord->dx3v(k);
	  vol = dx*dy*dz;
	  divB += ((bcc1(k,j,i+1) - bcc1(k,j,i-1))/(2.*dx) + (bcc2(k,j+1,i) - bcc2(k,j-1,i))/(2.* dy) + (bcc3(k+1,j,i) - bcc3(k-1,j,i))/(2. *dz))*vol;
	}
      }
    }
    return divB;
  }
#endif
  
} // namespace
