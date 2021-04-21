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
void TOV_ID(Real x1, Real x2, Real x3, ParameterInput *pin, 
		        AthenaArray<Real> &g, AthenaArray<Real> &g_inv, 
	                AthenaArray<Real> &dg_dx1, AthenaArray<Real> &dg_dx2,
	                AthenaArray<Real> &dg_dx3);
int tov_r_rhs(double dr, double *u, double *k);

int tov_r(double rhoc, double R, double k_adi, double gamma_adi, int *npts,
          double **p_r, double **p_m,
          double **p_rho, double **p_pre,double **p_phi,
          double **p_riso);
int     interp_locate(double *x, int Nx, double xval);
void    interp_lag4(double *f, double *x, int Nx, double xv,
                    double *fv_p, double *dfv_p, double *ddfv_p );

int RefinementCondition(MeshBlock *pmb);
// Global variables
Real gamma_adi, k_adi;  // hydro parameters
Real rhoc,R0;           //  Initial data parameters
Real rhomax;            //  Atmosphere setting variable
//Real fthr;
int npts;
Real *p_r,*p_m,*p_h,*p_rho,*p_pre,*p_phi,*p_riso; //Output for tov_r
} // namespace

//----------------------------------------------------------------------------------------
// Function for initializing global mesh properties
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)

void Mesh::InitUserMeshData(ParameterInput *pin, int res_flag) {
  // Read problem parameters

     rhoc = pin->GetReal("problem", "rhoc"); // Central value of energy density
     R0 = pin->GetReal("problem", "R0");  // Guess for initial star radius
     npts = pin->GetInteger("problem", "npts");  // number of radial pts for TOV solver
     k_adi = pin->GetReal("hydro", "k_adi");
     gamma_adi = pin->GetReal("hydro","gamma");
// Solve TOV equations, setting iniital data p_r, p_m, p_rho, p_pre, p_phi, p_riso, 
// all functions of schwarzschild radius
// Call to BAM code below
  tov_r(rhoc, R0, k_adi, gamma_adi, &npts,
            &p_r, &p_m, &p_rho,
            &p_pre, &p_phi,
            &p_riso);
	  
}



void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
// Allocate 3 user output variables: lapse, gxx, m
// leftover from cowling approx runs
	AllocateUserOutputVariables(12);
	return;
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {

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
  AthenaArray<Real> g, gi;
  g.NewAthenaArray(NMETRIC, iu+1);
  gi.NewAthenaArray(NMETRIC, iu+1);
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      pcoord->CellMetric(k, j, il, iu, g, gi);
      for (int i=il; i<=iu; ++i) {
	Real d1u1 = (phydro->w(IVX,k,j,i+1) - phydro->w(IVX,k,j,i-1))/pcoord->dx1v(i);
	Real d1u2 = (phydro->w(IVY,k,j,i+1) - phydro->w(IVY,k,j,i-1))/pcoord->dx1v(i);
	Real d1u3 = (phydro->w(IVZ,k,j,i+1) - phydro->w(IVZ,k,j,i-1))/pcoord->dx1v(i);
	Real d2u1 = (phydro->w(IVX,k,j+1,i) - phydro->w(IVX,k,j-1,i))/pcoord->dx2v(i);
	Real d2u2 = (phydro->w(IVY,k,j+1,i) - phydro->w(IVY,k,j-1,i))/pcoord->dx2v(i);
	Real d2u3 = (phydro->w(IVZ,k,j+1,i) - phydro->w(IVZ,k,j-1,i))/pcoord->dx2v(i);
	Real d3u1 = (phydro->w(IVX,k+1,j,i) - phydro->w(IVX,k-1,j,i))/pcoord->dx3v(i);
	Real d3u2 = (phydro->w(IVY,k+1,j,i) - phydro->w(IVY,k-1,j,i))/pcoord->dx3v(i);
	Real d3u3 = (phydro->w(IVZ,k+1,j,i) - phydro->w(IVZ,k-1,j,i))/pcoord->dx3v(i);

        Real r = sqrt(pow(pcoord->x1f(i),2.)+ pow(pcoord->x2f(j),2.)+  pow(pcoord->x3f(k),2.) );
	user_out_var(0,k,j,i) = sqrt(-g(I00,i)); // lapse
	user_out_var(1,k,j,i) = g(I11,i);        // gxx
	user_out_var(2,k,j,i) = 2.*r * (pow(g(I11,i),0.25)-1.); // Mass
	user_out_var(3,k,j,i) = d1u1;
	user_out_var(4,k,j,i) = d1u2;
	user_out_var(5,k,j,i) = d1u3;
	user_out_var(6,k,j,i) = d2u1;
	user_out_var(7,k,j,i) = d2u2;
	user_out_var(8,k,j,i) = d2u3;
	user_out_var(9,k,j,i) = d3u1;
	user_out_var(10,k,j,i) = d3u2;
	user_out_var(11,k,j,i) = d3u3;

	  }
	}
      }

	return;
}



//----------------------------------------------------------------------------------------
// Function for setting initial conditions
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
  // Parameters
  phydro->w.Fill(NAN);
  phydro->w1.Fill(NAN);
  pz4c->storage.u.Fill(NAN);
  pz4c->storage.adm.Fill(NAN);
  // Prepare index bounds
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

  Real rho, pgas, fatm, pgasmax;
  fatm = pin->GetReal("problem","fatm");
  // Prepare scratch arrays
  AthenaArray<Real> g, gi;
  g.NewAthenaArray(NMETRIC, iu+1);
  gi.NewAthenaArray(NMETRIC, iu+1);

     Real M = p_m[npts-1];     // Mass of TOV star 
  Real R = p_riso[npts-1];  // Isotropic Radius of TOV star
  printf("%.16e\n",R);
      Real r, dummy, phi, dphi, rsch, drsch;      
      rhomax=rhoc;
      pgasmax=k_adi*pow(rhomax,gamma_adi);
  // Initialize primitive values on CC grid
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
//      pcoord->CellMetric(k, j, il, iu, g, gi);
      for (int i=il; i<=iu; ++i) {

      r = sqrt(pow(pcoord->x1v(i),2.) +  pow(pcoord->x2v(j),2.) + pow(pcoord->x3v(k),2.)     );
      

	if (r<R){
//interpolate to star interior
       interp_lag4(p_rho, p_riso, npts, r,
                    &rho,  &dummy,&dummy);
     
   
//       if (rho>rhomax){
//		rhomax=rho;
//	}
//specific to EOS
	pgas = k_adi*pow(rho,gamma_adi); 
       	phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = rho;
        phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = pgas;
        phydro->w(IVX,k,j,i) = phydro->w1(IVX,k,j,i) = 0.0;
        phydro->w(IVY,k,j,i) = phydro->w1(IVY,k,j,i) = 0.0;
        phydro->w(IVZ,k,j,i) = phydro->w1(IVZ,k,j,i) = 0.0;
	} else {
//set exterior to atmos
        phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = rhomax*fatm;
        phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = k_adi*pow(rhomax*fatm,gamma_adi);
        phydro->w(IVX,k,j,i) = phydro->w1(IVX,k,j,i) = 0.0;
        phydro->w(IVY,k,j,i) = phydro->w1(IVY,k,j,i) = 0.0;
        phydro->w(IVZ,k,j,i) = phydro->w1(IVZ,k,j,i) = 0.0;
	}	
      }
    }
  }
// initialise metric variables on VC grid - setting alpha, beta, g_ij, K_ij
  for (int k=kl; k<=ku+1; ++k) {
    for (int j=jl; j<=ju+1; ++j) {
//      pcoord->CellMetric(k, j, il, iu, g, gi);
      for (int i=il; i<=iu+1; ++i) {

      r = sqrt(pow(pcoord->x1f(i),2.) +  pow(pcoord->x2f(j),2.) + pow(pcoord->x3f(k),2.)     );
      

	if (r<R){
      if (r < 10e-10){r += 1e-8;}// coordinate singularity at r=0
      interp_lag4(p_phi, p_riso, npts, r,
                  &phi,&dphi,  &dummy);
      interp_lag4(p_r, p_riso, npts, r,
                  &rsch,&drsch,  &dummy);
     
   
        pz4c->storage.u(Z4c::I_Z4c_alpha,k,j,i) = exp(phi); //Ths is alpha since beta is 0 take sqrt
        pz4c->storage.u(Z4c::I_Z4c_betax,k,j,i) = 0.0; 
        pz4c->storage.u(Z4c::I_Z4c_betay,k,j,i) = 0.0; 
        pz4c->storage.u(Z4c::I_Z4c_betaz,k,j,i) = 0.0; 
        pz4c->storage.adm(Z4c::I_ADM_gxx,k,j,i) = pow(rsch/r,2.);  // 
        pz4c->storage.adm(Z4c::I_ADM_gyy,k,j,i) = pow(rsch/r,2.); 
        pz4c->storage.adm(Z4c::I_ADM_gzz,k,j,i) = pow(rsch/r,2.); 
        pz4c->storage.adm(Z4c::I_ADM_gxy,k,j,i) = 0.0;
        pz4c->storage.adm(Z4c::I_ADM_gxz,k,j,i) = 0.0;
        pz4c->storage.adm(Z4c::I_ADM_gyz,k,j,i) = 0.0;
        pz4c->storage.adm(Z4c::I_ADM_Kxx,k,j,i) = 0.0; 
        pz4c->storage.adm(Z4c::I_ADM_Kyy,k,j,i) = 0.0; 
        pz4c->storage.adm(Z4c::I_ADM_Kzz,k,j,i) = 0.0; 
        pz4c->storage.adm(Z4c::I_ADM_Kxy,k,j,i) = 0.0;
        pz4c->storage.adm(Z4c::I_ADM_Kxz,k,j,i) = 0.0;
        pz4c->storage.adm(Z4c::I_ADM_Kyz,k,j,i) = 0.0;
//Set initial K = 0
	} else {
//Fix exterior schw. metric
        pz4c->storage.u(Z4c::I_Z4c_alpha,k,j,i) = ((r-M/2.)/(r+M/2.)); 
        pz4c->storage.u(Z4c::I_Z4c_betax,k,j,i) = 0.0; 
        pz4c->storage.u(Z4c::I_Z4c_betay,k,j,i) = 0.0; 
        pz4c->storage.u(Z4c::I_Z4c_betaz,k,j,i) = 0.0; 
        pz4c->storage.adm(Z4c::I_ADM_gxx,k,j,i) = pow((1.+0.5*M/r),4.); 
        pz4c->storage.adm(Z4c::I_ADM_gyy,k,j,i) = pow((1.+0.5*M/r),4.); 
        pz4c->storage.adm(Z4c::I_ADM_gzz,k,j,i) = pow((1.+0.5*M/r),4.); 
        pz4c->storage.adm(Z4c::I_ADM_gxy,k,j,i) = 0.0;
        pz4c->storage.adm(Z4c::I_ADM_gxz,k,j,i) = 0.0;
        pz4c->storage.adm(Z4c::I_ADM_gyz,k,j,i) = 0.0;
        pz4c->storage.adm(Z4c::I_ADM_Kxx,k,j,i) = 0.0; 
        pz4c->storage.adm(Z4c::I_ADM_Kyy,k,j,i) = 0.0; 
        pz4c->storage.adm(Z4c::I_ADM_Kzz,k,j,i) = 0.0; 
        pz4c->storage.adm(Z4c::I_ADM_Kxy,k,j,i) = 0.0;
        pz4c->storage.adm(Z4c::I_ADM_Kxz,k,j,i) = 0.0;
        pz4c->storage.adm(Z4c::I_ADM_Kyz,k,j,i) = 0.0;


	}	
      }
    }
  }
//Initialize remaining z4c variables
  pz4c->ADMToZ4c(pz4c->storage.adm,pz4c->storage.u);
  pz4c->ADMToZ4c(pz4c->storage.adm,pz4c->storage.u1); //?????
//Initialise coordinate class, CC metric
  pcoord->UpdateMetric();
//TODO can we update coarsec here? is coarse_u_ set yet?
  if(pmy_mesh->multilevel){
  pmr->pcoarsec->UpdateMetric();
  }
//  Initialise conserved variables
  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, il, iu, jl, ju,
                             kl, ku);
// Initialise VC matter (don't strictly need this here, will be caught in task list before used
pz4c->GetMatter(pz4c->storage.mat, pz4c->storage.adm, phydro->w);


  return;
}

//----------------------------------------------------------------------------------------
// Fixed boundary condition
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
//----------------------------------------------------------------------------------------
//Function for gr_user coordinate choice for cowling approx - not needed here
// Function that interpolates calculated TOV initial data to specified isotropic radius 
// for gr_user coordinate choice

void TOV_ID(Real x1, Real x2, Real x3, ParameterInput *pin,
            AthenaArray<Real> &g, AthenaArray<Real> &g_inv,
            AthenaArray<Real> &dg_dx1, AthenaArray<Real> &dg_dx2,
            AthenaArray<Real> &dg_dx3)
{



Real r;              // Isotropic radius of input coordinate
Real rsch,drsch;
Real phi,dphi;
Real dummy;



     r = sqrt(x1*x1+x2*x2+x3*x3);

     Real M = p_m[npts-1];     // Mass of TOV star 
     Real R = p_riso[npts-1];  // Isotropic Radius of TOV star



      if (r<R) {
// In star interior use numerically found values of metric components

// Interpolate initial data from function of schwarzschild radius to isotropic radius of given coordinate
      interp_lag4(p_r, p_riso, npts, r,
                  &rsch,&drsch,  &dummy);
      interp_lag4(p_phi, p_riso, npts, r,
                  &phi,&dphi,  &dummy);

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



//-----------------------------------------------------------------------------------
//Calculate right hand sides for TOV equations
//
/* tov r rhs */
int tov_r_rhs(double dr, double *u, double *k)
{
  double r   = u[0];
  double rho = u[1];
  double m   = u[2];
  double phi = u[3];
  double I   = u[4];
  double p,eps,dpdrho;



//  Set pressure and internal energy using equation of state
  p = k_adi * pow(rho,gamma_adi);
  eps = p / (rho*(gamma_adi-1.));
  dpdrho = gamma_adi*k_adi*pow(rho,gamma_adi-1.0);


  double e      = rho*(1.+eps);

  if (r==0) r=1e-10;
  double tmp1   = m+4.*PI*r*r*r*p;
  double tmp2   = r*r*(1.-2.*m/r);
  double tmp    = (r==0.)?0.:tmp1/tmp2;

  double drhodr = -(e+p) * tmp / dpdrho;
  double dmdr   = 4.*PI*r*r*e;
  double dphidr = tmp;
  double f      = sqrt(1.-2.*m/r);
  double dIdr   = ( 1.-f )/( r*f );

  k[0] = 0;
  k[1] = drhodr;
  k[2] = dmdr;
  k[3] = dphidr;
  k[4] = dIdr;

  return 0;
}




//------------------------------------------------------------------------------------
//Integrate TOV equations
//
/* tov r solver */
int tov_r(double rhoc, double R, double k_adi, double gamma_adi, int *npts, 
          double **p_r, double **p_m,
          double **p_rho, double **p_pre,double **p_phi, 
          double **p_riso)
{
  
  int n,v,i;
  int nvar=5;
  double **u = (double **) malloc((*npts)*sizeof(double*));
  for (n=0; n<*npts; n++)
    u[n] = (double *) malloc((nvar)*sizeof(double));
  double u1[nvar],u2[nvar],u3[nvar],k[nvar];
  double fact = 1./6.;
  
  double stp = R/(*npts);
  
  double pc,epslc;
//  Set central values of pressure internal energy using EOS
  pc = k_adi*pow(rhoc,gamma_adi);
  epslc = pc/(rhoc*(gamma_adi-1.));
  double ec = rhoc*(1.+epslc);
  
  u[0][0] = 0;
  u[0][1] = rhoc;
  u[0][2] = 0;
  u[0][3] = 0;
  u[0][4] = 0;
  
  printf("tov_r: solve TOV star (only once):\n");
  printf("    drho = %.16e npts = %d\n",stp, *npts);
  printf("    rhoc = %.16e\n",rhoc);
  printf("    ec   = %.16e\n",ec);
  printf("    pc   = %.16e\n",pc);
  
  double rhoo = u[0][1];
  int stop = 0;
  n=0;
  while (u[n][1]>0. && u[n][1]<=1.01*rhoo && stop==0) {

    stop += tov_r_rhs(stp, u[n], k);
    // u_1 = u + dt/2 k
    for (v=0; v<nvar; v++)

     u1[v] = u[n][v] + 0.5*stp*k[v]; 
    // r = rhs(u_1)
    stop += tov_r_rhs(stp, u1, k);
  
    // u_2 = u + dt/2 k
    for (v=0; v<nvar; v++)

    u2[v] = u[n][v] + 0.5*stp*k[v]; 

    // r = rhs(u_2)
    stop += tov_r_rhs(stp, u2, k);

    // u_3 = u + dt k
    for (v=0; v<nvar; v++)

    u3[v] = u[n][v] + stp*k[v];

    // r = rhs(u_3)
    stop += tov_r_rhs(stp, u3, k);
  
    // u = 1/6 ( -2 u + 2 u_1 + 4 u_2 + 2 u_3 + dt k ) 
    for (v=0; v<nvar; v++)

    u[n+1][v] = fact*( 2.*( - u[n][v] + u1[v] + u3[v] ) + 4.*u2[v] + stp*k[v] );
    
    u[n+1][0] += stp;
    
    if (n>=(*npts)-5) {
      u = (double **) realloc(u,(*npts*2)*sizeof(double*));
      for (i=*npts; i<*npts*2; i++)
        u[i] = (double *) malloc((nvar)*sizeof(double));
      *npts = (*npts)*2;
      printf("  expand grid\n");      
      //errorexit("");
    }
    rhoo = u[n][1];

    n++;
  }

  double p,dpdrho,phi,C, M,Mb, IR,phiR,phiRa;

  *npts = n;
  R     = u[*npts-1][0];
  M     = u[*npts-1][2];
  phiR  = u[*npts-1][3];
  IR    = u[*npts-1][4];
  phiRa = 0.5*log(1.-2.*M/R);
  C     = 1/(2*R) * (sqrt(R*R-2*M*R)+R-M) * exp(-IR);

  *p_r   = (double*) malloc (*npts*sizeof(double));
  *p_m   = (double*) malloc (*npts*sizeof(double));
  *p_rho = (double*) malloc (*npts*sizeof(double));
  *p_pre = (double*) malloc (*npts*sizeof(double));
  *p_phi = (double*) malloc (*npts*sizeof(double));
  *p_riso= (double*) malloc (*npts*sizeof(double));


  for (n=0; n<*npts; n++) {

    (*p_r)[n]   = u[n][0];
    (*p_rho)[n] = u[n][1];
    (*p_m)[n]   = u[n][2];
    (*p_phi)[n] = (u[n][3]-phiR + phiRa);
    (*p_pre)[n] = pow(u[n][1],gamma_adi) * k_adi;
    (*p_riso)[n]= (*p_r)[n] * C * exp(u[n][4]);

  }

  printf("    R    = %.16e   (%.16e)\n",R,(*p_riso)[*npts-1]);
  printf("    M    = %.16e\n",M);


  for (i=0; i<*npts; i++)
    free(u[i]);
  free(u);

  return 0;
}




//-----------------------------------------------------------------------------------------
//Check position of point that we are interpolating to 
int     interp_locate(double *x, int Nx, double xval)
{
  int ju,jm,jl;
  int ascnd;

  jl=-1;
  ju=Nx;

  if (xval <= x[0]) {
//    if (xval < x[0]) if (PR) printf("  pt to locate is outside (xval<xx).\n");
    return 0;
  } else if (xval >= x[Nx-1]) {
//    if (xval > x[Nx-1])if (PR)  printf("  pt to locate is outside (xval>xx).\n");
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
//4th order lagrangian interpolation

void    interp_lag4(double *f, double *x, int Nx, double xv,
                    double *fv_p, double *dfv_p, double *ddfv_p )
{
  /* Given the values in xv, it returns the interpolated values fv and 
  its 1st and 2nd derivatives dfv, ddfv of the fuction f(x) 
  Lagrangian 4 pts interpolation is used */
  
 // if (Nx < 4) errorexit(" too few points for interpolation");
  
  int i = interp_locate(x,Nx,xv);
    
  if( i < 1 ){
//    if (PR) printf(" too few points on the left => interpolation maybe be inaccurate! (v=%e)\n",xv);
    i = 1;
  } 
  if( i > (Nx-3) ){
//    if (1+PR) printf(" too few points on the right => interpolation maybe be inaccurate! (v=%e   -> %e %e)\n",xv, x[Nx-2],x[Nx-1]);
    i = Nx-3; 
  }
  
  double ximo =  x[i-1];
  double xi   =  x[i];
  double xipo =  x[i+1]; 
  double xipt =  x[i+2]; 
  
  double C1   = (f[i] - f[i-1])/(xi - ximo);
  double C2   = (-f[i] + f[i+1])/(-xi + xipo);
  double C3   = (-f[i+1] + f[i+2])/(-xipo + xipt);
  double CC1  = (-C1 + C2)/(-ximo + xipo);
  double CC2  = (-C2 + C3)/(-xi + xipt);
  double CCC1 = (-CC1 + CC2)/(-ximo + xipt);
      
  *fv_p   = f[i-1] + (-ximo + xv)*(C1 + (-xi + xv)*(CC1 + CCC1*(-xipo + xv)));
  *dfv_p  = C1 - (CC1 - CCC1*(xi + xipo - 2.*xv))*(ximo - xv)
      + (-xi + xv)*(CC1 + CCC1*(-xipo + xv));
  *ddfv_p = 2.*(CC1 - CCC1*(xi + ximo + xipo - 3.*xv));
}

//----------------------------------------------------------------------------------------
////! \fn
////  \brief refinement condition: refine at large gradients of velocity
//relic from cowling approx
int RefinementCondition(MeshBlock *pmb) {
  AthenaArray<Real> &w = pmb->phydro->w;
//  AthenaArray<Real> &g_inv = pmb->pcoord->g_inv;
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
                                                                                        

} // namespace
