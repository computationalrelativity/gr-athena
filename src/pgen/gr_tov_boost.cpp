//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gr_ntovboost.cpp
//  \brief Problem generator superposing boosted TOV stars 

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
#include "../mesh/mesh_refinement.hpp"
#include "../trackers/extrema_tracker.hpp"

// Configuration checking
#if not GENERAL_RELATIVITY
#error "This problem generator must be used with general relativity"
#endif

// Declarations
void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int il, int iu, int jl, int ju, int kl, int ku, int ngh);

Real MaxRho(MeshBlock* pmb, int iout);

namespace {

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

  // TOV solver funs
  int TOV_rhs(Real dr, Real *u, Real *k);
  int TOV_solve(TOVData * tov, Real rhoc, Real rmin, Real dr, int * npts);
  int interp_locate(Real *x, int Nx, Real xval);
  void interp_lag4(Real *f, Real *x, int Nx, Real xv,
		   Real *fv_p, Real *dfv_p, Real *ddfv_p );
  
  // 4D Boost funs
  Real invg4(Real gd[4][4], Real gu[4][4]);
  Real invg3(Real gd[3][3], Real gu[3][3]);
  void Gamma44(Real g[4][4], Real dg[4][4][4], Real Gamma[4][4][4]);
  void Gamma34(Real g[4][4], Real dg[4][4][4], Real Gamma[3][3][3]);
  void Gamma33(Real g[3][3], Real dg[3][3][3], Real Gamma[3][3][3]);
  void set_Lambda(Real LAMBDA[4][4], Real LAMBDAi[4][4], Real xix,Real xiy,Real xiz);

  void BoostedTOV(MeshBlock* pmb, TOVData * tov, Real pos[3], Real mom[3]);
  
  // Global variables
  Real gamma_adi, k_adi;  // hydro EOS parameters
  Real v_amp; // velocity amplitude for linear perturbations
  
  // Handle multiple TOVs
#define max_star_num (2)
  int ntov;
//  TOVData * tov[max_star_num] = {NULL};
  TOVData * tov[max_star_num];
  Real pos[max_star_num][3]; // position
  Real mom[max_star_num][3]; // momentum
  
} // namespace


//----------------------------------------------------------------------------------------
//!\fn void Mesh::InitUserMeshData(ParameterInput *pin, int res_flag)
// \brief  Function for initializing global mesh properties
// Inputs:
//   pin: input parameters 
// Outputs: (none)

void Mesh::InitUserMeshData(ParameterInput *pin) {
      std::stringstream msg;

  // Read problem parameters
  Real const rmin = pin->GetReal("problem", "rmin");   // minimum radius to start TOV integration
  Real const dr = pin->GetReal("problem", "dr");       // radial step for TOV integration 
  int const npts = pin->GetInteger("problem", "npts"); // number of max radial pts for TOV solver

  ntov = pin->GetOrAddInteger("problem", "number_of_stars",1);
  if (ntov<1) {
    msg << "### FATAL ERROR in function [ntovboost]"
	<< std::endl << "Need at least one star";
    ATHENA_ERROR(msg);
  }      
  if (ntov>max_star_num) {
        msg << "### FATAL ERROR in function [ntovboost]"
            << std::endl << "Too many stars, change: max_star_num = " << max_star_num;
        ATHENA_ERROR(msg);
  }      

  // EOS (gamma-law)
  k_adi = pin->GetReal("hydro", "k_adi");
  gamma_adi = pin->GetReal("hydro","gamma");
  
  // Compute TOVs 
  for (int s=0; s<ntov; ++s) {

    std::string s_str = std::to_string(s+1);	
    std::string parname;
    parname = "rhoc";
    parname += s_str;
    Real const rhoc = pin->GetReal("problem", parname); // central value of rest-mass density

    parname = "pos";
    parname += s_str;
    pos[s][0] = pin->GetOrAddReal("problem", parname+"x",0); // position
    pos[s][1] = pin->GetOrAddReal("problem", parname+"y",0);   
    pos[s][2] = pin->GetOrAddReal("problem", parname+"z",0);   

    parname = "mom";
    parname += s_str;
    mom[s][0] = pin->GetOrAddReal("problem", parname+"x",0); // momentum
    mom[s][1] = pin->GetOrAddReal("problem", parname+"y",0);   
    mom[s][2] = pin->GetOrAddReal("problem", parname+"z",0);   
    
    // Alloc 1D buffer for TOV 
    tov[s] = new TOVData;
    int _npts = tov[s]->npts = npts;
    for (int v = 0; v < itov_nv; ++v)
      tov[s]->data[v] = (Real*) malloc(npts*sizeof(Real));
  
    // Solve TOV equations, setting 1D inital data in tov->data
    TOV_solve(tov[s], rhoc, rmin, dr, &_npts);

  }

  //TODO(SB) Currently no check for overlapping stars!
  
  // Add max(rho) output.
  AllocateUserHistoryOutput(1);
  EnrollUserHistoryOutput(0, MaxRho, "max_rho", UserHistoryOperation::max);
  if(adaptive==true)
    EnrollUserRefinementCondition(RefinementCondition);
  return;
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

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
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
  
  // Prepare CC index bounds
  int ilcc = is - NGHOST;
  int iucc = ie + NGHOST;
  int jlcc = js;
  int jucc = je;
  if (block_size.nx2 > 1) {
    jlcc -= NGHOST;
    jucc += NGHOST;
  }
  int klcc = ks;
  int kucc = ke;
  if (block_size.nx3 > 1) {
    klcc -= NGHOST;
    kucc += NGHOST;
  }

  MB_info * mbi = pz4c->mbi;

  // Prepare CX (metric) index bounds
  int jlcx = mbi.il - mbi.ng;
  int jucx = mbi.iu + mbi.ng;
  int jlcx = mbi.jl;
  int jucx = mbi.ju;
  if (block_size.nx2 > 1) {
    jlcx -= mbi.ng;
    jucx += mbi.ng;
  }
  int klcx = mbi.kl;
  int kucx = mbi.ku;
  if (block_size.nx3 > 1) {
    klcx -= mbi.ng;
    kucx += mbi.ng;
  }
  
  // Storage

  // fill these:
  phydro->w_init.Fill(NAN);
  pz4c->storage.u_init.Fill(NAN);
  pz4c->storage.adm_init.Fill(NAN);

  // add to these:
  phydro->w.ZeroClear();
  pz4c->storage.u.ZeroClear();
  pz4c->storage.adm.ZeroClear();
//  MeshBlock *pmb = pmy_block_;
  Real const r2small = 1e-8;
  
  for (int s=0; s<ntov; ++s) {
    
    // Compute a boosted TOV, save on _init
    BoostedTOV(this, tov[s], pos[s], mom[s]);
    
    // Add _init to w, u, adm
    //TODO(SB) there should be a more compact way to do this copy!
    
    // Add hydro on CC
    for (int k=klcc; k<=kucc; ++k) {
      for (int j=jlcc; j<=jucc; ++j) {
	for (int i=ilcc; i<=iucc; ++i) {
	  phydro->w(IDN,k,j,i) += phydro->w_init(IDN,k,j,i);
	  phydro->w(IPR,k,j,i) += phydro->w_init(IPR,k,j,i);
	  phydro->w(IVX,k,j,i) += phydro->w_init(IVX,k,j,i);
	  phydro->w(IVY,k,j,i) += phydro->w_init(IVY,k,j,i);
	  phydro->w(IVZ,k,j,i) += phydro->w_init(IVZ,k,j,i);
	}
      }
    }
    
    // Add metric on VC
    
    for (int k=klcx; k<=kucx; ++k) {
      for (int j=jlcx; j<=jucx; ++j) {
	for (int i=ilcx; i<=iucx; ++i) {
	  for (int v=0; v<Z4c::N_ADM; ++v) {
	    pz4c->storage.adm(v,k,j,i) += pz4c->storage.adm_init(v,k,j,i);
	  }
	  pz4c->storage.u(Z4c::I_Z4c_alpha,k,j,i) += pz4c->storage.u_init(Z4c::I_Z4c_alpha,k,j,i);
	  pz4c->storage.u(Z4c::I_Z4c_betax,k,j,i) += pz4c->storage.u_init(Z4c::I_Z4c_betax,k,j,i);
	  pz4c->storage.u(Z4c::I_Z4c_betay,k,j,i) += pz4c->storage.u_init(Z4c::I_Z4c_betay,k,j,i);
	  pz4c->storage.u(Z4c::I_Z4c_betaz,k,j,i) += pz4c->storage.u_init(Z4c::I_Z4c_betaz,k,j,i);
	  if (s>0) {
	    // Subtract Mikowski
	    pz4c->storage.u(Z4c::I_Z4c_alpha,k,j,i) -= 1.; // lapse has a minus in front.
	    pz4c->storage.adm(Z4c::I_ADM_gxx,k,j,i) -= 1.;
	    pz4c->storage.adm(Z4c::I_ADM_gyy,k,j,i) -= 1.;
	    pz4c->storage.adm(Z4c::I_ADM_gzz,k,j,i) -= 1.;
	    pz4c->storage.adm(Z4c::I_ADM_psi4,k,j,i) -= 1;
	  }
	}
      }
    }
    
  } // ntov loop
  
  // Make sure everything is initialised
  //TODO(SB) there should be a more compact way to do this copy!

  // phydro->w1 = phydro->w_init = phydro->w; // this should work?!
  for (int k=klcc; k<=kucc; ++k) {
    for (int j=jlcc; j<=jucc; ++j) {
      for (int i=ilcc; i<=iucc; ++i) {
	phydro->w_init(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = phydro->w(IDN,k,j,i);
	phydro->w_init(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = phydro->w(IPR,k,j,i);
	phydro->w_init(IVX,k,j,i) = phydro->w1(IVX,k,j,i) = phydro->w(IVX,k,j,i);
	phydro->w_init(IVY,k,j,i) = phydro->w1(IVY,k,j,i) = phydro->w(IVY,k,j,i);
	phydro->w_init(IVZ,k,j,i) = phydro->w1(IVZ,k,j,i) = phydro->w(IVZ,k,j,i);
      }
    }
  }
  
  for (int k=klcx; k<=kucx; ++k) {
    for (int j=jlcx; j<=jucx; ++j) {
      for (int i=ilcx; i<=iucx; ++i) {
	for (int v=0; v<Z4c::N_ADM; ++v) {
	  pz4c->storage.adm_init(v,k,j,i) = pz4c->storage.adm(v,k,j,i);
	}
	pz4c->storage.u_init(Z4c::I_Z4c_alpha,k,j,i) = pz4c->storage.u1(Z4c::I_Z4c_alpha,k,j,i) = pz4c->storage.u(Z4c::I_Z4c_alpha,k,j,i);
	pz4c->storage.u_init(Z4c::I_Z4c_betax,k,j,i) = pz4c->storage.u1(Z4c::I_Z4c_betax,k,j,i) = pz4c->storage.u(Z4c::I_Z4c_betax,k,j,i);
	pz4c->storage.u_init(Z4c::I_Z4c_betay,k,j,i) = pz4c->storage.u1(Z4c::I_Z4c_betay,k,j,i) = pz4c->storage.u(Z4c::I_Z4c_betay,k,j,i); 
	pz4c->storage.u_init(Z4c::I_Z4c_betaz,k,j,i) = pz4c->storage.u1(Z4c::I_Z4c_betaz,k,j,i) = pz4c->storage.u(Z4c::I_Z4c_betaz,k,j,i);
      }
    }
  }
#if MAGNETIC_FIELDS_ENABLED 
//Bfield TODO
//NB EOS hardcoded here
  Real pgasmax = k_adi*SQR(pin->GetReal("problem","rhoc"));
  Real sep = std::abs(pos[0][0] -  pos[1][0]);
  Real pcut = pin->GetReal("problem","pcut")*pgasmax;
  Real b_amp = pin->GetReal("problem","b_amp");
  int magindex=pin->GetInteger("problem","magindex");
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





for(int k = ks-1; k<=ke+1; k++){
  for(int j = js-1; j<=je+1; j++){
  for(int i = is-1; i<=ie+1; i++){
          gamma_dd(0,0,i) = pz4c->ig->map3d_VC2CC(vcgamma_xx(k,j,i));
          gamma_dd(0,1,i) = pz4c->ig->map3d_VC2CC(vcgamma_xy(k,j,i));
          gamma_dd(0,2,i) = pz4c->ig->map3d_VC2CC(vcgamma_xz(k,j,i));
          gamma_dd(1,1,i) = pz4c->ig->map3d_VC2CC(vcgamma_yy(k,j,i));
          gamma_dd(1,2,i) = pz4c->ig->map3d_VC2CC(vcgamma_yz(k,j,i));
          gamma_dd(2,2,i) = pz4c->ig->map3d_VC2CC(vcgamma_zz(k,j,i));
}
   for(int i = is-1; i<=ie+1; i++){
//    Real detgamma = std::sqrt(Det3Metric(gamma_dd,i));
    Real detgamma = 1.0;

    bxcc(k,j,i) = - ((Atot(1,k+1,j,i) - Atot(1,k-1,j,i))/(2.0*pcoord->dx3v(k)))*detgamma;
    bycc(k,j,i) =  ((Atot(0,k+1,j,i) - Atot(0,k-1,j,i))/(2.0*pcoord->dx3v(k)))*detgamma;
    bzcc(k,j,i) = ( (Atot(1,k,j,i+1) - Atot(1,k,j,i-1))/(2.0*pcoord->dx1v(i))
                   - (Atot(0,k,j+1,i) - Atot(0,k,j-1,i))/(2.0*pcoord->dx2v(j)))*detgamma;

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




#endif
 
  // Initialize remaining z4c variables
  pz4c->ADMToZ4c(pz4c->storage.adm,pz4c->storage.u);
  pz4c->ADMToZ4c(pz4c->storage.adm,pz4c->storage.u1); // ???
  pz4c->ADMToZ4c(pz4c->storage.adm_init,pz4c->storage.u_init);
  
  // Initialise coordinate class, CC metric
  //TODO(SB) CHECK: Is this needed in full evo?
  pcoord->UpdateMetric();

  //TODO(WC) can we update coarsec here? is coarse_u_ set yet?
  if(pmy_mesh->multilevel){
    pmr->pcoarsec->UpdateMetric();
  }
  
  // Initialise conserved variables
  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, ilcc, iucc, jlcc, jucc, klcc, kucc);
  
  // Initialise VC matter
  //TODO(WC) (don't strictly need this here, will be caught in task list before used
  pz4c->GetMatter(pz4c->storage.mat, pz4c->storage.adm, phydro->w, pfield->bcc);
  pz4c->ADMConstraints(pz4c->storage.con,pz4c->storage.adm,pz4c->storage.mat,pz4c->storage.u);
  
  return;
}


//void Mesh::DeleteTemporaryUserMeshData() { //TODO update: current version error: no ‘void Mesh::DeleteTemporaryUserMeshData()’ member function declared in class ‘Mesh’
void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  // Free TOV data
  for (int s=0; s<ntov; ++s) {
    if (NULL != tov[s] ) { 
      for (int v = 0; v < itov_nv; ++v) {
	if (NULL != tov[s]->data[v]) {
	  free(tov[s]->data[v]);
	  tov[s]->data[v] = NULL;
	}
      }
      delete tov[s];
      tov[s] = NULL;
    }
  }
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

  int TOV_solve(TOVData * tov, Real rhoc, Real rmin, Real dr, int * npts)  {

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
  // Boosting functions

  Real invg4(Real gd[4][4], Real gu[4][4])
  {
    Real a = gd[0][0];
    Real b = gd[0][1];
    Real c = gd[0][2];
    Real d = gd[0][3];
    Real e = gd[1][1];
    Real f = gd[1][2];
    Real g = gd[1][3];
    Real h = gd[2][2];
    Real i = gd[2][3];
    Real j = gd[3][3];

    Real detg = -(std::pow(c,2)*std::pow(g,2)) + a*std::pow(g,2)*h + 
      std::pow(d,2)*(-std::pow(f,2) + e*h) + 2*b*c*g*i - 2*a*f*g*i - 
      std::pow(b,2)*std::pow(i,2) + a*e*std::pow(i,2) + 
      2*d*(c*f*g - b*g*h - c*e*i + b*f*i) + (std::pow(c,2)*e - 
					     2*b*c*f + a*std::pow(f,2) + std::pow(b,2)*h - a*e*h)*j;
    Real oodetg = 1./detg;
    
    gu[0][0] = oodetg*(std::pow(g,2)*h - 2*f*g*i + std::pow(f,2)*j + e*(std::pow(i,2) - h*j));
    gu[0][1] = oodetg*(-(d*g*h) + d*f*i + c*g*i - b*std::pow(i,2) - c*f*j + b*h*j);
    gu[0][2] = oodetg*(d*f*g - c*std::pow(g,2) - d*e*i + b*g*i + c*e*j - b*f*j);
    gu[0][3] = oodetg*(-(d*std::pow(f,2)) + c*f*g + d*e*h - b*g*h - c*e*i + b*f*i);
    gu[1][1] = oodetg*(std::pow(d,2)*h - 2*c*d*i + std::pow(c,2)*j + a*(std::pow(i,2) - h*j));
    gu[1][2] = oodetg*(-(std::pow(d,2)*f) + c*d*g + b*d*i - a*g*i - b*c*j + a*f*j);
    gu[1][3] = oodetg*(c*d*f - std::pow(c,2)*g - b*d*h + a*g*h + b*c*i - a*f*i);
    gu[2][2] = oodetg*(std::pow(d,2)*e - 2*b*d*g + std::pow(b,2)*j + a*(std::pow(g,2) - e*j));
    gu[2][3] = oodetg*(-(c*d*e) + b*d*f + b*c*g - a*f*g - std::pow(b,2)*i + a*e*i);
    gu[3][3] = oodetg*(std::pow(c,2)*e - 2*b*c*f + std::pow(b,2)*h + a*(std::pow(f,2) - e*h));
    
    gu[1][0] = gu[0][1];
    gu[2][0] = gu[0][2];
    gu[3][0] = gu[0][3];
    gu[2][1] = gu[1][2];
    gu[3][1] = gu[1][3];
    gu[3][2] = gu[2][3];
    
    return detg;
  }

  Real invg3(Real gd[3][3], Real gu[3][3])
  {
    
    gu[0][0] = gd[1][1]*gd[2][2] - gd[1][2]*gd[1][2];
    gu[0][1] = gd[0][2]*gd[1][2] - gd[0][1]*gd[2][2];
    gu[0][2] = gd[0][1]*gd[1][2] - gd[0][2]*gd[1][1];
    gu[1][1] = gd[0][0]*gd[2][2] - gd[0][2]*gd[0][2];
    gu[1][2] = gd[0][1]*gd[0][2] - gd[0][0]*gd[1][2];
    gu[2][2] = gd[0][0]*gd[1][1] - gd[0][1]*gd[0][1];
    
    Real detg = gd[0][0]*gu[0][0] + gd[0][1]*gu[0][1] + gd[0][2]*gu[0][2];
    Real oodetg = 1./detg;
    
    gu[0][0] *= oodetg;
    gu[0][1] *= oodetg;
    gu[0][2] *= oodetg;
    gu[1][1] *= oodetg;
    gu[1][2] *= oodetg;
    gu[2][2] *= oodetg;
    
    gu[1][0] = gu[0][1];
    gu[2][0] = gu[0][2];
    gu[2][1] = gu[1][2];
    
    return detg;
  }
  
  void Gamma44(Real g[4][4], Real dg[4][4][4], Real Gamma[4][4][4])
  {
    Real gi[4][4];
    
    invg4(g,gi);
    
    for (int o=0; o<=3; o++)
      for (int p=0; p<=3; p++)
	for (int q=0; q<=3; q++) {
	  Gamma[o][p][q] = 0.;
	  for (int r=0; r<=3; r++) {
	    Gamma[o][p][q] += 0.5*gi[o][r]* (dg[q][p][r] + dg[p][q][r] - dg[r][p][q]);
	  }
	}
    
  }

  void Gamma34(Real g[4][4], Real dg[4][4][4], Real Gamma[3][3][3])
  {
    Real gi[4][4];
    
    invg4(g,gi);
    
    for (int o=1; o<=3; o++)
      for (int p=1; p<=3; p++)
	for (int q=1; q<=3; q++) {
	  Gamma[o][p][q] = 0.;
	  for (int r=1; r<=3; r++) {
	    Gamma[o][p][q] += 0.5*(gi[o][r] - gi[0][o]*gi[0][r]/gi[0][0])*(dg[q][p][r] + dg[p][q][r] - dg[r][p][q]);
	  }
	}
  }
  
  void Gamma33(Real g[3][3], Real dg[3][3][3], Real Gamma[3][3][3])
  {
    Real gi[3][3];
    
    invg3(g,gi);
    
    for (int o=0; o<3; o++)
      for (int p=0; p<3; p++)
	for (int q=0; q<3; q++) {
	  Gamma[o][p][q] = 0.;
	  for (int r=0; r<3; r++) {
	    Gamma[o][p][q] += 0.5*gi[o][r]*(dg[q][p][r] + dg[p][q][r] - dg[r][p][q]);
	  }
	}
  }

  void set_Lambda(Real LAMBDA[4][4], Real LAMBDAi[4][4], Real xix,Real xiy,Real xiz)
  {
    // small saves the day if no boost is set and you have to divide by 0 inside LAMBDA
    Real const small = 1e-13;
    Real xi = std::sqrt(xix*xix + xiy*xiy + xiz*xiz) + small;
    Real gb = 1./std::sqrt(1.-xi*xi);
    
    LAMBDA[0][0] = gb;
    LAMBDA[0][1] = LAMBDA[1][0] = gb*xix;
    LAMBDA[0][2] = LAMBDA[2][0] = gb*xiy;
    LAMBDA[0][3] = LAMBDA[3][0] = gb*xiz;
    LAMBDA[1][1] = (1.+(gb-1.)*(xix*xix)/(xi*xi));
    LAMBDA[2][2] = (1.+(gb-1.)*(xiy*xiy)/(xi*xi));
    LAMBDA[3][3] = (1.+(gb-1.)*(xiz*xiz)/(xi*xi));
    LAMBDA[1][2] = LAMBDA[2][1] = (gb-1.)*xix*xiy/(xi*xi);
    LAMBDA[1][3] = LAMBDA[3][1] = (gb-1.)*xix*xiz/(xi*xi);
    LAMBDA[2][3] = LAMBDA[3][2] = (gb-1.)*xiy*xiz/(xi*xi);
    
    invg4(LAMBDA,LAMBDAi);
  }

  
  //----------------------------------------------------------------------------------------
  //! \fn void BoostedTOV(MeshBlock* pmb, TOVData * tov, Real pos[3], Real mom[3])
  // \brief Interpolate a boosted TOV on the MB
  //  Notes * this always operates on the _init memory, and overwrites there
  //          For the superposition one needs to copy things before adding the next star
  //
  
  void BoostedTOV(MeshBlock* pmb, TOVData * tov, Real pos[3], Real mom[3])
  {
    
  // Prepare CC index bounds
  int ilcc = pmb->is - NGHOST;
  int iucc = pmb->ie + NGHOST;
  int jlcc = pmb->js;
  int jucc = pmb->je;
  if (block_size.nx2 > 1) {
    jlcc -= NGHOST;
    jucc += NGHOST;
  }
  int klcc = pmb->ks;
  int kucc = pmb->ke;
  if (block_size.nx3 > 1) {
    klcc -= NGHOST;
    kucc += NGHOST;
  }

  MB_info * mbi = pmb->pz4c->mbi;

  // Prepare CX (metric) index bounds
  int jlcx = mbi.il - mbi.ng;
  int jucx = mbi.iu + mbi.ng;
  int jlcx = mbi.jl;
  int jucx = mbi.ju;
  if (block_size.nx2 > 1) {
    jlcx -= mbi.ng;
    jucx += mbi.ng;
  }
  int klcx = mbi.kl;
  int kucx = mbi.ku;
  if (block_size.nx3 > 1) {
    klcx -= mbi.ng;
    kucx += mbi.ng;
  }
  

    
    // Star mass & radius
    const Real M = tov->M;  // Mass of TOV star 
    const Real R = tov->Riso;  // Isotropic Radius of TOV star
    
    // Atmosphere 
    // Real rhomax = tov->data[itov_rho][0];
    // Real fatm = pin->GetReal("problem","fatm");
    // const Real rho_atm = rhomax * fatm;
    // const Real pre_atm = k_adi*std::pow(rhomax*fatm,gamma_adi);

    // Pontwise temp vars
    Real rho_kji, pgas_kji;
    Real lapse_kji, d_lapse_dr_kji, psi4_kji,d_psi4_dr_kji, dummy;
    
    // Boost stuff
    Real dlapse_kji[4], dpsi4_kji[4];
    Real g[4][4],  u[4],  delg[4][4][4],  Gamma[4][4][4];  // BEFORE BOOST
    Real gp[4][4], up[4], delgp[4][4][4], Gammap[4][4][4]; // AFTER  BOOST (p for primed)
    Real gip[4][4];
    Real betax_kji, betay_kji, betaz_kji;
    Real vx_kji, vy_kji, vz_kji;
    Real LAMBDA[4][4],LAMBDAi[4][4];
    Real x, y, z;   

 
    set_Lambda(LAMBDA,LAMBDAi, mom[0],mom[1],mom[2]);

    bool boostme = false;
    if ((SQR(mom[0])+SQR(mom[1])+SQR(mom[2]))>0.)
      boostme = true;
    
    // Initialise primitive values on CC grid
    // --------------------------------------
    
    for (int k=klcc; k<=kucc; ++k) {

      Real dz = pmb->pcoord->x3v(k) - pos[2];
      
      for (int j=jlcc; j<=jucc; ++j) {

	Real dy = pmb->pcoord->x2v(j) - pos[1];

	for (int i=ilcc; i<=iucc; ++i) {
	  
	  Real dx = pmb->pcoord->x1v(i) - pos[0];

	  // Set transformed coordinates
	  if (boostme) {
	    x = LAMBDA[1][1]*dx + LAMBDA[1][2]*dy + LAMBDA[1][3]*dz;
	    y = LAMBDA[2][1]*dx + LAMBDA[2][2]*dy + LAMBDA[2][3]*dz;
	    z = LAMBDA[3][1]*dx + LAMBDA[3][2]*dy + LAMBDA[3][3]*dz;
	  } else {
	    x = dx;
	    y = dy;
	    z = dz;
	  }
	  
	  // Isotropic radius
	  Real r = std::sqrt(x*x+y*y+z*z);
	  
	  // Set exterior to atmos
	  // (Let the EOS decide what to do)
	  rho_kji = 0.0;
	  pgas_kji = 0.0;
	  
	  // Exterior lapse and Psi4 and radial drvts on CC
	  lapse_kji = ((r-M/2.)/(r+M/2.)); 
	  d_lapse_dr_kji = M/((0.5*M+r)*(0.5*M+r));

	  psi4_kji = std::pow((1.+0.5*M/r),4); 
	  d_psi4_dr_kji = -2.*M*std::pow(1.+0.5*M/r,3)/(r*r);

	  // Interior
	  if (r<R) {
	    
	    // Interpolate rho to star interior
	    interp_lag4(tov->data[itov_rho], tov->data[itov_riso], tov->npts, r,
			&rho_kji, &dummy,&dummy);
	    
	    // Pressure from EOS
	    //TODO(SB) general EOS call 
	    pgas_kji = k_adi*pow(rho_kji,gamma_adi); 

	    // Lapse and Psi4 on CC
	    interp_lag4(tov->data[itov_lapse], tov->data[itov_riso], tov->npts, r,
			&lapse_kji, &d_lapse_dr_kji,  &dummy);
	    
	    interp_lag4(tov->data[itov_psi4], tov->data[itov_riso], tov->npts, r,
			&psi4_kji, &d_psi4_dr_kji,  &dummy);
	    	    
	  } 

	  if (boostme) {
	  
	    // Lapse and Psi4 Cartesian derivatives
	    dlapse_kji[1] = d_lapse_dr_kji * x/r;
	    dlapse_kji[2] = d_lapse_dr_kji * y/r;
	    dlapse_kji[3] = d_lapse_dr_kji * z/r;
	    
	    dpsi4_kji[1] = d_psi4_dr_kji * x/r;
	    dpsi4_kji[2] = d_psi4_dr_kji * y/r;
	    dpsi4_kji[3] = d_psi4_dr_kji * z/r;

	    // Transform the velocity
	    // ----------------------
	    
	    Real v2 = 0.0;
	    Real W  = 1.0; //1.0/std::sqrt(1.0 - v2);
	    u[0] = W/lapse_kji;
	    u[1] = u[2] = u[3] = 0.;
	    
	    // Set g_munu
	    g[0][0] = -(lapse_kji*lapse_kji);
	    for (int o=1; o<=3; o++)
	      g[0][o] = g[o][0] = 0.;
	    for (int o=1; o<=3; o++)
	      for (int p=1; p<=3; p++)
		g[o][p] = psi4_kji * (Real)(o==p);
	    
	    // Time derivative of g_munu
	    for (int o=0; o<=3; o++)
	      for (int p=0; p<=3; p++)
		delg[0][o][p] = 0.;
	    
	    // Space derivative of g_munu
	    for (int o=1; o<=3; o++) {
	      // lapse
	      delg[o][0][0] = -2. * lapse_kji * dlapse_kji[o]; 
	      // shift
	      for (int p=1; p<=3; p++)
		delg[o][p][0] = delg[o][0][p] = 0.;
	      // metric
	      for (int p=1; p<=3; p++)
		for (int q=1; q<=3; q++)
		  delg[o][p][q] = dpsi4_kji[o] * (Real)(q==p);
	    }
	    
	    // Contract the 4 Gamma
	    Gamma44(g,delg, Gamma);
	    
	    // Coordinate transformation for u^mu, g_munu
	    for (int o=0; o<=3; o++) {
	      up[o] = 0.;
	      for (int a=0; a<=3; a++) 
		up[o] += LAMBDAi[a][o] * u[a];
	    }
	    
	    for (int o=0; o<=3; o++) {
	      for (int p=0; p<=3; p++) {
		gp[o][p] = 0.;
		for (int a=0; a<=3; a++) {
		  for (int b=0; b<=3; b++) 
		    gp[o][p] += LAMBDA[o][a] * LAMBDA[p][b] * g[a][b];
		}
	      }
	    }
	    
	    // Compute the inverse gp^{munu}
	    invg4(gp,gip);
	    lapse_kji = std::sqrt(-1./gip[0][0]);
	    betax_kji = -gip[0][1]/gip[0][0];
	    betay_kji = -gip[0][2]/gip[0][0];
	    betaz_kji = -gip[0][3]/gip[0][0];
	    
	    // Set transformed u^mu values 
	    W       = up[0] * lapse_kji;
	    vx_kji  = up[1]/W + betax_kji/lapse_kji;
	    vy_kji  = up[2]/W + betay_kji/lapse_kji;
	    vz_kji  = up[3]/W + betaz_kji/lapse_kji;
	    
	    pmb->phydro->w_init(IDN, k, j, i) = rho_kji;
	    pmb->phydro->w_init(IPR, k, j, i) = pgas_kji;
	    pmb->phydro->w_init(IVX, k, j, i) = W * vx_kji;
	    pmb->phydro->w_init(IVY, k, j, i) = W * vy_kji;
	    pmb->phydro->w_init(IVZ, k, j, i) = W * vz_kji;

	  } else {

	    // unboosted, at most translated TOV
	    
	    pmb->phydro->w_init(IDN, k, j, i) = rho_kji;
	    pmb->phydro->w_init(IPR, k, j, i) = pgas_kji;
	    pmb->phydro->w_init(IVX, k, j, i) = 0.;
	    pmb->phydro->w_init(IVY, k, j, i) = 0.;
	    pmb->phydro->w_init(IVZ, k, j, i) = 0.;
	    
	  } // boostme
	  	  
	}
      }
    }
    
    // Initialise metric variables on VC grid 
    // --------------------------------------
    
    for (int k=klcx; k<=kucx; ++k) {

      Real dz = mbi.x3(k) - pos[2];
      
      for (int j=jlcx; j<=jucx; ++j) {

	Real dy = mbi.x2(j) - pos[1];
	for (int i=ilcx; i<=iucx; ++i) {

	  Real dx = mbi.x1(i) - pos[0];

	  // Set transformed coordinates
	  if (boostme) {
	    x = LAMBDA[1][1]*dx + LAMBDA[1][2]*dy + LAMBDA[1][3]*dz;
	    y = LAMBDA[2][1]*dx + LAMBDA[2][2]*dy + LAMBDA[2][3]*dz;
	    z = LAMBDA[3][1]*dx + LAMBDA[3][2]*dy + LAMBDA[3][3]*dz;
	  } else {
	    x = dx;
	    y = dy;
	    z = dz;
	  }

	  // Isotropic radius
	  Real r = std::sqrt(x*x+y*y+z*z);
	  
	  // Exterior lapse and Psi4 and radial drvts on VC
	  lapse_kji = ((r-M/2.)/(r+M/2.)); 
	  d_lapse_dr_kji = M/((0.5*M+r)*(0.5*M+r));

	  psi4_kji = std::pow((1.+0.5*M/r),4); 
	  d_psi4_dr_kji = -2.*M*std::pow(1.+0.5*M/r,3)/(r*r);

	  // Interior
	  if (r<R) {
	    
	    // Interior metric: lapse and Psi4
	    if (r == 0.) {
	      lapse_kji = tov->lapse_0;
	      d_lapse_dr_kji = 0.0;
	      psi4_kji = tov->psi4_0;
	      d_psi4_dr_kji = 0.0; 
	    } else {
	      interp_lag4(tov->data[itov_lapse], tov->data[itov_riso], tov->npts, r,
			  &lapse_kji, &d_lapse_dr_kji,  &dummy);
	      interp_lag4(tov->data[itov_psi4], tov->data[itov_riso], tov->npts, r,
			  &psi4_kji, &d_psi4_dr_kji,  &dummy);
	    }	
	  }

	  if (boostme) {
	  
	    // Lapse and Psi4 Cartesian derivatives
	    dlapse_kji[1] = d_lapse_dr_kji * x/r;
	    dlapse_kji[2] = d_lapse_dr_kji * y/r;
	    dlapse_kji[3] = d_lapse_dr_kji * z/r;
	    
	    dpsi4_kji[1] = d_psi4_dr_kji * x/r;
	    dpsi4_kji[2] = d_psi4_dr_kji * y/r;
	    dpsi4_kji[3] = d_psi4_dr_kji * z/r;
	    
	    // Transform the metric and extr.curv.
	    // -----------------------------------
	    
	    // Set g_munu
	    g[0][0] = -(lapse_kji*lapse_kji);
	    for (int o=1; o<=3; o++)
	      g[0][o] = g[o][0] = 0.;
	    for (int o=1; o<=3; o++)
	      for (int p=1; p<=3; p++)
		g[o][p] = psi4_kji * (Real)(o==p);
	    
	    // Time derivative of g_munu
	    for (int o=0; o<=3; o++)
	      for (int p=0; p<=3; p++)
		delg[0][o][p] = 0.;
	    
	    // Space derivative of g_munu
	    for (int o=1; o<=3; o++) {
	      // lapse
	      delg[o][0][0] = -2. * lapse_kji * dlapse_kji[o]; 
	      // shift
	      for (int p=1; p<=3; p++)
		delg[o][p][0] = delg[o][0][p] = 0.;
	      // metric
	      for (int p=1; p<=3; p++)
		for (int q=1; q<=3; q++)
		  delg[o][p][q] = dpsi4_kji[o] * (Real)(q==p);
	    }

	    // Contract the 4 Gamma
	    Gamma44(g,delg, Gamma);
	    
	    // Coordinate transformation for g_munu, Gamma^sigma_munu
	    for (int o=0; o<=3; o++) {
	      for (int p=0; p<=3; p++) {
		gp[o][p] = 0.;
		for (int a=0; a<=3; a++) {
		  for (int b=0; b<=3; b++) 
		    gp[o][p] += LAMBDA[o][a] * LAMBDA[p][b] * g[a][b];
		}
	      }
	    }
	    
	    for (int o=0; o<=3; o++) {
	      for (int p=0; p<=3; p++) {
		for (int q=0; q<=3; q++) {
		  Gammap[o][p][q] = 0.;
		  for (int a=0; a<=3; a++) {
		    for (int b=0; b<=3; b++) {
		      for (int c=0; c<=3; c++) {
			Gammap[o][p][q] += LAMBDAi[o][a] * LAMBDA[p][b] * LAMBDA[q][c] * Gamma[a][b][c];
			Gammap[o][p][q] += LAMBDAi[o][a] * 0.; // 0 because Lambda is constant
		      }
		    }
		  }
		}
	      }
	    }
	    
	    // Compute the inverse gp^{munu}
	    invg4(gp,gip);
	    lapse_kji = std::sqrt(-1./gip[0][0]);
	    betax_kji = -gip[0][1]/gip[0][0];
	    betay_kji = -gip[0][2]/gip[0][0];
	    betaz_kji = -gip[0][3]/gip[0][0];
	    
	    // Set lapse, shift, ADM metric, and extr. curvature	  
	    pmb->pz4c->storage.u_init(Z4c::I_Z4c_alpha,k,j,i) = lapse_kji;
	    pmb->pz4c->storage.u_init(Z4c::I_Z4c_betax,k,j,i) = betax_kji;
	    pmb->pz4c->storage.u_init(Z4c::I_Z4c_betay,k,j,i) = betay_kji; 
	    pmb->pz4c->storage.u_init(Z4c::I_Z4c_betaz,k,j,i) = betaz_kji;
	    
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_gxx,k,j,i) = gp[1][1];
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_gxy,k,j,i) = gp[1][2];
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_gxz,k,j,i) = gp[1][3];
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_gyy,k,j,i) = gp[2][2];
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_gyz,k,j,i) = gp[2][3];
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_gzz,k,j,i) = gp[3][3];
	    
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_Kxx,k,j,i) = -lapse_kji * Gammap[0][1][1];
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_Kxy,k,j,i) = -lapse_kji * Gammap[0][1][2];
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_Kxz,k,j,i) = -lapse_kji * Gammap[0][1][3];  
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_Kyy,k,j,i) = -lapse_kji * Gammap[0][2][2];
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_Kyz,k,j,i) = -lapse_kji * Gammap[0][2][3];	  
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_Kzz,k,j,i) = -lapse_kji * Gammap[0][3][3];

	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_psi4,k,j,i) = psi4_kji;


	  } else {

	    // unboosted metric
	    pmb->pz4c->storage.u_init(Z4c::I_Z4c_alpha,k,j,i) = lapse_kji;
	    pmb->pz4c->storage.u_init(Z4c::I_Z4c_betax,k,j,i) = 0.0;
	    pmb->pz4c->storage.u_init(Z4c::I_Z4c_betay,k,j,i) = 0.0;
	    pmb->pz4c->storage.u_init(Z4c::I_Z4c_betaz,k,j,i) = 0.0;
	    
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_gxx,k,j,i) = psi4_kji;
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_gxy,k,j,i) = 0.0;
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_gxz,k,j,i) = 0.0;
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_gyy,k,j,i) = psi4_kji;
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_gyz,k,j,i) = 0.0;
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_gzz,k,j,i) = psi4_kji;
	    
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_Kxx,k,j,i) = 0.0;
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_Kxy,k,j,i) = 0.0;
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_Kxz,k,j,i) = 0.0;
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_Kyy,k,j,i) = 0.0;
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_Kyz,k,j,i) = 0.0;
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_Kzz,k,j,i) = 0.0;
	    
	    pmb->pz4c->storage.adm_init(Z4c::I_ADM_psi4,k,j,i) = psi4_kji;
	    
	  } // boostme
	  
	}
      }
    }
    
    return;
  }
  
} // namespace
int RefinementCondition(MeshBlock *pmb)
{

//#ifdef TRACKER_EXTREMA

  Mesh * pmesh = pmb->pmy_mesh;
  ExtremaTracker * ptracker_extrema = pmesh->pextrema_tracker;

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

//#endif // TRACKER_EXTREMA
 return 0;
}
