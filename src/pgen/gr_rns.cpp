//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gr_rns.cpp
//  \brief Initial conditions for rotating neutron star from Stergioulas' RNS code
//         Requires the library:
//         https://bitbucket.org/bernuzzi/rnsc/src/master/

#include <cassert> // assert
#include <iostream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../z4c/z4c.hpp"
#include "../hydro/hydro.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"

#include "RNS.h" // https://bitbucket.org/bernuzzi/rnsc/src/master/

using namespace std;

static ini_data *data;

//int RefinementCondition(MeshBlock *pmb);

Real Maxrho(MeshBlock *pmb, int iout);
Real Minalp(MeshBlock *pmb, int iout);
Real L1rhodiff(MeshBlock *pmb, int iout);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin, int res_flag)
{
  AllocateUserHistoryOutput(3);
  EnrollUserHistoryOutput(0, Maxrho, "max-rho", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, L1rhodiff, "L1rhodiff");
  EnrollUserHistoryOutput(2, Minalp, "min-alp", UserHistoryOperation::min);

  if (!res_flag) {     
    string set_name = "problem";
    RNS_params_set_default();
    string inputfile = pin->GetOrAddString("problem", "filename", "tovgamma2.par");
    RNS_params_set_inputfile((char *) inputfile.c_str());    
    data = RNS_make_initial_data();
  }

  //if(adaptive==true)
  //EnrollUserRefinementCondition(RefinementCondition);
  
  return;
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin, int res_flag)
{
  if (!res_flag)
    RNS_finalise(data);
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Sets the initial conditions.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  
#ifdef Z4C_ASSERT_FINITE
  // as a sanity check (these should be over-written)
  pz4c->adm.psi4.Fill(NAN);
  pz4c->adm.g_dd.Fill(NAN);
  pz4c->adm.K_dd.Fill(NAN);

  pz4c->z4c.chi.Fill(NAN);
  pz4c->z4c.Khat.Fill(NAN);
  pz4c->z4c.Theta.Fill(NAN);
  pz4c->z4c.alpha.Fill(NAN);
  pz4c->z4c.Gam_u.Fill(NAN);
  pz4c->z4c.beta_u.Fill(NAN);
  pz4c->z4c.g_dd.Fill(NAN);
  pz4c->z4c.A_dd.Fill(NAN);

  /*
  pz4c->con.C.Fill(NAN);
  pz4c->con.H.Fill(NAN);
  pz4c->con.M.Fill(NAN);
  pz4c->con.Z.Fill(NAN);
  pz4c->con.M_d.Fill(NAN);
  */
#endif

  bool verbose = pin->GetOrAddBoolean("problem", "verbose", 0);

  Real rhoc = pin->GetReal("problem","rhoc");
  Real k_adi = pin->GetReal("hydro","k_adi");
  Real gamma_adi = pin->GetReal("hydro","gamma");
  Real fatm = pin->GetReal("problem","fatm");
  Real pres_pert = pin->GeOrAddtReal("problem","pres_pert",0);
  Real v_pert = pin->GetOrAddReal("problem","v_pert",0);

  MeshBlock * pmb = pmy_block;
  Coordinates * pco = pmb->pcoord;
  Z4c * pz4c = pmb->pz4c;

  // container with idx / grids pertaining z4c
  MB_info* mbi = &(pz4c->mbi);
  
  //---------------------------------------------------------------------------  
  // Interpolate ADM metric

  if(verbose)
    std::cout << "Interpolating ADM metric on current MeshBlock." << std::endl;
  
  int imin[3] = {0, 0, 0}; 
  int n[3] = {mbi->nn1, mbi->nn1, mbi->nn3};
  int sz = n[0] * n[1] * n[2];

  // temporary variables
  // this could be done instead by accessing and casting the Athena vars but
  // then it is coupled to implementation details etc.
  Real *gxx = new Real[sz], *gyy = new Real[sz], *gzz = new Real[sz];
  Real *gxy = new Real[sz], *gxz = new Real[sz], *gyz = new Real[sz];

  Real *Kxx = new Real[sz], *Kyy = new Real[sz], *Kzz = new Real[sz];
  Real *Kxy = new Real[sz], *Kxz = new Real[sz], *Kyz = new Real[sz];

  Real *alp = new Real[sz];
  Real *betax = new Real[sz], *betay = new Real[sz],*betaz = new Real[sz];

  Real *x = new Real[n[0]];
  Real *y = new Real[n[1]];
  Real *z = new Real[n[2]];

  // Populate coordinates
  for(int i = 0; i < n[0]; ++i) {
    x[i] = mbi->x1(i);
  }
  for(int i = 0; i < n[1]; ++i) {
    y[i] = mbi->x2(i);
  }
  for(int i = 0; i < n[2]; ++i) {
    z[i] = mbi->x3(i);
  }

  // Interpolate geometry
  RNS_Cartesian_interpolation
    (data, // struct containing the previously calculated solution
     imin, // min, max idxs of Cartesian Grid in the three directions
     n,    // TODO WC: check this!!!1
     n,    // total number of indices in each direction
     x,    // x,         // Cartesian coordinates
     y,    // y,
     z,    // z,
     alp,  // alp,       // lapse
     betax,  // betax,   // shift vector
     betay, // betay,
     betaz, // betaz,
     gxx,  // gxx,       // metric components
     gxy,  // gxy,
     gxz,  // gxz,
     gyy,  // gyy,
     gyz,  // gyz,
     gzz,  // gzz,
     Kxx,  // kxx,       // extrinsic curvature components
     Kxy,  // kxy,
     Kxz,  // kxz,
     Kyy,  // kyy,
     Kyz,  // kyz,
     Kzz,  // kzz
     NULL, // rho
     NULL, // epsl
     NULL, // vx
     NULL, // vy
     NULL, // vz
     NULL, // ux
     NULL, // uy
     NULL, // uz
     NULL  // pres
     );

  for (int k=0; k<mbi->nn3; ++k)
  for (int j=0; j<mbi->nn2; ++j)
  for (int i=0; i<mbi->nn1; ++i)
  {
  
    int flat_ix = i + n[0]*(j + n[1]*k);

    pz4c->storage.adm(Z4c::I_ADM_gxx,k,j,i) = gxx[flat_ix];
    pz4c->storage.adm(Z4c::I_ADM_gxy,k,j,i) = gxy[flat_ix];
    pz4c->storage.adm(Z4c::I_ADM_gxz,k,j,i) = gxz[flat_ix];
    pz4c->storage.adm(Z4c::I_ADM_gyy,k,j,i) = gyy[flat_ix];
    pz4c->storage.adm(Z4c::I_ADM_gyz,k,j,i) = gyz[flat_ix];
    pz4c->storage.adm(Z4c::I_ADM_gzz,k,j,i) = gzz[flat_ix];
    

    pz4c->storage.adm(Z4c::I_ADM_Kxx,k,j,i) = Kxx[flat_ix];
    pz4c->storage.adm(Z4c::I_ADM_Kxy,k,j,i) = Kxy[flat_ix];
    pz4c->storage.adm(Z4c::I_ADM_Kxz,k,j,i) = Kxz[flat_ix];
    pz4c->storage.adm(Z4c::I_ADM_Kyy,k,j,i) = Kyy[flat_ix];
    pz4c->storage.adm(Z4c::I_ADM_Kyz,k,j,i) = Kyz[flat_ix];
    pz4c->storage.adm(Z4c::I_ADM_Kzz,k,j,i) = Kzz[flat_ix];

    pz4c->storage.u(Z4c::I_Z4c_alpha,k,j,i) = alp[flat_ix];
    pz4c->storage.u(Z4c::I_Z4c_betax,k,j,i) = betax[flat_ix];
    pz4c->storage.u(Z4c::I_Z4c_betay,k,j,i) = betay[flat_ix];
    pz4c->storage.u(Z4c::I_Z4c_betaz,k,j,i) = betaz[flat_ix];

    //TODO what to do with psi4 buffer?
    //pz4c->storage.adm(Z4c::I_ADM_psi4,k,j,i) = 0.0;

  }

  delete gxx; delete gxy; delete gxz;
  delete gyy; delete gyz; delete gzz; 

  delete Kxx; delete Kxy; delete Kxz; 
  delete Kyy; delete Kyz; delete Kzz;

  delete alp;
  delete betax; delete betay; delete betaz;

  delete x; delete y; delete z;
  
  //---------------------------------------------------------------------------  
  // ADM-to-Z4c
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u1);

  //TODO Needed?
  pcoord->UpdateMetric();  
  if(pmy_mesh->multilevel){
    pmr->pcoarsec->UpdateMetric();
  }

  //---------------------------------------------------------------------------  
  // Interpolate primitives

  if(verbose)
    std::cout << "Interpolating primitives on current MeshBlock." << std::endl;

  n[0] = ncells1; n[1] = ncells2; n[2] = ncells3;
  sz = n[0] * n[1] * n[2];

  Real *rho = new Real[sz], *pres = new Real[sz];
  Real *ux = new Real[sz], *uy = new Real[sz], *uz = new Real[sz];
  
  Real *x = new Real[n[0]];
  Real *y = new Real[n[1]];
  Real *z = new Real[n[2]];

  // Populate coordinates
  for(int i = 0; i < n[0]; ++i) {
    x[i] = pcoord->x1v(i);
  }
  for(int i = 0; i < n[1]; ++i) {
    y[i] = pcoord->x2v(i);
  }
  for(int i = 0; i < n[2]; ++i) {
    z[i] = pcoord->x3v(i);
  }

  // Interpolate primitives
  RNS_Cartesian_interpolation
    (data, // struct containing the previously calculated solution
     imin, // min, max idxs of Cartesian Grid in the three directions
     n,    // 
     n,    // total number of indices in each direction
     x,    // x,         // Cartesian coordinates
     y,    // y,
     z,    // z,
     NULL,  // alp,       // lapse
     NULL,  // betax,   // shift vector
     NULL, // betay,
     NULL, // betaz,
     NULL,  // gxx,       // metric components
     NULL,  // gxy,
     NULL,  // gxz,
     NULL,  // gyy,
     NULL,  // gyz,
     NULL,  // gzz,
     NULL,  // kxx,       // extrinsic curvature components
     NULL,  // kxy,
     NULL,  // kxz,
     NULL,  // kyy,
     NULL,  // kyz,
     NULL,  // kzz
     rho, // rho
     NULL, // epsl - maybe for new EOS we want to take this
     NULL, // vx
     NULL, // vy
     NULL, // vz
     ux, // vx
     uy, // vy
     uz, // vz
     pres  // pres
     );

  // Atmosphere levels
  Real rho_atm = rhoc*fatm;
  Real pres_atm = k_adi*pow(rho_atm,gamma_adi); //TODO (SB) general EOS call
  
  for (int k=0; k<ncells3; ++k)
  for (int j=0; j<ncells2; ++j)
  for (int i=0; i<ncells1; ++i)
  {

    int flat_ix = i + n[0]*(j + n[1]*k);
    Real r = std::sqrt(x[i]*x[i]+y[j]*y[j]+z[k]*z[k]);
    
    phydro->w_init(IDN,k,j,i) = rho[flat_ix];
    phydro->w_init(IPR,k,j,i) =  pres[flat_ix];
    phydro->w_init(IVX, k, j, i) = ux[flat_ix];
    phydro->w_init(IVY, k, j, i) = uy[flat_ix];
    phydro->w_init(IVZ, k, j, i) = uz[flat_ix];

    // Add perturbations
    if (pres_pert) {
      phydro->w_init(IPR,k,j,i) -= pres_pert * pres[flat_ix];      
    }
    if (v_pert) {
      phydro->w_init(IVX, k, j, i) -= v_pert * std::cos(M_PI*r/(2.0*8.0))*x[i]/r;
      phydro->w_init(IVY, k, j, i) -= v_pert * std::cos(M_PI*r/(2.0*8.0))*y[j]/r;
      phydro->w_init(IVZ, k, j, i) -= v_pert * std::cos(M_PI*r/(2.0*8.0))*z[k]/r;
    }
    
    phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = phydro->w_init(IDN,k,j,i);
    phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = phydro->w_init(IPR,k,j,i);
    phydro->w(IVX,k,j,i) = phydro->w1(IVX,k,j,i) = phydro->w_init(IVX,k,j,i);
    phydro->w(IVY,k,j,i) = phydro->w1(IVY,k,j,i) = phydro->w_init(IVY,k,j,i);
    phydro->w(IVZ,k,j,i) = phydro->w1(IVZ,k,j,i) = phydro->w_init(IVZ,k,j,i);
    
  }

  delete rho;
  delete pres;
  delete ux; delete uy; delete uz;
  
  delete x; delete y; delete z;
  
  //---------------------------------------------------------------------------  
  // Initialise conserved variables

  if(verbose)
    std::cout << "Initializing conservatives on current MeshBlock." << std::endl;
  
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

  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, il, iu, jl, ju, kl, ku);

  //---------------------------------------------------------------------------  
  // Initialise matter & ADM constraints
  //TODO(WC) (don't strictly need this here, will be caught in task list before used
  
  if(verbose)
    std::cout << "Initializing matter and constraints on current MeshBlock." << std::endl;

  pz4c->GetMatter(pz4c->storage.mat, pz4c->storage.adm, phydro->w,pfield->bcc);
  pz4c->ADMConstraints(pz4c->storage.con,pz4c->storage.adm,pz4c->storage.mat,pz4c->storage.u);
  
#ifdef Z4C_ASSERT_FINITE
  pz4c->assert_is_finite_adm();
  pz4c->assert_is_finite_con();
  pz4c->assert_is_finite_mat();
  pz4c->assert_is_finite_z4c();
#endif 

  return;
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

Real Minalp(MeshBlock *pmb, int iout) {
  Real min_alp = 1.0e100;
  int is = pmb->is, ie = pmb->ie+1, js = pmb->js, je = pmb->je+1, ks = pmb->ks, ke = pmb->ke+1;
  AthenaArray<Real> alpha;
  alpha.InitWithShallowSlice(pmb->pz4c->storage.u,Z4c::I_Z4c_alpha,1);
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        min_alp = std::min(std::abs(alpha(k,j,i)), min_alp);
      }
    }
  }
  return min_alp;
}

Real L1rhodiff(MeshBlock *pmb, int iout) {
  Real L1rho = 0.0;
  Real vol,dx,dy,dz;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        dx = pmb->pcoord->dx1v(i);
        dy = pmb->pcoord->dx2v(j);
        dz = pmb->pcoord->dx3v(k);
        vol = dx*dy*dz;
        L1rho += std::abs(pmb->phydro->w(IDN,k,j,i) - pmb->phydro->w_init(IDN,k,j,i))*vol;
      }
    }
  }
  return L1rho;
}
