//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file puncture_z4c.cpp
//  \brief implementation of functions in the Z4c class for initializing puntures evolution

// C++ standard headers
#include <cmath> // pow

// Athena++ headers
#include "../../z4c/z4c.hpp"
#include "../../z4c/z4c_macro.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../mesh/mesh.hpp"
#include "../hydro.hpp"


#ifdef RNS
#include "RNS.h"
#endif
//----------------------------------------------------------------------------------------
// \!fn void Z4c::ADMTwoPunctures(AthenaArray<Real> & u)
// \brief Initialize ADM vars to two punctures

void Hydro::RNS_Metric(ParameterInput *pin, AthenaArray<Real> & u_adm, AthenaArray<Real> & u, AthenaArray<Real> & u1, ini_data *data)
{
  bool verbose = pin->GetOrAddBoolean("problem", "verbose", 0);
  
  //if(verbose)
  //  Z4c::DebugInfoVars();


  MeshBlock * pmb = pmy_block;
  Coordinates * pco = pmb->pcoord;
  Z4c * pz4c = pmb->pz4c;

  Z4c::ADM_vars adm;
  pz4c->SetADMAliases(u_adm, adm);
  Z4c::Z4c_vars z4c;
  pz4c->SetZ4cAliases(u, z4c);
  Z4c::Z4c_vars z4c_1;
  pz4c->SetZ4cAliases(u1, z4c_1);

  // Flat spacetime
//  ADMMinkowski(u_adm);

  //--

  // construct initial data set
  if(verbose)
    std::cout << "Generating rns data." << std::endl;

  if(verbose)
    std::cout << "Done!" << std::endl;

  // interpolate to ADM variables based on solution
  if(verbose)
    std::cout << "Interpolating current MeshBlock." << std::endl;

  int imin[3] = {0, 0, 0};

  // dimensions of block in each direction
  //int n[3] = {(*pmb).block_size.nx1 + 2 * GSIZEI,
  //            (*pmb).block_size.nx2 + 2 * GSIZEJ,
  //            (*pmb).block_size.nx3 + 2 * GSIZEK};
  int n[3] = {pz4c->mbi.nn1, pz4c->mbi.nn2, pz4c->mbi.nn3};
  int sz = n[0] * n[1] * n[2];
  // this could be done instead by accessing and casting the Athena vars but
  // then it is coupled to implementation details etc.
  Real *gxx = new Real[sz], *gyy = new Real[sz], *gzz = new Real[sz];
  Real *gxy = new Real[sz], *gxz = new Real[sz], *gyz = new Real[sz];

  Real *Kxx = new Real[sz], *Kyy = new Real[sz], *Kzz = new Real[sz];
  Real *Kxy = new Real[sz], *Kxz = new Real[sz], *Kyz = new Real[sz];
  Real *psi = new Real[sz];
  Real *alp = new Real[sz];
  Real *betax = new Real[sz], *betay = new Real[sz],*betaz = new Real[sz];

  Real *x = new Real[n[0]];
  Real *y = new Real[n[1]];
  Real *z = new Real[n[2]];

  // need to populate coordinates
  for(int ix_I = 0; ix_I < n[0]; ix_I++){
    //x[ix_I] = pco->x1v(ix_I);
    x[ix_I] = pz4c->mbi.x1(ix_I);
  }

  for(int ix_J = 0; ix_J < n[1]; ix_J++){
    //y[ix_J] = pco->x2v(ix_J);
    y[ix_J] = pz4c->mbi.x2(ix_J);
  }

  for(int ix_K = 0; ix_K < n[2]; ix_K++){
    //z[ix_K] = pco->x3v(ix_K);
    z[ix_K] = pz4c->mbi.x3(ix_K);
  }

  RNS_Cartesian_interpolation
    (data, // struct containing the previously calculated solution
     imin, // min, max idxs of Cartesian Grid in the three directions
     n,    // 
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


  int flat_ix;

  GLOOP3(k,j,i){
    flat_ix = i + n[0]*(j + n[1]*k);

    adm.g_dd(0, 0, k, j, i) = gxx[flat_ix];
    adm.g_dd(1, 1, k, j, i) = gyy[flat_ix];
    adm.g_dd(2, 2, k, j, i) = gzz[flat_ix];
    adm.g_dd(0, 1, k, j, i) = gxy[flat_ix];
    adm.g_dd(0, 2, k, j, i) = gxz[flat_ix];
    adm.g_dd(1, 2, k, j, i) = gyz[flat_ix];

    adm.K_dd(0, 0, k, j, i) = Kxx[flat_ix];
    adm.K_dd(1, 1, k, j, i) = Kyy[flat_ix];
    adm.K_dd(2, 2, k, j, i) = Kzz[flat_ix];
    adm.K_dd(0, 1, k, j, i) = Kxy[flat_ix];
    adm.K_dd(0, 2, k, j, i) = Kxz[flat_ix];
    adm.K_dd(1, 2, k, j, i) = Kyz[flat_ix];

    z4c_1.alpha(k,j,i) = z4c.alpha(k,j,i)        = alp[flat_ix];
    z4c_1.beta_u(0,k,j,i) = z4c.beta_u(0,k,j,i)     = betax[flat_ix];
    z4c_1.beta_u(1,k,j,i) = z4c.beta_u(1,k,j,i)     = betay[flat_ix];
    z4c_1.beta_u(2,k,j,i) = z4c.beta_u(2,k,j,i)     = betaz[flat_ix];
  }


  /*free(gxx); free(gyy); free(gzz);
  free(gxy); free(gxz); free(gyz);

  free(Kxx); free(Kyy); free(Kzz);
  free(Kxy); free(Kxz); free(Kyz);

  free(psi); free(alp);
  free(betax); free(betay); free(betaz);

  free(x); free(y); free(z);*/
  delete[] gxx;
  delete[] gyy;
  delete[] gzz;
  delete[] gxy;
  delete[] gxz;
  delete[] gyz;

  delete[] Kxx;
  delete[] Kyy;
  delete[] Kzz;
  delete[] Kxy;
  delete[] Kxz;
  delete[] Kyz;

  delete[] psi;
  delete[] alp;
  delete[] betax;
  delete[] betay;
  delete[] betaz;

  delete[] x;
  delete[] y;
  delete[] z;

  if(verbose)
    std::cout << "\n\n<-Z4c::ADMTwoPunctures\n\n";
}





void Hydro::RNS_Hydro(ParameterInput *pin, AthenaArray<Real> & w, AthenaArray<Real> & w1, AthenaArray<Real> & w_init, ini_data *data)
{
  bool verbose = pin->GetOrAddBoolean("problem", "verbose", 0);
  
  //if(verbose)
  //  Z4c::DebugInfoVars();

  Real rhoc = pin->GetReal("problem","rhoc");
  Real k_adi = pin->GetReal("hydro","k_adi");
  Real gamma_adi = pin->GetReal("hydro","gamma");
  Real fatm = pin->GetReal("problem","fatm");
  Real pres_pert = pin->GetReal("problem","pres_pert");
  Real v_pert = pin->GetReal("problem","v_pert");

  MeshBlock * pmb = pmy_block;
  Coordinates * pco = pmb->pcoord;
  Z4c * pz4c = pmb->pz4c;

  // Flat spacetime
//  ADMMinkowski(u_adm);

  //--

  // construct initial data set
  if(verbose)
    std::cout << "Generating two puncture data." << std::endl;

  if(verbose)
    std::cout << "Done!" << std::endl;

  // interpolate to ADM variables based on solution
  if(verbose)
    std::cout << "Interpolating current MeshBlock." << std::endl;

  int imin[3] = {0, 0, 0};

  // dimensions of block in each direction
  //int n[3] = {(*pmb).block_size.nx1 + 2 * GSIZEI,
  //            (*pmb).block_size.nx2 + 2 * GSIZEJ,
  //            (*pmb).block_size.nx3 + 2 * GSIZEK};
  int n[3] = {pz4c->mbi.nn1-1, pz4c->mbi.nn2-1, pz4c->mbi.nn3-1};

  int sz = n[0] * n[1] * n[2];
  // this could be done instead by accessing and casting the Athena vars but
  // then it is coupled to implementation details etc.
  Real *rho = new Real[sz], *pres = new Real[sz];
  Real *ux = new Real[sz], *uy = new Real[sz], *uz = new Real[sz];

  /*Real *gxx = new Real[sz], *gyy = new Real[sz], *gzz = new Real[sz];
  Real *gxy = new Real[sz], *gxz = new Real[sz], *gyz = new Real[sz];*/

  Real *x = new Real[n[0]];
  Real *y = new Real[n[1]];
  Real *z = new Real[n[2]];

  // need to populate coordinates
  for(int ix_I = 0; ix_I < n[0]; ix_I++){
    //x[ix_I] = pco->x1v(ix_I);
    x[ix_I] = pco->x1v(ix_I);
  }

  for(int ix_J = 0; ix_J < n[1]; ix_J++){
    //y[ix_J] = pco->x2v(ix_J);
    y[ix_J] = pco->x2v(ix_J);
  }

  for(int ix_K = 0; ix_K < n[2]; ix_K++){
    //z[ix_K] = pco->x3v(ix_K);
    z[ix_K] = pco->x3v(ix_K);
  }

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

  Real rho_atm = rhoc*fatm;
  Real pres_atm = k_adi*pow(rho_atm,gamma_adi);
  int flat_ix;
  double psi4;
  Real r;
//  Real vsq, Wlor;

  CLOOP3(k,j,i){
    flat_ix = i + n[0]*(j + n[1]*k);
    r = x[i]*x[i]+y[j]*y[j]+z[k]*z[k];
    r = std::sqrt(r);
/*
    vsq = vx[flat_idx]*vx[flat_idx]*gxx[flat_idx] + 
          vy[flat_idx]*vy[flat_idx]*gyy[flat_idx] +
          vz[flat_idx]*vz[flat_idx]*gzz[flat_idx] +
          2.0*vx[flat_idx]*vy[flat_idx]*gxy[flat_idx] +
          2.0*vx[flat_idx]*vz[flat_idx]*gxz[flat_idx] +
          2.0*vy[flat_idx]*vz[flat_idx]*gyz[flat_idx];

    Wlor = 1.0/(1.0-vsq);
*/
    if(rho[flat_ix] > rho_atm){
        w_init(IDN,k,j,i) = w1(IDN,k,j,i) = w(IDN,k,j,i) = rho[flat_ix];
        w_init(IPR,k,j,i) = w1(IPR,k,j,i) = w(IPR,k,j,i) = pres[flat_ix]- pres_pert*pres[flat_ix];
        w_init(IVX,k,j,i) = w1(IVX,k,j,i) = w(IVX,k,j,i) = ux[flat_ix]  - v_pert;
        w_init(IVY,k,j,i) = w1(IVY,k,j,i) = w(IVY,k,j,i) = uy[flat_ix]- v_pert;
        w_init(IVZ,k,j,i) = w1(IVZ,k,j,i) = w(IVZ,k,j,i) = uz[flat_ix]- v_pert;
    } else{
        w_init(IDN,k,j,i) = w1(IDN,k,j,i) = w(IDN,k,j,i) = rho_atm;
        w_init(IPR,k,j,i) = w1(IPR,k,j,i) = w(IPR,k,j,i) = pres_atm - pres_pert*pres_atm;
        w_init(IVX,k,j,i) = w1(IVX,k,j,i) = w(IVX,k,j,i) = 0.0;
        w_init(IVY,k,j,i) = w1(IVY,k,j,i) = w(IVY,k,j,i) = 0.0;
        w_init(IVZ,k,j,i) = w1(IVZ,k,j,i) = w(IVZ,k,j,i) = 0.0; 
    }

  }


  /*free(rho); free(pres);

  free(ux); free(uy); free(uz);


  free(x); free(y); free(z);*/
  delete[] rho;
  delete[] pres;
  delete[] ux;
  delete[] uy;
  delete[] uz;
  delete[] x;
  delete[] y;
  delete[] z;

  if(verbose)
    std::cout << "\n\nRNS ID\n\n";
}

