
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file blast.cpp
//  \brief Problem generator for spherical blast wave problem.  Works in Cartesian,
//         cylindrical, and spherical coordinates.  Contains post-processing code
//         to check whether blast is spherical for regression tests
//
// REFERENCE: P. Londrillo & L. Del Zanna, "High-order upwind schemes for
//   multidimensional MHD", ApJ, 530, 508 (2000), and references therein.

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <cstring>    // strcmp()
#include <sstream>
#include <stdexcept>
#include <string>
#include <fstream>
#include <iostream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"
#include "../utils/read_lorene.hpp"
#include "../z4c/z4c.hpp"

Real threshold;

#ifndef LORENE_EOS
#define LORENE_EOS (1)
#endif

int RefinementCondition(MeshBlock *pmb);
int interp_locate(Real *x, int Nx, Real xval);
void interp_lag4(Real *f, Real *x, int Nx, Real xv,
       Real *fv_p, Real *dfv_p, Real *ddfv_p );
double linear_interp(double *f, double *x, int n, double xv);

  std::string table_fname;
  LoreneTable * Table = NULL; // Lorene table object
  std::string filename, filename_Y; // Lorene table fnames

void Mesh::InitUserMeshData(ParameterInput *pin, int res_flag) {
  if (adaptive) {
    EnrollUserRefinementCondition(RefinementCondition);
    threshold = pin->GetReal("problem","thr");
  }
    filename   = pin->GetString("hydro", "lorene");
    filename_Y = pin->GetString("hydro", "lorene_Y");
    Table = new LoreneTable;
    ReadLoreneTable(filename, Table);
    ReadLoreneFractions(filename_Y, Table);
    ConvertLoreneTable(Table);
    // PS uses different floor to reprimand
    #if USETM
      Table->rho_atm = pin->GetReal("hydro", "dfloor"); 
    #endif
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Spherical blast wave test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  std::cout << "IN PG" << std::endl;
  Real rout = pin->GetReal("problem", "radius");
  Real rin  = rout - pin->GetOrAddReal("problem", "ramp", 0.0);
  Real dfloor   = pin->GetOrAddReal("hydro", "dfloor", 1.0);
  Real drat     = pin->GetOrAddReal("problem", "drat", 1.0);
  Real b0, angle;
  if (MAGNETIC_FIELDS_ENABLED) {
    b0 = pin->GetReal("problem", "b0");
    angle = (PI/180.0)*pin->GetReal("problem", "angle");
  }

  std::cout << "OUT PG" << std::endl;
  std::cout << std::flush;
  //phydro->w.Fill(NAN);
  //phydro->w1.Fill(NAN);
  //phydro->u.Fill(NAN);
  //phydro->temperature.Fill(NAN);
  //pz4c->storage.u.Fill(NAN);
  //pz4c->storage.adm.Fill(NAN);

  Real T_atmosphere = pin->GetReal("hydro", "tfloor");
  Real Y_atmosphere = pin->GetReal("hydro", "y0_atmosphere");

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

  // setup uniform ambient medium with spherical over-pressured region
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        Real rad;

        #if USETM
        for (int l=0; l<NSCALARS; l++) {
          pscalars->r(l,k,j,i) = 0.0;
          pscalars->s(l,k,j,i) = 0.0;
        }
        #endif
          Real x = pcoord->x1v(i);
          Real y = pcoord->x2v(j);
          Real z = pcoord->x3v(k);
          //std::cout << "x = " << x << ", y = " << y << ", z = " << z << std::endl;
          rad = std::sqrt(SQR(x) + SQR(y) + SQR(z));
        //std::cout << "rad = " << rad << ", rout = " << rout << std::endl;
        Real rho_kji = 0.0;
        Real pgas_kji = 0.0;
        Real Y_kji = Y_atmosphere;
        Real v_kji = 0.0;
        if (rad < rout) {
          if (rad < rin) {
            rho_kji = drat*dfloor;
          } else {
            Real f = (rad - rin) / (rout - rin);
            std::cout << "f = " << f << std::endl;
            Real log_den = (1.0-f) * std::log(drat*dfloor) + f * std::log(dfloor);
            std::cout << "drat = " << drat << ", dfloor = " << dfloor << ", log_den = " << log_den << std::endl;
            rho_kji = std::exp(log_den);
          }
            std::cout << "RHO = " << rho_kji << std::endl;
            std::cout << std::flush;
            Y_kji = linear_interp(Table->Y[0], Table->data[tab_logrho], Table->size, log(rho_kji));
            Real n_kji = rho_kji/(peos->GetEOS().GetBaryonMass());
            std::cout << "N = " << n_kji << std::endl;
            pgas_kji = peos->GetEOS().GetPressure(n_kji,T_atmosphere,&Y_kji);
            std::cout << "PGAS = " << pgas_kji << std::endl;
        }
        phydro->w_init(IDN, k, j, i) = rho_kji;
        phydro->w_init(IPR, k, j, i) = pgas_kji;
        phydro->w_init(IVX, k, j, i) = 0.0;
        phydro->w_init(IVY, k, j, i) = 0.0;
        phydro->w_init(IVZ, k, j, i) = 0.0;
        if(NSCALARS==1) {
          pscalars->r(0,k,j,i) = Y_kji;
        }
        phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = phydro->w_init(IDN,k,j,i);
        phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = phydro->w_init(IPR,k,j,i);
        phydro->w(IVX,k,j,i) = phydro->w1(IVX,k,j,i) = phydro->w_init(IVX,k,j,i);
        phydro->w(IVY,k,j,i) = phydro->w1(IVY,k,j,i) = phydro->w_init(IVY,k,j,i);
        phydro->w(IVZ,k,j,i) = phydro->w1(IVZ,k,j,i) = phydro->w_init(IVZ,k,j,i);
        phydro->temperature(k,j,i) = T_atmosphere;
      }
    }
  }
  for (int k=kl; k<=ku+1; ++k) {
    for (int j=jl; j<=ju+1; ++j) {
      for (int i=il; i<=iu+1; ++i) {
        pz4c->storage.u(Z4c::I_Z4c_alpha,k,j,i) = 1.0;
        pz4c->storage.u(Z4c::I_Z4c_betax,k,j,i) = 0.0; 
        pz4c->storage.u(Z4c::I_Z4c_betay,k,j,i) = 0.0; 
        pz4c->storage.u(Z4c::I_Z4c_betaz,k,j,i) = 0.0; 
        pz4c->storage.adm(Z4c::I_ADM_gxx,k,j,i) = 1.0;
        pz4c->storage.adm(Z4c::I_ADM_gyy,k,j,i) = 1.0;
        pz4c->storage.adm(Z4c::I_ADM_gzz,k,j,i) = 1.0;
        pz4c->storage.adm(Z4c::I_ADM_gxy,k,j,i) = 0.0;
        pz4c->storage.adm(Z4c::I_ADM_gxz,k,j,i) = 0.0;
        pz4c->storage.adm(Z4c::I_ADM_gyz,k,j,i) = 0.0;
        pz4c->storage.adm(Z4c::I_ADM_Kxx,k,j,i) = 0.0; 
        pz4c->storage.adm(Z4c::I_ADM_Kyy,k,j,i) = 0.0; 
        pz4c->storage.adm(Z4c::I_ADM_Kzz,k,j,i) = 0.0; 
        pz4c->storage.adm(Z4c::I_ADM_Kxy,k,j,i) = 0.0;
        pz4c->storage.adm(Z4c::I_ADM_Kxz,k,j,i) = 0.0;
        pz4c->storage.adm(Z4c::I_ADM_Kyz,k,j,i) = 0.0;
        pz4c->storage.adm(Z4c::I_ADM_psi4,k,j,i) = 1.0;
        pz4c->storage.u_init(Z4c::I_Z4c_alpha,k,j,i) = 1.0;
        pz4c->storage.u_init(Z4c::I_Z4c_betax,k,j,i) = 0.0; 
        pz4c->storage.u_init(Z4c::I_Z4c_betay,k,j,i) = 0.0; 
        pz4c->storage.u_init(Z4c::I_Z4c_betaz,k,j,i) = 0.0; 
        pz4c->storage.adm_init(Z4c::I_ADM_gxx,k,j,i) = 1.0;
        pz4c->storage.adm_init(Z4c::I_ADM_gyy,k,j,i) = 1.0;
        pz4c->storage.adm_init(Z4c::I_ADM_gzz,k,j,i) = 1.0;
        pz4c->storage.adm_init(Z4c::I_ADM_gxy,k,j,i) = 0.0;
        pz4c->storage.adm_init(Z4c::I_ADM_gxz,k,j,i) = 0.0;
        pz4c->storage.adm_init(Z4c::I_ADM_gyz,k,j,i) = 0.0;
        pz4c->storage.adm_init(Z4c::I_ADM_Kxx,k,j,i) = 0.0; 
        pz4c->storage.adm_init(Z4c::I_ADM_Kyy,k,j,i) = 0.0; 
        pz4c->storage.adm_init(Z4c::I_ADM_Kzz,k,j,i) = 0.0; 
        pz4c->storage.adm_init(Z4c::I_ADM_Kxy,k,j,i) = 0.0;
        pz4c->storage.adm_init(Z4c::I_ADM_Kxz,k,j,i) = 0.0;
        pz4c->storage.adm_init(Z4c::I_ADM_Kyz,k,j,i) = 0.0;
        pz4c->storage.adm_init(Z4c::I_ADM_psi4,k,j,i) = 1.0;
      }
    }
  }

  pz4c->ADMToZ4c(pz4c->storage.adm,pz4c->storage.u);
  pz4c->ADMToZ4c(pz4c->storage.adm,pz4c->storage.u1); //?????
  pz4c->ADMToZ4c(pz4c->storage.adm_init,pz4c->storage.u_init);
  
  // Initialise coordinate class, CC metric
  pcoord->UpdateMetric();
  //
  //     //TODO(WC) can we update coarsec here? is coarse_u
  //     _ set yet?
  if(pmy_mesh->multilevel){
    pmr->pcoarsec->UpdateMetric();
  }
  // initialize interface B and total energy
  if (MAGNETIC_FIELDS_ENABLED) {
    pfield->b.x1f.ZeroClear();
    pfield->b.x2f.ZeroClear();
    pfield->b.x3f.ZeroClear();
    pfield->bcc.ZeroClear();
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu+1; ++i) {
          pfield->b.x1f(k,j,i) = b0 * std::cos(angle);
        }
      }
    }
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju+1; ++j) {
        for (int i=il; i<=iu; ++i) {
          pfield->b.x2f(k,j,i) = b0 * std::sin(angle);
        }
      }
    }
    for (int k=kl; k<=ku+1; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          pfield->b.x3f(k,j,i) = 0.0;
        }
      }
    }
    //for (int k=kl; k<=ku; ++k) {
    //  for (int j=jl; j<=ju; ++j) {
    //    for (int i=il; i<=iu; ++i) {
    //      phydro->u(IEN,k,j,i) += 0.5*b0*b0;
    //    }
    //  }
    //} 
    pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, il,iu,jl,ju,kl,ku);
  }
  peos->PrimitiveToConserved(phydro->w, pscalars->r, pfield->bcc, phydro->u, pscalars->s, pcoord, 
                             il, iu, jl, ju, kl, ku);
  pz4c->GetMatter(pz4c->storage.mat, pz4c->storage.adm, phydro->w, pscalars->r, pfield->bcc);
  pz4c->ADMConstraints(pz4c->storage.con,pz4c->storage.adm,pz4c->storage.mat,pz4c->storage.u); 

  return;
}

// refinement condition: check the maximum pressure gradient
int RefinementCondition(MeshBlock *pmb) {
  AthenaArray<Real> &w = pmb->phydro->w;
  Real maxeps = 0.0;
  if (pmb->pmy_mesh->f3) {
    for (int k=pmb->ks-1; k<=pmb->ke+1; k++) {
      for (int j=pmb->js-1; j<=pmb->je+1; j++) {
        for (int i=pmb->is-1; i<=pmb->ie+1; i++) {
          Real eps = std::sqrt(SQR(0.5*(w(IPR,k,j,i+1) - w(IPR,k,j,i-1)))
                               +SQR(0.5*(w(IPR,k,j+1,i) - w(IPR,k,j-1,i)))
                               +SQR(0.5*(w(IPR,k+1,j,i) - w(IPR,k-1,j,i))))/w(IPR,k,j,i);
          maxeps = std::max(maxeps, eps);
        }
      }
    }
  } else if (pmb->pmy_mesh->f2) {
    int k = pmb->ks;
    for (int j=pmb->js-1; j<=pmb->je+1; j++) {
      for (int i=pmb->is-1; i<=pmb->ie+1; i++) {
        Real eps = std::sqrt(SQR(0.5*(w(IPR,k,j,i+1) - w(IPR,k,j,i-1)))
                             + SQR(0.5*(w(IPR,k,j+1,i) - w(IPR,k,j-1,i))))/w(IPR,k,j,i);
        maxeps = std::max(maxeps, eps);
      }
    }
  } else {
    return 0;
  }

  if (maxeps > threshold) return 1;
  if (maxeps < 0.25*threshold) return -1;
  return 0;
}

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
