//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file m1_fiducial_velocity.cpp
//  \brief computes the fiducial velocity (CC) on a mesh block 

// C++ standard headers
//#include <cmath> 
#include <sstream>

// Athena++ headers
#include "../mesh/mesh.hpp"
#include "../z4c/z4c.hpp"
#include "../hydro/hydro.hpp"
#include "m1.hpp"

void M1::CalcFiducialVelocity()
{  
  if (M1_CALCFIDUCIALVELOCITY_OFF) return;
  M1_DEBUG_PR("in: CalcFiducialVelocity");

  MeshBlock * pmb = pmy_block;
  if (fiducial_velocity == "fluid") {
    GCLOOP2(k,j) {
      for(int a = 0; a < NDIM; ++a) {
	GCLOOP1(i) {
	  fidu.vel_u(a,k,j,i) = pmb->phydro->w(IVX+a,k,j,i);
        }
      }
    }
  } else if (fiducial_velocity == "mixed") {  
    GCLOOP3(k,j,i) {
      Real const rho = pmb->phydro->w(IDN,k,j,i);
      Real const fac = 1.0/std::max(rho, fiducial_vel_rho_fluid);
      for(int a = 0; a < NDIM; ++a) {
	fidu.vel_u(a,k,j,i) = pmb->phydro->w(IVX+a,k,j,i) * rho * fac;      
      }
    }
  } else if (fiducial_velocity == "zero") {
    fidu.vel_u.ZeroClear();
    fidu.Wlorentz.Fill(1.0);
    return;
  } else if (fiducial_velocity == "test") {
    return;
  } else {
    std::ostringstream msg;
    msg << "Unknown fiducial velocity " << fiducial_velocity << std::endl;
    ATHENA_ERROR(msg);
  }
  
  // Here fidu.vel_u contains utilde^i = W v^i
  // compute Lorentz factor and store v^i
  
  // Calculating here w_lorentz means an extra VC2CC and the storage.
  // Another option is to calculate it online every time needed;
  // Note in the latter case (part of) the CC metric is computed in the places W is required.
  
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> vc_adm_g_dd;
  vc_adm_g_dd.InitWithShallowSlice(pmy_block->pz4c->storage.adm, Z4c::I_ADM_gxx);
  
  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> g_dd;
  g_dd.NewTensorPointwise();
  
  GCLOOP3(k,j,i) {
    // Go from ADM 3-metric VC (AthenaArray/Tensor)
    // to ADM 3-metric on CC at ijk (TensorPointwise)       
    for(int a = 0; a < NDIM; ++a) {
      for(int b = a; b < NDIM; ++b) {
	g_dd(a,b) = VCInterpolation(vc_adm_g_dd(a,b),k,j,i);
      }
    }
    // Compute W Lorentz from utilde
    Real const W = GetWLorentz_from_utilde(fidu.vel_u(0,k,j,i),
                                           fidu.vel_u(1,k,j,i),
                                           fidu.vel_u(2,k,j,i),
                                           g_dd(0,0), g_dd(0,1), g_dd(0,2),
                                           g_dd(1,1), g_dd(1,2), g_dd(2,2),
                                           nullptr, nullptr, nullptr,  nullptr);
    fidu.Wlorentz(k,j,i) = W;
    Real const ooW = 1.0/W;
    for(int a = 0; a < NDIM; ++a) {
      fidu.vel_u(a,k,j,i) *= ooW;
    }
  }
  
  g_dd.DeleteTensorPointwise(); 
}
