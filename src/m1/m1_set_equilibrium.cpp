//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file m1_set equilibrium.cpp
//  \brief set equilibrium state

// C++ standard headers
#include <cassert>
#include <algorithm>
#include <cmath> 

// Athena++ headers
#include "../mesh/mesh.hpp"
#include "../hydro/hydro.hpp"
#include "../z4c/z4c.hpp"
#include "m1.hpp"
#include "fake_rates.hpp"

using namespace utils;

//TODO NeutrinoDensity() etc routines. The class FakeRates can be used for testing but below code needs to be slightly adapted

void M1::SetEquilibrium(AthenaArray<Real> & u)
{
  MeshBlock * pmb = pmy_block;
  ParameterInput * pin = new ParameterInput;
  FakeRates * fr = new FakeRates(pin, 1, 1);
  Lab_vars vec;
  SetLabVarsAliases(u, vec);  
  //SetLabVarsAliases(storage.u, vec);

  // Rad_vars rad;
  // SetRadVarsAliases(storage.u_rad, rad);  
  // RadMat_vars rmat;
  // SetRadMatVarsAliases(storage.radmat, rmat);
  // Fidu_vars fidu;
  // SetFiduVarsAliases(storage.intern, fidu);
  
  // Pointwise 4D tensors used in the loop
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> g_dd;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> beta_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 0> alpha;  
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> g_uu;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> n_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> n_d;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 2> gamma_ud;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> u_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> u_d;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> F_d;
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> P_dd;
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> T_dd;
    
  g_dd.NewTensorPointwise();
  beta_u.NewTensorPointwise();
  alpha.NewTensorPointwise();
  g_uu.NewTensorPointwise();
  n_u.NewTensorPointwise();
  n_d.NewTensorPointwise();
  gamma_ud.NewTensorPointwise();
  u_u.NewTensorPointwise();
  u_d.NewTensorPointwise();
  F_d.NewTensorPointwise();
  P_dd.NewTensorPointwise();
  T_dd.NewTensorPointwise();

    
  // Go through cells
  CLOOP3(k,j,i) {

    //TODO: fixme, need new eos/c2p
    Real const rho = pmb->phydro->w(IDN,k,j,i);
    Real const temperature = 0; //pmb->phydro->w(IDN,k,j,i);
    Real const Y_e = 0; //pmb->phydro->w(IDN,k,j,i);

    // Skip this point if it is at low density
    if (rho < equilibrium_rho_min) {
      continue;
    }

    // Go from ADM 3-metric VC (AthenaArray/Tensor)
    // to ADM 4-metric on CC at ijk (TensorPointwise) 
    Get4Metric_VC2CCinterp(pmb, k,j,i,
			   pmb->pz4c->storage.u, pmb->pz4c->storage.adm,
			   g_dd, beta_u, alpha);    
    Get4Metric_Inv(g_dd, beta_u, alpha, g_uu);
    Get4Metric_Normal(alpha, beta_u, n_u);
    Get4Metric_NormalForm(alpha, n_d);
    Get4Metric_SpaceProj(n_u, n_d, gamma_ud);
    Real const detg = SpatialDet(g_dd(1,1), g_dd(1,2), g_dd(1,3), 
				 g_dd(2,2), g_dd(2,3), g_dd(3,3));
    Real const volform = std::sqrt(detg);
    
    // Fiducial vel.
    Real const Wlorentz = fidu.Wlorentz(k,j,i);
      uvel(alpha(), beta_u(1), beta_u(2), beta_u(3), Wlorentz,
	 fidu.vel_u(0,k,j,i), fidu.vel_u(1,k,j,i), fidu.vel_u(2,k,j,i), 
	 &u_u(0), &u_u(1), &u_u(2), &u_u(3));    
    tensor::contract(g_dd, u_u, u_d);

    // Current implementation is restricted.
    assert(nspecies == 3);
    assert(ngroups == 1);
    
    //
    // Compute the optically thick weak equilibrium
    Real nudens_0[3], nudens_1[3];
    int ierr = fr->NeutrinoDensity(rho, temperature, Y_e, &nudens_0[0], &nudens_0[1]);
//			       &nudens_1[0], &nudens_1[1], &nudens_1[2]);
    assert(!ierr);

    for (int ig = 0; ig < nspecies*ngroups; ++ig) {
    
      //
      // Set to equilibrium in the fluid frame
      rad.Ht(  k,j,i, ig) = 0.0;
      rad.H(0, k,j,i, ig) = 0.0;
      rad.H(1, k,j,i, ig) = 0.0;
      rad.H(2, k,j,i, ig) = 0.0;
      rad.chi( k,j,i, ig) = 1.0/3.0;

      Real const J = nudens_1[ig]*volform;
      rad.J(k,j,i, ig) = J;

      
      //
      // Compute quantities in the lab frame
      for (int a = 0; a < MDIM; ++a)
	for (int b = 0; b < 4; ++b) {
	  T_dd(a,b) = (4./3.) * J * u_d(a) * u_d(b) +
	    (1./3.) * J * g_dd(a,b);
	}

      vec.E(k,j,i,ig) = calc_J_from_rT(T_dd, n_u);
      calc_H_from_rT(T_dd, n_u, gamma_ud, F_d);
      apply_floor(g_uu, &vec.E(k,j,i,ig), F_d);

      unpack_F_d(F_d,
		 &vec.F_d(0, k,j,i,ig),
		 &vec.F_d(1, k,j,i,ig),
		 &vec.F_d(2, k,j,i,ig));
      
      calc_K_from_rT(T_dd, gamma_ud, P_dd);
      unpack_P_dd(P_dd,
		  &rad.P_dd(0,0, k,j,i,ig),&rad.P_dd(0,1, k,j,i,ig),&rad.P_dd(0,2, k,j,i,ig),
		  &rad.P_dd(1,1, k,j,i,ig),&rad.P_dd(1,2, k,j,i,ig),&rad.P_dd(2,2, k,j,i,ig));
      
      //
      // Now compute neutrino number density
      rad.nnu(k,j,i,ig) = nudens_0[ig]*volform;
      vec.N(k,j,i,ig) = std::max(Wlorentz * rad.nnu(k,j,i,ig), rad_N_floor);

    } // ig looop
    
  } // CLOOP3

  g_dd.DeleteTensorPointwise();
  beta_u.DeleteTensorPointwise();
  alpha.DeleteTensorPointwise();
  g_uu.DeleteTensorPointwise();
  n_u.DeleteTensorPointwise();
  gamma_ud.DeleteTensorPointwise();
  u_u.DeleteTensorPointwise();
  u_d.DeleteTensorPointwise();
  F_d.DeleteTensorPointwise();
  P_dd.DeleteTensorPointwise();
  T_dd.DeleteTensorPointwise();
  
}
