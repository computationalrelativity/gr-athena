//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file m1_opacity.cpp
//  \brief set the neutrino opacities

// C++ standard headers
#include <cassert>
#include <cmath> 
#include <sstream>

// Athena++ headers
#include "m1.hpp"
#include "../z4c/z4c.hpp"
#include "../mesh/mesh.hpp"
#include "../coordinates/coordinates.hpp"
#include "../hydro/hydro.hpp"

#define CGS_GCC (1.619100425158886e-18)

//----------------------------------------------------------------------------------------
// Opacity function scheduled in the tasklist

void M1::CalcOpacity(Real const dt, AthenaArray<Real> & u)
{
  if (M1_CALCOPACITY_OFF) return;
  M1_DEBUG_PR("in: CalcOpacity");
  
  // Zero by default
  rmat.abs_0.ZeroClear();
  rmat.abs_1.ZeroClear();
  rmat.eta_0.ZeroClear();
  rmat.eta_1.ZeroClear();
  rmat.scat_1.ZeroClear();

  // Note each type has its own routines 
  if (opacities == "zero") {
    return;
  } else if (opacities == "fake") {
    CalcOpacityFake(dt,u);      
  } else if (opacities == "neutrino") {
    CalcOpacityNeutrinos(dt,u);
  } else if (opacities == "photon") {
    CalcOpacityPhotons(dt,u);
  } else {
    std::ostringstream msg;
    msg << "Unknown opacities " << opacities << std::endl;
    ATHENA_ERROR(msg);
  }
  
}

//----------------------------------------------------------------------------------------
// Fake opacities for testing

void M1::CalcOpacityFake(Real const dt, AthenaArray<Real> & u)
{
  Real *eta_0 = new Real[ngroups*nspecies];
  Real *eta_1 = new Real[ngroups*nspecies];
  Real *abs_0 = new Real[ngroups*nspecies];
  Real *abs_1 = new Real[ngroups*nspecies];
  Real *sca_1 = new Real[ngroups*nspecies];
  int ierr[5];
  
  GCLOOP3(k,j,i) {
    if (m1_mask(k,j,i)) {
      continue;
    }

    //
    // Matter variables (dummy)
    Real const rho = 1.0; 
    Real const temperature = 0; 
    Real const Ye = 0; 

    //
    // Set opacities (for all species & groups)
    ierr[0] = fake_opac->Emission(rho,temperature,Ye, eta_0);
    ierr[1] = fake_opac->Emission(rho,temperature,Ye, eta_1);
    ierr[2] = fake_opac->Absorption_abs(rho,temperature,Ye, abs_0);
    ierr[3] = fake_opac->Absorption_abs(rho,temperature,Ye, abs_1);
    ierr[4] = fake_opac->Absorption_sca(rho,temperature,Ye, sca_1);
    for (int r=0; r<5; ++r)
      assert(!ierr[r]);
    
    //
    // Store opacities
    for (int ig = 0; ig < ngroups*nspecies; ++ig) {
      rmat.eta_0 (ig,k,j,i) = eta_0 [ig];
      rmat.eta_1 (ig,k,j,i) = eta_1 [ig];
      rmat.abs_0 (ig,k,j,i) = abs_0 [ig];
      rmat.abs_1 (ig,k,j,i) = abs_1 [ig];
      rmat.scat_1(ig,k,j,i) = sca_1 [ig];
    }
  }

  delete[] eta_0;
  delete[] eta_1;
  delete[] abs_0;
  delete[] abs_1;
  delete[] sca_1;
  
}

//----------------------------------------------------------------------------------------
// Neutrino opacities

void M1::CalcOpacityNeutrinos(Real const dt, AthenaArray<Real> & u)
{
  // Current implementation is restricted.
  assert(ngroups == 1);
  assert(nspecies > 1);

  MeshBlock * pmb = pmy_block;

  Rad_vars vec;
  SetRadVarsAliases(u, vec);
  
  // Minimum grid spacing
  Real const dx = pmb->pcoord->dx1v(0);
  Real const dy = pmb->pcoord->dx2v(1);
  Real const dz = pmb->pcoord->dx3v(2);
  Real const delta = std::min(dx, std::min(dy, dz));
  
  // For 4 species neutrino transport we need to divide the nux luminosity by 2
  Real const nux_weight = (nspecies == 3 ? 1.0 : 0.5);

  // Temp buffers for neutrino quantities
  // (hardcoded for nspecies<=4)
  //TODO: allocate memory for 'nspecies'
  Real nudens_0[4], nudens_1[4], chi_loc[4];
  Real eta_0_loc[4], eta_1_loc[4];
  Real abs_0_loc[4], abs_1_loc[4];
  Real scat_0_loc[4], scat_1_loc[4];
  Real nudens_0_trap[4], nudens_1_trap[4];
  Real nudens_0_thin[4], nudens_1_thin[4];
  
  // Pointwise 4D tensors used in the loop
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> g_dd;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> beta_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 0> alpha;  
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> g_uu;

  g_dd.NewTensorPointwise();
  beta_u.NewTensorPointwise();
  alpha.NewTensorPointwise();
  g_uu.NewTensorPointwise();

  //
  // Go through cells
  GCLOOP3(k,j,i) {
    if (m1_mask(k,j,i)) {
      continue;
    }
    
    //
    // Go from ADM 3-metric VC (AthenaArray/Tensor)
    // to ADM 4-metric on CC at ijk (TensorPointwise) 
    Get4Metric_VC2CCinterp(pmb, k,j,i,
			   pmb->pz4c->storage.u, pmb->pz4c->storage.adm,
			   g_dd, beta_u, alpha);
    Get4Metric_Inv(g_dd, beta_u, alpha, g_uu);
    Real const detg = SpatialDet(g_dd(1,1), g_dd(1,2), g_dd(1,3), 
				 g_dd(2,2), g_dd(2,3), g_dd(3,3));
    Real const oovolform = 1.0/(std::sqrt(detg));
    
    //
    // Matter variables FIXME: call to finite-T EOS
    Real const rho = 0.0; //pmb->phydro->w(IDN,k,j,i);
    Real const temperature = 0; //pmb->phydro->w(IDN,k,j,i);
    Real const Y_e = 0; //pmb->phydro->w(IDN,k,j,i);
    
    // Local neutrino quantities (undesitized)
    for (int is = 0; is < nspecies; ++is) {
      nudens_0[is] = rad.nnu(is,k,j,i) * oovolform;
      nudens_1[is] = rad.J(is,k,j,i) * oovolform;
      chi_loc[is] = rad.chi(is,k,j,i);
    }
    if (nspecies == 4) {
      nudens_0[2] += rad.nnu(3, i, j, k) * oovolform;
      nudens_1[2] += rad.J(3, i, j, k) * oovolform;
    }
    
    // Get emissivities and opacities
    int ierr = NeutrinoRates(
			     rho, temperature, Y_e,
			     nudens_0[0], nudens_1[0], chi_loc[0],
			     nudens_0[1], nudens_1[1], chi_loc[1],
			     nudens_0[2], nudens_1[2], chi_loc[2],
			     &eta_0_loc[0], &eta_0_loc[1], &eta_0_loc[2],
			     &eta_1_loc[0], &eta_1_loc[1], &eta_1_loc[2],
			     &abs_0_loc[0], &abs_0_loc[1], &abs_0_loc[2],
			     &abs_1_loc[0], &abs_1_loc[1], &abs_1_loc[2],
			     &scat_0_loc[0], &scat_0_loc[1], &scat_0_loc[2],
			     &scat_1_loc[0], &scat_1_loc[1], &scat_1_loc[2]);
    assert(!ierr);
    eta_0_loc[2] = nux_weight*eta_0_loc[2];
    eta_1_loc[2] = nux_weight*eta_1_loc[2];
    eta_0_loc[3] = eta_0_loc[2];
    eta_1_loc[3] = eta_1_loc[2];
    abs_0_loc[3] = abs_0_loc[2];
    abs_1_loc[3] = abs_1_loc[2];
    scat_0_loc[3] = scat_0_loc[2];
    scat_1_loc[3] = scat_1_loc[2];
    
    // An effective optical depth used to decide whether to compute
    // the black body function for neutrinos assuming neutrino trapping
    // or at a fixed temperature and Ye
    Real const tau = std::min(
			      sqrt(abs_1_loc[0]*(abs_1_loc[0] + scat_1_loc[0])),
			      sqrt(abs_1_loc[1]*(abs_1_loc[1] + scat_1_loc[1])))*dt;
    
    // Compute the neutrino black body functions assuming trapped neutrinos
    if (opacity_tau_trap >= 0 && tau > opacity_tau_trap) {
      Real temperature_trap, Y_e_trap;
      ierr = WeakEquilibrium(
			     rho, temperature, Y_e,
			     nudens_0[0], nudens_0[1], nudens_0[2],
			     nudens_1[0], nudens_1[1], nudens_1[2],
			     &temperature_trap, &Y_e_trap,
			     &nudens_0_trap[0], &nudens_0_trap[1], &nudens_0_trap[2],
			     &nudens_1_trap[0], &nudens_1_trap[1], &nudens_1_trap[2]);
      if (ierr) {
	// Try to recompute the weak equilibrium using neglecting
	// current neutrino data
	ierr = WeakEquilibrium(
			       rho, temperature, Y_e,
			       0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			       &temperature_trap, &Y_e_trap,
			       &nudens_0_trap[0], &nudens_0_trap[1], &nudens_0_trap[2],
			       &nudens_1_trap[0], &nudens_1_trap[1], &nudens_1_trap[2]);
	if (ierr) {
	  std::ostringstream msg;
	  msg << "Could not find the weak equilibrium!" << std::endl;
	  //msg << "Iteration = " << iteration << std::endl;
	  msg << "(i, j, k) = (" << i << ", " << j << ", " << k << ")\n";
	  msg << "(x, y, z) = (" << pmb->pcoord->x1v(i) << ", " << pmb->pcoord->x2v(j) << ", "
	      << pmb->pcoord->x3v(k) << ")\n";
	  msg << "rho = " << rho << std::endl;
	  msg << "temperature = " << temperature << std::endl;
	  msg << "Y_e = " << Y_e << std::endl;
	  msg << "alp = " << alpha() << std::endl;
	  msg << "nudens_0 = " << nudens_0[0] << " " << nudens_0[1]
	      << " " << nudens_0[2] << std::endl;
	  msg << "nudens_1 = " << nudens_1[0] << " " << nudens_1[1]
	      << " " << nudens_1[2] << std::endl;
	  ATHENA_ERROR(msg);
	}
      }
      assert(isfinite(nudens_0_trap[0]));
      assert(isfinite(nudens_0_trap[1]));
      assert(isfinite(nudens_0_trap[2]));
      assert(isfinite(nudens_1_trap[0]));
      assert(isfinite(nudens_1_trap[1]));
      assert(isfinite(nudens_1_trap[2]));
      nudens_0_trap[2] = nux_weight*nudens_0_trap[2];
      nudens_1_trap[2] = nux_weight*nudens_1_trap[2];
      nudens_0_trap[3] = nudens_0_trap[2];
      nudens_1_trap[3] = nudens_1_trap[2];
    }
    
    // Compute the neutrino black body function assuming fixed temperature and Y_e
    ierr = NeutrinoDensity(
			   rho, temperature, Y_e,
			   &nudens_0_thin[0], &nudens_0_thin[1], &nudens_0_thin[2],
			   &nudens_1_thin[0], &nudens_1_thin[1], &nudens_1_thin[2]);
    assert(!ierr);
    nudens_0_thin[2] = nux_weight*nudens_0_thin[2];
    nudens_1_thin[2] = nux_weight*nudens_1_thin[2];
    nudens_0_thin[3] = nudens_0_thin[2];
    nudens_1_thin[3] = nudens_1_thin[2];
    
    // Combine optically thin and optically thick limits
    for (int is = 0; is < nspecies; ++is) {
      
      // Set the neutrino black body function
      Real my_nudens_0, my_nudens_1;
      if (opacity_tau_trap < 0 || tau <= opacity_tau_trap) {
	my_nudens_0 = nudens_0_thin[is];
	my_nudens_1 = nudens_1_thin[is];
      }
      else if (tau > opacity_tau_trap + opacity_tau_delta) {
	my_nudens_0 = nudens_0_trap[is];
	my_nudens_1 = nudens_1_trap[is];
      }
      else {
	Real const lam = (tau - opacity_tau_trap)/opacity_tau_delta;
	my_nudens_0 = lam*nudens_0_trap[is] + (1 - lam)*nudens_0_thin[is];
	my_nudens_1 = lam*nudens_1_trap[is] + (1 - lam)*nudens_1_thin[is];
      }
      
      // Set the neutrino energies
      rmat.nueave(is,k,j,i) = my_nudens_1/my_nudens_0;
      
      // Store opacities
      rmat.scat_1(is,k,j,i) = scat_1_loc[is];
      rmat.abs_0(is,k,j,i) = abs_0_loc[is];
      rmat.abs_1(is,k,j,i) = abs_1_loc[is];
      
      // Enforce Kirchhoff's laws.
      rmat.eta_0(is,k,j,i) = rmat.abs_0(is,k,j,i)*my_nudens_0;
      rmat.eta_1(is,k,j,i) = rmat.abs_1(is,k,j,i)*my_nudens_1;
    }
         
  } // CLOOP
  
  g_dd.DeleteTensorPointwise();
  beta_u.DeleteTensorPointwise();
  alpha.DeleteTensorPointwise();
  g_uu.DeleteTensorPointwise();
  
}

//----------------------------------------------------------------------------------------
// Photon opacities

void M1::CalcOpacityPhotons(Real const dt, AthenaArray<Real> & u)
{
  // Current implementation is restricted.
  assert(ngroups == 1);
  assert(nspecies == 1);
  
  MeshBlock * pmb = pmy_block;

  Rad_vars vec;
  SetRadVarsAliases(u, vec);
  
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> vc_z4c_alpha;
  vc_z4c_alpha.InitWithShallowSlice(pmb->pz4c->storage.u, Z4c::I_Z4c_alpha);
    
  // Minimum grid spacing
  Real const dx = pmb->pcoord->dx1v(0);
  Real const dy = pmb->pcoord->dx2v(1);
  Real const dz = pmb->pcoord->dx3v(2);
  Real const delta = std::min(dx, std::min(dy, dz));

  // Pointwise 4D tensors used in the loop
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> g_dd;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> beta_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 0> alpha;  
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> g_uu;

  g_dd.NewTensorPointwise();
  beta_u.NewTensorPointwise();
  alpha.NewTensorPointwise();
  g_uu.NewTensorPointwise();

  //
  // Go through cells
  GCLOOP3(k,j,i) {
    if (m1_mask(k,j,i)) {
      continue;
    }
    
    //
    // Go from ADM 3-metric VC (AthenaArray/Tensor)
    // to ADM 4-metric on CC at ijk (TensorPointwise) 
    Get4Metric_VC2CCinterp(pmb, k,j,i,
			   pmb->pz4c->storage.u, pmb->pz4c->storage.adm,
			   g_dd, beta_u, alpha);
    Get4Metric_Inv(g_dd, beta_u, alpha, g_uu);
    Real const detg = SpatialDet(g_dd(1,1), g_dd(1,2), g_dd(1,3), 
				 g_dd(2,2), g_dd(2,3), g_dd(3,3));
    Real const oovolform = 1.0/(std::sqrt(detg));
    
    //
    // Matter variables FIXME: call to finite-T EOS
    Real const rho = 1.0; //pmb->phydro->w(IDN,k,j,i);
    Real const temperature = 0; //pmb->phydro->w(IDN,k,j,i);
    Real const Y_e = 0; //pmb->phydro->w(IDN,k,j,i);

    //
    // Set & store opacities
    int ierr = photon_opac->PhotonOpacity(
					  rho, temperature, Y_e,
					  &rmat.abs_1(0,k,j,i), &rmat.scat_1(0,k,j,i));
    assert(!ierr);
    rmat.eta_1(0,k,j,i) = rmat.abs_1(0,k,j,i) * photon_opac->PhotonBlackBody(temperature);
    
  } // CLOOP
  
  g_dd.DeleteTensorPointwise();
  beta_u.DeleteTensorPointwise();
  alpha.DeleteTensorPointwise();
  g_uu.DeleteTensorPointwise();
  
}
