//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file m1_opacity.cpp
//  \brief set the opacities

//TODO NeutrinoDensity() etc routines. The class FakeRates can be used for testing but below code needs to be slightly adapted

// C++ standard headers
#include <cassert>
#include <cmath> 

// Athena++ headers
#include "m1.hpp"
#include "../z4c/z4c.hpp"
#include "../mesh/mesh.hpp"
#include "../coordinates/coordinates.hpp"
#include "../hydro/hydro.hpp"
#include "fake_rates.hpp"

#define CGS_GCC (1.619100425158886e-18)

//int NeutrinoDensity(Real const, Real const, Real const, Real *, Real *, Real *, Real *, Real *, Real *); // TODO: define this
//int NeutrinoEmission(Real const, Real const, Real const, Real *, Real *, Real *, Real *, Real *, Real *); // TODO: define this
//int NeutrinoOpacity(Real const, Real const, Real const, Real *, Real *, Real *, Real *, Real *, Real *); // TODO: define this
//int NeutrinoAbsorptionRate(Real const, Real const, Real const, Real *, Real *, Real *, Real *, Real *, Real *); // TODO: define this
                                                                                                        
void M1::CalcOpacity(ParameterInput * pin, AthenaArray<Real> & u)
{
  MeshBlock * pmb = pmy_block;
  FakeRates * fr;

  fr = new FakeRates(pin, 1, 1);

  Rad_vars vec;
  SetRadVarsAliases(u, vec);
  
  // RadMat_vars rad;
  // SetRadVarsAliases(storage.u_rad, rad);
  // RadMat_vars rmat;
  // SetRadMatVarsAliases(storage.radmat, rmat);
  // Fidu_vars fidu;
  // SetFiduVarsAliases(storage.intern, fidu);

  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> vc_z4c_alpha;
  vc_z4c_alpha.InitWithShallowSlice(pmb->pz4c->storage.u, Z4c::I_Z4c_alpha);
  
  // Zero by default
  rmat.abs_0.ZeroClear();
  rmat.abs_1.ZeroClear();
  rmat.eta_0.ZeroClear();
  rmat.eta_1.ZeroClear();
  rmat.scat_1.ZeroClear();

  // Minimum grid spacing
  Real const dx = pmb->pcoord->dx1v(0);
  Real const dy = pmb->pcoord->dx2v(1);
  Real const dz = pmb->pcoord->dx3v(2);
  Real const delta = std::min(dx, std::min(dy, dz));

  // Go through cells
  CLOOP3(k,j,i) {
    if (rad.mask(k,j,i)) {
      // already set to zero
      continue;
    }
    
    // Current implementation is restricted.
    assert(nspecies == 3);
    assert(ngroups == 1);
    
    //TODO: fixme, need new eos/c2p
    Real const rho = pmb->phydro->w(IDN,k,j,i);
    Real const temperature = 0; //pmb->phydro->w(IDN,k,j,i);
    Real const Y_e = 0; //pmb->phydro->w(IDN,k,j,i);
    
    // Get the local thermodynamic equilibrium neutrino density and energy
    Real nudens_0[3], nudens_1[3];
    //int ierr = fr->NeutrinoDensity(rho, temperature, Y_e,
		//	                         &nudens_0[0], &nudens_0[1], &nudens_0[2],
		//	                         &nudens_1[0], &nudens_1[1], &nudens_1[2]);
    // TODO: check rates
    int ierr = fr->NeutrinoDensity(rho, temperature, Y_e, &nudens_0[0], &nudens_0[1]);

    assert(!ierr);
    assert(isfinite(nudens_0[0]));
    assert(isfinite(nudens_0[1]));
    assert(isfinite(nudens_0[2]));
    assert(isfinite(nudens_1[0]));
    assert(isfinite(nudens_1[1]));
    assert(isfinite(nudens_1[2]));

    // Get the local neutrino emissivities
    Real eta_0_loc[3], eta_1_loc[3];
    ierr = fr->NeutrinoEmission(rho, temperature, Y_e, &eta_0_loc[0]);
		//	    &eta_0_loc[0], &eta_0_loc[1], &eta_0_loc[2],
		//	    &eta_1_loc[0], &eta_1_loc[1], &eta_1_loc[2]);

    assert(!ierr);
    assert(isfinite(eta_0_loc[0]));
    assert(isfinite(eta_0_loc[1]));
    assert(isfinite(eta_0_loc[2]));
    assert(isfinite(eta_1_loc[0]));
    assert(isfinite(eta_1_loc[1]));
    assert(isfinite(eta_1_loc[2]));

    // Get the transport opacity (absorption + scattering)
    Real kappa_0_loc[3], kappa_1_loc[3];
    ierr = fr->NeutrinoOpacity(rho, temperature, Y_e, &kappa_0_loc[0]);
		//	   &kappa_0_loc[0], &kappa_0_loc[1], &kappa_0_loc[2],
		//	   &kappa_1_loc[0], &kappa_1_loc[1], &kappa_1_loc[2]);
    
    assert(!ierr);
    assert(isfinite(kappa_0_loc[0]));
    assert(isfinite(kappa_0_loc[1]));
    assert(isfinite(kappa_0_loc[2]));
    assert(isfinite(kappa_1_loc[0]));
    assert(isfinite(kappa_1_loc[1]));
    assert(isfinite(kappa_1_loc[2]));
    
    // Get the absorption opacity (absorption + scattering)
    Real abs_0_loc[3], abs_1_loc[3];
    ierr = fr->NeutrinoAbsorptionRate(rho, temperature, Y_e, &abs_0_loc[0]);
		//		  &abs_0_loc[0], &abs_0_loc[1], &abs_0_loc[2],
		//		  &abs_1_loc[0], &abs_1_loc[1], &abs_1_loc[2]);
    
    assert(!ierr);
    assert(isfinite(abs_0_loc[0]));
    assert(isfinite(abs_0_loc[1]));
    assert(isfinite(abs_0_loc[2]));
    assert(isfinite(abs_1_loc[0]));
    assert(isfinite(abs_1_loc[1]));
    assert(isfinite(abs_1_loc[2]));
    
    // Compute the cell optical depths
    Real tau[3] = {
      std::sqrt(abs_1_loc[0]*(kappa_1_loc[0]))*delta,
      std::sqrt(abs_1_loc[1]*(kappa_1_loc[1]))*delta,
      std::sqrt(abs_1_loc[2]*(kappa_1_loc[2]))*delta,
    };

    // Below we might need the lapse at (k,j,i) at CC:
    Real const alpha = pmb->pz4c->ig->map3d_VC2CC(vc_z4c_alpha(k,j,i));
    
    // Correct cross-sections for incoming neutrino energy
    for (int ig = 0; ig < ngroups*nspecies; ++ig) {

      // Extract scattering opacity
      rmat.scat_1(k,j,i,ig) = (kappa_1_loc[ig] - abs_1_loc[ig]);

      // Correct absorption opacities for non-LTE effects at low optical depth
      Real corr_fac = 1.0;
      if (tau[ig] < opacity_equil_depth) {
	      Real const Gamma = fidu.Wlorentz(k,j,i) - alpha*rad.Ht(k,j,i,ig)/rad.J(k,j,i,ig);	
	      corr_fac = (Gamma*rad.J(k,j,i,ig)/vec.nnu(k,j,i,ig)) * (nudens_0[ig]/nudens_1[ig]); // TODO: check rad and vec
	
	      if (!isfinite(corr_fac)) {
	        corr_fac = 1.0;
	      }
      }
      
      corr_fac *= corr_fac;
      corr_fac = std::max(1.0, std::min(corr_fac, opacity_corr_fac_max));
      
      // Set absorption opacities
      rmat.abs_0(k,j,i,ig) = corr_fac * abs_0_loc[ig];
      rmat.abs_1(k,j,i,ig) = corr_fac * abs_1_loc[ig];
      
      // Set emissivities and Enforce Kirchhoff's laws
      rmat.eta_0(k,j,i,ig) = nudens_0[ig] * rmat.abs_0(k,j,i,ig);
      rmat.eta_1(k,j,i,ig) = nudens_1[ig] * rmat.abs_1(k,j,i,ig);
    }
    
  }

}
