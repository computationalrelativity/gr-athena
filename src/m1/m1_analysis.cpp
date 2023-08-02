//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file m1_analysis.cpp
//  \brief perform additional calculation on radiation fields

// C++ standard headers
#include <cmath> 

// Athena++ headers
#include "m1.hpp"


void M1::Analyse(AthenaArray<Real> const & u)
{  
  MeshBlock * pmb = pmy_block;

  Lab_vars vec;
  SetLabVarsAliases(u, vec);  
  
  Real const mb = AverageBaryonMass(); //TODO fix this somewhere
  Real * enu = new Real [ngroups*nspecies];

  // Pointwise 4D tensors used in the loop
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> g_dd;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> beta_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 0> alpha;  
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> g_uu;    
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> u_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> F_d;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> F_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> H_d;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> H_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> fnu_u;  
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> r_d; 

  g_dd.NewTensorPointwise();
  beta_u.NewTensorPointwise();
  alpha.NewTensorPointwise();
  g_uu.NewTensorPointwise();
  u_u.NewTensorPointwise();
  F_d.NewTensorPointwise();
  F_u.NewTensorPointwise();
  H_d.NewTensorPointwise();
  H_u.NewTensorPointwise();
  fnu_u.NewTensorPointwise();
  r_d.NewTensorPointwise();

  
  // Go through cells
  CLOOP3(k,j,i) {

    Real const x = pmb->pcoord->x1v(i);;
    Real const y = pmb->pcoord->x2v(j);;
    Real const z = pmb->pcoord->x3v(k);;

    
    //
    // Go from ADM 3-metric VC (AthenaArray/Tensor)
    // to ADM 4-metric on CC at ijk (TensorPointwise) 
    Get4Metric_VC2CCinterp(pmb, k,j,i,
			   pmb->pz4c->storage.u, pmb->pz4c->storage.adm,
			   &g_dd, &beta_u, &alpha);
    Get4Metric_Inv(g_dd, beta_u, alpha, &g_uu);
    Real const detg = SpatialDet(g_dd(1,1), g_dd(1,2), g_dd(1,3), 
				 g_dd(2,2), g_dd(2,3), g_dd(3,3));
    Real const oovolform = 1.0/(std::sqrt(detg));

    
    //
    // Fluid quantities
    Real const rho = pmb->phydro->w(IDN,k,j,i);
    Real const egas = pmb->phydro->w(IDE,k,j,i);//TODO fixme! what is stored?
    //Real const eps = pmb->phydro->w(IDE,k,j,i);
    //Real const egas = rho*(1.0 + eps);

    Real etot  = egas;
    Real const oonb = 1.0/(rho/mb);

    
    //
    // Neutrino fractions
    for (int ig = 0; ig < ngroups*nspecies; ++ig) {
      rdia.ynu(k,j,i,ig) = rad.nnu(k,j,i,ig) * oovolform * oonb;
      enu[ig] = rad.J(k,j,i,ig) * oovolform;
      etot += enu[ig];
    }

    
    //
    // Neutrino energies
    for (int ig = 0; ig < ngroups*nspecies; ++ig) {
      rdia.znu(k,j,i,ig) = enu[ig]/etot;
    }

    
    //
    // Radial fluxes
    r_d(0) = 0.0;
    r_d(1) = x;
    r_d(2) = y;
    r_d(3) = z;
    Real const rr = std::sqrt(tensor::dot(g_uu, r_d, r_d));
    Real const irr = 1.0/rr;
    uvel(alpha(), beta_u(1), beta_u(2), beta_u(3), fidu.Wlorentz(k,j,i),
	 fidu.vel_u(0), fidu.vel_u(1), fidu.vel_u(2), 
	 &u_u(0), &u_u(1), &u_u(2), &u_u(3));    

    for (int ig = 0; ig < ngroups*nspecies; ++ig) {

            pack_F_d(beta_u(1), beta_u(2), beta_u(3),
	       vec.F_d(0,k,j,i,ig),
	       vec.F_d(1,k,j,i,ig),
	       vec.F_d(2,k,j,i,ig),
	       &F_d);

	    pack_H_d(rad.Ht(k,j,i,ig),
		     rad.H(0,k,j,i,ig), rad.H(1,k,j,i,ig), rad.H(2,k,j,i,ig),
		     &H_d);
	    
	    tensor::contract(g_uu, F_d, &F_u);
	    tensor::contract(g_uu, H_d, &H_u);

	    Real sum = 0.0;
	    for (int a = 1; a < 4; ++a) {
	      sum += r_d(a) * irr *
		calc_E_flux(alpha(), beta_u, vec.E(k,j,i,ig), F_u, a);
	    }
	    rdia.radial_flux_1(k,j,i,ig) = sum;
	    
	    assemble_fnu(u_u, rad.J(k,j,i,ig), H_u, &fnu_u);
	    Real const Gamma = alpha() * fnu_u(0);
	    // Note that nnu is densitized here
	    Real const nnu = vec.N(k,j,i,ig)/Gamma;

	    rdia.radial_flux_0(k,j,i,ig) = alpha() * irr * nnu * tensor::dot(fnu_u, r_d);

    }
    
    
  } //CLOOP3

  delete[] enu;

  g_dd.DeleteTensorPointwise();
  beta_u.DeleteTensorPointwise();
  alpha.DeleteTensorPointwise();
  g_uu.DeleteTensorPointwise();
  u_u.DeleteTensorPointwise();
  F_d.DeleteTensorPointwise();
  F_u.DeleteTensorPointwise();
  H_d.DeleteTensorPointwise();
  H_u.DeleteTensorPointwise();
  fnu_u.DeleteTensorPointwise();
  r_d.DeleteTensorPointwise();

}
