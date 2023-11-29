//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file m1_fluxes.cpp
//  \brief calculate the numerical fluxes and update the r.h.s.

// C++ standard headers
#include <cassert>
#include <cmath>
#include <sstream>

// Athena++ headers
#include "m1.hpp"
#include "../z4c/z4c.hpp"
#include "../z4c/z4c_macro.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/tensor.hpp"

#define SQ(X) ((X)*(X))
#define TINY (1e-10)

// 1D index on scratch space with directional index i
#define GFINDEX1D(i, ig, iv)				\
  ((iv) + (ig)*N_Lab + (i)*(N_Lab * ngroups*nspecies))

using namespace utils;

namespace {
  Real minmod2(Real rl, Real rp, Real th) {
    return std::min(1.0, std::min(th*rl, th*rp));
  } 
}

#define M1_FLUXX_SET_ZERO (0)
#define M1_FLUXY_SET_ZERO (1)
#define M1_FLUXZ_SET_ZERO (1)

//----------------------------------------------------------------------------------------
// \fn void M1::AddFluxDivergence()
// \brief Add the flux divergence to the RHS (see analogous Hydro method)

void M1::AddFluxDivergence(AthenaArray<Real> & u_rhs) {
  //M1_DEBUG_PR("in: AddFluxDivergence");
  
  MeshBlock *pmb = pmy_block;
  AthenaArray<Real> &x1flux = storage.flux[X1DIR];
  AthenaArray<Real> &x2flux = storage.flux[X2DIR];
  AthenaArray<Real> &x3flux = storage.flux[X3DIR];
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  AthenaArray<Real> &x1area = x1face_area_, &x2area = x2face_area_,
    &x2area_p1 = x2face_area_p1_, &x3area = x3face_area_,
    &x3area_p1 = x3face_area_p1_, &vol = cell_volume_, &dflx = dflx_;
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      
      // calculate x1-flux divergence
      pmb->pcoord->Face1Area(k, j, is, ie+1, x1area); //SB(FIXME) for GR, this works in master_cx_matter, but not in matter_* branches!
      for (int iv=0; iv<N_Lab; ++iv) {
        for (int ig=0; ig<ngroups*nspecies; ++ig) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            //dflx(iv,ig,i) = (x1area(i+1)*x1flux(iv,ig,k,j,i+1) - x1area(i)*x1flux(iv,ig,k,j,i));
	    dflx(iv,ig,i) = (x1flux(iv,ig,k,j,i+1) - x1flux(iv,ig,k,j,i));
	  }
        }
      }
      
      // calculate x2-flux divergence
      if (pmb->block_size.nx2 > 1) {
        pmb->pcoord->Face2Area(k, j  , is, ie, x2area   );
        pmb->pcoord->Face2Area(k, j+1, is, ie, x2area_p1);
        for (int iv=0; iv<N_Lab; ++iv) {
          for (int ig=0; ig<ngroups*nspecies; ++ig) {
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
	      dflx(iv,ig,i) += (x2area_p1(i)*x2flux(iv,ig,k,j+1,i) - x2area(i)*x2flux(iv,ig,k,j,i));
            }
          }
        }
      }
      
      // calculate x3-flux divergence
      if (pmb->block_size.nx3 > 1) {
        pmb->pcoord->Face3Area(k  , j, is, ie, x3area   );
        pmb->pcoord->Face3Area(k+1, j, is, ie, x3area_p1);
        for (int iv=0; iv<N_Lab; ++iv) {
          for (int ig=0; ig<ngroups*nspecies; ++ig) {
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
              dflx(iv,ig,i) += (x3area_p1(i)*x3flux(iv,ig,k+1,j,i) - x3area(i)*x3flux(iv,ig,k,j,i));
            }
          }
        }
      }
      
      // update conserved variables
      pmb->pcoord->CellVolume(k, j, is, ie, vol);
      for (int iv=0; iv<N_Lab; ++iv) {
        for (int ig=0; ig<ngroups*nspecies; ++ig) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            u_rhs(iv,ig,k,j,i) -= dflx(iv,ig,i)/vol(i); //TODO CHECK THIS vol(i)
          }
        }
      }
      
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
// \!fn void M1::CalcFluxes(AthenaArray<Real> & u)
// \brief Compute the numerical fluxes using a simple 2nd order flux-limited method
//        with high-Peclet limit fix. Cf. Hydro::CalculateFluxes(...)

void M1::CalcFluxes(AthenaArray<Real> & u)
{
  M1_DEBUG_PR("in: CalcFluxes");
  
  MeshBlock * pmb = pmy_block;
  int const is = pmb->is; int const js = pmb->js; int const ks = pmb->ks;
  int const ie = pmb->ie; int const je = pmb->je; int const ke = pmb->ke;
  
  Lab_vars vec;
  SetLabVarsAliases(u, vec);  
  
  // Grid data
  Real const delta[3] = {
    pmb->pcoord->dx1v(0),
    pmb->pcoord->dx1v(1),
    pmb->pcoord->dx1v(0),
  };
  int const ncells[3] = {
    pmb->ncells1,
    pmb->ncells2,
    pmb->ncells3,
  };
  
  // Pointwise 4D tensors used in the loops (MDIM)
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> g_dd;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> beta_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 0> alpha;  
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> g_uu;    

  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> u_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> v_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> H_d;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> H_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> F_d;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> F_u;
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> P_dd;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 2> P_ud;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> fnu_u;

  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> gamma_uu; 
  
  g_dd.NewTensorPointwise();
  beta_u.NewTensorPointwise();
  alpha.NewTensorPointwise();
  g_uu.NewTensorPointwise();

  u_u.NewTensorPointwise();
  v_u.NewTensorPointwise();
  H_d.NewTensorPointwise();
  H_u.NewTensorPointwise();
  F_d.NewTensorPointwise();
  F_u.NewTensorPointwise();
  P_dd.NewTensorPointwise();
  P_ud.NewTensorPointwise();
  fnu_u.NewTensorPointwise();

  gamma_uu.NewTensorPointwise(); // NDIM !
  
  // Scratch space
  Real * cons;
  Real * flux;
  Real * cmax = nullptr;

  //--------------------------------------------------------------------------------------
  // i-direction
  if (!M1_FLUXX_SET_ZERO) M1_DEBUG_PR("in: CalcFluxes direction i");
  int dir = X1DIR; // NB several TensorPointwise are 4D!
  AthenaArray<Real> &x1flux = storage.flux[dir];
  
  try {
    cons = new Real[ N_Lab * ngroups*nspecies * ncells[dir] ];
    flux = new Real[ N_Lab * ngroups*nspecies * ncells[dir] ];
    cmax = new Real[ N_Lab * ngroups*nspecies * ncells[dir] ];
  } catch (std::bad_alloc &e) {
    std::stringstream msg;
    msg << "Out of memory!" << std::endl;
    ATHENA_ERROR(msg);
  }
  
  // set the loop limits 
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      
      // ----------------------------------------------
      // 1st pass compute the fluxes
      for (int i = 0; i < ncells[dir]; ++i) {
	int Id = i; // directional index for scratch buffers
	
	// From ADM 3-metric VC (AthenaArray/Tensor) to 
	// ADM 4-metric on CC at ijk (TensorPointwise) 
	Get4Metric_VC2CCinterp(pmb, k,j,i,				      
			       pmb->pz4c->storage.u, pmb->pz4c->storage.adm,  
			       g_dd, beta_u, alpha);			      
	Get4Metric_Inv_Inv3(g_dd, beta_u, alpha, g_uu, gamma_uu);	      
	uvel(alpha(), beta_u(1), beta_u(2), beta_u(3), fidu.Wlorentz(k,j,i),  
	     fidu.vel_u(0,k,j,i), fidu.vel_u(1,k,j,i), fidu.vel_u(2,k,j,i),   
	     &u_u(0), &u_u(1), &u_u(2), &u_u(3));				 
	pack_v_u(fidu.vel_u(0,k,j,i), fidu.vel_u(1,k,j,i), fidu.vel_u(2,k,j,i),  v_u);  
	
	for (int ig = 0; ig < ngroups*nspecies; ++ig) {			 
	  
	  pack_F_d(beta_u(1), beta_u(2), beta_u(3),			 
		   vec.F_d(0,ig,k,j,i),					 
		   vec.F_d(1,ig,k,j,i),					 
		   vec.F_d(2,ig,k,j,i),					 
		   F_d);
	  pack_H_d(rad.Ht(ig,k,j,i),					 
		   rad.H(0,ig,k,j,i), rad.H(1,ig,k,j,i), rad.H(2,ig,k,j,i),  
		   H_d);						 
	  pack_P_dd(beta_u(1), beta_u(2), beta_u(3),			 
		    rad.P_dd(0,0,ig,k,j,i), rad.P_dd(0,1,ig,k,j,i), rad.P_dd(1,1,ig,k,j,i),  
		    rad.P_dd(1,1,ig,k,j,i), rad.P_dd(1,2,ig,k,j,i), rad.P_dd(2,2,ig,k,j,i),  
		    P_dd);						 
	  tensor::contract(g_uu, H_d, H_u);				 
	  tensor::contract(g_uu, F_d, F_u);				 
	  tensor::contract(g_uu, P_dd, P_ud);				 
	  assemble_fnu(u_u, rad.J(ig,k,j,i), H_u, fnu_u);		 
	  Real const Gamma = compute_Gamma(fidu.Wlorentz(k,j,i), v_u,	 
					   rad.J(ig,k,j,i), vec.E(ig,k,j,i), F_d,  
					   rad_E_floor, rad_eps);	 
	  
	  Real nnu;							 
	  (void)nnu;							 
	  if (nspecies > 1)						 
	    nnu = vec.N(ig,k,j,i)/Gamma;				 
	  
	  // Scratch buffers in direction Id
	  cons[GFINDEX1D(Id, ig, I_Lab_Fx)] = vec.F_d(0,ig,k,j,i);		 
	  cons[GFINDEX1D(Id, ig, I_Lab_Fy)] = vec.F_d(1,ig,k,j,i);		 
	  cons[GFINDEX1D(Id, ig, I_Lab_Fz)] = vec.F_d(2,ig,k,j,i);		 
	  cons[GFINDEX1D(Id, ig, I_Lab_E)] = vec.E(ig,k,j,i);			 
	  if (nspecies > 1)						 
	    cons[GFINDEX1D(Id, ig, I_Lab_N)] = vec.N(ig,k,j,i);		 
	  
	  assert(isfinite(cons[GFINDEX1D(Id, ig, I_Lab_Fx)]));			 
	  assert(isfinite(cons[GFINDEX1D(Id, ig, I_Lab_Fy)]));			 
	  assert(isfinite(cons[GFINDEX1D(Id, ig, I_Lab_Fz)]));			 
	  assert(isfinite(cons[GFINDEX1D(Id, ig, I_Lab_E)]));			 
	  if (nspecies > 1)						 
	    assert(isfinite(cons[GFINDEX1D(Id, ig, I_Lab_N)]));		 
	  
	  flux[GFINDEX1D(Id, ig, I_Lab_Fx)] =					 
	    calc_F_flux(alpha(), beta_u, F_d, P_ud, dir+1, 1);		 
	  flux[GFINDEX1D(Id, ig, I_Lab_Fy)] =					 
	    calc_F_flux(alpha(), beta_u, F_d, P_ud, dir+1, 2);		 
	  flux[GFINDEX1D(Id, ig, I_Lab_Fz)] =					 
	    calc_F_flux(alpha(), beta_u, F_d, P_ud, dir+1, 3);		 
	  flux[GFINDEX1D(Id, ig, I_Lab_E)] =					 
	    calc_E_flux(alpha(), beta_u, vec.E(ig,k,j,i), F_u, dir+1);	 
	  if (nspecies > 1)						 
	    flux[GFINDEX1D(Id, ig, I_Lab_N)] =					 
	      alpha() * nnu * fnu_u(dir+1);				 
	  
	  assert(isfinite(flux[GFINDEX1D(Id, ig, I_Lab_Fx)]));			 
	  assert(isfinite(flux[GFINDEX1D(Id, ig, I_Lab_Fy)]));			 
	  assert(isfinite(flux[GFINDEX1D(Id, ig, I_Lab_Fz)]));			 
	  assert(isfinite(flux[GFINDEX1D(Id, ig, I_Lab_E)]));			 
	  if (nspecies > 1)						 
	    assert(isfinite(flux[GFINDEX1D(Id, ig, I_Lab_N)]));		 
	  
	  // Eigenvalues in the optically thin limit
	  //
	  // Using the optically thin eigenvalues seems to cause
	  // problems in some situations, possibly because the
	  // optically thin closure is acausal in certain
	  // conditions, so this is commented for now.
#if (M1_USE_EIGENVALUES_THIN)
	  Real const F2 = tensor::dot(F_u, F_d);
	  Real const F = std::sqrt(F2);
	  Real const fx = F_u(dir+1)*(F > 0 ? 1/F : 0);
	  Real const ffx = F_u(dir+1)*(F2 > 0 ? 1/F2 : 0);
	  Real const lam[3] = {
	    alpha()*fx - beta_u(dir+1),
	    - alpha()*fx - beta_u(dir+1),
	    alpha()*vec.E(ig,k,j,i)*ffx - beta_u(dir+1),
	  };
	  Real const cM1 = std::max(std::abs(lam[0]),
				    std::max(std::abs(lam[1]), std::abs(lam[2])));
	  //TODO optically thick characteristic speeds and combination
#endif
	  // Speed of light -- note that gamma_uu has NDIM=3 but beta has MDIM
	  Real const clam[2] = {
	    alpha()*std::sqrt(gamma_uu(dir,dir)) - beta_u(dir+1),
	    - alpha()*std::sqrt(gamma_uu(dir,dir)) - beta_u(dir+1),
	  };
	  Real const clight = std::max(std::abs(clam[0]), std::abs(clam[1]));
          
	  // In some cases the eigenvalues in the thin limit
	  // overestimate the actual characteristic speed and can
	  // become larger than c
	  cmax[GFINDEX1D(Id, ig, 0)] = clight;
	  // = std::min(clight, cM1);
	  
	}  // ig loop 
      } // i loop
            
      // ----------------------------------------------
      // 2nd pass store the num fluxes
      for (int i = is-1; i <= ie; ++i) {
      	int Id = i;
	
	for (int ig = 0; ig < ngroups*nspecies; ++ig) {		
	  
	  Real avg_abs_1 = rmat.abs_1(ig,k,j,i);			
	  Real avg_scat_1 = rmat.scat_1(ig,k,j,i);			
	  avg_abs_1 += rmat.abs_1(ig,k,j,i+1);				
	  avg_scat_1 += rmat.scat_1(ig,k,j,i+1);
	  
	  // Remove dissipation at high Peclet numbers 
	  Real kapa = 0.5*(avg_abs_1 + avg_scat_1); 
	  Real A = 1.0;
	  if (kapa*delta[dir] > 1.0) {
	    A = std::min(1.0, 1.0/(delta[dir]*kapa));
	    A = std::max(A, mindiss);
	  }
	  
	  for (int iv = 0; iv < N_Lab; ++iv) { 
	    Real const ujm = cons[GFINDEX1D(Id-1, ig, iv)]; 
	    Real const uj = cons[GFINDEX1D(Id, ig, iv)]; 
	    Real const ujp = cons[GFINDEX1D(Id+1, ig, iv)]; 
	    Real const ujpp = cons[GFINDEX1D(Id+2, ig, iv)]; 
	    
	    Real const fj = flux[GFINDEX1D(Id, ig, iv)]; 
	    Real const fjp = flux[GFINDEX1D(Id+1, ig, iv)]; 
	    
	    Real const cc = cmax[GFINDEX1D(Id, ig, 0)]; 
	    Real const ccp = cmax[GFINDEX1D(Id+1, ig, 0)]; 
	    Real const cmx = std::max(cc, ccp); 
	    
	    Real const dup = ujpp - ujp; 
	    Real const duc = ujp - uj; 
	    Real const dum = uj - ujm;
	    
	    bool sawtooth = false; 
	    Real phi = 0; 
	    if (dup*duc > 0 && dum*duc > 0) { 
	      phi = minmod2(dum/duc, dup/duc, minmod_theta); 
	    } else if (dup*duc < 0 && dum*duc < 0) { 
	      sawtooth = true; 
	    } 
	    assert(isfinite(phi)); 
	    
	    Real const flux_low = 0.5*(fj + fjp - cmx*(ujp - uj));
	    Real const flux_high = 0.5*(fj + fjp);
	    
	    Real flux_num = flux_high
	      - (sawtooth ? 1.0 : A)*(1.0 - phi)*(flux_high - flux_low); 
#if M1_FLUXX_SET_ZERO
	    flux_num = 0.0;
#endif
	    x1flux(iv,ig,k,j,i+1) = flux_num; // Note THC (Athena++) stores F_{i+1/2} (F_{i-1/2})!

	    
	    char sbuf[128]; sprintf(sbuf," iv = %d  (k,j,i+1) = (%d,%d,%d)  flux_x= %e", iv, k,j,i+1, flux_num); M1_DEBUG_PR(sbuf); 

	    
	  } // iv loop 
	} // ig loop
	
      }//i loop
      
    } // j loop
  } // k loop
  
  delete[] cons;
  delete[] flux;
  delete[] cmax;



  

  //TODO FIX The scratch arrays order to the Lab variabled indexes!!!


  
  
  //--------------------------------------------------------------------------------------
  // j-direction

  if (pmb->pmy_mesh->f2) {

    if (!M1_FLUXY_SET_ZERO) M1_DEBUG_PR("in: CalcFluxes direction j");
    
    int dir = X2DIR;
    AthenaArray<Real> &x2flux = storage.flux[dir];
    
    try {
      cons = new Real[ N_Lab * ngroups*nspecies * ncells[dir] ];
      flux = new Real[ N_Lab * ngroups*nspecies * ncells[dir] ];
      cmax = new Real[ N_Lab * ngroups*nspecies * ncells[dir] ];
    } catch (std::bad_alloc &e) {
      std::stringstream msg;
      msg << "Out of memory!" << std::endl;
      ATHENA_ERROR(msg);
    }
    
    // set the loop limits
    for (int i=is; i<=ie; ++i) {
      for (int k=ks; k<=ke; ++k) {
	
	// ----------------------------------------------
	// 1st pass compute the fluxes
	for (int j = 0; j < ncells[dir]; ++j) {
	  int Id = j; // directional index for scratch buffers
	  
	  // From ADM 3-metric VC (AthenaArray/Tensor) to 
	  // ADM 4-metric on CC at ijk (TensorPointwise) 
	  Get4Metric_VC2CCinterp(pmb, k,j,i,				      
				 pmb->pz4c->storage.u, pmb->pz4c->storage.adm,  
				 g_dd, beta_u, alpha);			      
	  Get4Metric_Inv_Inv3(g_dd, beta_u, alpha, g_uu, gamma_uu);	      
	  uvel(alpha(), beta_u(1), beta_u(2), beta_u(3), fidu.Wlorentz(k,j,i),  
	       fidu.vel_u(0,k,j,i), fidu.vel_u(1,k,j,i), fidu.vel_u(2,k,j,i),   
	       &u_u(0), &u_u(1), &u_u(2), &u_u(3));				 
	  pack_v_u(fidu.vel_u(0,k,j,i), fidu.vel_u(1,k,j,i), fidu.vel_u(2,k,j,i),  v_u);  
	  
	  for (int ig = 0; ig < ngroups*nspecies; ++ig) {			 
	    
	    pack_F_d(beta_u(1), beta_u(2), beta_u(3),			 
		     vec.F_d(0,ig,k,j,i),					 
		     vec.F_d(1,ig,k,j,i),					 
		     vec.F_d(2,ig,k,j,i),					 
		     F_d);						 
	    pack_H_d(rad.Ht(ig,k,j,i),					 
		     rad.H(0,ig,k,j,i), rad.H(1,ig,k,j,i), rad.H(2,ig,k,j,i),  
		     H_d);						 
	    pack_P_dd(beta_u(1), beta_u(2), beta_u(3),			 
		      rad.P_dd(0,0,ig,k,j,i), rad.P_dd(0,1,ig,k,j,i), rad.P_dd(1,1,ig,k,j,i),  
		      rad.P_dd(1,1,ig,k,j,i), rad.P_dd(1,2,ig,k,j,i), rad.P_dd(2,2,ig,k,j,i),  
		      P_dd);						 
	    tensor::contract(g_uu, H_d, H_u);				 
	    tensor::contract(g_uu, F_d, F_u);				 
	    tensor::contract(g_uu, P_dd, P_ud);				 
	    assemble_fnu(u_u, rad.J(ig,k,j,i), H_u, fnu_u);		 
	    Real const Gamma = compute_Gamma(fidu.Wlorentz(k,j,i), v_u,	 
					     rad.J(ig,k,j,i), vec.E(ig,k,j,i), F_d,  
					     rad_E_floor, rad_eps);	 
	    
	    Real nnu;							 
	    (void)nnu;							 
	    if (nspecies > 1)						 
	      nnu = vec.N(ig,k,j,i)/Gamma;				 
	    
	    // Scratch buffers in direction Id
	    cons[GFINDEX1D(Id, ig, 0)] = vec.F_d(0,ig,k,j,i);		 
	    cons[GFINDEX1D(Id, ig, 1)] = vec.F_d(1,ig,k,j,i);		 
	    cons[GFINDEX1D(Id, ig, 2)] = vec.F_d(2,ig,k,j,i);		 
	    cons[GFINDEX1D(Id, ig, 3)] = vec.E(ig,k,j,i);			 
	    if (nspecies > 1)						 
	      cons[GFINDEX1D(Id, ig, 4)] = vec.N(ig,k,j,i);		 
	    
	    assert(isfinite(cons[GFINDEX1D(Id, ig, 0)]));			 
	    assert(isfinite(cons[GFINDEX1D(Id, ig, 1)]));			 
	    assert(isfinite(cons[GFINDEX1D(Id, ig, 2)]));			 
	    assert(isfinite(cons[GFINDEX1D(Id, ig, 3)]));			 
	    if (nspecies > 1)						 
	      assert(isfinite(cons[GFINDEX1D(Id, ig, 4)]));		 
	    
	    flux[GFINDEX1D(Id, ig, 0)] =					 
	      calc_F_flux(alpha(), beta_u, F_d, P_ud, dir+1, 1);		 
	    flux[GFINDEX1D(Id, ig, 1)] =					 
	      calc_F_flux(alpha(), beta_u, F_d, P_ud, dir+1, 2);		 
	    flux[GFINDEX1D(Id, ig, 2)] =					 
	      calc_F_flux(alpha(), beta_u, F_d, P_ud, dir+1, 3);		 
	    flux[GFINDEX1D(Id, ig, 3)] =					 
	      calc_E_flux(alpha(), beta_u, vec.E(ig,k,j,i), F_u, dir+1);	 
	    if (nspecies > 1)						 
	      flux[GFINDEX1D(Id, ig, 4)] =					 
		alpha() * nnu * fnu_u(dir+1);				 
	    
	    assert(isfinite(flux[GFINDEX1D(Id, ig, 0)]));			 
	    assert(isfinite(flux[GFINDEX1D(Id, ig, 1)]));			 
	    assert(isfinite(flux[GFINDEX1D(Id, ig, 2)]));			 
	    assert(isfinite(flux[GFINDEX1D(Id, ig, 3)]));			 
	    if (nspecies > 1)						 
	      assert(isfinite(flux[GFINDEX1D(Id, ig, 4)]));		 
	    
	    // Eigenvalues in the optically thin limit
	    //
	    // Using the optically thin eigenvalues seems to cause
	    // problems in some situations, possibly because the
	    // optically thin closure is acausal in certain
	    // conditions, so this is commented for now.
#if (M1_USE_EIGENVALUES_THIN)
	    Real const F2 = tensor::dot(F_u, F_d);
	    Real const F = std::sqrt(F2);
	    Real const fx = F_u(dir+1)*(F > 0 ? 1/F : 0);
	    Real const ffx = F_u(dir+1)*(F2 > 0 ? 1/F2 : 0);
	    Real const lam[3] = {
	      alpha()*fx - beta_u(dir+1),
	      - alpha()*fx - beta_u(dir+1),
	      alpha()*vec.E(ig,k,j,i)*ffx - beta_u(dir+1),
	    };
	    Real const cM1 = std::max(std::abs(lam[0]),
				      std::max(std::abs(lam[1]), std::abs(lam[2])));
	    //TODO optically thick characteristic speeds and combination
#endif
	    // Speed of light -- note that gamma_uu has NDIM=3 but beta has MDIM
	    Real const clam[2] = {
	      alpha()*std::sqrt(gamma_uu(dir,dir)) - beta_u(dir+1),
	      - alpha()*std::sqrt(gamma_uu(dir,dir)) - beta_u(dir+1),
	    };
	    Real const clight = std::max(std::abs(clam[0]), std::abs(clam[1]));
	    
	    // In some cases the eigenvalues in the thin limit
	    // overestimate the actual characteristic speed and can
	    // become larger than c
	    cmax[GFINDEX1D(Id, ig, 0)] = clight;
	    // = std::min(clight, cM1);
	    
	  }  // ig loop 
	} // j loop
	
	// ----------------------------------------------
	// 2nd pass store the num fluxes
	for (int j = js-1; j <= je; ++j) { 
	  int Id = j;
	  
	  for (int ig = 0; ig < ngroups*nspecies; ++ig) {		
  								
	    Real avg_abs_1 = rmat.abs_1(ig,k,j,i);			
	    Real avg_scat_1 = rmat.scat_1(ig,k,j,i);			
	    avg_abs_1 += rmat.abs_1(ig,k,j+1,i);				
	    avg_scat_1 += rmat.scat_1(ig,k,j+1,i);
	    
	    // Remove dissipation at high Peclet numbers 
	    Real kapa = 0.5*(avg_abs_1 + avg_scat_1); 
	    Real A = 1.0;
	    if (kapa*delta[dir] > 1.0) {
	      A = std::min(1.0, 1.0/(delta[dir]*kapa));
	      A = std::max(A, mindiss);
	    }

	    for (int iv = 0; iv < N_Lab; ++iv) { 
	      Real const ujm = cons[GFINDEX1D(Id-1, ig, iv)]; 
	      Real const uj = cons[GFINDEX1D(Id, ig, iv)]; 
	      Real const ujp = cons[GFINDEX1D(Id+1, ig, iv)]; 
	      Real const ujpp = cons[GFINDEX1D(Id+2, ig, iv)]; 
	      
	      Real const fj = flux[GFINDEX1D(Id, ig, iv)]; 
	      Real const fjp = flux[GFINDEX1D(Id+1, ig, iv)]; 
	      
	      Real const cc = cmax[GFINDEX1D(Id, ig, 0)]; 
	      Real const ccp = cmax[GFINDEX1D(Id+1, ig, 0)]; 
	      Real const cmx = std::max(cc, ccp); 
	      
	      Real const dup = ujpp - ujp; 
	      Real const duc = ujp - uj; 
	      Real const dum = uj - ujm; 
	      
	      bool sawtooth = false; 
	      Real phi = 0; 
	      if (dup*duc > 0 && dum*duc > 0) { 
		phi = minmod2(dum/duc, dup/duc, minmod_theta); 
	      } else if (dup*duc < 0 && dum*duc < 0) { 
	      sawtooth = true; 
	      } 
	    assert(isfinite(phi)); 
	    
	    Real const flux_low = 0.5*(fj + fjp - cmx*(ujp - uj));
	    Real const flux_high = 0.5*(fj + fjp);
	    
	    Real flux_num = flux_high - (sawtooth ? 1.0 : A)*(1.0 - phi)*(flux_high - flux_low); 
#if M1_FLUXY_SET_ZERO
	    flux_num = 0.0;
#endif
	    x2flux(iv,ig,k,j+1,i) = flux_num; 
	    
	    } // iv loop 
	  } // ig loop
	  
	} // j loop
	
      } // k loop
    } // i loop

    delete[] cons;
    delete[] flux;
    delete[] cmax;

  }
  
  //--------------------------------------------------------------------------------------
  // k-direction
  
  if (pmb->pmy_mesh->f3) {

    if (!M1_FLUXZ_SET_ZERO) M1_DEBUG_PR("in: CalcFluxes direction k");
    
    int dir = X3DIR;
    AthenaArray<Real> &x3flux = storage.flux[dir];

    try {
      cons = new Real[ N_Lab * ngroups*nspecies * ncells[dir] ];
      flux = new Real[ N_Lab * ngroups*nspecies * ncells[dir] ];
      cmax = new Real[ N_Lab * ngroups*nspecies * ncells[dir] ];
    } catch (std::bad_alloc &e) {
      std::stringstream msg;
      msg << "Out of memory!" << std::endl;
      ATHENA_ERROR(msg);
    }
  
    // set the loop limits
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
	
	// ----------------------------------------------
	// 1st pass compute the fluxes
	for (int k = 0; k < ncells[dir]; ++k) {
	  int Id = k; // directional index for scratch buffers
	  
	  // From ADM 3-metric VC (AthenaArray/Tensor) to 
	  // ADM 4-metric on CC at ijk (TensorPointwise) 
	  Get4Metric_VC2CCinterp(pmb, k,j,i,				      
				 pmb->pz4c->storage.u, pmb->pz4c->storage.adm,  
				 g_dd, beta_u, alpha);			      
	  Get4Metric_Inv_Inv3(g_dd, beta_u, alpha, g_uu, gamma_uu);	      
	  uvel(alpha(), beta_u(1), beta_u(2), beta_u(3), fidu.Wlorentz(k,j,i),  
	       fidu.vel_u(0,k,j,i), fidu.vel_u(1,k,j,i), fidu.vel_u(2,k,j,i),   
	       &u_u(0), &u_u(1), &u_u(2), &u_u(3));				 
	  pack_v_u(fidu.vel_u(0,k,j,i), fidu.vel_u(1,k,j,i), fidu.vel_u(2,k,j,i),  v_u);  
	  
	  for (int ig = 0; ig < ngroups*nspecies; ++ig) {			 
	    
	    pack_F_d(beta_u(1), beta_u(2), beta_u(3),			 
		     vec.F_d(0,ig,k,j,i),					 
		     vec.F_d(1,ig,k,j,i),					 
		     vec.F_d(2,ig,k,j,i),					 
		     F_d);						 
	    pack_H_d(rad.Ht(ig,k,j,i),					 
		     rad.H(0,ig,k,j,i), rad.H(1,ig,k,j,i), rad.H(2,ig,k,j,i),  
		     H_d);						 
	    pack_P_dd(beta_u(1), beta_u(2), beta_u(3),			 
		      rad.P_dd(0,0,ig,k,j,i), rad.P_dd(0,1,ig,k,j,i), rad.P_dd(1,1,ig,k,j,i),  
		      rad.P_dd(1,1,ig,k,j,i), rad.P_dd(1,2,ig,k,j,i), rad.P_dd(2,2,ig,k,j,i),  
		      P_dd);						 
	    tensor::contract(g_uu, H_d, H_u);				 
	    tensor::contract(g_uu, F_d, F_u);				 
	    tensor::contract(g_uu, P_dd, P_ud);				 
	    assemble_fnu(u_u, rad.J(ig,k,j,i), H_u, fnu_u);		 
	    Real const Gamma = compute_Gamma(fidu.Wlorentz(k,j,i), v_u,	 
					     rad.J(ig,k,j,i), vec.E(ig,k,j,i), F_d,  
					     rad_E_floor, rad_eps);	 
	    
	    Real nnu;							 
	    (void)nnu;							 
	    if (nspecies > 1)						 
	      nnu = vec.N(ig,k,j,i)/Gamma;				 
	    
	    // Scratch buffers in direction Id
	    cons[GFINDEX1D(Id, ig, 0)] = vec.F_d(0,ig,k,j,i);		 
	    cons[GFINDEX1D(Id, ig, 1)] = vec.F_d(1,ig,k,j,i);		 
	    cons[GFINDEX1D(Id, ig, 2)] = vec.F_d(2,ig,k,j,i);		 
	    cons[GFINDEX1D(Id, ig, 3)] = vec.E(ig,k,j,i);			 
	    if (nspecies > 1)						 
	      cons[GFINDEX1D(Id, ig, 4)] = vec.N(ig,k,j,i);		 
	    
	    assert(isfinite(cons[GFINDEX1D(Id, ig, 0)]));			 
	    assert(isfinite(cons[GFINDEX1D(Id, ig, 1)]));			 
	    assert(isfinite(cons[GFINDEX1D(Id, ig, 2)]));			 
	    assert(isfinite(cons[GFINDEX1D(Id, ig, 3)]));			 
	    if (nspecies > 1)						 
	      assert(isfinite(cons[GFINDEX1D(Id, ig, 4)]));		 
	    
	    flux[GFINDEX1D(Id, ig, 0)] =					 
	      calc_F_flux(alpha(), beta_u, F_d, P_ud, dir+1, 1);		 
	    flux[GFINDEX1D(Id, ig, 1)] =					 
	      calc_F_flux(alpha(), beta_u, F_d, P_ud, dir+1, 2);		 
	    flux[GFINDEX1D(Id, ig, 2)] =					 
	      calc_F_flux(alpha(), beta_u, F_d, P_ud, dir+1, 3);		 
	    flux[GFINDEX1D(Id, ig, 3)] =					 
	      calc_E_flux(alpha(), beta_u, vec.E(ig,k,j,i), F_u, dir+1);	 
	    if (nspecies > 1)						 
	      flux[GFINDEX1D(Id, ig, 4)] =					 
		alpha() * nnu * fnu_u(dir+1);				 
	    
	    assert(isfinite(flux[GFINDEX1D(Id, ig, 0)]));			 
	    assert(isfinite(flux[GFINDEX1D(Id, ig, 1)]));			 
	    assert(isfinite(flux[GFINDEX1D(Id, ig, 2)]));			 
	    assert(isfinite(flux[GFINDEX1D(Id, ig, 3)]));			 
	    if (nspecies > 1)						 
	      assert(isfinite(flux[GFINDEX1D(Id, ig, 4)]));		 
	    
	    // Eigenvalues in the optically thin limit
	    //
	    // Using the optically thin eigenvalues seems to cause
	    // problems in some situations, possibly because the
	    // optically thin closure is acausal in certain
	    // conditions, so this is commented for now.
#if (M1_USE_EIGENVALUES_THIN)
	    Real const F2 = tensor::dot(F_u, F_d);
	    Real const F = std::sqrt(F2);
	    Real const fx = F_u(dir+1)*(F > 0 ? 1/F : 0);
	    Real const ffx = F_u(dir+1)*(F2 > 0 ? 1/F2 : 0);
	    Real const lam[3] = {
	      alpha()*fx - beta_u(dir+1),
	      - alpha()*fx - beta_u(dir+1),
	      alpha()*vec.E(ig,k,j,i)*ffx - beta_u(dir+1),
	    };
	    Real const cM1 = std::max(std::abs(lam[0]),
				      std::max(std::abs(lam[1]), std::abs(lam[2])));
	    //TODO optically thick characteristic speeds and combination
#endif
	    // Speed of light -- note that gamma_uu has NDIM=3 but beta has MDIM
	    Real const clam[2] = {
	      alpha()*std::sqrt(gamma_uu(dir,dir)) - beta_u(dir+1),
	      - alpha()*std::sqrt(gamma_uu(dir,dir)) - beta_u(dir+1),
	    };
	    Real const clight = std::max(std::abs(clam[0]), std::abs(clam[1]));
          
	    // In some cases the eigenvalues in the thin limit
	    // overestimate the actual characteristic speed and can
	    // become larger than c
	    cmax[GFINDEX1D(Id, ig, 0)] = clight;
	    // = std::min(clight, cM1);
	  
	  } // ig loop
	  
	} // k loop

	// ----------------------------------------------
	// 2nd pass store the num fluxes
	for (int k = ks-1; k <= ke; ++k) { 
	  int Id = k;
	  
	  for (int ig = 0; ig < ngroups*nspecies; ++ig) {		
	    
	    Real avg_abs_1 = rmat.abs_1(ig,k,j,i);			
	    Real avg_scat_1 = rmat.scat_1(ig,k,j,i);			
	    avg_abs_1 += rmat.abs_1(ig,k+1,j,i);				
	    avg_scat_1 += rmat.scat_1(ig,k+1,j,i);
	    
	    // Remove dissipation at high Peclet numbers 
	    Real kapa = 0.5*(avg_abs_1 + avg_scat_1); 
	    Real A = 1.0;  
	    if (kapa*delta[dir] > 1.0) {			
	      A = std::max(1.0/(delta[dir]*kapa), mindiss);	
	    }							
	    
	    for (int iv = 0; iv < N_Lab; ++iv) { 
	      Real const ujm = cons[GFINDEX1D(Id-1, ig, iv)]; 
	      Real const uj = cons[GFINDEX1D(Id, ig, iv)]; 
	      Real const ujp = cons[GFINDEX1D(Id+1, ig, iv)]; 
	      Real const ujpp = cons[GFINDEX1D(Id+2, ig, iv)]; 
	    
	      Real const fj = flux[GFINDEX1D(Id, ig, iv)]; 
	      Real const fjp = flux[GFINDEX1D(Id+1, ig, iv)]; 
	    
	      Real const cc = cmax[GFINDEX1D(Id, ig, 0)]; 
	      Real const ccp = cmax[GFINDEX1D(Id+1, ig, 0)]; 
	      Real const cmx = std::max(cc, ccp); 
	    
	      Real const dup = ujpp - ujp;
	      Real const duc = ujp - uj; 
	      Real const dum = uj - ujm; 
	    
	      bool sawtooth = false; 
	      Real phi = 0; 
	      if (dup*duc > 0 && dum*duc > 0) { 
		phi = minmod2(dum/duc, dup/duc, minmod_theta); 
	      } else if (dup*duc < 0 && dum*duc < 0) { 
		sawtooth = true; 
	      } 
	      assert(isfinite(phi)); 
	      
	      Real const flux_low = 0.5*(fj + fjp - cmx*(ujp - uj));
	      Real const flux_high = 0.5*(fj + fjp);
	      
	      Real flux_num = flux_high - (sawtooth ? 1.0 : A)*(1.0 - phi)*(flux_high - flux_low); 
#if M1_FLUXZ_SET_ZERO
	      flux_num = 0.0;
#endif
	      x3flux(iv,ig,k+1,j,i) = flux_num; 
	      
	    } // iv loop 
	  } // ig loop
		    
	}// k loop
	
      } // i loop
    } // j loop 
    
   delete[] cons;
   delete[] flux;
   delete[] cmax;
      
  }

  //--------------------------------------------------------------------------------------

  g_dd.DeleteTensorPointwise();
  beta_u.DeleteTensorPointwise();
  alpha.DeleteTensorPointwise();
  g_uu.DeleteTensorPointwise();
  
  u_u.DeleteTensorPointwise();
  v_u.DeleteTensorPointwise();
  H_d.DeleteTensorPointwise();
  H_u.DeleteTensorPointwise();
  F_d.DeleteTensorPointwise();
  F_u.DeleteTensorPointwise();
  P_dd.DeleteTensorPointwise();
  P_ud.DeleteTensorPointwise();
  fnu_u.DeleteTensorPointwise();

  gamma_uu.DeleteTensorPointwise();

}

