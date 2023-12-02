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
  ((iv) + (ig)*nvars + (i)*(nvars * ngroups*nspecies))

#define PINDEX1D(ig, iv) \
    ((iv) + (ig)*nvars)

using namespace utils;

namespace {
  Real minmod2(Real rl, Real rp, Real th) {
    return std::min(1.0, std::min(th*rl, th*rp));
  } 
}

#define M1_FLUXX_SET_ZERO (0)
#define M1_FLUXY_SET_ZERO (1)
#define M1_FLUXZ_SET_ZERO (1)

#define test_thc_mode (0) // compile with 2 ghosts.

//----------------------------------------------------------------------------------------
// \fn void M1::AddFluxDivergence()
// \brief Add the flux divergence to the RHS (see analogous Hydro method)

void M1::AddFluxDivergence(AthenaArray<Real> & u_rhs) {
  M1_DEBUG_PR("in: AddFluxDivergence");
  
  MeshBlock *pmb = pmy_block;
  AthenaArray<Real> &x1flux = storage.flux[X1DIR];
  AthenaArray<Real> &x2flux = storage.flux[X2DIR];
  AthenaArray<Real> &x3flux = storage.flux[X3DIR];
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  AthenaArray<Real> &x1area = x1face_area_, &x2area = x2face_area_,
    &x2area_p1 = x2face_area_p1_, &x3area = x3face_area_,
    &x3area_p1 = x3face_area_p1_, &vol = cell_volume_, &dflx = dflx_;

#if (test_thc_mode)
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int iv=0; iv<N_Lab; ++iv) {
        for (int ig=0; ig<ngroups*nspecies; ++ig) {
          for (int i=is; i<=ie; ++i) {
            u_rhs(iv,ig,k,j,i) = x1flux(iv,ig,k,j,i);

	    //char sbuf[128]; sprintf(sbuf," k=%d j=%d i=%d   iv=%d  ig=%d   rhs = %e", k,j,i, iv, ig, u_rhs(iv,ig,k,j,i)); M1_DEBUG_PR(sbuf); 

	    
	  }
	}
      }
    }
  }
  return;
#endif
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {


      pmb->pcoord->CellVolume(k, j, is, ie, vol);
      
      // calculate x1-flux divergence
      pmb->pcoord->Face1Area(k, j, is, ie+1, x1area); //SB(FIXME) for GR, this will work in master_cx_matter, but not in matter_* branches!
      for (int iv=0; iv<N_Lab; ++iv) {
        for (int ig=0; ig<ngroups*nspecies; ++ig) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            dflx(iv,ig,i) = (x1area(i+1)*x1flux(iv,ig,k,j,i+1) - x1area(i)*x1flux(iv,ig,k,j,i));

	    char sbuf[128]; sprintf(sbuf,"div: k=%d j=%d i=%d   iv=%d  ig=%d   flux = %e  area=%e dflx=%e  RHS=%e", k,j,i, iv, ig, x1flux(iv,ig,k,j,i), x1area(i), dflx(iv,ig,i), -dflx(iv,ig,i)/vol(i)); M1_DEBUG_PR(sbuf); 
	    
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
            u_rhs(iv,ig,k,j,i) -= dflx(iv,ig,i)/vol(i); 


	    char sbuf[128]; sprintf(sbuf," k=%d j=%d i=%d   iv=%d  ig=%d   rhs = %e (dflx=%e  vol=%e)",
				    k,j,i, iv, ig, u_rhs(iv,ig,k,j,i), dflx(iv,ig,i), vol(i)); M1_DEBUG_PR(sbuf); 



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
    pmb->pcoord->dx2v(0),
    pmb->pcoord->dx3v(0),
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

  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> gamma_uu; // (NDIM)
  
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

  // For scratch errors
  int const nvars = (nspecies > 1 ? N_Lab : N_Lab-1);
  int mapiv [] = {
    I_Lab_Fx, I_Lab_Fy, I_Lab_Fz,
    I_Lab_E,
    I_Lab_N,
  };
  
  //--------------------------------------------------------------------------------------
  for (int dir = 0; dir < NDIM; ++dir) {

    if (dir==1 && !pmb->pmy_mesh->f2) continue;
    if (dir==2 && !pmb->pmy_mesh->f3) continue;

    AthenaArray<Real> &xdirflux = storage.flux[dir];
    
    int index[3];
    int pts[3];
    int beg[3];
    int end[3];
    int shift[3];

    Real flux_num[N_Lab];
    
    // We have to leave as the most internal loop the one on the
    // direction. For this reason we will remap the usual indices
    // i,j,k into different points of index[:].
    int ii, ij, ik;
    switch(dir) {
    case 0:
      ii = 2;
      ij = 1;
      ik = 0;
      
      pts[0] = ncells[2];
      pts[1] = ncells[1];
      pts[2] = ncells[0];

      beg[0] = ks;   // M1_NGHOST
      beg[1] = js;
      beg[2] = is-1; 
      
      end[0] = ke+1; // pts - M1_NGHOST
      end[1] = je+1;
      end[2] = ie+1;

      shift[0] = 1;
      shift[1] = 0;
      shift[2] = 0;
      
      break;
    case 1:
      ii = 1;
      ij = 2;
      ik = 0;
      
      pts[0] = ncells[2];
      pts[1] = ncells[0];
      pts[2] = ncells[1];

      beg[0] = ks;   
      beg[1] = is;
      beg[2] = js-1;

      end[0] = ke+1; 
      end[1] = ie+1;
      end[2] = je+1;

      shift[0] = 0;
      shift[1] = 1;
      shift[2] = 0;
      
      break;
    case 2:
      ii = 1;
      ij = 0;
      ik = 2;
      
      pts[0] = ncells[1];
      pts[1] = ncells[0];
      pts[2] = ncells[2];

      beg[0] = js;   
      beg[1] = is;
      beg[2] = ks-1;

      end[0] = je+1; 
      end[1] = ie+1;
      end[2] = ke+1;

      shift[0] = 0;
      shift[1] = 0;
      shift[2] = 1;
      
      break;
    }

#if (test_thc_mode)
    beg[0] = M1_NGHOST;
    beg[1] = M1_NGHOST;
    beg[2] = M1_NGHOST-1;

    end[0] = pts[0] - M1_NGHOST;
    end[1] = pts[1] - M1_NGHOST;
    end[2] = pts[2] - M1_NGHOST;
#endif
    
    // Indices aliases
    int & i = index[ii];
    int & j = index[ij];
    int & k = index[ik];
    
    // Actual indices
    int __i, __j, __k;
    
    // Scratch space
    // size = nvars * ngroups * nspecies * ncells
    Real * cons;
    Real * flux;
    Real * cmax = nullptr;
    
#if (test_thc_mode)
    Real * flux_jm = NULL;
    Real * flux_jp = NULL;
    Real * d_ptr   = NULL;
#endif

    xdirflux.ZeroClear();
    
    try {
      cons = new Real[ nvars * ngroups*nspecies * ncells[dir] ];
      flux = new Real[ nvars * ngroups*nspecies * ncells[dir] ];
      cmax = new Real[ nvars * ngroups*nspecies * ncells[dir] ];
      
#if (test_thc_mode)
      flux_jm = new Real[nvars*ngroups*nspecies];
      flux_jp = new Real[nvars*ngroups*nspecies];
#endif
      
    } catch (std::bad_alloc &e) {
      std::stringstream msg;
      msg << "Out of memory!" << std::endl;
      ATHENA_ERROR(msg);
    }
    
    
    for (__i = beg[0]; __i < end[0]; ++__i) {
      for (__j = beg[1]; __j < end[1]; ++__j) {
	
	
	// ----------------------------------------------
	// 1st pass compute the fluxes
	for (__k = 0; __k < pts[2]; ++__k) {
	  index[0] = __i;
	  index[1] = __j;
	  index[2] = __k;
	  
	  // 	char sbuf[128]; sprintf(sbuf,"k = %d j = %d i = %d", k,j,i); M1_DEBUG_PR(sbuf);
	  
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

	  // M1_DEBUG_PR("g_uu");
	  //   for (int a = 0; a < MDIM; ++a)
	  //     for (int b = 0; b < MDIM; ++b)
	  // 	M1_DEBUG_PR(g_uu(a,b));
	  
	  for (int ig = 0; ig < ngroups*nspecies; ++ig) {			 
	    
	    pack_F_d(beta_u(1), beta_u(2), beta_u(3),			 
		     vec.F_d(0,ig,k,j,i),					 
		     vec.F_d(1,ig,k,j,i),					 
		     vec.F_d(2,ig,k,j,i),					 
		     F_d);
	    
	    // M1_DEBUG_PR("F_d");
	    // for (int a = 0; a < 3; ++a)
	    //   M1_DEBUG_PR(vec.F_d(a,ig,k,j,i));
	    // M1_DEBUG_PR("packed F_d");
	    // for (int a = 0; a < MDIM; ++a)
	    //   M1_DEBUG_PR(F_d(a));
	    
	    pack_H_d(rad.Ht(ig,k,j,i),					 
		     rad.H(0,ig,k,j,i), rad.H(1,ig,k,j,i), rad.H(2,ig,k,j,i),  
		     H_d);
	    
	    for (int a = 0; a < NDIM; ++a) {
	      for (int b = a; b < NDIM; ++b) {
		
		assert(isfinite(rad.P_dd(a,b,ig,k,j,i)));
		//sprintf(sbuf," in Flux: a = %d b = %d ig = %d P_dd= %e", a,b,ig, rad.P_dd(a,b,ig,k,j,i));M1_DEBUG_PR(sbuf); 
	      }
	    }				 
	    pack_P_dd(beta_u(1), beta_u(2), beta_u(3),			 
		      rad.P_dd(0,0,ig,k,j,i), rad.P_dd(0,1,ig,k,j,i), rad.P_dd(1,1,ig,k,j,i),  
		      rad.P_dd(1,1,ig,k,j,i), rad.P_dd(1,2,ig,k,j,i), rad.P_dd(2,2,ig,k,j,i),  
		      P_dd);						 
	    
	    // M1_DEBUG_PR("P_dd");
 	    // for (int a = 0; a < MDIM; ++a)
	    //   for (int b = 0; b < MDIM; ++b)
	    // 	M1_DEBUG_PR(P_dd(a,b));
	    // M1_DEBUG_PR("rad.P_dd");
 	    // for (int a = 0; a < 3; ++a)
	    //   for (int b = 0; b < 3; ++b)
	    // 	M1_DEBUG_PR(rad.P_dd(a,b,ig,k,j,i));
	    // M1_DEBUG_PR("rad.J");
	    // M1_DEBUG_PR(rad.J(ig,k,j,i));

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
	    
	    // Scratch buffers 
	    cons[GFINDEX1D(__k, ig, 0)] = vec.F_d(0,ig,k,j,i);		 
	    cons[GFINDEX1D(__k, ig, 1)] = vec.F_d(1,ig,k,j,i);		 
	    cons[GFINDEX1D(__k, ig, 2)] = vec.F_d(2,ig,k,j,i);		 
	    cons[GFINDEX1D(__k, ig, 3)] = vec.E(ig,k,j,i);			 
	    if (nspecies > 1)						 
	      cons[GFINDEX1D(__k, ig, 4)] = vec.N(ig,k,j,i);		 

	    for (int iv = 0; iv < nvars; ++iv)
	      assert(isfinite(cons[GFINDEX1D(__k, ig, iv)]));			 

	    // M1_DEBUG_PR("alpha");
	    // M1_DEBUG_PR(alpha());
	    // M1_DEBUG_PR("beta_u");
	    // for (int a = 0; a < MDIM; ++a)
	    //   M1_DEBUG_PR(beta_u(a));
	    // M1_DEBUG_PR("F_u");
	    // for (int a = 0; a < MDIM; ++a)
	    //   M1_DEBUG_PR(F_u(a));
	    // M1_DEBUG_PR("     P_dd    P_ud");
 	    // for (int a = 0; a < MDIM; ++a)
	    //   for (int b = 0; b < MDIM; ++b) {
	    // 	char sbuf[128]; sprintf(sbuf,"%d %d  %e  %e",a,b,P_dd(a,b),P_ud(a,b)); M1_DEBUG_PR(sbuf);
	    //   }
	    
	    flux[GFINDEX1D(__k, ig, 0)] =					 
	      calc_F_flux(alpha(), beta_u, F_d, P_ud, dir+1, 1);		 
	    flux[GFINDEX1D(__k, ig, 1)] =					 
	      calc_F_flux(alpha(), beta_u, F_d, P_ud, dir+1, 2);		 
	    flux[GFINDEX1D(__k, ig, 2)] =					 
	      calc_F_flux(alpha(), beta_u, F_d, P_ud, dir+1, 3);		 
	    flux[GFINDEX1D(__k, ig, 3)] =					 
	      calc_E_flux(alpha(), beta_u, vec.E(ig,k,j,i), F_u, dir+1);	 
	    if (nspecies > 1)						 
	      flux[GFINDEX1D(__k, ig, 4)] = alpha() * nnu * fnu_u(dir+1);		 

	    for (int iv = 0; iv < nvars; ++iv)
	      assert(isfinite(flux[GFINDEX1D(__k, ig, iv)]));			 

	    // for (int iv = 0; iv < nvars; ++iv) {
	    //   char sbuf[128]; sprintf(sbuf,"(%d,%d,%d) h=%g iv = %d var = %d flux = %e",k,j,i,delta[dir],iv,mapiv[iv], flux[GFINDEX1D(__k, ig, iv)]); M1_DEBUG_PR(sbuf);
	    // }
	    
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
	    
	    //M1_DEBUG_PR("clight=");M1_DEBUG_PR(clight);
	    
	    // In some cases the eigenvalues in the thin limit
	    // overestimate the actual characteristic speed and can
	    // become larger than c
	    cmax[GFINDEX1D(__k, ig, 0)] = clight;
	    // = std::min(clight, cM1);
	    
	  }  // ig loop 
	} // __k loop
	
#if (test_thc_mode)
	// Cleanup the flux buffer
	memset(flux_jm, 0, nvars*ngroups*nspecies*sizeof(Real));
	memset(flux_jp, 0, nvars*ngroups*nspecies*sizeof(Real));
#endif
	
	// ----------------------------------------------
	// 2nd pass store the num fluxes
	for (__k = beg[2]; __k < end[2]; ++__k) {
	  index[0] = __i;
	  index[1] = __j;
	  index[2] = __k;
          
	  for (int ig = 0; ig < ngroups*nspecies; ++ig) {		
	    
	    Real avg_abs_1 = 0.5*(rmat.abs_1(ig,k,j,i)
				  + rmat.abs_1(ig,k+shift[2],j+shift[1],i+shift[0]));
	    Real avg_scat_1 = 0.5*(rmat.scat_1(ig,k,j,i)
				   + rmat.scat_1(ig,k+shift[2],j+shift[1],i+shift[0]));
	    
	    // Remove dissipation at high Peclet numbers 
	    Real kapa = 0.5*(avg_abs_1 + avg_scat_1); 
	    Real A = 1.0;
	    if (kapa*delta[dir] > 1.0) {
	      A = std::min(1.0, 1.0/(delta[dir]*kapa));
	      A = std::max(A, mindiss);
	    }
	  
	    for (int iv = 0; iv < nvars; ++iv) { 
	      Real const ujm = cons[GFINDEX1D(__k-1, ig, iv)]; 
	      Real const uj = cons[GFINDEX1D(__k, ig, iv)]; 
	      Real const ujp = cons[GFINDEX1D(__k+1, ig, iv)]; 
	      Real const ujpp = cons[GFINDEX1D(__k+2, ig, iv)]; 
	      
	      Real const fj = flux[GFINDEX1D(__k, ig, iv)]; 
	      Real const fjp = flux[GFINDEX1D(__k+1, ig, iv)]; 
	      
	      Real const cc = cmax[GFINDEX1D(__k, ig, 0)]; 
	      Real const ccp = cmax[GFINDEX1D(__k+1, ig, 0)]; 
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
	      
	      Real const flux_low =
		0.5*(fj + fjp - cmx*(ujp - uj));
	      Real const flux_high = 0.5*(fj + fjp);
	      
	      flux_num[iv] = flux_high
		- (sawtooth ? 1.0 : A)*(1.0 - phi)*(flux_high - flux_low); 
	      
	      if (M1_FLUXX_SET_ZERO && dir==0) flux_num[iv] = 0.0;
	      if (M1_FLUXY_SET_ZERO && dir==1) flux_num[iv] = 0.0;
	      if (M1_FLUXZ_SET_ZERO && dir==2) flux_num[iv] = 0.0;
	      
	      //M1_DEBUG_PR(flux_num[iv]);
	      //char sbuf[128]; sprintf(sbuf,"(%d,%d,%d) h=%g iv = %d  var = %d ig = %d  fluxnum = %e",k,j,i,delta[dir],iv,mapiv[iv],ig, flux_num[iv]); M1_DEBUG_PR(sbuf);
	      
#if (test_thc_mode)
	      
	      flux_jp[PINDEX1D(ig, iv)] = flux_num[iv];
	      
	      xdirflux(mapiv[iv],
		       ig, k,j,i) += 1.0/delta[dir]*(
						     flux_jm[PINDEX1D(ig, iv)] -
						     flux_jp[PINDEX1D(ig, iv)])*
		static_cast<Real>(
				  i >= M1_NGHOST
				  && i <  ncells[0] - M1_NGHOST
				  && j >= M1_NGHOST
				  && j <  ncells[1] - M1_NGHOST
				  && k >= M1_NGHOST
				  && k <  ncells[2] - M1_NGHOST);
	      
#else

	      xdirflux(mapiv[iv], ig, k+shift[2],j+shift[1],i+shift[0]) = flux_num[iv];
	      
#endif
	      
	      //char sbuf[128]; sprintf(sbuf,"(%d,%d,%d) h=%g iv = %d  var = %d ig = %d  fluxnum = %e",k,j,i,delta[dir],iv,mapiv[iv],ig, xdirflux(mapiv[iv],ig, k,j,i)); M1_DEBUG_PR(sbuf);
	      
	    
	    } // iv loop 
	    
	    
	  } // ig loop
	  
#if (test_thc_mode)
	  // Rotate flux pointer
	  d_ptr = flux_jm;
	  flux_jm = flux_jp;
	  flux_jp = d_ptr;
#endif
	  
	}// __k loop
	
      } // __j loop
    } // __i loop
    
    delete[] cons;
    delete[] flux;
    delete[] cmax;
    
  } // dir loop
  
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

