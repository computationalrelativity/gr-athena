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

// TODO: Check this: need to run over ngroups*nspecies
#define GFINDEX3D(i,j,k) \
    (i + lsh[0]*(j+lsh[1]*k))

#define TINY (1e-10)

#define GFINDEX1D(__k, ig, iv) \
    ((iv) + (ig)*5 + (__k)*(5*ngroups*nspecies))

#define PINDEX1D(ig, iv) \
    ((iv) + (ig)*5)

using namespace utils;

Real compute_Gamma(Real const W,
		 		   				 TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & v_u,
		 							 Real const J, Real const E,
									 TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & F_d);

namespace {
  Real minmod2(Real rl, Real rp, Real th) {
    return std::min(1.0, std::min(th*rl, th*rp));
  } 
}

//----------------------------------------------------------------------------------------
// \!fn void M1::CalcFluxes(AthenaArray<Real> & u, AthenaArray<Real> & u_rhs)
// \brief Compute the numerical fluxes using a simple 2nd order flux-limited method
//        with high-Peclet limit fix. The fluxes are then added to the RHSs.
void M1::CalcFluxes(AthenaArray<Real> & u, AthenaArray<Real> & u_rhs)
{
  MeshBlock * pmb = pmy_block;
  
  Lab_vars vec;
  SetLabVarsAliases(u, vec);  
  Lab_vars vec_rhs;
  SetLabVarsAliases(u_rhs, vec_rhs);

  // Aliases for the RHS pointers
  Real ** rhs = new Real * [ngroups*nspecies*5];
  for (int ig = 0; ig < ngroups*nspecies; ++ig) {
    rhs[PINDEX1D(ig, 0)] = &vec_rhs.N  (   ig, 0,0,0);
    rhs[PINDEX1D(ig, 1)] = &vec_rhs.F_d(0, ig, 0,0,0);
    rhs[PINDEX1D(ig, 2)] = &vec_rhs.F_d(1, ig, 0,0,0);
    rhs[PINDEX1D(ig, 3)] = &vec_rhs.F_d(2, ig, 0,0,0);
    rhs[PINDEX1D(ig, 4)] = &vec_rhs.E  (   ig, 0,0,0);
  }

  // Grid data
  Real const delta[3] = {
    pmb->pcoord->dx1v(0),
    pmb->pcoord->dx1v(1),
    pmb->pcoord->dx1v(0),
  };
  Real const idelta[3] = {
    1.0/pmb->pcoord->dx1v(0),
    1.0/pmb->pcoord->dx1v(1),
    1.0/pmb->pcoord->dx1v(0),
  };
  int const ncells[3] = {
    pmb->ncells1,
    pmb->ncells2,
    pmb->ncells3,
  };

  // Pointwise 4D tensors used in the loop
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> g_dd;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> beta_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 0> alpha;  
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> g_uu;    

  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> gamma_uu; // NDIM !

  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> u_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> v_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> H_d;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> H_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> F_d;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> F_u;
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> P_dd;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 2> P_ud;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> fnu_u;
    
  g_dd.NewTensorPointwise();
  beta_u.NewTensorPointwise();
  alpha.NewTensorPointwise();
  g_uu.NewTensorPointwise();
  gamma_uu.NewTensorPointwise();

  u_u.NewTensorPointwise();
  v_u.NewTensorPointwise();
  H_d.NewTensorPointwise();
  H_u.NewTensorPointwise();
  F_d.NewTensorPointwise();
  F_u.NewTensorPointwise();
  P_dd.NewTensorPointwise();
  P_ud.NewTensorPointwise();
  fnu_u.NewTensorPointwise();

  // Go through directions
  for (int dir = 0; dir < NDIM; ++dir) {
    int index[3];
    int lsh[3];

    // We have to leave as the most internal loop the one on the
    // direction. For this reason we will remap the usual indices
    // i,j,k into different points of index[:].
    int ii, ij, ik;
    switch(dir) {
			case 0:
				ii = 2;
				ij = 1;
				ik = 0;

				lsh[0] = ncells[2];
				lsh[1] = ncells[1];
				lsh[2] = ncells[0];
				
				break;
			case 1:
				ii = 1;
				ij = 2;
				ik = 0;
				
				lsh[0] = ncells[2];
				lsh[1] = ncells[0];
				lsh[2] = ncells[1];
				
				break;
			case 2:
				ii = 1;
				ij = 0;
				ik = 2;
				
				lsh[0] = ncells[1];
				lsh[1] = ncells[0];
				lsh[2] = ncells[2];
				
				break;
    }
    
    // Indices aliases
    int & i = index[ii];
    int & j = index[ij];
    int & k = index[ik];
    
    // Actual indices
    int __i, __j, __k;
    
    // Scratch space
    Real * cons;
    Real * flux;
    Real * cmax    = nullptr;
    Real * flux_jm = nullptr;
    Real * flux_jp = nullptr;
    Real * d_ptr   = nullptr;
    try {
      cons = new Real[5*ngroups*nspecies*lsh[2]];
      flux = new Real[5*ngroups*nspecies*lsh[2]];
      cmax = new Real[5*ngroups*nspecies*lsh[2]];
      flux_jm = new Real[5*ngroups*nspecies];
      flux_jp = new Real[5*ngroups*nspecies];
    } catch (std::bad_alloc &e) {
      std::stringstream msg;
      msg << "Out of memory!" << std::endl;
      ATHENA_ERROR(msg);
    }
    
    //TODO check the order of these loops!
    for (__i = M1_NGHOST; __i < lsh[0] - M1_NGHOST; ++__i) {
      for (__j = M1_NGHOST; __j < lsh[1] - M1_NGHOST; ++__j) {

        // ----------------------------------------------
        // 1st pass compute the fluxes
        // . this could be made into a separate Cactus scheduled
        //   function to avoid some duplicated calculations
        for (__k = 0; __k < lsh[2]; ++__k) {
	  
          index[0] = __i;
          index[1] = __j;
          index[2] = __k;
          int const ijk = GFINDEX3D(i,j,k);
          // Go from ADM 3-metric VC (AthenaArray/Tensor)
          // to ADM 4-metric on CC at ijk (TensorPointwise)
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
                                             rad.J(ig,k,j,i), vec.E(ig,k,j,i), F_d);

            // Note that nnu is densitized here
            Real const nnu = vec.N(ig,k,j,i)/Gamma;

            cons[GFINDEX1D(__k, ig, 0)] = vec.N(ig,k,j,i);
            cons[GFINDEX1D(__k, ig, 1)] = vec.F_d(0,ig,k,j,i);
            cons[GFINDEX1D(__k, ig, 2)] = vec.F_d(1,ig,k,j,i);
            cons[GFINDEX1D(__k, ig, 3)] = vec.F_d(2,ig,k,j,i);
            cons[GFINDEX1D(__k, ig, 4)] = vec.E(ig,k,j,i);
            
            assert(isfinite(cons[GFINDEX1D(__k, ig, 0)]));
            assert(isfinite(cons[GFINDEX1D(__k, ig, 1)]));
            assert(isfinite(cons[GFINDEX1D(__k, ig, 2)]));
            assert(isfinite(cons[GFINDEX1D(__k, ig, 3)]));
            assert(isfinite(cons[GFINDEX1D(__k, ig, 4)]));
            
            flux[GFINDEX1D(__k, ig, 0)] =
                alpha() * nnu * fnu_u(dir+1);
            flux[GFINDEX1D(__k, ig, 1)] =
                calc_F_flux(alpha(), beta_u, F_d, P_ud, dir+1, 1);
            flux[GFINDEX1D(__k, ig, 2)] =
                calc_F_flux(alpha(), beta_u, F_d, P_ud, dir+1, 2);
            flux[GFINDEX1D(__k, ig, 3)] =
                calc_F_flux(alpha(), beta_u, F_d, P_ud, dir+1, 3);
            flux[GFINDEX1D(__k, ig, 4)] =
                calc_E_flux(alpha(), beta_u, vec.E(ig,k,j,i), F_u, dir+1);
            
            assert(isfinite(flux[GFINDEX1D(__k, ig, 0)]));
            assert(isfinite(flux[GFINDEX1D(__k, ig, 1)]));
            assert(isfinite(flux[GFINDEX1D(__k, ig, 2)]));
            assert(isfinite(flux[GFINDEX1D(__k, ig, 3)]));
            assert(isfinite(flux[GFINDEX1D(__k, ig, 4)]));
            
            // Eigenvalues in the optically thin limit
            //
            // Using the optically thin eigenvalues seems to cause
            // problems in some situations, possibly because the
            // optically thin closure is acausal in certain
            // conditions, so this is commented for now.
#if 0
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
	    
	    // TODO optically thick characteristic speeds and combination
#endif
            // Speed of light -- note that gamma_uu has NDIM=3
            Real const clam[2] = {
                alpha()*std::sqrt(gamma_uu(dir,dir)) - beta_u(dir+1),
                - alpha()*std::sqrt(gamma_uu(dir,dir)) - beta_u(dir+1),
            };
            Real const clight = std::max(std::abs(clam[0]), std::abs(clam[1]));
            
            // In some cases the eigenvalues in the thin limit
            // overestimate the actual characteristic speed and can
            // become larger than c
            cmax[GFINDEX1D(__k, ig, 0)] = clight;
            // = std::min(clight, cM1);
          }
        }
	
        // Cleanup the flux buffer
        memset(flux_jm, 0, 5*ngroups*nspecies*sizeof(Real));
        memset(flux_jp, 0, 5*ngroups*nspecies*sizeof(Real));
          
        // ----------------------------------------------
        // 2nd pass update the RHS
        for (__k = M1_NGHOST-1; __k < lsh[2]-M1_NGHOST; ++__k) {
          index[0] = __i;
          index[1] = __j;
          index[2] = __k;
          int const ijk = GFINDEX3D(i,j,k);
        
          for (int ig = 0; ig < ngroups*nspecies; ++ig) {

            Real avg_abs_1 = rmat.abs_1(ig,k,j,i);
            Real avg_scat_1 = rmat.scat_1(ig,k,j,i);
            
            index[2]++;
            
            avg_abs_1 += rmat.abs_1(ig,k,j,i);
            avg_scat_1 += rmat.scat_1(ig,k,j,i);
            
            index[2]--;
            
            // Remove dissipation at high Peclet numbers
            Real kapa = 0.5*(avg_abs_1 + avg_scat_1);
            Real A = 1.0;
            if (kapa*delta[dir] > 1.0) {
              A = std::max(1.0/(delta[dir]*kapa), mindiss);
            }

            for (int iv = 0; iv < 5; ++iv) {
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
                phi = minmod2(dum/duc, dup/duc, 1.0);
              } else if (dup*duc < 0 && dum*duc < 0) {
                sawtooth = true;
              }
              assert(isfinite(phi));

              Real const flux_low = 0.5*(fj + fjp - cmx*(ujp - uj));
              Real const flux_high = 0.5*(fj + fjp);
              
              Real flux_num = flux_high -
                              (sawtooth ? 1 : A)*(1 - phi)*(flux_high - flux_low);
              flux_jp[PINDEX1D(ig, iv)] = flux_num;

              if (!rad.mask(k,j,i)) {
          
                //TODO check this pointer!!!
                rhs[PINDEX1D(ig, iv)][ijk] += idelta[dir]*(flux_jm[PINDEX1D(ig, iv)] -
                                                           flux_jp[PINDEX1D(ig, iv)]) *
                                                           static_cast<Real>(i >= M1_NGHOST
                                                               && i <  ncells[0] - M1_NGHOST
                                                               && j >= M1_NGHOST
                                                               && j <  ncells[1] - M1_NGHOST
                                                               && k >= M1_NGHOST
                                                               && k <  ncells[2] - M1_NGHOST);
                assert(isfinite(rhs[PINDEX1D(ig, iv)][ijk]));
              }
            } //iv loop
          } //ig loop
          // Rotate flux pointer
          d_ptr = flux_jm;
          flux_jm = flux_jp;
          flux_jp = d_ptr;
	      }
      }
      delete[] cons;
      delete[] flux;
      delete[] cmax;
      delete[] flux_jm;
      delete[] flux_jp;
    }
  } // dir loop

  delete[] rhs;
    
  g_dd.DeleteTensorPointwise();
  beta_u.DeleteTensorPointwise();
  alpha.DeleteTensorPointwise();
  g_uu.DeleteTensorPointwise();
  
  gamma_uu.DeleteTensorPointwise();

  u_u.DeleteTensorPointwise();
  v_u.DeleteTensorPointwise();
  H_d.DeleteTensorPointwise();
  H_u.DeleteTensorPointwise();
  F_d.DeleteTensorPointwise();
  F_u.DeleteTensorPointwise();
  P_dd.DeleteTensorPointwise();
  P_ud.DeleteTensorPointwise();
  fnu_u.DeleteTensorPointwise();
}

