//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file m1_calc_update.cpp
//  \brief update the radiation on the mesh block

// C++ standard headers
#include <algorithm> // max
#include <cmath> // sqrt
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

// Athena++ headers
#include "m1.hpp"
#include "../z4c/z4c.hpp"
#include "../z4c/z4c_macro.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../hydro/hydro.hpp"

using namespace utils;

//----------------------------------------------------------------------------------------
// Function to update the radiation field 

void M1::CalcUpdate(Real const dt, AthenaArray<Real> & u_p, AthenaArray<Real> & u_c,
		    AthenaArray<Real> & u_rhs)
{

  M1_DEBUG_PR(" ");
  M1_DEBUG_PR(M1_SRC_METHOD);
  M1_DEBUG_PR("in: CalcUpdate, dt = ");M1_DEBUG_PR(dt);
  M1_DEBUG_PR(" ");
  
  MeshBlock * pmb = pmy_block;

  // Disable GSL error handler
  gsl_error_handler_t * gsl_err = gsl_set_error_handler_off();

  gsl_root_fsolver * gsl_solver_1d =
    gsl_root_fsolver_alloc(gsl_root_fsolver_brent);
  gsl_multiroot_fdfsolver * gsl_solver_nd =
    gsl_multiroot_fdfsolver_alloc(gsl_multiroot_fdfsolver_hybridsj, 4);
  
  closure_t closure_fun;
  if (closure == "Eddington") {
    closure_fun = eddington;
  }
  else if (closure == "Kershaw") {
    closure_fun = kershaw;
  }
  else if (closure == "Minerbo") {
    closure_fun = minerbo;
  }
  else if (closure == "thin") {
    closure_fun = thin;
  }
  else {
    std::ostringstream msg;
    msg << "Unknown closure " << closure << "\n";
    ATHENA_ERROR(msg);
  }

  // Steps
  // 1. F^m   = F^k + dt/2 [ A[F^k] + S[F^m]   ]
  // 2. F^k+1 = F^k + dt   [ A[F^m] + S[F^k+1] ]
  // At each step we solve an implicit problem in the form
  //    F = F^* + cdt S[F]
  // Where F^* = F^k + cdt A

  //TimeIntegratorStage--; //TODO: fixme
  
  Real mb = 0.0;
  (void)mb;
  if (nspecies > 1) {
    mb = 1.0; //FIXME available from fake_opac or neutrino_opac
  }
  
  //TODO: fix ptrs to fluid vars 3D grid vars (see also below)
  //densxp.InitWithShallowSlice(pmb->phydro->u,IDN,1);
  //densxn.InitWithShallowSlice(pmb->phydro->u,IDN,1);
  //sconx.InitWithShallowSlice(pmb->phydro->u,IVX,1);
  //scony.InitWithShallowSlice(pmb->phydro->u,IVY,1);
  //sconz.InitWithShallowSlice(pmb->phydro->u,IVZ,1);
  //tau.InitWithShallowSlice(pmb->phydro->u,IEN,1);
  
  Lab_vars vec_p;
  SetLabVarsAliases(u_p, vec_p);
  Lab_vars vec;
  SetLabVarsAliases(u_c, vec);  
  Lab_vars vec_rhs;
  SetLabVarsAliases(u_rhs, vec_rhs);
  
  // Pointwise 4D tensors used in the loop
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> g_dd;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> beta_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 0> alpha;  
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> g_uu;    
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> n_d;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> n_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 2> gamma_ud;

  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> F_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> S_d;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> tS_d;
    
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> u_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> u_d;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 2> proj_ud;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> v_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> v_d;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> H_d;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> Hstar_d;
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> P_dd;
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> rT_dd;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> H_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> fnu_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> F_d;

  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> Hnew_d;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> Fstar_d;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> Fnew_d;

  g_dd.NewTensorPointwise();
  beta_u.NewTensorPointwise();
  alpha.NewTensorPointwise();
  g_uu.NewTensorPointwise();
  gamma_ud.NewTensorPointwise();
  n_d.NewTensorPointwise();
  n_u.NewTensorPointwise();

  F_u.NewTensorPointwise();
  S_d.NewTensorPointwise();
  tS_d.NewTensorPointwise();
  
  u_u.NewTensorPointwise();
  u_d.NewTensorPointwise();
  proj_ud.NewTensorPointwise();
  v_u.NewTensorPointwise();
  v_d.NewTensorPointwise();
  H_d.NewTensorPointwise();
  Hstar_d.NewTensorPointwise();
  P_dd.NewTensorPointwise();
  rT_dd.NewTensorPointwise();
  H_u.NewTensorPointwise();
  fnu_u.NewTensorPointwise();
  F_d.NewTensorPointwise();

  Hnew_d.NewTensorPointwise();
  Fstar_d.NewTensorPointwise();
  Fnew_d.NewTensorPointwise();
  
  CLOOP3(k,j,i) {
    
    net.abs(k,j,i) = 0;
    net.heat(k,j,i) = 0;
    
    if (m1_mask(k,j,i)) {
      continue;
    }

    // Go from ADM 3-metric VC (AthenaArray/Tensor)
    // to ADM 4-metric on CC at ijk (TensorPointwise) 
    Get4Metric_VC2CCinterp(pmb, k,j,i,
                           pmb->pz4c->storage.u, pmb->pz4c->storage.adm,
                           g_dd, beta_u, alpha);    
    Get4Metric_Inv(g_dd, beta_u, alpha, g_uu);
    Get4Metric_NormalForm(alpha, n_d);
    Get4Metric_Normal(alpha, beta_u, n_u);
    Get4Metric_SpaceProj(n_u, n_d, gamma_ud);

    //TODO: check following:
    Real const volform = std::sqrt(SpatialDet(g_dd));
      
    uvel(alpha(), beta_u(1), beta_u(2), beta_u(3), fidu.Wlorentz(k,j,i),
	       fidu.vel_u(0,k,j,i), fidu.vel_u(1,k,j,i), fidu.vel_u(2,k,j,i), 
	       &u_u(0), &u_u(1), &u_u(2), &u_u(3));    
    
    tensor::contract(g_dd, u_u, u_d);
    calc_proj(u_d, u_u, proj_ud);

    pack_v_u(fidu.vel_u(0,k,j,i), fidu.vel_u(1,k,j,i), fidu.vel_u(2,k,j,i), v_u);
    tensor::contract(g_dd, v_u, v_d);

    Real const fidu_w_lorentz = fidu.Wlorentz(k,j,i);
    
    //
    // Source RHS are stored here
    Real DrE[ngroups*nspecies];
    Real DrFx[ngroups*nspecies];
    Real DrFy[ngroups*nspecies];
    Real DrFz[ngroups*nspecies];
    Real DrN[ngroups*nspecies];
    Real DDxp[ngroups*nspecies];

    
    //
    // Step 1 -- compute the sources
    for (int ig = 0; ig < ngroups*nspecies; ++ig) {

#if (M1_SRC_METHOD == M1_SRC_METHOD_EXPL)
      //
      // Radiation fields
      Real E = vec.E(ig,k,j,i);
      pack_F_d(beta_u(1), beta_u(2), beta_u(3),
               vec.F_d(0,ig,k,j,i),
               vec.F_d(1,ig,k,j,i),
               vec.F_d(2,ig,k,j,i),
               F_d);
      tensor::contract(g_uu, F_d, F_u);
      
      //
      // Compute radiation quantities in the fluid frame
      Real J = rad.J(ig,k,j,i);
      Real const Gamma = compute_Gamma(fidu_w_lorentz, v_u, J, E, F_d,
				       rad_E_floor, rad_eps);

      pack_H_d(rad.Ht(ig,k,j,i),
               rad.H(0,ig,k,j,i), rad.H(1,ig,k,j,i), rad.H(2,ig,k,j,i),
               H_d);

      //
      // Compute radiation sources
      calc_rad_sources(rmat.eta_1(ig,k,j,i)*volform, 
		       rmat.abs_1(ig,k,j,i), rmat.scat_1(ig,k,j,i), u_d, J, H_d,
		       S_d);
      DrE[ig] = dt*calc_rE_source(alpha(), n_u, S_d);
		
      calc_rF_source(alpha(), gamma_ud, S_d, tS_d);
      DrFx[ig] = dt * tS_d(1);
      DrFy[ig] = dt * tS_d(2);
      DrFz[ig] = dt * tS_d(3);

      if (nspecies > 1) {
	      DrN[ig] = dt*alpha()*(volform*rmat.eta_0(ig,k,j,i) -  
			      rmat.abs_0(ig,k,j,i)*vec.N(ig,k,j,i)/Gamma);
      }
      
#else

      //
      // Here we boost to the fluid frame, compute fluid matter
      // interaction, and boost back. These values are used as
      // initial guess for the implicit solve.
      
      //      
      // Advect radiation
      Real Estar = vec_p.E(ig,k,j,i) + 0.5*dt*vec_rhs.E(ig,k,j,i);
      pack_F_d(beta_u(1), beta_u(2), beta_u(3),
          vec_p.F_d(0,ig,k,j,i) + 0.5*dt * vec_rhs.F_d(0,ig,k,j,i),
          vec_p.F_d(1,ig,k,j,i) + 0.5*dt * vec_rhs.F_d(1,ig,k,j,i),
          vec_p.F_d(2,ig,k,j,i) + 0.5*dt * vec_rhs.F_d(2,ig,k,j,i),
          Fstar_d);
      apply_floor(g_uu, &Estar, Fstar_d);
      Real Nstar = 0; (void)Nstar;
      if (nspecies > 1) {
        Nstar = std::max(vec_p.N(ig,k,j,i) + 0.5*dt * vec_rhs.N(ig,k,j,i), rad_N_floor);
      }
      Real Enew;

      //
      // Compute quantities in the fluid frame
      calc_closure_pt(pmb, i, j, k, ig,
                      closure_fun, gsl_solver_1d, g_dd, g_uu, n_d,
                      fidu_w_lorentz, u_u, v_d,
                      proj_ud, Estar, Fstar_d, &rad.chi(ig,k,j,i), P_dd);
      assemble_rT(n_d, Estar, Fstar_d, P_dd, rT_dd);
      
      Real const Jstar = calc_J_from_rT(rT_dd, u_u);

      calc_H_from_rT(rT_dd, u_u, proj_ud, Hstar_d);

      //
      // Estimate interaction with matteer
      Real const dtau = dt/fidu_w_lorentz;
      Real Jnew = (Jstar + dtau*rmat.eta_1(ig,k,j,i)*volform)/(1. + dtau*rmat.abs_1(ig,k,j,i));

      // Only three components of H^a are independent; H^0 is found by
      // requiring H^a u_a = 0
      Real const khat = (rmat.abs_1(ig,k,j,i) + rmat.scat_1(ig,k,j,i));

      for (int a = 1; a < 4; ++a) {
        Hnew_d(a) = Hstar_d(a)/(1 + dtau*khat);
      }
      Hnew_d(0) = 0.0;
      for (int a = 1; a < 4; ++a) {
        Hnew_d(0) -= Hnew_d(a)*(u_u(a)/u_u(0));
      }
      
      //
      // Update Tmunu
      Real const H2 = tensor::dot(g_uu, Hnew_d, Hnew_d);

#if (M1_SRC_METHOD==M1_SRC_BOOST)
      Real const xi = sqrt(H2)*(Jnew > rad_E_floor ? 1/Jnew : 0);
      rad.chi(ig,k,j,i) = closure_fun(xi);
#else // this is really only important in the thick limit, so take chi = 1/3
      rad.chi(ig,k,j,i) = 1.0/3.0;
#endif
      
      Real const dthick = 3.*(1. - rad.chi(ig,k,j,i))/2.;
      Real const dthin = 1. - dthick;
      
      for(int a = 0; a < 4; ++a) {
        for(int b = a; b < 4; ++b) {
          rT_dd(a,b) = Jnew*u_d(a)*u_d(b) + Hnew_d(a)*u_d(b) + Hnew_d(b)*u_d(a) +
            dthin*Jnew*(Hnew_d(a)*Hnew_d(b)*(H2 > 0 ? 1/H2 : 0)) +
            dthick*Jnew*(g_dd(a,b) + u_d(a)*u_d(b))/3.0;
        }
      }
    
      //
      // Boost back to the lab frame
      Enew = calc_J_from_rT(rT_dd, n_u);
      calc_H_from_rT(rT_dd, n_u, gamma_ud, Fnew_d);
      apply_floor(g_uu, &Enew, Fnew_d);
      
#if (M1_SRC_METHOD == M1_SRC_METHOD_IMPL)

      // Compute interaction with matter
      source_update_pt(pmb, i, j, k, ig,
            closure_fun, gsl_solver_1d, gsl_solver_nd, dt/2,
            alpha(), g_dd, g_uu, n_d, n_u, gamma_ud, u_d, u_u,
            v_d, v_u, proj_ud, fidu.Wlorentz(k,j,i), Estar, Fstar_d,
            Estar, Fstar_d,
            &rad.chi(ig,k,j,i), rmat.eta_1(ig,k,j,i), rmat.abs_1(ig,k,j,i),
            rmat.scat_1(ig,k,j,i),
            &Enew, Fnew_d);
      apply_floor(g_uu, &Enew, Fnew_d);
      //
      // Update closure
      apply_closure(g_dd, g_uu, n_d, fidu.Wlorentz(k,j,i),
                    u_u, v_d, proj_ud, Enew, Fnew_d, rad.chi(k,j,i), P_dd);

      //
      // Compute new radiation energy density in the fluid frame
      assemble_rT(n_d, Enew, Fnew_d, P_dd, rT_dd);
      Jnew = calc_J_from_rT(rT_dd, u_u);

#endif // (M1_SRC_METHOD == M1_SRC_IMPL)

      //
      // Compute changes in radiation energy and momentum
      DrE[ig]  = Enew - Estar;
      DrFx[ig] = Fnew_d(1) - Fstar_d(1);
      DrFy[ig] = Fnew_d(2) - Fstar_d(2);
      DrFz[ig] = Fnew_d(3) - Fstar_d(3);
    
      if (nspecies > 1) {
        // Compute updated Gamma
        Real const Gamma = compute_Gamma(fidu_w_lorentz, v_u, Jnew, Enew, Fnew_d,
                rad_E_floor, rad_eps);
        if (source_therm_limit < 0 || dt*rmat.abs_0(ig,k,j,i) < source_therm_limit) {
          //
          // N^k+1 = N^* + dt ( eta - abs N^k+1 )
          Real Nnew = (Nstar + dt*alpha()*volform*rmat.eta_0(ig,k,j,i))/ 
            (1.0 + dt*alpha()*rmat.abs_0(ig,k,j,i)/Gamma);
          DrN[ig]  = Nnew - Nstar;
        } else {
          //
          // The neutrino number density is updated assuming the neutrino
          // average energies are those of the equilibrium

          //TODO define and set the nueave variables!!!
            //DrN[ig] = (nueave(ig,k,j,i) > 0 ? Gamma*Jnew/nueave(ig,k,j,i) - Nstar : 0.0);
        }
      } 

#endif // (THC_M1_SRC_METHOD == THC_M1_SRC_EXPL)
      
      //
      // Fluid lepton sources
      if (nspecies > 1) {
        DDxp[ig] = -mb*(DrN[ig]*(ig == 0) - DrN[ig]*(ig == 1));
      }
    } // ig loop
    
    //TODO: fix ptrs to fluid vars below!
    
    //
    // Step 2 -- limit the sources    
    Real theta = 1.0;
    if (source_limiter >= 0) {
      theta = 1.0;

      Real DTau_sum = 0.0;
      for (int ig = 0; ig < ngroups*nspecies; ++ig) {
        Real Estar = vec_p.E(ig,k,j,i) + dt * vec_rhs.E(ig,k,j,i);
        if (DrE[ig] < 0) {
          theta = std::min(-source_limiter*std::max(Estar, 0.0)/DrE[ig], theta);
        }
        DTau_sum -= DrE[ig];
      } // ig loop
      if (DTau_sum < 0) {
        theta = std::min(-source_limiter/DTau_sum, theta);
        //theta = std::min(0.0, theta);
      }
    
      if (nspecies > 1) {
        Real DDxp_sum = 0.0;
        for (int ig = 0; ig < ngroups*nspecies; ++ig) {
          Real Nstar = vec_p.N(ig,k,j,i) + dt * vec_rhs.N(ig,k,j,i);
          if (DrN[ig] < 0) {
            theta = std::min(-source_limiter*std::max(Nstar, 0.0)/DrN[ig], theta);
          }
          DDxp_sum += DDxp[ig];
        }
      
        //Real const DYe = DDxp_sum/pmb->phydro->u(IDN,k,j,i);
        //if (DYe < 0) {
          //FIXME theta = min(source_limiter*max(source_Ye_max - XXX.Y_e(k,j,i), 0.0)/DYe, theta);
        //}
        //else if (DYe < 0) {
          //FIXME theta = min(source_limiter*min(source_Ye_min - XXX.Y_e(k,j,i), 0.0)/DYe, theta);
        //}
      }
      theta = std::max(0.0, theta);
    } // source limiter
    
    //
    // Step 3 -- update fields
    for (int ig = 0; ig < ngroups*nspecies; ++ig) {

      //
      // Update radiation quantities
      Real E = vec_p.E(ig,k,j,i)     + dt * vec_rhs.E(ig,k,j,i)     + theta*DrE[ig];
      F_d(1) = vec_p.F_d(0,ig,k,j,i) + dt * vec_rhs.F_d(0,ig,k,j,i) + theta*DrFx[ig];
      F_d(2) = vec_p.F_d(1,ig,k,j,i) + dt * vec_rhs.F_d(1,ig,k,j,i) + theta*DrFy[ig];
      F_d(3) = vec_p.F_d(2,ig,k,j,i) + dt * vec_rhs.F_d(2,ig,k,j,i) + theta*DrFz[ig];

      apply_floor(g_uu, &E, F_d);

      Real N = 0; (void)N;
      if (nspecies > 1) {
        N = vec_p.N(ig,k,j,i) + dt * vec_rhs.N(ig,k,j,i)    + theta*DrN[ig];
        N = std::max(N, rad_N_floor);
      }
            
      //
      // Compute back reaction on the fluid
      // NOTE: fluid backreaction is only needed at the last substep

      //FIXME ME!!!
      //if (backreact && 0 == TimeIntegratorStage) {

      // Current implementation is restricted.
      assert (ngroups == 1);
    
      //TODO: fix ptrs to fluid vars
      //sconx(k,j,i)  -= theta*DrFx[ig];
      //scony(k,j,i)  -= theta*DrFy[ig];
      //sconz(k,j,i)  -= theta*DrFz[ig];
      //tau(k,j,i)    -= theta*DrE[ig];
    
      net.heat(k,j,i) -= theta*DrE[ig];

      if (nspecies > 1) {
        //densxp(k,j,i)  += theta*DDxp[ig];
        //densxn(k,j,i)  -= theta*DDxp[ig];
        net.abs(k,j,i) += theta*DDxp[ig];
      }

      //} // backreact
    
      //
      // Save updated results into grid functions
      vec.E(ig,k,j,i) = E;
      unpack_F_d(F_d, &vec.F_d(0,ig,k,j,i), &vec.F_d(1,ig,k,j,i), &vec.F_d(2,ig,k,j,i));
      if (nspecies > 1) vec.N(ig,k,j,i) = N;

      assert(isfinite(vec.E(ig,k,j,i)));
      assert(isfinite(vec.F_d(0,ig,k,j,i)));
      assert(isfinite(vec.F_d(1,ig,k,j,i)));
      assert(isfinite(vec.F_d(2,ig,k,j,i)));
       
    } // ig loop
  } // CLOOP3
  
  g_dd.DeleteTensorPointwise();
  beta_u.DeleteTensorPointwise();
  alpha.DeleteTensorPointwise();
  g_uu.DeleteTensorPointwise();
  n_d.DeleteTensorPointwise();
  n_u.DeleteTensorPointwise();
  gamma_ud.DeleteTensorPointwise();

  F_u.DeleteTensorPointwise();
  S_d.DeleteTensorPointwise();
  tS_d.DeleteTensorPointwise();
  
  u_u.DeleteTensorPointwise();
  u_d.DeleteTensorPointwise();
  proj_ud.DeleteTensorPointwise();
  v_u.DeleteTensorPointwise();
  v_d.DeleteTensorPointwise();
  H_d.DeleteTensorPointwise();
  Hstar_d.DeleteTensorPointwise();
  P_dd.DeleteTensorPointwise();
  rT_dd.DeleteTensorPointwise();
  H_u.DeleteTensorPointwise();
  fnu_u.DeleteTensorPointwise();
  F_d.DeleteTensorPointwise();
  
  Hnew_d.DeleteTensorPointwise();
  Fstar_d.DeleteTensorPointwise();
  Fnew_d.DeleteTensorPointwise();
  
  gsl_root_fsolver_free(gsl_solver_1d);
  gsl_multiroot_fdfsolver_free(gsl_solver_nd);
      
  // Restore GSL error handler
  gsl_set_error_handler(gsl_err);
 
}

//----------------------------------------------------------------------------------------
// Function to update the radiation field without sources

void M1::CalcUpdate_advection(Real const dt, AthenaArray<Real> & u_p, AthenaArray<Real> & u_c,
			      AthenaArray<Real> & u_rhs)
{
  M1_DEBUG_PR(" ");
  M1_DEBUG_PR("in: CalcUpdate_advection, dt = ");M1_DEBUG_PR(dt);
  M1_DEBUG_PR(" ");
  
  MeshBlock * pmb = pmy_block;

  Lab_vars vec_p;
  SetLabVarsAliases(u_p, vec_p);
  Lab_vars vec;
  SetLabVarsAliases(u_c, vec);  
  Lab_vars vec_rhs;
  SetLabVarsAliases(u_rhs, vec_rhs);
  
  CLOOP3(k,j,i) {
    
    for (int ig = 0; ig < ngroups*nspecies; ++ig) {

      vec.E(ig,k,j,i) = std::max(vec_p.E(ig,k,j,i) + dt * vec_rhs.E(ig,k,j,i),
				   rad_E_floor);
      assert(isfinite(vec.E(ig,k,j,i)));
      
      for (int a = 0; a < 3; ++a) {
	vec.F_d(a,ig,k,j,i) = vec_p.F_d(a,ig,k,j,i) + dt * vec_rhs.F_d(a,ig,k,j,i);
	assert(isfinite(vec.F_d(a,ig,k,j,i)));
      }
      
      //if (nspecies > 1) {
      vec.N(ig,k,j,i) = std::max(vec_p.N(ig,k,j,i) + dt * vec_rhs.N(ig,k,j,i),
				   rad_N_floor);
      assert(isfinite(vec.N(ig,k,j,i)));
      //}
      
    } // ig loop
    
  } // CLOOP3
  
}
