//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file m1_closure.cpp
//  \brief Calculate the M1 closure

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>

// Athena++ headers
#include "m1.hpp"
#include "../mesh/mesh.hpp"
#include "../coordinates/coordinates.hpp"

#if Z4C_ENABLED
#include "../z4c/z4c_macro.hpp"
#endif

#define SQ(X) ((X)*(X))

using namespace utils;

void M1::apply_closure(TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_dd,
                       TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_uu,
                       TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & n_d,
                       Real const w_lorentz,
                       TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_u,
                       TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & v_d,
                       TensorPointwise<Real, Symmetries::NONE, MDIM, 2> const & proj_ud,
                       Real const E,
                       TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & F_d,
                       Real const chi,
                       TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> & P_dd)
{
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> Pthin_dd;
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> Pthick_dd;

  Pthin_dd.NewTensorPointwise();
  Pthick_dd.NewTensorPointwise();

  calc_Pthin(g_uu, E, F_d, Pthin_dd);
  calc_Pthick(g_dd, g_uu, n_d, w_lorentz, v_d, E, F_d, Pthick_dd);

  Real const dthick = 1.5*(1. - chi);
  Real const dthin = 1. - dthick;

  for (int a = 0; a < MDIM; ++a) {
    for (int b = a; b < MDIM; ++b) {
      P_dd(a,b) = dthick*Pthick_dd(a,b) + dthin*Pthin_dd(a,b);
    }
  }
  Pthin_dd.DeleteTensorPointwise();
  Pthick_dd.DeleteTensorPointwise();
}

namespace {
  
  struct Parameters {
    Parameters(closure_t _closure,
               TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & _g_dd,
               TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & _g_uu,
               TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & _n_d,
               Real const _w_lorentz,
               TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & _u_u,
               TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & _v_d,
               TensorPointwise<Real, Symmetries::NONE, MDIM, 2> const & _proj_ud,
               Real const _E,
               TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & _F_d):      
               closure(_closure), g_dd(_g_dd), g_uu(_g_uu), n_d(_n_d),
               w_lorentz(_w_lorentz), u_u(_u_u), v_d(_v_d), proj_ud(_proj_ud),
               E(_E), F_d(_F_d) {}
               closure_t closure;
               TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_dd;
               TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_uu;
               TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & n_d;
               Real const w_lorentz;
               TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_u;
               TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & v_d;
               TensorPointwise<Real, Symmetries::NONE, MDIM, 2> const & proj_ud;
               Real const E;
               TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & F_d;
  };

  void print_stuff(MeshBlock * pmb,
                   int const i, int const j, int const k,
                   int const ig,
                   Parameters const * p,
                   std::ostream & ss)
  {

    // CC coords
    Real x1v = pmb->pcoord->x1v(i);
    Real x2v = pmb->pcoord->x2v(j);
    Real x3v = pmb->pcoord->x3v(k);

    // VC coords
    //Real x1f = pmb->pcoord->x1f(i);
    //Real x2f = pmb->pcoord->x2f(j);
    //Real x3f = pmb->pcoord->x3f(k);
    
    ss << "(i, j, k) = (" << i << ", " << j << ", " << k << ")\n";
    ss << "ig = " << ig << std::endl;
    ss << "(x, y, z) = (" << x1v << ", " << x2v << ", " << x3v << ")\n";

    // ss << "abs_0 = " << abs_0(k,j,i) << std::endl;
    // ss << "abs_1 = " << abs_1(k,j,i) << std::endl;
    // ss << "eta_0 = " << eta_0(k,j,i) << std::endl;
    // ss << "eta_1 = " << eta_1(k,j,i) << std::endl;
    // ss << "scat_1 = " << scat_1(k,j,i) << std::endl;
    
    //TODO: get hydro vars as well here
    //      this need the new eos/c2p!!!
    // ss << "rho = " << rho[ijk] << endl;
    // ss << "temperature = " << temperature[ijk] << endl;
    // ss << "Y_e = " << Y_e[ijk] << endl;

    //TODO: not sure printing these is useful,
    //      we already print the 4D metric interp.ed at the cell
    /*
    //Z4c::Z4c_vars z4c;
    //Z4c::SetZ4cliases(pmb->pz4c->storage.u, z4c);
    ss << "alp = " << z4c.alpha(k,j,i) << endl;
    ss << "beta = (" << z4c.beta(0,k,j,i) << ", "
                     << z4c.beta(1,k,j,i) << ", "
                     << z4c.beta(2,k,j,i) << ")\n";
    */
    
    ss << "g_uu = (";
    for (int a = 0; a < MDIM; ++a) {
      for (int b = 0; b < MDIM; ++b) {
        ss << p->g_uu(a,b) << ", ";
      }
    }

    ss << "\b\b)\n";
    ss << "g_dd = (";
    for (int a = 0; a < MDIM; ++a) {
      for (int b = 0; b < MDIM; ++b) {
        ss << p->g_dd(a,b) << ", ";
      }
    }

    ss << "\b\b)\n";
    ss << "w_lorentz = " << p->w_lorentz << std::endl;
    ss << "n_d = (";
    for (int a = 0; a < MDIM; ++a) {
      ss << p->n_d(a) << ", ";
    }

    ss << "\b\b)\n";
    ss << "u_u = (";
    for (int a = 0; a < MDIM; ++a) {
      ss << p->u_u(a) << ", ";
    }
    
    ss << "\b\b)\n";
    ss << "v_d = (";
    for (int a = 0; a < MDIM; ++a) {
      ss << p->v_d(a) << ", ";
    }

    ss << "\b\b)\n";
    ss << "E = " << p->E << std::endl;
    ss << "F_d = (";
    for (int a = 0; a < MDIM; ++a) {
        ss << p->F_d(a) << ", ";
    }

    ss << "\b\b)\n";
  }
} // anonymous namespace

  // Function to rootfind in order to determine the closure
double M1::zFunction(double xi, void * params) // Parameters * p)
{
  Parameters *p = reinterpret_cast<Parameters *>(params);

  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> P_dd;
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> rT_dd;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> H_d;

  P_dd.NewTensorPointwise();
  rT_dd.NewTensorPointwise();
  H_d.NewTensorPointwise();

  apply_closure(p->g_dd, p->g_uu, p->n_d, p->w_lorentz, p->u_u, p->v_d,
                p->proj_ud, p->E, p->F_d, p->closure(xi), P_dd);

  assemble_rT(p->n_d, p->E, p->F_d, P_dd, rT_dd);
  Real const J = calc_J_from_rT(rT_dd, p->u_u);

  calc_H_from_rT(rT_dd, p->u_u, p->proj_ud, H_d);
  Real const H2 = tensor::dot(p->g_uu, H_d, H_d);

  P_dd.DeleteTensorPointwise();
  rT_dd.DeleteTensorPointwise();
  H_d.DeleteTensorPointwise();

  return SQ(J*xi) - H2;
}

  //static double zFunctionWrapper(double x, void* obj) {
  //  M1* self = static_cast<M1*>(obj);
  //  return self->zFunction(x);
 // }

//double zFunction_gsl(double xi, void * params)
//{
//  Parameters * pgsl = reinterpret_cast<Parameters *>(params);
//  return pgsl->statePtr->zFunction(xi, params);
//}

Real flux_factor(TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_uu,
                 Real const J,
                 TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & H_d)
{
  Real xi = (J > 0 ? tensor::dot(g_uu, H_d, H_d)/SQ(J) : 0);
  return std::max(0.0, std::min(xi, 1.0));
}

Real eddington(Real const xi) {
  return 1.0/3.0;
}

Real minerbo(Real const xi) {
  //TODO: Isn't here xi actually xi*xi?
  return 1.0/3.0 + xi*xi*(6.0 - 2.0*xi + 6.0*xi*xi)/15.0;
}

Real thin(Real const xi) {
  return 1.0;
}

void M1::calc_closure_pt(MeshBlock * pmb,
                         int const i, int const j, int const k,
                         int const ig,
                         closure_t closure_fun,
                         gsl_root_fsolver * fsolver,
                         TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_dd,
                         TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_uu,
                         TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & n_d,
                         Real const w_lorentz,
                         TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_u,
                         TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & v_d,
                         TensorPointwise<Real, Symmetries::NONE, MDIM, 2> const & proj_ud,
                         Real const E,
                         TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & F_d,
                         Real * chi,
                         TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> & P_dd)
{    
  // These are special cases for which no root finding is needed
  if (closure_fun == eddington) {
    *chi = 1./3.;
    apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud,
		              E, F_d, *chi, P_dd);
    return;
  } else if (closure_fun == thin) {
    *chi = 1.0;
    apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud,
		              E, F_d, *chi, P_dd);
    return;
  }
  
  Parameters params(closure_fun, g_dd, g_uu, n_d,
		                w_lorentz, u_u, v_d, proj_ud, E, F_d);
  gsl_function F;
  F.function = &zFunction;
  F.params = reinterpret_cast<void *>(&params);
  
  double x_lo = 0.0;
  double x_hi = 1.0;
  
  int ierr = gsl_root_fsolver_set(fsolver, &F, x_lo, x_hi);
  // No root, most likely because of high velocities in the fluid
  // We default to optically thin closure
  if (ierr == GSL_EINVAL) {
    *chi = 1.0;
    apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud,
		              E, F_d, *chi, P_dd);
    return;
  } else if (ierr == GSL_EBADFUNC) {
    std::ostringstream msg;
    msg << "NaN or Inf found in closure!\n";
    print_stuff(pmb, i, j, k, ig, &params, msg);
    ATHENA_ERROR(msg);
  } else if (ierr != 0) {
    std::ostringstream msg;
    msg << "Unexpected error in gsl_root_fsolver_iterate,  error code \""
	      << ierr << "\"\n";
    print_stuff(pmb, i, j, k, ig, &params, msg);
    ATHENA_ERROR(msg);
  }
  
  // Rootfinding
  int iter = 0;
  do {
    ++iter;
    ierr = gsl_root_fsolver_iterate(fsolver);
    // Some nans in the evaluation. This should not happen.
    if (ierr == GSL_EBADFUNC) {
      std::ostringstream msg;
      msg << "NaNs or Infs found when computing the closure!\n";
      print_stuff(pmb, i, j, k, ig, &params, msg);
      ATHENA_ERROR(msg);
    } else if (ierr != 0) {
      std::ostringstream msg;
      msg << "Unexpected error in gsl_root_fsolver_iterate,  error code \""
	        << ierr << "\"\n";
      print_stuff(pmb, i, j, k, ig, &params, msg);
      ATHENA_ERROR(msg);
    }
    *chi = closure_fun(gsl_root_fsolver_root(fsolver));
    x_lo = gsl_root_fsolver_x_lower(fsolver);
    x_hi = gsl_root_fsolver_x_upper(fsolver);
    ierr = gsl_root_test_interval(x_lo, x_hi, closure_epsilon, 0);
  } while (ierr == GSL_CONTINUE && iter < closure_maxiter);
  
  if (ierr != GSL_SUCCESS) {
    std::ostringstream msg;
    msg << "Maximum number of iterations exceeded "
           "when computing the M1 closure\n";
    std::cout << msg.str();
  }
  
  // We are done, update the closure with the newly found chi
  apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud,
		            E, F_d, *chi, P_dd);
}

//----------------------------------------------------------------------------------------
// \!fn Real M1::CalcClosure(AthenaArray<Real> const & u)
// \brief computes the closure on a mesh block

void M1::CalcClosure(AthenaArray<Real> & u)
{
  MeshBlock * pmb = pmy_block;
  
  // Disable GSL error handler
  gsl_error_handler_t * gsl_err = gsl_set_error_handler_off();
  
  closure_t closure_fun;  
  if (closure == "Eddington") {
    closure_fun = eddington;
  } else if (closure == "Minerbo") {
    closure_fun = minerbo;
  } else if (closure == "thin") {
    closure_fun = thin;
  } else {
    std::ostringstream msg;
    msg << "Unknown closure " << closure << "\n";
    ATHENA_ERROR(msg);
  }
  
  gsl_root_fsolver * gsl_solver = gsl_root_fsolver_alloc(gsl_root_fsolver_brent);

  Lab_vars vec;
  //SetLabVarsAliases(storage.u, vec);
  SetLabVarsAliases(u, vec);  
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
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> n_d;

  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> u_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> u_d;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 2> proj_ud;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> v_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> v_d;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> H_d;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> H_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> fnu_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> F_d;
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> P_dd;
  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> T_dd;
    
  g_dd.NewTensorPointwise();
  beta_u.NewTensorPointwise();
  alpha.NewTensorPointwise();
  g_uu.NewTensorPointwise();
  n_d.NewTensorPointwise();

  u_u.NewTensorPointwise();
  u_d.NewTensorPointwise();
  proj_ud.NewTensorPointwise();
  v_u.NewTensorPointwise();
  v_d.NewTensorPointwise();
  H_d.NewTensorPointwise();
  H_u.NewTensorPointwise();
  fnu_u.NewTensorPointwise();
  F_d.NewTensorPointwise();
  P_dd.NewTensorPointwise();
  T_dd.NewTensorPointwise();
  
  //GCLOOP3(k,j,i) {
  CLOOP3(k,j,i) {
    for (int ig = 0; ig < nspecies*ngroups; ++ig) {
      if (rad.mask(k,j,i)) {
	      rad.J(ig,k,j,i) = 0;
	      for (int a = 0; a < NDIM; ++a) {
          rad.Ht(a,ig,k,j,i) = 0;
	        rad.H(a,ig,k,j,i)  = 0;
        }
	      for (int a = 0; a < NDIM; ++a) {
	        for (int b = a; b < NDIM; ++b) {
	          rad.P_dd(a,b,ig,k,j,i) = 0;
          }
        }
	      continue;
      }
    }

    // Go from ADM 3-metric VC (AthenaArray/Tensor)
    // to ADM 4-metric on CC at ijk (TensorPointwise) 
    Get4Metric_VC2CCinterp(pmb, k,j,i,
                           pmb->pz4c->storage.u, pmb->pz4c->storage.adm,
                           g_dd, beta_u, alpha);    
    Get4Metric_Inv(g_dd, beta_u, alpha, g_uu);
    Get4Metric_NormalForm(alpha, n_d);

    Real const W = fidu.Wlorentz(k,j,i);

    uvel(alpha(), beta_u(1), beta_u(2), beta_u(3), W,
	       fidu.vel_u(0,k,j,i), fidu.vel_u(1,k,j,i), fidu.vel_u(2,k,j,i), 
	       &u_u(0), &u_u(1), &u_u(2), &u_u(3));    

    tensor::contract(g_dd, u_u, u_d);
    calc_proj(u_d, u_u, proj_ud);
    pack_v_u(fidu.vel_u(0,k,j,i), fidu.vel_u(1,k,j,i), fidu.vel_u(2,k,j,i),  v_u);

    tensor::contract(g_dd, v_u, v_d);
  
   for (int ig = 0; ig < nspecies*ngroups; ++ig) {
      Real const W = fidu.Wlorentz(k,j,i);
      pack_F_d(beta_u(1), beta_u(2), beta_u(3),
	             vec.F_d(0,ig,k,j,i),
	             vec.F_d(1,ig,k,j,i),
	             vec.F_d(2,ig,k,j,i),
	             F_d);
      
      // chi, P_ab
      calc_closure_pt(pmb, i, j, k, ig,
                      closure_fun, gsl_solver, g_dd, g_uu, n_d,
                      W, u_u, v_d, proj_ud, vec.E(ig,k,j,i), F_d,
                      &rad.chi(ig,k,j,i), P_dd);
      unpack_P_dd(P_dd,
		              &rad.P_dd(0,0,ig,k,j,i), &rad.P_dd(0,1,ig,k,j,i), &rad.P_dd(0,2,ig,k,j,i),
                  &rad.P_dd(1,1,ig,k,j,i), &rad.P_dd(1,2,ig,k,j,i),
                  &rad.P_dd(2,2,ig,k,j,i));
      
      for (int a = 0; a < NDIM; ++a) {
	      for (int b = a; b < a; ++b) {
	        assert(isfinite(rad.P_dd(a,b,ig,k,j,i)));
        }
      }
      // J
      // TODO: Check this E
      assemble_rT(n_d, lab.E(ig,k,j,i), F_d, P_dd, T_dd);
      
      rad.J(ig,k,j,i) = calc_J_from_rT(T_dd, u_u);      

      assert(isfinite(rad.J(ig,k,j,i)));
      
      // H_a
      calc_H_from_rT(T_dd, u_u, proj_ud, H_d);

      unpack_H_d(H_d,
                 &rad.Ht(ig,k,j,i),
                 &rad.H(1,ig,k,j,i),
                 &rad.H(2,ig,k,j,i),
                 &rad.H(3,ig,k,j,i));

      assert(isfinite(rad.Ht(ig,k,j,i)));
      for (int a = 0; a < NDIM; ++a) {
	      assert(isfinite(rad.H(a,ig,k,j,i)));
      }
      
      // nnu
      tensor::contract(g_uu, H_d, H_u);

      assemble_fnu(u_u, rad.J(ig,k,j,i), H_u, fnu_u);

      Real const Gamma = alpha()*fnu_u(0);

      rad.nnu(ig,k,j,i) = vec.N(ig,k,j,i)/Gamma;

    } // ig loop
  } // CLOOP (k,j,i)
  
  g_dd.DeleteTensorPointwise();
  beta_u.DeleteTensorPointwise();
  alpha.DeleteTensorPointwise();
  g_uu.DeleteTensorPointwise();
  n_d.DeleteTensorPointwise();

  fidu.vel_u.DeleteAthenaTensor();
  u_u.DeleteTensorPointwise();
  u_d.DeleteTensorPointwise();
  proj_ud.DeleteTensorPointwise();
  v_u.DeleteTensorPointwise();
  v_d.DeleteTensorPointwise();
  H_d.DeleteTensorPointwise();
  H_u.DeleteTensorPointwise();
  fnu_u.DeleteTensorPointwise();
  F_d.DeleteTensorPointwise();
  P_dd.DeleteTensorPointwise();
  T_dd.DeleteTensorPointwise();
  
  gsl_root_fsolver_free(gsl_solver);
  
  // Restore GSL error handler
  gsl_set_error_handler(gsl_err);
  
}

