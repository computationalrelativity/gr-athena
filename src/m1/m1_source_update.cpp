//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file m1_source_update.cpp
//  \brief solves the implicit problem
//           q^new = q^star + dt S[q^new]
//         at a point
// The source term is S^a = (eta - ka J) u^a - (ka + ks) H^a and includes also emission.

// C++ standard headers
#include <cmath> // pow
#include <sstream>

// Athena++ headers
#include "m1.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"

#if Z4C_ENABLED
#include "../z4c/z4c.hpp"
#include "../z4c/z4c_macro.hpp"
#endif

#define SQ(X) ((X)*(X))
#define TINY (1e-10)

using namespace utils;
//----------------------------------------------------------------------------------------
// Low level kernel computing the Jacobian matrix

// TODO: check where to define or read this
Real source_epsabs = 1e-3,source_epsrel = 1e-3;
int source_maxiter = 100;

namespace {
  
  inline Real radM1_set_dthin(Real chi) { return 1.5*chi-0.5; }
  inline Real radM1_set_dthick(Real chi){ return 1.5*(1.0-chi); }
  
  void __source_jacobian_low_level(
				   Real *qpre, Real Fup[4], Real F2,
				   Real chi,
				   Real kapa, Real kaps,
				   Real vup[4], Real vdown[4], Real v2,
				   Real W,
				   Real alpha,
				   Real cdt,
				   Real *qstar,
				   gsl_matrix * J)
  {
    const Real kapas = kapa+kaps;
    const Real alpW  = alpha * W;
    
    const Real dthin = radM1_set_dthin(chi);
    const Real dthick = radM1_set_dthick(chi);
    
    const Real vx = vdown[1];
    const Real vy = vdown[2];
    const Real vz = vdown[3];
    const Real W2 = SQ(W);
    const Real W3 = W2*W;
    
    const Real vdotF = Fup[1]*vdown[1] + Fup[2]*vdown[2] + Fup[3]*vdown[3];
    const Real normF = sqrt(F2);
    const Real inormF = (normF > 0 ? 1/normF : 0);
    const Real vdothatf = vdotF*inormF;
    const Real vdothatf2 = SQ(vdothatf);
    const Real hatfx = qpre[1]*inormF; // hatf_i
    const Real hatfy = qpre[2]*inormF;
    const Real hatfz = qpre[3]*inormF;
    const Real hatfupx = Fup[1]*inormF; // hatf^i
    const Real hatfupy = Fup[2]*inormF;
    const Real hatfupz = Fup[3]*inormF;
    const Real e = qpre[0];
    const Real eonormF = std::min(e*inormF, 1.0); // with factor dthin ...
    
    // drvts of J
    Real JdE = W2 + dthin*vdothatf2*W2 + (dthick*(3 - 2*W2)*(-1 + W2))/(1 + 2*W2);
    
    Real JdFv = 2*W2*(-1 + (dthin*eonormF*vdothatf) + (2*dthick*(-1 + W2))/(1 + 2*W2));
    Real JdFf = (-2*dthin*eonormF*vdothatf2*W2);
    
    Real JdFx = JdFv * vup[1] + JdFf * hatfupx;
    Real JdFy = JdFv * vup[2] + JdFf * hatfupy;
    Real JdFz = JdFv * vup[3] + JdFf * hatfupz;
    
    // drvts of Hi
    Real HdEv = W3*(-1 - dthin*vdothatf2 + (dthick*(-3 + 2*W2))/(1 + 2*W2));
    Real HdEf = -(dthin*vdothatf*W);
    
    Real HxdE = HdEv * vx + HdEf * hatfx;
    Real HydE = HdEv * vy + HdEf * hatfy;
    Real HzdE = HdEv * vz + HdEf * hatfz;
    
    Real HdFdelta = (1 - dthick*v2 - (dthin*eonormF*vdothatf))*W;
    Real HdFvv = (2*(1 - dthin*eonormF*vdothatf)*W3) + dthick*W*(2 - 2*W2 + 1/(-1 - 2*W2));
    Real HdFff = (2*dthin*eonormF*vdothatf*W);
    Real HdFvf = (2*dthin*eonormF*vdothatf2*W3);
    Real HdFfv = -(dthin*eonormF*W);
    
    Real HxdFx = HdFdelta + HdFvv * vx * vup[1] + HdFff * hatfx * hatfupx + HdFvf * vx * hatfupx + HdFfv * hatfx * vup[1];
    Real HydFx = HdFvv * vy * vup[1] + HdFff * hatfy * hatfupx + HdFvf * vy * hatfupx + HdFfv * hatfy * vup[1];
    Real HzdFx = HdFvv * vz * vup[1] + HdFff * hatfz * hatfupx + HdFvf * vz * hatfupx + HdFfv * hatfz * vup[1];
    
    Real HxdFy = HdFvv * vx * vup[2] + HdFff * hatfx * hatfupy + HdFvf * vx * hatfupy + HdFfv * hatfx * vup[2];
    Real HydFy = HdFdelta + HdFvv * vy * vup[2] + HdFff * hatfy * hatfupy + HdFvf * vy * hatfupy + HdFfv * hatfy * vup[2];
    Real HzdFy = HdFvv * vz * vup[2] + HdFff * hatfz * hatfupy + HdFvf * vz * hatfupy + HdFfv * hatfz * vup[2];
    
    Real HxdFz = HdFvv * vx * vup[3] + HdFff * hatfx * hatfupz + HdFvf * vx * hatfupz + HdFfv * hatfx * vup[3];
    Real HydFz = HdFvv * vy * vup[3] + HdFff * hatfy * hatfupz + HdFvf * vy * hatfupz + HdFfv * hatfy * vup[3];
    Real HzdFz = HdFdelta + HdFvv * vz * vup[3] + HdFff * hatfz * hatfupz + HdFvf * vz * hatfupz + HdFfv * hatfz * vup[3];
    
    // Build the Jacobian
    Real J00 = - alpW * ( kapas - kaps * JdE);
    
    Real J0x = + alpW * kaps * JdFx - alpW * kapas * vup[1];
    Real J0y = + alpW * kaps * JdFy - alpW * kapas * vup[2];
    Real J0z = + alpW * kaps * JdFz - alpW * kapas * vup[3];
    
    Real Jx0 = - alpha * ( kapas * HxdE + W * kapa * vx * JdE );
    Real Jy0 = - alpha * ( kapas * HydE + W * kapa * vy * JdE );
    Real Jz0 = - alpha * ( kapas * HzdE + W * kapa * vz * JdE );
    
    Real Jxx = - alpha * ( kapas * HxdFx + W * kapa * vx * JdFx );
    Real Jxy = - alpha * ( kapas * HxdFy + W * kapa * vx * JdFy );
    Real Jxz = - alpha * ( kapas * HxdFz + W * kapa * vx * JdFz );
    
    Real Jyy = - alpha * ( kapas * HydFx + W * kapa * vy * JdFx );
    Real Jyx = - alpha * ( kapas * HydFy + W * kapa * vy * JdFy );
    Real Jyz = - alpha * ( kapas * HydFz + W * kapa * vy * JdFz );
    
    Real Jzx = - alpha * ( kapas * HzdFx + W * kapa * vz * JdFx );
    Real Jzy = - alpha * ( kapas * HzdFy + W * kapa * vz * JdFy );
    Real Jzz = - alpha * ( kapas * HzdFz + W * kapa * vz * JdFz );

    // Store Jacobian into J
    Real A_data[4][4] = { 1 - cdt*J00, - cdt*J0x, - cdt*J0y, - cdt*J0z,
      - cdt*Jx0, 1 - cdt*Jxx, - cdt*Jxy, - cdt*Jxz,
      - cdt*Jy0, - cdt*Jyx, 1 - cdt*Jyy, - cdt*Jyz,
      - cdt*Jz0, - cdt*Jzx, - cdt*Jzy, 1 - cdt*Jzz, };
    for (int a = 0; a < 4; ++a)
      for (int b = 0; b < 4; ++b) {
	gsl_matrix_set(J, a, b, A_data[a][b]);
      }
    
  }
  
} // namespace

//----------------------------------------------------------------------------------------
// Setup for the GSL solver

namespace {
  
  struct Params {
      Params(MeshBlock * _pmb,
             int const _i,
             int const _j,
             int const _k,
             int const _ig,
             closure_t _closure,
             gsl_root_fsolver  * _gsl_solver_1d,
             Real const _cdt,
             Real const _alp,
             TensorPointwise<Real, Symmetries::SYM2, MDIM, 2>const & _g_dd,
             TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & _g_uu,
             TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & _n_d,
             TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & _n_u,
             TensorPointwise<Real, Symmetries::NONE, MDIM, 2> const & _gamma_ud,
             TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & _u_d,
             TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & _u_u,
             TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & _v_d,
             TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & _v_u,
             TensorPointwise<Real, Symmetries::NONE, MDIM, 2> const & _proj_ud,
             Real const _W,
             Real const _Estar,
             TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & _Fstar_d,
             Real _chi,
             Real const _eta,
             Real const _kabs,
             Real const _kscat):
              pmb(_pmb),
              i(_i), j(_j), k(_k), ig(_ig),
              closure(_closure), gsl_solver_1d(_gsl_solver_1d),
              cdt(_cdt),
              alp(_alp), g_dd(_g_dd), g_uu(_g_uu), n_d(_n_d), n_u(_n_u),
              gamma_ud(_gamma_ud),
              u_d(_u_d), u_u(_u_u), v_d(_v_d), v_u(_v_u), proj_ud(_proj_ud), W(_W),
              Estar(_Estar), Fstar_d(_Fstar_d), chi(_chi),
              eta(_eta), kabs(_kabs), kscat(_kscat) {}
             MeshBlock * pmb;
             int const i;
             int const j;
             int const k;
             int const ig;
             closure_t closure;
             gsl_root_fsolver * gsl_solver_1d;
             Real const cdt;
             Real const alp;
             TensorPointwise<Real, Symmetries::SYM2, MDIM, 2>const & g_dd;
             TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_uu;
             TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & n_d;
             TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & n_u;
             TensorPointwise<Real, Symmetries::NONE, MDIM, 2> const & gamma_ud;
             TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_d;
             TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_u;
             TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & v_d;
             TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & v_u;
             TensorPointwise<Real, Symmetries::NONE, MDIM, 2> const & proj_ud;
             Real const W;
             Real const Estar;
             TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & Fstar_d;
             Real chi;
             Real const eta;
             Real const kabs;
             Real const kscat;
             
             Real E;
             TensorPointwise<Real, Symmetries::NONE, MDIM, 1> F_d;
             TensorPointwise<Real, Symmetries::NONE, MDIM, 1> F_u;
             TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> P_dd;
             TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> T_dd;
             Real J;
             TensorPointwise<Real, Symmetries::NONE, MDIM, 1> H_d;
             TensorPointwise<Real, Symmetries::NONE, MDIM, 1> S_d;
             Real Edot;
             TensorPointwise<Real, Symmetries::NONE, MDIM, 1> tS_d;
  };

  int prepare_closure(gsl_vector const * q, Params * p)
  {
    M1 * pm1 = p->pmb->pm1;
        
    p->E = std::max(gsl_vector_get(q, 0), 0.0);
    if (p->E < 0) {
      return GSL_EBADFUNC;
    }
    // ASK: how to deal with this    
    pm1->pack_F_d(-p->alp * p->n_u(1), -p->alp * p->n_u(2), -p->alp * p->n_u(3),
                  gsl_vector_get(q, 1), gsl_vector_get(q, 2), gsl_vector_get(q, 3),
                  p->F_d);
    tensor::contract(p->g_uu, p->F_d, p->F_u);
    
    pm1->calc_closure_pt(p->pmb, p->i, p->j, p->k, p->ig,
                         p->closure, p->gsl_solver_1d, p->g_dd, p->g_uu, p->n_d, p->W,
                         p->u_u, p->v_d, p->proj_ud, p->E, p->F_d,
                         &p->chi, p->P_dd);
    
    return GSL_SUCCESS;
  }

  int prepare_sources(gsl_vector const * q, Params * p)
  { 
    M1 * pm1 = p->pmb->pm1;
    pm1->assemble_rT(p->n_d, p->E, p->F_d, p->P_dd, p->T_dd);
    
    p->J = pm1->calc_J_from_rT(p->T_dd, p->u_u);
    pm1->calc_H_from_rT(p->T_dd, p->u_u, p->proj_ud, p->H_d);
      
    pm1->calc_rad_sources(p->eta, p->kabs, p->kscat, p->u_d, p->J, p->H_d, p->S_d);
    
    p->Edot = pm1->calc_rE_source(p->alp, p->n_u, p->S_d);
    pm1->calc_rF_source(p->alp, p->gamma_ud, p->S_d, p->tS_d);
    
    return GSL_SUCCESS;
  }

  int prepare(gsl_vector const * q, Params * p)
  {
    int ierr = prepare_closure(q, p);
    if (ierr != GSL_SUCCESS) {
      return ierr;
    }

    ierr = prepare_sources(q, p);
    if (ierr != GSL_SUCCESS) {
      return ierr;
    }

    return GSL_SUCCESS;
  }

  // Function to rootfind for
  //    f(q) = q - q^* - dt S[q]
  int impl_func_val(gsl_vector const * q, void * params, gsl_vector * f)
  {
    Params * p = reinterpret_cast<Params *>(params);
    int ierr = prepare(q, p);
    if (ierr != GSL_SUCCESS) return ierr;

#define EVALUATE_ZFUNC              \
    gsl_vector_set(f, 0, gsl_vector_get(q, 0) - p->Estar      - p->cdt * p->Edot); \
    gsl_vector_set(f, 1, gsl_vector_get(q, 1) - p->Fstar_d(1) - p->cdt * p->tS_d(1)); \
    gsl_vector_set(f, 2, gsl_vector_get(q, 2) - p->Fstar_d(2) - p->cdt * p->tS_d(2)); \
    gsl_vector_set(f, 3, gsl_vector_get(q, 3) - p->Fstar_d(3) - p->cdt * p->tS_d(3));

    EVALUATE_ZFUNC

    return GSL_SUCCESS;
  }

  // Jacobian of the implicit function
  int impl_func_jac(gsl_vector const * q, void * params, gsl_matrix * J)
  {
    Params * p = reinterpret_cast<Params *>(params);

    int ierr = prepare(q, p);
    if (ierr != GSL_SUCCESS) {
      return ierr;
    }

#define EVALUATE_ZJAC              \
    Real m_q[] = {p->E, p->F_d(1), p->F_d(2), p->F_d(3)};    \
    Real m_Fup[] = {p->F_u(0), p->F_u(1), p->F_u(2), p->F_u(3)}; \
    Real m_F2 = tensor::dot(p->F_u, p->F_d);         \
    Real m_chi = p->chi;             \
    Real m_kscat = p->kscat;             \
    Real m_kabs = p->kabs;             \
    Real m_vup[] = {p->v_u(0), p->v_u(1), p->v_u(2), p->v_u(3)}; \
    Real m_vdw[] = {p->v_d(0), p->v_d(1), p->v_d(2), p->v_d(3)}; \
    Real m_v2 = tensor::dot(p->v_u, p->v_d);         \
    Real m_W = p->W;               \
    Real m_alpha = p->alp;            \
    Real m_cdt = p->cdt;            \
    Real m_qstar[] = {p->Estar, p->Fstar_d(1), p->Fstar_d(2), p->Fstar_d(3)}; \
    __source_jacobian_low_level(m_q, m_Fup, m_F2, m_chi, m_kscat, m_kabs, \
        m_vup, m_vdw, m_v2, m_W, m_alpha, m_cdt, m_qstar, J);    

    EVALUATE_ZJAC;

    return GSL_SUCCESS;
  }

  // Function and Jacobian evaluation
  int impl_func_val_jac(gsl_vector const * q, void * params, gsl_vector * f, gsl_matrix * J)
  {
    Params * p = reinterpret_cast<Params *>(params);

    int ierr = prepare(q, p);
    if (ierr != GSL_SUCCESS) {
      return ierr;
    }

    EVALUATE_ZFUNC
    EVALUATE_ZJAC

    return GSL_SUCCESS;
  }

#undef EVALUATE_ZFUNC
#undef EVALUATE_ZJAC

} // namespace


//----------------------------------------------------------------------------------------
// Source update at one point

void M1::source_update_pt(
    MeshBlock * pmb,
    int const i,
    int const j,
    int const k,
    int const ig,
    closure_t closure_fun,
    gsl_root_fsolver * gsl_solver_1d,
    gsl_multiroot_fdfsolver * gsl_solver_nd,
    Real const cdt,
    Real const alp,
    TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_dd,
    TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_uu,
    TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & n_d,
    TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & n_u,
    TensorPointwise<Real, Symmetries::NONE, MDIM, 2> const & gamma_ud,
    TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_d,
    TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_u,
    TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & v_d,
    TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & v_u,
    TensorPointwise<Real, Symmetries::NONE, MDIM, 2> const & proj_ud,
    Real const W,
    Real const Eold,
    TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & Fold_d,
    Real const Estar,
    TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & Fstar_d,
    Real * chi,
    Real const eta,
    Real const kabs,
    Real const kscat,
    Real * Enew,
    TensorPointwise<Real, Symmetries::NONE, MDIM, 1> Fnew_d)
{
  Params p(pmb, i, j, k, ig,
           closure_fun, gsl_solver_1d, cdt, alp, g_dd, g_uu, n_d, n_u,
           gamma_ud, u_d, u_u, v_d, v_u, proj_ud, W, Estar, Fstar_d, *chi, eta,
           kabs, kscat);
        
  gsl_multiroot_function_fdf zfunc = {
      impl_func_val,
      impl_func_jac,
      impl_func_val_jac,
      4, &p};
  
  Real qold[] = {Eold, Fold_d(1), Fold_d(2), Fold_d(3)};
  gsl_vector_view xold = gsl_vector_view_array(qold, 4);
  
  // Initial guess for the solution
  Real q[4] = {*Enew, Fnew_d(1), Fnew_d(2), Fnew_d(3)};
  gsl_vector_view x = gsl_vector_view_array(q, 4);

  int ierr = gsl_multiroot_fdfsolver_set(gsl_solver_nd, &zfunc, &x.vector);
  int iter = 0;
  do {
    ierr = gsl_multiroot_fdfsolver_iterate(gsl_solver_nd);
    iter++;
    // The nonlinear solver is stuck: bailing out
    if (ierr == GSL_ENOPROG || ierr == GSL_ENOPROGJ) {
#ifdef BREAK_ON_ENOPROG
      break;
#else
      *Enew = Eold;
      Fnew_d(0) = Fold_d(0);
      Fnew_d(1) = Fold_d(1);
      Fnew_d(2) = Fold_d(2);
      Fnew_d(3) = Fold_d(3);
      return;
#endif
    } else if (ierr == GSL_EBADFUNC) {
    // NaNs or Infs are found, this should not have happened
      std::ostringstream msg;
      msg << "NaNs or Infs found in the implicit solve!";
      ATHENA_ERROR(msg);
    } else if (ierr != 0) {
      std::ostringstream msg;
      msg << "Unexpected error in "
	           "gsl_multirootroot_fdfsolver_iterate, error code " << ierr << std::endl;
      ATHENA_ERROR(msg);
    }
    ierr = gsl_multiroot_test_delta(gsl_solver_nd->dx, gsl_solver_nd->x,
				                            source_epsabs, source_epsrel);
  } while (ierr == GSL_CONTINUE && iter < source_maxiter);
  
  *Enew = gsl_vector_get(gsl_solver_nd->x, 0);
  Fnew_d(1) = gsl_vector_get(gsl_solver_nd->x, 1);
  Fnew_d(2) = gsl_vector_get(gsl_solver_nd->x, 2);
  Fnew_d(3) = gsl_vector_get(gsl_solver_nd->x, 3);
  // F_0 = g_0i F^i = beta_i F^i = beta^i F_i
  Fnew_d(0) = - alp*n_u(1)*Fnew_d(1)
              - alp*n_u(2)*Fnew_d(2)
              - alp*n_u(3)*Fnew_d(3);
  
  prepare_closure(gsl_solver_nd->x, &p);
  *chi = p.chi;
}
