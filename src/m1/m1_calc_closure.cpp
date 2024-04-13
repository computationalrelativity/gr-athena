// c++
#include <iostream>

// Athena++ headers
#include "m1.hpp"
#include "m1_macro.hpp"
#include "m1_utils.hpp"

// External libraries
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>

// ============================================================================
namespace M1::Closures {
// ============================================================================

// P_{i j} in Eq.(15) of [1] - works with densitized variables
//
void ClosureThin(M1 * pm1,
                 const Real weight,
                 const int ix_g,
                 const int ix_s,
                 AT_N_sym & sp_P_dd_,
                 const int k, const int j,
                 const int il, const int iu)
{
  AT_N_vec & sp_F_d  = pm1->lab.sp_F_d(ix_g,ix_s);
  AT_C_sca & sc_E    = pm1->lab.sc_E(  ix_g,ix_s);

  // point to scratch
  AT_C_sca & sc_norm2_F_ = pm1->scratch.sc_A_;

  Assemble::sp_norm2_(sc_norm2_F_,
                      sp_F_d,
                      pm1->geom.sp_g_dd,
                      k, j, il, iu);

  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    const Real F2 = sc_norm2_F_(i);
    const Real fac = (F2 > 0) ? sc_E(k,j,i) / F2
                              : 0.0;
    sp_P_dd_(a,b,i) = weight * fac * sp_F_d(a,k,j,i) * sp_F_d(b,k,j,i);
  }
}

void AddClosureThin(M1 * pm1,
                    const Real weight,
                    const int ix_g,
                    const int ix_s,
                    AT_N_sym & sp_P_dd)
{
  const int il = 0;
  const int iu = pm1->mbi.nn1-1;

  AT_N_sym & sp_P_dd_ = pm1->scratch.sp_sym_A_;

  M1_GLOOP2(k,j)
  {
    ClosureThin(pm1, weight, ix_g, ix_s, sp_P_dd_, k, j, il, iu);
    Assemble::ScratchAddToDense(sp_P_dd, sp_P_dd_, k, j, il, iu);
  }
}

// Function sets:
// lab_aux.sp_P_dd += wei * P_dd (thick)
void ClosureThick(M1 * pm1,
                  const Real weight,
                  const int ix_g,
                  const int ix_s,
                  AT_N_sym & sp_P_dd_,
                  const int k, const int j,
                  const int il, const int iu)
{

  AT_N_vec & sp_F_d = pm1->lab.sp_F_d(ix_g,ix_s);
  AT_C_sca & sc_E   = pm1->lab.sc_E(  ix_g,ix_s);

  AT_N_vec & sp_H_d = pm1->rad.sp_H_d(ix_g,ix_s);
  AT_C_sca & sc_J   = pm1->rad.sc_J(  ix_g,ix_s);

  AT_N_vec & sp_v_u = pm1->fidu.sp_v_u;
  AT_N_vec & sp_v_d = pm1->fidu.sp_v_d;
  AT_C_sca & sc_W   = pm1->fidu.sc_W;

  AT_N_sym & sp_g_dd = pm1->geom.sp_g_dd;

  // point to available scratches
  AT_C_sca & sc_dot_Fv_ = pm1->scratch.sc_A_;
  AT_C_sca & W2_        = pm1->scratch.sc_B_;

  // --------------------------------------------------------------------------
  Assemble::sc_dot_dense_sp_(sc_dot_Fv_,
                             sp_F_d,
                             sp_v_u,
                             k, j, il, iu);

  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    W2_(i) = SQR(sc_W(k,j,i));
  }

  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sc_J(k,j,i) = 3.0 / (2.0 * W2_(i) + 1.0) * (
      (2.0 * W2_(i) - 1.0) * sc_E(k,j,i) -
      2.0 * W2_(i) * sc_dot_Fv_(i)
    );
  }

  for (int a=0; a<N; ++a)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sp_H_d(a,k,j,i) = (
      1.0 / sc_W(k,j,i) * sp_F_d(a,k,j,i) +
      sc_W(k,j,i) * sp_v_d(a,k,j,i) / (2.0 * W2_(i) + 1) *
        (
          (4.0 * W2_(i) + 1) * sc_dot_Fv_(i) -
          4.0 * W2_(i) * sc_E(k,j,i)
        )
    );
  }

  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sp_P_dd_(a,b,i) = weight * (
      4.0 * ONE_3RD * W2_(i) * sc_J(k,j,i) *
      sp_v_d(a,k,j,i) * sp_v_d(b,k,j,i) +
      sc_W(k,j,i) * (
        sp_v_d(a,k,j,i) * sp_H_d(b,k,j,i) +
        sp_v_d(b,k,j,i) * sp_H_d(a,k,j,i)
      ) +
      ONE_3RD * sc_J(k,j,i) * sp_g_dd(a,b,k,j,i)
    );
  }

}

void AddClosureThick(M1 * pm1,
                     const Real weight,
                     const int ix_g,
                     const int ix_s,
                     AT_N_sym & sp_P_dd)
{
  const int il = 0;
  const int iu = pm1->mbi.nn1-1;

  AT_N_sym & sp_P_dd_ = pm1->scratch.sp_sym_B_;

  M1_GLOOP2(k,j)
  {
    ClosureThick(pm1, weight, ix_g, ix_s, sp_P_dd_, k, j, il, iu);
    Assemble::ScratchAddToDense(sp_P_dd, sp_P_dd_, k, j, il, iu);
  }
}
// ============================================================================
} // namespace M1::Closures
// ============================================================================

// ============================================================================
namespace M1::Closures::Minerbo {
// ============================================================================

// Eddington factor
inline Real chi(const Real xi)
{
  const Real xi2 = SQR(xi);
  return ONE_3RD + xi2 / 15.0 * (6.0 - 2.0 * xi + 6 * xi2);
}

// Required data during rootfinding procedure
struct DataRootfinder {
  AT_N_sym & sp_g_uu;

  AT_C_sca & sc_E;
  AT_N_vec & sp_F_d;
  AT_N_sym & sp_P_dd;
  AT_C_sca & sc_chi;
  AT_C_sca & sc_J;
  AT_C_sca & sc_H_t;
  AT_N_vec & sp_H_d;
  AT_C_sca & sc_W;
  AT_N_vec & sp_v_u;
  AT_N_vec & sp_v_d;

  // scratch
  AT_C_sca & sc_;
  AT_N_sym & sp_P_tn_dd_;
  AT_N_sym & sp_P_tk_dd_;

  int i, j, k;
};

// Function to find root of
Real R(Real xi, void *par)
{
  DataRootfinder * drf = reinterpret_cast<DataRootfinder*>(par);

  const int i = drf->i;
  const int j = drf->j;
  const int k = drf->k;

  // assemble P_dd
  drf->sc_chi(k,j,i) = chi(xi);

  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  {
    drf->sp_P_dd(a,b,k,j,i) = 0.5 * (
      (3.0 * drf->sc_chi(k,j,i) - 1.0) * drf->sp_P_tn_dd_(a,b,i) +
      3.0 * (1.0 - drf->sc_chi(k,j,i)) * drf->sp_P_tk_dd_(a,b,i)
    );
  }

  const Real W2 = SQR(drf->sc_W(k,j,i));
  Real dotFv = 0.0;

  for (int a=0; a<N; ++a)
  {
    dotFv += drf->sp_F_d(a,k,j,i) * drf->sp_v_u(a,k,j,i);
  }

  // assemble J
  drf->sc_J(k,j,i) = drf->sc_E(k,j,i) - 2.0 * dotFv;
  for (int a=0; a<N; ++a)
  for (int b=0; b<N; ++b)
  {
    drf->sc_J(k,j,i) -= (
      drf->sp_P_dd(a,b,k,j,i) *
      drf->sp_v_u(a,k,j,i) *
      drf->sp_v_u(b,k,j,i)
    );
  }

  drf->sc_J(k,j,i) = W2 * drf->sc_J(k,j,i);

  // assemble H_t
  drf->sc_H_t(k,j,i) = drf->sc_W(k,j,i) * (
    drf->sc_E(k,j,i) - drf->sc_J(k,j,i) - dotFv
  );

  // assemble H_d
  for (int a=0; a<N; ++a)
  {
    drf->sp_H_d(a,k,j,i) = (
      drf->sp_F_d(a,k,j,i) - drf->sc_J(k,j,i) * drf->sp_v_d(a,k,j,i)
    );

    for (int b=0; b<N; ++b)
    {
      drf->sp_H_d(a,k,j,i) -= (
        drf->sp_v_u(b,k,j,i) * drf->sp_P_dd(b,a,k,j,i)
      );
    }

    drf->sp_H_d(a,k,j,i) = drf->sc_W(k,j,i) * drf->sp_H_d(a,k,j,i);
  }

  // assemble sc_H_st
  Real sc_H2_st = -SQR(drf->sc_H_t(k,j,i));
  for (int a=0; a<N; ++a)
  for (int b=0; b<N; ++b)
  {
    sc_H2_st += (
      drf->sp_g_uu(a,b,k,j,i) *
      drf->sp_H_d(a,k,j,i) *
      drf->sp_H_d(b,k,j,i)
    );
  }

  // assemble R
  // const Real R_ = (std::abs(drf->sc_E(k,j,i)) > 0)
  //   ? (SQR(xi) * SQR(drf->sc_J(k,j,i)) - sc_H2_st) / drf->sc_E(k,j,i)
  //   : 0.0;

  const Real R_ = (SQR(xi * drf->sc_J(k,j,i)) - sc_H2_st);

  return R_;
}

void AddClosure(M1 * pm1,
                const Real weight,
                const int ix_g,
                const int ix_s,
                AT_N_sym & sp_P_dd)
{
  // Admissible ranges for xi entering Eddington factor chi(xi)
  const Real xi_min = 0.0;
  const Real xi_max = 1.0;

  const int il = 0;
  const int iu = pm1->mbi.nn1-1;

  AT_N_sym & sp_g_dd = pm1->geom.sp_g_dd;
  AT_N_sym & sp_g_uu = pm1->geom.sp_g_uu;

  // point to scratches
  AT_N_sym & sp_P_tn_dd_ = pm1->scratch.sp_sym_A_;
  AT_N_sym & sp_P_tk_dd_ = pm1->scratch.sp_sym_B_;

  // initialize struct for root-finding
  DataRootfinder drf {
    sp_g_uu,
    pm1->lab.sc_E(  ix_g,ix_s),
    pm1->lab.sp_F_d(ix_g,ix_s),
    sp_P_dd,
    pm1->lab_aux.sc_chi(ix_g,ix_s),
    pm1->rad.sc_J(  ix_g,ix_s),
    pm1->rad.sc_H_t(ix_g,ix_s),
    pm1->rad.sp_H_d(ix_g,ix_s),
    pm1->fidu.sc_W,
    pm1->fidu.sp_v_u,
    pm1->fidu.sp_v_d,
    pm1->scratch.sc_A_,
    sp_P_tn_dd_,
    sp_P_tk_dd_
  };

  // setup GSL ----------------------------------------------------------------
  gsl_set_error_handler_off();

  gsl_function R_;
  R_.function = &R;
  R_.params = &drf;

  gsl_root_fsolver * gsl_solver = gsl_root_fsolver_alloc(
    gsl_root_fsolver_brent
  );

  auto gsl_err_kill = [&](const int status)
  {
    std::ostringstream msg;
    msg << "M1::Closures::Minerbo::AddClosure unexpected error: ";
    msg << status;

    std::cout << msg.str().c_str() << std::endl;

    ATHENA_ERROR(msg);
  };

  auto gsl_err_warn = [&]()
  {
    std::ostringstream msg;
    msg << "M1::Closures::Minerbo::AddClosure maxiter=";
    msg << pm1->opt.max_iter_C << " exceeded";
    std::cout << msg.str().c_str() << std::endl;
  };
  // --------------------------------------------------------------------------

  // main loop
  // int iter_tot = 0;

  M1_GLOOP2(k,j)
  {
    Closures::ClosureThin( pm1, 1.0, ix_g, ix_s, sp_P_tn_dd_, k, j, il, iu);
    Closures::ClosureThick(pm1, 1.0, ix_g, ix_s, sp_P_tk_dd_, k, j, il, iu);

    M1_GLOOP1(i)
    {
      drf.i = i;
      drf.j = j;
      drf.k = k;

      int status = gsl_root_fsolver_set(gsl_solver, &R_, xi_min, xi_max);

      switch (status)
      {
        case (GSL_EINVAL):  // bracketing failed (revert to thin closure)
        {
          Assemble::ScratchToDense(sp_P_dd, sp_P_tn_dd_, k, j, i, i);
          break;
        }
        case (0):
        {
          // root-finding loop
          Real loc_xi_min = xi_min;
          Real loc_xi_max = xi_max;

          status = GSL_CONTINUE;
          for (int iter=1;
               iter<=pm1->opt.max_iter_C && status == GSL_CONTINUE;
               ++iter)
          {
            status = gsl_root_fsolver_iterate(gsl_solver);

            if (status != GSL_SUCCESS)
            {
              break;
            }
            else if (status)
            {
              sp_P_tn_dd_.array().print_all();
              sp_P_tk_dd_.array().print_all();

              std::cout << "E" << std::endl;;
              std::cout << drf.sc_E(k,j,i) << std::endl;

              std::cout << "F_d" << std::endl;;
              std::cout << drf.sp_F_d(0,k,j,i) << std::endl;
              std::cout << drf.sp_F_d(1,k,j,i) << std::endl;
              std::cout << drf.sp_F_d(2,k,j,i) << std::endl;

              gsl_err_kill(status);
            }

            loc_xi_min = gsl_root_fsolver_x_lower(gsl_solver);
            loc_xi_max = gsl_root_fsolver_x_upper(gsl_solver);

            status = gsl_root_test_interval(
              loc_xi_min, loc_xi_max, pm1->opt.eps_C, 0
            );

            // if (iter > 0.5 * static_cast<Real>(pm1->opt.max_iter_C))
            // {
            //   std::cout << "iter slow w/" << std::endl;
            //   std::cout << pm1->lab_aux.sc_chi(ix_g,ix_s)(k,j,i) << std::endl;

            //   std::cout << R_.function(xi_min, &drf) << std::endl;
            //   std::cout << R_.function(gsl_root_fsolver_root(gsl_solver), &drf) << std::endl;

            // }

            // if (status == GSL_SUCCESS)
            // {
            //   std::cout << iter << std::endl;
            // }

            // iter_tot++;
          }

          if (status != GSL_SUCCESS)
          {
            gsl_err_warn();
          }

          // Final call of R to update sp_P_dd with new root info.
          R_.function(gsl_root_fsolver_root(gsl_solver), &drf);

          break;
        }
        default:
        {
            std::cout << "sp_P_tn_dd_" << std::endl;;
            std::cout << sp_P_tn_dd_(0,0,i) << std::endl;
            std::cout << sp_P_tn_dd_(0,1,i) << std::endl;
            std::cout << sp_P_tn_dd_(0,2,i) << std::endl;
            std::cout << sp_P_tn_dd_(1,1,i) << std::endl;
            std::cout << sp_P_tn_dd_(1,2,i) << std::endl;
            std::cout << sp_P_tn_dd_(2,2,i) << std::endl;

            std::cout << "sp_P_tk_dd_" << std::endl;;
            std::cout << sp_P_tk_dd_(0,0,i) << std::endl;
            std::cout << sp_P_tk_dd_(0,1,i) << std::endl;
            std::cout << sp_P_tk_dd_(0,2,i) << std::endl;
            std::cout << sp_P_tk_dd_(1,1,i) << std::endl;
            std::cout << sp_P_tk_dd_(1,2,i) << std::endl;
            std::cout << sp_P_tk_dd_(2,2,i) << std::endl;

            std::cout << "E" << std::endl;;
            std::cout << drf.sc_E(k,j,i) << std::endl;

            std::cout << "F_d" << std::endl;;
            std::cout << drf.sp_F_d(0,k,j,i) << std::endl;
            std::cout << drf.sp_F_d(1,k,j,i) << std::endl;
            std::cout << drf.sp_F_d(2,k,j,i) << std::endl;
          gsl_err_kill(status);
        }
      }
    }
  }

  // const int nn = pm1->mbi.nn1 * pm1->mbi.nn2 * pm1->mbi.nn3;
  // std::cout << static_cast<Real>(iter_tot) / nn << std::endl;

  // cleanup ------------------------------------------------------------------
  gsl_root_fsolver_free(gsl_solver);
  // TODO: bug - this can't be deactivated properly with OMP
  // gsl_set_error_handler(NULL);  // restore default handler
}

// ============================================================================
} // namespace M1::Closures::Minerbo
// ============================================================================


// ============================================================================
namespace M1 {
// ============================================================================

// ----------------------------------------------------------------------------
// Computes the closure on a mesh block
void M1::CalcClosure(AthenaArray<Real> & u)
{
  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    pm1->lab_aux.sp_P_dd(ix_g,ix_s).ZeroClear();

    switch (opt.closure_variety)
    {
      case (opt_closure_variety::thin):
      {
        Closures::AddClosureThin(this,
                                 1.0, ix_g, ix_s,
                                 lab_aux.sp_P_dd(ix_g,ix_s));
        lab_aux.sc_chi(ix_g,ix_s).Fill(1.0);
        break;
      }
      case (opt_closure_variety::thick):
      {
        Closures::AddClosureThick(this,
                                  1.0, ix_g, ix_s,
                                  lab_aux.sp_P_dd(ix_g,ix_s));
        lab_aux.sc_chi(ix_g,ix_s).Fill(ONE_3RD);
        break;
      }
      case (opt_closure_variety::Minerbo):
      {
        Closures::Minerbo::AddClosure(this,
                                      1.0, ix_g, ix_s,
                                      lab_aux.sp_P_dd(ix_g,ix_s));
        break;
      }
      default:
      {
        assert(false);
        std::exit(0);
      }
    }
  }

  return;
}

// ----------------------------------------------------------------------------
// Map (closed) Eulerian fields (E, F_d, P_dd) to (J, H_d)
void M1::CalcFiducialFrame(AthenaArray<Real> & u)
{
  vars_Lab U_n { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u, U_n);

  AT_C_sca & sc_W = pm1->fidu.sc_W;
  AT_N_vec & sp_v_u = pm1->fidu.sp_v_u;
  AT_N_vec & sp_v_d = pm1->fidu.sp_v_d;

  // point to scratches
  AT_C_sca & dotFv_ = pm1->scratch.sc_A_;
  AT_C_sca & sc_W2_ = pm1->scratch.sc_B_;

  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    AT_C_sca & sc_E    = U_n.sc_E(  ix_g,ix_s);
    AT_N_vec & sp_F_d  = U_n.sp_F_d(ix_g,ix_s);
    AT_N_sym & sp_P_dd = lab_aux.sp_P_dd(ix_g,ix_s);

    AT_C_sca & sc_H_t = pm1->rad.sc_H_t(ix_g,ix_s);
    AT_N_vec & sp_H_d = pm1->rad.sp_H_d(ix_g,ix_s);
    AT_C_sca & sc_J   = pm1->rad.sc_J(  ix_g,ix_s);

    M1_GLOOP2(k,j)
    {
      dotFv_.ZeroClear();

      for (int a=0; a<N; ++a)
      M1_GLOOP1(i)
      {
        dotFv_(i) += sp_F_d(a,k,j,i) * sp_v_u(a,k,j,i);
      }

      M1_GLOOP1(i)
      {
        sc_W2_(i) = SQR(sc_W(k,j,i));
      }

      // assemble J
      M1_GLOOP1(i)
      {
        sc_J(k,j,i) = (
          sc_E(k,j,i) - 2.0 * dotFv_(i)
        );
      }

      for (int a=0; a<N; ++a)
      for (int b=0; b<N; ++b)
      M1_GLOOP1(i)
      {
        sc_J(k,j,i) += (
          sp_P_dd(a,b,k,j,i) *
          sp_v_u(a,k,j,i) *
          sp_v_u(b,k,j,i)
        );
      }

      M1_GLOOP1(i)
      {
        sc_J(k,j,i) = sc_W2_(i) * sc_J(k,j,i);
      }

      // assemble H_t
      M1_GLOOP1(i)
      {
        sc_H_t(k,j,i) = sc_W(k,j,i) * (
          sc_E(k,j,i) - sc_J(k,j,i) - dotFv_(i)
        );
      }

      // assemble H_d
      for (int a=0; a<N; ++a)
      {
        M1_GLOOP1(i)
        {
          sp_H_d(a,k,j,i) = (
            sp_F_d(a,k,j,i) - sc_J(k,j,i) * sp_v_d(a,k,j,i)
          );
        }

        for (int b=0; b<N; ++b)
        M1_GLOOP1(i)
        {
          sp_H_d(a,k,j,i) -= (
            sp_v_u(b,k,j,i) * sp_P_dd(b,a,k,j,i)
          );
        }

        M1_GLOOP1(i)
        {
          sp_H_d(a,k,j,i) = sc_W(k,j,i) * sp_H_d(a,k,j,i);
        }
      }
    }
  }
  return;
}


// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//