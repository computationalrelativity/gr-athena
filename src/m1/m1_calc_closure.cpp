// c++
#include <iostream>

// Athena++ headers
#include "m1.hpp"
#include "m1_calc_closure.hpp"
#include "m1_containers.hpp"
#include "m1_utils.hpp"

// External libraries
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>

// ============================================================================
namespace M1::Closures {
// ============================================================================

void InfoDump(M1 * pm1, const int ix_g, const int ix_s,
              const int k, const int j, const int i)
{
  std::cout << "(k,j,i) " << k << "," << j << "," << i << "\n";
  std::cout << "sc_W "    << pm1->fidu.sc_W(  k,j,i) << "\n";
  std::cout << "sc_xi "   << pm1->lab_aux.sc_xi(ix_g,ix_s)(k,j,i) << "\n";
  std::cout << "sc_chi "  << pm1->lab_aux.sc_chi(ix_g,ix_s)(k,j,i) << "\n";
  std::cout << "sc_E "    << pm1->lab.sc_E(ix_g,ix_s)(k,j,i) << "\n";

  std::cout << "sp_F_d:" << "\n";

  for (int a=0; a<N; ++a)
  {
    std::cout << "a " << a << " "
              << pm1->lab.sp_F_d(ix_g,ix_s)(a,k,j,i) << "\n";
  }

  std::cout << "sc_J:" << pm1->rad.sc_J(ix_g,ix_s)(k,j,i) << "\n";
  std::cout << "sc_H_t:" << pm1->rad.sc_H_t(ix_g,ix_s)(k,j,i) << "\n";

  std::cout << "sp_H_d:" << "\n";

  for (int a=0; a<N; ++a)
  {
    std::cout << "a " << a << " "
              << pm1->rad.sp_H_d(ix_g,ix_s)(a,k,j,i) << "\n";
  }

  std::cout << "sp_P_dd:" << "\n";

  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  {
    std::cout << "(a,b) " << a << "," << b << " "
              << pm1->lab_aux.sp_P_dd(ix_g,ix_s)(a,b,k,j,i) << "\n";
  }
}

// sliced operations ----------------------------------------------------------

// P_{i j} in Eq.(15) of [1] - works with densitized variables
//
void ClosureThin(M1 * pm1,
                 AT_N_sym & sp_P_dd_,
                 const AT_C_sca & sc_E,
                 const AT_N_vec & sp_F_d,
                 const int k, const int j, const int i)
{
  const Real nF2 = Assemble::sp_norm2__(sp_F_d, pm1->geom.sp_g_uu, k, j, i);

  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  {
    const Real fac = (nF2 > 0) ? sc_E(k,j,i) / nF2
                               : 0.0;
    sp_P_dd_(a,b,i) = fac * sp_F_d(a,k,j,i) * sp_F_d(b,k,j,i);
  }
}

// Function sets:
// lab_aux.sp_P_dd += wei * P_dd (thick)
void ClosureThick(M1 * pm1,
                  AT_N_sym & sp_P_dd_,
                  const Real dotFv,
                  const AT_C_sca & sc_E,
                  const AT_N_vec & sp_F_d,
                  const int k, const int j, const int i)
{
  AT_N_vec & sp_v_d = pm1->fidu.sp_v_d;
  AT_C_sca & sc_W   = pm1->fidu.sc_W;

  AT_N_sym & sp_g_dd = pm1->geom.sp_g_dd;

  // --------------------------------------------------------------------------

  const Real W    = sc_W(k,j,i);
  const Real oo_W = 1.0 / W;
  const Real W2   = SQR(W);

  const Real J_tk = 3.0 / (2.0 * W2 + 1.0) * (
    (2.0 * W2 - 1.0) * sc_E(k,j,i) - 2.0 * W2 * dotFv
  );

  const Real fac_H_tk = W /  (2.0 * W2 + 1.0) * (
    (4.0 * W2 + 1.0) * dotFv - 4.0 * W2 * sc_E(k,j,i)
  );

  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  {
    const Real H_a_tk = oo_W * sp_F_d(a,k,j,i) +
                        fac_H_tk * sp_v_d(a,k,j,i);
    const Real H_b_tk = oo_W * sp_F_d(b,k,j,i) +
                        fac_H_tk * sp_v_d(b,k,j,i);

    sp_P_dd_(a,b,i) = (
      4.0 * ONE_3RD * W2 * J_tk * sp_v_d(a,k,j,i) * sp_v_d(b,k,j,i) +
      W * (sp_v_d(a,k,j,i) * H_b_tk +
           sp_v_d(b,k,j,i) * H_a_tk) +
      ONE_3RD * J_tk * sp_g_dd(a,b,k,j,i)
    );
  }
}

void ClosureThick(M1 * pm1,
                  AT_N_sym & sp_P_dd_,
                  const AT_C_sca & sc_E,
                  const AT_N_vec & sp_F_d,
                  const int k, const int j, const int i)
{
  const Real dotFv = Assemble::sc_dot_dense_sp__(
    sp_F_d,
    pm1->fidu.sp_v_u,
    k, j, i
  );
  ClosureThick(pm1, sp_P_dd_, dotFv, sc_E, sp_F_d, k, j, i);
}

// ============================================================================
} // namespace M1::Closures
// ============================================================================

// ============================================================================
namespace M1::Closures::Minerbo {
// ============================================================================

Real R(Real xi, void *par)
{
  DataRootfinder * drf = reinterpret_cast<DataRootfinder*>(par);

  const int i = drf->i;
  const int j = drf->j;
  const int k = drf->k;

  // assemble P_dd
  drf->sc_xi( k,j,i) = xi;
  drf->sc_chi(k,j,i) = chi(xi);

  sp_P_dd__(
    drf->sp_P_dd, drf->sc_chi, drf->sp_P_tn_dd_, drf->sp_P_tk_dd_,
    k, j, i
  );

  const Real W  = drf->sc_W(k,j,i);
  const Real W2 = SQR(W);

  drf->dotFv = Assemble::sc_dot_dense_sp__(
    drf->sp_F_d,
    drf->sp_v_u,
    k, j, i
  );

  // assemble J
  drf->sc_J(k,j,i) = Assemble::sc_J__(
    W2, drf->dotFv, drf->sc_E, drf->sp_v_u, drf->sp_P_dd,
    k, j, i
  );

  // drf->sc_J(k,j,i) = std::max(drf->sc_J(k,j,i), 1e-14);

  // assemble H_t
  drf->sc_H_t(k,j,i) = Assemble::sc_H_t__(
    W, drf->dotFv, drf->sc_E, drf->sc_J,
    k, j, i
  );

  // assemble H_d
  Assemble::sp_H_d__(drf->sp_H_d, W, drf->sc_J, drf->sp_F_d,
                     drf->sp_v_d, drf->sp_v_u, drf->sp_P_dd,
                     k, j, i);

  // assemble sc_H_st
  Real sc_H2_st = Assemble::sc_H2_st__(
    drf->sc_H_t, drf->sp_H_d, drf->sp_g_uu,
    k, j, i
  );

  // assemble R
  // const Real R_ = (std::abs(drf->sc_E(k,j,i)) > 0)
  //   ? (SQR(xi) * SQR(drf->sc_J(k,j,i)) - sc_H2_st) / drf->sc_E(k,j,i)
  //   : 0.0;

  const Real R_ = (SQR(xi * drf->sc_J(k,j,i)) - sc_H2_st);

  drf->Z_xi_im2 = drf->Z_xi_im1;
  drf->Z_xi_im1 = drf->Z_xi_i;
  drf->Z_xi_i = R_;

  drf->xi_im2 = drf->xi_im1;
  drf->xi_im1 = drf->xi_i;
  drf->xi_i   = xi;

  return R_;
}

Real dR(Real xi, void *par)
{
  DataRootfinder * drf = reinterpret_cast<DataRootfinder*>(par);

  const int i = drf->i;
  const int j = drf->j;
  const int k = drf->k;

  const Real W  = drf->sc_W(k,j,i);
  const Real W2 = SQR(W);

  // derivative of Z_xi
  Real dJ (0);
  Real dHH (0);

  const Real J  = drf->sc_J(  k,j,i);
  const Real Hn = drf->sc_H_t(k,j,i);

  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  {
    drf->sp_dP_dd_(a,b,i) = 3.0 / 5.0 * (
      drf->sp_P_tn_dd_(a,b,i) - drf->sp_P_tk_dd_(a,b,i)
    ) * xi * (2.0 - xi + 4.0 * SQR(xi));
  }

  for (int a=0; a<N; ++a)
  for (int b=0; b<N; ++b)
  {
    dJ += drf->sp_v_u(a,k,j,i) *
          drf->sp_v_u(b,k,j,i) *
          drf->sp_dP_dd_(a,b,i);
  }

  dJ = W * dJ;

  const Real dHn = -W * dJ;

  for (int a=0; a<N; ++a)
  {
    drf->sp_dH_d_(a,i) = drf->sp_v_d(a,k,j,i) * dJ;
    for (int b=0; b<N; ++b)
    {
      drf->sp_dH_d_(a,i) += drf->sp_v_u(b,k,j,i) * drf->sp_dP_dd_(a,b,i);
    }
    drf->sp_dH_d_(a,i) = -W * drf->sp_dH_d_(a,i);
  }

  for (int a=0; a<N; ++a)
  for (int b=0; b<N; ++b)
  {
    dHH += drf->sp_g_uu(a,b,k,j,i) *
           drf->sp_dH_d_(a,i) *
           drf->sp_H_d(b,k,j,i);
  }

  const Real dR_ = 2 * xi * SQR(J) + 2 * J * SQR(xi) * dJ + 2 * Hn * dHn - dHH;
  drf->dZ_xi_im2 = drf->Z_xi_im1;
  drf->dZ_xi_im1 = drf->Z_xi_i;
  drf->dZ_xi_i = dR_;

  return dR_;
}

void RdR(Real xi, void *par, Real *r, Real *dr)
{
  *r  = R(xi, par);
  *dr = dR(xi, par);
}

void AddClosureP(M1 * pm1,
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

  // fiducial quantities
  AT_C_sca & sc_W   = pm1->fidu.sc_W;
  AT_N_vec & sp_v_u = pm1->fidu.sp_v_u;
  AT_N_vec & sp_v_d = pm1->fidu.sp_v_d;

  AT_C_sca & sc_E   = pm1->lab.sc_E(  ix_g,ix_s);
  AT_N_vec & sp_F_d = pm1->lab.sp_F_d(ix_g,ix_s);
  AT_C_sca & sc_chi = pm1->lab_aux.sc_chi(ix_g,ix_s);
  AT_C_sca & sc_xi  = pm1->lab_aux.sc_xi( ix_g,ix_s);

  AT_C_sca & sc_J   = pm1->rad.sc_J(  ix_g,ix_s);
  AT_C_sca & sc_H_t = pm1->rad.sc_H_t(ix_g,ix_s);
  AT_N_vec & sp_H_d = pm1->rad.sp_H_d(ix_g,ix_s);

  // required geometric quantities
  AT_N_sym & sp_g_dd = pm1->geom.sp_g_dd;
  AT_N_sym & sp_g_uu = pm1->geom.sp_g_uu;

  // point to scratches
  AT_N_vec & sp_dH_d_    = pm1->scratch.sp_vec_A_;
  AT_N_sym & sp_P_tn_dd_ = pm1->scratch.sp_sym_A_;
  AT_N_sym & sp_P_tk_dd_ = pm1->scratch.sp_sym_B_;
  AT_N_sym & sp_dP_dd_   = pm1->scratch.sp_sym_C_;

  // initialize struct for root-finding
  DataRootfinder drf {
    sp_g_uu,
    sc_E,
    sp_F_d,
    sp_P_dd,
    sc_chi,
    sc_xi,
    sc_J,
    sc_H_t,
    sp_H_d,
    pm1->fidu.sc_W,
    pm1->fidu.sp_v_u,
    pm1->fidu.sp_v_d,
    sp_dH_d_,
    sp_P_tn_dd_,
    sp_P_tk_dd_,
    sp_dP_dd_
  };

  std::array<Real, 1> iI_xi;

  // main loop ----------------------------------------------------------------
  int iter_tot = 0;

  M1_GLOOP2(k,j)
  {
    M1_GLOOP1(i)
    {
      const Real dotFv = Assemble::sc_dot_dense_sp__(sp_F_d, sp_v_u, k, j, i);

      Closures::ClosureThin( pm1, sp_P_tn_dd_, sc_E, sp_F_d, k, j, i);
      Closures::ClosureThick(pm1, sp_P_tk_dd_, dotFv, sc_E, sp_F_d, k, j, i);


      const int iter_max_C = pm1->opt.max_iter_C;
      int pit = 0;     // iteration counter
      int rit = 0;     // restart counter
      const int iter_max_R = pm1->opt.max_iter_C_rst;  // max restarts
      Real w_opt = pm1->opt.w_opt_ini_C;  // underrelaxation factor
      Real e_C_abs_tol = pm1->opt.eps_C;
      // maximum error amplification factor between iters.
      Real fac_PA = pm1->opt.fac_amp_C;

      // retain values for potential restarts
      iI_xi[0] = sc_xi(k,j,i);

      Real e_abs_old = std::numeric_limits<Real>::infinity();
      Real e_abs_cur = 0;

      // solver loop ----------------------------------------------------------
      drf.i = i;
      drf.j = j;
      drf.k = k;

      const Real W = sc_W(k,j,i);
      const Real oo_W = 1.0 / W;
      const Real W2   = SQR(W);

      do
      {
        pit++;

        // Check limits
        // const Real xi_old = sc_xi(k,j,i);

        // if ((pit == 1) && (rit == 0))
        // {
        //   const Real err_l = std::abs(R(xi_min, &drf));
        //   const Real err_r = std::abs(R(xi_max, &drf));
        //   const Real err_m = std::abs(R(xi_old, &drf));

        //   Real xi = (err_l < err_r) ? xi_min : xi_max;
        //   xi = (err_m < std::min(err_l, err_r)) ? xi_old : xi;

        //   if (std::min(err_m, std::min(err_l, err_r)) < e_C_abs_tol)
        //   {
        //     e_abs_cur = std::min(err_m, std::min(err_l, err_r));
        //     std::abs(R(xi, &drf));
        //     break;
        //   }
        // }

        // const Real xi = xi_old;
        // sc_E(k,j,i) = std::max(sc_E(k,j,i), 1e-14);

        Real Z_xi = R(sc_xi(k,j,i), &drf);

        // if (std::abs(Z_xi) < e_C_abs_tol)
        // {
        //   break;
        // }

        Real sc_xi_can = sc_xi(k,j,i) - w_opt * Z_xi;
        // enforce non-negative values
        // sc_xi(k,j,i) = std::min(std::max(sc_xi_can, xi_min), xi_max);
        sc_xi(k,j,i) = sc_xi_can;

        if (sc_xi_can < xi_min)
        {
          e_abs_cur = std::abs(xi_max - drf.xi_im1);
          e_abs_old = e_abs_cur;

          sc_xi_can = xi_min;
          R(sc_xi_can, &drf);
          pit = 0;
        }
        else if (sc_xi_can > xi_max)
        {
          e_abs_cur = std::abs(xi_max - drf.xi_im1);
          e_abs_old = e_abs_cur;

          // e_abs_cur = std::numeric_limits<Real>::infinity();
          // e_abs_old = e_abs_cur;

          sc_xi_can = xi_max;
          R(sc_xi_can, &drf);
          pit = 0;
        }
        else
        {
          sc_xi(k,j,i) = sc_xi_can;
          e_abs_cur = std::abs(drf.xi_i - drf.xi_im1);
        }


        // e_abs_cur = std::abs(Z_xi);
        // e_abs_cur = std::abs(1 - (drf.xi_i + 1) / (drf.xi_im1 + 1));
        // e_abs_cur = std::abs(drf.xi_i - drf.xi_im2);

        // if((sc_xi_can < xi_min)||(sc_xi_can > xi_max))
        // {
        //   sc_xi(k,j,i) = std::min(std::max(sc_xi_can, xi_min), xi_max);
        //   R(sc_xi(k,j,i), &drf);
        //   e_abs_cur = 0;
        //   break;
        // }

        // if (pit > 1)
        // if (std::abs(1 - (drf.xi_i + 1) / (drf.xi_im1 + 1)) < 1e-5)
        // {
        //   e_abs_cur = 0;
        //   break;
        // }

        // if (pit > 3)
        // {
        //   const Real p0 = drf.xi_im1;
        //   const Real p1 = drf.xi_i;
        //   const Real p2 = sc_xi(k,j,i);

        //   const Real pe = p0-SQR(p1-p0)/(p2-2*p1+p0);
        //   sc_xi(k,j,i) = pe;

        // }

        // e_abs_cur = std::abs(1 - (drf.xi_i + 1) / (drf.xi_im1 + 1));

        e_abs_cur = std::abs(drf.xi_i - drf.xi_im1);

        // if (std::abs(1 - (drf.xi_i + 1) / (drf.xi_im1 + 1)) < 1e-5)
        // {
        //   e_abs_cur = 0;
        //   break;
        // }


        if ((e_abs_cur > fac_PA * e_abs_old))
        {
          // halve underrelaxation and recover old values
          w_opt = w_opt / 2;
          sc_xi(k,j,i) = iI_xi[0];

          // restart iteration
          e_abs_old = std::numeric_limits<Real>::infinity();
          pit = 0;
          rit++;

          if (rit > iter_max_R)
          {
            std::ostringstream msg;
            msg << "M1::Closures::Minerbo::AddClosureP max restarts exceeded.";
            std::cout << msg.str().c_str() << "\n";
            std::cout << e_abs_cur << "\n";
            std::cout << sc_xi_can << "\n";
            std::cout << Z_xi << "\n";
            std::cout << w_opt << "\n";
            std::cout << "more info: \n";
            std::cout << drf.xi_i << "\n";
            std::cout << drf.xi_im1 << "\n";
            std::cout << drf.xi_im2 << "\n";
            std::cout << "Z_xi: \n";
            std::cout << drf.Z_xi_i << "\n";
            std::cout << drf.Z_xi_im1 << "\n";
            std::cout << drf.Z_xi_im2 << "\n";
            Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);
            std::exit(0);
          }
        }
        else
        {
          e_abs_old = e_abs_cur;
          sc_chi(k,j,i) = chi(sc_xi(k,j,i));

          /*
          if ((sc_xi(k,j,i) < xi_min))
          {
            sc_xi(k,j,i) = xi_min;
            sc_chi(k,j,i) = chi(sc_xi(k,j,i));

            for (int a=0; a<N; ++a)
            for (int b=a; b<N; ++b)
            {
              sp_P_dd(a,b,k,j,i) = sp_P_tk_dd_(a,b,i);
            }
          }
          else if ((sc_xi(k,j,i) > xi_max))
          {
            sc_xi(k,j,i) = xi_max;
            sc_chi(k,j,i) = chi(sc_xi(k,j,i));

            for (int a=0; a<N; ++a)
            for (int b=a; b<N; ++b)
            {
              sp_P_dd(a,b,k,j,i) = sp_P_tn_dd_(a,b,i);
            }
          }
          */


          if (pm1->opt.reset_thin)
          {
            if ((sc_xi(k,j,i) < xi_min) || (sc_xi(k,j,i) > xi_max))
            {
              sc_xi(k,j,i) = xi_max;
              sc_chi(k,j,i) = chi(sc_xi(k,j,i));

              for (int a=0; a<N; ++a)
              for (int b=a; b<N; ++b)
              {
                sp_P_dd(a,b,k,j,i) = sp_P_tn_dd_(a,b,i);
              }

              e_abs_cur = 0.0; // forces iter breakout
            }
          }

        }

      } while ((pit < iter_max_C) && (e_abs_cur >= e_C_abs_tol));



      if ((sc_xi(k,j,i) < xi_min) || sc_xi(k,j,i) > xi_max)
      {
        std::cout << "M1::Closures::Minerbo::AddClosureP:\n";
        std::cout << "outside [0,1] \n";
        std::cout << "Tol. not achieved: (pit,rit,e_abs_cur) ";
        std::cout << pit << "," << rit << "," << e_abs_cur << "\n";
        std::cout << e_abs_cur << "\n";
        std::cout << w_opt << "\n";
        std::cout << "more info: \n";
        std::cout << drf.xi_i << "\n";
        std::cout << drf.xi_im1 << "\n";
        std::cout << drf.xi_im2 << "\n";
        std::cout << 1 - (drf.xi_i + 1) / (drf.xi_im1 + 1) << "\n";
        std::cout << "Z_xi: \n";
        std::cout << drf.Z_xi_i << "\n";
        std::cout << drf.Z_xi_im1 << "\n";
        std::cout << drf.Z_xi_im2 << "\n";

        Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);

        std::cout << sc_xi(k,j,i) << "\n";
        std::cout << iI_xi[0] << "\n";

        std::exit(0);
      }


      iter_tot += pit;

      if ((e_abs_cur > e_C_abs_tol) && pm1->opt.verbose_iter_C)
      {
        std::cout << "M1::Closures::Minerbo::AddClosureP:\n";
        std::cout << "Tol. not achieved: (pit,rit,e_abs_cur) ";
        std::cout << pit << "," << rit << "," << e_abs_cur << "\n";
        std::cout << k << "," << j << "," << i << "\n";
        std::cout << e_abs_cur << "\n";
        std::cout << w_opt << "\n";
        std::cout << "more info: \n";
        std::cout << drf.xi_i << "\n";
        std::cout << drf.xi_im1 << "\n";
        std::cout << drf.xi_im2 << "\n";
        std::cout << 1 - (drf.xi_i + 1) / (drf.xi_im1 + 1) << "\n";
        std::cout << "Z_xi: \n";
        std::cout << drf.Z_xi_i << "\n";
        std::cout << drf.Z_xi_im1 << "\n";
        std::cout << drf.Z_xi_im2 << "\n";

        Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);
        std::exit(0);
      }
    }


  }

  // const int nn = pm1->mbi.nn1 * pm1->mbi.nn2 * pm1->mbi.nn3;
  // std::cout << static_cast<Real>(iter_tot) / nn << std::endl;

}

void AddClosureN(M1 * pm1,
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

  // fiducial quantities
  AT_C_sca & sc_W   = pm1->fidu.sc_W;
  AT_N_vec & sp_v_u = pm1->fidu.sp_v_u;
  AT_N_vec & sp_v_d = pm1->fidu.sp_v_d;

  AT_C_sca & sc_E   = pm1->lab.sc_E(  ix_g,ix_s);
  AT_N_vec & sp_F_d = pm1->lab.sp_F_d(ix_g,ix_s);
  AT_C_sca & sc_chi = pm1->lab_aux.sc_chi(ix_g,ix_s);
  AT_C_sca & sc_xi  = pm1->lab_aux.sc_xi( ix_g,ix_s);

  AT_C_sca & sc_J   = pm1->rad.sc_J(  ix_g,ix_s);
  AT_C_sca & sc_H_t = pm1->rad.sc_H_t(ix_g,ix_s);
  AT_N_vec & sp_H_d = pm1->rad.sp_H_d(ix_g,ix_s);

  // required geometric quantities
  AT_N_sym & sp_g_dd = pm1->geom.sp_g_dd;
  AT_N_sym & sp_g_uu = pm1->geom.sp_g_uu;

  // point to scratches
  AT_N_vec & sp_dH_d_    = pm1->scratch.sp_vec_A_;
  AT_N_sym & sp_P_tn_dd_ = pm1->scratch.sp_sym_A_;
  AT_N_sym & sp_P_tk_dd_ = pm1->scratch.sp_sym_B_;
  AT_N_sym & sp_dP_dd_   = pm1->scratch.sp_sym_C_;

  // initialize struct for root-finding
  DataRootfinder drf {
    sp_g_uu,
    sc_E,
    sp_F_d,
    sp_P_dd,
    sc_chi,
    sc_xi,
    sc_J,
    sc_H_t,
    sp_H_d,
    pm1->fidu.sc_W,
    pm1->fidu.sp_v_u,
    pm1->fidu.sp_v_d,
    sp_dH_d_,
    sp_P_tn_dd_,
    sp_P_tk_dd_,
    sp_dP_dd_
  };

  std::array<Real, 1> iI_xi;

  // main loop ----------------------------------------------------------------
  int iter_tot = 0;

  M1_GLOOP2(k,j)
  {
    M1_GLOOP1(i)
    {
      const Real dotFv = Assemble::sc_dot_dense_sp__(sp_F_d, sp_v_u, k, j, i);

      Closures::ClosureThin( pm1, sp_P_tn_dd_, sc_E, sp_F_d, k, j, i);
      Closures::ClosureThick(pm1, sp_P_tk_dd_, dotFv, sc_E, sp_F_d, k, j, i);

      const int iter_max_C = pm1->opt.max_iter_C;
      int pit = 0;     // iteration counter
      int rit = 0;     // restart counter
      const int iter_max_R = pm1->opt.max_iter_C_rst;  // max restarts
      Real w_opt = pm1->opt.w_opt_ini_C;  // underrelaxation factor
      Real e_C_abs_tol = pm1->opt.eps_C;
      // maximum error amplification factor between iters.
      Real fac_PA = pm1->opt.fac_amp_C;

      // retain values for potential restarts
      iI_xi[0] = sc_xi(k,j,i);

      Real e_abs_old = std::numeric_limits<Real>::infinity();
      Real e_abs_cur = 0;

      // solver loop ----------------------------------------------------------
      drf.i = i;
      drf.j = j;
      drf.k = k;

      const Real W = sc_W(k,j,i);
      const Real oo_W = 1.0 / W;
      const Real W2   = SQR(W);

      do
      {
        pit++;

        // const Real xi_old = sc_xi(k,j,i);

        // if ((pit == 1) && (rit == 0))
        // {
        //   const Real err_l = std::abs(R(xi_min, &drf));
        //   const Real err_r = std::abs(R(xi_max, &drf));
        //   const Real err_m = std::abs(R(xi_old, &drf));

        //   Real xi = (err_l < err_r) ? xi_min : xi_max;
        //   xi = (err_m < std::min(err_l, err_r)) ? xi_old : xi;

        //   if (std::min(err_m, std::min(err_l, err_r)) < e_C_abs_tol)
        //   {
        //     e_abs_cur = std::min(err_m, std::min(err_l, err_r));
        //     std::abs(R(xi, &drf));
        //     break;
        //   }
        // }

        // const Real xi = xi_old;

        // sc_E(k,j,i) = std::max(sc_E(k,j,i), 1e-12);

        const Real xi  = sc_xi(k,j,i);
        Real Z_xi = R(xi, &drf);

        // if (std::abs(Z_xi) < 1e-20)
        // {
        //   e_abs_cur = 0;
        //   break;
        // }

        Real dZ_xi = dR(xi, &drf);

        // Apply Newton iterate with fallback
        Real D = 0;
        Real sc_xi_can = std::numeric_limits<Real>::infinity();


        if ((std::abs(dZ_xi) > pm1->opt.eps_C_N * std::abs(Z_xi)) && (rit == 0))
        {
          // Newton
          D = Z_xi / dZ_xi;
          sc_xi_can = sc_xi(k,j,i) - D;
          // if (pit == 1)
          // {
          // }
          // else
          // {

          // }

          // if ((sc_xi_can < xi_min) || (sc_xi_can > xi_max))
          // {
          //   // Picard
          //   D = w_opt * Z_xi;
          //   sc_xi_can = sc_xi(k,j,i) - D;
          // }
        }
        else
        {
          // Picard
          D = w_opt * Z_xi;
          sc_xi_can = sc_xi(k,j,i) - D;

          // sc_xi_can = std::min(std::max(sc_xi_can, xi_min), xi_max);
          // sc_xi(k,j,i) = xi_max;
          // R(xi_max, &drf);
          // D = 0;
          // Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);
          // std::exit(0);

          // if (sc_xi_can < xi_min)
          // {
          //   sc_xi_can = xi_min;
          //   R(sc_xi_can, &drf);
          //   pit = 0;
          // }
          // else if (sc_xi_can > xi_max)
          // {
          //   sc_xi_can = xi_max;
          //   R(sc_xi_can, &drf);
          //   pit = 0;
          // }
        }

        // enforce non-negative values
        // sc_xi(k,j,i) = std::min(std::max(sc_xi_can, xi_min), xi_max);


        // sc_xi(k,j,i) = std::min(std::abs(sc_xi_can), xi_max);
        sc_xi(k,j,i) = sc_xi_can;

        // if (sc_xi(k,j,i) < xi_min)
        // {
        //   sc_xi(k,j,i) = xi_min;
        //   R(sc_xi(k,j,i), &drf);
        //   pit = 0;
        // }

        // if (sc_xi(k,j,i) > xi_max)
        // {
        //   sc_xi(k,j,i) = xi_max;
        //   R(sc_xi(k,j,i), &drf);
        //   pit = 0;
        // }

        // if (sc_xi(k,j,i) > xi_max)
        // {
        //   sc_xi(k,j,i) = xi_max;
        //   R(sc_xi(k,j,i), &drf);
        //   e_abs_cur = 0;
        //   break;
        // }

        // R(sc_xi(k,j,i), &drf);

        // scale error tol by step
        // e_abs_cur = std::abs(D);
        e_abs_cur = std::abs(D / w_opt);


        // e_abs_cur = std::abs(1 - (drf.xi_i + 1) / (drf.xi_im1 + 1));

        // if (std::abs(1 - (drf.xi_i + 1) / (drf.xi_im1 + 1)) < 1e-3)
        // {
        //   e_abs_cur = 0;
        //   break;
        // }

        if ((e_abs_cur > fac_PA * e_abs_old) && (pit > 30))
        {
          std::cout << pit << std::endl;
          std::cout << e_abs_cur << std::endl;
          std::cout << e_abs_old << std::endl;
          std::cout << Z_xi << std::endl;
          std::cout << D << std::endl;
          std::cout << dZ_xi << std::endl;
          std::cout << sc_xi_can << std::endl;
          std::cout << sc_xi(k,j,i) << std::endl;
          Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);

          std::cout << drf.xi_i << "\n";
          std::cout << drf.xi_im1 << "\n";
          std::cout << drf.xi_im2 << "\n";
          std::exit(0);

          // halve underrelaxation and recover old values
          w_opt = w_opt / 2;
          sc_xi(k,j,i) = iI_xi[0];

          // restart iteration
          e_abs_old = std::numeric_limits<Real>::infinity();
          pit = 0;
          rit++;

          if (rit > iter_max_R)
          {
            std::ostringstream msg;
            msg << "M1::Closures::Minerbo::AddClosureN max restarts exceeded.";
            std::cout << msg.str().c_str() << std::endl;
            Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);
            std::exit(0);
          }
        }
        else
        {
          e_abs_old = e_abs_cur;
          sc_chi(k,j,i) = chi(sc_xi(k,j,i));

          if (pm1->opt.reset_thin)
          {
            if ((sc_xi(k,j,i) < xi_min) || (sc_xi(k,j,i) > xi_max))
            {
              sc_xi(k,j,i) = xi_max;
              sc_chi(k,j,i) = chi(sc_xi(k,j,i));

              for (int a=0; a<N; ++a)
              for (int b=a; b<N; ++b)
              {
                sp_P_dd(a,b,k,j,i) = sp_P_tn_dd_(a,b,i);
              }

              e_abs_cur = 0.0; // forces iter breakout
            }
          }

        }

      } while ((pit < iter_max_C) && (e_abs_cur >= e_C_abs_tol));

      iter_tot += pit;

      if ((e_abs_cur > e_C_abs_tol) && pm1->opt.verbose_iter_C)
      {
        std::cout << "M1::Closures::Minerbo::AddClosureN:\n";
        std::cout << "Tol. not achieved: (pit,rit,e_abs_cur) ";
        std::cout << pit << "," << rit << "," << e_abs_cur << "\n";
        Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);
        // std::exit(0);
      }

      // sc_xi(k,j,i) = std::min(std::max(sc_xi(k,j,i), xi_min), xi_max);

    }


  }

  // const int nn = pm1->mbi.nn1 * pm1->mbi.nn2 * pm1->mbi.nn3;
  // std::cout << static_cast<Real>(iter_tot) / nn << std::endl;

}

void ClosureMinerbo(M1 * pm1,
                    AT_C_sca & sc_xi,
                    AT_C_sca & sc_chi,
                    AT_N_sym & sp_P_dd,
                    AT_C_sca & sc_E,
                    AT_N_vec & sp_F_d,
                    AT_C_sca & sc_J,
                    AT_C_sca & sc_H_t,
                    AT_N_vec & sp_H_d,
                    const int ix_g, const int ix_s,
                    const int k, const int j, const int i)
{
  // Admissible ranges for xi entering Eddington factor chi(xi)
  const Real xi_min = 0.0;
  const Real xi_max = 1.0;

  // required geometric quantities
  AT_N_sym & sp_g_uu = pm1->geom.sp_g_uu;

  // point to scratches
  AT_N_vec & sp_dH_d_    = pm1->scratch.sp_vec_A_;
  AT_N_sym & sp_P_tn_dd_ = pm1->scratch.sp_sym_A_;
  AT_N_sym & sp_P_tk_dd_ = pm1->scratch.sp_sym_B_;
  AT_N_sym & sp_dP_dd_   = pm1->scratch.sp_sym_C_;

  // initialize struct for root-finding
  DataRootfinder drf {
    sp_g_uu,
    sc_E,
    sp_F_d,
    sp_P_dd,
    sc_chi,
    sc_xi,
    sc_J,
    sc_H_t,
    sp_H_d,
    pm1->fidu.sc_W,
    pm1->fidu.sp_v_u,
    pm1->fidu.sp_v_d,
    sp_dH_d_,
    sp_P_tn_dd_,
    sp_P_tk_dd_,
    sp_dP_dd_
  };

  // setup GSL ----------------------------------------------------------------
  gsl_function R_;
  R_.function = &R;
  R_.params = &drf;

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

  drf.i = i;
  drf.j = j;
  drf.k = k;

  const Real dotFv = Assemble::sc_dot_dense_sp__(sp_F_d,
                                                 pm1->fidu.sp_v_u,
                                                 k, j, i);

  Closures::ClosureThin( pm1, sp_P_tn_dd_, sc_E, sp_F_d, k, j, i);
  Closures::ClosureThick(pm1, sp_P_tk_dd_, dotFv, sc_E, sp_F_d, k, j, i);

  int status = gsl_root_fsolver_set(pm1->gsl_brent_solver,
                                    &R_, xi_min, xi_max);

  /*
  typedef struct
    {
      double a, b, c, d, e;
      double fa, fb, fc;
    }
  brent_state_t;

  brent_state_t * state = static_cast<brent_state_t*>(gsl_solver->state);
  */

  switch (status)
  {
    case (GSL_EINVAL):  // bracketing failed (revert to thin closure)
    {
      Assemble::PointToDense(sp_P_dd, sp_P_tn_dd_, k, j, i);
      break;
    }
    case (0):
    {
      // root-finding loop
      Real loc_xi_min = xi_min;
      Real loc_xi_max = xi_max;

      status = GSL_CONTINUE;
      int iter = 0;
      do
      {
        iter++;
        status = gsl_root_fsolver_iterate(pm1->gsl_brent_solver);

        if (status != GSL_SUCCESS)
        {
          break;
        }
        else if (status)
        {
          sp_P_tn_dd_.array().print_all();
          sp_P_tk_dd_.array().print_all();

          Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);

          gsl_err_kill(status);
        }

        loc_xi_min = gsl_root_fsolver_x_lower(pm1->gsl_brent_solver);
        loc_xi_max = gsl_root_fsolver_x_upper(pm1->gsl_brent_solver);

        status = gsl_root_test_interval(
          loc_xi_min, loc_xi_max, pm1->opt.eps_C, 0
        );

      } while (iter<=pm1->opt.max_iter_C && status == GSL_CONTINUE);

      if ((status != GSL_SUCCESS) && pm1->opt.verbose_iter_C)
      {
        gsl_err_warn();
      }

      // Update P_dd with root
      sc_xi( k,j,i) = pm1->gsl_brent_solver->root;
      sc_chi(k,j,i) = chi(sc_xi(k,j,i));

      sp_P_dd__(sp_P_dd, sc_chi, sp_P_tn_dd_, sp_P_tk_dd_, k, j, i);

      break;
    }
    default:
    {
      Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);
      gsl_err_kill(status);
    }
  }

}

void ClosureMinerboN(M1 * pm1,
                     AT_C_sca & sc_xi,
                     AT_C_sca & sc_chi,
                     AT_N_sym & sp_P_dd,
                     AT_C_sca & sc_E,
                     AT_N_vec & sp_F_d,
                     AT_C_sca & sc_J,
                     AT_C_sca & sc_H_t,
                     AT_N_vec & sp_H_d,
                     const int ix_g, const int ix_s,
                     const int k, const int j, const int i)
{
  // if (sc_E(k,j,i) < 1e-13)
  // {
  //   AT_N_sym & sp_P_tn_dd_ = pm1->scratch.sp_sym_A_;

  //   Closures::ClosureThin( pm1, sp_P_tn_dd_, sc_E, sp_F_d, k, j, i);
  //   Assemble::PointToDense(sp_P_dd, sp_P_tn_dd_, k, j, i);

  //   sc_xi(k,j,i) = 1;
  //   sc_chi(k,j,i) = 1;
  //   return;
  // }

  // Admissible ranges for xi entering Eddington factor chi(xi)
  const Real xi_min = 0.0;
  const Real xi_max = 1.0;

  // required geometric quantities
  AT_N_sym & sp_g_uu = pm1->geom.sp_g_uu;

  // point to scratches
  AT_N_vec & sp_dH_d_    = pm1->scratch.sp_vec_A_;
  AT_N_sym & sp_P_tn_dd_ = pm1->scratch.sp_sym_A_;
  AT_N_sym & sp_P_tk_dd_ = pm1->scratch.sp_sym_B_;
  AT_N_sym & sp_dP_dd_   = pm1->scratch.sp_sym_C_;

  // initialize struct for root-finding
  DataRootfinder drf {
    sp_g_uu,
    sc_E,
    sp_F_d,
    sp_P_dd,
    sc_chi,
    sc_xi,
    sc_J,
    sc_H_t,
    sp_H_d,
    pm1->fidu.sc_W,
    pm1->fidu.sp_v_u,
    pm1->fidu.sp_v_d,
    sp_dH_d_,
    sp_P_tn_dd_,
    sp_P_tk_dd_,
    sp_dP_dd_
  };

  // setup GSL ----------------------------------------------------------------
  gsl_function_fdf R_;
  R_.f   = &R;
  R_.df  = &dR;
  R_.fdf = &RdR;
  R_.params = &drf;

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

  drf.i = i;
  drf.j = j;
  drf.k = k;

  const Real dotFv = Assemble::sc_dot_dense_sp__(sp_F_d,
                                                 pm1->fidu.sp_v_u,
                                                 k, j, i);

  Closures::ClosureThin( pm1, sp_P_tn_dd_, sc_E, sp_F_d, k, j, i);
  Closures::ClosureThick(pm1, sp_P_tk_dd_, dotFv, sc_E, sp_F_d, k, j, i);

  int status = gsl_root_fdfsolver_set(pm1->gsl_newton_solver,
                                      &R_, sc_xi(k,j,i));


  bool revert_brent = false;

  switch (status)
  {
    case (0):
    {
      // root-finding loop
      Real loc_xim1 = sc_xi(k,j,i);
      Real loc_xi   = sc_xi(k,j,i);

      status = GSL_CONTINUE;
      int iter = 0;
      do
      {
        iter++;
        status = gsl_root_fdfsolver_iterate(pm1->gsl_newton_solver);

        if (status != GSL_SUCCESS)
        {
          revert_brent = true;
          break;
        }

        loc_xim1 = loc_xi;
        loc_xi = gsl_root_fdfsolver_root(pm1->gsl_newton_solver);

        status = gsl_root_test_delta(
          loc_xim1, loc_xi, pm1->opt.eps_C, 0
        );

        if ((loc_xi<xi_min) || (loc_xi>xi_max))
        {
          revert_brent = true;
          break;
        }

      } while (iter<=pm1->opt.max_iter_C && status == GSL_CONTINUE);

      // Update P_dd with root
      sc_xi( k,j,i) = pm1->gsl_newton_solver->root;
      sc_chi(k,j,i) = chi(sc_xi(k,j,i));

      sp_P_dd__(sp_P_dd, sc_chi, sp_P_tn_dd_, sp_P_tk_dd_, k, j, i);

      break;
    }
    default:
    {
      revert_brent = true;
    }
  }

  if (revert_brent)
  {
    Closures::Minerbo::ClosureMinerbo(
      pm1, sc_xi, sc_chi, sp_P_dd,
      sc_E,
      sp_F_d,
      sc_J,
      sc_H_t,
      sp_H_d,
      ix_g, ix_s,
      k, j, i);
    return;
  }
}

void ClosureMinerboN_(M1 * pm1,
                     AT_C_sca & sc_xi,
                     AT_C_sca & sc_chi,
                     AT_N_sym & sp_P_dd,
                     AT_C_sca & sc_E,
                     AT_N_vec & sp_F_d,
                     AT_C_sca & sc_J,
                     AT_C_sca & sc_H_t,
                     AT_N_vec & sp_H_d,
                     const int ix_g, const int ix_s,
                     const int k, const int j, const int i)
{
  // Admissible ranges for xi entering Eddington factor chi(xi)
  const Real xi_min = 0.0;
  const Real xi_max = 1.0;

  // required geometric quantities
  AT_N_sym & sp_g_uu = pm1->geom.sp_g_uu;

  // point to scratches
  AT_N_vec & sp_dH_d_    = pm1->scratch.sp_vec_A_;
  AT_N_sym & sp_P_tn_dd_ = pm1->scratch.sp_sym_A_;
  AT_N_sym & sp_P_tk_dd_ = pm1->scratch.sp_sym_B_;
  AT_N_sym & sp_dP_dd_   = pm1->scratch.sp_sym_C_;

  // initialize struct for root-finding
  DataRootfinder drf {
    sp_g_uu,
    sc_E,
    sp_F_d,
    sp_P_dd,
    sc_chi,
    sc_xi,
    sc_J,
    sc_H_t,
    sp_H_d,
    pm1->fidu.sc_W,
    pm1->fidu.sp_v_u,
    pm1->fidu.sp_v_d,
    sp_dH_d_,
    sp_P_tn_dd_,
    sp_P_tk_dd_,
    sp_dP_dd_
  };

  // try Newton; revert to Brent below on failure / excursion from bracket

  // solver loop --------------------------------------------------------------
  drf.i = i;
  drf.j = j;
  drf.k = k;

  const Real dotFv = Assemble::sc_dot_dense_sp__(sp_F_d, pm1->fidu.sp_v_u,
                                                 k, j, i);

  Closures::ClosureThin( pm1, sp_P_tn_dd_, sc_E, sp_F_d, k, j, i);
  Closures::ClosureThick(pm1, sp_P_tk_dd_, dotFv, sc_E, sp_F_d, k, j, i);

  const int iter_max_C = pm1->opt.max_iter_C;
  int pit = 0;     // iteration counter
  int rit = 0;     // restart counter
  const int iter_max_R = pm1->opt.max_iter_C_rst;  // max restarts
  Real w_opt = pm1->opt.w_opt_ini_C;  // underrelaxation factor
  Real e_C_abs_tol = pm1->opt.eps_C;
  // maximum error amplification factor between iters.
  Real fac_PA = pm1->opt.fac_amp_C;

  Real e_abs_old = std::numeric_limits<Real>::infinity();
  Real e_abs_cur = 0;

  bool do_newton = true;

  auto is_valid = [&](const Real xi)
  {
    return (xi >= xi_min) && (xi <= xi_max);
  };


  drf.xi_i = std::numeric_limits<Real>::infinity();

  bool revert_brent = false;

  do
  {
    pit++;

    const Real xi  = sc_xi(k,j,i);
    Real Z_xi = R(xi, &drf);

    // break-fast?
    if ((std::abs(drf.xi_i-drf.xi_im1) < e_C_abs_tol))
    {
      break;
    }

    Real dZ_xi = dR(xi, &drf);

    // Apply Newton iterate with fallback
    Real D = 0;
    Real sc_xi_can = std::numeric_limits<Real>::infinity();

    if ((std::abs(dZ_xi) > pm1->opt.eps_C_N))
    {
      // Newton
      D = Z_xi / dZ_xi;
      sc_xi_can = sc_xi(k,j,i) - D;
    }
    else
    {
      revert_brent = true;
      break;
    }

    if (D > (xi_max - xi_min))
    {
      revert_brent = true;
      break;
    }

    e_abs_cur = std::abs(drf.xi_i - sc_xi_can);

    if ((e_abs_cur > fac_PA * e_abs_old))
    {
      // stagnated
      revert_brent = true;
      break;
    }
    else
    {
      e_abs_old = e_abs_cur;
      sc_xi(k,j,i) = sc_xi_can;
      sc_chi(k,j,i) = chi(sc_xi(k,j,i));
    }

  } while ((pit < iter_max_C) &&
           (e_abs_cur >= e_C_abs_tol));

  if (revert_brent || !is_valid(sc_xi(k,j,i)))
  {
    Closures::Minerbo::ClosureMinerbo(
      pm1, sc_xi, sc_chi, sp_P_dd,
      sc_E,
      sp_F_d,
      sc_J,
      sc_H_t,
      sp_H_d,
      ix_g, ix_s,
      k, j, i);
  }
  else
  {
    if (!is_valid(sc_xi(k,j,i)))
    {
      sc_xi(k,j,i) = std::max(
        std::min(xi_max, sc_xi(k,j,i)), xi_min
      );
      sc_chi(k,j,i) = chi(sc_xi(k,j,i));

    }
    sp_P_dd__(sp_P_dd, sc_chi, sp_P_tn_dd_, sp_P_tk_dd_, k, j, i);
    return;
  }

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

  vars_Lab U { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u, U);

  // fix ranges ---------------------------------------------------------------
  const int IL = 0;
  const int IU = mbi.nn1 - 1;

  const int JL = 0;
  const int JU = mbi.nn2 - 1;

  const int KL = 0;
  const int KU = mbi.nn3 - 1;

  /*
  for (int k=KL; k<=KU; ++k)
  for (int j=JL; j<=JU; ++j)
  for (int i=IL; i<=IU; ++i)
  {
    MaskThreshold(k,j,i);
  }
  */

  // iterate over species -----------------------------------------------------
  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    // get current g,s --------------------------------------------------------
    AT_N_sym & sp_P_dd = lab_aux.sp_P_dd(ix_g,ix_s);

    AT_C_sca & sc_E   = U.sc_E(  ix_g,ix_s);
    AT_N_vec & sp_F_d = U.sp_F_d(ix_g,ix_s);

    AT_C_sca & sc_chi = lab_aux.sc_chi(ix_g,ix_s);
    AT_C_sca & sc_xi  = lab_aux.sc_xi( ix_g,ix_s);

    AT_C_sca & sc_J   = rad.sc_J(  ix_g,ix_s);
    AT_C_sca & sc_H_t = rad.sc_H_t(ix_g,ix_s);
    AT_N_vec & sp_H_d = rad.sp_H_d(ix_g,ix_s);

    sp_P_dd.ZeroClear();

    // select closure ---------------------------------------------------------
    switch (opt.closure_variety)
    {
      case (opt_closure_variety::thin):
      {
        AT_N_sym & sp_P_tn_dd_ = scratch.sp_sym_A_;

        sc_chi.ZeroClear();
        sc_xi.ZeroClear();

        for (int k=KL; k<=KU; ++k)
        for (int j=JL; j<=JU; ++j)
        for (int i=IL; i<=IU; ++i)
        if (pm1->MaskGet(k,j,i))
        {
          Closures::ClosureThin(this, sp_P_tn_dd_, sc_E, sp_F_d,
                                k, j, i);
          Assemble::PointToDense(sp_P_dd, sp_P_tn_dd_, k, j, i);
          sc_xi( k,j,i) = 1.0;
          sc_chi(k,j,i) = 1.0;
        }

        break;
      }
      case (opt_closure_variety::thick):
      {
        AT_N_sym & sp_P_tk_dd_ = scratch.sp_sym_A_;

        sc_chi.ZeroClear();
        sc_xi.ZeroClear();

        for (int k=KL; k<=KU; ++k)
        for (int j=JL; j<=JU; ++j)
        for (int i=IL; i<=IU; ++i)
        if (pm1->MaskGet(k,j,i))
        {
          Closures::ClosureThick(this, sp_P_tk_dd_, sc_E, sp_F_d,
                                 k, j, i);
          Assemble::PointToDense(sp_P_dd, sp_P_tk_dd_, k, j, i);
          sc_xi( k,j,i) = 0.0;
          sc_chi(k,j,i) = ONE_3RD;
        }

        break;
      }
      case (opt_closure_variety::Minerbo):
      {
        for (int k=KL; k<=KU; ++k)
        for (int j=JL; j<=JU; ++j)
        for (int i=IL; i<=IU; ++i)
        if (pm1->MaskGet(k,j,i))
        {
          Closures::Minerbo::ClosureMinerbo(
            this, sc_xi, sc_chi, sp_P_dd,
            sc_E,
            sp_F_d,
            sc_J,
            sc_H_t,
            sp_H_d,
            ix_g, ix_s,
            k, j, i);
        }

        break;
      }
      case (opt_closure_variety::MinerboP):
      {
        Closures::Minerbo::AddClosureP(this,
                                       1.0, ix_g, ix_s,
                                       lab_aux.sp_P_dd(ix_g,ix_s));
        break;
      }
      case (opt_closure_variety::MinerboN):
      {
        // Closures::Minerbo::AddClosureN(this,
        //                                1.0, ix_g, ix_s,
        //                                lab_aux.sp_P_dd(ix_g,ix_s));

        for (int k=KL; k<=KU; ++k)
        for (int j=JL; j<=JU; ++j)
        for (int i=IL; i<=IU; ++i)
        if (pm1->MaskGet(k,j,i))
        {
          Closures::Minerbo::ClosureMinerboN(
            this, sc_xi, sc_chi, sp_P_dd,
            sc_E,
            sp_F_d,
            sc_J,
            sc_H_t,
            sp_H_d,
            ix_g, ix_s,
            k, j, i);
        }
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
//
// This isn't needed for the core algorithm but may be for data-dumping
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