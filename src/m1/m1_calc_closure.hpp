#ifndef M1_CALC_CLOSURE_HPP
#define M1_CALC_CLOSURE_HPP

// c++
// ...

// Athena++ classes headers
#include "m1.hpp"
#include "m1_containers.hpp"
#include "m1_macro.hpp"
#include "m1_utils.hpp"


// ============================================================================
namespace M1::Closures {
// ============================================================================

void InfoDump(M1 * pm1, const int ix_g, const int ix_s,
              const int k, const int j, const int i);

void ClosureThin(M1 * pm1,
                 const Real weight,
                 const int ix_g,
                 const int ix_s,
                 AT_N_sym & sp_P_dd_,
                 const int k, const int j,
                 const int il, const int iu);

void ClosureThin(M1 * pm1,
                 AT_N_sym & sp_P_dd_,
                 const AT_C_sca & sc_E,
                 const AT_N_vec & sp_F_d,
                 const int k, const int j, const int i);

void ClosureThick(M1 * pm1,
                  const Real weight,
                  const int ix_g,
                  const int ix_s,
                  AT_N_sym & sp_P_dd_,
                  const int k, const int j,
                  const int il, const int iu);

void ClosureThick(M1 * pm1,
                  AT_N_sym & sp_P_dd_,
                  const Real dotFv,
                  const AT_C_sca & sc_E,
                  const AT_N_vec & sp_F_d,
                  const int k, const int j, const int i);

// ============================================================================
}  // M1::Closures
// ============================================================================

// ============================================================================
namespace M1::Closures::Minerbo {
// ============================================================================

// Required data during rootfinding procedure
struct DataRootfinder {
  AT_N_sym & sp_g_uu;

  AT_C_sca & sc_E;
  AT_N_vec & sp_F_d;
  AT_N_sym & sp_P_dd;
  AT_C_sca & sc_chi;
  AT_C_sca & sc_xi;
  AT_C_sca & sc_J;
  AT_C_sca & sc_H_t;
  AT_N_vec & sp_H_d;
  AT_C_sca & sc_W;
  AT_N_vec & sp_v_u;
  AT_N_vec & sp_v_d;

  // scratch
  AT_N_vec & sp_dH_d_;
  AT_N_sym & sp_P_tn_dd_;
  AT_N_sym & sp_P_tk_dd_;
  AT_N_sym & sp_dP_dd_;

  // scratch - scalar reductions
  Real dotFv;

  // grid
  int i, j, k;

  // retain iterations
  Real Z_xi_im2;
  Real Z_xi_im1;
  Real Z_xi_i;

  Real dZ_xi_im2;
  Real dZ_xi_im1;
  Real dZ_xi_i;

  Real xi_im2;
  Real xi_im1;
  Real xi_i;
};

// Function to find root of.
// Warning: P_dd, xi, chi, etc are modified in par struct
// Note: use with par as DataRootfinder
Real R( Real xi, void *par);

// Assumed to be called immediately after R
Real dR(Real xi, void *par);

// Eddington factor
inline Real chi(const Real xi)
{
  const Real xi2 = SQR(xi);
  return ONE_3RD + xi2 / 15.0 * (6.0 - 2.0 * xi + 6 * xi2);
}

inline void sp_P_dd(
  AT_N_sym & sp_tar_dd,
  const AT_C_sca & sc_chi,
  const AT_N_sym & sp_P_tn_dd_,
  const AT_N_sym & sp_P_tk_dd_,
  const int k, const int j,
  const int il, const int iu)
{

  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
   sp_tar_dd(a,b,k,j,i) = 0.5 * (
      (3.0 * sc_chi(k,j,i) - 1.0) * sp_P_tn_dd_(a,b,i) +
      3.0 * (1.0 - sc_chi(k,j,i)) * sp_P_tk_dd_(a,b,i)
    );
  }
}

inline void sp_P_dd(
  AT_N_sym & sp_tar_dd,
  const AT_C_sca & sc_chi,
  const AT_N_sym & sp_P_tn_dd_,
  const AT_N_sym & sp_P_tk_dd_,
  const int k, const int j, const int i)
{

  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  {
   sp_tar_dd(a,b,k,j,i) = 0.5 * (
      (3.0 * sc_chi(k,j,i) - 1.0) * sp_P_tn_dd_(a,b,i) +
      3.0 * (1.0 - sc_chi(k,j,i)) * sp_P_tk_dd_(a,b,i)
    );
  }
}

// ============================================================================
} // namespace M1::Closures::Minerbo
// ============================================================================


#endif // M1_CALC_CLOSURE_HPP

