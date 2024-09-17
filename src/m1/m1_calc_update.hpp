#ifndef M1_CALC_UPDATE_HPP
#define M1_CALC_UPDATE_HPP

// c++
// ...

// Athena++ classes headers
#include "m1.hpp"
#include "m1_containers.hpp"
#include "m1_macro.hpp"
#include "m1_utils.hpp"

// ============================================================================
namespace M1::Update {
// ============================================================================

// structure that stores state information
struct StateMetaVector {
  M1 & pm1;
  const int ix_g;
  const int ix_s;

  AT_C_sca & sc_nG;
  AT_C_sca & sc_E;
  AT_N_vec & sp_F_d;

  AT_C_sca & sc_n;
  AT_C_sca & sc_chi;
  AT_C_sca & sc_xi;
  AT_N_sym & sp_P_dd;

  AT_C_sca & sc_J;
  AT_C_sca & sc_H_t;
  AT_N_vec & sp_H_d;

  AT_C_sca & sc_eta_0;
  AT_C_sca & sc_kap_a_0;

  AT_C_sca & sc_eta;
  AT_C_sca & sc_kap_a;
  AT_C_sca & sc_kap_s;

  AT_C_sca & sc_avg_nrg;

  // Store state prior to iteration (for fallback)
  std::array<Real, 1> U_0_xi;
  std::array<Real, 1> U_0_E;
  std::array<Real, N> U_0_F_d;
  std::array<Real, 1> U_0_nG;

  // For iteration
  std::array<Real, 1> Z_xi;
  std::array<Real, 1> Z_E;
  std::array<Real, N> Z_F_d;

  void FallbackStore(const int k, const int j, const int i)
  {
    Assemble::PointAthenaTensorToArray(U_0_xi,  sc_xi,   k, j, i);
    Assemble::PointAthenaTensorToArray(U_0_E,   sc_E,   k, j, i);
    Assemble::PointAthenaTensorToArray(U_0_F_d, sp_F_d, k, j, i);
    Assemble::PointAthenaTensorToArray(U_0_nG,  sc_nG,  k, j, i);
  }

  void Fallback(const int k, const int j, const int i)
  {
    Assemble::PointArrayToAthenaTensor(sc_xi,  U_0_xi,  k, j, i);
    Assemble::PointArrayToAthenaTensor(sc_E,   U_0_E,   k, j, i);
    Assemble::PointArrayToAthenaTensor(sp_F_d, U_0_F_d, k, j, i);
    Assemble::PointArrayToAthenaTensor(sc_nG,  U_0_nG,  k, j, i);
  }

};

struct SourceMetaVector {
  M1 & pm1;
  const int ix_g;
  const int ix_s;

  AT_C_sca & sc_S0;
  AT_C_sca & sc_S1;
  AT_N_vec & sp_S1_d;
};

StateMetaVector ConstructStateMetaVector(
  M1 & pm1, M1::vars_Lab & vlab,
  const int ix_g, const int ix_s);

SourceMetaVector ConstructSourceMetaVector(
  M1 & pm1, M1::vars_Source & vsrc,
  const int ix_g, const int ix_s);

void AddSourceMatter(
  M1 & pm1,
  const StateMetaVector & C,  // state to utilize
  StateMetaVector & I,        // add source here
  SourceMetaVector & S,
  const int k, const int j, const int i);


// If appled return true, otherwise false
template <class V>
inline bool NonFiniteToZero(
  M1 & pm1, V & C,
  const int k, const int j, const int i)
{
  bool floor_reset = !std::isfinite(C.sc_E(k,j,i));
  for (int a=0; a<N; ++a)
  {
    floor_reset = floor_reset or !std::isfinite(C.sp_F_d(a,k,j,i));
  }

  if (floor_reset)
  {
    C.sc_E(k,j,i) = 0;
    for (int a=0; a<N; ++a)
    {
      C.sp_F_d(a,k,j,i) = 0;
    }
  }
  return floor_reset;
}

// If applied return true, otherwise false
template <class V>
inline bool ApplyFloors(
  M1 & pm1, V & C,
  const int k, const int j, const int i)
{
  const bool floor_applied = C.sc_E(k,j,i) < pm1.opt.fl_E;
  C.sc_E(k,j,i) = std::max(C.sc_E(k,j,i), pm1.opt.fl_E);
  return floor_applied;
}

// Require | F | / E <= 1
//
// Enforce this by setting: F_i -> F_i / (\Vert F \Vert_\gamma / E)
//
// If enforced return true, otherwise false
template <class V>
inline bool EnforceCausality(
  M1 & pm1, V & C,
  const int k, const int j, const int i)
{
  /*
  const Real norm2F = Assemble::sp_norm2__(C.sp_F_d, pm1.geom.sp_g_uu,
                                           k, j, i);

  if (norm2F > 0)
  if (SQR(C.sc_E(k,j,i)) > norm2F)
  {
    const Real normF = std::sqrt(norm2F);
    const Real fac = normF / C.sc_E(k,j,i);  // BD: TODO - add eps to this in case |F|=E

    const Real rfac = 1.0 / (fac + pm1.opt.eps_ec_fac);
    for (int a=0; a<N; ++a)
    {
      C.sp_F_d(a,k,j,i) = C.sp_F_d(a,k,j,i) * rfac;
    }
    return true;
  }

  return false;
  */

  bool rescale = false;
  for (int a=0; a<N; ++a)
  {
    if (1 < std::abs(C.sp_F_d(a,k,j,i)) / C.sc_E(k,j,i))
    {
      rescale = true;
    }
  }

  if (rescale)
  {
    const Real norm2F = Assemble::sp_norm2__(C.sp_F_d, pm1.geom.sp_g_uu,
                                             k, j, i);
    const Real normF = std::sqrt(norm2F);
    const Real fac = normF / C.sc_E(k,j,i);

    const Real rfac = 1.0 / (fac + pm1.opt.eps_ec_fac);
    for (int a=0; a<N; ++a)
    {
      C.sp_F_d(a,k,j,i) = C.sp_F_d(a,k,j,i) * rfac;
    }
  }

  return rescale;
}

// ============================================================================
} // namespace M1::Update
// ============================================================================


#endif // M1_CALC_UPDATE_HPP

