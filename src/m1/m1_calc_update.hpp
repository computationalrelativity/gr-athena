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
  AT_D_vec & st_H_u;

  AT_C_sca & sc_eta_0;
  AT_C_sca & sc_kap_a_0;

  AT_C_sca & sc_eta;
  AT_C_sca & sc_kap_a;
  AT_C_sca & sc_kap_s;

  AT_C_sca & sc_avg_nrg;

  // Store state prior to iteration (for fallback)
  std::array<Real, 1> U_0_xi;
  std::array<Real, 1> U_0_chi;
  std::array<Real, 1> U_0_E;
  std::array<Real, N> U_0_F_d;
  std::array<Real, 1> U_0_nG;

  // For iteration
  std::array<Real, 1> Z_xi;
  std::array<Real, 1> Z_chi;
  std::array<Real, 1> Z_E;
  std::array<Real, N> Z_F_d;

  void FallbackStore(const int k, const int j, const int i)
  {
    Assemble::PointAthenaTensorToArray(U_0_xi,  sc_xi,  k, j, i);
    Assemble::PointAthenaTensorToArray(U_0_chi, sc_chi, k, j, i);
    Assemble::PointAthenaTensorToArray(U_0_E,   sc_E,   k, j, i);
    Assemble::PointAthenaTensorToArray(U_0_F_d, sp_F_d, k, j, i);
    Assemble::PointAthenaTensorToArray(U_0_nG,  sc_nG,  k, j, i);
  }

  void Fallback(const int k, const int j, const int i)
  {
    Assemble::PointArrayToAthenaTensor(sc_xi,  U_0_xi,  k, j, i);
    Assemble::PointArrayToAthenaTensor(sc_chi, U_0_chi, k, j, i);
    Assemble::PointArrayToAthenaTensor(sc_E,   U_0_E,   k, j, i);
    Assemble::PointArrayToAthenaTensor(sp_F_d, U_0_F_d, k, j, i);
    Assemble::PointArrayToAthenaTensor(sc_nG,  U_0_nG,  k, j, i);
  }

};

struct SourceMetaVector {
  M1 & pm1;
  const int ix_g;
  const int ix_s;

  AT_C_sca & sc_nG;
  AT_C_sca & sc_E;
  AT_N_vec & sp_F_d;
};

StateMetaVector ConstructStateMetaVector(
  M1 & pm1, M1::vars_Lab & vlab,
  const int ix_g, const int ix_s);

SourceMetaVector ConstructSourceMetaVector(
  M1 & pm1, M1::vars_Source & vsrc,
  const int ix_g, const int ix_s);

// For components (E, F_d) perform tar <- src
inline void Copy_E_F_d(StateMetaVector &dst,
                       const StateMetaVector &src,
                       const int k, const int j, const int i)
{
  dst.sc_E(k,j,i) = src.sc_E(k,j,i);
  for (int a=0; a<N; ++a)
  {
    dst.sp_F_d(a,k,j,i) = src.sp_F_d(a,k,j,i);
  }
}

// For components (nG, ) perform tar <- src
inline void Copy_nG(StateMetaVector &dst,
                    const StateMetaVector &src,
                    const int k, const int j, const int i)
{
  dst.sc_nG(k,j,i) = src.sc_nG(k,j,i);
}

// For components (nG, E, F_d) perform tar <- src
inline void Copy_nG_E_F_d(StateMetaVector &dst,
                          const StateMetaVector &src,
                          const int k, const int j, const int i)
{
  Copy_nG(dst, src, k, j, i);
  Copy_E_F_d(dst, src, k, j, i);
}

// For components (E, F_d) perform V <- V + sca * S
template <class T>
inline void InPlaceScalarMulAdd_E_F_d(const Real sca,
                                      StateMetaVector &V,
                                      const T &S,
                                      const int k, const int j, const int i)
{
  V.sc_E(k,j,i) += sca * S.sc_E(k,j,i);
  for (int a=0; a<N; ++a)
  {
    V.sp_F_d(a,k,j,i) += sca * S.sp_F_d(a,k,j,i);
  }
}

// For components (nG, ) perform V <- V + sca * S
template <class T>
inline void InPlaceScalarMulAdd_nG(const Real sca,
                                   StateMetaVector &V,
                                   const T &S,
                                   const int k, const int j, const int i)
{
  V.sc_nG(k,j,i) += sca * S.sc_nG(k,j,i);
}

// For components (nG, E, F_d) perform V <- V + sca * S
template <class T>
inline void InPlaceScalarMulAdd_nG_E_F_d(const Real sca,
                                         StateMetaVector &V,
                                         const T &S,
                                         const int k, const int j, const int i)
{
  InPlaceScalarMulAdd_nG(sca, V, S, k, j, i);
  InPlaceScalarMulAdd_E_F_d(sca, V, S, k, j, i);
}

// For components (nG, E, F_d) perform V <- V + sca * (X + Y)
template <class T, class U>
inline void InPlaceScalarMulAdd_nG_E_F_d(const Real sca,
                                         StateMetaVector &V,
                                         const T &X,
                                         const U &Y,
                                         const int k, const int j, const int i)
{
  V.sc_nG(k,j,i) += sca * (X.sc_nG(k,j,i) + Y.sc_nG(k,j,i));
  V.sc_E(k,j,i) += sca * (X.sc_E(k,j,i) + Y.sc_E(k,j,i));

  for (int a=0; a<N; ++a)
  {
    V.sp_F_d(a,k,j,i) += sca * (X.sp_F_d(a,k,j,i) + Y.sp_F_d(a,k,j,i));
  }
}

// For components (E, F_d) perform S <- sca * S
template <class T>
inline void InPlaceScalarMul_E_F_d(const Real sca,
                                   T &S,
                                   const int k, const int j, const int i)
{
  S.sc_E(k,j,i) *= sca;
  for (int a=0; a<N; ++a)
  {
    S.sp_F_d(a,k,j,i) *= sca;
  }
}

// For components (nG, ) perform S <- sca * S
template <class T>
inline void InPlaceScalarMul_nG(const Real sca,
                                T &S,
                                const int k, const int j, const int i)
{
  S.sc_nG(k,j,i) *= sca;
}

// For components (nG, E, F_d) perform S <- sca * S
template <class T>
inline void InPlaceScalarMul_nG_E_F_d(const Real sca,
                                      T &S,
                                      const int k, const int j, const int i)
{
  S.sc_nG(k,j,i) *= sca;
  S.sc_E(k,j,i) *= sca;
  for (int a=0; a<N; ++a)
  {
    S.sp_F_d(a,k,j,i) *= sca;
  }
}

// If appled return true, otherwise false
template <class V>
inline bool NonFiniteToZero(
  M1 & pm1, V & C,
  const int k, const int j, const int i)
{
  // Check if finite, faster to sum all values and then check if result
  Real val = C.sc_E(k,j,i) + C.sc_nG(k,j,i);
  for (int a=0; a<N; ++a)
  {
    val += C.sp_F_d(a,k,j,i);
  }

  const bool floor_reset = !std::isfinite(val);

  if (floor_reset)
  {
    C.sc_E(k,j,i) = pm1.opt.fl_E;
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

  if (floor_applied)
  {
    for (int a=0; a<N; ++a)
    {
      C.sp_F_d(a,k,j,i) = 0;
    }
  }

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
  const Real norm2F = Assemble::sp_norm2__(C.sp_F_d, pm1.geom.sp_g_uu,
                                           k, j, i);

  if (norm2F > 0)
  if (SQR(C.sc_E(k,j,i)) < norm2F)
  {
    const Real normF = std::sqrt(norm2F);
    const Real fac = normF / C.sc_E(k,j,i);

    const Real rfac = 1.0 / (fac + pm1.opt.eps_ec_fac);
    for (int a=0; a<N; ++a)
    {
      C.sp_F_d(a,k,j,i) = C.sp_F_d(a,k,j,i) * rfac;
    }
    return true;
  }

  return false;

  /*
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
  */
}

// Enforce physicality on (E, F_d) components of V
inline void EnforcePhysical_E_F_d(
  M1 & pm1,
  Update::StateMetaVector & V,
  const int k, const int j, const int i)
{
  NonFiniteToZero(pm1, V, k, j, i);
  ApplyFloors(pm1, V, k, j, i);
  if (pm1.opt.enforce_causality)
  {
    EnforceCausality(pm1, V, k, j, i);
  }
}

// Enforce physicality on (nG, ) components of V
inline void EnforcePhysical_nG(
  M1 & pm1,
  Update::StateMetaVector & V,
  const int k, const int j, const int i)
{
  V.sc_nG(k,j,i) = std::max(V.sc_nG(k,j,i), pm1.opt.fl_nG);
}

// ============================================================================
} // namespace M1::Update
// ============================================================================


#endif // M1_CALC_UPDATE_HPP

