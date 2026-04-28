#ifndef UTILS_GRID_THETA_PHI_FIELDS_HPP
#define UTILS_GRID_THETA_PHI_FIELDS_HPP
//========================================================================================
// GR-Athena++
//========================================================================================
//! \file grid_theta_phi_fields.hpp
//  \brief Derivative-bundled fields on a theta-phi grid.
//
//  DTensorField<T, sym, ndim, rank>      - grid-stored tensor (heap per point)
//  DTensorFieldPoint<T, sym, ndim, rank> - point-stored tensor (stack)
//  DMultipoleField<T>                    - multipole-decomposed scalar (heap
//  per lm)
//
//  Each bundles a quantity with its radial/time derivative flavors:
//    D00 = value, D10 = d_r, D20 = d^2_r, D01 = d_t, D11 = d_r d_t
//
//  Access is by arity: value = rank + grid_dims args, derivative = 1 + rank +
//  grid_dims.
//    gamma(a, b, i, j)       - value     (rank-2 grid, 4 args)
//    gamma(D10, a, b, i, j)  - d_r       (rank-2 grid, 5 args)
//    alpha(i, j)             - value     (rank-0 grid, 2 args)
//    alpha(D10, i, j)        - d_r       (rank-0 grid, 3 args)
//    h00(lm, c)              - value     (multipole, 2 args)
//    h00(D10, lm, c)         - d_r       (multipole, 3 args)

#include <array>

#include "../athena_tensor.hpp"

namespace gra::grids::theta_phi
{

// ============================================================================
// Derivative multi-index constants for radial/time derivatives.
// Dpq notation: p = radial order, q = time order.
// ============================================================================
namespace ix_DRT
{
constexpr int D00    = 0;  // value
constexpr int D10    = 1;  // d_r
constexpr int D20    = 2;  // d^2_r
constexpr int D01    = 3;  // d_t
constexpr int D11    = 4;  // d_r d_t
constexpr int NDERIV = 5;
}  // namespace ix_DRT

// ============================================================================
// Forward declarations
// ============================================================================
template <typename T, TensorSymm sym, int ndim, int rank>
class DTensorField;

template <typename T, TensorSymm sym, int ndim, int rank>
class DTensorFieldPoint;

// ============================================================================
// DTensorField - grid storage (rank 0)
// ============================================================================
template <typename T, TensorSymm sym, int ndim>
class DTensorField<T, sym, ndim, 0>
{
  using AT = AthenaTensor<T, sym, ndim, 0, TensorStorage::Grid>;

  public:
  // Allocate all derivative slots on an (ntheta x nphi) grid.
  void Allocate(int nth, int nph)
  {
    for (auto& t : d_)
      t.NewAthenaTensor(nth, nph);
  }

  // -- Value access: operator()(i, j) ----------------------------------------
  T& operator()(int const i, int const j)
  {
    return d_[0](i, j);
  }
  T operator()(int const i, int const j) const
  {
    return d_[0](i, j);
  }

  // -- Derivative access: operator()(d, i, j) --------------------------------
  T& operator()(int const d, int const i, int const j)
  {
    return d_[d](i, j);
  }
  T operator()(int const d, int const i, int const j) const
  {
    return d_[d](i, j);
  }

  // -- Raw slot access -------------------------------------------------------
  AT& operator[](int const d)
  {
    return d_[d];
  }
  const AT& operator[](int const d) const
  {
    return d_[d];
  }

  private:
  std::array<AT, ix_DRT::NDERIV> d_;
};

// ============================================================================
// DTensorField - grid storage (rank 1)
// ============================================================================
template <typename T, TensorSymm sym, int ndim>
class DTensorField<T, sym, ndim, 1>
{
  using AT = AthenaTensor<T, sym, ndim, 1, TensorStorage::Grid>;

  public:
  void Allocate(int nth, int nph)
  {
    for (auto& t : d_)
      t.NewAthenaTensor(nth, nph);
  }

  // -- Value access: operator()(a, i, j) -------------------------------------
  T& operator()(int const a, int const i, int const j)
  {
    return d_[0](a, i, j);
  }
  T operator()(int const a, int const i, int const j) const
  {
    return d_[0](a, i, j);
  }

  // -- Derivative access: operator()(d, a, i, j) -----------------------------
  T& operator()(int const d, int const a, int const i, int const j)
  {
    return d_[d](a, i, j);
  }
  T operator()(int const d, int const a, int const i, int const j) const
  {
    return d_[d](a, i, j);
  }

  // -- Raw slot access -------------------------------------------------------
  AT& operator[](int const d)
  {
    return d_[d];
  }
  const AT& operator[](int const d) const
  {
    return d_[d];
  }

  private:
  std::array<AT, ix_DRT::NDERIV> d_;
};

// ============================================================================
// DTensorField - grid storage (rank 2)
// ============================================================================
template <typename T, TensorSymm sym, int ndim>
class DTensorField<T, sym, ndim, 2>
{
  using AT = AthenaTensor<T, sym, ndim, 2, TensorStorage::Grid>;

  public:
  void Allocate(int nth, int nph)
  {
    for (auto& t : d_)
      t.NewAthenaTensor(nth, nph);
  }

  // -- Value access: operator()(a, b, i, j) ----------------------------------
  T& operator()(int const a, int const b, int const i, int const j)
  {
    return d_[0](a, b, i, j);
  }
  T operator()(int const a, int const b, int const i, int const j) const
  {
    return d_[0](a, b, i, j);
  }

  // -- Derivative access: operator()(d, a, b, i, j) --------------------------
  T& operator()(int const d,
                int const a,
                int const b,
                int const i,
                int const j)
  {
    return d_[d](a, b, i, j);
  }
  T operator()(int const d,
               int const a,
               int const b,
               int const i,
               int const j) const
  {
    return d_[d](a, b, i, j);
  }

  // -- Raw slot access -------------------------------------------------------
  AT& operator[](int const d)
  {
    return d_[d];
  }
  const AT& operator[](int const d) const
  {
    return d_[d];
  }

  private:
  std::array<AT, ix_DRT::NDERIV> d_;
};

// ============================================================================
// DTensorFieldPoint - point storage (rank 0)
// ============================================================================
template <typename T, TensorSymm sym, int ndim>
class DTensorFieldPoint<T, sym, ndim, 0>
{
  using AT = AthenaTensor<T, sym, ndim, 0, TensorStorage::Point>;

  public:
  // -- Value access: operator()() --------------------------------------------
  T& operator()()
  {
    return d_[0]();
  }
  T operator()() const
  {
    return d_[0]();
  }

  // -- Derivative access: operator()(d) --------------------------------------
  T& operator()(int const d)
  {
    return d_[d]();
  }
  T operator()(int const d) const
  {
    return d_[d]();
  }

  // -- Raw slot access -------------------------------------------------------
  AT& operator[](int const d)
  {
    return d_[d];
  }
  const AT& operator[](int const d) const
  {
    return d_[d];
  }

  void ZeroClear()
  {
    for (auto& t : d_)
      t.ZeroClear();
  }

  private:
  std::array<AT, ix_DRT::NDERIV> d_;
};

// ============================================================================
// DTensorFieldPoint - point storage (rank 1)
// ============================================================================
template <typename T, TensorSymm sym, int ndim>
class DTensorFieldPoint<T, sym, ndim, 1>
{
  using AT = AthenaTensor<T, sym, ndim, 1, TensorStorage::Point>;

  public:
  // -- Value access: operator()(a) -------------------------------------------
  T& operator()(int const a)
  {
    return d_[0](a);
  }
  T operator()(int const a) const
  {
    return d_[0](a);
  }

  // -- Derivative access: operator()(d, a) -----------------------------------
  T& operator()(int const d, int const a)
  {
    return d_[d](a);
  }
  T operator()(int const d, int const a) const
  {
    return d_[d](a);
  }

  // -- Raw slot access -------------------------------------------------------
  AT& operator[](int const d)
  {
    return d_[d];
  }
  const AT& operator[](int const d) const
  {
    return d_[d];
  }

  void ZeroClear()
  {
    for (auto& t : d_)
      t.ZeroClear();
  }

  private:
  std::array<AT, ix_DRT::NDERIV> d_;
};

// ============================================================================
// DTensorFieldPoint - point storage (rank 2)
// ============================================================================
template <typename T, TensorSymm sym, int ndim>
class DTensorFieldPoint<T, sym, ndim, 2>
{
  using AT = AthenaTensor<T, sym, ndim, 2, TensorStorage::Point>;

  public:
  // -- Value access: operator()(a, b) ----------------------------------------
  T& operator()(int const a, int const b)
  {
    return d_[0](a, b);
  }
  T operator()(int const a, int const b) const
  {
    return d_[0](a, b);
  }

  // -- Derivative access: operator()(d, a, b) --------------------------------
  T& operator()(int const d, int const a, int const b)
  {
    return d_[d](a, b);
  }
  T operator()(int const d, int const a, int const b) const
  {
    return d_[d](a, b);
  }

  // -- Raw slot access -------------------------------------------------------
  AT& operator[](int const d)
  {
    return d_[d];
  }
  const AT& operator[](int const d) const
  {
    return d_[d];
  }

  void ZeroClear()
  {
    for (auto& t : d_)
      t.ZeroClear();
  }

  private:
  std::array<AT, ix_DRT::NDERIV> d_;
};

// ============================================================================
// DMultipoleField - multipole-decomposed scalar with derivative slots.
//
// Each slot is an AthenaArray<T>(lmpoints, 2) indexed by (lm, c)
// where c in {Re, Im}.  Arity-based access matches DTensorField:
//   f(lm, c)        - value (D00)
//   f(d, lm, c)     - derivative slot d
//   f[d]            - raw AthenaArray<T>& for slot d
// ============================================================================
template <typename T>
class DMultipoleField
{
  public:
  void Allocate(int lmpoints)
  {
    for (auto& a : d_)
      a.NewAthenaArray(lmpoints, 2);
  }

  // -- Value access: operator()(lm, c)
  // ----------------------------------------
  T& operator()(int const lm, int const c)
  {
    return d_[0](lm, c);
  }
  T operator()(int const lm, int const c) const
  {
    return d_[0](lm, c);
  }

  // -- Derivative access: operator()(d, lm, c)
  // --------------------------------
  T& operator()(int const d, int const lm, int const c)
  {
    return d_[d](lm, c);
  }
  T operator()(int const d, int const lm, int const c) const
  {
    return d_[d](lm, c);
  }

  // -- Raw slot access -------------------------------------------------------
  AthenaArray<T>& operator[](int const d)
  {
    return d_[d];
  }
  const AthenaArray<T>& operator[](int const d) const
  {
    return d_[d];
  }

  private:
  std::array<AthenaArray<T>, ix_DRT::NDERIV> d_;
};

}  // namespace gra::grids::theta_phi

#endif  // UTILS_GRID_THETA_PHI_FIELDS_HPP
