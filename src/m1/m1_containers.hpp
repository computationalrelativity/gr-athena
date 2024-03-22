#ifndef M1_CONTAINERS_HPP
#define M1_CONTAINERS_HPP

#include "m1_macro.hpp"

// c++
#include <vector>

// Athena++ classes headers
<<<<<<< HEAD
#include "../athena_aliases.hpp"
#include "../utils/tensor.hpp"
=======
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../athena_tensor.hpp"
// #include "../utils/tensor.hpp"

// ============================================================================
namespace M1 {
// ============================================================================
>>>>>>> cc2a74d6 (Namespace M1 to avoid future headaches. Couple (mock) external fluid / geometry)

// typedef ====================================================================

// define some types to make everything more readable

<<<<<<< HEAD
using namespace gra::aliases;
=======
static const int D = M1_NDIM + 1;
static const int N = M1_NDIM;

typedef AthenaArray< Real>                         AA;
// spatial hypersurface
typedef AthenaTensor<Real, TensorSymm::NONE, N, 0> AT_N_sca;
typedef AthenaTensor<Real, TensorSymm::NONE, N, 1> AT_N_vec;
typedef AthenaTensor<Real, TensorSymm::SYM2, N, 2> AT_N_sym;
>>>>>>> cc2a74d6 (Namespace M1 to avoid future headaches. Couple (mock) external fluid / geometry)

// symmetric tensor derivatives
typedef AthenaTensor<Real, TensorSymm::SYM2, N, 3> AT_N_D1sym;

// space-time fields
typedef AthenaTensor<Real, TensorSymm::NONE, D, 0> AT_D_sca;
typedef AthenaTensor<Real, TensorSymm::NONE, D, 1> AT_D_vec;
typedef AthenaTensor<Real, TensorSymm::SYM2, D, 2> AT_D_sym;

// Treat group & species as super-indices with this nested structure
typedef AthenaArray<AT_N_sca> GS_AT_N_sca;
typedef AthenaArray<AT_N_vec> GS_AT_N_vec;

/*
typedef utils::tensor::TensorPointwise<
  Real,
  utils::tensor::Symmetries::NONE,
  N,
  0> TP_N_sca;

typedef utils::tensor::TensorPointwise<
  Real,
  utils::tensor::Symmetries::NONE,
  N,
  1> TP_N_vec;

typedef utils::tensor::TensorPointwise<
  Real,
  utils::tensor::Symmetries::SYM2,
  N,
  2> TP_N_sym;
*/


// Class ======================================================================

// Note that GS containers need to have something assigned (e.g. via alias)
// or there will be problems.

// Treat group & species as super-indices with this nested structure
template<typename T>
class GroupSpeciesContainer
{

public:
  GroupSpeciesContainer() {}
  GroupSpeciesContainer(int N_GRPS, int N_SPCS)
    :
    N_GRPS(N_GRPS),
    N_SPCS(N_SPCS)
  {
    data_.resize(N_GRPS * N_SPCS);
  }
  ~GroupSpeciesContainer()
  {
    for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
    for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
    {
      data_[ix_map(ix_g,ix_s)].DeleteAthenaTensor();
    }
  }

  T & operator()(int const ix_g, int const ix_s)
  {
    return data_[ix_map(ix_g,ix_s)];
  }

  T operator()(int const ix_g, int const ix_s) const
  {
    return data_[ix_map(ix_g,ix_s)];
  }

  inline int ix_map(int const ix_g, int const ix_s)
  {
    return ix_s + N_SPCS * ix_g;
  }

private:
  const int N_GRPS = 0;
  const int N_SPCS = 0;

  std::vector<T> data_;

};

// Treat group & species as super-indices with this nested structure
template<typename T>
class GroupSpeciesFluxContainer
{

public:
  GroupSpeciesFluxContainer() {}
  GroupSpeciesFluxContainer(int N_GRPS, int N_SPCS)
    :
    N_GRPS(N_GRPS),
    N_SPCS(N_SPCS)
  {
    data_.resize(N_GRPS * N_SPCS * M1_NDIM);
  }
  ~GroupSpeciesFluxContainer()
  {
    for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
    for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
    for (int ix_f=0; ix_f<M1_NDIM; ++ix_f)
    {
      data_[ix_map(ix_g,ix_s,ix_f)].DeleteAthenaTensor();
    }
  }

  T & operator()(int const ix_g, int const ix_s, int const ix_f)
  {
    return data_[ix_map(ix_g,ix_s,ix_f)];
  }

  T operator()(int const ix_g, int const ix_s, int const ix_f) const
  {
    return data_[ix_map(ix_g,ix_s,ix_f)];
  }

  inline int ix_map(int const ix_g, int const ix_s, int const ix_f)
  {
    return ix_f + M1_NDIM * (ix_s + N_SPCS * ix_g);
  }

private:
  const int N_GRPS = 0;
  const int N_SPCS = 0;

  std::vector<T> data_;

};

// ============================================================================
} // namespace M1
// ============================================================================

#endif // M1_CONTAINERS_HPP

//
// :D
//