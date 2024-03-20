#ifndef M1_CONTAINERS_HPP
#define M1_CONTAINERS_HPP

#include "m1_macro.hpp"

// c++
#include <vector>

// Athena++ classes headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../athena_tensor.hpp"
#include "../utils/tensor.hpp"

// typedef ====================================================================

// define some types to make everything more readable
namespace {

static const int D = M1_NDIM + 1;
static const int N = M1_NDIM;

typedef AthenaArray< Real>                         AA;
typedef AthenaTensor<Real, TensorSymm::NONE, N, 0> AT_N_sca;
typedef AthenaTensor<Real, TensorSymm::NONE, N, 1> AT_N_vec;
typedef AthenaTensor<Real, TensorSymm::SYM2, N, 2> AT_N_sym;

// Treat group & species as super-indices with this nested structure
typedef AthenaArray<AT_N_sca> GS_AT_N_sca;
typedef AthenaArray<AT_N_vec> GS_AT_N_vec;

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

}

// Class ======================================================================

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
    data_.reserve(N_GRPS * N_SPCS);
  }
  ~GroupSpeciesContainer()
  {
    for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
    for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
    {
      data_[ix_s + N_SPCS * ix_g].DeleteAthenaTensor();
    }
  }

  T & operator()(int const ix_g, int const ix_s)
  {
    return data_[ix_s + N_SPCS * ix_g];
  }

  T operator()(int const ix_g, int const ix_s) const
  {
    return data_[ix_s + N_SPCS * ix_g];
  }

private:
  const int N_GRPS = 0;
  const int N_SPCS = 0;

  std::vector<T> data_;

};


#endif // M1_CONTAINERS_HPP
