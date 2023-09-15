#ifndef LINEAR_ALGEBRA_HPP_
#define LINEAR_ALGEBRA_HPP_
//! \file linear_algebra.hpp
//  \brief small linear algebra functions

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../athena_tensor.hpp"

// ----------------------------------------------------------------------------
// New impl. here
namespace LinearAlgebra {

// compute spatial determinant of a 3x3  matrix
static inline Real Det3Metric(
  Real const gxx, Real const gxy, Real const gxz,
  Real const gyy, Real const gyz, Real const gzz)
{
  return (- SQR(gxz)*gyy + 2*gxy*gxz*gyz - gxx*SQR(gyz) -
          SQR(gxy)*gzz + gxx*gyy*gzz);
}

static inline Real Det3Metric(
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> const & g,
  int const i)
{
  return Det3Metric(
    g(0,0,i), g(0,1,i), g(0,2,i),
    g(1,1,i), g(1,2,i), g(2,2,i)
  );
}

static inline Real Det3Metric(
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> const & g,
  int const k, int const j, int const i)
{
  return Det3Metric(
    g(0,0,k,j,i), g(0,1,k,j,i), g(0,2,k,j,i),
    g(1,1,k,j,i), g(1,2,k,j,i), g(2,2,k,j,i)
  );
}

// compute inverse of a 3x3 matrix
static inline void Inv3Metric(
  Real const detginv,
  Real const gxx, Real const gxy, Real const gxz,
  Real const gyy, Real const gyz, Real const gzz,
  Real * uxx, Real * uxy, Real * uxz,
  Real * uyy, Real * uyz, Real * uzz)
{
  *uxx = (-SQR(gyz) + gyy*gzz)*detginv;
  *uxy = (gxz*gyz  - gxy*gzz)*detginv;
  *uyy = (-SQR(gxz) + gxx*gzz)*detginv;
  *uxz = (-gxz*gyy + gxy*gyz)*detginv;
  *uyz = (gxy*gxz  - gxx*gyz)*detginv;
  *uzz = (-SQR(gxy) + gxx*gyy)*detginv;
}

// compute trace of a rank 2 covariant spatial tensor
static inline Real TraceRank2(
  Real const detginv,
  Real const gxx, Real const gxy, Real const gxz,
  Real const gyy, Real const gyz, Real const gzz,
  Real const Axx, Real const Axy, Real const Axz,
  Real const Ayy, Real const Ayz, Real const Azz)
{
  return detginv*(
    - 2.*Ayz*gxx*gyz + Axx*gyy*gzz +  gxx*(Azz*gyy + Ayy*gzz)
    + 2.*(gxz*(Ayz*gxy - Axz*gyy + Axy*gyz) + gxy*(Axz*gyz - Axy*gzz))
    - Azz*SQR(gxy) - Ayy*SQR(gxz) - Axx*SQR(gyz)
  );
}

// inner product of 3-vec
static inline Real InnerProductSlicedVec3Metric(
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> const & u,
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> const & g,
  int const k, int const j, int const i)
{
  return (u(0,i)*u(0,i)*g(0,0,k,j,i) +
          u(1,i)*u(1,i)*g(1,1,k,j,i) +
          u(2,i)*u(2,i)*g(2,2,k,j,i) +
          2.0*u(0,i)*u(1,i)*g(0,1,k,j,i) +
          2.0*u(0,i)*u(2,i)*g(0,2,k,j,i) +
          2.0*u(1,i)*u(2,i)*g(1,2,k,j,i));
}

// indicial manipulations
static inline void SlicedVecMet3Contraction(
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> & v_res,
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> const & v_src,
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> const & met3_src,
  const int k, const int j, const int i
)
{
  #pragma omp unroll full
  for (int a=0; a<3; ++a)
  {
    v_res(a,i) = (v_src(0,i)*met3_src(a,0,k,j,i) +
                  v_src(1,i)*met3_src(a,1,k,j,i) +
                  v_src(2,i)*met3_src(a,2,k,j,i));
  }
}


}  // namespace LinearAlgebra

// implementation details (for templates) =====================================
// #include "linear_algebra.tpp"
// ============================================================================

//
// :D
//

#endif // LINEAR_ALGEBRA_HPP_
