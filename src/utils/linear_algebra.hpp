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
// New impl. here; leverage polymorphism to make this more readable elsewhere
namespace LinearAlgebra {

// compute spatial determinant of a 3x3  matrix
inline Real Det3Metric(
  Real const gxx, Real const gxy, Real const gxz,
  Real const gyy, Real const gyz, Real const gzz)
{
  return (- SQR(gxz)*gyy + 2*gxy*gxz*gyz - gxx*SQR(gyz) -
          SQR(gxy)*gzz + gxx*gyy*gzz);
}

inline Real Det3Metric(
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> const & g,
  int const i)
{
  return Det3Metric(
    g(0,0,i), g(0,1,i), g(0,2,i),
    g(1,1,i), g(1,2,i), g(2,2,i)
  );
}

inline Real Det3Metric(
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> const & g,
  int const k, int const j, int const i)
{
  return Det3Metric(
    g(0,0,k,j,i), g(0,1,k,j,i), g(0,2,k,j,i),
    g(1,1,k,j,i), g(1,2,k,j,i), g(2,2,k,j,i)
  );
}

// compute inverse of a 3x3 symmetric matrix
inline void Inv3Metric(
  Real const oodetg,
  Real const gxx, Real const gxy, Real const gxz,
  Real const gyy, Real const gyz, Real const gzz,
  Real * uxx, Real * uxy, Real * uxz,
  Real * uyy, Real * uyz, Real * uzz)
{
  *uxx = (-SQR(gyz) + gyy*gzz)*oodetg;
  *uxy = (gxz*gyz  - gxy*gzz)*oodetg;
  *uyy = (-SQR(gxz) + gxx*gzz)*oodetg;
  *uxz = (-gxz*gyy + gxy*gyz)*oodetg;
  *uyz = (gxy*gxz  - gxx*gyz)*oodetg;
  *uzz = (-SQR(gxy) + gxx*gyy)*oodetg;
}

inline void Inv3Metric(
  AthenaTensor<Real, TensorSymm::NONE, 3, 0> const & oodetg,
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> const & g_dd,
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> & g_uu,
  int const i)
{
  Inv3Metric(
    oodetg(i),
    g_dd(0,0,i), g_dd(0,1,i), g_dd(0,2,i),
    g_dd(1,1,i), g_dd(1,2,i), g_dd(2,2,i),
    &g_uu(0,0,i), &g_uu(0,1,i), &g_uu(0,2,i),
    &g_uu(1,1,i), &g_uu(1,2,i), &g_uu(2,2,i)
  );
}

inline void Inv3Metric(
  AthenaTensor<Real, TensorSymm::NONE, 3, 0> const & oodetg,
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> const & g_dd,
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> & g_uu,
  int const il, int const iu)
{
  for (int i=il; i<=iu; ++i)
  {
    Inv3Metric(
      oodetg(i),
      g_dd(0,0,i), g_dd(0,1,i), g_dd(0,2,i),
      g_dd(1,1,i), g_dd(1,2,i), g_dd(2,2,i),
      &g_uu(0,0,i), &g_uu(0,1,i), &g_uu(0,2,i),
      &g_uu(1,1,i), &g_uu(1,2,i), &g_uu(2,2,i)
    );
  }
}

// BD: TODO shift target to first slot
inline void Inv3Metric(
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> const & g_dd,
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> & g_uu,
  int const il, int const iu)
{
  for (int i=il; i<=iu; ++i)
  {
    const Real oodetg = 1. / Det3Metric(g_dd, i);
    Inv3Metric(
      oodetg,
      g_dd(0,0,i), g_dd(0,1,i), g_dd(0,2,i),
      g_dd(1,1,i), g_dd(1,2,i), g_dd(2,2,i),
      &g_uu(0,0,i), &g_uu(0,1,i), &g_uu(0,2,i),
      &g_uu(1,1,i), &g_uu(1,2,i), &g_uu(2,2,i)
    );
  }
}

// compute trace of a rank 2 covariant spatial tensor
inline Real TraceRank2(
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
inline Real InnerProductSlicedVec3Metric(
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

inline Real InnerProductSlicedVec3Metric(
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> const & u,
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> const & g,
  int const i)
{
  return (u(0,i)*u(0,i)*g(0,0,i) +
          u(1,i)*u(1,i)*g(1,1,i) +
          u(2,i)*u(2,i)*g(2,2,i) +
          2.0*u(0,i)*u(1,i)*g(0,1,i) +
          2.0*u(0,i)*u(2,i)*g(0,2,i) +
          2.0*u(1,i)*u(2,i)*g(1,2,i));
}

inline void InnerProductSlicedVec3Metric(
  AthenaTensor<Real, TensorSymm::NONE, 3, 0> & norm2_v,
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> const & v,
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> const & g,
  int const il, int const iu)
{
  for (int i=il; i<=iu; ++i)
  {
    norm2_v(i) = InnerProductSlicedVec3Metric(
      v, g, i
    );
  }
}


// indicial manipulations
inline void SlicedVecMet3Contraction(
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> & v_dst,
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> const & v_src,
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> const & met3_src,
  const int k, const int j, const int i
)
{
  // #pragma omp unroll full
  for (int a=0; a<3; ++a)
  {
    v_dst(a,i) = (v_src(0,i)*met3_src(a,0,k,j,i) +
                  v_src(1,i)*met3_src(a,1,k,j,i) +
                  v_src(2,i)*met3_src(a,2,k,j,i));
  }
}

inline void SlicedVecMet3Contraction(
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> & v_dst,
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> const & v_src,
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> const & met3_src,
  const int il, const int iu
)
{
  // #pragma omp unroll full
  for (int a=0; a<3; ++a)
  {
    for (int i=il; i<=iu; ++i)
    {
      v_dst(a,i) = (v_src(0,i)*met3_src(a,0,i) +
                    v_src(1,i)*met3_src(a,1,i) +
                    v_src(2,i)*met3_src(a,2,i));
    }
  }
}

template<typename T>
inline void ApplyLinearTransform(
  const AthenaArray<T> & T_mat,
  AthenaArray<T>       & tar_vec,
  const AthenaArray<T> & src_vec
)
{
  const int D = T_mat.GetDim1();
  for (int jx=0; jx<D; ++jx)
  {
    tar_vec(jx) = 0;
    for (int ix=0; ix<D; ++ix)
    {
      tar_vec(jx) += T_mat(jx,ix) * src_vec(ix);
    }
  }
}

template<typename T,int D>
inline void ApplyLinearTransform(
  const AthenaArray<T> & T_mat,
  AthenaTensor<T, TensorSymm::NONE, D, 1>       & dst,
  const AthenaTensor<T, TensorSymm::NONE, D, 1> & src,
  const int il, const int iu
)
{

  for (int a=0; a<D; ++a)
  for (int i=il; i<=iu; ++i)
  {
    dst(a,i) = 0;
  }

  for (int a=0; a<D; ++a)
  for (int b=0; b<D; ++b)
  for (int i=il; i<=iu; ++i)
  {
    dst(a,i) += T_mat(a,b) * src(b,i);
  }

}

template<typename T,int D>
inline void ApplyLinearTransform(
  const AthenaArray<T> & T_mat_A,
  const AthenaArray<T> & T_mat_B,
  AthenaTensor<T, TensorSymm::SYM2, D, 2>       & dst,
  const AthenaTensor<T, TensorSymm::SYM2, D, 2> & src,
  const int il, const int iu
)
{
  for (int b=0; b<D; ++b)
  for (int a=0; a<=b; ++a)
  for (int i=il; i<=iu; ++i)
  {
    dst(a,b,i) = 0;
  }

  for (int b=0; b<D; ++b)
  for (int a=0; a<=b; ++a)
  for (int l=0; l<D; ++l)
  for (int m=0; m<D; ++m)
  for (int i=il; i<=iu; ++i)
  {
    dst(a,b,i) += T_mat_A(a,l) * T_mat_B(b,m) * src(l,m,i);
  }

}

template<typename T,int D>
inline void ApplyLinearTransform(
  const AthenaArray<T> & T_mat_A,
  const AthenaArray<T> & T_mat_B,
  const AthenaArray<T> & T_mat_C,
  AthenaTensor<T, TensorSymm::SYM2, D, 3>       & dst,
  const AthenaTensor<T, TensorSymm::SYM2, D, 3> & src,
  const int il, const int iu
)
{
  for (int c=0; c<D; ++c)
  for (int b=0; b<D; ++b)
  for (int a=0; a<=b; ++a)
  for (int i=il; i<=iu; ++i)
  {
    dst(c,a,b,i) = 0;
  }

  for (int c=0; c<D; ++c)
  for (int b=0; b<D; ++b)
  for (int a=0; a<=b; ++a)
  for (int l=0; l<D; ++l)
  for (int n=0; n<D; ++n)
  for (int m=0; m<D; ++m)
  for (int i=il; i<=iu; ++i)
  {
    dst(c,a,b,i) += T_mat_A(c,l) * T_mat_B(a,m) * T_mat_C(b,n) * src(l,m,n,i);
  }


}

// compute norm-squared with (v_u, g_dd) or (v_d, g_uu)
template<typename T>
void MetricNorm2Vector(
  AthenaTensor<T, TensorSymm::NONE,       3, 0> & v_norm2,
  const AthenaTensor<T, TensorSymm::NONE, 3, 1> & v_d,
  const AthenaTensor<T, TensorSymm::SYM2, 3, 2> & g_dd,
  const int il, const int iu
)
{
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> v_u( iu+1);
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> g_uu(iu+1);

  // switch form of metric
  LinearAlgebra::Inv3Metric(g_dd, g_uu, il, iu);
  // raise idx on vec
  LinearAlgebra::SlicedVecMet3Contraction(
    v_u, v_d, g_uu,
    il, iu
  );
  // compute norm squared
  LinearAlgebra::InnerProductSlicedVec3Metric(
    v_norm2, v_u, g_dd,
    il, iu
  );
}

// Construct space-time (ST) metric from ADM decomposed quantities
template<typename T, int D>
void Assemble_ST_Metric_dd(
  AthenaTensor<T, TensorSymm::SYM2, D, 2>         & st_g_dd,
  const AthenaTensor<T, TensorSymm::SYM2, D-1, 2> & sp_g_dd,
  const AthenaTensor<T, TensorSymm::NONE, D-1, 0> & alpha,
  const AthenaTensor<T, TensorSymm::NONE, D-1, 1> & sp_beta_d,
  const int il, const int iu)
{
  // need <beta,beta>_sp_g
  AthenaTensor<Real, TensorSymm::NONE, D-1, 0> sp_norm2_beta(iu+1);
  MetricNorm2Vector(sp_norm2_beta, sp_beta_d, sp_g_dd, il, iu);

  for (int i=il; i<=iu; ++i)
  {
    st_g_dd(0,0,i) = -SQR(alpha(i)) + sp_norm2_beta(i);
  }

  for (int b=0; b<D-1; ++b)  // spatial ranges
  {
    for (int i=il; i<=iu; ++i)
    {
      st_g_dd(0,b+1,i) = sp_beta_d(b,i);
    }

    for (int a=0; a<=b; ++a)
    for (int i=il; i<=iu; ++i)
    {
      st_g_dd(a+1,b+1,i) = sp_g_dd(a,b,i);
    }
  }
}

template<typename T, int D>
void Assemble_ST_Metric_uu(
  AthenaTensor<T, TensorSymm::SYM2, D, 2>         & st_g_uu,
  const AthenaTensor<T, TensorSymm::SYM2, D-1, 2> & sp_g_uu,
  const AthenaTensor<T, TensorSymm::NONE, D-1, 0> & alpha,
  const AthenaTensor<T, TensorSymm::NONE, D-1, 1> & sp_beta_u,
  const int il, const int iu)
{
  AthenaTensor<Real, TensorSymm::NONE, D-1, 0> ooalpha2(iu+1);

  for (int i=il; i<=iu; ++i)
  {
    ooalpha2(i) = 1. / SQR(alpha(i));
  }

  for (int i=il; i<=iu; ++i)
  {
    st_g_uu(0,0,i) = -ooalpha2(i);
  }

  for (int b=0; b<D-1; ++b)  // spatial ranges
  {
    for (int a=0; a<=b; ++a)
    for (int i=il; i<=iu; ++i)
    {
      st_g_uu(a+1,b+1,i) = (sp_g_uu(a,b,i) -
                            ooalpha2(i)*sp_beta_u(a,i)*sp_beta_u(b,i));
    }

    for (int i=il; i<=iu; ++i)
    {
      st_g_uu(0,b+1,i) = sp_beta_u(b,i) * ooalpha2(i);
    }

  }
}

template<typename T, int D>
void ExtractFrom_ST_Metric_dd(
  const AthenaTensor<T, TensorSymm::SYM2, D, 2> & st_g_dd,
  AthenaTensor<T, TensorSymm::SYM2, D-1, 2>     & sp_g_dd,
  AthenaTensor<T, TensorSymm::NONE, D-1, 0>     & alpha,
  AthenaTensor<T, TensorSymm::NONE, D-1, 1>     & sp_beta_d,
  const int il, const int iu
)
{
  for (int a=0; a<D-1; ++a)
  for (int i=il; i<=iu; ++i)
  {
    sp_beta_d(a,i) = st_g_dd(0,a+1,i);
  }

  for (int a=0; a<D-1; ++a)
  for (int b=0; b<=a; ++b)
  for (int i=il; i<=iu; ++i)
  {
    sp_g_dd(b,a,i) = st_g_dd(b+1,a+1,i);
  }

  // lapse needs <beta,beta>_sp_g  (must come after above)
  AthenaTensor<Real, TensorSymm::NONE, D-1, 0> sp_norm2_beta(iu+1);
  MetricNorm2Vector(sp_norm2_beta, sp_beta_d, sp_g_dd, il, iu);

  for (int i=il; i<=iu; ++i)
  {
    alpha(i) = std::sqrt(sp_norm2_beta(i)-st_g_dd(0,0,i));
  }


}

template<typename T, int D>
void ExtractFrom_ST_Metric_uu(
  const AthenaTensor<T, TensorSymm::SYM2, D, 2> & st_g_uu,
  AthenaTensor<T, TensorSymm::SYM2, D-1, 2>     & sp_g_uu,
  AthenaTensor<T, TensorSymm::NONE, D-1, 0>     & alpha,
  AthenaTensor<T, TensorSymm::NONE, D-1, 1>     & sp_beta_u,
  const int il, const int iu
)
{
  for (int i=il; i<=iu; ++i)
  {
    alpha(i) = 1. / std::sqrt(-st_g_uu(0,0,i));
  }

  for (int a=0; a<D-1; ++a)
  for (int i=il; i<=iu; ++i)
  {
    sp_beta_u(a,i) = st_g_uu(0,a+1,i) * SQR(alpha(i));
  }

  for (int a=0; a<D-1; ++a)
  for (int b=0; b<=a; ++b)
  for (int i=il; i<=iu; ++i)
  {
    sp_g_uu(b,a,i) = (st_g_uu(b+1,a+1,i)+
                      st_g_uu(0,a+1,i)*sp_beta_u(b,i));
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
