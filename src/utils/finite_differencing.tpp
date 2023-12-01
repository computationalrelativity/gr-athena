// headers ====================================================================
// c / c++
// #include <algorithm>

// Athena++ headers
#include "finite_differencing.hpp"
// ============================================================================

// ============================================================================
namespace FiniteDifference {

// ctor / dtor ----------------------------------------------------------------
inline Uniform::Uniform(const int nn1, const Real dx1)
{
  Uniform(nn1, 0, dx1, 0);
}

inline Uniform::Uniform(
  const int nn1, const int nn2,  const Real dx1, const Real dx2)
{
  Uniform(nn1, nn2, 0, dx1, dx2, 0);
}

inline Uniform::Uniform(
  const int nn1, const int nn2, const int nn3,
  const Real dx1, const Real dx2, const Real dx3)
{
  diss_scaling = pow(2, -2*NGHOST)*(NGHOST % 2 == 0 ? -1 : 1);

  stride[0] = 1;
  stride[1] = (nn2 > 1) ? nn1 : 0;
  stride[2] = (nn3 > 1) ? nn2 * nn1 : 0;

  idx[0] = 1.0 / dx1;
  idx[1] = (nn2 > 1) ? 1.0 / dx2 : 0.0;
  idx[2] = (nn3 > 1) ? 1.0 / dx3 : 0.0;

#ifdef DBG_SYMMETRIZE_FD
  cidx1[0] = 1.0 / (dx1 * c1::coeff_lcm);
  cidx1[1] = (nn2 > 1) ? (1.0 / (dx2 * c1::coeff_lcm)) : 0.0;
  cidx1[2] = (nn3 > 1) ? (1.0 / (dx3 * c1::coeff_lcm)) : 0.0;

  cidx2[0] = SQR(1.0 / dx1) / c2::coeff_lcm;
  cidx2[1] = (nn2 > 1) ? SQR(1.0 / dx2) / c2::coeff_lcm : 0.0;
  cidx2[2] = (nn3 > 1) ? SQR(1.0 / dx3) / c2::coeff_lcm : 0.0;

  cidx1_lo[0] = 1.0 / (dx1 * c1_lo::coeff_lcm);
  cidx1_lo[1] = (nn2 > 1) ? (1.0 / (dx2 * c1_lo::coeff_lcm)) : 0.0;
  cidx1_lo[2] = (nn3 > 1) ? (1.0 / (dx3 * c1_lo::coeff_lcm)) : 0.0;

  cidxd[0] = diss_scaling / (dx1 * cd::coeff_lcm);
  cidxd[1] = (nn2 > 1) ? (diss_scaling / (dx2 * cd::coeff_lcm)) : 0.0;
  cidxd[2] = (nn3 > 1) ? (diss_scaling / (dx3 * cd::coeff_lcm)) : 0.0;

  // lop left / right
  lidx_l1[0] = 1.0 / (dx1 * ll1::coeff_lcm);
  lidx_l1[1] = (nn2 > 1) ? (1.0 / (dx2 * ll1::coeff_lcm)) : 0.0;
  lidx_l1[2] = (nn3 > 1) ? (1.0 / (dx3 * ll1::coeff_lcm)) : 0.0;

  lidx_r1[0] = 1.0 / (dx1 * lr1::coeff_lcm);
  lidx_r1[1] = (nn2 > 1) ? (1.0 / (dx2 * lr1::coeff_lcm)) : 0.0;
  lidx_r1[2] = (nn3 > 1) ? (1.0 / (dx3 * lr1::coeff_lcm)) : 0.0;

#endif // DBG_SYMMETRIZE_FD
}

inline Real Uniform::Dx(int dir, Real & u)
{
#ifdef DBG_SYMMETRIZE_FD

  // // 1 NN
  // Real * pu = &u;
  // Real out = (
  //   (-pu[-1 * stride[dir]] + pu[1 * stride[dir]])
  // );
  // return out * idx[dir] / 2.0;

  // // 2 NN
  // Real * pu = &u;
  // Real out = (
  //   1.0 * ( pu[-2 * stride[dir]] - pu[2 * stride[dir]]) +
  //   8.0 * (-pu[-1 * stride[dir]] + pu[1 * stride[dir]])
  // );
  // return out * idx[dir] / 12.0;

  // // 3 NN
  // Real * pu = &u;
  // Real out = (
  //    1.0 * (-pu[-3 * stride[dir]] + pu[3 * stride[dir]]) +
  //    9.0 * ( pu[-2 * stride[dir]] - pu[2 * stride[dir]]) +
  //   45.0 * (-pu[-1 * stride[dir]] + pu[1 * stride[dir]])
  // );
  // return out * idx[dir] * (1. / 60.0);

  // Real * pu_l = &u - c1::offset*stride[dir];
  // Real * pu_r = &pu_l[c1::width-1];

  // Real out(0.);
  // for(int n1 = 0; n1 < c1::nghost; ++n1)
  // {
  //   const int n_ = n1*stride[dir];
  //   out += c1::coeff[n1] * (pu_l[n_] - pu_r[-n_]);
  //   // int const n2  = c1::width - n1 - 1;
  //   // out += c1::coeff[n1] * (pu[n1*stride[dir]] - pu[n2*stride[dir]]);
  // }

  Real * pu = &u - c1::offset*stride[dir];

  Real out(0.);
  for(int n1 = 0; n1 < c1::nghost; ++n1)
  {
    int const n2  = c1::width - n1 - 1;
    out += c1::coeff[n1] * (pu[n1*stride[dir]] - pu[n2*stride[dir]]);
  }

  return out * cidx1[dir];
#else

  Real * pu = &u - s1::offset*stride[dir];

  Real out(0.);
  for(int n1 = 0; n1 < s1::nghost; ++n1)
  {
    int const n2  = s1::width - n1 - 1;
    Real const c1 = s1::coeff[n1] * pu[n1*stride[dir]];
    Real const c2 = s1::coeff[n2] * pu[n2*stride[dir]];
    out += (c1 + c2);
  }
  out += s1::coeff[s1::nghost] * pu[s1::nghost*stride[dir]];
  return out * idx[dir];

#endif // DBG_SYMMETRIZE_FD

}

inline Real Uniform::Ds(int dir, Real & u)
{
#ifdef DBG_SYMMETRIZE_FD

  Real * pu = &u - c1_lo::offset*stride[dir];

  Real out(0.);
  for(int n1 = 0; n1 < c1_lo::nghost; ++n1)
  {
    int const n2  = c1_lo::width - n1 - 1;
    out += c1_lo::coeff[n1] * (pu[n1*stride[dir]] - pu[n2*stride[dir]]);
  }
  return out * cidx1_lo[dir];

#else

  Real * pu = &u;
  return 0.5 * idx[dir] * (pu[stride[dir]] - pu[-stride[dir]]);

#endif // DBG_SYMMETRIZE_FD
}

inline Real Uniform::Lx(int dir, Real & vx, Real & u)
{
#ifdef DBG_SYMMETRIZE_FD

  Real * pu = &u;

  Real dl(0.);
  for(int n = 0; n < ll1::width; ++n)
  {
    dl += ll1::coeff[n] * pu[(n - ll1::offset)*stride[dir]];
  }

  Real dr(0.);
  for(int n = lr1::width-1; n >= 0; --n)
  {
    dr += lr1::coeff[n] * pu[(n - lr1::offset)*stride[dir]];
  }

  // lidx_l1[dir] == lidx_r1[dir]
  return ((vx < 0) ? (vx * dl) : (vx * dr)) * lidx_l1[dir];

#else

  Real * pu = &u;

  Real dl(0.);
  for(int n = 0; n < sl::width; ++n)
  {
    dl += sl::coeff[n] * pu[(n - sl::offset)*stride[dir]];
  }

  Real dr(0.);
  for(int n = sr::width-1; n >= 0; --n)
  {
    dr += sr::coeff[n] * pu[(n - sr::offset)*stride[dir]];
  }

  return ((vx < 0) ? (vx * dl) : (vx * dr)) * idx[dir];

#endif // DBG_SYMMETRIZE_FD
}

inline Real Uniform::Dxx(int dir, Real & u)
{

#ifdef DBG_SYMMETRIZE_FD

  // // 1 NN
  // Real * pu = &u;
  // Real out = (
  //   (pu[-1 * stride[dir]] + pu[1 * stride[dir]])
  // ) - 2. * pu[0];
  // return out * SQR(idx[dir]);

  // // 2 NN
  // Real * pu = &u;
  // Real out = (
  //   + 1.0 * (-pu[-2 * stride[dir]] - pu[2 * stride[dir]])
  //   + 16.0 * (pu[-1 * stride[dir]] + pu[1 * stride[dir]])
  // ) - 30.0 * pu[0];
  // return out * SQR(idx[dir]) / 12.0;

  // // 3 NN
  // Real * pu = &u;
  // Real out = (
  //   + 1.0 *   ( pu[-3 * stride[dir]] + pu[3 * stride[dir]])
  //   + 13.5 *  (-pu[-2 * stride[dir]] - pu[2 * stride[dir]])
  //   + 135.0 * ( pu[-1 * stride[dir]] + pu[1 * stride[dir]])
  // ) - 245.0 * pu[0];
  // return out * SQR(idx[dir]) / 90.0;

  Real * pu = &u - c2::offset*stride[dir];

  Real out(0.);
  for(int n1 = 0; n1 < c2::nghost; ++n1)
  {
    int const n2  = c2::width - n1 - 1;
    out += c2::coeff[n1] * (pu[n1*stride[dir]] + pu[n2*stride[dir]]);
  }
  out += c2::coeff[c2::nghost] * pu[c2::nghost*stride[dir]];
  return out * cidx2[dir];

#else

  Real * pu = &u - s2::offset*stride[dir];

  Real out(0.);
  for(int n1 = 0; n1 < s2::nghost; ++n1)
  {
    int const n2  = s2::width - n1 - 1;
    Real const c1 = s2::coeff[n1] * pu[n1*stride[dir]];
    Real const c2 = s2::coeff[n2] * pu[n2*stride[dir]];
    out += (c1 + c2);
  }
  out += s2::coeff[s2::nghost] * pu[s2::nghost*stride[dir]];
  return out * SQR(idx[dir]);

#endif // DBG_SYMMETRIZE_FD

}

inline Real Uniform::Dxy(int dirx, int diry, Real & u)
{
#ifdef DBG_SYMMETRIZE_FD

  Real * pu = &u - c1::offset*(stride[dirx] + stride[diry]);
  Real out(0.);

  for (int nx1 = 0; nx1 < c1::nghost; ++nx1)
  {
    const int nx2 = c1::width - nx1 - 1;

    for (int ny1 = 0; ny1 < c1::nghost; ++ny1)
    {
      const int ny2 = c1::width - ny1 - 1;

      const Real v11 = pu[nx1*stride[dirx] + ny1*stride[diry]];
      const Real v22 = pu[nx2*stride[dirx] + ny2*stride[diry]];

      const Real v21 = -pu[nx2*stride[dirx] + ny1*stride[diry]];
      const Real v12 = -pu[nx1*stride[dirx] + ny2*stride[diry]];

      const Real c11 = c1::coeff[nx1] * c1::coeff[ny1];
      out += c11 * FloatingPoint::sum_associative(v11, v22, v21, v12);

      // compensated
      // out += 0.5 * c11 * (std::max({ca, cb, cc}) + std::min({ca, cb, cc}));
    }
  }

  return out * cidx1[dirx] * cidx1[diry];

#else

  Real * pu = &u - s1::offset*(stride[dirx] + stride[diry]);
  Real out(0.);

  for(int nx1 = 0; nx1 < s1::nghost; ++nx1) {
    int const nx2 = s1::width - nx1 - 1;
    for(int ny1 = 0; ny1 < s1::nghost; ++ny1) {
      int const ny2 = s1::width - ny1 - 1;

      Real const c11 = s1::coeff[nx1] * s1::coeff[ny1] * pu[nx1*stride[dirx] + ny1*stride[diry]];
      Real const c12 = s1::coeff[nx1] * s1::coeff[ny2] * pu[nx1*stride[dirx] + ny2*stride[diry]];
      Real const c21 = s1::coeff[nx2] * s1::coeff[ny1] * pu[nx2*stride[dirx] + ny1*stride[diry]];
      Real const c22 = s1::coeff[nx2] * s1::coeff[ny2] * pu[nx2*stride[dirx] + ny2*stride[diry]];

      Real const ca = (1./6.)*((c11 + c12) + (c21 + c22));
      Real const cb = (1./6.)*((c11 + c21) + (c12 + c22));
      Real const cc = (1./6.)*((c11 + c22) + (c12 + c21));

      out += ((ca + cb) + cc) + ((ca + cc) + cb);
    }
    int const ny = s1::nghost;

    Real const c1 = s1::coeff[nx1] * s1::coeff[ny] * pu[nx1*stride[dirx] + ny*stride[diry]];
    Real const c2 = s1::coeff[nx2] * s1::coeff[ny] * pu[nx2*stride[dirx] + ny*stride[diry]];

    out += (c1 + c2);
  }
  int const nx = s1::nghost;
  for(int ny1 = 0; ny1 < s1::nghost; ++ny1) {
    int const ny2 = s1::width - ny1 - 1;

    Real const c1 = s1::coeff[nx] * s1::coeff[ny1] * pu[nx*stride[dirx] + ny1*stride[diry]];
    Real const c2 = s1::coeff[nx] * s1::coeff[ny2] * pu[nx*stride[dirx] + ny2*stride[diry]];

    out += (c1 + c2);
  }
  int const ny = s1::nghost;
  out += s1::coeff[nx] * s1::coeff[ny] * pu[nx*stride[dirx] + ny*stride[diry]];

  // const Real diff = std::abs(Dxy_(dirx, diry, u) - out * idx[dirx] * idx[diry]);
  // if (diff > 1e-14)
  //   std::cout << "diff: " << diff << std::endl;

  return out * idx[dirx] * idx[diry];

#endif // DBG_SYMMETRIZE_FD
}

inline Real Uniform::Diss(int dir, Real & u, Real diss)
{
#ifdef DBG_SYMMETRIZE_FD

  Real * pu = &u - cd::offset*stride[dir];

  Real out(0.);
  // #pragma omp simd reduction(+:out)
  for(int n1 = 0; n1 < cd::nghost; ++n1)
  {
    int const n2  = cd::width - n1 - 1;
    out += cd::coeff[n1] * (pu[n1*stride[dir]] + pu[n2*stride[dir]]);
  }
  out += cd::coeff[cd::nghost] * pu[cd::nghost*stride[dir]];
  return out * cidxd[dir] * diss;

#else

  Real * pu = &u - sd::offset*stride[dir];

  Real out(0.);
  for(int n1 = 0; n1 < sd::nghost; ++n1)
  {
    int const n2  = sd::width - n1 - 1;
    Real const c1 = sd::coeff[n1] * pu[n1*stride[dir]];
    Real const c2 = sd::coeff[n2] * pu[n2*stride[dir]];
    out += (c1 + c2);
  }
  out += sd::coeff[sd::nghost] * pu[sd::nghost*stride[dir]];

  return out * idx[dir] * diss * diss_scaling;

#endif // DBG_SYMMETRIZE_FD
}

inline Real Uniform::Dx_ho(int dir, Real & u)
{
  Real * pu = &u - cd_hd::offset*stride[dir];

  Real out(0.);
  for(int n1 = 0; n1 < cd_hd::nghost; ++n1) {
    int const n2  = cd_hd::width - n1 - 1;
    Real const c1 = cd_hd::coeff[n1] * pu[n1*stride[dir]];
    Real const c2 = cd_hd::coeff[n2] * pu[n2*stride[dir]];
    out += (c1 + c2);
  }
  out += cd_hd::coeff[cd_hd::nghost] * pu[cd_hd::nghost*stride[dir]];
  return out * POW4(idx[dir])*POW3(idx[dir]);
}

} // namespace FiniteDifference
// ============================================================================
