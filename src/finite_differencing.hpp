#ifndef FINITE_DIFFERENCING_HPP_
#define FINITE_DIFFERENCING_HPP_

//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file finite_differencing.hpp
//  \brief High-performance finite-differencing kernel

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "utils/floating_point.hpp"


namespace FiniteDifference {

// Centered finite differencing stencils
// * degree : Degree of the derivative, eg, 1 for 1st derivative
// * nghost : Number of ghost points used for the derivative
template<int degree_, int nghost_>
class FDCenteredStencil {
public:
  // Degree of the derivative to be approximated
  enum {degree = degree_};
  // Number of ghost points required for the differencing
  enum {nghost = nghost_};
  // Position at which the derivative is computed wrt the beginning of the stencil
  enum {offset = nghost_};
  // Width of the stencil
  enum {width = 2*nghost_ + 1};
  // Finite differencing coefficients
  static Real const coeff[width];
};

// Choose the right order for the dissipation operators
template<int nghost_>
class FDDissChoice {
public:
  enum {degree = 2*(nghost_ + 1)};
  enum {nghost = nghost_ + 1};
};

// Choose the left or right biased stencil
template<int degree_, int nghost_>
class FDBiasedChoice {
public:
  enum {degree = degree_};
  enum {nghost = nghost_+1};
  enum {lopsize = nghost_};
};
template<>
class FDBiasedChoice<1, 1> {
public:
  enum {degree = 1};
  enum {nghost = 2};
  enum {lopsize = 0};
};
template<>
class FDBiasedChoice<1, 2> {
public:
  enum {degree = 1};
  enum {nghost = 3};
  enum {lopsize = 1};
};
template<>
class FDBiasedChoice<1, 3> {
public:
  enum {degree = 1};
  enum {nghost = 4};
  enum {lopsize = 2};
};
template<>
class FDBiasedChoice<1, 4> {
public:
  enum {degree = 1};
  enum {nghost = 5};
  enum {lopsize = 3};
};

// Left-biased finite differencing stencils
// * degree  : Degree of the derivative, eg, 1 for 1st derivative
// * nghost  : Number of ghost points used for the derivative
// * lopsize : Number of points to the right of the derivative point
template<int degree_, int nghost_, int lopsize_>
class FDLeftBiasedStencil {
public:
  // Degree of the derivative to be approximated
  enum {degree = degree_};
  // Number of ghost points required for the differencing
  enum {nghost = nghost_};
  // Position at which the derivative is computed wrt the beginning of the stencil
  enum {offset = nghost_};
  // Width of the stencil
  enum {width = nghost_ + lopsize_ + 1};
  // Finite differencing coefficients
  static Real const coeff[width];
};

// Right-biased finite differencing stencils
// * degree : Degree of the derivative, eg, 1 for 1st derivative
// * nghost : Number of ghost points used for the derivative
// * lopsize : Number of points to the left of the derivative point
template<int degree_, int nghost_, int lopsize_>
class FDRightBiasedStencil {
public:
  // Degree of the derivative to be approximated
  enum {degree = degree_};
  // Number of ghost points required for the differencing
  enum {nghost = nghost_};
  // Position at which the derivative is computed wrt the beginning of the stencil
  enum {offset = lopsize_};
  // Width of the stencil
  enum {width = nghost_ + lopsize_ + 1};
  // Finite differencing coefficients
  static Real const coeff[width];
};

#ifdef DBG_SYMMETRIZE_FD
// Rewrite with denominator LCM factoring to mitigate some error ==============

// Odd degree centered finite differencing stencils
// * degree : Degree of the derivative, eg, 1, 3, ...
// * nghost : Number of ghost points used for the derivative
template<int degree_, int nghost_>
class FDStencilCenteredDegreeOdd {
public:
  // Degree of the derivative to be approximated
  enum {degree = degree_};
  // Number of ghost points required for the differencing
  enum {nghost = nghost_};
  // Position at which the derivative is computed wrt the beginning of the
  // stencil
  enum {offset = nghost_};
  // width of stencil
  enum {width = 2 * nghost_ + 1};
  // coefficient data width
  enum {coeff_width = nghost_};
  // Finite differencing coefficients (these need to be divided by coeff_lcm)
  static Real const coeff[coeff_width];
  static Real const coeff_lcm;
};

// Even degree centered finite differencing stencils
// * degree : Degree of the derivative, eg, 2, 4, ...
// * nghost : Number of ghost points used for the derivative
template<int degree_, int nghost_>
class FDStencilCenteredDegreeEven {
public:
  // Degree of the derivative to be approximated
  enum {degree = degree_};
  // Number of ghost points required for the differencing
  enum {nghost = nghost_};
  // Position at which the derivative is computed wrt the beginning of the
  // stencil
  enum {offset = nghost_};
  // width of stencil
  enum {width = 2 * nghost_ + 1};
  // coefficient data width
  enum {coeff_width = nghost_ + 1};
  // Finite differencing coefficients (these need to be divided by coeff_lcm)
  static Real const coeff[coeff_width];
  static Real const coeff_lcm;
};

// Lop-sided finite differencing stencils
// Left:  [o o o x o o o] -> [o o o o x o o - -]
// Right: [o o o x o o o] -> [- - o o x o o o o]


// Left-biased finite differencing stencils
// * degree  : Degree of the derivative, eg, 1 for 1st derivative
// * nghost  : Number of ghost points used for the derivative
// * lopsize : Number of points to the right of the derivative point
template<int degree_, int nghost_, int lopsize_>
class FDStencilBiasedLeft {
public:
  // Degree of the derivative to be approximated
  enum {degree = degree_};
  // Number of ghost points required for the differencing
  enum {nghost = nghost_};
  // Position at which the derivative is computed wrt the beginning of the stencil
  enum {offset = nghost_};
  // Width of the stencil
  enum {width = nghost_ + lopsize_ + 1};
  // Finite differencing coefficients
  static Real const coeff[width];
  static Real const coeff_lcm;
};

// Right-biased finite differencing stencils
// * degree : Degree of the derivative, eg, 1 for 1st derivative
// * nghost : Number of ghost points used for the derivative
// * lopsize : Number of points to the left of the derivative point
template<int degree_, int nghost_, int lopsize_>
class FDStencilBiasedRight {
public:
  // Degree of the derivative to be approximated
  enum {degree = degree_};
  // Number of ghost points required for the differencing
  enum {nghost = nghost_};
  // Position at which the derivative is computed wrt the beginning of the stencil
  enum {offset = lopsize_};
  // Width of the stencil
  enum {width = nghost_ + lopsize_ + 1};
  // Finite differencing coefficients
  static Real const coeff[width];
  static Real const coeff_lcm;
};

#endif // DBG_SYMMETRIZE_FD


class Uniform
{
  // internal definitions -----------------------------------------------------
  private:
    int stride[3];
    Real idx[3];
    Real diss_scaling;

#ifdef DBG_SYMMETRIZE_FD
    // 1st deg derivative stencil
    typedef FDStencilCenteredDegreeOdd<1, NGHOST-1> c1;
    Real cidx1[3];

    // 2nd deg derivative stencil
    typedef FDStencilCenteredDegreeEven<2, NGHOST-1> c2;
    Real cidx2[3];

    // 1st deg derivative stencil, low order
    typedef FDStencilCenteredDegreeOdd<1, 1> c1_lo;
    Real cidx1_lo[3];

    // diss deg derivative stencil
    typedef FDStencilCenteredDegreeEven<2 * NGHOST, NGHOST> cd;
    Real cidxd[3];

    // lop-sided
    typedef FDStencilBiasedLeft<
        FDBiasedChoice<1, NGHOST-1>::degree,
        FDBiasedChoice<1, NGHOST-1>::nghost,
        FDBiasedChoice<1, NGHOST-1>::lopsize
      > ll1;
    Real lidx_l1[3];

    typedef FDStencilBiasedRight<
        FDBiasedChoice<1, NGHOST-1>::degree,
        FDBiasedChoice<1, NGHOST-1>::nghost,
        FDBiasedChoice<1, NGHOST-1>::lopsize
      > lr1;
    Real lidx_r1[3];
#else
    // 1st derivative stecil
    typedef FDCenteredStencil<1, NGHOST-1> s1;
    // 2nd derivative stencil
    typedef FDCenteredStencil<2, NGHOST-1> s2;
    // dissipation operator
    typedef FDCenteredStencil<
      FDDissChoice<NGHOST-1>::degree,
      FDDissChoice<NGHOST-1>::nghost
      > sd;
    // left-biased derivative
    typedef FDLeftBiasedStencil<
        FDBiasedChoice<1, NGHOST-1>::degree,
        FDBiasedChoice<1, NGHOST-1>::nghost,
        FDBiasedChoice<1, NGHOST-1>::lopsize
      > sl;
    // right-biased derivative
    typedef FDRightBiasedStencil<
        FDBiasedChoice<1, NGHOST-1>::degree,
        FDBiasedChoice<1, NGHOST-1>::nghost,
        FDBiasedChoice<1, NGHOST-1>::lopsize
      > sr;
#endif // DBG_SYMMETRIZE_FD

  // expose functionality -----------------------------------------------------
  public:

    // 1st derivative (high order centered)
#ifdef DBG_SYMMETRIZE_FD
    inline Real Dx(int dir, Real & u) {
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
      // return out * idx[dir] / 60.0;

      Real * pu = &u - c1::offset*stride[dir];

      Real out(0.);
      for(int n1 = 0; n1 < c1::nghost; ++n1) {
        int const n2  = c1::width - n1 - 1;
        out += c1::coeff[n1] * (pu[n1*stride[dir]] - pu[n2*stride[dir]]);
      }
      return out * cidx1[dir];
    }
#else
    inline Real Dx(int dir, Real & u) {
      Real * pu = &u - s1::offset*stride[dir];

      Real out(0.);
      for(int n1 = 0; n1 < s1::nghost; ++n1) {
        int const n2  = s1::width - n1 - 1;
        Real const c1 = s1::coeff[n1] * pu[n1*stride[dir]];
        Real const c2 = s1::coeff[n2] * pu[n2*stride[dir]];
        out += (c1 + c2);
      }
      out += s1::coeff[s1::nghost] * pu[s1::nghost*stride[dir]];
      return out * idx[dir];
    }
#endif // DBG_SYMMETRIZE_FD

    // 1st derivative 2nd order centered
#ifdef DBG_SYMMETRIZE_FD
    inline Real Ds(int dir, Real & u) {
      Real * pu = &u - c1_lo::offset*stride[dir];

      Real out(0.);
      for(int n1 = 0; n1 < c1_lo::nghost; ++n1) {
        int const n2  = c1_lo::width - n1 - 1;
        out += c1_lo::coeff[n1] * (pu[n1*stride[dir]] - pu[n2*stride[dir]]);
      }
      return out * cidx1_lo[dir];
    }
#else
    inline Real Ds(int dir, Real & u) {
      Real * pu = &u;
      return 0.5 * idx[dir] * (pu[stride[dir]] - pu[-stride[dir]]);
    }
#endif // DBG_SYMMETRIZE_FD

    // Advective derivative
    // The advective derivative is for an equation in the form
    //    d_t u = vx d_x u
    // So negative vx means advection from the *left* to the *right*, so we use
    // *left* biased FD stencils
#ifdef DBG_SYMMETRIZE_FD
    inline Real Lx(int dir, Real & vx, Real & u) {
      Real * pu = &u;

      Real dl(0.);
      for(int n = 0; n < ll1::width; ++n) {
        dl += ll1::coeff[n] * pu[(n - ll1::offset)*stride[dir]];
      }

      Real dr(0.);
      for(int n = lr1::width-1; n >= 0; --n) {
        dr += lr1::coeff[n] * pu[(n - lr1::offset)*stride[dir]];
      }

      // lidx_l1[dir] == lidx_r1[dir]
      return ((vx < 0) ? (vx * dl) : (vx * dr)) * lidx_l1[dir];
    }
#else
    inline Real Lx(int dir, Real & vx, Real & u) {
      Real * pu = &u;

      Real dl(0.);
      for(int n = 0; n < sl::width; ++n) {
        dl += sl::coeff[n] * pu[(n - sl::offset)*stride[dir]];
      }

      Real dr(0.);
      for(int n = sr::width-1; n >= 0; --n) {
        dr += sr::coeff[n] * pu[(n - sr::offset)*stride[dir]];
      }



      return ((vx < 0) ? (vx * dl) : (vx * dr)) * idx[dir];
    }
#endif // DBG_SYMMETRIZE_FD

    // Homogeneous 2nd derivative
#ifdef DBG_SYMMETRIZE_FD
    inline Real Dxx(int dir, Real & u) {
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
      for(int n1 = 0; n1 < c2::nghost; ++n1) {
        int const n2  = c2::width - n1 - 1;
        out += c2::coeff[n1] * (pu[n1*stride[dir]] + pu[n2*stride[dir]]);
      }
      out += c2::coeff[c2::nghost] * pu[c2::nghost*stride[dir]];
      return out * cidx2[dir];
    }
#else
    inline Real Dxx(int dir, Real & u) {
      Real * pu = &u - s2::offset*stride[dir];

      Real out(0.);
      for(int n1 = 0; n1 < s2::nghost; ++n1) {
        int const n2  = s2::width - n1 - 1;
        Real const c1 = s2::coeff[n1] * pu[n1*stride[dir]];
        Real const c2 = s2::coeff[n2] * pu[n2*stride[dir]];
        out += (c1 + c2);
      }
      out += s2::coeff[s2::nghost] * pu[s2::nghost*stride[dir]];
      return out * SQR(idx[dir]);
    }
#endif // DBG_SYMMETRIZE_FD

    // Mixed 2nd derivative
#ifdef DBG_SYMMETRIZE_FD
    inline Real Dxy(int dirx, int diry, Real & u) {
      Real * pu = &u - c1::offset*(stride[dirx] + stride[diry]);
      Real out(0.);

      for(int nx1 = 0; nx1 < c1::nghost; ++nx1) {
        int const nx2 = c1::width - nx1 - 1;
        for(int ny1 = 0; ny1 < c1::nghost; ++ny1) {
          int const ny2 = c1::width - ny1 - 1;
          // out += c1::coeff[nx1] * c1::coeff[ny1] * (
          //   ( pu[nx1*stride[dirx] + ny1*stride[diry]] + pu[nx2*stride[dirx] + ny2*stride[diry]]) -
          //   ( pu[nx2*stride[dirx] + ny1*stride[diry]] + pu[nx1*stride[dirx] + ny2*stride[diry]])
          // );

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
    }
#else
    inline Real Dxy(int dirx, int diry, Real & u) {
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
    }
#endif // DBG_SYMMETRIZE_FD

    // Kreiss-Oliger dissipation operator
#ifdef DBG_SYMMETRIZE_FD
    inline Real Diss(int dir, Real & u, Real diss) {
      Real * pu = &u - cd::offset*stride[dir];

      Real out(0.);
      for(int n1 = 0; n1 < cd::nghost; ++n1) {
        int const n2  = cd::width - n1 - 1;
        out += cd::coeff[n1] * (pu[n1*stride[dir]] + pu[n2*stride[dir]]);
      }
      out += cd::coeff[cd::nghost] * pu[cd::nghost*stride[dir]];
      return out * cidxd[dir] * diss;
    }
#else
    inline Real Diss(int dir, Real & u, Real diss) {
      Real * pu = &u - sd::offset*stride[dir];

      Real out(0.);
      for(int n1 = 0; n1 < sd::nghost; ++n1) {
        int const n2  = sd::width - n1 - 1;
        Real const c1 = sd::coeff[n1] * pu[n1*stride[dir]];
        Real const c2 = sd::coeff[n2] * pu[n2*stride[dir]];
        out += (c1 + c2);
      }
      out += sd::coeff[sd::nghost] * pu[sd::nghost*stride[dir]];

      return out * idx[dir] * diss * diss_scaling;
    }
#endif // DBG_SYMMETRIZE_FD


  // ctor / dtor --------------------------------------------------------------
  public:
    Uniform(const int nn1, const Real dx1);
    Uniform(
      const int nn1, const int nn2, const Real dx1, const Real dx2);
    Uniform(
      const int nn1, const int nn2, const int nn3,
      const Real dx1, const Real dx2, const Real dx3);
    ~Uniform();

};

} // namespace FiniteDifference

#endif
