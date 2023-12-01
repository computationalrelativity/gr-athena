#ifndef FINITE_DIFFERENCING_HPP_
#define FINITE_DIFFERENCING_HPP_

//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file finite_differencing.hpp
//  \brief High-performance finite-differencing kernel

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "floating_point.hpp"

namespace FiniteDifference {

// Centered finite differencing stencils
// * degree : Degree of the derivative, eg, 1 for 1st derivative
// * nghost : Number of ghost points used for the derivative
template<int degree_, int nghost_>
class FDCenteredStencil
{
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
class FDDissChoice
{
public:
  enum {degree = 2*(nghost_ + 1)};
  enum {nghost = nghost_ + 1};
};

// Choose the left or right biased stencil
template<int degree_, int nghost_>
class FDBiasedChoice
{
public:
  enum {degree = degree_};
  enum {nghost = nghost_+1};
  enum {lopsize = nghost_};
};
template<>
class FDBiasedChoice<1, 1>
{
public:
  enum {degree = 1};
  enum {nghost = 2};
  enum {lopsize = 0};
};
template<>
class FDBiasedChoice<1, 2>
{
public:
  enum {degree = 1};
  enum {nghost = 3};
  enum {lopsize = 1};
};
template<>
class FDBiasedChoice<1, 3>
{
public:
  enum {degree = 1};
  enum {nghost = 4};
  enum {lopsize = 2};
};
template<>
class FDBiasedChoice<1, 4>
{
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
class FDLeftBiasedStencil
{
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
class FDRightBiasedStencil
{
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
class FDStencilCenteredDegreeOdd
{
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
class FDStencilCenteredDegreeEven
{
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
class FDStencilBiasedLeft
{
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
class FDStencilBiasedRight
{
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

  // higher degree derivatives
  // BD: TODO, generalize suitably...
  typedef FDCenteredStencil<7, 4> cd_hd;


  // expose functionality -----------------------------------------------------
  public:

    // 1st derivative (high order centered)
    inline Real Dx(int dir, Real & u);

    // 1st derivative 2nd order centered
    inline Real Ds(int dir, Real & u);

    // Advective derivative
    // The advective derivative is for an equation in the form
    //    d_t u = vx d_x u
    // So negative vx means advection from the *left* to the *right*, so we use
    // *left* biased FD stencils
    inline Real Lx(int dir, Real & vx, Real & u);

    // Homogeneous 2nd derivative
    inline Real Dxx(int dir, Real & u);

    // Mixed 2nd derivative
    inline Real Dxy(int dirx, int diry, Real & u);

    // Kreiss-Oliger dissipation operator
    inline Real Diss(int dir, Real & u, Real diss);

    // Higher degree derivatives (error estimation)
    inline Real Dx_ho(int dir, Real & u);

  // ctor / dtor --------------------------------------------------------------
  public:
    inline Uniform(const int nn1, const Real dx1);
    inline Uniform(
      const int nn1, const int nn2, const Real dx1, const Real dx2);
    inline Uniform(
      const int nn1, const int nn2, const int nn3,
      const Real dx1, const Real dx2, const Real dx3);
    inline ~Uniform() { }

};

} // namespace FiniteDifference

// implementation details (for templates) =====================================
// order here is important!
#include "finite_differencing_stencils.tpp"
#include "finite_differencing.tpp"
// ============================================================================


#endif
