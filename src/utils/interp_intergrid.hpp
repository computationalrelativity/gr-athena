#ifndef INTERP_INTERGRID_HPP_
#define INTERP_INTERGRID_HPP_
//! \file interp_intergrid.hpp
//  \brief prototypes of utility functions to pack/unpack buffers
// C headers
// C++ headers
// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "interp_univariate.hpp"
//----------------------------------------------------------------------------------------
// \!fn Real CCInterpolation(AthenaArray<Real> &in, int k, int j, int i)
// \brief Perform cubic interpolation from cell-centered grid to vertex.
inline Real CCInterpolation(const AthenaArray<Real> &in, int k, int j, int i) {
  // interpolation coefficients
  // ordering is (i, j, k) = +/- (1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1),
  //                             (1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2)
  const Real coeff[8] = {729.0, -81.0, -81.0, 9.0, -81.0, 9.0, 9.0, -1.0};
  const Real den   = 4096.0;
  // Cells are located to the right of their corresponding vertices. So, i=1 corresponds
  // to a physical cell index a=0.
  const int off = 1;
  Real sum = 0;
  // Loop in reverse; the higher-index coefficients tend to have smaller contributions,
  // so this *may* limit round-off error.
  for (int c = 2; c > 0; --c) {
    for (int b = 2; b > 0; --b) {
      for (int a = 2; a > 0; --a) {
        int index = (a - off) + 2*((b - off) + 2*(c - off));
        // Grab the cube around vertex and add it up.
        Real lll = in(k - c, j - b, i - a);
        Real llu = in(k - c, j - b, i + a - off);
        Real lul = in(k - c, j + b - off, i - a);
        Real luu = in(k - c, j + b - off, i + a - off);
        Real ull = in(k + c - off, j - b, i - a);
        Real ulu = in(k + c - off, j - b, i + a - off);
        Real uul = in(k + c - off, j + b - off, i - a);
        Real uuu = in(k + c - off, j + b - off, i + a - off);
        // Attempt to add up the cube in a way that preserves symmetry.
        sum += coeff[index]*( ((lll + uuu) + (lul + ulu)) + ((llu + uul) + (luu + ull)) );
      }
    }
  }
  return sum/den;
}
//---------------------------------------------------------------------------------------
// \!fn Real VCInterpolation(AthenaArray<Real> &in, int k, int j, int i)
// \brief Perform linear interpolation to the desired cell-centered grid index.
inline Real VCInterpolation(const AthenaArray<Real> &in, int k, int j, int i) {
  return 0.125*(((in(k, j, i) + in(k + 1, j + 1, i + 1)) // lower-left-front to upper-right-back
               + (in(k+1, j+1, i) + in(k, j, i+1))) // upper-left-back to lower-right-front
               +((in(k, j+1, i) + in(k + 1, j, i+1)) // lower-left-back to upper-right-front
               + (in(k+1, j, i) + in(k, j+1, i+1)))); // upper-left-front to lower-right-back
}
//---------------------------------------------------------------------------------------
// \!fn Real VCReconstruct(AthenaArray<Real> &in, int k, int j, int i)
// \brief Perform linear interpolation to the desired face-centered grid index.
template<int dir>
inline Real VCReconstruct(AthenaArray<Real> &in, int k, int j, int i);
template<>
inline Real VCReconstruct<0>(AthenaArray<Real> &in, int k, int j, int i) {
  return 0.25*((in(k, j, i) + in(k + 1, j + 1, i)) +
               (in(k + 1, j, i) + in(k, j + 1, i)));
}
template<>
inline Real VCReconstruct<1>(AthenaArray<Real> &in, int k, int j, int i) {
  return 0.25*((in(k, j, i) + in(k + 1, j, i + 1)) +
               (in(k + 1, j, i) + in(k, j, i + 1)));
}
template<>
inline Real VCReconstruct<2>(AthenaArray<Real> &in, int k, int j, int i) {
  return 0.25*((in(k, j, i) + in(k, j + 1, i + 1)) +
               (in(k, j + 1, i) + in(k, j, i + 1)));
}
inline Real VCReconstruct(int dir, AthenaArray<Real> &in, int k, int j, int i) {
  switch(dir) {
    case 0:
      return VCReconstruct<0>(in, k, j, i);
    case 1:
      return VCReconstruct<1>(in, k, j, i);
    case 2:
      return VCReconstruct<2>(in, k, j, i);
    default:
      abort();
  }
}
//---------------------------------------------------------------------------------------
// \!fn Real VCDiff(AthenaArray<Real> &in, int k, int j, int i)
// \brief Evaluates the undivided derivative of a vertex centered variable at cell centers.
template<int dir>
inline Real VCDiff(const AthenaArray<Real> &in, int k, int j, int i);
template<>
inline Real VCDiff<0>(const AthenaArray<Real> &in, int k, int j, int i) {
  const Real coeff[2] = {9./8., -1./24.};
  Real stencil[4]; // values at the cell faces
  int off = 1;
  for (int a = -1; a < 3; ++a) {
    stencil[a + off] = 0.25*((in(k, j, i + a) + in(k + 1, j + 1, i + a)) +
                             (in(k + 1, j, i + a) + in(k, j + 1, i + a)));
  }
  return coeff[1]*(stencil[3] - stencil[0]) + coeff[0]*(stencil[2] - stencil[1]);
}
template<>
inline Real VCDiff<1>(const AthenaArray<Real> &in, int k, int j, int i) {
  const Real coeff[2] = {9./8., -1./24.};
  Real stencil[4]; // values at the cell faces
  int off = 1;
  for (int a = -1; a < 3; ++a) {
    stencil[a + off] = 0.25*((in(k, j + a, i) + in(k + 1, j + a, i + 1)) +
                             (in(k + 1, j + a, i) + in(k, j + a, i + 1)));
  }
  return coeff[1]*(stencil[3] - stencil[0]) + coeff[0]*(stencil[2] - stencil[1]);
}
template<>
inline Real VCDiff<2>(const AthenaArray<Real> &in, int k, int j, int i) {
  const Real coeff[2] = {9./8., -1./24.};
  Real stencil[4]; // values at the cell faces
  int off = 1;
  for (int a = -1; a < 3; ++a) {
    stencil[a + off] = 0.25*((in(k + a, j, i) + in(k + a, j + 1, i + 1)) +
                             (in(k + a, j + 1, i) + in(k + a, j, i + 1)));
  }
  return coeff[1]*(stencil[3] - stencil[0]) + coeff[0]*(stencil[2] - stencil[1]);
}
inline Real VCDiff(int dir, const AthenaArray<Real> &in, int k, int j, int i) {
  switch(dir) {
    case 0:
      return VCDiff<0>(in, k, j, i);
    case 1:
      return VCDiff<1>(in, k, j, i);
    case 2:
      return VCDiff<2>(in, k, j, i);
    default:
      abort();
  }
}
#endif // INTERP_INTERGRID_HPP_

