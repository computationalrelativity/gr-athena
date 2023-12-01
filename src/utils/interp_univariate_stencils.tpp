// headers ====================================================================
// c / c++
// #include <algorithm>

// Athena++ headers
#include "../athena.hpp"
#include "interp_univariate.hpp"

// ----------------------------------------------------------------------------
namespace InterpUniform {

// right biased stencils
template<>
inline Real const InterpolateLagrangeUniformBiasR<2>::coeff[] = {
  2., -1.
};
template<>
inline Real const InterpolateLagrangeUniformBiasR<3>::coeff[] = {
  3., -3., 1.
};
template<>
inline Real const InterpolateLagrangeUniformBiasR<4>::coeff[] = {
  4., -6., 4., -1.
};
template<>
inline Real const InterpolateLagrangeUniformBiasR<5>::coeff[] = {
  5., -10., 10., -5., 1.
};
template<>
inline Real const InterpolateLagrangeUniformBiasR<6>::coeff[] = {
  6., -15., 20., -15., 6., -1.
};
template<>
inline Real const InterpolateLagrangeUniformBiasR<7>::coeff[] = {
  7., -21., 35., -35., 21., -7., 1.
};
template<>
inline Real const InterpolateLagrangeUniformBiasR<8>::coeff[] = {
  8., -28., 56., -70., 56., -28., 8., -1.
};
template<>
inline Real const InterpolateLagrangeUniformBiasR<9>::coeff[] = {
  9., -36., 84., -126., 126., -84., 36., -9., 1.
};
template<>
inline Real const InterpolateLagrangeUniformBiasR<10>::coeff[] = {
  10., -45., 120., -210., 252., -210., 120., -45., 10., -1.
};

// left biased stencils
template<>
inline Real const InterpolateLagrangeUniformBiasL<2>::coeff[] = {
  -1., 2.
};
template<>
inline Real const InterpolateLagrangeUniformBiasL<3>::coeff[] = {
  1., -3., 3.
};
template<>
inline Real const InterpolateLagrangeUniformBiasL<4>::coeff[] = {
  -1., 4., -6., 4.
};
template<>
inline Real const InterpolateLagrangeUniformBiasL<5>::coeff[] = {
  1., -5., 10., -10., 5.
};
template<>
inline Real const InterpolateLagrangeUniformBiasL<6>::coeff[] = {
  -1., 6., -15., 20., -15., 6.
};
template<>
inline Real const InterpolateLagrangeUniformBiasL<7>::coeff[] = {
  1., -7., 21., -35., 35., -21., 7.
};
template<>
inline Real const InterpolateLagrangeUniformBiasL<8>::coeff[] = {
  -1., 8., -28., 56., -70., 56., -28., 8.
};
template<>
inline Real const InterpolateLagrangeUniformBiasL<9>::coeff[] = {
  1., -9., 36., -84., 126., -126., 84., -36., 9.
};
template<>
inline Real const InterpolateLagrangeUniformBiasL<10>::coeff[] = {
  -1., 10., -45., 120., -210., 252., -210., 120., -45., 10.
};


// centered stencils
template<>
inline Real const InterpolateLagrangeUniform<1>::coeff[1] = {
  1./2., // interp. to midpoint of stencil
};
template<>
inline Real const InterpolateLagrangeUniform<2>::coeff[2] = {
  -1./16., 9./16.,
};
template<>
inline Real const InterpolateLagrangeUniform<3>::coeff[3] = {
  3./256., -25./256., 75./128.,
};
template<>
inline Real const InterpolateLagrangeUniform<4>::coeff[4] = {
  -5./2048., 49./2048., -245./2048., 1225./2048.,
};
template<>
inline Real const InterpolateLagrangeUniform<5>::coeff[5] = {
  35./65536., -405./65536., 567./16384., -2205./16384., 19845./32768.,
};

// Stencils about central point e.g. {x_i-1, x_i, x_i+1} with dx unif.
// coeffs for interp. to x_i-dx/4.
//
// interp. to x_i+dx/4 by flipping stencil
template<>
inline Real const InterpolateLagrangeUniformChildren<1>::coeff[3] = {
  5.0 / 32.0, 30.0 / 32.0, -3.0 / 32.0
};

template<>
inline Real const InterpolateLagrangeUniformChildren<2>::coeff[5] = {
  -45.0 / 2048.0,
   420.0 / 2048.0,
   1890.0 / 2048.0,
  -252.0 / 2048.0,
   35.0 / 2048.0
  // -(27.0/512.0),
  // 21.0/64.0,
  // 189.0/256.0,
  // 0.0,
  // -(7.0/512.0)
};

template<>
inline Real const InterpolateLagrangeUniformChildren<3>::coeff[7] = {
   273.0 / 65536.0,
  -2574.0 / 65536.0,
   15015.0 / 65536.0,
   60060.0 / 65536.0,
  -9009.0 / 65536.0,
   2002.0 / 65536.0,
  -231.0 / 65536.0
  // 351.0/32768.0,
  // -(1287.0/16384.0),
  // 10725.0/32768.0,
  // 6435.0/8192.0,
  // -(1287.0/32768.0),
  // -(143.0/16384.0),
  // 99.0/32768.0
};

template<>
inline Real const InterpolateLagrangeUniformChildren<4>::coeff[9] = {
  -7293.0 / 8388608.0,
   79560.0 / 8388608.0,
  -437580.0 / 8388608.0,
   2042040.0 / 8388608.0,
   7657650.0 / 8388608.0,
  -1225224.0 / 8388608.0,
   340340.0 / 8388608.0,
  -67320.0 / 8388608.0,
   6435.0 / 8388608.0
  // -(2431.0/1048576.0),
  // 5525.0/262144.0,
  // -(12155.0/131072.0),
  // 85085.0/262144.0,
  // 425425.0/524288.0,
  // -(17017.0/262144.0),
  // 935.0/262144.0,
  // -(715.0/1048576.0)
};

} // InterpUniform

//
// :D
//
