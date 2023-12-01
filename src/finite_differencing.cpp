//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file finite_differencing.cpp
//  \brief High-performance finite differencing kernel

#include "finite_differencing.hpp"
#include <cmath>

namespace FiniteDifference {

// Centered finite differencing 1st derivative
template<>
Real const FDCenteredStencil<1, 1>::coeff[] = {
  -1./2., 0., 1./2.,
};

template<>
Real const FDCenteredStencil<1, 2>::coeff[] = {
  1./12., -2./3., 0., 2./3., -1./12.,
};

template<>
Real const FDCenteredStencil<1, 3>::coeff[] = {
  -1./60., 3./20., -3./4., 0., 3./4., -3./20., 1./60.,
};

template<>
Real const FDCenteredStencil<1, 4>::coeff[] = {
  1./280., -4./105., 1./5., -4./5., 0., 4./5., -1./5., 4./105., -1./280.,
};

template<>
Real const FDCenteredStencil<1, 5>::coeff[] = {
  -1./1260., 5./504., -5./84., 5./21., -5./6., 0., 5./6., -5./21., 5./84., -5./504., 1./1260.,
};

template<>
Real const FDCenteredStencil<1, 6>::coeff[] = {
  1./5544.,-1./385.,1./56.,-5./63.,15./56.,-6./7.,0.,6./7.,-15./56.,5./63.,-1./56.,1./385.,-1./5544.
};

// Centered finite differencing 2nd derivative
template<>
Real const FDCenteredStencil<2, 1>::coeff[] = {
  1., -2., 1.,
};

template<>
Real const FDCenteredStencil<2, 2>::coeff[] = {
  -1./12., 4./3., -5./2., 4./3., -1./12.,
};

template<>
Real const FDCenteredStencil<2, 3>::coeff[] = {
  1./90., -3./20., 3./2., -49./18., 3./2., -3./20., 1./90.,
};

template<>
Real const FDCenteredStencil<2, 4>::coeff[] = {
  -1./560., 8./315., -1./5., 8./5., -205./72., 8./5., -1./5., 8./315., -1./560.,
};

// add for testing
template<>
Real const FDCenteredStencil<2, 5>::coeff[] = {
  1./3150., -5./1008., 5./126., -5./21., 5./3., -5269./1800.,
  5./3., -5./21., 5./126., -5./1008., 1./3150.,
};

template<>
Real const FDCenteredStencil<2, 6>::coeff[] = {
  -1./16632., 2./1925., -1./112., 10./189., -15./56., 12./7., -5369./1800.,
  12./7., -15./56., 10./189., -1./112., 2./1925., -1./16632.,
};

template<>
Real const FDCenteredStencil<2, 7>::coeff[] = {
  1./84084., -7./30888., 7./3300., -7./528., 7./108., -7./24., 7./4.,
  -266681./88200., 7./4., -7./24., 7./108., -7./528., 7./3300., -7./30888.,
  1./84084.,
};

template<>
Real const FDCenteredStencil<2, 8>::coeff[] = {
 -1./411840., 16./315315., -2./3861., 112./32175., -7./396., 112./1485.,
 -14./45., 16./9., -1077749./352800., 16./9., -14./45., 112./1485.,
 -7./396., 112./32175., -2./3861., 16./315315., -1./411840.,
};


// High order derivative operators for Kreiss-Oliger dissipation
template<>
Real const FDCenteredStencil<4, 2>::coeff[] = {
  1., -4., 6., -4., 1.,
};

template<>
Real const FDCenteredStencil<6, 3>::coeff[] = {
  1., -6., 15., -20., 15., -6., 1.,
};

template<>
Real const FDCenteredStencil<8, 4>::coeff[] = {
  1., -8., 28., -56., 70., -56., 28., -8., 1.,
};

template<>
Real const FDCenteredStencil<10, 5>::coeff[] = {
  1., -10., 45., -120., 210., -252., 210., -120., 45., -10., 1.,
};

template<>
Real const FDCenteredStencil<12, 6>::coeff[] = {
  1., -12., 66., -220., 495., -792., 924., -792., 495., -220., 66., -12., 1.
};

template<>
Real const FDCenteredStencil<14, 7>::coeff[] = {
  1.,-14.,91.,-364.,1001.,-2002.,3003.,-3432.,3003.,-2002.,1001.,-364.,91.,-14.,1.
};

// Left-biased finite differencing 1st derivative
template<>
Real const FDLeftBiasedStencil<1, 1, 0>::coeff[] = {
  -1., 1.,
};

template<>
Real const FDLeftBiasedStencil<1, 2, 0>::coeff[] = {
  0.5, -2.0, 1.5,
};

template<>
Real const FDLeftBiasedStencil<1, 2, 1>::coeff[] = {
  1./6., -1., 1./2., 1./3.,
};

template<>
Real const FDLeftBiasedStencil<1, 3, 1>::coeff[] = {
  -1./12., 6./12., -18./12., +10./12., +3./12.,
};

template<>
Real const FDLeftBiasedStencil<1, 3, 2>::coeff[] = {
  -1./30., 1./4., -1., 1./3., 1./2., -1./20.,
};

template<>
Real const FDLeftBiasedStencil<1, 4, 2>::coeff[] = {
  1./60., -2./15., 1./2., -4./3., 7./12., 2./5., -1./30.,
};

template<>
Real const FDLeftBiasedStencil<1, 4, 3>::coeff[] = {
  1./140., -1./15., 3./10., -1., 1./4., 3./5., -1./10., 1./105.,
};

template<>
Real const FDLeftBiasedStencil<1, 5, 3>::coeff[] = {
  -1./280., 1./28., -1./6., 1./2., -5./4., 9./20., 1./2., -1./14., 1./168.,
};

template<>
Real const FDLeftBiasedStencil<1, 5, 4>::coeff[] ={
  -1./630., 1./56., -2./21., 1./3., -1., 1./5., 2./3., -1./7., 1./42., -1./504.,
};

template<>
Real const FDLeftBiasedStencil<1, 6, 5>::coeff[] ={
  1./2772.,-1./210.,5./168.,-5./42.,5./14.,-1.,1./6.,5./7.,-5./28.,5./126.,-1./168.,1./2310.
};

template<>
Real const FDLeftBiasedStencil<1, 7, 6>::coeff[] ={
  -1./12012.,1./792.,-1./110.,1./24.,-5./36.,3./8.,-1.,1./7.,3./4.,-5./24.,1./18.,-1./88.,1./660.,-1./10296.
};

// Right-biased finite differencing 1st derivative
template<>
Real const FDRightBiasedStencil<1, 1, 0>::coeff[] = {
  -1., 1.,
};

template<>
Real const FDRightBiasedStencil<1, 2, 0>::coeff[] = {
  -1.5, 2.0, -0.5,
};

template<>
Real const FDRightBiasedStencil<1, 2, 1>::coeff[] = {
  -1./3., -1./2., 1., -1./6.,
};

template<>
Real const FDRightBiasedStencil<1, 3, 1>::coeff[] = {
   -3./12., -10./12., 18./12., -6./12., 1./12.,
};

template<>
Real const FDRightBiasedStencil<1, 3, 2>::coeff[] = {
  1./20., -1./2., -1./3., 1., -1./4., 1./30.,
};

template<>
Real const FDRightBiasedStencil<1, 4, 2>::coeff[] = {
  1./30., -2./5., -7./12., 4./3., -1./2., 2./15., -1./60.,
};

template<>
Real const FDRightBiasedStencil<1, 4, 3>::coeff[] = {
  -1./105., 1./10., -3./5., -1./4., 1., -3./10., 1./15., -1./140.,
};

template<>
Real const FDRightBiasedStencil<1, 5, 3>::coeff[] = {
  -1./168., 1./14., -1./2., -9./20., 5./4., -1./2., 1./6., -1./28., 1./280.,
};

template<>
Real const FDRightBiasedStencil<1, 5, 4>::coeff[] = {
  1./504., -1./42., 1./7., -2./3., -1./5., 1., -1./3., 2./21., -1./56., 1./630.,
};

template<>
Real const FDRightBiasedStencil<1, 6, 5>::coeff[] = {
  -1./2310.,1./168.,-5./126.,5./28.,-5./7.,-1./6.,1.,-5./14.,5./42.,-5./168.,1./210.,-1./2772.
};

template<>
Real const FDRightBiasedStencil<1, 7, 6>::coeff[] = {
  1./10296.,-1./660.,1./88.,-1./18.,5./24.,-3./4.,-1./7.,1.,-3./8.,5./36.,-1./24.,1./110.,-1./792.,1./12012.
};

// Coefficients for biased derivatives 


// 1st derivative, 4th order
// This must be defined but should not be used for NGHOST = 3,
// so it is filled with NAN
template<>
Real const FDRightBiasedStencilBeyond<1, 3, 3>::coeff[] = {
           NAN, NAN, NAN, NAN, NAN,
};
template<>
Real const FDRightBiasedStencilBeyond<1, 3, 2>::coeff[] = {
           -25./12., +4., -3., 4./3., -1./4.,
};

template<>
Real const FDRightBiasedStencilBeyond<1, 3, 1>::coeff[] = {
           -1./4., -5./6., 3./2., -1./2., 1./12.,
};
// 2nd derivative, 4th order
// This must be defined but should not be used for NGHOST = 3,
// so it is filled with NAN
template<>
Real const FDRightBiasedStencilBeyond<2, 3, 3>::coeff[] = {
           NAN, NAN, NAN, NAN, NAN, NAN,
};
template<>
Real const FDRightBiasedStencilBeyond<2, 3, 2>::coeff[] = {
           15./4., -77./6., 107./6., -13., 61./12., -5./6.
};

template<>
Real const FDRightBiasedStencilBeyond<2, 3, 1>::coeff[] = {
           5./6., -5./4., -1./3., 7./6., -1./2., 1./12.,
};

// 1st derivative, 6th order
template<>
Real const FDRightBiasedStencilBeyond<1, 4, 3>::coeff[] = {
           -49./20., 6., -15./2., 20./3., -15./4., 6./5., -1./6.
};

template<>
Real const FDRightBiasedStencilBeyond<1, 4, 2>::coeff[] = {
           -1./6., -77./60., 5./2., -5./3., 5./6., -1./4., 1./30.,
};

template<>
Real const FDRightBiasedStencilBeyond<1, 4, 1>::coeff[] = {
           1./30., -2./5., -7./12., 4./3., -1./2., 2./15., -1./60.,
};

// 2nd derivative, 6th order
template<>
Real const FDRightBiasedStencilBeyond<2, 4, 3>::coeff[] = {
        469./90., -223./10., 879./20., -949./18., 41., -201./10., 1019./180., -7./10.
};

template<>
Real const FDRightBiasedStencilBeyond<2, 4, 2>::coeff[] = {
           7./10., -7./18., -27./10., 19./4., -67./18., 9./5., -1./2., 11./180.
};

template<>
Real const FDRightBiasedStencilBeyond<2, 4, 1>::coeff[] = {
        -11./180., 107./90., -21./10., 13./18., 17./36., -3./10., 4./45., -1./90.
};

// 1st derivative, 4th order
// This is always defined but should not be used for NGHOST = 3,
// so it is filled with NAN
template<>
Real const FDLeftBiasedStencilBeyond<1, 3, 3>::coeff[] = {
           NAN, NAN, NAN, NAN, NAN,
};
template<>
Real const FDLeftBiasedStencilBeyond<1, 3, 2>::coeff[] = {
           1./4., -4./3., 3., -4., 25./12.,
};

template<>
Real const FDLeftBiasedStencilBeyond<1, 3, 1>::coeff[] = {
           -1./12, 1./2., -3./2., 5./6., 1./4.,
};

// 2nd derivative, 4th order
// This must be defined but should not be used for NGHOST = 3,
// so it is filled with NAN
template<>
Real const FDLeftBiasedStencilBeyond<2, 3, 3>::coeff[] = {
           NAN, NAN, NAN, NAN, NAN, NAN,
};
template<>
Real const FDLeftBiasedStencilBeyond<2, 3, 2>::coeff[] = {
           -5./6., 61./12., -13., 107./6., -77./6., 15./4.,
};

template<>
Real const FDLeftBiasedStencilBeyond<2, 3, 1>::coeff[] = {
           1./12., -1./2., 7./6., -1./3., -5./4., 5./6.,
};

// 1st derivative, 6th order
template<>
Real const FDLeftBiasedStencilBeyond<1, 4, 3>::coeff[] = {
          1./6., -6./5., 15./4., -20./3., 15./2., -6., 49./20.,
};

template<>
Real const FDLeftBiasedStencilBeyond<1, 4, 2>::coeff[] = {
          -1./30., 1./4., -5./6., 5./3., -5./2., 77./60., 1./6.,
};

template<>
Real const FDLeftBiasedStencilBeyond<1, 4, 1>::coeff[] = {
          1./60., -2./15., 1./2., -4./3., 7./12., 2./5., -1./30.,
};

// 2nd derivative, 6th order
template<>
Real const FDLeftBiasedStencilBeyond<2, 4, 3>::coeff[] = {
        -7./10., 1019./180., -201./10., 41., -949./18., 879./20., -223./10., 469./90.,
};

template<>
Real const FDLeftBiasedStencilBeyond<2, 4, 2>::coeff[] = {
        11./180., -1./2., 9./5., -67./18., 19./4., -27./10., -7./18., 7./10.,
};

template<>
Real const FDLeftBiasedStencilBeyond<2, 4, 1>::coeff[] = {
        -1./90., 4./45., -3./10., 17./36., 13./18., -21./10., 107./90., -11./180.,
};

#ifdef DBG_SYMMETRIZE_FD
// Rewrite with denominator LCM factoring to mitigate some error ==============

// 1st degree -----------------------------------------------------------------

// 2nd order ------------------------------------------------------------------
template<>
Real const FDStencilCenteredDegreeOdd<1, 1>::coeff[] = {
  -1
};

template<>
Real const FDStencilCenteredDegreeOdd<1, 1>::coeff_lcm = 2;

// 4th order ------------------------------------------------------------------
template<>
Real const FDStencilCenteredDegreeOdd<1, 2>::coeff[] = {
  1, -8
};

template<>
Real const FDStencilCenteredDegreeOdd<1, 2>::coeff_lcm = 12;

// 6th order ------------------------------------------------------------------
template<>
Real const FDStencilCenteredDegreeOdd<1, 3>::coeff[] = {
  -1, 9, -45
};

template<>
Real const FDStencilCenteredDegreeOdd<1, 3>::coeff_lcm = 60;

// 8th order ------------------------------------------------------------------
template<>
Real const FDStencilCenteredDegreeOdd<1, 4>::coeff[] = {
  3, -32, 168, -672
};

template<>
Real const FDStencilCenteredDegreeOdd<1, 4>::coeff_lcm = 840;

// 10th order -----------------------------------------------------------------
template<>
Real const FDStencilCenteredDegreeOdd<1, 5>::coeff[] = {
  -2, 25, -150, 600, -2100
};

template<>
Real const FDStencilCenteredDegreeOdd<1, 5>::coeff_lcm = 2520;

// 12th order -----------------------------------------------------------------
template<>
Real const FDStencilCenteredDegreeOdd<1, 6>::coeff[] = {
  5, -72, 495, -2200, 7425, -23760
};

template<>
Real const FDStencilCenteredDegreeOdd<1, 6>::coeff_lcm = 27720;

// 14th order -----------------------------------------------------------------
template<>
Real const FDStencilCenteredDegreeOdd<1, 7>::coeff[] = {
  -15, 245, -1911, 9555, -35035, 105105, -315315
};

template<>
Real const FDStencilCenteredDegreeOdd<1, 7>::coeff_lcm = 360360;

// 2nd degree -----------------------------------------------------------------

// 2nd order ------------------------------------------------------------------
template<>
Real const FDStencilCenteredDegreeEven<2, 1>::coeff[] = {
  1, -2
};

template<>
Real const FDStencilCenteredDegreeEven<2, 1>::coeff_lcm = 1;

// 4th order ------------------------------------------------------------------
template<>
Real const FDStencilCenteredDegreeEven<2, 2>::coeff[] = {
  -1, 16, -30
};

template<>
Real const FDStencilCenteredDegreeEven<2, 2>::coeff_lcm = 12;

// 6th order ------------------------------------------------------------------
template<>
Real const FDStencilCenteredDegreeEven<2, 3>::coeff[] = {
  2, -27, 270, -490
};

template<>
Real const FDStencilCenteredDegreeEven<2, 3>::coeff_lcm = 180;

// 8th order ------------------------------------------------------------------
template<>
Real const FDStencilCenteredDegreeEven<2, 4>::coeff[] = {
  -9, 128, -1008, 8064, -14350
};

template<>
Real const FDStencilCenteredDegreeEven<2, 4>::coeff_lcm = 5040;

// 10th order -----------------------------------------------------------------
template<>
Real const FDStencilCenteredDegreeEven<2, 5>::coeff[] = {
  8, -125, 1000, -6000, 42000, -73766
};

template<>
Real const FDStencilCenteredDegreeEven<2, 5>::coeff_lcm = 25200;

// 12th order -----------------------------------------------------------------
template<>
Real const FDStencilCenteredDegreeEven<2, 6>::coeff[] = {
  -50,864,-7425,44000,-222750,1425600,-2480478
};

template<>
Real const FDStencilCenteredDegreeEven<2, 6>::coeff_lcm = 831600;

// 14th order -----------------------------------------------------------------
template<>
Real const FDStencilCenteredDegreeEven<2, 7>::coeff[] = {
  900, -17150, 160524, -1003275, 4904900, -22072050, 132432300, -228812298
};

template<>
Real const FDStencilCenteredDegreeEven<2, 7>::coeff_lcm = 75675600;

// higher degree --------------------------------------------------------------

// 4th degree -----------------------------------------------------------------
template<>
Real const FDStencilCenteredDegreeEven<4, 2>::coeff[] = {
  1, -4, 6
};

template<>
Real const FDStencilCenteredDegreeEven<4, 2>::coeff_lcm = 1;

// 6th degree -----------------------------------------------------------------
template<>
Real const FDStencilCenteredDegreeEven<6, 3>::coeff[] = {
  1, -6, 15, -20
};

template<>
Real const FDStencilCenteredDegreeEven<6, 3>::coeff_lcm = 1;

// 8th degree -----------------------------------------------------------------
template<>
Real const FDStencilCenteredDegreeEven<8, 4>::coeff[] = {
  1, -8, 28, -56, 70
};

template<>
Real const FDStencilCenteredDegreeEven<8, 4>::coeff_lcm = 1;

// 10th degree ----------------------------------------------------------------
template<>
Real const FDStencilCenteredDegreeEven<10, 5>::coeff[] = {
  1, -10, 45, -120, 210, -252
};

template<>
Real const FDStencilCenteredDegreeEven<10, 5>::coeff_lcm = 1;

// 12th degree ----------------------------------------------------------------
template<>
Real const FDStencilCenteredDegreeEven<12, 6>::coeff[] = {
  1, -12, 66, -220, 495, -792, 924
};

template<>
Real const FDStencilCenteredDegreeEven<12, 6>::coeff_lcm = 1;

// 14th degree ----------------------------------------------------------------
template<>
Real const FDStencilCenteredDegreeEven<14, 7>::coeff[] = {
  1, -14, 91, -364, 1001, -2002, 3003, -3432
};

template<>
Real const FDStencilCenteredDegreeEven<14, 7>::coeff_lcm = 1;

// 16th degree ----------------------------------------------------------------
template<>
Real const FDStencilCenteredDegreeEven<16, 8>::coeff[] = {
  1, -16, 120, -560, 1820, -4368, 8008, -11440, 12870
};

template<>
Real const FDStencilCenteredDegreeEven<16, 8>::coeff_lcm = 1;

// 1st lop --------------------------------------------------------------------

// 2nd order ------------------------------------------------------------------
template<>
Real const FDStencilBiasedLeft<1, 2, 0>::coeff[] = {
  1, -4, 3
};

template<>
Real const FDStencilBiasedLeft<1, 2, 0>::coeff_lcm = 2;

template<>
Real const FDStencilBiasedRight<1, 2, 0>::coeff[] = {
  -3, 4, -1
};

template<>
Real const FDStencilBiasedRight<1, 2, 0>::coeff_lcm = 2;

// 4th order ------------------------------------------------------------------
template<>
Real const FDStencilBiasedLeft<1, 3, 1>::coeff[] = {
  -1, 6, -18, 10, 3
};

template<>
Real const FDStencilBiasedLeft<1, 3, 1>::coeff_lcm = 12;

template<>
Real const FDStencilBiasedRight<1, 3, 1>::coeff[] = {
  -3, -10, 18, -6, 1
};

template<>
Real const FDStencilBiasedRight<1, 3, 1>::coeff_lcm = 12;

// 6th order ------------------------------------------------------------------
template<>
Real const FDStencilBiasedLeft<1, 4, 2>::coeff[] = {
  1, -8, 30, -80, 35, 24, -2
};

template<>
Real const FDStencilBiasedLeft<1, 4, 2>::coeff_lcm = 60;

template<>
Real const FDStencilBiasedRight<1, 4, 2>::coeff[] = {
  2, -24, -35, 80, -30, 8, -1
};

template<>
Real const FDStencilBiasedRight<1, 4, 2>::coeff_lcm = 60;

// 8th order ------------------------------------------------------------------
template<>
Real const FDStencilBiasedLeft<1, 5, 3>::coeff[] = {
  -3, 30, -140, 420, -1050, 378, 420, -60, 5
};

template<>
Real const FDStencilBiasedLeft<1, 5, 3>::coeff_lcm = 840;

template<>
Real const FDStencilBiasedRight<1, 5, 3>::coeff[] = {
  -5, 60, -420, -378, 1050, -420, 140, -30, 3
};

template<>
Real const FDStencilBiasedRight<1, 5, 3>::coeff_lcm = 840;

// 10th order -----------------------------------------------------------------
template<>
Real const FDStencilBiasedLeft<1, 6, 4>::coeff[] = {
  2, -24, 135, -480, 1260, -3024, 924, 1440, -270, 40, -3
};

template<>
Real const FDStencilBiasedLeft<1, 6, 4>::coeff_lcm = 2520;

template<>
Real const FDStencilBiasedRight<1, 6, 4>::coeff[] = {
  3, -40, 270, -1440, -924, 3024, -1260, 480, -135, 24, -2
};

template<>
Real const FDStencilBiasedRight<1, 6, 4>::coeff_lcm = 2520;

// 12th order -----------------------------------------------------------------
template<>
Real const FDStencilBiasedLeft<1, 7, 5>::coeff[] = {
  -5, 70, -462, 1925, -5775, 13860, -32340, 8580, 17325, -3850, 770, -105, 7
};

template<>
Real const FDStencilBiasedLeft<1, 7, 5>::coeff_lcm = 27720;

template<>
Real const FDStencilBiasedRight<1, 7, 5>::coeff[] = {
  -7, 105, -770, 3850, -17325, -8580, 32340, -13860, 5775, -1925, 462, -70, 5
};

template<>
Real const FDStencilBiasedRight<1, 7, 5>::coeff_lcm = 27720;

// 14th order -----------------------------------------------------------------
template<>
Real const FDStencilBiasedLeft<1, 8, 6>::coeff[] = {
  15, -240, 1820, -8736, 30030, -80080, 180180, -411840,
  96525, 240240, -60060, 14560, -2730, 336, -20
};

template<>
Real const FDStencilBiasedLeft<1, 8, 6>::coeff_lcm = 360360;

template<>
Real const FDStencilBiasedRight<1, 8, 6>::coeff[] = {
  20, -336, 2730, -14560, 60060, -240240, -96525, 411840,
  -180180, 80080, -30030, 8736, -1820, 240, -15
};

template<>
Real const FDStencilBiasedRight<1, 8, 6>::coeff_lcm = 360360;

#endif // DBG_SYMMETRIZE_FD

// ctor / dtor ----------------------------------------------------------------
Uniform::Uniform(const int nn1, const Real dx1)
{
  Uniform(nn1, 0, dx1, 0);
}

Uniform::Uniform(
  const int nn1, const int nn2,  const Real dx1, const Real dx2)
{
  Uniform(nn1, nn2, 0, dx1, dx2, 0);
}

Uniform::Uniform(
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

Uniform::~Uniform() { }

} // namespace FiniteDifference
