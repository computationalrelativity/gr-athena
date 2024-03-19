#ifndef M1_MACRO_HPP
#define M1_MACRO_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file m1_macro.hpp

#define M1_IX_IL                                                              \
  pm1->mbi.il

#define M1_IX_IU                                                              \
  pm1->mbi.iu

#define M1_IX_JL                                                              \
  pm1->mbi.jl

#define M1_IX_JU                                                              \
  pm1->mbi.ju

#define M1_IX_KL                                                              \
  pm1->mbi.kl

#define M1_IX_KU                                                              \
  pm1->mbi.ku

#define M1_GSIZEI                                                             \
  (pm1->mbi.ng)

#define M1_GSIZEJ                                                             \
  ((pm1->mbi.f2) ? (pm1->mbi.ng) : (0))

#define M1_GSIZEK                                                             \
  ((pm1->mbi.f3) ? (pm1->mbi.ng) : (0))

// 2D loop over k and j in the interior of the block
#define M1_ILOOP2(k,j)                                                        \
  for(int k = M1_IX_KL; k <= M1_IX_KU; ++k)                                   \
  for(int j = M1_IX_JL; j <= M1_IX_JU; ++j)

// 2D loop over k and j on the whole block
#define M1_GLOOP2(k,j)                                                        \
  for(int k = M1_IX_KL - M1_GSIZEK; k <= M1_IX_KU + M1_GSIZEK; ++k)           \
  for(int j = M1_IX_JL - M1_GSIZEJ; j <= M1_IX_JU + M1_GSIZEJ; ++j)

// 1D loop over i in the interior of the block
#define M1_ILOOP1(i)                                                          \
  _Pragma("omp simd")                                                         \
  for(int i = M1_IX_IL; i <= M1_IX_IU; ++i)

// 1D loop over i on the whole block
#define M1_GLOOP1(i)                                                          \
  _Pragma("omp simd")                                                         \
  for(int i = M1_IX_IL - M1_GSIZEI; i <= M1_IX_IU + M1_GSIZEI; ++i)

// 3D loop over the interior of the block
#define M1_ILOOP3(k,j,i)                                                      \
    M1_ILOOP2(k,j)                                                            \
    M1_ILOOP1(i)

// 3D loop over the whole block
#define M1_GLOOP3(k,j,i)                                                      \
    M1_GLOOP2(k,j)                                                            \
    M1_GLOOP1(i)

#endif
