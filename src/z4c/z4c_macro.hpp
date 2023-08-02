#ifndef Z4c_MACRO_HPP
#define Z4c_MACRO_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_macro.hpp
//  \brief common macros for the Z4c class

// BD: refactor scope - as this is useful elsewhere?
// BD: usage needs cleanup elsewhere

#if defined(Z4C_CC_ENABLED)
  #define SW_CC_CX_VC(a, b, c) \
    a
#elif defined(Z4C_CX_ENABLED)
  #define SW_CC_CX_VC(a, b, c) \
    b
#else
  #define SW_CC_CX_VC(a, b, c) \
    c
#endif

// BD: derive from mbi (f1,f2,f3) _or_ fix as 3 (Manifold dimension) ?
// #define NDIM    (mbi.ndim)
#define NDIM    (3)

#define IX_IL                                                                 \
  pz4c->mbi.il

#define IX_IU                                                                 \
  pz4c->mbi.iu

#define IX_JL                                                                 \
  pz4c->mbi.jl

#define IX_JU                                                                 \
  pz4c->mbi.ju

#define IX_KL                                                                 \
  pz4c->mbi.kl

#define IX_KU                                                                 \
  pz4c->mbi.ku

#define GSIZEI                                                                \
  (pz4c->mbi.ng)

#define GSIZEJ                                                                \
  ((pz4c->mbi.f2) ? (pz4c->mbi.ng) : (0))

#define GSIZEK                                                                \
  ((pz4c->mbi.f3) ? (pz4c->mbi.ng) : (0))

// 2D loop over k and j in the interior of the block
#define ILOOP2(k,j)                                                           \
  for(int k = IX_KL; k <= IX_KU; ++k)                                         \
  for(int j = IX_JL; j <= IX_JU; ++j)

// 2D loop over k and j on the whole block
#define GLOOP2(k,j)                                                           \
  for(int k = IX_KL - GSIZEK; k <= IX_KU + GSIZEK; ++k)                       \
  for(int j = IX_JL - GSIZEJ; j <= IX_JU + GSIZEJ; ++j)

// 1D loop over i in the interior of the block
#define ILOOP1(i)                                                             \
  _Pragma("omp simd")                                                         \
  for(int i = IX_IL; i <= IX_IU; ++i)

// 1D loop over i on the whole block
#define GLOOP1(i)                                                             \
  _Pragma("omp simd")                                                         \
  for(int i = IX_IL - GSIZEI; i <= IX_IU + GSIZEI; ++i)

// 3D loop over the interior of the block
#define ILOOP3(k,j,i)                                                         \
    ILOOP2(k,j)                                                               \
    ILOOP1(i)

// 3D loop over the whole block
#define GLOOP3(k,j,i)                                                         \
    GLOOP2(k,j)                                                               \
    GLOOP1(i)

#endif
