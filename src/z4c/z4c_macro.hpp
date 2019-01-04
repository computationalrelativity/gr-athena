#ifndef Z4c_MACRO_HPP
#define Z4c_MACRO_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_macro.hpp
//  \brief common macros for the Z4c class

// 2D loop over k and j in the interior of the block
#define ILOOP2(k,j)                                                           \
  for(int k = pmy_block->ks; k <= pmy_block->ke; ++k)                         \
  for(int j = pmy_block->js; j <= pmy_block->je; ++j)

// 2D loop over k and j on the whole block
#define GLOOP2(k,j)                                                           \
  for(int k = pmy_block->ks - NGHOST; k <= pmy_block->ke + NGHOST; ++k)       \
  for(int j = pmy_block->js - NGHOST; j <= pmy_block->je + NGHOST; ++j)

// 1D loop over i in the interior of the block
#define ILOOP1(i)                                                             \
  _Pragma("omp simd")                                                         \
  for(int i = pmy_block->is; i <= pmy_block->ie; ++i)

// 1D loop over i in the interior of the block
#define GLOOP1(i)                                                             \
  _Pragma("omp simd")                                                         \
  for(int i = pmy_block->is - NGHOST; i <= pmy_block->ie + NGHOST; ++i)

// 3D loop over the interior of the block
#define ILOOP3(k,j,i)                                                         \
    ILOOP2(k,j)                                                               \
    ILOOP1(i)

// 3D loop over the whole block
#define GLOOP3(k,j,i)                                                         \
    GLOOP2(k,j)                                                               \
    GLOOP1(i)

#endif
