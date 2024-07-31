#ifndef M1_MACRO_HPP
#define M1_MACRO_HPP

// ============================================================================
namespace M1 {
// ============================================================================

// Various constants ----------------------------------------------------------

// TODO: move to sane place with all units
#ifndef M1_UNITS_CGS_GCC
#define M1_UNITS_CGS_GCC (1.619100425158886e-18)  // CGS density conv. fact
#endif

#ifndef M1_NDIM
#define M1_NDIM 3
#endif

#ifndef M1_NGHOST_MIN
#define M1_NGHOST_MIN 2
#endif

// Indicial magic -------------------------------------------------------------
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

#define M1_FSIZEI                                                             \
  (M1_NGHOST_MIN)

#define M1_FSIZEJ                                                             \
  ((pm1->mbi.f2) ? (M1_NGHOST_MIN) : (0))

#define M1_FSIZEK                                                             \
  ((pm1->mbi.f3) ? (M1_NGHOST_MIN) : (0))


// 2D loop over k and j in the interior of the block
#define M1_ILOOP2(k,j)                                                        \
  for(int k = M1_IX_KL; k <= M1_IX_KU; ++k)                                   \
  for(int j = M1_IX_JL; j <= M1_IX_JU; ++j)

// 2D loop over k and j over flux-salient nodes
#define M1_FLOOP2(k,j)                                                        \
  for(int k = M1_IX_KL - M1_FSIZEK; k <= M1_IX_KU + M1_FSIZEK; ++k)           \
  for(int j = M1_IX_JL - M1_FSIZEJ; j <= M1_IX_JU + M1_FSIZEJ; ++j)

// 2D loop over k and j on the whole block
#define M1_GLOOP2(k,j)                                                        \
  for(int k = M1_IX_KL - M1_GSIZEK; k <= M1_IX_KU + M1_GSIZEK; ++k)           \
  for(int j = M1_IX_JL - M1_GSIZEJ; j <= M1_IX_JU + M1_GSIZEJ; ++j)

// 1D loop over i in the interior of the block
#define M1_ILOOP1(i)                                                          \
  _Pragma("omp simd")                                                         \
  for(int i = M1_IX_IL; i <= M1_IX_IU; ++i)

// 1D loop over i over flux-salient nodes
#define M1_FLOOP1(i)                                                          \
  _Pragma("omp simd")                                                         \
  for(int i = M1_IX_IL - M1_FSIZEI; i <= M1_IX_IU + M1_FSIZEI; ++i)

// 1D loop over i on the whole block
#define M1_GLOOP1(i)                                                          \
  _Pragma("omp simd")                                                         \
  for(int i = M1_IX_IL - M1_GSIZEI; i <= M1_IX_IU + M1_GSIZEI; ++i)

// 3D loop over the interior of the block
#define M1_ILOOP3(k,j,i)                                                      \
    M1_ILOOP2(k,j)                                                            \
    M1_ILOOP1(i)

// 3D loop over flux-salient nodes
#define M1_FLOOP3(k,j,i)                                                      \
    M1_FLOOP2(k,j)                                                            \
    M1_FLOOP1(i)

// 3D loop over the whole block
#define M1_GLOOP3(k,j,i)                                                      \
    M1_GLOOP2(k,j)                                                            \
    M1_GLOOP1(i)

// ============================================================================
} // namespace M1
// ============================================================================

#endif  // M1_MACRO_HPP

//
// :D
//