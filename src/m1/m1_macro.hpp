#ifndef M1_MACRO_HPP
#define M1_MACRO_HPP

// ============================================================================
namespace M1 {
// ============================================================================

// Various constants ----------------------------------------------------------

// BD: TODO - move to sane place with all units
#ifndef M1_UNITS_CGS_GCC
#define M1_UNITS_CGS_GCC (1.619100425158886e-18)  // CGS density conv. fact
#endif

#ifndef M1_NDIM
#define M1_NDIM 3
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

// if positivity preservation utilized, extend here masking loops
#define M1_MSIZEI                                                             \
  pm1->opt.flux_lo_fallback

#define M1_MSIZEJ                                                             \
  ((pm1->mbi.f2) ? (pm1->opt.flux_lo_fallback) : (0))

#define M1_MSIZEK                                                             \
  ((pm1->mbi.f3) ? (pm1->opt.flux_lo_fallback) : (0))

#define M1_FSIZEI                                                             \
  (1 + pm1->opt.flux_lo_fallback)

#define M1_FSIZEJ                                                             \
  ((pm1->mbi.f2) ? (1 + pm1->opt.flux_lo_fallback) : (0))

#define M1_FSIZEK                                                             \
  ((pm1->mbi.f3) ? (1 + pm1->opt.flux_lo_fallback) : (0))

// if positivity preservation utilized, extend here reconstruction loops
#define M1_RSIZEI                                                             \
  pm1->opt.flux_lo_fallback

#define M1_RSIZEJ                                                             \
  ((pm1->mbi.f2) ? (pm1->opt.flux_lo_fallback) : (0))

#define M1_RSIZEK                                                             \
  ((pm1->mbi.f3) ? (pm1->opt.flux_lo_fallback) : (0))

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

// 2D loop over k and j for mask nodes
#define M1_MLOOP2(k,j)                                                        \
  for(int k = M1_IX_KL-M1_MSIZEK; k <= M1_IX_KU+M1_MSIZEK; ++k)               \
  for(int j = M1_IX_JL-M1_MSIZEJ; j <= M1_IX_JU+M1_MSIZEJ; ++j)

// 2D loop over k and j over flux-salient nodes
#define M1_FLOOP2(k,j)                                                        \
  for(int k = M1_IX_KL-M1_FSIZEK; k <= M1_IX_KU+M1_FSIZEK+1; ++k)             \
  for(int j = M1_IX_JL-M1_FSIZEJ; j <= M1_IX_JU+M1_FSIZEJ+1; ++j)

// 2D loop over k and j on the whole block
#define M1_GLOOP2(k,j)                                                        \
  for(int k = M1_IX_KL-M1_GSIZEK; k <= M1_IX_KU+M1_GSIZEK; ++k)               \
  for(int j = M1_IX_JL-M1_GSIZEJ; j <= M1_IX_JU+M1_GSIZEJ; ++j)

// 1D loop over i in the interior of the block
#define M1_ILOOP1(i)                                                          \
  _Pragma("omp simd")                                                         \
  for(int i = M1_IX_IL; i <= M1_IX_IU; ++i)

// 1D loop over i for mask nodes
#define M1_MLOOP1(i)                                                          \
  _Pragma("omp simd")                                                         \
  for(int i = M1_IX_IL-M1_MSIZEI; i <= M1_IX_IU+M1_MSIZEI; ++i)

// 1D loop over i over flux-salient nodes
#define M1_FLOOP1(i)                                                          \
  _Pragma("omp simd")                                                         \
  for(int i = M1_IX_IL-M1_FSIZEI; i <= M1_IX_IU+M1_FSIZEI+1; ++i)

// 1D loop over i on the whole block
#define M1_GLOOP1(i)                                                          \
  _Pragma("omp simd")                                                         \
  for(int i = M1_IX_IL-M1_GSIZEI; i <= M1_IX_IU+M1_GSIZEI; ++i)

// 3D loop over the interior of the block
#define M1_ILOOP3(k,j,i)                                                      \
    M1_ILOOP2(k,j)                                                            \
    M1_ILOOP1(i)

// 3D loop over the interior + 1 ghost
#define M1_MLOOP3(k,j,i)                                                      \
    M1_MLOOP2(k,j)                                                            \
    M1_MLOOP1(i)

// 3D loop over flux-salient nodes
#define M1_FLOOP3(k,j,i)                                                      \
    M1_FLOOP2(k,j)                                                            \
    M1_FLOOP1(i)

// 3D loop over the whole block
#define M1_GLOOP3(k,j,i)                                                      \
    M1_GLOOP2(k,j)                                                            \
    M1_GLOOP1(i)

// 3D loop for reconstruction

// flux_idx : i-1/2

#define M1_RLOOP2_1(k,j)                                                      \
  for(int k = M1_IX_KL-M1_RSIZEK; k <= M1_IX_KU+M1_RSIZEK; ++k)               \
  for(int j = M1_IX_JL-M1_RSIZEJ; j <= M1_IX_JU+M1_RSIZEJ; ++j)
#define M1_RLOOP1_1(i)                                                        \
  _Pragma("omp simd")                                                         \
  for(int i = M1_IX_IL-M1_RSIZEI; i <= M1_IX_IU+M1_RSIZEI+1; ++i)
#define M1_RLOOP3_1(k,j,i)                                                    \
    M1_RLOOP2_1(k,j)                                                          \
    M1_RLOOP1_1(i)

// flux_idx : j-1/2
#define M1_RLOOP2_2(k,j)                                                      \
  for(int k = M1_IX_KL-M1_RSIZEK; k <= M1_IX_KU+M1_RSIZEK; ++k)               \
  for(int j = M1_IX_JL-M1_RSIZEJ; j <= M1_IX_JU+M1_RSIZEJ+1; ++j)
#define M1_RLOOP1_2(i)                                                        \
  _Pragma("omp simd")                                                         \
  for(int i = M1_IX_IL-M1_RSIZEI; i <= M1_IX_IU+M1_RSIZEI; ++i)
#define M1_RLOOP3_2(k,j,i)                                                    \
    M1_RLOOP2_2(k,j)                                                          \
    M1_RLOOP1_2(i)

// flux_idx : k-1/2
#define M1_RLOOP2_3(k,j)                                                      \
  for(int k = M1_IX_KL-M1_RSIZEK; k <= M1_IX_KU+M1_RSIZEK+1; ++k)             \
  for(int j = M1_IX_JL-M1_RSIZEJ; j <= M1_IX_JU+M1_RSIZEJ; ++j)
#define M1_RLOOP1_3(i)                                                        \
  _Pragma("omp simd")                                                         \
  for(int i = M1_IX_IL-M1_RSIZEI; i <= M1_IX_IU+M1_RSIZEI; ++i)
#define M1_RLOOP3_3(k,j,i)                                                    \
    M1_RLOOP2_3(k,j)                                                          \
    M1_RLOOP1_3(i)

// 3D loop for limiting mask
#define M1_LLOOP2_1(k,j)                                                      \
  for(int k = M1_IX_KL-M1_FSIZEK; k <= M1_IX_KU+M1_FSIZEK; ++k)               \
  for(int j = M1_IX_JL-M1_FSIZEJ; j <= M1_IX_JU+M1_FSIZEJ; ++j)
#define M1_LLOOP1_1(i)                                                        \
  _Pragma("omp simd")                                                         \
  for(int i = M1_IX_IL-M1_FSIZEI; i <= M1_IX_IU+M1_FSIZEI+1; ++i)
#define M1_LLOOP3_1(k,j,i)                                                    \
    M1_LLOOP2_1(k,j)                                                          \
    M1_LLOOP1_1(i)

// flux_idx : j-1/2
#define M1_LLOOP2_2(k,j)                                                      \
  for(int k = M1_IX_KL-M1_FSIZEK; k <= M1_IX_KU+M1_FSIZEK; ++k)               \
  for(int j = M1_IX_JL-M1_FSIZEJ; j <= M1_IX_JU+M1_FSIZEJ+1; ++j)
#define M1_LLOOP1_2(i)                                                        \
  _Pragma("omp simd")                                                         \
  for(int i = M1_IX_IL-M1_FSIZEI; i <= M1_IX_IU+M1_FSIZEI; ++i)
#define M1_LLOOP3_2(k,j,i)                                                    \
    M1_LLOOP2_2(k,j)                                                          \
    M1_LLOOP1_2(i)

// flux_idx : k-1/2
#define M1_LLOOP2_3(k,j)                                                      \
  for(int k = M1_IX_KL-M1_FSIZEK; k <= M1_IX_KU+M1_FSIZEK+1; ++k)             \
  for(int j = M1_IX_JL-M1_FSIZEJ; j <= M1_IX_JU+M1_FSIZEJ; ++j)
#define M1_LLOOP1_3(i)                                                        \
  _Pragma("omp simd")                                                         \
  for(int i = M1_IX_IL-M1_FSIZEI; i <= M1_IX_IU+M1_FSIZEI; ++i)
#define M1_LLOOP3_3(k,j,i)                                                    \
    M1_LLOOP2_3(k,j)                                                          \
    M1_LLOOP1_3(i)

// ============================================================================
} // namespace M1
// ============================================================================

#endif  // M1_MACRO_HPP

//
// :D
//
