#ifndef WAVE_MACRO_HPP
#define WAVE_MACRO_HPP

#if WAVE_CC_ENABLED
  #define WAVE_SW_CC_CX_VC(a, b, c)                                                 \
    a
#elif WAVE_CX_ENABLED
  #define WAVE_SW_CC_CX_VC(a, b, c)                                                 \
    b
#else
  #define WAVE_SW_CC_CX_VC(a, b, c)                                                 \
    c
#endif

#if WAVE_CC_ENABLED || WAVE_CX_ENABLED
  #define WAVE_SW_CCX_VC(a, b)                                                      \
    a
#else
  #define WAVE_SW_CCX_VC(a, b)                                                      \
    b
#endif

// BD: allow replacement slots with function-like specification
#if WAVE_CC_ENABLED
  #define WAVE_FCN_CC_CX_VC(A, B, C) \
    A
#elif WAVE_CX_ENABLED
  #define WAVE_FCN_CC_CX_VC(A, B, C) \
    B
#else
  #define WAVE_FCN_CC_CX_VC(A, B, C) \
    C
#endif

#endif // WAVE_MACRO_HPP

//
// :D
//