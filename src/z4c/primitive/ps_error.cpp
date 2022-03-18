//! \file ps_error.cpp
//  \brief Defines labels for error enumerator.
#include "ps_error.hpp"

std::string Primitive::ErrorString[10] = {"SUCCESS",
                                          "RHO_TOO_BIG",
                                          "RHO_TOO_SMALL",
                                          "NANS_IN_CONS",
                                          "MAG_TOO_BIG",
                                          "BRACKETING_FAILED",
                                          "NO_SOLUTION",
                                          "CONS_FLOOR",
                                          "PRIM_FLOOR",
                                          "CONS_ADJUSTED"};
