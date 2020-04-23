#ifndef HYDRO_CONFORMAL_
#define HYDRO_CONFORMAL_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file conformal.hpp
//  \brief declarations for Conformal class

// Athena++ headers
#include "../athena.hpp"

class Conformal {
  public:
    //! Proper time derivative of the conformal factor as a function of conformal time
    /*! This function should be provided by the problem generator */
    Real expansionVelocity(Real tau);
    //! Conformal factor as a function of conformal time
    /*! This function should be provided by the problem generator */
    Real conformalFactor(Real tau);
    //! Physical time as a function of conformal time
    /*! This function should be provided by the problem generator */
    Real physicalTime(Real tau);
};

#endif