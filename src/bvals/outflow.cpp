//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file outflow.cpp
//  \brief implementation of outflow BCs in each dimension

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../z4c/z4c.hpp"
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
//! \fn void OutflowInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          AthenaArray<Real> & waveu, AthenaArray<Real> &z4c, FaceField &b,
//                          Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions, inner x1 boundary

void OutflowInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    AthenaArray<Real> & waveu, AthenaArray<Real> &z4c, FaceField &b,
                    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
  if (HYDRO_ENABLED) {
    for (int n=0; n<(NHYDRO); ++n) {
      for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          prim(n,k,j,is-i) = prim(n,k,j,is);
        }
      }}
    }
  }

  // extrapolate wave equation variables at 4th order
  if (WAVE_ENABLED) {
    for (int n = 0; n < 2; ++n) {
      for (int k = ks; k <= ke; ++k)
      for (int j = js; j <= js; ++j) {
#pragma omp simd
        for (int i = is-1; i >= is-ngh; --i) {
          waveu(n,k,j,i) = 4.*waveu(n,k,j,i+1) - 6.*waveu(n,k,j,i+2) +
                           4.*waveu(n,k,j,i+3) - 1.*waveu(n,k,j,i+4);
        }
      }
    }
  }

  // extrapolate Z4c variables at 4th order
  if (Z4C_ENABLED) {
    for (int n = 0; n < Z4c::N_Z4c; ++n) {
      for (int k = ks; k <= ke; ++k)
      for (int j = js; j <= js; ++j) {
#pragma omp simd
        for (int i = is-1; i >= is-ngh; --i) {
          z4c(n,k,j,i) = 4.*z4c(n,k,j,i+1) - 6.*z4c(n,k,j,i+2) +
                         4.*z4c(n,k,j,i+3) - 1.*z4c(n,k,j,i+4);
        }
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x1f(k,j,(is-i)) = b.x1f(k,j,is);
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x2f(k,j,(is-i)) = b.x2f(k,j,is);
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x3f(k,j,(is-i)) = b.x3f(k,j,is);
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void OutflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          AthenaArray<Real> & waveu, AthenaArray<Real> &z4c, FaceField &b,
//                          Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions, outer x1 boundary

void OutflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    AthenaArray<Real> & waveu, AthenaArray<Real> &z4c, FaceField &b,
                    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
  if (HYDRO_ENABLED) {
    for (int n=0; n<(NHYDRO); ++n) {
      for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          prim(n,k,j,ie+i) = prim(n,k,j,ie);
          }
      }}
    }
  }

  // extrapolate wave equation variables at 4th order
  if (WAVE_ENABLED) {
    for (int n = 0; n < 2; ++n) {
      for (int k = ks; k <= ke; ++k)
      for (int j = js; j <= js; ++j) {
#pragma omp simd
        for (int i = ie+1; i <= ie+ngh; ++i) {
          waveu(n,k,j,i) = 4.*waveu(n,k,j,i-1) - 6.*waveu(n,k,j,i-2) +
                           4.*waveu(n,k,j,i-3) - 1.*waveu(n,k,j,i-4);
        }
      }
    }
  }

  // extrapolate Z4c variables at 4th order
  if (Z4C_ENABLED) {
    for (int n = 0; n < Z4c::N_Z4c; ++n) {
      for (int k = ks; k <= ke; ++k)
      for (int j = js; j <= js; ++j) {
#pragma omp simd
        for (int i = ie+1; i <= ie+ngh; ++i) {
          z4c(n,k,j,i) = 4.*z4c(n,k,j,i-1) - 6.*z4c(n,k,j,i-2) +
                         4.*z4c(n,k,j,i-3) - 1.*z4c(n,k,j,i-4);
        }
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x1f(k,j,(ie+i+1)) = b.x1f(k,j,(ie+1));
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x2f(k,j,(ie+i)) = b.x2f(k,j,ie);
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x3f(k,j,(ie+i)) = b.x3f(k,j,ie);
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void OutflowInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          AthenaArray<Real> & waveu, AthenaArray<Real> &z4c, FaceField &b,
//                          Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions, inner x2 boundary

void OutflowInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    AthenaArray<Real> & waveu, AthenaArray<Real> &z4c, FaceField &b,
                    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
  if (HYDRO_ENABLED) {
    for (int n=0; n<(NHYDRO); ++n) {
      for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          prim(n,k,js-j,i) = prim(n,k,js,i);
        }
      }}
    }
  }

  // extrapolate wave equation variables at 4th order
  if (WAVE_ENABLED) {
    for (int n = 0; n < 2; ++n) {
      for (int k = ks; k <= ke; ++k)
      for (int j = js-1; j >= js-ngh; --j) {
#pragma omp simd
        for (int i = is; i <= ie; ++i) {
          waveu(n,k,j,i) = 4.*waveu(n,k,j+1,i) - 6.*waveu(n,k,j+2,i) +
                           4.*waveu(n,k,j+3,i) - 1.*waveu(n,k,j+4,i);
        }
      }
    }
  }

  // extrapolate Z4c variables at 4th order
  if (Z4C_ENABLED) {
    for (int n = 0; n < Z4c::N_Z4c; ++n) {
      for (int k = ks; k <= ke; ++k)
      for (int j = js-1; j >= js-ngh; --j) {
#pragma omp simd
        for (int i = is; i <= ie; ++i) {
          z4c(n,k,j,i) = 4.*z4c(n,k,j+1,i) - 6.*z4c(n,k,j+2,i) +
                         4.*z4c(n,k,j+3,i) - 1.*z4c(n,k,j+4,i);
        }
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f(k,(js-j),i) = b.x1f(k,js,i);
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f(k,(js-j),i) = b.x2f(k,js,i);
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f(k,(js-j),i) = b.x3f(k,js,i);
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void OutflowOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          AthenaArray<Real> & waveu, AthenaArray<Real> &z4c, FaceField &b,
//                          Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions, outer x2 boundary

void OutflowOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    AthenaArray<Real> & waveu, AthenaArray<Real> &z4c, FaceField &b,
                    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
  if (HYDRO_ENABLED) {
    for (int n=0; n<(NHYDRO); ++n) {
      for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          prim(n,k,je+j,i) = prim(n,k,je,i);
        }
      }}
    }
  }

  // extrapolate wave equation variables at 4th order
  if (WAVE_ENABLED) {
    for (int n = 0; n < 2; ++n) {
      for (int k = ks; k <= ke; ++k)
      for (int j = je+1; j <= je+ngh; ++j) {
#pragma omp simd
        for (int i = is; i <= ie; ++i) {
          waveu(n,k,j,i) = 4.*waveu(n,k,j-1,i) - 6.*waveu(n,k,j-2,i) +
                           4.*waveu(n,k,j-3,i) - 1.*waveu(n,k,j-4,i);
        }
      }
    }
  }

  // extrapolate Z4c variables at 4th order
  if (Z4C_ENABLED) {
    for (int n = 0; n < Z4c::N_Z4c; ++n) {
      for (int k = ks; k <= ke; ++k)
      for (int j = je+1; j <= je+ngh; ++j) {
#pragma omp simd
        for (int i = is; i <= ie; ++i) {
          z4c(n,k,j,i) = 4.*z4c(n,k,j-1,i) - 6.*z4c(n,k,j-2,i) +
                         4.*z4c(n,k,j-3,i) - 1.*z4c(n,k,j-4,i);
        }
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f(k,(je+j  ),i) = b.x1f(k,(je  ),i);
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f(k,(je+j+1),i) = b.x2f(k,(je+1),i);
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f(k,(je+j  ),i) = b.x3f(k,(je  ),i);
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void OutflowInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          AthenaArray<Real> & waveu, AthenaArray<Real> &z4c, FaceField &b,
//                          Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions, inner x3 boundary

void OutflowInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    AthenaArray<Real> & waveu, AthenaArray<Real> &z4c, FaceField &b,
                    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
  if (HYDRO_ENABLED) {
    for (int n=0; n<(NHYDRO); ++n) {
      for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          prim(n,ks-k,j,i) = prim(n,ks,j,i);
        }
      }}
    }
  }

  // extrapolate wave equation variables at 4th order
  if (WAVE_ENABLED) {
    for (int n = 0; n < 2; ++n) {
      for (int k = ks-1; k >= ks-ngh; --k)
      for (int j = js; j <= js; ++j) {
#pragma omp simd
        for (int i = is; i <= ie; ++i) {
          waveu(n,k,j,i) = 4.*waveu(n,k+1,j,i) - 6.*waveu(n,k+2,j,i) +
                           4.*waveu(n,k+3,j,i) - 1.*waveu(n,k+4,j,i);
        }
      }
    }
  }

  // extrapolate Z4c variables at 4th order
  if (Z4C_ENABLED) {
    for (int n = 0; n < Z4c::N_Z4c; ++n) {
      for (int k = ks-1; k >= ks-ngh; --k)
      for (int j = js; j <= js; ++j) {
#pragma omp simd
        for (int i = is; i <= ie; ++i) {
          z4c(n,k,j,i) = 4.*z4c(n,k+1,j,i) - 6.*z4c(n,k+2,j,i) +
                         4.*z4c(n,k+3,j,i) - 1.*z4c(n,k+4,j,i);
        }
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f((ks-k),j,i) = b.x1f(ks,j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f((ks-k),j,i) = b.x2f(ks,j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f((ks-k),j,i) = b.x3f(ks,j,i);
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void OutflowOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          AthenaArray<Real> & waveu, AthenaArray<Real> &z4c, FaceField &b,
//                          Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions, outer x3 boundary

void OutflowOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    AthenaArray<Real> & waveu, AthenaArray<Real> &z4c, FaceField &b,
                    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
  if (HYDRO_ENABLED) {
    for (int n=0; n<(NHYDRO); ++n) {
      for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          prim(n,ke+k,j,i) = prim(n,ke,j,i);
        }
      }}
    }
  }

  // extrapolate wave equation variables at 4th order
  if (WAVE_ENABLED) {
    for (int n = 0; n < 2; ++n) {
      for (int k = ke+1; k <= ke+ngh; ++k)
      for (int j = js; j <= js; ++j) {
#pragma omp simd
        for (int i = is; i <= ie; ++i) {
          waveu(n,k,j,i) = 4.*waveu(n,k-1,j,i) - 6.*waveu(n,k-2,j,i) +
                           4.*waveu(n,k-3,j,i) - 1.*waveu(n,k-4,j,i);
        }
      }
    }
  }

  // extrapolate Z4c variables at 4th order
  if (Z4C_ENABLED) {
    for (int n = 0; n < Z4c::N_Z4c; ++n) {
      for (int k = ke+1; k <= ke+ngh; ++k)
      for (int j = js; j <= js; ++j) {
#pragma omp simd
        for (int i = is; i <= ie; ++i) {
          z4c(n,k,j,i) = 4.*z4c(n,k-1,j,i) - 6.*z4c(n,k-2,j,i) +
                         4.*z4c(n,k-3,j,i) - 1.*z4c(n,k-4,j,i);
        }
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f((ke+k  ),j,i) = b.x1f((ke  ),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f((ke+k  ),j,i) = b.x2f((ke  ),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f((ke+k+1),j,i) = b.x3f((ke+1),j,i);
      }
    }}
  }

  return;
}
