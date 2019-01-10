//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file reflect.cpp
//  \brief implementation of reflecting BCs in each dimension

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../z4c/z4c.hpp"
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ReflectInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          AthenaArray<Real> &z4c, FaceField &b, const Real time, const Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief REFLECTING boundary conditions, inner x1 boundary

void ReflectInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    AthenaArray<Real> &z4c, FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones, reflecting v1
  if (HYDRO_ENABLED) {
    for (int n=0; n<(NHYDRO); ++n) {
      if (n==(IVX)) {
        for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            prim(IVX,k,j,is-i) = -prim(IVX,k,j,(is+i-1));  // reflect 1-velocity
          }
        }}
      } else {
        for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            prim(n,k,j,is-i) = prim(n,k,j,(is+i-1));
          }
        }}
      }
    }
  }

  // copy spacetime variables into ghost zones, reflecting when needed
  if (Z4C_ENABLED) {
    for (int n = 0; n < Z4c::N_Z4c; ++n) {
      if (n == Z4c::I_Z4c_gxy || n == Z4c::I_Z4c_gxz ||
          n == Z4c::I_Z4c_Axy || n == Z4c::I_Z4c_Axz ||
          n == Z4c::I_Z4c_Gamx || n == Z4c::I_Z4c_betax) {
        for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            z4c(n,k,j,is-i) = -z4c(n,k,j,(is+i-1));
          }
        }}
      } else {
        for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            z4c(n,k,j,is-i) = z4c(n,k,j,(is+i-1));
          }
        }}
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones, reflecting b1
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x1f(k,j,(is-i)) = -b.x1f(k,j,(is+i  ));  // reflect 1-field
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x2f(k,j,(is-i)) =  b.x2f(k,j,(is+i-1));
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x3f(k,j,(is-i)) =  b.x3f(k,j,(is+i-1));
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ReflectOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          AthenaArray<Real> &z4c, FaceField &b, const Real time, const Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief REFLECTING boundary conditions, outer x1 boundary

void ReflectOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    AthenaArray<Real> &z4c, FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones, reflecting v1
  if (HYDRO_ENABLED) {
    for (int n=0; n<(NHYDRO); ++n) {
      if (n==(IVX)) {
        for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            prim(IVX,k,j,ie+i) = -prim(IVX,k,j,(ie-i+1));  // reflect 1-velocity
          }
        }}
      } else {
        for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            prim(n,k,j,ie+i) = prim(n,k,j,(ie-i+1));
          }
        }}
      }
    }
  }

  // copy spacetime variables into ghost zones, reflecting when needed
  if (Z4C_ENABLED) {
    for (int n = 0; n < Z4c::N_Z4c; ++n) {
      if (n == Z4c::I_Z4c_gxy || n == Z4c::I_Z4c_gxz ||
          n == Z4c::I_Z4c_Axy || n == Z4c::I_Z4c_Axz ||
          n == Z4c::I_Z4c_Gamx || n == Z4c::I_Z4c_betax) {
        for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            z4c(n,k,j,ie+i) = -z4c(n,k,j,(ie-i+1));
          }
        }}
      } else {
        for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            z4c(n,k,j,ie+i) = z4c(n,k,j,(ie-i+1));
          }
        }}
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones, reflecting b1
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x1f(k,j,(ie+i+1)) = -b.x1f(k,j,(ie-i+1));  // reflect 1-field
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x2f(k,j,(ie+i  )) =  b.x2f(k,j,(ie-i+1));
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x3f(k,j,(ie+i  )) =  b.x3f(k,j,(ie-i+1));
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ReflecInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                         AthenaArray<Real> &z4c, FaceField &b, const Real time, const Real dt,
//                         int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief REFLECTING boundary conditions, inner x2 boundary

void ReflectInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    AthenaArray<Real> &z4c, FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones, reflecting v2
  if (HYDRO_ENABLED) {
    for (int n=0; n<(NHYDRO); ++n) {
      if (n==(IVY)) {
        for (int k=ks; k<=ke; ++k) {
        for (int j=1; j<=ngh; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            prim(IVY,k,js-j,i) = -prim(IVY,k,js+j-1,i);  // reflect 2-velocity
          }
        }}
      } else {
        for (int k=ks; k<=ke; ++k) {
        for (int j=1; j<=ngh; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            prim(n,k,js-j,i) = prim(n,k,js+j-1,i);
          }
        }}
      }
    }
  }

  // copy spacetime variables into ghost zones, reflecting when needed
  if (Z4C_ENABLED) {
    for (int n = 0; n < Z4c::N_Z4c; ++n) {
      if (n == Z4c::I_Z4c_gxy || n == Z4c::I_Z4c_gyz ||
          n == Z4c::I_Z4c_Axy || n == Z4c::I_Z4c_Ayz ||
          n == Z4c::I_Z4c_Gamy || n == Z4c::I_Z4c_betay) {
        for (int k=ks; k<=ke; ++k) {
        for (int j=1; j<=ngh; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            z4c(n,k,js-j,i) = -z4c(n,k,js+j-1,i);
          }
        }}
      } else {
        for (int k=ks; k<=ke; ++k) {
        for (int j=1; j<=ngh; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            z4c(n,k,js-j,i) = z4c(n,k,js+j-1,i);
          }
        }}
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones, reflecting b2
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f(k,(js-j),i) =  b.x1f(k,(js+j-1),i);
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f(k,(js-j),i) = -b.x2f(k,(js+j  ),i);  // reflect 2-field
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f(k,(js-j),i) =  b.x3f(k,(js+j-1),i);
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ReflectOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          AthenaArray<Real> &z4c, FaceField &b, const Real time, const Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief REFLECTING boundary conditions, outer x2 boundary

void ReflectOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    AthenaArray<Real> &z4c, FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones, reflecting v2
  if (HYDRO_ENABLED) {
    for (int n=0; n<(NHYDRO); ++n) {
      if (n==(IVY)) {
        for (int k=ks; k<=ke; ++k) {
        for (int j=1; j<=ngh; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            prim(IVY,k,je+j,i) = -prim(IVY,k,je-j+1,i);  // reflect 2-velocity
          }
        }}
      } else {
        for (int k=ks; k<=ke; ++k) {
        for (int j=1; j<=ngh; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            prim(n,k,je+j,i) = prim(n,k,je-j+1,i);
          }
        }}
      }
    }
  }

  // copy spacetime variables into ghost zones, reflecting when needed
  if (Z4C_ENABLED) {
    for (int n = 0; n < Z4c::N_Z4c; ++n) {
      if (n == Z4c::I_Z4c_gxy || n == Z4c::I_Z4c_gyz ||
          n == Z4c::I_Z4c_Axy || n == Z4c::I_Z4c_Ayz ||
          n == Z4c::I_Z4c_Gamy || n == Z4c::I_Z4c_betay) {
        for (int k=ks; k<=ke; ++k) {
        for (int j=1; j<=ngh; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            z4c(n,k,je+j,i) = -z4c(n,k,je-j+1,i);
          }
        }}
      } else {
        for (int k=ks; k<=ke; ++k) {
        for (int j=1; j<=ngh; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            z4c(n,k,je+j,i) = z4c(n,k,je-j+1,i);
          }
        }}
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones, reflecting b2
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f(k,(je+j  ),i) =  b.x1f(k,(je-j+1),i);
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f(k,(je+j+1),i) = -b.x2f(k,(je-j+1),i);  // reflect 2-field
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f(k,(je+j  ),i) =  b.x3f(k,(je-j+1),i);
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ReflectInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          AthenaArray<Real> &z4c, FaceField &b, const Real time, const Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief REFLECTING boundary conditions, inner x3 boundary

void ReflectInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    AthenaArray<Real> &z4c, FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones, reflecting v3
  if (HYDRO_ENABLED) {
    for (int n=0; n<(NHYDRO); ++n) {
      if (n==(IVZ)) {
        for (int k=1; k<=ngh; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            prim(IVZ,ks-k,j,i) = -prim(IVZ,ks+k-1,j,i);  // reflect 3-velocity
          }
        }}
      } else {
        for (int k=1; k<=ngh; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            prim(n,ks-k,j,i) = prim(n,ks+k-1,j,i);
          }
        }}
      }
    }
  }

  // copy spacetime variables into ghost zones, reflecting when needed
  if (Z4C_ENABLED) {
    for (int n = 0; n < Z4c::N_Z4c; ++n) {
      if (n == Z4c::I_Z4c_gxz || n == Z4c::I_Z4c_gyz ||
          n == Z4c::I_Z4c_Axz || n == Z4c::I_Z4c_Ayz ||
          n == Z4c::I_Z4c_Gamz || n == Z4c::I_Z4c_betaz) {
        for (int k=1; k<=ngh; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            z4c(n,ks-k,j,i) = -z4c(n,ks+k-1,j,i);
          }
        }}
      } else {
        for (int k=1; k<=ngh; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            z4c(n,ks-k,j,i) = z4c(n,ks+k-1,j,i);
          }
        }}
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones, reflecting b3
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f((ks-k),j,i) =  b.x1f((ks+k-1),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f((ks-k),j,i) =  b.x2f((ks+k-1),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f((ks-k),j,i) = -b.x3f((ks+k  ),j,i);  // reflect 3-field
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ReflectOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          AthenaArray<Real> &z4c, FaceField &b, const Real time, const Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief REFLECTING boundary conditions, outer x3 boundary

void ReflectOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    AthenaArray<Real> &z4c, FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones, reflecting v3
  if (HYDRO_ENABLED) {
    for (int n=0; n<(NHYDRO); ++n) {
      if (n==(IVZ)) {
        for (int k=1; k<=ngh; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            prim(IVZ,ke+k,j,i) = -prim(IVZ,ke-k+1,j,i);  // reflect 3-velocity
          }
        }}
      } else {
        for (int k=1; k<=ngh; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            prim(n,ke+k,j,i) = prim(n,ke-k+1,j,i);
          }
        }}
      }
    }
  }

  // copy spacetime variables into ghost zones, reflecting when needed
  if (Z4C_ENABLED) {
    for (int n = 0; n < Z4c::N_Z4c; ++n) {
      if (n == Z4c::I_Z4c_gxz || n == Z4c::I_Z4c_gyz ||
          n == Z4c::I_Z4c_Axz || n == Z4c::I_Z4c_Ayz ||
          n == Z4c::I_Z4c_Gamz || n == Z4c::I_Z4c_betaz) {
        for (int k=1; k<=ngh; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            z4c(n,ke+k,j,i) = -z4c(n,ke-k+1,j,i);
          }
        }}
      } else {
        for (int k=1; k<=ngh; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            z4c(n,ke+k,j,i) = z4c(n,ke-k+1,j,i);
          }
        }}
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones, reflecting b3
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f((ke+k  ),j,i) =  b.x1f((ke-k+1),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f((ke+k  ),j,i) =  b.x2f((ke-k+1),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f((ke+k+1),j,i) = -b.x3f((ke-k+1),j,i);  // reflect 3-field
      }
    }}
  }

  return;
}
