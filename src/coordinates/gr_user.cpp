//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gr_user.cpp
//  \brief Used for arbitrary stationary coordinates in general relativity, with all
//  functions evaluated numerically based on the metric.
//  Original implementation by CJ White.

// C headers

// C++ headers
#include <cmath>  // sqrt()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "coordinates.hpp"

namespace {
// Declarations
Real Determinant(const AthenaArray<Real> &g);
Real Determinant(Real a11, Real a12, Real a13, Real a21, Real a22, Real a23,
                 Real a31, Real a32, Real a33);
Real Determinant(Real a11, Real a12, Real a21, Real a22);
void CalculateTransformation(
    const AthenaArray<Real> &g,
    const AthenaArray<Real> &g_inv, int face, AthenaArray<Real> &transformation);
} // namespace

//----------------------------------------------------------------------------------------
// GRUser Constructor
// Inputs:
//   pmb: pointer to MeshBlock containing this grid
//   pin: pointer to runtime inputs
//   flag: true if object is for coarse grid only in an AMR calculation

GRUser::GRUser(MeshBlock *pmb, ParameterInput *pin, bool flag)
    : Coordinates(pmb, pin, flag) {
  // Set object names
  RegionSize& block_size = pmy_block->block_size;

  // set more indices
  int ill = il - ng;
  int iuu = iu + ng;
  int jll, juu;
  if (block_size.nx2 > 1) {
    jll = jl - ng;
    juu = ju + ng;
  } else {
    jll = jl;
    juu = ju;
  }
  int kll, kuu;
  if (block_size.nx3 > 1) {
    kll = kl - ng;
    kuu = ku + ng;
  } else {
    kll = kl;
    kuu = ku;
  }

  // Initialize volume-averaged coordinates and spacings: r-direction
  for (int i=ill; i<=iuu; ++i) {
    Real r_m = x1f(i);
    Real r_p = x1f(i+1);
    x1v(i) = 0.5 * (r_m + r_p);  // at least 2nd-order accurate
  }
  for (int i=ill; i<=iuu-1; ++i) {
    dx1v(i) = x1v(i+1) - x1v(i);
  }

  // Initialize volume-averaged coordinates and spacings: theta-direction
  if (block_size.nx2 == 1) {
    Real theta_m = x2f(jl);
    Real theta_p = x2f(jl+1);
    x2v(jl) = 0.5 * (theta_m + theta_p);  // at least 2nd-order accurate
    dx2v(jl) = dx2f(jl);
  } else {
    for (int j=jll; j<=juu; ++j) {
      Real theta_m = x2f(j);
      Real theta_p = x2f(j+1);
      x2v(j) = 0.5 * (theta_m + theta_p);  // at least 2nd-order accurate
    }
    for (int j=jll; j<=juu-1; ++j) {
      dx2v(j) = x2v(j+1) - x2v(j);
    }
  }

  // Initialize volume-averaged coordinates and spacings: phi-direction
  if (block_size.nx3 == 1) {
    Real phi_m = x3f(kl);
    Real phi_p = x3f(kl+1);
    x3v(kl) = 0.5 * (phi_m + phi_p);  // at least 2nd-order accurate
    dx3v(kl) = dx3f(kl);
  } else {
    for (int k=kll; k<=kuu; ++k) {
      Real phi_m = x3f(k);
      Real phi_p = x3f(k+1);
      x3v(k) = 0.5 * (phi_m + phi_p);  // at least 2nd-order accurate
    }
    for (int k=kll; k<=kuu-1; ++k) {
      dx3v(k) = x3v(k+1) - x3v(k);
    }
  }

  // Allocate arrays for geometric quantities
  metric_cell_kji_.NewAthenaArray(2, NMETRIC, nc3, nc2, nc1);
  if (!coarse_flag) {
    coord_vol_kji_.NewAthenaArray(nc3, nc2, nc1);
    coord_area1_kji_.NewAthenaArray(nc3, nc2, nc1+1);
    coord_area2_kji_.NewAthenaArray(nc3, nc2+1, nc1);
    coord_area3_kji_.NewAthenaArray(nc3+1, nc2, nc1);
    coord_len1_kji_.NewAthenaArray(nc3+1, nc2+1, nc1);
    coord_len2_kji_.NewAthenaArray(nc3+1, nc2, nc1+1);
    coord_len3_kji_.NewAthenaArray(nc3, nc2+1, nc1+1);
    coord_width1_kji_.NewAthenaArray(nc3, nc2, nc1);
    coord_width2_kji_.NewAthenaArray(nc3, nc2, nc1);
    coord_width3_kji_.NewAthenaArray(nc3, nc2, nc1);
    coord_src_kji_.NewAthenaArray(3, NMETRIC, nc3, nc2, nc1);
    metric_face1_kji_.NewAthenaArray(2, NMETRIC, nc3, nc2, nc1+1);
    metric_face2_kji_.NewAthenaArray(2, NMETRIC, nc3, nc2+1, nc1);
    metric_face3_kji_.NewAthenaArray(2, NMETRIC, nc3+1, nc2, nc1);
    trans_face1_kji_.NewAthenaArray(2, NMETRIC, nc3, nc2, nc1+1);
    trans_face2_kji_.NewAthenaArray(2, NMETRIC, nc3, nc2+1, nc1);
    trans_face3_kji_.NewAthenaArray(2, NMETRIC, nc3+1, nc2, nc1);
    g_.NewAthenaArray(NMETRIC, nc1+1);
    gi_.NewAthenaArray(NMETRIC, nc1+1);
  }

  // Allocate scratch arrays
  AthenaArray<Real> g, g_inv, dg_dx1, dg_dx2, dg_dx3, transformation;
  g.NewAthenaArray(NMETRIC);
  g_inv.NewAthenaArray(NMETRIC);
  dg_dx1.NewAthenaArray(NMETRIC);
  dg_dx2.NewAthenaArray(NMETRIC);
  dg_dx3.NewAthenaArray(NMETRIC);
  if (!coarse_flag) {
    transformation.NewAthenaArray(2, NTRIANGULAR);
  }

  // Calculate cell-centered geometric quantities
  for (int k=kll; k<=kuu; ++k) {
    for (int j=jll; j<=juu; ++j) {
      for (int i=ill; i<=iuu; ++i) {
        // Get position and separations
        Real x1 = x1v(i);
        Real x2 = x2v(j);
        Real x3 = x3v(k);
        Real dx1 = dx1f(i);
        Real dx2 = dx2f(j);
        Real dx3 = dx3f(k);

        // Calculate metric coefficients
        Metric(x1, x2, x3, pin, g, g_inv, dg_dx1, dg_dx2, dg_dx3);

        // Calculate volumes
        if (!coarse_flag) {
          Real det = Determinant(g);
          coord_vol_kji_(k,j,i) = std::sqrt(-det) * dx1 * dx2 * dx3;
        }

        // Calculate widths
        if (!coarse_flag) {
          coord_width1_kji_(k,j,i) = std::sqrt(g(I11)) * dx1;
          coord_width2_kji_(k,j,i) = std::sqrt(g(I22)) * dx2;
          coord_width3_kji_(k,j,i) = std::sqrt(g(I33)) * dx3;
        }

        // Store metric derivatives
        if (!coarse_flag) {
          for (int m = 0; m < NMETRIC; ++m) {
            coord_src_kji_(0,m,k,j,i) = dg_dx1(m);
            coord_src_kji_(1,m,k,j,i) = dg_dx2(m);
            coord_src_kji_(2,m,k,j,i) = dg_dx3(m);
          }
        }

        // Set metric coefficients
        for (int n = 0; n < NMETRIC; ++n) {
          metric_cell_kji_(0,n,k,j,i) = g(n);
          metric_cell_kji_(1,n,k,j,i) = g_inv(n);
        }
      }
    }
  }

  // Calculate x1-face-centered geometric quantities
  if (!coarse_flag) {
    for (int k=kll; k<=kuu; ++k) {
      for (int j=jll; j<=juu; ++j) {
        for (int i=ill; i<=iuu+1; ++i) {
          // Get position and separations
          Real x1 = x1f(i);
          Real x2 = x2v(j);
          Real x3 = x3v(k);
          Real dx2 = dx2f(j);
          Real dx3 = dx3f(k);

          // Calculate metric coefficients
          Metric(x1, x2, x3, pin, g, g_inv, dg_dx1, dg_dx2, dg_dx3);

          // Calculate areas
          Real det = Determinant(g);
          coord_area1_kji_(k,j,i) = std::sqrt(-det) * dx2 * dx3;

          // Set metric coefficients
          for (int n = 0; n < NMETRIC; ++n) {
            metric_face1_kji_(0,n,k,j,i) = g(n);
            metric_face1_kji_(1,n,k,j,i) = g_inv(n);
          }

          // Calculate frame transformation
          CalculateTransformation(g, g_inv, 1, transformation);
          for (int n = 0; n < 2; ++n) {
            for (int m = 0; m < NTRIANGULAR; ++m) {
              trans_face1_kji_(n,m,k,j,i) = transformation(n,m);
            }
          }
        }
      }
    }
  }

  // Calculate x2-face-centered geometric quantities
  if (!coarse_flag) {
    for (int k=kll; k<=kuu; ++k) {
      for (int j=jll; j<=juu+1; ++j) {
        for (int i=ill; i<=iuu; ++i) {
          // Get position and separations
          Real x1 = x1v(i);
          Real x2 = x2f(j);
          Real x3 = x3v(k);
          Real dx1 = dx1f(i);
          Real dx3 = dx3f(k);

          // Calculate metric coefficients
          Metric(x1, x2, x3, pin, g, g_inv, dg_dx1, dg_dx2, dg_dx3);

          // Calculate areas
          Real det = Determinant(g);
          coord_area2_kji_(k,j,i) = std::sqrt(-det) * dx1 * dx3;

          // Set metric coefficients
          for (int n = 0; n < NMETRIC; ++n) {
            metric_face2_kji_(0,n,k,j,i) = g(n);
            metric_face2_kji_(1,n,k,j,i) = g_inv(n);
          }

          // Calculate frame transformation
          CalculateTransformation(g, g_inv, 2, transformation);
          for (int n = 0; n < 2; ++n) {
            for (int m = 0; m < NTRIANGULAR; ++m) {
              trans_face2_kji_(n,m,k,j,i) = transformation(n,m);
            }
          }
        }
      }
    }
  }

  // Calculate x3-face-centered geometric quantities
  if (!coarse_flag) {
    for (int k=kll; k<=kuu+1; ++k) {
      for (int j=jll; j<=juu; ++j) {
        for (int i=ill; i<=iuu; ++i) {
          // Get position and separations
          Real x1 = x1v(i);
          Real x2 = x2v(j);
          Real x3 = x3f(k);
          Real dx1 = dx1f(i);
          Real dx2 = dx2f(j);

          // Calculate metric coefficients
          Metric(x1, x2, x3, pin, g, g_inv, dg_dx1, dg_dx2, dg_dx3);

          // Calculate areas
          Real det = Determinant(g);
          coord_area3_kji_(k,j,i) = std::sqrt(-det) * dx1 * dx2;

          // Set metric coefficients
          for (int n = 0; n < NMETRIC; ++n) {
            metric_face3_kji_(0,n,k,j,i) = g(n);
            metric_face3_kji_(1,n,k,j,i) = g_inv(n);
          }

          // Calculate frame transformation
          CalculateTransformation(g, g_inv, 3, transformation);
          for (int n = 0; n < 2; ++n) {
            for (int m = 0; m < NTRIANGULAR; ++m) {
              trans_face3_kji_(n,m,k,j,i) = transformation(n,m);
            }
          }
        }
      }
    }
  }

  // Calculate x1-edge-centered geometric quantities
  if (!coarse_flag) {
    for (int k=kll; k<=kuu+1; ++k) {
      for (int j=jll; j<=juu+1; ++j) {
        for (int i=ill; i<=iuu; ++i) {
          // Get position and separation
          Real x1 = x1v(i);
          Real x2 = x2f(j);
          Real x3 = x3f(k);
          Real dx1 = dx1f(i);

          // Calculate metric coefficients
          Metric(x1, x2, x3, pin, g, g_inv, dg_dx1, dg_dx2, dg_dx3);

          // Calculate lengths
          Real det = Determinant(g);
          coord_len1_kji_(k,j,i) = std::sqrt(-det) * dx1;
        }
      }
    }
  }

  // Calculate x2-edge-centered geometric quantities
  if (!coarse_flag) {
    for (int k=kll; k<=kuu+1; ++k) {
      for (int j=jll; j<=juu; ++j) {
        for (int i=ill; i<=iuu+1; ++i) {
          // Get position and separation
          Real x1 = x1f(i);
          Real x2 = x2v(j);
          Real x3 = x3f(k);
          Real dx2 = dx2f(j);

          // Calculate metric coefficients
          Metric(x1, x2, x3, pin, g, g_inv, dg_dx1, dg_dx2, dg_dx3);

          // Calculate lengths
          Real det = Determinant(g);
          coord_len2_kji_(k,j,i) = std::sqrt(-det) * dx2;
        }
      }
    }
  }

  // Calculate x3-edge-centered geometric quantities
  if (!coarse_flag) {
    for (int k=kll; k<=kuu; ++k) {
      for (int j=jll; j<=juu+1; ++j) {
        for (int i=ill; i<=iuu+1; ++i) {
          // Get position and separation
          Real x1 = x1f(i);
          Real x2 = x2f(j);
          Real x3 = x3v(k);
          Real dx3 = dx3f(k);

          // Calculate metric coefficients
          Metric(x1, x2, x3, pin, g, g_inv, dg_dx1, dg_dx2, dg_dx3);

          // Calculate lengths
          Real det = Determinant(g);
          coord_len3_kji_(k,j,i) = std::sqrt(-det) * dx3;
        }
      }
    }
  }
}


//----------------------------------------------------------------------------------------
// EdgeXLength functions: compute physical length at cell edge-X as vector
// Edge1(i,j,k) located at (i,j-1/2,k-1/2), i.e. (x1v(i), x2f(j), x3f(k))
// Edge2(i,j,k) located at (i-1/2,j,k-1/2), i.e. (x1f(i), x2v(j), x3f(k))
// Edge3(i,j,k) located at (i-1/2,j-1/2,k), i.e. (x1f(i), x2f(j), x3v(k))

void GRUser::Edge1Length(const int k, const int j, const int il, const int iu,
                         AthenaArray<Real> &lengths) {
  // \Delta L \approx \sqrt{-g} \Delta x^1
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    lengths(i) = coord_len1_kji_(k,j,i);
  }
  return;
}

void GRUser::Edge2Length(const int k, const int j, const int il, const int iu,
                         AthenaArray<Real> &lengths) {
  // \Delta L \approx \sqrt{-g} \Delta x^2
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    lengths(i) = coord_len2_kji_(k,j,i);
  }
  return;
}

void GRUser::Edge3Length(const int k, const int j, const int il, const int iu,
                         AthenaArray<Real> &lengths) {
  // \Delta L \approx \sqrt{-g} \Delta x^3
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    lengths(i) = coord_len3_kji_(k,j,i);
  }
  return;
}

//----------------------------------------------------------------------------------------
// GetEdgeXLength functions: return length of edge-X at (i,j,k)

Real GRUser::GetEdge1Length(const int k, const int j, const int i) {
  // \Delta L \approx \sqrt{-g} \Delta x^1
  return coord_len1_kji_(k,j,i);
}

Real GRUser::GetEdge2Length(const int k, const int j, const int i) {
  // \Delta L \approx \sqrt{-g} \Delta x^2
  return coord_len2_kji_(k,j,i);
}

Real GRUser::GetEdge3Length(const int k, const int j, const int i) {
  // \Delta L \approx \sqrt{-g} \Delta x^3
  return coord_len3_kji_(k,j,i);
}

//----------------------------------------------------------------------------------------
// CenterWidthX functions: return physical width in X-dir at (i,j,k) cell-center

void GRUser::CenterWidth1(const int k, const int j, const int il, const int iu,
                          AthenaArray<Real> &dx1) {
  // \Delta W \approx \sqrt{g_{11}} \Delta x^1
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    dx1(i) = coord_width1_kji_(k,j,i);
  }
  return;
}

void GRUser::CenterWidth2(const int k, const int j, const int il, const int iu,
                          AthenaArray<Real> &dx2) {
  // \Delta W \approx \sqrt{g_{22}} \Delta x^2
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    dx2(i) = coord_width2_kji_(k,j,i);
  }
  return;
}

void GRUser::CenterWidth3(const int k, const int j, const int il, const int iu,
                          AthenaArray<Real> &dx3) {
  // \Delta W \approx \sqrt{g_{33}} \Delta x^3
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    dx3(i) = coord_width3_kji_(k,j,i);
  }
  return;
}

//----------------------------------------------------------------------------------------
// FaceXArea functions: compute area of face with normal in X-dir as vector
// Inputs:
//   k,j: x3- and x2-indices
//   il,iu: x1-index bounds
// Outputs:
//   areas: 1D array of interface areas orthogonal to X-face

void GRUser::Face1Area(const int k, const int j, const int il, const int iu,
                       AthenaArray<Real> &areas) {
  // \Delta A \approx \sqrt{-g} \Delta x^2 \Delta x^3
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    areas(i) = coord_area1_kji_(k,j,i);
  }
  return;
}

void GRUser::Face2Area(const int k, const int j, const int il, const int iu,
                       AthenaArray<Real> &areas) {
  // \Delta A \approx \sqrt{-g} \Delta x^1 \Delta x^3
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    areas(i) = coord_area2_kji_(k,j,i);
  }
  return;
}

void GRUser::Face3Area(const int k, const int j, const int il, const int iu,
                       AthenaArray<Real> &areas) {
  // \Delta A \approx \sqrt{-g} \Delta x^1 \Delta x^2
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    areas(i) = coord_area3_kji_(k,j,i);
  }
  return;
}

//----------------------------------------------------------------------------------------
// GetFaceXArea functions: return area of face with normal in X-dir at (i,j,k)
// Inputs:
//   k,j,i: x3-, x2-, and x1-indices
// return:
//   interface area orthogonal to X-face

Real GRUser::GetFace1Area(const int k, const int j, const int i) {
  // \Delta A \approx \sqrt{-g} \Delta x^2 \Delta x^3
  return coord_area1_kji_(k,j,i);
}

Real GRUser::GetFace2Area(const int k, const int j, const int i) {
  // \Delta A \approx \sqrt{-g} \Delta x^1 \Delta x^3
  return coord_area2_kji_(k,j,i);
}

Real GRUser::GetFace3Area(const int k, const int j, const int i) {
  // \Delta A \approx \sqrt{-g} \Delta x^1 \Delta x^2
  return coord_area3_kji_(k,j,i);
}

//----------------------------------------------------------------------------------------
// Cell Volume function: compute volume of cell as vector
// Inputs:
//   k,j: x3- and x2-indices
//   il,iu: x1-index bounds
// Outputs:
//   volumes: 1D array of cell volumes

void GRUser::CellVolume(const int k, const int j, const int il, const int iu,
                        AthenaArray<Real> &volumes) {
  // \Delta V \approx \sqrt{-g} \Delta x^1 \Delta x^2 \Delta x^3
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    volumes(i) = coord_vol_kji_(k,j,i);
  }
  return;
}

//----------------------------------------------------------------------------------------
// GetCellVolume: returns cell volume at (i,j,k)
// Inputs:
//   k,j,i: phi-, theta-, and r-indices
// Outputs:
//   returned value: cell volume

Real GRUser::GetCellVolume(const int k, const int j, const int i) {
  // \Delta V \approx \sqrt{-g} \Delta x^1 \Delta x^2 \Delta x^3
  return coord_vol_kji_(k,j,i);
}

//----------------------------------------------------------------------------------------
// Function for computing cell-centered and face-centered metric coefficients
// Inputs:
//   k,j: x3- and x2-indices
//   il,iu: x1-index bounds
// Outputs:
//   g: array of metric components in 1D
//   g_inv: array of inverse metric components in 1D

void GRUser::CellMetric(const int k, const int j, const int il, const int iu,
                        AthenaArray<Real> &g, AthenaArray<Real> &g_inv) {
  for (int n = 0; n < NMETRIC; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      g(n,i) = metric_cell_kji_(0,n,k,j,i);
      g_inv(n,i) = metric_cell_kji_(1,n,k,j,i);
    }
  }
  return;
}

void GRUser::Face1Metric(const int k, const int j, const int il, const int iu,
                         AthenaArray<Real> &g, AthenaArray<Real> &g_inv) {
  for (int n = 0; n < NMETRIC; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      g(n,i) = metric_face1_kji_(0,n,k,j,i);
      g_inv(n,i) = metric_face1_kji_(1,n,k,j,i);
    }
  }
  return;
}

void GRUser::Face2Metric(const int k, const int j, const int il, const int iu,
                         AthenaArray<Real> &g, AthenaArray<Real> &g_inv) {
  for (int n = 0; n < NMETRIC; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      g(n,i) = metric_face2_kji_(0,n,k,j,i);
      g_inv(n,i) = metric_face2_kji_(1,n,k,j,i);
    }
  }
  return;
}

void GRUser::Face3Metric(const int k, const int j, const int il, const int iu,
                         AthenaArray<Real> &g, AthenaArray<Real> &g_inv) {
  for (int n = 0; n < NMETRIC; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      g(n,i) = metric_face3_kji_(0,n,k,j,i);
      g_inv(n,i) = metric_face3_kji_(1,n,k,j,i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
// Function for raising covariant components of a vector
// Inputs:
//   a_0,a_1,a_2,a_3: covariant components of vector
//   k,j,i: indices of cell in which transformation is desired
// Outputs:
//   pa0,pa1,pa2,pa3: pointers to contravariant 4-vector components

void GRUser::RaiseVectorCell(Real a_0, Real a_1, Real a_2, Real a_3, int k, int j, int i,
                             Real *pa0, Real *pa1, Real *pa2, Real *pa3) {
  // Extract metric coefficients
  const Real &g00 = metric_cell_kji_(1,I00,k,j,i);
  const Real &g01 = metric_cell_kji_(1,I01,k,j,i);
  const Real &g02 = metric_cell_kji_(1,I02,k,j,i);
  const Real &g03 = metric_cell_kji_(1,I03,k,j,i);
  const Real &g10 = metric_cell_kji_(1,I01,k,j,i);
  const Real &g11 = metric_cell_kji_(1,I11,k,j,i);
  const Real &g12 = metric_cell_kji_(1,I12,k,j,i);
  const Real &g13 = metric_cell_kji_(1,I13,k,j,i);
  const Real &g20 = metric_cell_kji_(1,I02,k,j,i);
  const Real &g21 = metric_cell_kji_(1,I12,k,j,i);
  const Real &g22 = metric_cell_kji_(1,I22,k,j,i);
  const Real &g23 = metric_cell_kji_(1,I23,k,j,i);
  const Real &g30 = metric_cell_kji_(1,I03,k,j,i);
  const Real &g31 = metric_cell_kji_(1,I13,k,j,i);
  const Real &g32 = metric_cell_kji_(1,I23,k,j,i);
  const Real &g33 = metric_cell_kji_(1,I33,k,j,i);

  // Set raised components
  *pa0 = g00*a_0 + g01*a_1 + g02*a_2 + g03*a_3;
  *pa1 = g10*a_0 + g11*a_1 + g12*a_2 + g13*a_3;
  *pa2 = g20*a_0 + g21*a_1 + g22*a_2 + g23*a_3;
  *pa3 = g30*a_0 + g31*a_1 + g32*a_2 + g33*a_3;
  return;
}

//----------------------------------------------------------------------------------------
// Function for lowering contravariant components of a vector
// Inputs:
//   a0,a1,a2,a3: contravariant components of vector
//   k,j,i: indices of cell in which transformation is desired
// Outputs:
//   pa_0,pa_1,pa_2,pa_3: pointers to covariant 4-vector components

void GRUser::LowerVectorCell(Real a0, Real a1, Real a2, Real a3, int k, int j, int i,
                             Real *pa_0, Real *pa_1, Real *pa_2, Real *pa_3) {
  // Extract metric coefficients
  const Real &g_00 = metric_cell_kji_(0,I00,k,j,i);
  const Real &g_01 = metric_cell_kji_(0,I01,k,j,i);
  const Real &g_02 = metric_cell_kji_(0,I02,k,j,i);
  const Real &g_03 = metric_cell_kji_(0,I03,k,j,i);
  const Real &g_10 = metric_cell_kji_(0,I01,k,j,i);
  const Real &g_11 = metric_cell_kji_(0,I11,k,j,i);
  const Real &g_12 = metric_cell_kji_(0,I12,k,j,i);
  const Real &g_13 = metric_cell_kji_(0,I13,k,j,i);
  const Real &g_20 = metric_cell_kji_(0,I02,k,j,i);
  const Real &g_21 = metric_cell_kji_(0,I12,k,j,i);
  const Real &g_22 = metric_cell_kji_(0,I22,k,j,i);
  const Real &g_23 = metric_cell_kji_(0,I23,k,j,i);
  const Real &g_30 = metric_cell_kji_(0,I03,k,j,i);
  const Real &g_31 = metric_cell_kji_(0,I13,k,j,i);
  const Real &g_32 = metric_cell_kji_(0,I23,k,j,i);
  const Real &g_33 = metric_cell_kji_(0,I33,k,j,i);

  // Set lowered components
  *pa_0 = g_00*a0 + g_01*a1 + g_02*a2 + g_03*a3;
  *pa_1 = g_10*a0 + g_11*a1 + g_12*a2 + g_13*a3;
  *pa_2 = g_20*a0 + g_21*a1 + g_22*a2 + g_23*a3;
  *pa_3 = g_30*a0 + g_31*a1 + g_32*a2 + g_33*a3;
  return;
}

namespace {
//----------------------------------------------------------------------------------------
// Functions for calculating determinant
// Inputs:
//   g: array of covariant metric coefficients
//   a11,a12,a13,a21,a22,a23,a31,a32,a33: elements of matrix
//   a11,a12,a21,a22: elements of matrix
// Outputs:
//   returned value: determinant

Real Determinant(const AthenaArray<Real> &g) {
  const Real &a11 = g(I00);
  const Real &a12 = g(I01);
  const Real &a13 = g(I02);
  const Real &a14 = g(I03);
  const Real &a21 = g(I01);
  const Real &a22 = g(I11);
  const Real &a23 = g(I12);
  const Real &a24 = g(I13);
  const Real &a31 = g(I02);
  const Real &a32 = g(I12);
  const Real &a33 = g(I22);
  const Real &a34 = g(I23);
  const Real &a41 = g(I03);
  const Real &a42 = g(I13);
  const Real &a43 = g(I23);
  const Real &a44 = g(I33);
  Real det = a11 * Determinant(a22, a23, a24, a32, a33, a34, a42, a43, a44)
             - a12 * Determinant(a21, a23, a24, a31, a33, a34, a41, a43, a44)
             + a13 * Determinant(a21, a22, a24, a31, a32, a34, a41, a42, a44)
             - a14 * Determinant(a21, a22, a23, a31, a32, a33, a41, a42, a43);
  return det;
}

Real Determinant(Real a11, Real a12, Real a13, Real a21, Real a22, Real a23,
                 Real a31, Real a32, Real a33) {
  Real det = a11 * Determinant(a22, a23, a32, a33)
             - a12 * Determinant(a21, a23, a31, a33)
             + a13 * Determinant(a21, a22, a31, a32);
  return det;
}

Real Determinant(Real a11, Real a12, Real a21, Real a22) {
  return a11 * a22 - a12 * a21;
}

//----------------------------------------------------------------------------------------
// Function for calculating frame transformation coefficients
// Inputs:
//   g,g_inv: arrays of covariant and contravariant metric coefficients
//   face: 1, 2, or 3 depending on which face is considered
// Outputs:
//   transformation: array of transformation coefficients

void CalculateTransformation(
    const AthenaArray<Real> &g,
    const AthenaArray<Real> &g_inv, int face, AthenaArray<Real> &transformation) {
  // Prepare indices
  int index[4][4];
  index[0][0] = I00;
  index[0][1] = I01; index[0][2] = I02; index[0][3] = I03;
  index[1][1] = I11; index[1][2] = I12; index[1][3] = I13;
  index[2][1] = I12; index[2][2] = I22; index[2][3] = I23;
  index[3][1] = I13; index[3][2] = I23; index[3][3] = I33;

  // Shift indices according to face
  int i0 = 0;
  int i1 = face;
  int i2 = 1 + face%3;
  int i3 = 1 + (face+1)%3;

  // Extract metric coefficients
  const Real &g_22 = g(index[i2][i2]);
  const Real &g_23 = g(index[i2][i3]);
  const Real &g_33 = g(index[i3][i3]);
  const Real &g00 = g_inv(index[i0][i0]);
  const Real &g01 = g_inv(index[i0][i1]);
  const Real &g02 = g_inv(index[i0][i2]);
  const Real &g03 = g_inv(index[i0][i3]);
  const Real &g11 = g_inv(index[i1][i1]);
  const Real &g12 = g_inv(index[i1][i2]);
  const Real &g13 = g_inv(index[i1][i3]);

  // Calculate intermediate quantities
  Real aa = -1.0 / std::sqrt(-g00);
  Real bb = 1.0 / std::sqrt(g00 * (g00 * g11 - SQR(g01)));
  Real cc = 1.0 / std::sqrt(g_33);
  Real dd = 1.0 / std::sqrt(g_33 * (g_22 * g_33 - SQR(g_23)));
  Real ee = g01 * g12 - g11 * g02;
  Real ff = g01 * g02 - g00 * g12;
  Real gg = g01 * g13 - g11 * g03;
  Real hh = g01 * g03 - g00 * g13;
  Real ii = SQR(bb)/cc * g00 * (gg + ee * g_23/g_33);
  Real jj = SQR(bb)/cc * g00 * (hh + ff * g_23/g_33);

  // Set local-to-global transformation coefficients
  transformation(0,T00) = aa * g00;
  transformation(0,T10) = aa * g01;
  transformation(0,T20) = aa * g02;
  transformation(0,T30) = aa * g03;
  transformation(0,T11) = bb * (g01 * g01 - g00 * g11);
  transformation(0,T21) = bb * (g01 * g02 - g00 * g12);
  transformation(0,T31) = bb * (g01 * g03 - g00 * g13);
  transformation(0,T22) = dd * g_33;
  transformation(0,T32) = -dd * g_23;
  transformation(0,T33) = cc;

  // Set global-to-local transformation coefficients
  transformation(1,T00) = -aa;
  transformation(1,T10) = bb * g01;
  transformation(1,T11) = -bb * g00;
  transformation(1,T20) = SQR(bb)*ee/dd * g00/g_33;
  transformation(1,T21) = SQR(bb)*ff/dd * g00/g_33;
  transformation(1,T22) = 1.0 / (dd * g_33);
  transformation(1,T30) = ii;
  transformation(1,T31) = jj;
  transformation(1,T32) = 1.0/cc * g_23/g_33;
  transformation(1,T33) = 1.0/cc;
  return;
}
} // namespace
