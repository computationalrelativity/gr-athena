#ifndef UTILS_GRID_THETA_PHI_HPP
#define UTILS_GRID_THETA_PHI_HPP
//========================================================================================
// GR-Athena++
//========================================================================================
//! \file grid_theta_phi.hpp
//  \brief Shared theta-phi spherical grid with pre-built interpolator pools.
//
//  Eliminates duplicated grid setup, Gauss-Legendre quadrature,
//  point-ownership flagging, and per-point heap-allocated interpolators across
//  AHF, WaveExtractRWZ, and Ejecta.
//
//  Usage:
//    GridThetaPhi<InterpCC, InterpVC> grid;
//    grid.Initialize(ntheta, nphi, "midpoint");   // or "gausslegendre"
//    // fill grid.x_cart(d, i, j) with Cartesian coords of sphere points
//    grid.Prepare(pmesh, use_cc, use_vc);
//    // use grid.interp_pool_cc[grid.mask_interp_idx(i,j)].eval(...)
//    grid.TearDown();
//
//  Template parameters:
//    InterpCC  -  interpolator type for cell-centered pool  (e.g. hydro grid)
//    InterpVC  -  interpolator type for vertex-centered pool (e.g. Z4c on VC)
//                 Defaults to InterpCC when only one pool type is needed.

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "../athena.hpp"
#include "../athena_aliases.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../defs.hpp"
#include "../mesh/mesh.hpp"

using namespace gra::aliases;

// ============================================================================
// Free function: compute the determinant of the induced 2-metric on a
// surface embedded in 3D, at a single point.
//
//   rr      - surface radius r(th,ph)
//   rr_dth  - dr/dth
//   rr_dph  - dr/dph
//   sinth, costh, sinph, cosph - trig values at this point
//   g_xx, g_xy, g_xz, g_yy, g_yz, g_zz - 3-metric components
//
// Returns det(h) = h_thth h_phph - h_thph^2, clamped >= 0.
// ============================================================================
inline Real SurfaceElement2D(const Real rr,
                             const Real rr_dth,
                             const Real rr_dph,
                             const Real sinth,
                             const Real costh,
                             const Real sinph,
                             const Real cosph,
                             const Real g_xx,
                             const Real g_xy,
                             const Real g_xz,
                             const Real g_yy,
                             const Real g_yz,
                             const Real g_zz)
{
  // Tangent vector d(x,y,z)/dth
  const Real dXdth_0 = (rr_dth * sinth + rr * costh) * cosph;
  const Real dXdth_1 = (rr_dth * sinth + rr * costh) * sinph;
  const Real dXdth_2 = rr_dth * costh - rr * sinth;

  // Tangent vector d(x,y,z)/dph
  const Real dXdph_0 = (rr_dph * cosph - rr * sinph) * sinth;
  const Real dXdph_1 = (rr_dph * sinph + rr * cosph) * sinth;
  const Real dXdph_2 = rr_dph * costh;

  // Induced 2-metric h_{AB} = g_{ab} e^a_A e^b_B
  // Expand the double contraction with g symmetric:
  //   h = sum_{a,b} g(a,b) * e_A(a) * e_B(b)
  // Using packed symmetric metric: g_xx, g_xy, g_xz, g_yy, g_yz, g_zz
  const Real g[3][3]  = { { g_xx, g_xy, g_xz },
                          { g_xy, g_yy, g_yz },
                          { g_xz, g_yz, g_zz } };
  const Real eA[2][3] = { { dXdth_0, dXdth_1, dXdth_2 },
                          { dXdph_0, dXdph_1, dXdph_2 } };

  Real h[2][2] = { { 0.0, 0.0 }, { 0.0, 0.0 } };
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
    {
      const Real gab = g[a][b];
      h[0][0] += eA[0][a] * eA[0][b] * gab;
      h[0][1] += eA[0][a] * eA[1][b] * gab;
      h[1][1] += eA[1][a] * eA[1][b] * gab;
    }

  Real deth = h[0][0] * h[1][1] - h[0][1] * h[0][1];
  if (deth < 0.0)
    deth = 0.0;

  return deth;
}

// ============================================================================

template <typename InterpCC, typename InterpVC = InterpCC>
class GridThetaPhi
{
  public:
  // ---- Grid dimensions -----------------------------------------------------
  int ntheta = 0;
  int nphi   = 0;

  // ---- Grid node arrays (1D) -----------------------------------------------
  AthenaArray<Real> th_grid;  // size ntheta
  AthenaArray<Real> ph_grid;  // size nphi

  // ---- Precomputed trigonometric values (1D)
  // ---------------------------------
  AthenaArray<Real> sin_theta, cos_theta;  // size ntheta
  AthenaArray<Real> sin_phi, cos_phi;      // size nphi

  // ---- Quadrature weights (2D: ntheta x nphi) ------------------------------
  AthenaArray<Real> weights;

  // ---- Cartesian coordinates of sphere points (3D: 3 x ntheta x nphi) ------
  //  x_cart(0,i,j) = x,  x_cart(1,i,j) = y,  x_cart(2,i,j) = z
  //  Filled by the consumer before calling Prepare().
  AthenaArray<Real> x_cart;

  // ---- Contravariant Jacobian d(r,th,ph)/d(x,y,z) on the sphere
  // ---------------
  //  con_J(A, a, i, j)     - first derivatives,  A in {r,th,ph}, a in {x,y,z}
  //  con_J2(A, a, b, i, j) - second derivatives,  symmetric in (a,b)
  //  Filled by ComputeConJacobian().
  AT_N_T2 con_J;
  AT_N_VS2 con_J2;

  // ---- Point ownership (2D: ntheta x nphi) ---------------------------------
  AthenaArray<MeshBlock*> mask_mb;   // owning MeshBlock, nullptr if unowned
  AthenaArray<int> mask_interp_idx;  // index into pool(s), -1 if unowned

  // ---- Interpolator pools (sparse: one entry per owned point)
  // ---------------
  std::vector<InterpCC> interp_pool_cc;
  std::vector<InterpVC> interp_pool_vc;

  // ---- State ---------------------------------------------------------------
  bool has_cc   = false;
  bool has_vc   = false;
  bool prepared = false;

  // ==========================================================================
  // Initialize grid arrays and quadrature weights.
  // Supported methods: "midpoint", "gausslegendre".
  // ==========================================================================
  void Initialize(const int nth, const int nph, const std::string& quadrature)
  {
    ntheta = nth;
    nphi   = nph;

    // Validate nphi is even
    if (nphi % 2 != 0)
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in GridThetaPhi::Initialize" << std::endl
          << "nphi must be even, got " << nphi << std::endl;
      ATHENA_ERROR(msg);
    }

    // Allocate grid arrays
    th_grid.NewAthenaArray(ntheta);
    ph_grid.NewAthenaArray(nphi);
    weights.NewAthenaArray(ntheta, nphi);

    // Phi grid is always uniform midpoint
    const Real dph = 2.0 * PI / nphi;
    for (int j = 0; j < nphi; ++j)
      ph_grid(j) = dph * (0.5 + j);

    if (quadrature == "midpoint")
    {
      const Real dth = PI / ntheta;
      for (int i = 0; i < ntheta; ++i)
        th_grid(i) = dth * (0.5 + i);

      for (int i = 0; i < ntheta; ++i)
      {
        const Real dcosth = std::sin(th_grid(i)) * dth;
        for (int j = 0; j < nphi; ++j)
        {
          weights(i, j) = dcosth * dph;
        }
      }
    }
    else if (quadrature == "gausslegendre")
    {
      if (ntheta != nphi / 2)
      {
        std::stringstream msg;
        msg << "### FATAL ERROR in GridThetaPhi::Initialize" << std::endl
            << "gausslegendre requires ntheta == nphi/2, got ntheta=" << ntheta
            << " nphi=" << nphi << std::endl;
        ATHENA_ERROR(msg);
      }

      std::vector<Real> gl_nodes(ntheta);
      std::vector<Real> gl_weights(ntheta);

      GLQuadNodesWeights(
        -1.0, 1.0, gl_nodes.data(), gl_weights.data(), ntheta);

      for (int i = 0; i < ntheta; ++i)
        th_grid(i) = std::acos(gl_nodes[i]);

      for (int i = 0; i < ntheta; ++i)
      {
        for (int j = 0; j < nphi; ++j)
        {
          weights(i, j) = gl_weights[i] * dph;
        }
      }
    }
    else
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in GridThetaPhi::Initialize" << std::endl
          << "unknown quadrature method: " << quadrature << std::endl;
      ATHENA_ERROR(msg);
    }

    // Precompute trigonometric values
    sin_theta.NewAthenaArray(ntheta);
    cos_theta.NewAthenaArray(ntheta);
    for (int i = 0; i < ntheta; ++i)
    {
      sin_theta(i) = std::sin(th_grid(i));
      cos_theta(i) = std::cos(th_grid(i));
    }
    sin_phi.NewAthenaArray(nphi);
    cos_phi.NewAthenaArray(nphi);
    for (int j = 0; j < nphi; ++j)
    {
      sin_phi(j) = std::sin(ph_grid(j));
      cos_phi(j) = std::cos(ph_grid(j));
    }

    // Allocate Cartesian coordinate array (filled by consumer before Prepare)
    x_cart.NewAthenaArray(3, ntheta, nphi);

    // Allocate Jacobian arrays
    con_J.NewAthenaTensor(ntheta, nphi);
    con_J2.NewAthenaTensor(ntheta, nphi);

    // Allocate ownership arrays
    mask_mb.NewAthenaArray(ntheta, nphi);
    mask_interp_idx.NewAthenaArray(ntheta, nphi);
    mask_interp_idx.Fill(-1);
  }

  // ==========================================================================
  // Build interpolator pools from the current x_cart positions.
  //
  //  use_cc  -  build cell-centered pool  (origin from x1v, size from ncells)
  //  use_vc  -  build vertex-centered pool (origin from x1f, size from nverts)
  //
  //  The consumer must fill x_cart before calling this.
  //  Uses PointContainedExclusive (exclusive upper bounds, prevents
  //  double-counting at MeshBlock boundaries).
  // ==========================================================================
  void Prepare(const Mesh* pmesh, const bool use_cc, const bool use_vc)
  {
    if (prepared)
      return;

    has_cc = use_cc;
    has_vc = use_vc;

    // Clear ownership state
    mask_mb.Fill(nullptr);
    mask_interp_idx.Fill(-1);

    // Reserve pool capacity (upper bound: every grid point)
    const int N_pts_max = ntheta * nphi;
    if (has_cc)
      interp_pool_cc.reserve(N_pts_max);
    if (has_vc)
      interp_pool_vc.reserve(N_pts_max);

    const std::vector<MeshBlock*>& pmb_array = pmesh->GetMeshBlocksCached();

    for (MeshBlock* pmb : pmb_array)
    {
      // Cell-centered grid info
      const Real origin_cc[3] = { pmb->pcoord->x1v(0),
                                  pmb->pcoord->x2v(0),
                                  pmb->pcoord->x3v(0) };
      const Real delta_cc[3]  = { pmb->pcoord->dx1v(0),
                                  pmb->pcoord->dx2v(0),
                                  pmb->pcoord->dx3v(0) };
      const int size_cc[3]    = { pmb->ncells1, pmb->ncells2, pmb->ncells3 };

      // Vertex-centered grid info
      const Real origin_vc[3] = { pmb->pcoord->x1f(0),
                                  pmb->pcoord->x2f(0),
                                  pmb->pcoord->x3f(0) };
      const Real delta_vc[3]  = { pmb->pcoord->dx1f(0),
                                  pmb->pcoord->dx2f(0),
                                  pmb->pcoord->dx3f(0) };
      const int size_vc[3]    = { pmb->nverts1, pmb->nverts2, pmb->nverts3 };

      for (int i = 0; i < ntheta; ++i)
      {
        for (int j = 0; j < nphi; ++j)
        {
          const Real x1 = x_cart(0, i, j);
          const Real x2 = x_cart(1, i, j);
          const Real x3 = x_cart(2, i, j);

          if (!pmb->PointContainedExclusive(x1, x2, x3))
            continue;

          mask_mb(i, j) = pmb;

          const Real coord[3] = { x1, x2, x3 };
          const int idx       = static_cast<int>(has_cc ? interp_pool_cc.size()
                                                  : interp_pool_vc.size());
          mask_interp_idx(i, j) = idx;

          if (has_cc)
            interp_pool_cc.emplace_back(origin_cc, delta_cc, size_cc, coord);

          if (has_vc)
            interp_pool_vc.emplace_back(origin_vc, delta_vc, size_vc, coord);
        }
      }
    }

    prepared = true;
  }

  // ==========================================================================
  // Tear down interpolator pools and clear ownership masks.
  // ==========================================================================
  void TearDown()
  {
    if (!prepared)
      return;

    mask_mb.Fill(nullptr);
    mask_interp_idx.Fill(-1);

    interp_pool_cc.clear();
    interp_pool_vc.clear();

    prepared = false;
  }

  // ==========================================================================
  // Reinitialize after AMR: tear down stale pools, rebuild with fresh
  // MeshBlock pointers.  Remembers which pools were active.
  // ==========================================================================
  void Reinitialize(const Mesh* pmesh)
  {
    const bool prev_cc = has_cc;
    const bool prev_vc = has_vc;
    TearDown();
    Prepare(pmesh, prev_cc, prev_vc);
  }

  // ==========================================================================
  // Accessors
  // ==========================================================================
  bool IsOwned(const int i, const int j) const
  {
    return mask_mb(i, j) != nullptr;
  }

  int tpindex(const int i, const int j) const
  {
    return i * nphi + j;
  }

  Real dtheta() const
  {
    return PI / ntheta;
  }

  Real dphi() const
  {
    return 2.0 * PI / nphi;
  }

  // ==========================================================================
  // Fill x_cart from a center point and per-point radii.
  //
  //  center  -  Cartesian coordinates of the sphere center (3 elements)
  //  rr      -  surface radius at each grid point (ntheta x nphi)
  //  bitant  -  if true, reflect z < 0 to z > 0 (bitant symmetry wrt z=0)
  //
  //  Uses precomputed sin_theta, cos_theta, sin_phi, cos_phi.
  // ==========================================================================
  void FillCartesianCoords(const Real center[3],
                           const AthenaArray<Real>& rr,
                           const bool bitant)
  {
    for (int i = 0; i < ntheta; ++i)
    {
      const Real sth = sin_theta(i);
      const Real cth = cos_theta(i);
      for (int j = 0; j < nphi; ++j)
      {
        const Real sph  = sin_phi(j);
        const Real cph  = cos_phi(j);
        x_cart(0, i, j) = center[0] + rr(i, j) * sth * cph;
        x_cart(1, i, j) = center[1] + rr(i, j) * sth * sph;
        Real z          = center[2] + rr(i, j) * cth;
        if (bitant)
          z = std::abs(z);
        x_cart(2, i, j) = z;
      }
    }
  }

  // ==========================================================================
  // Fill x_cart from a center point and a constant radius.
  //
  //  center  -  Cartesian coordinates of the sphere center (3 elements)
  //  radius  -  constant surface radius
  //  bitant  -  if true, reflect z < 0 to z > 0 (bitant symmetry wrt z=0)
  // ==========================================================================
  void FillCartesianCoords(const Real center[3],
                           const Real radius,
                           const bool bitant)
  {
    for (int i = 0; i < ntheta; ++i)
    {
      const Real sth = sin_theta(i);
      const Real cth = cos_theta(i);
      for (int j = 0; j < nphi; ++j)
      {
        const Real sph  = sin_phi(j);
        const Real cph  = cos_phi(j);
        x_cart(0, i, j) = center[0] + radius * sth * cph;
        x_cart(1, i, j) = center[1] + radius * sth * sph;
        Real z          = center[2] + radius * cth;
        if (bitant)
          z = std::abs(z);
        x_cart(2, i, j) = z;
      }
    }
  }

  // ==========================================================================
  // Compute the contravariant Jacobian d(r,th,ph)/d(x,y,z) and its second
  // derivatives at every grid point.  Uses the surface radii rr(i,j) and
  // precomputed trig values.
  //
  //  Spherical index ordering:  A = 0 -> r,  1 -> th,  2 -> ph
  //  Cartesian index ordering:  a = 0 -> x,  1 -> y,  2 -> z
  //
  //  Returns false if any surface point has rp < min_radius.
  // ==========================================================================
  bool ComputeConJacobian(const AthenaArray<Real>& rr, const Real min_radius)
  {
    for (int i = 0; i < ntheta; ++i)
    {
      const Real sth = sin_theta(i);
      const Real cth = cos_theta(i);

      for (int j = 0; j < nphi; ++j)
      {
        const Real sph = sin_phi(j);
        const Real cph = cos_phi(j);

        // Cartesian coordinates relative to center
        const Real xp = rr(i, j) * sth * cph;
        const Real yp = rr(i, j) * sth * sph;
        const Real zp = rr(i, j) * cth;

        const Real rp   = std::sqrt(xp * xp + yp * yp + zp * zp);
        const Real rhop = std::sqrt(xp * xp + yp * yp);

        if (rp < min_radius)
          return false;

        const Real _divrp    = 1.0 / rp;
        const Real _divrp3   = SQR(_divrp) * _divrp;
        const Real _divrp4   = SQR(_divrp) * SQR(_divrp);
        const Real _divrhop  = 1.0 / rhop;
        const Real _divrhop2 = SQR(_divrhop);
        const Real _divrhop3 = _divrhop2 * _divrhop;
        const Real _divrhop4 = SQR(_divrhop2);
        const Real xp2       = SQR(xp);
        const Real yp2       = SQR(yp);
        const Real zp2       = SQR(zp);

        // First derivatives: con_J(A, a, i, j) = dA/da
        // A=0: dr/dx^a
        con_J(0, 0, i, j) = xp * _divrp;
        con_J(0, 1, i, j) = yp * _divrp;
        con_J(0, 2, i, j) = zp * _divrp;

        // A=1: dtheta/dx^a
        con_J(1, 0, i, j) = zp * xp * (SQR(_divrp) * _divrhop);
        con_J(1, 1, i, j) = zp * yp * (SQR(_divrp) * _divrhop);
        con_J(1, 2, i, j) = -rhop * SQR(_divrp);

        // A=2: dphi/dx^a
        con_J(2, 0, i, j) = -yp * _divrhop2;
        con_J(2, 1, i, j) = xp * _divrhop2;
        con_J(2, 2, i, j) = 0.0;

        // Second derivatives: con_J2(A, a, b, i, j) = d^2 A/da db (sym in a,b)
        // A=0: d^2 r/dx^a dx^b
        con_J2(0, 0, 0, i, j) = _divrp - xp2 * _divrp3;
        con_J2(0, 0, 1, i, j) = -xp * yp * _divrp3;
        con_J2(0, 0, 2, i, j) = -xp * zp * _divrp3;
        con_J2(0, 1, 1, i, j) = _divrp - yp2 * _divrp3;
        con_J2(0, 1, 2, i, j) = -yp * zp * _divrp3;
        con_J2(0, 2, 2, i, j) = _divrp - zp2 * _divrp3;

        // A=1: d^2 theta/dx^a dx^b
        con_J2(1, 0, 0, i, j) =
          zp * (-2.0 * SQR(xp2) - xp2 * yp2 + SQR(yp2) + zp2 * yp2) *
          (_divrp4 * _divrhop3);
        con_J2(1, 0, 1, i, j) = -xp * yp * zp * (3.0 * xp2 + 3.0 * yp2 + zp2) *
                                (_divrp4 * _divrhop3);
        con_J2(1, 0, 2, i, j) = xp * (xp2 + yp2 - zp2) * (_divrp4 * _divrhop);
        con_J2(1, 1, 1, i, j) =
          zp * (-2.0 * SQR(yp2) - yp2 * xp2 + SQR(xp2) + zp2 * xp2) *
          (_divrp4 * _divrhop3);
        con_J2(1, 1, 2, i, j) = yp * (xp2 + yp2 - zp2) * (_divrp4 * _divrhop);
        con_J2(1, 2, 2, i, j) = 2.0 * zp * rhop * _divrp4;

        // A=2: d^2 phi/dx^a dx^b
        con_J2(2, 0, 0, i, j) = 2.0 * yp * xp * _divrhop4;
        con_J2(2, 0, 1, i, j) = (yp2 - xp2) * _divrhop4;
        con_J2(2, 0, 2, i, j) = 0.0;
        con_J2(2, 1, 1, i, j) = -2.0 * yp * xp * _divrhop4;
        con_J2(2, 1, 2, i, j) = 0.0;
        con_J2(2, 2, 2, i, j) = 0.0;
      }
    }
    return true;
  }

  private:
  // ==========================================================================
  // Gauss-Legendre quadrature nodes and weights on [a, b].
  // ==========================================================================
  static void GLQuadNodesWeights(const Real a,
                                 const Real b,
                                 Real* x,
                                 Real* w,
                                 const int n)
  {
    constexpr Real tol = 1e-14;

    const int m   = (n + 1) / 2;
    const Real xm = 0.5 * (b + a);
    const Real xl = 0.5 * (b - a);

    for (int i = 1; i <= m; ++i)
    {
      Real z = std::cos(PI * (i - 0.25) / (n + 0.5));
      Real z1, pp;
      do
      {
        Real p1 = 1.0;
        Real p2 = 0.0;
        for (int j = 1; j <= n; ++j)
        {
          Real p3 = p2;
          p2      = p1;
          p1      = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
        }
        pp = n * (z * p1 - p2) / (z * z - 1.0);
        z1 = z;
        z  = z1 - p1 / pp;
      } while (std::fabs(z - z1) > tol);

      x[i - 1] = xm - xl * z;
      x[n - i] = xm + xl * z;
      w[i - 1] = 2.0 * xl / ((1.0 - z * z) * pp * pp);
      w[n - i] = w[i - 1];
    }
  }
};

#endif  // UTILS_GRID_THETA_PHI_HPP
