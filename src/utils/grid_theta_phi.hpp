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
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../defs.hpp"
#include "../mesh/mesh.hpp"

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

  // ---- Quadrature weights (2D: ntheta x nphi) ------------------------------
  AthenaArray<Real> weights;

  // ---- Cartesian coordinates of sphere points (3D: 3 x ntheta x nphi) ------
  //  x_cart(0,i,j) = x,  x_cart(1,i,j) = y,  x_cart(2,i,j) = z
  //  Filled by the consumer before calling Prepare().
  AthenaArray<Real> x_cart;

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
    if ((nphi + 1) % 2 == 0)
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

    // Allocate Cartesian coordinate array (filled by consumer before Prepare)
    x_cart.NewAthenaArray(3, ntheta, nphi);

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
