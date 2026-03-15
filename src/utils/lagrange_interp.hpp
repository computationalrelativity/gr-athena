#ifndef LAGRANGE_INTERP_HPP_
#define LAGRANGE_INTERP_HPP_

//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file lagrange.hpp
//  \brief Lagrange interpolation

#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>

#include "../athena.hpp"

// If this is uncommented always use symmetric operators
#ifndef LOCALINTERP_SYMMETRIC
#define LOCALINTERP_SYMMETRIC
#endif

// Use fuzzy comparison for whether a coord is "half-way" between known knodes
// Comparison is done based on relative error and below factor of machine eps
#ifndef LOCALINTERP_MID_FUZZY
#define LOCALINTERP_MID_FUZZY
#define LOCALINTERP_MID_FUZZY_FAC 10
#endif


template<int order, bool compute_diff = false>
class LagrangeInterp1D {
  public:
    LagrangeInterp1D(
        //! [in] Grid origin
        Real const origin,
        //! [in] Grid spacing
        Real const delta,
        //! [in] Number of grid points
        int siz,
        //! [in] Interpolation point
        Real const coord):
      m_origin(origin),
      m_delta(delta),
      m_delta_inv(1.0/m_delta),
      m_siz(siz),
      m_coord(coord),
      m_mid_flag(false),
      m_npoint(order + 1),
      m_out_of_bounds(false) {
      // Check if we are in the middle between two grid points
#ifdef LOCALINTERP_SYMMETRIC
      if(0 == order % 2) {
        int idx = std::lrint(std::floor((m_coord - m_origin)*m_delta_inv));

        // original conditional read:
        // m_coord - (idx*m_delta + m_origin) == 0.5 * m_delta
        Real cond_l = m_coord - (idx*m_delta + m_origin);
        Real cond_r = 0.5 * m_delta;

        bool cond;

#ifdef LOCALINTERP_MID_FUZZY
        Real eps = std::numeric_limits<Real>::epsilon();
        cond = std::abs(1 - cond_l / cond_r) < eps * LOCALINTERP_MID_FUZZY_FAC;
#else
        cond = cond_l == cond_r
#endif

        if(cond) {
          m_mid_flag = true;
          ++m_npoint;
        }
        else {
          m_mid_flag = false;
        }

      }
#endif
      // First point (from the left) of the interpolation stencil
      m_point = std::lrint(std::floor((m_coord - m_origin)*m_delta_inv
          - 0.5*(order - 1)));
#ifdef LOCALINTERP_SYMMETRIC
      if(m_mid_flag) {
          m_point -= 1;
      }
#endif

      // Shift the interpolation stencil if out of the grid
      int shift = m_point;
      if(shift < 0) {
        m_point -= shift;
        m_out_of_bounds = true;
      }
      shift = m_point + order - (m_siz - 1) + m_mid_flag;
      if(shift > 0) {
        m_point -= shift;
        m_out_of_bounds = true;
      }

      Real xp[order+2];
      for(int i = 0; i <= order; ++i) {
        xp[i] = (m_point + i) * m_delta + m_origin;
      }
#ifdef LOCALINTERP_SYMMETRIC
      if(0 == order % 2 && m_mid_flag) {
        xp[order+1] = (m_point + order + 1) * m_delta + m_origin;
      }
#endif
      m_calc_coeff_lr(xp, &m_coeff_lr[0]);
      if constexpr (compute_diff) {
        m_calc_coeff_diff_lr(xp, &m_coeff_lr[0], &m_coeff_diff_lr[0]);
      }
#ifdef LOCALINTERP_SYMMETRIC
      if(0 == order % 2 && m_mid_flag) {
        m_calc_coeff_rl(&xp[1], &m_coeff_rl[0]);
        if constexpr (compute_diff) {
          m_calc_coeff_diff_rl(&xp[1], &m_coeff_rl[0], &m_coeff_diff_rl[0]);
        }
      }
      else {
        std::memcpy(m_coeff_rl, m_coeff_lr, sizeof(m_coeff_lr));
        if constexpr (compute_diff) {
          std::memcpy(m_coeff_diff_rl, m_coeff_diff_lr, sizeof(m_coeff_diff_lr));
        }
      }
#endif
    }

    // Rule of five: delete copy (self-referential access patterns make copies
    // error-prone); default move is safe now that reference members have been
    // replaced with accessor methods.
    LagrangeInterp1D(const LagrangeInterp1D&) = delete;
    LagrangeInterp1D& operator=(const LagrangeInterp1D&) = delete;
    LagrangeInterp1D(LagrangeInterp1D&&) = default;
    LagrangeInterp1D& operator=(LagrangeInterp1D&&) = default;
    ~LagrangeInterp1D() = default;

    //! Evaluates the interpolator
    template<typename T>
    T eval(
        //! [in] must be offset so that vals[0] = vals["point"]
        T const * const vals,
        //! [in] stride used to access vals
        int const stride
        ) const {
      T out_lr = 0;
      for(int i = 0; i <= order; ++i) {
        out_lr += static_cast<T>(m_coeff_lr[i]) * vals[i*stride];
      }
#ifdef LOCALINTERP_SYMMETRIC
      if (m_mid_flag) {
        T out_rl = 0;
        for(int i = order; i >= 0; --i) {
          out_rl += static_cast<T>(m_coeff_rl[i]) * vals[(i + 1)*stride];
        }
        return T(0.5)*(out_lr + out_rl);
      }
#endif
      return out_lr;
    }

    //! Evaluates the derivative interpolator
    template<typename T>
    T eval_diff(
        //! [in] must be offset so that vals[0] = vals["point"]
        T const * const vals,
        //! [in] stride used to access vals
        int const stride
        ) const {
      static_assert(compute_diff,
          "eval_diff() requires compute_diff=true");
      T out_lr = 0;
      for(int i = 0; i <= order; ++i) {
        out_lr += static_cast<T>(m_coeff_diff_lr[i]) * vals[i*stride];
      }
#ifdef LOCALINTERP_SYMMETRIC
      if (m_mid_flag) {
        T out_rl = 0;
        for(int i = order; i >= 0; --i) {
          out_rl += static_cast<T>(m_coeff_diff_rl[i]) * vals[(i + 1)*stride];
        }
        return T(0.5)*(out_lr + out_rl);
      }
#endif
      return out_lr;
    }
  private:
    // Compute the Lagrange interpolation coefficients on a given stencil
    void m_calc_coeff_lr(
        Real const * const xp,
        Real * const coeff
        ) const {
#define TYPECASE(I0, TEST, I1, OP)                                            \
      for(int j = 0; j <= order; ++j) {                                       \
        Real num = 1.0;                                                       \
        Real den = 1.0;                                                       \
        for(int i = I0; i TEST I1; OP i) {                                    \
          if(i == j) {                                                        \
            continue;                                                         \
          }                                                                   \
          num = num * (m_coord - xp[i]);                                      \
          den = den * (xp[j] - xp[i]);                                        \
        }                                                                     \
        coeff[j] = num/den;                                                   \
      }
      TYPECASE(0, <=, order, ++)
    }
    void m_calc_coeff_rl(
        Real const * const xp,
        Real * const coeff
        ) const {
      TYPECASE(order, >=, 0, --)
#undef TYPECASE
    }

    // Compute the Lagrange derivative coefficients on a given stancil
    void m_calc_coeff_diff_lr(
        Real const * const xp,
        Real const * const coeff,
        Real * const coeff_diff
        ) const {
#define TYPECASE(M0, TEST, M1, OP)                                            \
      for (int i = 0; i <= order; ++i) {                                      \
        Real fac = 0.0;                                                       \
        for (int m = M0; m TEST M1; OP m) {                                   \
          if (i == m) {                                                       \
            continue;                                                         \
          }                                                                   \
          fac += 1.0/(m_coord - xp[m]);                                       \
        }                                                                     \
        coeff_diff[i] = coeff[i]*fac;                                         \
      }
      TYPECASE(0, <=, order, ++)
    }
    void m_calc_coeff_diff_rl(
        Real const * const xp,
        Real const * const coeff,
        Real * const coeff_diff
        ) const {
      TYPECASE(order, >=, 0, --)
#undef TYPECASE
    }
  private:
    Real m_origin;
    Real m_delta;
    Real m_delta_inv;
    int m_siz;

    Real m_coord;

    // If true we have an asymmetric stencil, but the interpolation point is
    // exactly in the middle between two grid points. In this case we need to
    // average the results obtained from the interpolation on two separate
    // stencils.
    bool m_mid_flag;
    // First point (going from left to right) of the stencil
    int m_point;
    // Number of points needed for the interpolation
    int m_npoint;
    // The stencil was shifted to avoid going out of bounds
    bool m_out_of_bounds;

    // Interpolation coefficients for interpolation from left to right
    Real m_coeff_lr[order+1];
#ifdef LOCALINTERP_SYMMETRIC
    // Interpolation coefficients for interpolation from right to left
    Real m_coeff_rl[order+1];
#endif
    // Interpolation coefficients for derivative from left to right
    Real m_coeff_diff_lr[order+1];
#ifdef LOCALINTERP_SYMMETRIC
    // Interpolation coefficients for derivative from right to left
    Real m_coeff_diff_rl[order+1];
#endif
  public:
    //! Index of the first point of the interpolation stencil
    int point() const { return m_point; }
    //! Number of points needed for the interpolation
    int npoint() const { return m_npoint; }
    //! The stencil was shifted to avoid going out of bounds
    bool out_of_bounds() const { return m_out_of_bounds; }
};

template<int ndim, int D>
class NextStencil {
  public:
    enum { value = D - 1 };
};

template<int ndim>
class NextStencil<ndim, 0> {
  public:
    enum { value = 0 };
};

// Multi-dimensional Lagrange interpolation
//
// This class assumes that
// . the grid is uniformly spaced in each direction, different grid spacing in
//   different direction is allowed;
// . the fastest running index in the data is that corrsponding to the first dimension
// . the stride used to access the data along the first dimension is 1
template<int order, int ndim, bool compute_diff = false>
class LagrangeInterpND {
  public:
    LagrangeInterpND(
        //! [in] Grid origin
        Real const origin[ndim],
        //! [in] Grid spacing
        Real const delta[ndim],
        //! [in] Number of grid points
        int const siz[ndim],
        //! [in] Interpolation point
        Real const coord[ndim]):
        m_out_of_bounds(false) {
      for(int d = 0; d < ndim; ++d) {
        m_origin[d] = origin[d];
        m_delta[d] = delta[d];
        m_siz[d] = siz[d];
        m_coord[d] = coord[d];
        mp_interp[d] = new (&m_interp_scratch[d][0])
          LagrangeInterp1D<order, compute_diff>(m_origin[d], m_delta[d],
              m_siz[d], m_coord[d]);
        m_out_of_bounds = m_out_of_bounds || mp_interp[d]->out_of_bounds();
      }
      // Precompute strides for the D==0 base case of m_fill_stencil
      m_stride[0] = 1;
      for(int d = 1; d < ndim; ++d)
        m_stride[d] = m_stride[d-1] * m_siz[d-1];
    }

    // Rule of five -------------------------------------------------------
    // Delete copy - mp_interp[] contains self-pointers into m_interp_scratch
    // that would dangle after a shallow copy.
    LagrangeInterpND(const LagrangeInterpND&) = delete;
    LagrangeInterpND& operator=(const LagrangeInterpND&) = delete;

    // Move constructor: bitwise-copy all storage, then fix up the self-pointers
    // in mp_interp[] to reference this object's own scratch buffers.
    LagrangeInterpND(LagrangeInterpND&& o) noexcept
        : m_out_of_bounds(o.m_out_of_bounds) {
      std::memcpy(m_origin, o.m_origin, sizeof(m_origin));
      std::memcpy(m_delta, o.m_delta, sizeof(m_delta));
      std::memcpy(m_siz, o.m_siz, sizeof(m_siz));
      std::memcpy(m_stride, o.m_stride, sizeof(m_stride));
      std::memcpy(m_coord, o.m_coord, sizeof(m_coord));
      std::memcpy(m_interp_scratch, o.m_interp_scratch, sizeof(m_interp_scratch));
      for (int d = 0; d < ndim; ++d) {
        mp_interp[d] = reinterpret_cast<LagrangeInterp1D<order, compute_diff>*>(
            &m_interp_scratch[d][0]);
      }
    }

    // Move assignment: same logic as move constructor.
    LagrangeInterpND& operator=(LagrangeInterpND&& o) noexcept {
      if (this != &o) {
        m_out_of_bounds = o.m_out_of_bounds;
        std::memcpy(m_origin, o.m_origin, sizeof(m_origin));
        std::memcpy(m_delta, o.m_delta, sizeof(m_delta));
        std::memcpy(m_siz, o.m_siz, sizeof(m_siz));
        std::memcpy(m_stride, o.m_stride, sizeof(m_stride));
        std::memcpy(m_coord, o.m_coord, sizeof(m_coord));
        std::memcpy(m_interp_scratch, o.m_interp_scratch, sizeof(m_interp_scratch));
        for (int d = 0; d < ndim; ++d) {
          mp_interp[d] = reinterpret_cast<LagrangeInterp1D<order, compute_diff>*>(
              &m_interp_scratch[d][0]);
        }
      }
      return *this;
    }

    ~LagrangeInterpND() = default;
    // -------------------------------------------------------------------- 

    // Evaluate interpolation
    // der is the derivative direction (-1 for no derivative)
    template<typename T, int der=-1>
    T eval(
        //! [in] Grid function to interpolate
        T const * const gf
        ) const {
      m_fill_stencil<T, ndim-1, der>(gf);
      if constexpr (ndim - 1 == der) {
        return mp_interp[ndim-1]->eval_diff(m_vals[ndim-1], 1);
      }
      else {
        return mp_interp[ndim-1]->eval(m_vals[ndim-1], 1);
      }
    }
  private:
    // Recursively fill the stencil used for the interpolation
    template<typename T, int D, int der>
    void m_fill_stencil(
        T const * const gf
        ) const {
      static_assert(D >= 0 && D < ndim);
      if constexpr (D == 0) {
        int gidx = mp_interp[0]->point();
        for(int d = 1; d < ndim; ++d) {
          gidx += m_stride[d] * (mp_interp[d]->point() + m_eval_pos[d]);
        }
        std::memcpy(&m_vals[0][0], &gf[gidx], mp_interp[0]->npoint()*sizeof(T));
      }
      else {
        for(m_eval_pos[D] = 0; m_eval_pos[D] < mp_interp[D]->npoint(); ++m_eval_pos[D]) {
          m_fill_stencil<T, NextStencil<ndim, D>::value, der>(gf);
          if constexpr (D - 1 == der) {
            m_vals[D][m_eval_pos[D]] = mp_interp[D-1]->eval_diff(m_vals[D-1], 1);
          }
          else {
            m_vals[D][m_eval_pos[D]] = mp_interp[D-1]->eval(m_vals[D-1], 1);
          }
        }
      }
    }
  private:
    Real m_origin[ndim];
    Real m_delta[ndim];
    int m_siz[ndim];
    int m_stride[ndim];

    Real m_coord[ndim];
    bool m_out_of_bounds;

    // 1D interpolators (constructed via placement new into scratch storage)
    LagrangeInterp1D<order, compute_diff> * mp_interp[ndim];
    // Scratch space used for placement new
    char m_interp_scratch[ndim][sizeof(LagrangeInterp1D<order, compute_diff>)];

    // Scratch buffers for eval() - mutable so eval() can remain const
    mutable Real m_vals[ndim][order+2];
    mutable int m_eval_pos[ndim];
  public:
    //! True if any dimension's stencil was shifted to stay in bounds
    bool out_of_bounds() const { return m_out_of_bounds; }
};


#endif
