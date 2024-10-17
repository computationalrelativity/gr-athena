#ifndef NUMTOOLS_ROOT_HPP
#define NUMTOOLS_ROOT_HPP

//! \file numtools_root.hpp
//  \author Jacob Fields
//
//  \brief Declares some functions for root-finding.

#include <cmath>
#include <iostream>
#include <limits>
#include "../../athena.hpp"

namespace NumTools {

class Root {
  private:
  public:
    /// Maximum number of iterations
    unsigned int iterations;

    /// Only used for benchmarking, not thread-safe.
    int last_count;

    Root() : iterations(30) {}

    Root(Root const&) = delete;
    void operator=(Root const&) = delete;

    // FalsePosition {{{

    //! \brief Find the root of a functor f using false position.
    //
    // Find the root of a generic functor taking at least one argument. The first
    // argument is assumed to be the quantity of interest. All other arguments are
    // assumed to be constant parameters for the function. The root-finding method
    // is the Illinois variant of false position.
    //
    // \param[in]  f  The functor to find a root for. Its root function must take at
    //                least one argument.
    // \param[in,out]  lb  The lower bound for the root.
    // \param[in,out]  ub  The upper bound for the root.
    // \param[out]  x  The location of the root.
    // \param[in]  args  Additional arguments required by f.

    template<class Functor, class ... Types>
    inline bool FalsePosition(Functor&& f, Real &lb, Real &ub, Real& x,  Real tol, Types ... args) {
      int side = 0;
      Real ftest;
      unsigned int count = 0;
      last_count = 0;
      // Get our initial bracket.
      Real flb = f(lb, args...);
      Real fub = f(ub, args...);
      Real xold;
      x = lb;
      // If one of the bounds is already within tolerance of the root, we can skip all of this.
      if (std::fabs(flb) <= tol) {
        x = lb;
        return true;
      }
      else if (std::fabs(fub) <= tol) {
        x = ub;
        return true;
      }
      if (flb*fub > 0) {
        return false;
      }
      do {
        xold = x;
        // Calculate the new root position.
        x = (fub*lb - flb*ub)/(fub - flb);
        count++;

        if (std::fabs((x-xold)/x) <= tol) {
          return true;
        }

        // Calculate f at the prospective root.
        ftest = f(x,args...);

        // Check the sign of f. If f is on the same side as the lower bound, then we adjust
        // the lower bound. Similarly, if f is on the same side as the upper bound, we 
        // adjust the upper bound. If ftest falls on the same side twice, we weight one of
        // the sides to force the new root to fall on the other side. This allows us to
        // whittle down both sides at once and get better average convergence.
        if (ftest*flb >= 0) {
          flb = ftest;
          lb = x;
          if (side == 1) {
            fub /= 2.0;
          }
          side = 1;
        }
        else {
          fub = ftest;
          ub = x;
          if (side == -1) {
            flb /= 2.0;
          }
          side = -1;
        }
      }
      while (count < iterations);
      last_count = count;

      // Return success if we're below the tolerance, otherwise report failure.
      return fabs((x-xold)/x) <= tol;
    }
    
    // }}}

    // Chandrupatla {{{

    //! \brief Find the root of a functor f using Chandrupatla's method
    //
    // Find the root of a generic functor taking at least one argument. The first
    // argument is assumed to be the quantity of interest. All other arguments are
    // assumed to be constant parameters for the function. The root-finding method
    // is Chandrupatla's method, a simpler alternative to Brent's method with
    // comparable performance.
    //
    // \param[in]  f  The functor to find a root for. Its root function must take at
    //                least one argument.
    // \param[in,out]  lb  The lower bound for the root.
    // \param[in,out]  ub  The upper bound for the root.
    // \param[out]  x  The location of the root.
    // \param[in]  args  Additional arguments required by f.

    template<class Functor, class ... Types>
    inline bool Chandrupatla(Functor&& f, Real &lb, Real &ub, Real& x, Real tol, Types ... args) {
      unsigned int count = 0;
      last_count = 0;
      // Get our initial bracket.
      Real flb = f(lb, args...);
      Real fub = f(ub, args...);
      x = lb;
      // If one of the bounds is already within tolerance of the root, we can skip all of this.
      if (std::fabs(flb) <= tol) {
        x = lb;
        return true;
      }
      else if (std::fabs(fub) <= tol) {
        x = ub;
        return true;
      }
      // Make sure the bracket is valid
      if (flb*fub > 0) {
        return false;
      }
      Real t = 0.5;
      Real x1, x2, x3, f1, f2, f3, ftest;
      Real phi1, xi1;
      x1 = ub;
      x2 = lb;
      f1 = fub;
      f2 = flb;
      do {
        // Estimate the new root position
        x = x1 + t*(x2 - x1);
        count++;
        // Calculate f at the prospective root
        ftest = f(x, args...);
        if (std::fabs((x-x1)/x) <= tol) {
          break;
        }
        // Check the sign of ftest to determine the new bounds
        if (ftest*f1 >= 0) {
          x3 = x1;
          x1 = x;
          f3 = f1;
          f1 = ftest;
        }
        else {
          x3 = x2;
          x2 = x1;
          x1 = x;
          f3 = f2;
          f2 = f1;
          f1 = ftest;
        }
        // Check if we're in the region of validity for quadratic interpolation.
        phi1 = (f1 - f2)/(f3 - f2);
        xi1 = (x1 - x2)/(x3 - x2);
        if (1.0 - std::sqrt(1.0 - xi1) < phi1 && phi1 < std::sqrt(xi1)) {
          // Perform quadratic interpolation
          t = f1/(f3 - f2)*(f3/(f1 - f2) + (x3 - x1)/(x2 - x1)*f2/(f3 - f1));
        }
        else {
          // Perform bisection instead
          t = 0.5;
        }
      }
      while (count < iterations);
      last_count = count;

      // Return success if we're below the tolerance, otherwise report failure.
      return fabs((x-x1)/x) <= tol;
    }

    // }}}

    // NewtonSafe {{{
    /*! \brief Find the root of a function f using a safe Newton solve.
     *
     * A safe Newton solve performs a Newton-Raphson solve, but it also brackets the
     * root using bisection to ensure that the result converges.
     *
     * \param[in]     f     The functor to find a root for. Its root function must take
     *                      at least one argument.
     * \param[in,out] lb    The lower bound for the root of f.
     * \param[in,out] ub    The upper bound for the root of f.
     * \param[out]    x     The root of f.
     * \param[in]     args  Additional arguments required by f.
     */
    template<class Functor, class ... Types>
    inline bool NewtonSafe(Functor&& f, Real &lb, Real &ub, Real& x, Real tol, Types ... args) {
      Real fx;
      Real dfx;
      Real xold;
      unsigned int count = 0;
      //last_count = 0;
      // We first need to ensure that the bracket is valid.
      Real fub, flb;
      f(flb, dfx, lb, args...);
      f(fub, dfx, ub, args...);
      if (flb*fub > 0) {
        return 0;
      }
      // If one of the roots is already within tolerance, then
      // we don't need to do the solve.
      if (std::fabs(flb) <= tol) {
        x = lb;
        return true;
      }
      else if (std::fabs(fub) <= tol) {
        x = ub;
        return true;
      }
      // Since we already had to evaluate the function at the bounds,
      // we can predict our starting position using false position.
      x = (fub*lb - flb*ub)/(fub - flb);
      do {
        xold = x;
        // Calculate f and df at point x.
        f(fx, dfx, x, args...);
        x = x - fx/dfx;
        // Check that the root is bounded properly.
        if (x > ub || x < lb) {
          // Revert to bisection if the root is not converging.
          x = 0.5*(ub + lb);
          //f(fx, dfx, x, args...);
        }
        // Correct the bounds.
        if (fx*flb > 0) {
          flb = fx;
          lb = xold;
        }
        else if (fx*fub > 0) {
          fub = fx;
          ub = xold;
        }
        count++;
      }
      while (std::fabs((xold-x)/x) > tol && count < iterations);
      //last_count = count;

      // Return success if we're below the tolerance, otherwise report failure.
      return std::fabs((x-xold)/x) <= tol;
    }
    // }}}

  // --------------------------------------------------------------------------
  // Ported from BOOST
  template <class T>
  inline int sign (const T& a)
  {
    return (a == 0) ? 0 : (a < 0) ? -1 : 1;
  }

  template <class T>
  inline constexpr T min_value() noexcept
  {
    return std::numeric_limits<T>::min();
  }

  template <class T>
  inline constexpr T max_value() noexcept
  {
    return std::numeric_limits<T>::max();
  }

  template <class T>
  inline constexpr T epsilon() noexcept
  {
    return std::numeric_limits<T>::epsilon();
  }

  template <class T>
  inline T safe_div(const T& num, const T&denom, const T& r)
  {
    if(std::abs(denom) < 1)
    {
      if(std::abs(denom * max_value<T>()) <= std::abs(num))
        return r;
    }
    return num / denom;
  }

  template <class F, class T, class ... Types>
  void bracket(F f, T& a, T& b, T c, T& fa, T& fb, T& d, T& fd, Types ... args)
  {
    //
    // Given a point c inside the existing enclosing interval
    // [a, b] sets a = c if f(c) == 0, otherwise finds the new
    // enclosing interval: either [a, c] or [c, b] and sets
    // d and fd to the point that has just been removed from
    // the interval.  In other words d is the third best guess
    // to the root.
    //
    T tol = epsilon<T>() * 2;
    //
    // If the interval [a,b] is very small, or if c is too close
    // to one end of the interval then we need to adjust the
    // location of c accordingly:
    //
    if((b - a) < 2 * tol * a)
    {
        c = a + (b - a) / 2;
    }
    else if(c <= a + std::abs(a) * tol)
    {
        c = a + std::abs(a) * tol;
    }
    else if(c >= b - std::abs(b) * tol)
    {
        c = b - std::abs(b) * tol;
    }
    //
    // OK, lets invoke f(c):
    //
    T fc = f(c, args...);
    //
    // if we have a zero then we have an exact solution to the root:
    //
    if(fc == 0)
    {
        a = c;
        fa = 0;
        d = 0;
        fd = 0;
        return;
    }
    //
    // Non-zero fc, update the interval:
    //
    if(sign(fa) * sign(fc) < 0)
    {
        d = b;
        fd = fb;
        b = c;
        fb = fc;
    }
    else
    {
        d = a;
        fd = fa;
        a = c;
        fa= fc;
    }
  }

  template <class T>
  inline T secant_interpolate(const T& a, const T& b, const T& fa, const T& fb)
  {
    //
    // Performs standard secant interpolation of [a,b] given
    // function evaluations f(a) and f(b).  Performs a bisection
    // if secant interpolation would leave us very close to either
    // a or b.  Rationale: we only call this function when at least
    // one other form of interpolation has already failed, so we know
    // that the function is unlikely to be smooth with a root very
    // close to a or b.
    //

    T tol = epsilon<T>() * 5;
    T c = a - (fa / (fb - fa)) * (b - a);
    if((c <= a + std::abs(a) * tol) || (c >= b - std::abs(b) * tol))
        return (a + b) / 2;
    return c;
  }

  template <class T>
  T quadratic_interpolate(const T& a, const T& b, T const& d,
                          const T& fa, const T& fb, T const& fd,
                          unsigned count)
  {
    //
    // Performs quadratic interpolation to determine the next point,
    // takes count Newton steps to find the location of the
    // quadratic polynomial.
    //
    // Point d must lie outside of the interval [a,b], it is the third
    // best approximation to the root, after a and b.
    //
    // Note: this does not guarantee to find a root
    // inside [a, b], so we fall back to a secant step should
    // the result be out of range.
    //
    // Start by obtaining the coefficients of the quadratic polynomial:
    //
    T B = safe_div(T(fb - fa), T(b - a), max_value<T>());
    T A = safe_div(T(fd - fb), T(d - b), max_value<T>());
    A = safe_div(T(A - B), T(d - a), T(0));

    if(A == 0)
    {
        // failure to determine coefficients, try a secant step:
        return secant_interpolate(a, b, fa, fb);
    }
    //
    // Determine the starting point of the Newton steps:
    //
    T c;
    if(sign(A) * sign(fa) > 0)
    {
        c = a;
    }
    else
    {
        c = b;
    }
    //
    // Take the Newton steps:
    //
    for(unsigned i = 1; i <= count; ++i)
    {
        //c -= safe_div(B * c, (B + A * (2 * c - a - b)), 1 + c - a);
        c -= safe_div(T(fa+(B+A*(c-b))*(c-a)), T(B + A * (2 * c - a - b)), T(1 + c - a));
    }
    if((c <= a) || (c >= b))
    {
        // Oops, failure, try a secant step:
        c = secant_interpolate(a, b, fa, fb);
    }
    return c;
  }

  template <class T>
  T cubic_interpolate(const T& a, const T& b, const T& d,
                      const T& e, const T& fa, const T& fb,
                      const T& fd, const T& fe)
  {
    //
    // Uses inverse cubic interpolation of f(x) at points
    // [a,b,d,e] to obtain an approximate root of f(x).
    // Points d and e lie outside the interval [a,b]
    // and are the third and forth best approximations
    // to the root that we have found so far.
    //
    // Note: this does not guarantee to find a root
    // inside [a, b], so we fall back to quadratic
    // interpolation in case of an erroneous result.
    //
    T q11 = (d - e) * fd / (fe - fd);
    T q21 = (b - d) * fb / (fd - fb);
    T q31 = (a - b) * fa / (fb - fa);
    T d21 = (b - d) * fd / (fd - fb);
    T d31 = (a - b) * fb / (fb - fa);

    T q22 = (d21 - q11) * fb / (fe - fb);
    T q32 = (d31 - q21) * fa / (fd - fa);
    T d32 = (d31 - q21) * fd / (fd - fa);
    T q33 = (d32 - q22) * fa / (fe - fa);
    T c = q31 + q32 + q33 + a;

    if((c <= a) || (c >= b))
    {
        // Out of bounds step, fall back to quadratic interpolation:
        c = quadratic_interpolate(a, b, d, fa, fb, fd, 3);
    }

    return c;
  }

  template <class F, class T, class ... Types>
  T toms748_solve(F && f, const T& ax, const T& bx, const T & tol, Types ... args)
  {
    //
    // Main entry point and logic for Toms Algorithm 748
    // root finder.
    //
    unsigned int max_iter = iterations;

    auto tolf = [&](const T &a, const T &b)
    {
      return std::abs(a-b) <= tol * std::min(a,b);
    };

    unsigned int count = max_iter;
    T a, b, fa, fb, c, u, fu, a0, b0, d, fd, e, fe;
    static const T mu = 0.5f;

    // initialise a, b and fa, fb:
    a = ax;
    b = bx;
    if(a >= b)
    {
      std::cout << "Domain error a>=b" << std::endl;
      assert(false);
    }

    // eval rather than pass
    // fa = fax;
    // fb = fbx;
    fa = f(a, args...);
    fb = f(b, args...);

    if(tolf(a, b) || (fa == 0) || (fb == 0))
    {
        max_iter = 0;
        if(fa == 0)
          b = a;
        else if(fb == 0)
          a = b;

        // return boost::math::make_pair(a, b);
        return 0.5 * (b - a);
    }

    if(sign(fa) * sign(fb) > 0)
    {
      std::cout << "Parameters a and b do not bracket the root" << std::endl;
      assert(false);
    }

    // dummy value for fd, e and fe:
    fe = e = fd = 1e5F;

    if(fa != 0)
    {
        //
        // On the first step we take a secant step:
        //
        c = secant_interpolate(a, b, fa, fb);
        bracket(f, a, b, c, fa, fb, d, fd, args...);
        --count;

        if(count && (fa != 0) && !tolf(a, b))
        {
          //
          // On the second step we take a quadratic interpolation:
          //
          c = quadratic_interpolate(a, b, d, fa, fb, fd, 2);
          e = d;
          fe = fd;
          bracket(f, a, b, c, fa, fb, d, fd, args...);
          --count;
        }
    }

    while(count && (fa != 0) && !tolf(a, b))
    {
        // save our brackets:
        a0 = a;
        b0 = b;
        //
        // Starting with the third step taken
        // we can use either quadratic or cubic interpolation.
        // Cubic interpolation requires that all four function values
        // fa, fb, fd, and fe are distinct, should that not be the case
        // then variable prof will get set to true, and we'll end up
        // taking a quadratic step instead.
        //
        T min_diff = min_value<T>() * 32;
        bool prof = (fabs(fa - fb) < min_diff) || (fabs(fa - fd) < min_diff) ||
                    (fabs(fa - fe) < min_diff) || (fabs(fb - fd) < min_diff) ||
                    (fabs(fb - fe) < min_diff) || (fabs(fd - fe) < min_diff);
        if (prof) {
          c = quadratic_interpolate(a, b, d, fa, fb, fd, 2);
        } else {
          c = cubic_interpolate(a, b, d, e, fa, fb, fd, fe);
        }
        //
        // re-bracket, and check for termination:
        //
        e = d;
        fe = fd;
        bracket(f, a, b, c, fa, fb, d, fd, args...);
        if((0 == --count) || (fa == 0) || tolf(a, b))
          break;

        //
        // Now another interpolated step:
        //
        prof = (fabs(fa - fb) < min_diff) || (fabs(fa - fd) < min_diff) ||
               (fabs(fa - fe) < min_diff) || (fabs(fb - fd) < min_diff) ||
               (fabs(fb - fe) < min_diff) || (fabs(fd - fe) < min_diff);
        if (prof) {
          c = quadratic_interpolate(a, b, d, fa, fb, fd, 3);
        } else {
          c = cubic_interpolate(a, b, d, e, fa, fb, fd, fe);
        }
        //
        // Bracket again, and check termination condition, update e:
        //
        bracket(f, a, b, c, fa, fb, d, fd, args...);
        if((0 == --count) || (fa == 0) || tolf(a, b))
          break;

        //
        // Now we take a double-length secant step:
        //
        if(fabs(fa) < fabs(fb))
        {
          u = a;
          fu = fa;
        }
        else
        {
          u = b;
          fu = fb;
        }
        c = u - 2 * (fu / (fb - fa)) * (b - a);
        if(fabs(c - u) > (b - a) / 2)
        {
          c = a + (b - a) / 2;
        }
        //
        // Bracket again, and check termination condition:
        //
        e = d;
        fe = fd;
        bracket(f, a, b, c, fa, fb, d, fd, args...);

        if((0 == --count) || (fa == 0) || tolf(a, b))
          break;
        //
        // And finally... check to see if an additional bisection step is
        // to be taken, we do this if we're not converging fast enough:
        //
        if((b - a) < mu * (b0 - a0))
          continue;
        //
        // bracket again on a bisection:
        //
        e = d;
        fe = fd;
        bracket(f, a, b, T(a + (b - a) / 2), fa, fb, d, fd, args...);
        --count;
    } // while loop

    max_iter -= count;
    if(fa == 0)
    {
        b = a;
    }
    else if(fb == 0)
    {
        a = b;
    }

    // return boost::math::make_pair(a, b);
    const Real root = 0.5 * (b - a);
    f(root, args...);
    return root;
  }

};

} // namespace

#endif
