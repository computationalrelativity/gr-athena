#ifndef NUMTOOLS_ROOT_H
#define NUMTOOLS_ROOT_H

/*!
 * \file numtoolsroot.h
 * \author Jacob Fields
 *
 * \brief Declares some functions for root-finding.
 */

#include <cmath>

#include "../../athena.hpp"

#ifdef DEBUG_ROOTS
#include <iostream>
#endif

namespace NumTools{
  class Root {
    private:
      Root() : iterations(30), tol(1e-15) {}
    public:
      /// The number of iterations to permit during a root solve.
      unsigned int iterations;
      /// The iterations from the last call to the root solve.
      //unsigned int count;
      /// The tolerance for a root solve.
      Real tol;
      /// The golden ratio, used by the ITP method
      const Real phi = 1.618033988749895;

      static inline Root& get_instance() {
        static Root instance;

        return instance;
      }

      Root(Root const&) = delete;
      void operator=(Root const&) = delete;

      // false_position {{{
      /*!
       * \fn template<class ... Types> bool false_position(Real (*f)(Real, Types...), Real &lb, Real &ub, Real &x, Types ... args)
       *
       * \brief Find the root of a function f using false position.
       *
       * Find the root of a generic function taking at least one argument. The first argument
       * is assumed to be the quantity of interest. All other arguments are assumed to be
       * constant parameters for the function. The root-finding method is the Illinois variant
       * of false position.
       *
       * False position works by evaluating the test function at both bounds, then looking for
       * the x-intercept of the line between these points. It typically has superlinear 
       * convergence. If one bound is drawn in toward the root more quickly than the other, the
       * method can stagnate. The Illinois variant handles these cases by forcing the intersection
       * to occur on the other side of the root if it appears on the same side twice in a row.
       *
       * \param[in]  f    The function to find a root for. It must take at least one argument.
       * \param[in,out]  lb   The lower bound for the root.
       * \param[in,out]  ub   The upper bound for the root.
       * \param[out] x    The location of the root.
       * \param[in]  args Additional arguments required by f.
       */
      template<class ... Types> 
      bool false_position(Real (*f)(Real, Types...), Real &lb, Real &ub, Real& x, Types ... args){
        int side = 0;
        Real ftest;
        unsigned int count = 0;
        // Get our initial bracket.
        Real flb = f(lb, args...);
        Real fub = f(ub, args...);
        Real xold;
        x = lb;
        // If one of the bounds is already within tolerances of the root, we can skip all of this.
        if (std::fabs(flb) <= tol) {
          x = lb;
          return true;
        }
        else if (std::fabs(fub) <= tol) {
          x = ub;
          return true;
        }
        if (flb*fub > 0){
          #ifdef DEBUG_ROOTS
          std::cout << "Failed to bracket the root.\n";
          #endif
          return false;
        }
        do{
          xold = x;
          // Calculate the new root position.
          x = (fub*lb - flb*ub)/(fub - flb);
          count++;
          // Calculate f at the prospective root.
          ftest = f(x,args...);
          if(std::fabs((x-xold)/x) <= tol){
            break;
          }
          // Check the sign of f. If f is on the same side as the lower bound, then we adjust
          // the lower bound. Similarly, if f is on the same side as the upper bound, we 
          // adjust the upper bound. If ftest falls on the same side twice, we weight one of
          // the sides to force the new root to fall on the other side. This allows us to
          // whittle down both sides at once and get better average convergence.
          if(ftest*flb >= 0){
            flb = ftest;
            lb = x;
            if (side == 1){
              fub /= 2.0;
            }
            side = 1;
          }
          else{
            fub = ftest;
            ub = x;
            if (side == -1){
              flb /= 2.0;
            }
            side = -1;
          }
        }
        while(count < iterations);

  #ifdef DEBUG_ROOTS
        printf("Root solve iterations: %d out of %d\n",count,iterations);
        printf("Root solve accuracy: %g\n",ftest);
  #endif

        // Return success if we're below the tolerance, otherwise report failure.
        //return fabs(ftest) <= tol;
        return std::fabs((x-xold)/x) <= tol;
      }
      // }}}
      
      // newton_raphson {{{
      /*!
       * \fn template<class ... Types> bool newton_raphson(void (*f)(Real&, Real&, Real, Types...), Real &x, Types ... args)
       *
       * \brief Find the root of a function f using a Newton-Raphson solver.
       *
       * Find the root of a function f with at least three arguments using a Newton-Raphson solver. It is
       * assumed that f takes three arguments. The first argument must be a reference to store the 
       * function value. The second argument must be a reference to store the derivative. The third
       * argument is the point to evaluate both the function and its derivative at. Any additional
       * arguments are assumed to be constants.
       *
       * The Newton-Raphson method is one of the standard methods for root-finding. It evaluates
       * a function f and its derivative df at a point x, then calculates the tangent line at
       * that point and predicts that the root is located at its x-intercept. For well-behaved
       * problems, it provides quadratic convergence. For poorly behaved roots, it will diverge.
       * For best results, use on functions with one well defined root and monotonic behavior.
       *
       * \param[in]     f    The function to use for the root solve.
       * \param[in,out] x    The initial guess for the root (in) and the location of the root (out).
       * \param[in]     args Any additional arguments required by f.
       */
      template<class ... Types>
      bool newton_raphson(void (*f)(Real&, Real&, Real, Types...), Real &x, Types ... args){
        Real fx;
        Real dfx;
        Real xold;
        unsigned int count = 0;
        do{
          xold = x;
          // Calculate f and df at point x.
          f(fx, dfx, x, args...);
          x = x - fx/dfx;
          count++;
        }
        while (std::fabs((xold-x)/x) > tol && count < iterations);

  #ifdef DEBUG_ROOTS
        printf("Root solve iterations: %d out of %d\n", count, iterations);
        printf("Root solve accuracy: %g\n", fx);
  #endif

        // Return success if we're below the tolerance, otherwise report failure.
        //return fabs(fx) <= tol;
        return std::fabs((xold-x)/x) <= tol;
      }
      // }}}

      // newton_safe {{{
      /*!
       * \fn template<class ... Types> bool newton_safe(void (*f)(Real&, Real&, Real, Types...), Real &lb, Real &ub, Real &x, Types ... args)
       *
       * \brief Find the root of a function f using a safe Newton solve.
       *
       * A safe Newton solve performs a Newton-Raphson solve, but it also brackets the
       * root using bisection to ensure that the result converges.
       *
       * \param[in]     f    The function to use for the root solve.
       * \param[in,out] lb   The lower bound for the root of .
       * \param[in,out] ub   The upper bound for the root of f.
       * \param[out]    x    The root of f.
       * \param[in]     args Additional arguments required by f.
       */
      template<class ... Types>
      bool newton_safe(void (*f)(Real&, Real&, Real, Types...), Real& lb, Real& ub, 
          Real& x, Types ... args) {
        Real fx;
        Real dfx;
        Real xold;
        unsigned int count = 0;
        // We first need to ensure that the bracket is valid.
        Real fub, flb;
        f(flb, dfx, lb, args...);
        f(fub, dfx, ub, args...);
        if (flb*fub > 0) {
          #ifdef DEBUG_ROOTS
          printf("Failed to bracket root.\n");
          #endif
          return 0;
        }
        // If one of the roots is already within tolerances, then
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
            f(fx, dfx, x, args...);
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
   #ifdef DEBUG_ROOTS
        printf("Root solve iterations: %d out of %d\n", count, iterations);
        printf("Root solve accuracy: %g\n", fx);
   #endif

        // Return success if we're below the tolerance, otherwise report failure.
        //return fabs(fx) <= tol;
        return std::fabs((xold-x)/x) <= tol;
      }
      // }}}
  };
};

#endif
