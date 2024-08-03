// C headers

// C++ headers
#include <cmath>  // pow(), trig functions
#include <iomanip>
#include <iostream>   // endl
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "coordinates.hpp"

// ----------------------------------------------------------------------------
SphericalPolarUniform::SphericalPolarUniform(MeshBlock *pmb,
                                             ParameterInput *pin,
                                             bool flag)
  : Coordinates(pmb, pin, flag)
{
  RegionSize& block_size = pmy_block->block_size;

  // x1-direction -------------------------------------------------------------
  for (int i=il-ng; i<=iu+ng; ++i)
  {
    x1v(i) = 0.5*(x1f(i+1) + x1f(i));
  }
  for (int i=il-ng; i<=iu+ng-1; ++i)
  {
    dx1v(i) = x1v(i+1) - x1v(i);
  }

  // x2-direction -------------------------------------------------------------
  if (pmb->block_size.nx2 == 1)
  {
    x2v(jl) = 0.5*(x2f(jl+1) + x2f(jl));
    dx2v(jl) = dx2f(jl);
  }
  else
  {
    for (int j=jl-ng; j<=ju+ng-1; ++j)
    {
      dx2v(j) = x2f(j+1) - x2f(j);
    }

    for (int j=jl-ng; j<=ju+ng; ++j)
    {
      x2v(j) = 0.5*(x2f(j+1) + x2f(j));
    }
  }

  // x3-direction -------------------------------------------------------------
  if (pmb->block_size.nx3 == 1)
  {
    x3v(kl) = 0.5*(x3f(kl+1) + x3f(kl));
    dx3v(kl) = dx3f(kl);
  }
  else
  {
    for (int k=kl-ng; k<=ku+ng-1; ++k)
    {
      dx3v(k) = x3v(k+1) - x3v(k);
    }

    for (int k=kl-ng; k<=ku+ng; ++k)
    {
      x3v(k) = 0.5*(x3f(k+1) + x3f(k));
    }
  }

  // Set up finite differencing -----------------------------------------------
  fd_is_defined = true;
  fd_cc = new FiniteDifference::Uniform(
    nc1, nc2, nc3,
    dx1v(0), dx2v(0), dx3v(0)
  );

  fd_cx = new FiniteDifference::Uniform(
    cx_nc1, cx_nc2, cx_nc3,
    dx1v(0), dx2v(0), dx3v(0)
  );

  fd_vc = new FiniteDifference::Uniform(
    nv1, nv2, nv3,
    dx1f(0), dx2f(0), dx3f(0)
  );
  // --------------------------------------------------------------------------
}


//
// :D
//