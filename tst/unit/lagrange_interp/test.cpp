#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "lagrange_interp.hpp"

double poly(double x, double y, double c[3][3]) {
  double out = 0.;
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    out += c[i][j] * pow(x, i) * pow(y, j);
  }
  return out;
}

void make_grid(double xmin, double dx, int siz, double * xp) {
  for (int i = 0; i < siz; ++i) {
    xp[i] = xmin + i*dx;
  }
}

int main(void) {
  // Select some random coefficients
  double c[3][3];
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    c[i][j] = 2.*(std::rand()/RAND_MAX - 1.0);
  }

  // Setup a grid
  int const Nx      = 10;
  double const xmin = -2.;
  double const xmax = 2.;
  double const dx   = (xmax - xmin)/(Nx - 1);
  double * x = new double[Nx];
  make_grid(xmin, dx, Nx, x);

  int const Ny      = 10;
  double const ymin = -2.;
  double const ymax = 2.;
  double const dy   = (ymax - ymin)/(Ny - 1);
  double * y = new double[Ny];
  make_grid(ymin, dy, Ny, y);

  double * gf = new double[Nx*Ny];
  for (int i = 0; i < Nx; ++i)
  for (int j = 0; j < Ny; ++j) {
    gf[j*Nx + i] = poly(x[i], y[j], c);
  }

  double origin[] = {xmin, ymin};
  double delta[] = {dx, dy};
  int siz[] = {Nx, Ny};
  {
    double coord[] = {M_PI/3., std::exp(1)};
    LagrangeInterpND<2, 2> interp(origin, delta, siz, coord);

    double Igf = interp.eval(gf);
    double Egf = poly(coord[0], coord[1], c);

    assert (std::abs(Igf - Egf) < 100*std::numeric_limits<double>::epsilon());
  }

  for (int i = 0; i < Nx; ++i)
  for (int j = 0; j < Ny; ++j) {
    gf[j*Nx + i] = sin(M_PI*x[i])*sin(2*M_PI*y[j]);
  }

  {
    double coord[2][2] = {{-0.5, 0.5}, {0.5, -0.5}};
    LagrangeInterpND<3, 2> interp1(origin, delta, siz, coord[0]);
    LagrangeInterpND<3, 2> interp2(origin, delta, siz, coord[1]);

    double Igf1 = interp1.eval(gf);
    double Igf2 = interp1.eval(gf);

    assert (std::abs(Igf1 - Igf2) < std::numeric_limits<double>::epsilon());
  }

  delete[] gf;
  delete[] y;
  delete[] x;
}
