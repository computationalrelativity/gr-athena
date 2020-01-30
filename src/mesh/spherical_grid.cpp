//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file spherical_grid.cpp
//  \brief implements SphericalGrid and SphericalPatch

#include <cmath>
#include <list>

#include "../coordinates/coordinates.hpp"
#include "spherical_grid.hpp"

SphericalGrid::SphericalGrid(int nlev, Real rad): GeodesicGrid(nlev), m_r(rad) { }

void SphericalGrid::Position(int ic, Real * x, Real * y, Real * z) const {
  Real theta, phi;
  GeodesicGrid::PositionPolar(ic, &theta, &phi);
  *x = m_r*std::sin(theta)*std::cos(phi);
  *y = m_r*std::sin(theta)*std::sin(phi);
  *z = m_r*std::cos(theta);
}

Real SphericalGrid::ComputeWeight(int ic) const {
  return m_r*m_r*GeodesicGrid::ComputeWeight(ic);
}

Real SphericalGrid::ArcLength(int ic1, int ic2) const {
  return m_r*GeodesicGrid::ArcLength(ic1, ic2);
}

SphericalPatch::SphericalPatch(SphericalGrid const * psphere, MeshBlock const * pblock, collocation_t coll):
    coll(coll), pm_interp(nullptr), pm_sphere(psphere), pm_block(pblock) {
  MeshBlock const * pmb = pm_block;
  Coordinates const * pmc = pm_block->pcoord;

  Real xmin, xmax, ymin, ymax, zmin, zmax;
  Real origin[3];
  Real delta[3];
  int size[3];

  switch (coll) {
    cell:
      xmin = pmc->x1v(pmb->is);
      xmax = pmc->x1v(pmb->ie);
      ymin = pmc->x2v(pmb->js);
      ymax = pmc->x2v(pmb->je);
      zmin = pmc->x3v(pmb->ks);
      zmax = pmc->x3v(pmb->ke);
      origin[0] = pmc->x1v(0);
      origin[1] = pmc->x2v(0);
      origin[3] = pmc->x3v(0);
      size[0] = pmb->block_size.nx1 + 2*(NGHOST);
      size[1] = pmb->block_size.nx2 + 2*(NGHOST);
      size[2] = pmb->block_size.nx3 + 2*(NGHOST);
      break;
    vertex:
      xmin = pmc->x1f(pmb->is);
      xmax = pmc->x1f(pmb->ie);
      ymin = pmc->x2f(pmb->js);
      ymax = pmc->x2f(pmb->je);
      zmin = pmc->x3f(pmb->ks);
      zmax = pmc->x3f(pmb->ke);
      origin[0] = pmc->x1f(0);
      origin[1] = pmc->x2f(0);
      origin[3] = pmc->x3f(0);
      size[0] = pmb->block_size.nx1 + 2*(NGHOST) + 1;
      size[1] = pmb->block_size.nx2 + 2*(NGHOST) + 1;
      size[2] = pmb->block_size.nx3 + 2*(NGHOST) + 1;
      break;
  }
  delta[0] = pmc->dx1v(0);
  delta[1] = pmc->dx2v(0);
  delta[3] = pmc->dx3v(0);

  // Loop over all points to find those belonging to this spherical patch
  int const np = pm_sphere->NumVertices();
  m_map.reserve(np);
  for (int ic = 0; ic < np; ++ic) {
    Real x, y, z;
    pm_sphere->Position(ic, &x, &y, &z);
    if (x >= xmin && x <= xmax && y >= ymin && y <= ymax && z <= zmin && z >= zmax) {
      m_map.push_back(ic);
    }
  }
  m_map.shrink_to_fit();
  m_n = m_map.size();

  pm_interp = new LagrangeInterpND<2*NGHOST, 3> *[m_n];
  for (int i = 0; i < m_n; ++i) {
    Real coord[3];
    pm_sphere->Position(m_map[i], &coord[0], &coord[1], &coord[2]);
    pm_interp[i] = new LagrangeInterpND<2*NGHOST, 3>(origin, delta, size, coord);
  }
}

SphericalPatch::~SphericalPatch() {
  for (int i = 0; i < m_n; ++i) {
    delete pm_interp[i];
  }
  delete[] pm_interp;
}

void SphericalPatch::interpToSpherical(Real const * src, Real * dst) const {
  for (int i = 0; i < m_n; ++i) {
    dst[i] = pm_interp[i]->eval(src);
  }
}

void SphericalPatch::InterpToSpherical(AthenaArray<Real> const & src, AthenaArray<Real> * dst) const {
  assert (src.GetDim2() == dst->GetDim2());
  assert (dst->GetDim1() == m_n);
  AthenaArray<Real> mySrc, myDst;
  int const nvars = src.GetDim2();
  for (int iv = 0; iv < nvars; ++iv) {
    mySrc.InitWithShallowSlice(const_cast<AthenaArray<Real>&>(src), iv, 1);
    myDst.InitWithShallowSlice(*dst, iv, 1);
    interpToSpherical(mySrc.data(), myDst.data());
  }
}

void SphericalPatch::mergeData(Real const * src, Real * dst) const {
  for (int i = 0; i < m_n; ++i) {
    dst[m_map[i]] = src[i];
  }
}

void SphericalPatch::MergeData(AthenaArray<Real> const & src, AthenaArray<Real> * dst) const {
  assert (src.GetDim2() == dst->GetDim2());
  assert (dst->GetDim1() == pm_sphere->NumVertices());
  assert (src.GetDim1() == m_n);
  AthenaArray<Real> mySrc, myDst;
  int const nvars = src.GetDim2();
  for (int iv = 0; iv < nvars; ++iv) {
    mySrc.InitWithShallowSlice(const_cast<AthenaArray<Real>&>(src), iv, 1);
    myDst.InitWithShallowSlice(*dst, iv, 1);
    mergeData(mySrc.data(), myDst.data());
  }
}
