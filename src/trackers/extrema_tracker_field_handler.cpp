// Interpolation of various fields to a point
// Write as column data

// c/c++
#include <cstddef>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <unistd.h>

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// Athena++
#include "extrema_tracker.hpp"

// for registration of control field..
#if M1_ENABLED
#include "../m1/m1.hpp"
#endif

#if MAGNETIC_FIELDS_ENABLED
#include "../field/field.hpp"
#endif

#if FLUID_ENABLED
#include "../hydro/hydro.hpp"
#endif

#if WAVE_ENABLED
#include "../wave/wave.hpp"
#endif

#if Z4C_ENABLED
#include "../z4c/z4c.hpp"
#endif

// ----------------------------------------------------------------------------

namespace {

Real DoInterpolateCC(
  MeshBlock * pmb,
  AA & field_cc,
  const Real n,
  const Real x, const Real y, const Real z
)
{
  ExtremaTracker * pet = pmb->pmy_mesh->ptracker_extrema;

  // Uniform grid spacing assumed
  const int ndim = pmb->pmy_mesh->ndim;

  Real origin[ndim];
  Real ds[ndim];
  int sz[ndim];
  Real coord[ndim];

  // populate salient data in this block
  switch (ndim)
  {
    case 3:
    {
      origin[2] = pmb->pcoord->x3v(0);
      sz[2] = pmb->ncells3;
      ds[2] = pmb->pcoord->dx3v(0);
      coord[2] = z;
    }
    case 2:
    {
      origin[1] = pmb->pcoord->x2v(0);
      sz[1] = pmb->ncells2;
      ds[1] = pmb->pcoord->dx2v(0);
      coord[1] = y;
    }
    case 1:
    {
      origin[0] = pmb->pcoord->x1v(0);
      sz[0] = pmb->ncells1;
      ds[0] = pmb->pcoord->dx1v(0);
      coord[0] = x;
      break;
    }
    default:
    {
      std::cout << "DoInterpolateCC requires ndim<=3" << std::endl;
      assert(false);
    }
  }

  AA slice_cc;
  slice_cc.InitWithShallowSlice(field_cc, n, 1);
  Real value_interpolated = std::numeric_limits<Real>::quiet_NaN();

  switch (ndim)
  {
    case 3:
    {
      typedef LagrangeInterpND<2*(NGHOST-1), 3> Interp_Lag3;
      Interp_Lag3 * pinterp3 = new Interp_Lag3(origin, ds, sz, coord);

      value_interpolated = pinterp3->eval(&(slice_cc(0,0,0)));

      delete pinterp3;
      break;
    }
    case 2:
    {
      typedef LagrangeInterpND<2*(NGHOST-1), 2> Interp_Lag2;
      Interp_Lag2 * pinterp2 = new Interp_Lag2(origin, ds, sz, coord);

      value_interpolated = pinterp2->eval(&(slice_cc(0,0,0)));

      delete pinterp2;
      break;
    }
    case 1:
    {
      typedef LagrangeInterpND<2*(NGHOST-1), 1> Interp_Lag1;
      Interp_Lag1 * pinterp1 = new Interp_Lag1(origin, ds, sz, coord);

      value_interpolated = pinterp1->eval(&(slice_cc(0)));

      delete pinterp1;
      break;
    }
    default:
    {
      std::cout << "DoInterpolateCC requires ndim<=3" << std::endl;
      assert(false);
    }
  }

  return value_interpolated;
}

}

// ----------------------------------------------------------------------------

// Interpolate everything that is possible to interpolate
void ExtremaTracker::TryInterpolateAndWriteFields(
  MeshBlock * pmb,
  int num_tracker, int iter, Real time
)
{
  const int n = num_tracker;

  // if no evaluation or point not contained in current MeshBlock do nothing
  if (!evaluate_fields(n-1) ||
      !pmb->PointContainedExclusive(c_x1(n-1), c_x2(n-1), c_x3(n-1)))
  {
    return;
  }

  // tracker in current MeshBlock and we want to interpolate then dump data ---
  const Real x = c_x1(n-1);
  const Real y = c_x2(n-1);
  const Real z = c_x3(n-1);

  Hydro * ph = pmb->phydro;
  PassiveScalars * ps = pmb->pscalars;
  Field * pf = pmb->pfield;
  Z4c * pz4c = pmb->pz4c;

  // try access file ----------------------------------------------------------
  bool new_file = true;
  std::string fn = output_filename + std::to_string(n) + ".evf.txt";
  if (access(fn.c_str(), F_OK) == 0)
  {
    new_file = false;
  }

  FILE* pofile = fopen(fn.c_str(), "a");

  if (pofile == nullptr)
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in ExtremaTracker::"
        << "TryInterpolateAndWriteFields \n"
        << "Could not open file '" << fn << "' for writing!";
    throw std::runtime_error(msg.str());
  }

  // --------------------------------------------------------------------------

  // push interpolated values to this stream ----------------------------------
  int ix_var = 0;
  std::ostringstream oss_header; // only needed if file doesn't exist

  if (new_file)
  {
    oss_header << "# Values of fields @ tracker number: " << n << "\n";
  }

  std::ostringstream oss;

  // scientific has no effect on integers
  oss.setf(std::ios::scientific, std::ios::floatfield);

  // formatting functions -----------------------------------------------------
  auto push_num_int = [&](int v)
  {
    oss << std::left << std::setw(13);
    oss << v;
  };

  auto push_num_Real = [&](Real v)
  {
    if (v >= 0)
    {
      // leading space so + and - align (+ not shown)
      oss << " ";
    }
    oss << std::setprecision(FPRINTF_PREC) << " " << v;
  };
  // --------------------------------------------------------------------------

  if (new_file)
  {
    oss_header << "#";
    oss_header << " [" << ++ix_var << "]=iter";
    oss_header << " [" << ++ix_var << "]=Time";
    oss_header << " [" << ++ix_var << "]=T.x";
    oss_header << " [" << ++ix_var << "]=T.y";
    oss_header << " [" << ++ix_var << "]=T.z";
  }

  push_num_int(iter);
  push_num_Real(time);
  push_num_Real(c_x1(n-1));
  push_num_Real(c_x2(n-1));
  push_num_Real(c_x3(n-1));

  // Use macros to suppress things safeley
#if FLUID_ENABLED
  // if (FLUID_ENABLED)
  {
    for (int ix=0; ix<Hydro::ixn_cons::N; ++ix)
    {
      const Real interp_val = DoInterpolateCC(pmb, ph->u, ix, x, y, z);
      push_num_Real(interp_val);

      if (new_file)
      {
        oss_header << " [" << ++ix_var << "]=";
        oss_header << Hydro::ixn_cons::names[ix];
      }
    }

    for (int ix=0; ix<Hydro::ixn_prim::N; ++ix)
    {
      const Real interp_val = DoInterpolateCC(pmb, ph->w, ix, x, y, z);
      push_num_Real(interp_val);

      if (new_file)
      {
        oss_header << " [" << ++ix_var << "]=";
        oss_header << Hydro::ixn_prim::names[ix];
      }
    }

    for (int ix=0; ix<HydroDerivedIndex::NDRV_HYDRO; ++ix)
    {
      const Real interp_val = DoInterpolateCC(
        pmb, ph->derived_ms, ix, x, y, z
      );
      push_num_Real(interp_val);

      if (new_file)
      {
        oss_header << " [" << ++ix_var << "]=";
        oss_header << Hydro::ixn_derived_ms::names[ix];
      }
    }
  }
#endif // FLUID_ENABLED

#if NSCALARS > 0
  // if (NSCALARS > 0)
  {
    for (int ix=0; ix<NSCALARS; ++ix)
    {
      const Real interp_val = DoInterpolateCC(pmb, ps->s, ix, x, y, z);
      push_num_Real(interp_val);

      if (new_file)
      {
        oss_header << " [" << ++ix_var << "]=";
        oss_header << "passive_scalar.s_" << ix;
      }
    }

    for (int ix=0; ix<NSCALARS; ++ix)
    {
      const Real interp_val = DoInterpolateCC(pmb, ps->r, ix, x, y, z);
      push_num_Real(interp_val);

      if (new_file)
      {
        oss_header << " [" << ++ix_var << "]=";
        oss_header << "passive_scalar.r_" << ix;
      }
    }
  }
#endif // NSCALARS > 0

#if MAGNETIC_FIELDS_ENABLED
  // if (MAGNETIC_FIELDS_ENABLED)
  {
    for (int ix=0; ix<Field::ixn_cc::N; ++ix)
    {
      const Real interp_val = DoInterpolateCC(pmb, ph->w, ix, x, y, z);
      push_num_Real(interp_val);

      if (new_file)
      {
        oss_header << " [" << ++ix_var << "]=";
        oss_header << Field::ixn_cc::names[ix];
      }
    }

    for (int ix=0; ix<FieldDerivedIndex::NDRV_FIELD; ++ix)
    {
      const Real interp_val = DoInterpolateCC(
        pmb, pf->derived_ms, ix, x, y, z
      );
      push_num_Real(interp_val);

      if (new_file)
      {
        oss_header << " [" << ++ix_var << "]=";
        oss_header << Field::ixn_derived_ms::names[ix];
      }
    }
  }
#endif // MAGNETIC_FIELDS_ENABLED

#if Z4C_ENABLED
  // if (Z4C_ENABLED)
  {
#if defined(Z4C_VC_ENABLED)
    // support could be extended for this, would require VC interp.
    assert(false);
#endif

    // z4c state-vector variables
    for (int ix=0; ix<Z4c::N_Z4c; ++ix)
    {
      const Real interp_val = DoInterpolateCC(
        pmb, pz4c->storage.u,
        ix, x, y, z);
      push_num_Real(interp_val);

      if (new_file)
      {
        oss_header << " [" << ++ix_var << "]=";
        oss_header << Z4c::Z4c_names[ix];
      }
    }

    for (int ix=0; ix<Z4c::N_ADM; ++ix)
    {
      const Real interp_val = DoInterpolateCC(
        pmb, pz4c->storage.adm,
        ix, x, y, z);
      push_num_Real(interp_val);

      if (new_file)
      {
        oss_header << " [" << ++ix_var << "]=";
        oss_header << Z4c::ADM_names[ix];
      }
    }

    for (int ix=0; ix<Z4c::N_CON; ++ix)
    {
      const Real interp_val = DoInterpolateCC(
        pmb, pz4c->storage.con,
        ix, x, y, z);
      push_num_Real(interp_val);

      if (new_file)
      {
        oss_header << " [" << ++ix_var << "]=";
        oss_header << Z4c::Constraint_names[ix];
      }
    }

    if (FLUID_ENABLED)
    {
      for (int ix=0; ix<Z4c::N_MAT; ++ix)
      {
        const Real interp_val = DoInterpolateCC(
          pmb, pz4c->storage.mat,
          ix, x, y, z);
        push_num_Real(interp_val);

        if (new_file)
        {
          oss_header << " [" << ++ix_var << "]=";
          oss_header << Z4c::Matter_names[ix];
        }
      }
    }

    for (int ix=0; ix<Z4c::N_AUX_EXTENDED; ++ix)
    {
      const Real interp_val = DoInterpolateCC(
        pmb, pz4c->storage.aux_extended,
        ix, x, y, z);
      push_num_Real(interp_val);

      if (new_file)
      {
        oss_header << " [" << ++ix_var << "]=";
        oss_header << Z4c::Aux_Extended_names[ix];
      }
    }

  }
#endif // Z4C_ENABLED

  if (new_file)
  {
    oss_header << "\n";
    std::string out_header = oss_header.str();
    fwrite(out_header.data(), 1, out_header.size(), pofile);
  }

  oss << "\n";
  std::string out = oss.str();
  fwrite(out.data(), 1, out.size(), pofile);
  fclose(pofile);
}

//
// :D
//