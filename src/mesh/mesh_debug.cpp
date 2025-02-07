// C headers

// C++ headers
#include <cstdio>
#include <iostream>
#include <limits>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_aliases.hpp"
#include "../bvals/bvals.hpp"
#include "../globals.hpp"
#include "mesh.hpp"

#include "../hydro/hydro.hpp"
#include "../field/field.hpp"
#include "../m1/m1.hpp"
#include "../scalars/scalars.hpp"
#include "../wave/wave.hpp"
#include "../z4c/z4c.hpp"

using namespace gra::aliases;

void Mesh::DebugMesh(const Real x3__,
                     const Real x2__,
                     const Real x1__)
{
  /*
  std::vector<MeshBlock*> pmb_array;
  GetMeshBlocksMyRank(pmb_array);

  const int nmb = pmb_array.size();

  for (int ni = 0; ni < nmb; ++ni)
  {
    MeshBlock * pmb = pmb_array[ni];
    Hydro * ph = pmb->phydro;
    PassiveScalars * ps = pmb->pscalars;
    Z4c * pz = pmb->pz4c;
    EquationOfState * peos = pmb->peos;

    auto p_nearest = [&](AA & coords, const Real x, const int N_ax)
    {
      int ix_min = 0;
      Real dist_min = +std::numeric_limits<Real>::infinity();

      for (int ix=0; ix<N_ax; ++ix)
      {
        if (std::abs(x - coords(ix)) < dist_min)
        {
          dist_min = std::abs(x - coords(ix));
          ix_min = ix;
        }
      }
      return ix_min;
    };

    if (pmb->PointContainedExclusive(x1__, x2__, x3__))
    {
      int i1, i2, i3;

      AA & x1_ = pmb->pcoord->x1v;
      AA & x2_ = pmb->pcoord->x2v;
      AA & x3_ = pmb->pcoord->x3v;

      i1 = p_nearest(x1_, x1__, pmb->ncells1);
      i2 = p_nearest(x2_, x2__, pmb->ncells2);
      i3 = p_nearest(x3_, x3__, pmb->ncells3);

      i1 = 0;
      i2 = 0;
      i3 = 0;

      // hydro sector
      std::printf("ph->u\n");
      for (int n=0; n<ph->u.GetDim(4); ++n)
      {
        std::printf(
          "%.16g @ (%.3g,%.3g,%.3g)\n",
          ph->u(n,i3,i2,i1),
          x3_(i3), x2_(i2), x1_(i1)
        );
      }

      std::printf("ph->u1\n");
      for (int n=0; n<ph->u1.GetDim(4); ++n)
      {
        std::printf(
          "%.16g @ (%.3g,%.3g,%.3g)\n",
          ph->u1(n,i3,i2,i1),
          x3_(i3), x2_(i2), x1_(i1)
        );
      }

      std::printf("ph->w\n");
      for (int n=0; n<ph->w.GetDim(4); ++n)
      {
        std::printf(
          "%.16g @ (%.3g,%.3g,%.3g)\n",
          ph->w(n,i3,i2,i1),
          x3_(i3), x2_(i2), x1_(i1)
        );
      }

      std::printf("ph->w1\n");
      for (int n=0; n<ph->w1.GetDim(4); ++n)
      {
        std::printf(
          "%.16g @ (%.3g,%.3g,%.3g)\n",
          ph->w1(n,i3,i2,i1),
          x3_(i3), x2_(i2), x1_(i1)
        );
      }


      // passive scalars
      if (NSCALARS > 0)
      {
        std::printf("ps->s\n");
        for (int n=0; n<ps->s.GetDim(4); ++n)
        {
          std::printf(
            "%.16g @ (%.3g,%.3g,%.3g)\n",
            ps->s(n,i3,i2,i1),
            x3_(i3), x2_(i2), x1_(i1)
          );
        }

        std::printf("ps->s1\n");
        for (int n=0; n<ps->s.GetDim(4); ++n)
        {
          std::printf(
            "%.16g @ (%.3g,%.3g,%.3g)\n",
            ps->s1(n,i3,i2,i1),
            x3_(i3), x2_(i2), x1_(i1)
          );
        }

        std::printf("ps->r\n");
        for (int n=0; n<ps->r.GetDim(4); ++n)
        {
          std::printf(
            "%.16g @ (%.3g,%.3g,%.3g)\n",
            ps->r(n,i3,i2,i1),
            x3_(i3), x2_(i2), x1_(i1)
          );
        }
      }

      // grav. sector
      std::printf("pz->storage.u\n");
      for (int n=0; n<pz->storage.u.GetDim(4); ++n)
      {
        std::printf(
          "%.16g @ (%.3g,%.3g,%.3g)\n",
          pz->storage.u(n,i3,i2,i1),
          x3_(i3), x2_(i2), x1_(i1)
        );
      }

      std::printf("pz->storage.u1\n");
      for (int n=0; n<pz->storage.u1.GetDim(4); ++n)
      {
        std::printf(
          "%.16g @ (%.3g,%.3g,%.3g)\n",
          pz->storage.u1(n,i3,i2,i1),
          x3_(i3), x2_(i2), x1_(i1)
        );
      }

      std::printf("pz->storage.adm\n");
      for (int n=0; n<pz->storage.adm.GetDim(4); ++n)
      {
        std::printf(
          "%.16g @ (%.3g,%.3g,%.3g)\n",
          pz->storage.adm(n,i3,i2,i1),
          x3_(i3), x2_(i2), x1_(i1)
        );
      }


      // pz->storage.u1.ZeroClear();
      // ph->u1.ZeroClear();
      // ps->s1.ZeroClear();

      Real sum_abs_fields (0.);

      auto fcn_array_abs_sum = [&](AA & to_sum)
      {
        Real sum (0.);
        for (int n=0; n<to_sum.GetDim(4); ++n)
        for (int k=0; k<=pmb->ncells3-1; ++k)
        for (int j=0; j<=pmb->ncells2-1; ++j)
        for (int i=0; i<=pmb->ncells1-1; ++i)
        // CC_GLOOP3(k, j, i)
        {
          sum += std::abs(to_sum(n,k,j,i));
        }

        return sum;
      };

      auto fcn_array_abs_sum_0 = [&](AA & to_sum)
      {
        Real sum (0.);
        // CC_GLOOP3(k, j, i)
        for (int k=0; k<=pmb->ncells3-1; ++k)
        for (int j=0; j<=pmb->ncells2-1; ++j)
        for (int i=0; i<=pmb->ncells1-1; ++i)
        {
          sum += std::abs(to_sum(0,k,j,i));
        }

        return sum;
      };


      sum_abs_fields = 0;
      sum_abs_fields += fcn_array_abs_sum(ph->u);
      // sum_abs_fields += fcn_array_abs_sum(ph->u1);

      std::printf("ph->u: %.16g\n", std::log10(sum_abs_fields));

      sum_abs_fields = 0;
      sum_abs_fields += fcn_array_abs_sum(ph->w);
      std::printf("ph->w: %.16g\n", std::log10(sum_abs_fields));

      sum_abs_fields = 0;
      sum_abs_fields += fcn_array_abs_sum_0(ph->w);
      std::printf("ph->w_0: %.16g\n", std::log10(sum_abs_fields));

      sum_abs_fields = 0;
      sum_abs_fields += fcn_array_abs_sum(ph->w1);
      std::printf("ph->w1: %.16g\n", std::log10(sum_abs_fields));

      if (NSCALARS > 0)
      {
        sum_abs_fields = 0;
        sum_abs_fields += fcn_array_abs_sum(ps->s);
        std::printf("ps->s: %.16g\n", std::log10(sum_abs_fields));
        sum_abs_fields = 0;
        sum_abs_fields += fcn_array_abs_sum(ps->r);
        std::printf("ps->r: %.16g\n", std::log10(sum_abs_fields));

      }

      sum_abs_fields = 0;
      sum_abs_fields += fcn_array_abs_sum(pz->storage.u);
      std::printf("pz->storage.u: %.16g\n", std::log10(sum_abs_fields));

      sum_abs_fields = 0;
      sum_abs_fields += fcn_array_abs_sum(pz->storage.adm);
      std::printf("pz->storage.adm: %.16g\n", std::log10(sum_abs_fields));

      sum_abs_fields = 0;
      sum_abs_fields += fcn_array_abs_sum(pz->storage.mat);
      std::printf("pz->storage.mat: %.16g\n", std::log10(sum_abs_fields));

      std::printf("floors: %.16g %.16g\n",
        peos->GetEOS().GetDensityFloor(),
        peos->GetEOS().GetTemperatureFloor()
      );


    }
  }
  */
}


//
// :D
//
