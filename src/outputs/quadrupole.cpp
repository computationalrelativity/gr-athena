//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file quadrupole.cpp
//  \brief writes mass quadrupole first time derivative data.

// C headers

// C++ headers
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

// Athena++ headers
#include "../defs.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../z4c/z4c.hpp"
#include "../wave/wave.hpp"
#include "../mesh/mesh.hpp"
#include "../scalars/scalars.hpp"
#include "outputs.hpp"

// dot(Ixx), dot(Ixy), dot(Ixz), dot(Iyy), dot(Iyz), dot(Izz)
#define NQUAD_VARS 6

//! \fn void OutputType::QuadrupoleFile()
//  \brief Writes a txt file
void HistoryOutput::WriteQuadFile(Mesh *pm, ParameterInput *pin, bool flag)
{
  MeshBlock *pmb = pm->pblock;
  Real real_max = std::numeric_limits<Real>::max();
  Real real_min = std::numeric_limits<Real>::min();
  AthenaArray<Real> vol(pmb->ncells1);

  std::unique_ptr<Real[]> quad_data(new Real[NQUAD_VARS]);
  // initialize built-in variable sums to 0.0
  for (int n=0; n<NQUAD_VARS; ++n) quad_data[n] = 0.0;

  int ix_cons_dens; 

  // Loop over MeshBlocks
  while (pmb != nullptr) {
    Hydro *phyd = pmb->phydro;

    // Sum history variables over cells.  Note ghost cells are never included in sums
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          // NEW_OUTPUT_TYPES:

          Coordinates *pcoord = pmb->pcoord;
          const Real x = pcoord->x1v(i);
          const Real y = pcoord->x2v(j);
          const Real z = pcoord->x3v(k);

          int isum = 0;
          if (FLUID_ENABLED) {
            // Hydro conserved variables:
            Real& u_d  = phyd->u(IDN,k,j,i);
            Real& u_vx = phyd->w(IVX,k,j,i);
            Real& u_vy = phyd->w(IVY,k,j,i);
            Real& u_vz = phyd->w(IVZ,k,j,i);

#if FLUID_ENABLED
            if (pm->presc->opt.rescale_conserved_density)
            {
              ix_cons_dens = isum;
            }
#endif      

	  Real x_v_diag = x*u_vx + y*u_vy + z*u_vz;

          quad_data[isum++] += vol(i)*u_d*(2.0*u_vx*x - (2.0/3.0)*x_v_diag);
          quad_data[isum++] += vol(i)*u_d*(u_vx*y + u_vy*x);
          quad_data[isum++] += vol(i)*u_d*(u_vx*z + u_vz*x);
          quad_data[isum++] += vol(i)*u_d*(2.0*u_vy*y - (2.0/3.0)*x_v_diag);
          quad_data[isum++] += vol(i)*u_d*(u_vy*z + u_vz*y);
          quad_data[isum++] += vol(i)*u_d*(2.0*u_vz*z - (2.0/3.0)*x_v_diag);

          }

        }
      }
    }

   pmb = pmb->next;

  } // end loop over MeshBlocks

#ifdef MPI_PARALLEL
  // sum built-in/predefined quad_data[] over all ranks
  if (Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, quad_data.get(), NQUAD_VARS, MPI_ATHENA_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
  } else {
    MPI_Reduce(quad_data.get(), quad_data.get(), NQUAD_VARS, MPI_ATHENA_REAL, MPI_SUM, 0, 
               MPI_COMM_WORLD);
  }
#endif


  // only the master rank writes the file
  // create filename: "file_basename" + ".hst".  There is no file number.
  if (Globals::my_rank == 0) {
    std::string fname;
    fname.assign(output_params.file_basename);
    fname.append("_quadrupole.txt");

    // open file for output
    FILE *pfile;
    std::stringstream msg;

    // This bool allows to rewrite header below, and is useful
    // if the output folder changes from the restart.
    // This should be harmless otherwise.
    bool new_file = true;
    if (access(fname.c_str(), F_OK) == 0) {
      //printf("Found %s!\n", fname.c_str());
      new_file = false;
    }

    if ((pfile = std::fopen(fname.c_str(),"a")) == nullptr) {
      msg << "### FATAL ERROR in function [OutputType::HistoryFile]" << std::endl
          << "Output file '" << fname << "' could not be opened";
      ATHENA_ERROR(msg);
    }

    // If this is the first output, write header
    if (output_params.file_number == 0 || new_file) {
      // NEW_OUTPUT_TYPES:

      int iout = 1;
      // descriptor + hash is first line ---
      std::string ver("# GR-Athena++ (");
      ver.append(GIT_HASH);
      ver.append(") Mass Quadrupole first time derivative data\n");
      std::fprintf(pfile,"%s", ver.c_str());
      // -----------------------------------

      std::fprintf(pfile,"# [%d]=time ", iout++);
      std::fprintf(pfile,"[%d]=dt ", iout++);
      if (FLUID_ENABLED) {
        std::fprintf(pfile,"[%d]=dot(Ixx) ", iout++);
        std::fprintf(pfile,"[%d]=dot(Ixy) ", iout++);
        std::fprintf(pfile,"[%d]=dot(Ixz) ", iout++);
        std::fprintf(pfile,"[%d]=dot(Iyy) ", iout++);
        std::fprintf(pfile,"[%d]=dot(Iyz) ", iout++);
        std::fprintf(pfile,"[%d]=dot(Izz) ", iout++);
      }

      std::fprintf(pfile,"\n");
    }

    // write variables
    std::fprintf(pfile, output_params.data_format.c_str(), pm->time);
    std::fprintf(pfile, output_params.data_format.c_str(), pm->dt);
    for (int n=0; n<NQUAD_VARS; ++n)
      std::fprintf(pfile, output_params.data_format.c_str(), quad_data[n]);
    std::fprintf(pfile,"\n"); // terminate line
    std::fclose(pfile);
 
  }

  // increment counters, clean up
  //output_params.file_number++;
  //output_params.next_time += output_params.dt;
  //pin->SetInteger(output_params.block_name, "file_number", output_params.file_number);
  //pin->SetReal(output_params.block_name, "next_time", output_params.next_time);
  return;

}

