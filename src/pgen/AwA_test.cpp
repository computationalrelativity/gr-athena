//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file apples.cpp
//  \brief Initial conditions for Apples with Apples Test

#include <cassert> // assert
#include <iostream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../z4c/z4c.hpp"

using namespace std;

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Sets the initial conditions.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{

  string test = pin->GetOrAddString("problem", "test", "minkowski");

  if(test == "robust_stab") {
    pz4c->ADMRobustStability(pz4c->storage.adm);
    //Gauge
  }
  else if(test == "linear_wave1") {
      pz4c->ADMLinearWave1(pz4c->storage.adm);
      //Gauge
  }
  else if(test == "linear_wave2") {
      pz4c->ADMLinearWave2(pz4c->storage.adm);
      //Gauge
  }
  else if(test == "gauge_wave1") {
      pz4c->ADMGaugeWave1(pz4c->storage.adm);
      //Gauge
  }
  else if(test == "gauge_wave2") {
      pz4c->ADMGaugeWave2(pz4c->storage.adm);
      //Gauge
  }
  else if(test == "gauge_wave_lapse") {
      pz4c->GaugeGaugeWaveLapse(pz4c->storage.u);
      //Gauge
  }
  else if(test == "gauge_wave_lapse_shift") {
      pz4c->GaugeGaugeWaveLapseShift(pz4c->storage.u);
      //Gauge
  }
  else { // Minkowski test
    pz4c->ADMMinkowski(pz4c->storage.adm);
    pz4c->GaugeGaugeWaveLapse(pz4c->storage.u); //Setting gauge
  }

  //Constructing Z4c vars from ADM ones:
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);

  return;
}
