// See m1.hpp for description / references.

// c++
#include <map>
#include <codecvt>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

// Athena++ headers
#include "m1.hpp"

// ============================================================================
namespace M1 {
// ============================================================================

void M1::PopulateOptionsClosure(ParameterInput *pin)
{
  const std::string option_block {"M1_closure"};
  std::ostringstream msg;

  // access wrappers
  auto GoA_Real = [&](const std::string & name, const Real default_value)
  {
    return pin->GetOrAddReal(option_block, name, default_value);
  };

  auto GoA_int = [&](const std::string & name, const int default_value)
  {
    return pin->GetOrAddInteger(option_block, name, default_value);
  };

  auto GoA_bool = [&](const std::string & name, const int default_value)
  {
    return pin->GetOrAddInteger(option_block, name, default_value);
  };

  auto GoA_str = [&](const std::string & name,
                     const std::string & default_value)
  {
    return pin->GetOrAddString(option_block, name, default_value);
  };
  // --------------------------------------------------------------------------

  {
    static const std::map<std::string, opt_closure_variety> opt_var {
      { "thin",     opt_closure_variety::thin},
      { "thick",    opt_closure_variety::thick},
      { "Minerbo",  opt_closure_variety::Minerbo},
      { "MinerboN", opt_closure_variety::MinerboN},
      { "MinerboP", opt_closure_variety::MinerboP},
      { "MinerboB", opt_closure_variety::MinerboB},
    };

    auto itr = opt_var.find(GoA_str("variety", "thin"));
    if (itr != opt_var.end())
    {
      opt_closure.variety = itr->second;
    }
    else
    {
      msg << "M1_closure/variety unknown" << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  opt_closure.eps_tol     = GoA_Real("eps_tol",     1e-10);
  opt_closure.eps_Z_o_E   = GoA_Real("eps_Z_o_E",   1e-20);
  opt_closure.fac_Z_o_E   = GoA_Real("fac_Z_o_E",   0.01);
  opt_closure.w_opt_ini   = GoA_Real("w_opt_init",  1.0);
  opt_closure.fac_err_amp = GoA_Real("fac_err_amp", 1.11);

  opt_closure.iter_max     = GoA_int("iter_max", 32);
  opt_closure.iter_max_rst = GoA_int("iter_max_rst", 5);

  opt_closure.fallback_thin = GoA_bool("fallback_thin", false);

  opt_closure.use_Ostrowski = GoA_bool("use_Ostrowski", false);
  opt_closure.use_Neighbor  = GoA_bool("use_Neighbor",  false);

  opt_closure.verbose = GoA_bool("verbose", false);
}

void M1::PopulateOptionsSolver(ParameterInput *pin)
{
  const std::string option_block {"M1_solver"};
  std::ostringstream msg;

  // access wrappers
  auto GoA_Real = [&](const std::string & name, const Real default_value)
  {
    return pin->GetOrAddReal(option_block, name, default_value);
  };

  auto GoA_int = [&](const std::string & name, const int default_value)
  {
    return pin->GetOrAddInteger(option_block, name, default_value);
  };

  auto GoA_bool = [&](const std::string & name, const int default_value)
  {
    return pin->GetOrAddInteger(option_block, name, default_value);
  };

  auto GoA_str = [&](const std::string & name,
                     const std::string & default_value)
  {
    return pin->GetOrAddString(option_block, name, default_value);
  };
  // --------------------------------------------------------------------------

  {
    static const std::map<std::string, opt_integration_strategy> opt_strat {
      { "full_explicit", opt_integration_strategy::full_explicit},
      { "semi_implicit_PicardFrozenP",
        opt_integration_strategy::semi_implicit_PicardFrozenP},
      { "semi_implicit_PicardMinerboP",
        opt_integration_strategy::semi_implicit_PicardMinerboP},
      { "semi_implicit_PicardMinerboPC",
        opt_integration_strategy::semi_implicit_PicardMinerboPC},
      { "semi_implicit_HybridsJFrozenP",
        opt_integration_strategy::semi_implicit_HybridsJFrozenP},
      { "semi_implicit_HybridsJMinerbo",
        opt_integration_strategy::semi_implicit_HybridsJMinerbo},
      { "semi_implicit_Hybrids",
        opt_integration_strategy::semi_implicit_Hybrids}
    };

    auto itr = opt_strat.find(GoA_str("strategy", "full_explicit"));
    if (itr != opt_strat.end())
    {
      opt_solver.strategy = itr->second;
    }
    else
    {
      msg << "M1_solver/strategy unknown" << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  opt_solver.eps_tol     = GoA_Real("eps_tol",     1e-10);
  opt_solver.w_opt_ini   = GoA_Real("w_opt_init",  1.0);
  opt_solver.fac_err_amp = GoA_Real("fac_err_amp", 1.11);

  opt_solver.iter_max     = GoA_int("iter_max", 128);
  opt_solver.iter_max_rst = GoA_int("iter_max_rst", 10);

  opt_solver.use_Neighbor = GoA_bool("use_Neighbor",  false);

  opt_solver.verbose = GoA_bool("verbose", false);
}

void M1::PopulateOptions(ParameterInput *pin)
{
  std::string tmp;
  std::ostringstream msg;

  opt.use_split_step = pin->GetOrAddBoolean("problem",
                                            "use_split_step",
                                            false);

  PopulateOptionsClosure(pin);
  PopulateOptionsSolver( pin);

  { // fluxes
    tmp = pin->GetOrAddString("M1", "characteristics_variety", "approximate");
    if (tmp == "approximate")
    {
      opt.characteristics_variety = opt_characteristics_variety::approximate;
    }
    else if (tmp == "exact_thin")
    {
      opt.characteristics_variety = opt_characteristics_variety::exact_thin;
    }
    else if (tmp == "exact_thick")
    {
      opt.characteristics_variety = opt_characteristics_variety::exact_thick;
    }
    else if (tmp == "exact_Minerbo")
    {
      opt.characteristics_variety = opt_characteristics_variety::exact_Minerbo;
    }
    else
    {
      msg << "M1/characteristics_variety unknown" << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  { // fiducial
    tmp = pin->GetOrAddString("M1", "fiducial_velocity", "fluid");
    if (tmp == "fluid")
    {
      opt.fiducial_velocity = opt_fiducial_velocity::fluid;
    }
    else if (tmp == "mixed")
    {
      opt.fiducial_velocity = opt_fiducial_velocity::mixed;
    }
    else if (tmp == "zero")
    {
      opt.fiducial_velocity = opt_fiducial_velocity::zero;
    }
    else if (tmp == "none")
    {
      opt.fiducial_velocity = opt_fiducial_velocity::none;
    }
    else
    {
      msg << "M1/fiducial_velocity unknown" << std::endl;
      ATHENA_ERROR(msg);
    }

    opt.fiducial_velocity_rho_fluid = pin->GetOrAddReal(
      "M1", "fiducial_velocity_rho_fluid", 0.0) * M1_UNITS_CGS_GCC;
  }

  { // tol / ad-hoc
    opt.fl_E = pin->GetOrAddReal("M1", "fl_E", 1e-15);
    opt.fl_J = pin->GetOrAddReal("M1", "fl_J", 1e-15);
    opt.eps_E = pin->GetOrAddReal("M1", "eps_E", 1e-5);
    opt.eps_J = pin->GetOrAddReal("M1", "eps_J", 1e-10);
    opt.enforce_causality = pin->GetOrAddBoolean(
      "M1", "enforce_causality", true);
    opt.eps_ec_fac = pin->GetOrAddReal("M1", "eps_ec_fac", 1e-15);

    opt.min_flux_A = pin->GetOrAddReal("M1", "min_flux_A", 0);
  }

  { // coupling
    opt.couple_sources_ADM = pin->GetOrAddBoolean("M1",
                                                  "couple_sources_ADM",
                                                  false);

    opt.couple_sources_hydro = pin->GetOrAddBoolean("M1",
                                                    "couple_sources_hydro",
                                                    false);

    opt.couple_sources_Y_e = pin->GetOrAddBoolean("M1",
                                                  "couple_sources_Y_e",
                                                  false);

  }

  // debugging
  opt.value_inject = pin->GetOrAddBoolean("problem", "value_inject", false);
}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//