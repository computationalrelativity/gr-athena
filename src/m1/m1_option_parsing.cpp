// See m1.hpp for description / references.

// c++
#include <limits>
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
    return pin->GetOrAddBoolean(option_block, name, default_value);
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
      { "Kershaw",  opt_closure_variety::Kershaw}
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

  {
    static const std::map<std::string, opt_closure_method> opt_var {
      { "none",        opt_closure_method::none},
      { "gsl_Brent",   opt_closure_method::gsl_Brent},
      { "gsl_Newton",  opt_closure_method::gsl_Newton},
      { "custom_NB",   opt_closure_method::custom_NB},
      { "custom_NAB",  opt_closure_method::custom_NAB},
      { "custom_ONAB", opt_closure_method::custom_ONAB}
    };

    auto itr = opt_var.find(GoA_str("method", "gsl_Brent"));
    if (itr != opt_var.end())
    {
      opt_closure.method = itr->second;
    }
    else
    {
      msg << "M1_closure/method unknown" << std::endl;
      ATHENA_ERROR(msg);
    }

  }

  // various settings for methods
  opt_closure.eps_tol     = GoA_Real("eps_tol",     1e-10);
  opt_closure.eps_Z_o_E   = GoA_Real("eps_Z_o_E",   1e-20);
  opt_closure.fac_Z_o_E   = GoA_Real("fac_Z_o_E",   0.01);

  opt_closure.fallback_brent = GoA_bool("fallback_brent", true);
  opt_closure.fallback_thin = GoA_bool("fallback_thin", false);


  opt_closure.w_opt_ini   = GoA_Real("w_opt_init",  1.0);
  opt_closure.fac_err_amp = GoA_Real("fac_err_amp", 1.11);

  opt_closure.iter_max     = GoA_int("iter_max", 32);
  opt_closure.iter_max_rst = GoA_int("iter_max_rst", 5);


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
    return pin->GetOrAddBoolean(option_block, name, default_value);
  };

  auto GoA_str = [&](const std::string & name,
                     const std::string & default_value)
  {
    return pin->GetOrAddString(option_block, name, default_value);
  };
  // --------------------------------------------------------------------------

  {
    static const std::map<std::string, opt_integration_strategy> opt_strat {
      { "do_nothing", opt_integration_strategy::do_nothing},
      { "full_explicit", opt_integration_strategy::full_explicit},
      { "explicit_approximate_semi_implicit",
        opt_integration_strategy::explicit_approximate_semi_implicit},
      { "semi_implicit_Hybrids",
        opt_integration_strategy::semi_implicit_Hybrids},
      { "semi_implicit_HybridsJ",
        opt_integration_strategy::semi_implicit_HybridsJ}
    };

    auto get_solver = [&](std::string name, std::string default_method)
    {
      opt_integration_strategy ret;

      auto itr = opt_strat.find(GoA_str(name, default_method));
      if (itr != opt_strat.end())
      {
        ret = itr->second;
      }
      else
      {
        msg << "M1_solver/" << name << " unknown" << std::endl;
        ATHENA_ERROR(msg);
      }

      return ret;
    };

    std::string default_method = "full_explicit";

    opt_solver.solvers.non_stiff   = get_solver("solver_non_stiff",   default_method);
    opt_solver.solvers.stiff       = get_solver("solver_stiff",       default_method);
    opt_solver.solvers.scattering  = get_solver("solver_scattering",  default_method);
    opt_solver.solvers.equilibrium = get_solver("solver_equilibrium", default_method);

    opt_solver.solver_reduce_to_common = pin->GetOrAddBoolean(
      "M1_solver",
      "solver_reduce_to_common",
      false);

    opt_solver.solver_explicit_nG = pin->GetOrAddBoolean(
      "M1_solver",
      "solver_explicit_nG",
      false);

  }

  opt_solver.eps_a_tol     = GoA_Real("eps_tol",     1e-10);
  opt_solver.eps_r_tol     = GoA_Real("eps_tol",     1e-10);
  opt_solver.w_opt_ini   = GoA_Real("w_opt_init",  1.0);
  opt_solver.fac_err_amp = GoA_Real("fac_err_amp", 1.11);

  opt_solver.thick_tol = GoA_bool("thick_tol", false);
  opt_solver.thick_npg = GoA_bool("thick_npg", false);

  opt_solver.iter_max     = GoA_int("iter_max", 128);
  opt_solver.iter_max_rst = GoA_int("iter_max_rst", 10);

  opt_solver.use_Neighbor = GoA_bool("use_Neighbor",  false);

  opt_solver.src_lim = GoA_Real("src_lim", -1.0);

  opt_solver.limit_src_fluid = GoA_bool("limit_src_fluid",  false);
  opt_solver.limit_src_radiation = GoA_bool("limit_src_radiation",  false);

  opt_solver.use_Neighbor = GoA_bool("use_Neighbor",  false);

  opt_solver.src_lim_Ye_min = GoA_Real("src_lim_Ye_min", -1.0);
  opt_solver.src_lim_Ye_max = GoA_Real("src_lim_Ye_max", -1.0);

  opt_solver.src_lim_thick      = GoA_Real("src_lim_thick",      -1.0);
  opt_solver.src_lim_scattering = GoA_Real("src_lim_scattering", -1.0);

  opt_solver.limit_full_radiation = GoA_bool("limit_full_radiation", false);
  opt_solver.full_lim = GoA_Real("full_lim", -1.0);

  opt_solver.fb_rat_sl_E = GoA_Real("fb_rat_sl_E", -1.0);
  opt_solver.fb_rat_sl_F_d = GoA_Real("fb_rat_sl_F_d", -1.0);
  opt_solver.fb_rat_sl_nG = GoA_Real("fb_rat_sl_nG", -1.0);

  opt_solver.equilibrium_enforce = GoA_bool("equilibrium_enforce", false);
  opt_solver.equilibrium_initial = GoA_bool("equilibrium_initial", false);
  opt_solver.equilibrium_sources = GoA_bool("equilibrium_sources", false);
  opt_solver.equilibrium_evolve = GoA_bool("equilibrium_evolve", false);
  opt_solver.equilibrium_evolve_use_euler = GoA_bool(
    "equilibrium_evolve_use_euler", false
  );
  opt_solver.equilibrium_E_F_d = GoA_bool("equilibrium_E_F_d", false);
  opt_solver.equilibrium_n_nG = GoA_bool("equilibrium_n_nG", false);
  opt_solver.equilibrium_use_thick = GoA_bool("equilibrium_use_thick", false);

  opt_solver.equilibrium_src_nG = GoA_bool("equilibrium_src_nG", false);
  opt_solver.equilibrium_src_E_F_d = GoA_bool("equilibrium_src_E_F_d", false);
  opt_solver.equilibrium_use_diff_src = GoA_bool("equilibrium_use_diff_src",
                                                 false);

  opt_solver.equilibrium_zeta = GoA_Real("equilibrium_zeta", 0.0);

  opt_solver.eql_rho_min = GoA_Real("eql_rho_min", 0.0);
  opt_solver.tra_rho_min = GoA_Real(
    "tra_rho_min", opt_solver.eql_rho_min
  );

  opt_solver.flux_lo_fallback_tau_min = GoA_Real(
    "flux_lo_fallback_tau_min", -1
  );
  opt_solver.flux_lo_fallback_Ye_min = GoA_Real(
    "flux_lo_fallback_Ye_min", -1
  );
  opt_solver.flux_lo_fallback_Ye_max = GoA_Real(
    "flux_lo_fallback_Ye_max", -1
  );

  opt_solver.verbose = GoA_bool("verbose", false);


  // options related to solver (excision)
  opt_excision.excise_m1_damping =
      pin->GetOrAddBoolean("excision", "excise_m1_damping", false);

  opt_excision.m1_damping_factor =
    pin->GetOrAddReal("excision", "m1_damping_factor", 0.69);

  opt_excision.m1_disable_ahf_eql =
    pin->GetOrAddBoolean("excision", "m1_disable_ahf_eql", false);

  opt_excision.m1_disable_ahf_opac =
    pin->GetOrAddBoolean("excision", "m1_disable_ahf_opac", false);
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
    else if (tmp == "mixed")
    {
      opt.characteristics_variety = opt_characteristics_variety::mixed;
    }
    else if (tmp == "exact_thin")
    {
      opt.characteristics_variety = opt_characteristics_variety::exact_thin;
    }
    else if (tmp == "exact_thick")
    {
      opt.characteristics_variety = opt_characteristics_variety::exact_thick;
    }
    else if (tmp == "exact_closure")
    {
      opt.characteristics_variety = opt_characteristics_variety::exact_closure;
    }
    else
    {
      msg << "M1/characteristics_variety unknown" << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  { // flux style
    tmp = pin->GetOrAddString("M1", "flux_variety", "HybridizeMinModA");
    if (tmp == "LO")
    {
      opt.flux_variety = opt_flux_variety::LO;
    }
    else if (tmp == "HO")
    {
      opt.flux_variety = opt_flux_variety::HO;
    }
    else if (tmp == "HybridizeMinMod")
    {
      opt.flux_variety = opt_flux_variety::HybridizeMinMod;
    }
    else if (tmp == "HybridizeMinModA")
    {
      opt.flux_variety = opt_flux_variety::HybridizeMinModA;
    }
    else if (tmp == "HybridizeMinModB")
    {
      opt.flux_variety = opt_flux_variety::HybridizeMinModB;
    }
    else if (tmp == "HybridizeMinModC")
    {
      opt.flux_variety = opt_flux_variety::HybridizeMinModC;
    }
    else if (tmp == "HybridizeMinModD")
    {
      opt.flux_variety = opt_flux_variety::HybridizeMinModD;
    }
    else if (tmp == "HybridizeMinModE")
    {
      opt.flux_variety = opt_flux_variety::HybridizeMinModE;
    }
    else if (tmp == "RiemannHLLEmod")
    {
      opt.flux_variety = opt_flux_variety::RiemannHLLEmod;
    }
    else
    {
      msg << "M1/flux_variety unknown" << std::endl;
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
    opt.fl_E  = pin->GetOrAddReal("M1", "fl_E",    1e-15);
    opt.fl_J  = pin->GetOrAddReal("M1", "fl_J",    1e-15);
    opt.fl_nG = pin->GetOrAddReal("M1", "fl_nG",   1e-50);
    opt.fl_nF2 = pin->GetOrAddReal("M1", "fl_nF2", 1e-80);
    opt.eps_E = pin->GetOrAddReal("M1", "eps_E", 1e-5);
    opt.eps_J = pin->GetOrAddReal("M1", "eps_J", 1e-10);
    opt.enforce_causality = pin->GetOrAddBoolean(
      "M1", "enforce_causality", true);
    opt.enforce_finite = pin->GetOrAddBoolean(
      "M1", "enforce_finite", true);
    opt.eps_ec_fac = pin->GetOrAddReal("M1", "eps_ec_fac", 1e-15);

    opt.min_flux_Theta = pin->GetOrAddReal("M1", "min_flux_Theta", 0);
  }

  { // flux limiter / hybridization settings
    opt.flux_limiter_use_mask = pin->GetOrAddBoolean(
      "M1", "flux_limiter_use_mask", false);
    opt.flux_limiter_nn = pin->GetOrAddBoolean(
      "M1", "flux_limiter_nn", false);
    opt.flux_limiter_multicomponent = pin->GetOrAddBoolean(
      "M1", "flux_limiter_multicomponent", false);

    if (!opt.flux_limiter_use_mask)
    {
      if (opt.flux_limiter_multicomponent || opt.flux_limiter_nn)
      {
        msg << "M1/flux_limiter_multicomponent &  M1/flux_limiter_nn require";
        msg << "M1/flux_limiter_use_mask=true" << std::endl;
        ATHENA_ERROR(msg);
      }
    }

    opt.flux_lo_fallback_E = pin->GetOrAddBoolean(
      "M1", "flux_lo_fallback_E", false);
    opt.flux_lo_fallback_nG = pin->GetOrAddBoolean(
      "M1", "flux_lo_fallback_nG", false);

    opt.flux_lo_fallback = (opt.flux_lo_fallback_E || opt.flux_lo_fallback_nG);

    opt.flux_lo_fallback_eql_ho = pin->GetOrAddBoolean(
      "M1", "flux_lo_fallback_eql_ho", false);

    opt.flux_lo_fallback_first_stage = pin->GetOrAddBoolean(
      "M1", "flux_lo_fallback_first_stage", true);

    opt.flux_lo_fallback_species = pin->GetOrAddBoolean(
      "M1", "flux_lo_fallback_species", false);

    opt.flux_lo_fallback_mask_reset_all_stages = pin->GetOrAddBoolean(
      "M1", "flux_lo_fallback_mask_reset_all_stages", true);

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

  // BD: TODO- should remove this option entirely and assume true
  opt.retain_equilibrium = pin->GetOrAddBoolean("M1",
                                                "retain_equilibrium",
                                                true);

  // debugging
  opt.value_inject = pin->GetOrAddBoolean("problem", "value_inject", false);

  opt.disable_fluxes = pin->GetOrAddBoolean("M1", "disable_fluxes", false);

  // fix issues
  opt.zero_fix_sources = pin->GetOrAddBoolean("M1", "zero_fix_sources", true);
}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//