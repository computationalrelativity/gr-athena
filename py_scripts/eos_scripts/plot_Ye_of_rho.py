import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ============================================================
# 1. Read the Y_e(rho) table
# ============================================================

filename = "/home/mi58rip/gr-athena/eos_tables/ye_of_rho_luis.dat"

data = np.loadtxt(filename)

log10_rho_table = data[:, 0]
rho_table = 10.0**log10_rho_table
ye_table = data[:, 1]


# ============================================================
# 2. Liebendörfer Y_e(rho) parameterization
#
#    Parameters that are FIXED by the C++ implementation:
#
#       Ye_1       = 0.5
#       log10_rhoH = 15
#
#    Parameters to fit:
#
#       log10_rho1
#       log10_rho2
#       Ye_2
#       Ye_c
#       Ye_H
# ============================================================

def Ye_of_rho(
    rho,
    log10_rho1,
    log10_rho2,
    Ye_2,
    Ye_c,
    Ye_H,
    Ye_1=0.5,
    log10_rhoH=15.0,
):

    log10_rho = np.log10(rho)

    # --------------------------------------------------------
    # x parameter
    # --------------------------------------------------------

    x = (
        2.0 * log10_rho
        - log10_rho2
        - log10_rho1
    ) / (log10_rho2 - log10_rho1)

    x = np.clip(x, -1.0, 1.0)

    abs_x = np.abs(x)

    # --------------------------------------------------------
    # Low/intermediate density branch
    # --------------------------------------------------------

    Ye_low = (
        0.5 * (Ye_2 + Ye_1)
        + 0.5 * x * (Ye_2 - Ye_1)
        + Ye_c
        * (
            1.0
            - abs_x
            + 4.0 * abs_x
            * (abs_x - 0.5)
            * (abs_x - 1.0)
        )
    )

    # --------------------------------------------------------
    # High-density linear branch
    # --------------------------------------------------------

    m = (
        Ye_H - Ye_2
    ) / (
        log10_rhoH - log10_rho2
    )

    Ye_high = (
        Ye_2
        + m * (log10_rho - log10_rho2)
    )

    # --------------------------------------------------------
    # Piecewise function
    # --------------------------------------------------------

    return np.where(
        log10_rho > log10_rho2,
        Ye_high,
        Ye_low,
    )


# ============================================================
# 3. Residual function for least_squares
#
#    We parameterize rho2 as:
#
#       log10_rho2 = log10_rho1 + delta
#
#    with delta > 0.
#
#    This guarantees rho2 > rho1.
# ============================================================

def residuals(params, log10_rho, ye):

    log10_rho1, delta, Ye_2, Ye_c, Ye_H = params

    log10_rho2 = log10_rho1 + delta

    rho = 10.0**log10_rho

    ye_fit = Ye_of_rho(
        rho,
        log10_rho1,
        log10_rho2,
        Ye_2,
        Ye_c,
        Ye_H,
    )

    return ye_fit - ye


# ============================================================
# 4. Initial guess
# ============================================================

initial_guess = np.array([
    7.700,              # log10_rho1
    13.200 - 7.700,     # delta = log10_rho2 - log10_rho1
    0.285,              # Ye_2
    0.035,              # Ye_c
    0.265,              # Ye_H
])


# ============================================================
# 5. Physical bounds
#
#    delta > 0 guarantees rho2 > rho1.
# ============================================================

lower_bounds = np.array([
    5.0,        # log10_rho1
    0.1,        # delta
    0.0,        # Ye_2
    0.0,        # Ye_c
    0.0,        # Ye_H
])

upper_bounds = np.array([
    14.0,       # log10_rho1
    9.0,        # delta
    0.5,        # Ye_2
    0.5,        # Ye_c
    0.5,        # Ye_H
])


# ============================================================
# 6. Perform nonlinear least-squares fit
# ============================================================

result = least_squares(
    residuals,
    initial_guess,
    bounds=(lower_bounds, upper_bounds),
    args=(log10_rho_table, ye_table),

    # High accuracy
    xtol=1e-14,
    ftol=1e-14,
    gtol=1e-14,

    # More robust numerical differentiation
    diff_step=1e-5,

    # Let optimizer work sufficiently long
    max_nfev=100000,

    verbose=1,
)


# ============================================================
# 7. Extract fitted parameters
# ============================================================

log10_rho1_fit = result.x[0]
delta_fit       = result.x[1]
Ye_2_fit        = result.x[2]
Ye_c_fit        = result.x[3]
Ye_H_fit        = result.x[4]

log10_rho2_fit = log10_rho1_fit + delta_fit


# ============================================================
# 8. Calculate fitted curve
# ============================================================

ye_best = Ye_of_rho(
    rho_table,
    log10_rho1_fit,
    log10_rho2_fit,
    Ye_2_fit,
    Ye_c_fit,
    Ye_H_fit,
)


# Initial guess curve
ye_initial = Ye_of_rho(
    rho_table,
    initial_guess[0],
    initial_guess[0] + initial_guess[1],
    initial_guess[2],
    initial_guess[3],
    initial_guess[4],
)


# ============================================================
# 9. Error diagnostics
# ============================================================

residual = ye_table - ye_best

SSE = np.sum(residual**2)
MSE = np.mean(residual**2)
RMSE = np.sqrt(MSE)
MAE = np.mean(np.abs(residual))
MAX_ERROR = np.max(np.abs(residual))


print()
print("=" * 70)
print("FIT RESULTS")
print("=" * 70)

print(f"log10_rho1 = {log10_rho1_fit:.10f}")
print(f"log10_rho2 = {log10_rho2_fit:.10f}")
print(f"Ye_2       = {Ye_2_fit:.10f}")
print(f"Ye_c       = {Ye_c_fit:.10f}")
print(f"Ye_H       = {Ye_H_fit:.10f}")

print()
print("Fixed parameters:")
print(f"Ye_1       = 0.5")
print(f"log10_rhoH = 15.0")

print()
print("Fit diagnostics:")
print(f"SSE        = {SSE:.12e}")
print(f"MSE        = {MSE:.12e}")
print(f"RMSE       = {RMSE:.12e}")
print(f"MAE        = {MAE:.12e}")
print(f"Max error  = {MAX_ERROR:.12e}")

print()
print("Optimizer:")
print(f"Success    = {result.success}")
print(f"Message    = {result.message}")
print(f"NFEV       = {result.nfev}")
print("=" * 70)


# ============================================================
# 10. Plot Y_e(rho)
# ============================================================

fig, ax = plt.subplots(figsize=(5, 4))

ax.plot(
    rho_table,
    ye_table,
    color="navy",
    linewidth=1.2,
    label=r"$Y_e$ table",
)

ax.plot(
    rho_table,
    ye_best,
    color="darkgreen",
    linewidth=1.2,
    linestyle="--",
    label=r"$Y_e$ fitted Liebendörfer",
)

ax.plot(
    rho_table,
    ye_initial,
    color="black",
    linewidth=1.0,
    linestyle="-.",
    label=r"$Y_e$ initial guess",
)

ax.set_xscale("log")

ax.set_xlabel(r"$\rho$ (g/cm$^3$)", fontsize=12)
ax.set_ylabel(r"$Y_e$", fontsize=12)

ax.grid(True, linestyle="--", alpha=0.7)

ax.minorticks_on()

ax.tick_params(
    axis="both",
    which="both",
    direction="in",
    top=True,
    right=False,
)

ax.legend(fontsize=9, loc="best")

plt.tight_layout()

plt.savefig(
    "/home/mi58rip/gr-athena/plots/eos/"
    "ye_vs_rho_table_vs_analytical.png",
    dpi=600,
)

plt.close(fig)


# ============================================================
# 11. Plot residual
# ============================================================

fig, ax = plt.subplots(figsize=(5, 4))

ax.plot(
    rho_table,
    residual,
    linewidth=1.0,
)

ax.axhline(
    0.0,
    linestyle="--",
)

ax.set_xscale("log")

ax.set_xlabel(r"$\rho$ (g/cm$^3$)", fontsize=12)
ax.set_ylabel(
    r"$Y_e^{\rm table}-Y_e^{\rm fit}$",
    fontsize=12,
)

ax.grid(True, linestyle="--", alpha=0.7)

ax.minorticks_on()

ax.tick_params(
    axis="both",
    which="both",
    direction="in",
    top=True,
    right=False,
)

plt.tight_layout()

plt.savefig(
    "/home/mi58rip/gr-athena/plots/eos/"
    "ye_vs_rho_fit_residual.png",
    dpi=600,
)

plt.close(fig)