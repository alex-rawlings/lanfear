#pragma once

// Orbit integration in a fixed analytical potential. Each orbit is independent,
// so the batch routine is a simple OpenMP loop over orbits; MPI decomposition
// across ranks happens one level up, in python (mpi4py). The C++ layer knows
// nothing about MPI.
//
// For >1e6 particles we cannot keep every trajectory in memory, so the batch
// routine streams per-orbit summary statistics (energy conservation, radial and
// per-axis extents, angular-momentum behaviour) and discards the trajectory.
// Full trajectories are available for individual orbits via integrate_orbit(),
// for plotting and for developing the later FFT/classification stages.

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <boost/numeric/odeint.hpp>

#include "scf_potential.hpp"

namespace lanfear {

using OrbitState = std::array<double, 6>;  // (x, y, z, vx, vy, vz), HO units

// Per-orbit summary. The flattened column order is defined by
// summary_columns() / write_summary() below and mirrored in python.
struct OrbitSummary {
    double status = 0;       // 0 ok, 1 period estimate failed, 2 NaN encountered
    double period = 0;       // estimated orbital period (sets the time unit)
    double t_total = 0;      // total integration time = n_periods * period
    double energy0 = 0;      // initial specific energy 0.5 v^2 + Phi
    double energy_mean = 0;
    double energy_drift = 0; // max |E - E0| / |E0| over samples
    double r_min = 0, r_max = 0, r_mean = 0;
    double x_abs_max = 0, y_abs_max = 0, z_abs_max = 0;   // box semi-axes
    double Lx_mean = 0, Ly_mean = 0, Lz_mean = 0;
    double Lx_abs_mean = 0, Ly_abs_mean = 0, Lz_abs_mean = 0;
    double Lx_sign_changes = 0, Ly_sign_changes = 0, Lz_sign_changes = 0;
    // Minimum distance from each principal axis (the "tube hole"): a tube orbit
    // circulating about axis a keeps rho_a_min > 0, while a box passes near it.
    double rho_x_min = 0, rho_y_min = 0, rho_z_min = 0;
    // Shape (second-moment) tensor <x_i x_j>, time-averaged over the orbit. Its
    // smallest eigenvalue vanishes for a planar orbit (rosette) in any
    // orientation; the eigenvalue ordering encodes the orbit's shape.
    double Sxx = 0, Syy = 0, Szz = 0, Sxy = 0, Sxz = 0, Syz = 0;
};

constexpr std::size_t kSummaryCols = 30;

inline const char* const* summary_columns() {
    static const char* const cols[kSummaryCols] = {
        "status",     "period",      "t_total",     "energy0",
        "energy_mean", "energy_drift", "r_min",       "r_max",
        "r_mean",     "x_abs_max",   "y_abs_max",   "z_abs_max",
        "Lx_mean",    "Ly_mean",     "Lz_mean",     "Lx_abs_mean",
        "Ly_abs_mean", "Lz_abs_mean", "Lx_sign_changes", "Ly_sign_changes",
        "Lz_sign_changes", "rho_x_min", "rho_y_min", "rho_z_min",
        "Sxx", "Syy", "Szz", "Sxy", "Sxz", "Syz"};
    return cols;
}

inline void write_summary(const OrbitSummary& s, double* out) {
    out[0] = s.status;        out[1] = s.period;       out[2] = s.t_total;
    out[3] = s.energy0;       out[4] = s.energy_mean;  out[5] = s.energy_drift;
    out[6] = s.r_min;         out[7] = s.r_max;        out[8] = s.r_mean;
    out[9] = s.x_abs_max;     out[10] = s.y_abs_max;   out[11] = s.z_abs_max;
    out[12] = s.Lx_mean;      out[13] = s.Ly_mean;     out[14] = s.Lz_mean;
    out[15] = s.Lx_abs_mean;  out[16] = s.Ly_abs_mean; out[17] = s.Lz_abs_mean;
    out[18] = s.Lx_sign_changes; out[19] = s.Ly_sign_changes;
    out[20] = s.Lz_sign_changes;
    out[21] = s.rho_x_min;    out[22] = s.rho_y_min;   out[23] = s.rho_z_min;
    out[24] = s.Sxx; out[25] = s.Syy; out[26] = s.Szz;
    out[27] = s.Sxy; out[28] = s.Sxz; out[29] = s.Syz;
}

// Local circular period at the initial radius: T = 2*pi / sqrt(a_r / r), where
// a_r is the inward radial acceleration. Returns 0 if the point is unbound
// (outward net radial force) or at the origin. `Pot` is any type exposing
// potential(x,y,z) and acceleration(x,y,z) (SCFPotential, DiscPotential, ...).
template <class Pot>
inline double estimate_period(const Pot& pot, const OrbitState& s) {
    const double r = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    if (r <= 0.0) return 0.0;
    const auto a = pot.acceleration(s[0], s[1], s[2]);
    const double a_radial = -(a[0] * s[0] + a[1] * s[1] + a[2] * s[2]) / r;
    if (a_radial <= 0.0) return 0.0;
    return 2.0 * M_PI / std::sqrt(a_radial / r);
}

namespace detail {

// Equations of motion: dx/dt = v, dv/dt = a(x). Freezes on NaN so odeint cannot
// spin on a diverged orbit.
template <class Pot>
struct EquationsOfMotion {
    const Pot& pot;
    bool nan_hit = false;
    void operator()(const OrbitState& s, OrbitState& dsdt, double /*t*/) {
        if (std::isnan(s[0]) || std::isnan(s[1]) || std::isnan(s[2])) {
            nan_hit = true;
            dsdt.fill(0.0);
            return;
        }
        dsdt[0] = s[3];
        dsdt[1] = s[4];
        dsdt[2] = s[5];
        const auto a = pot.acceleration(s[0], s[1], s[2]);
        dsdt[3] = a[0];
        dsdt[4] = a[1];
        dsdt[5] = a[2];
    }
};

// Streaming accumulator over the sampled states.
template <class Pot>
struct Accumulator {
    const Pot& pot;
    OrbitSummary s;
    std::vector<double>* trajectory;  // optional (x,y,z,vx,vy,vz) per sample
    std::size_t n = 0;
    double e_sum = 0;
    double lx_sum = 0, ly_sum = 0, lz_sum = 0;
    double lx_abs_sum = 0, ly_abs_sum = 0, lz_abs_sum = 0;
    int lx_sign = 0, ly_sign = 0, lz_sign = 0;

    static int sgn(double v) { return (v > 0) - (v < 0); }
    void update_sign(double v, int& prev, double& changes) {
        const int cur = sgn(v);
        if (cur != 0) {
            if (prev != 0 && cur != prev) changes += 1;
            prev = cur;
        }
    }

    void operator()(const OrbitState& st, double /*t*/) {
        const double x = st[0], y = st[1], z = st[2];
        const double vx = st[3], vy = st[4], vz = st[5];
        const double r = std::sqrt(x * x + y * y + z * z);
        const double v2 = vx * vx + vy * vy + vz * vz;
        const double e = 0.5 * v2 + pot.potential(x, y, z);
        const double Lx = y * vz - z * vy;
        const double Ly = z * vx - x * vz;
        const double Lz = x * vy - y * vx;
        // Distance from each principal axis.
        const double rho_x = std::sqrt(y * y + z * z);
        const double rho_y = std::sqrt(x * x + z * z);
        const double rho_z = std::sqrt(x * x + y * y);

        if (n == 0) {
            s.energy0 = e;
            s.r_min = r;
            s.r_max = r;
            s.rho_x_min = rho_x;
            s.rho_y_min = rho_y;
            s.rho_z_min = rho_z;
        } else {
            s.r_min = std::min(s.r_min, r);
            s.r_max = std::max(s.r_max, r);
            s.rho_x_min = std::min(s.rho_x_min, rho_x);
            s.rho_y_min = std::min(s.rho_y_min, rho_y);
            s.rho_z_min = std::min(s.rho_z_min, rho_z);
        }
        s.r_mean += r;
        e_sum += e;
        s.energy_drift = std::max(s.energy_drift,
                                  std::abs(e - s.energy0) /
                                      (std::abs(s.energy0) + 1e-300));
        s.x_abs_max = std::max(s.x_abs_max, std::abs(x));
        s.y_abs_max = std::max(s.y_abs_max, std::abs(y));
        s.z_abs_max = std::max(s.z_abs_max, std::abs(z));
        lx_sum += Lx; ly_sum += Ly; lz_sum += Lz;
        lx_abs_sum += std::abs(Lx);
        ly_abs_sum += std::abs(Ly);
        lz_abs_sum += std::abs(Lz);
        s.Sxx += x * x; s.Syy += y * y; s.Szz += z * z;
        s.Sxy += x * y; s.Sxz += x * z; s.Syz += y * z;
        update_sign(Lx, lx_sign, s.Lx_sign_changes);
        update_sign(Ly, ly_sign, s.Ly_sign_changes);
        update_sign(Lz, lz_sign, s.Lz_sign_changes);

        if (trajectory) {
            trajectory->insert(trajectory->end(),
                               {x, y, z, vx, vy, vz});
        }
        ++n;
    }

    void finalise() {
        if (n == 0) return;
        const double inv = 1.0 / static_cast<double>(n);
        s.r_mean *= inv;
        s.energy_mean = e_sum * inv;
        s.Lx_mean = lx_sum * inv; s.Ly_mean = ly_sum * inv; s.Lz_mean = lz_sum * inv;
        s.Lx_abs_mean = lx_abs_sum * inv;
        s.Ly_abs_mean = ly_abs_sum * inv;
        s.Lz_abs_mean = lz_abs_sum * inv;
        s.Sxx *= inv; s.Syy *= inv; s.Szz *= inv;
        s.Sxy *= inv; s.Sxz *= inv; s.Syz *= inv;
    }
};

}  // namespace detail

// Integrate a single orbit for n_periods estimated periods, sampling n_samples
// uniformly spaced points. If `trajectory` is non-null it is filled with the
// (n_samples x 6) states, row-major.
template <class Pot>
inline OrbitSummary integrate_orbit(const Pot& pot, OrbitState state,
                                    int n_periods, int n_samples,
                                    double abs_tol, double rel_tol,
                                    std::vector<double>* trajectory = nullptr) {
    // Nudge exact zeros off the coordinate axes / origin.
    for (double& c : state)
        if (c == 0.0) c = 1e-12;

    OrbitSummary summary;
    const double T = estimate_period(pot, state);
    summary.period = T;
    summary.t_total = n_periods * T;
    if (!(T > 0.0) || n_samples < 2) {
        summary.status = 1;
        return summary;
    }
    const double dt_out = summary.t_total / (n_samples - 1);

    namespace ode = boost::numeric::odeint;
    auto stepper = ode::make_dense_output(
        abs_tol, rel_tol, ode::runge_kutta_dopri5<OrbitState>());
    detail::EquationsOfMotion<Pot> sys{pot};
    detail::Accumulator<Pot> acc{pot, summary, trajectory};
    if (trajectory) {
        trajectory->clear();
        trajectory->reserve(static_cast<std::size_t>(n_samples) * 6);
    }

    try {
        ode::integrate_const(stepper, std::ref(sys), state, 0.0,
                             summary.t_total, dt_out, std::ref(acc));
    } catch (...) {
        acc.s.status = 2;
    }
    acc.finalise();
    OrbitSummary out = acc.s;
    if (sys.nan_hit || std::isnan(out.energy_mean)) out.status = 2;
    return out;
}

// Integrate a batch of orbits (OpenMP over orbits). `states` is n_orbits x 6
// row-major (HO units); `out_summary` is n_orbits x kSummaryCols row-major.
template <class Pot>
inline void integrate_batch(const Pot& pot, const double* states,
                            std::size_t n_orbits, int n_periods, int n_samples,
                            double abs_tol, double rel_tol,
                            double* out_summary) {
    #pragma omp parallel for schedule(dynamic, 8)
    for (std::size_t i = 0; i < n_orbits; ++i) {
        OrbitState s;
        for (int j = 0; j < 6; ++j) s[j] = states[i * 6 + j];
        const OrbitSummary summary = integrate_orbit(
            pot, s, n_periods, n_samples, abs_tol, rel_tol, nullptr);
        write_summary(summary, out_summary + i * kSummaryCols);
    }
}

}  // namespace lanfear
