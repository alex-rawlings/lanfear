#pragma once

// Orbit integration in an analytical potential that is either static or
// rotates rigidly (figure rotation) at a constant pattern speed Omega. Each
// orbit is independent, so the batch routine is a simple OpenMP loop over
// orbits; MPI decomposition across ranks happens one level up, in python
// (mpi4py). The C++ layer knows nothing about MPI.
//
// Orbits are always integrated in the inertial frame. A rotating figure is
// represented by rotating the potential itself, Phi(x, t) = Phi_body(R(t)^T x),
// where R(t) turns by |Omega| t about the fixed axis Omega / |Omega| and the
// body (figure) frame coincides with the inertial frame at t = 0. There are
// therefore no fictitious (Coriolis/centrifugal) forces in the equations of
// motion; their effect appears in the body-frame view of the orbit. With
// Omega = 0 the rotation is skipped entirely and the integration is exactly
// that of the static potential.
//
// The per-orbit diagnostics are measured in the co-rotating body frame, where
// the potential is static and a regular orbit is a steady 3-torus: positions
// x_b = R^T x and co-rotating velocities v_rot = dx_b/dt = R^T v - Omega x x_b.
// The conserved quantity is the Jacobi integral E_J = E - Omega . L, which
// reduces to the energy E for a static potential.
//
// For >1e6 particles we cannot keep every trajectory in memory, so the batch
// routine streams per-orbit summary statistics (Jacobi-integral conservation,
// radial and per-axis extents, angular-momentum behaviour) and discards the
// trajectory. Full trajectories are available for individual orbits via
// integrate_orbit(), for plotting and for developing the later
// FFT/classification stages.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <vector>

#include <boost/numeric/odeint.hpp>

#include "scf_potential.hpp"

namespace lanfear {

// Report batch-integration progress to the console at every 10% of orbits
// completed. `completed` is a shared atomic counter incremented once per finished
// orbit (so it stays correct under OpenMP); call this exactly once per orbit.
// Because the counter values are unique, each 10% boundary is crossed by exactly
// one thread, so at most ten lines ("10% ... 100% of particles integrated") are
// printed with no duplicates.
inline void report_orbit_progress(std::atomic<std::size_t>& completed,
                                  std::size_t n_orbits) {
    const std::size_t done = completed.fetch_add(1) + 1;
    const int decile = static_cast<int>((done * 10) / n_orbits);
    const int prev_decile = static_cast<int>(((done - 1) * 10) / n_orbits);
    if (decile != prev_decile) {
        #pragma omp critical(lanfear_progress)
        {
            std::printf("%d%% of particles integrated\n", decile * 10);
            std::fflush(stdout);
        }
    }
}

using OrbitState = std::array<double, 6>;  // (x, y, z, vx, vy, vz), HO units
using Vec3 = std::array<double, 3>;

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

inline double dot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Rigid rotation of the potential (the figure) at constant pattern speed
// `omega` (angular-velocity vector, HO units). The body frame coincides with
// the inertial frame at t = 0 and turns by |omega| t about the fixed axis
// omega / |omega|, so omega has the same components in both frames. A zero
// omega is not rotating: to_body/to_inertial then return their input
// unchanged, so the static case is reproduced exactly.
struct FigureRotation {
    Vec3 omega{0.0, 0.0, 0.0};
    Vec3 axis{0.0, 0.0, 1.0};  // unit rotation axis (unused when not rotating)
    double rate = 0.0;         // |omega|
    bool rotating = false;

    FigureRotation() = default;
    explicit FigureRotation(const Vec3& w) : omega(w) {
        rate = std::sqrt(dot(w, w));
        rotating = rate > 0.0;
        if (rotating) axis = {w[0] / rate, w[1] / rate, w[2] / rate};
    }

    // Rodrigues rotation of v by angle theta about `axis`.
    Vec3 rotate(const Vec3& v, double theta) const {
        const double c = std::cos(theta), s = std::sin(theta);
        const Vec3 nxv = cross(axis, v);
        const double nv = dot(axis, v) * (1.0 - c);
        return {v[0] * c + nxv[0] * s + axis[0] * nv,
                v[1] * c + nxv[1] * s + axis[1] * nv,
                v[2] * c + nxv[2] * s + axis[2] * nv};
    }
    // Inertial -> body components at time t (R(t)^T v).
    Vec3 to_body(const Vec3& v, double t) const {
        return rotating ? rotate(v, -rate * t) : v;
    }
    // Body -> inertial components at time t (R(t) v).
    Vec3 to_inertial(const Vec3& v, double t) const {
        return rotating ? rotate(v, rate * t) : v;
    }
};

// Per-orbit summary. The flattened column order is defined by
// summary_columns() / write_summary() below and mirrored in python. Every
// quantity is measured in the co-rotating body frame (see the header comment);
// for a static potential that is simply the inertial frame.
struct OrbitSummary {
    double status = 0;       // 0 ok, 1 period estimate failed, 2 NaN encountered
    double period = 0;       // estimated orbital period (sets the time unit)
    double t_total = 0;      // total integration time = n_periods * period * factor
    // Initial Jacobi integral E_J = 0.5 v^2 + Phi - Omega . L (inertial v, L);
    // the specific energy 0.5 v^2 + Phi for a static potential.
    double energy0 = 0;
    double energy_mean = 0;
    double energy_drift = 0; // max |E_J - E_J0| / |E_J0| over samples
    double r_min = 0, r_max = 0, r_mean = 0;
    double x_abs_max = 0, y_abs_max = 0, z_abs_max = 0;   // box semi-axes
    // Angular momentum x_b x v_rot in the co-rotating frame (the inertial
    // angular momentum for a static potential).
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
    // Inner/outer x-tube morphology (Frigo et al. 2021; orbit-analysis). At the
    // orbit's z=0 crossings, compare the max |y| in an |x| centre strip vs an
    // |x| border strip. The ratio y_centre_max / y_border_max is < 1 for an
    // inner x-tube (pinched waist: widest in y at the x-ends) and >= 1 for an
    // outer x-tube (widest at the centre). Set to a large value when the border
    // strip has no crossings, so it reads as outer by default.
    double x_tube_ratio = 0;
    // Pericentre estimate: the minimum radius over every right-hand-side
    // evaluation of the integrator (each adaptive step and its stages), not
    // just the output samples. r_min is the minimum over the (uniformly spaced)
    // samples, which can step straight over a fast pericentre passage of a
    // plunging orbit; r_peri resolves it because the adaptive stepper shortens
    // its steps where the orbit is fastest. Used to flag orbits that reach a
    // central black-hole binary.
    double r_peri = 0;
    // Estimated orbital period in the co-rotating body frame of a figure
    // rotating at |Omega_p|: 2 pi / |Omega_c - |Omega_p||, with Omega_c =
    // 2 pi / period the local circular frequency (the worst case: a prograde
    // circular orbit, which the frame slows down). It diverges at corotation.
    // Equal to `period` for a static potential. The integration is lengthened
    // so that it spans n_periods of the longer of the two periods (see
    // body_period_factor()).
    double body_period = 0;
};

constexpr std::size_t kSummaryCols = 33;

// Default cap on how much longer than n_periods inertial periods an orbit is
// integrated to cover n_periods body-frame periods (see body_period_factor()).
constexpr int kMaxBodyPeriodFactor = 8;

inline const char* const* summary_columns() {
    static const char* const cols[kSummaryCols] = {
        "status",     "period",      "t_total",     "energy0",
        "energy_mean", "energy_drift", "r_min",       "r_max",
        "r_mean",     "x_abs_max",   "y_abs_max",   "z_abs_max",
        "Lx_mean",    "Ly_mean",     "Lz_mean",     "Lx_abs_mean",
        "Ly_abs_mean", "Lz_abs_mean", "Lx_sign_changes", "Ly_sign_changes",
        "Lz_sign_changes", "rho_x_min", "rho_y_min", "rho_z_min",
        "Sxx", "Syy", "Szz", "Sxy", "Sxz", "Syz", "x_tube_ratio",
        "r_peri", "body_period"};
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
    out[30] = s.x_tube_ratio;
    out[31] = s.r_peri;
    out[32] = s.body_period;
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

// Body-frame period 2 pi / |Omega_c - |Omega_p|| for an inertial period
// `period` (Omega_c = 2 pi / period) and pattern speed magnitude `omega_p`;
// `period` itself when static, infinite exactly at corotation.
inline double body_frame_period(double period, double omega_p) {
    if (!(omega_p > 0.0)) return period;
    const double rate = std::abs(2.0 * M_PI / period - omega_p);
    return rate > 0.0 ? 2.0 * M_PI / rate : std::numeric_limits<double>::infinity();
}

// Power-of-two factor f (1 <= f <= max_factor) by which an orbit's integration
// is lengthened, and its sample count raised, so that the window spans about
// n_periods body-frame periods: the power of two nearest (in log) to
// body_period / period, so the window covers at least n_periods / sqrt(2)
// body-frame periods unless capped. Near corotation the body-frame frequencies
// of the orbit tend to zero, and a window of n_periods inertial periods would
// cover too few body-frame cycles for the frequency analysis to resolve them.
// (Rounding up instead would double every prograde orbit's integration, as
// body_period / period = 1 / (1 - epsilon) exceeds 1 for any rotation.) Powers
// of two keep the sampling interval unchanged and the sample count a power of
// two (the FFT uses the largest power-of-two prefix). max_factor <= 1 disables
// the lengthening.
inline int body_period_factor(double period, double body_period, int max_factor) {
    int factor = 1;
    while (factor < max_factor && factor * M_SQRT2 * period < body_period)
        factor *= 2;
    return std::min(factor, std::max(max_factor, 1));
}

namespace detail {

// Inertial-frame equations of motion: dx/dt = v, dv/dt = a(x, t), with
// a(x, t) = R(t) a_body(R(t)^T x) for a rotating figure (just a_body(x) when
// static). Freezes on NaN so odeint cannot spin on a diverged orbit. Also
// records the minimum radius over every evaluation (the pericentre estimate
// OrbitSummary::r_peri; the radius is the same in either frame).
template <class Pot>
struct EquationsOfMotion {
    const Pot& pot;
    const FigureRotation& rotation;
    bool nan_hit = false;
    double r_eval_min = std::numeric_limits<double>::infinity();
    void operator()(const OrbitState& s, OrbitState& dsdt, double t) {
        if (std::isnan(s[0]) || std::isnan(s[1]) || std::isnan(s[2])) {
            nan_hit = true;
            dsdt.fill(0.0);
            return;
        }
        r_eval_min = std::min(
            r_eval_min, std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]));
        dsdt[0] = s[3];
        dsdt[1] = s[4];
        dsdt[2] = s[5];
        if (rotation.rotating) {
            const Vec3 xb = rotation.to_body({s[0], s[1], s[2]}, t);
            const auto ab = pot.acceleration(xb[0], xb[1], xb[2]);
            const Vec3 a = rotation.to_inertial({ab[0], ab[1], ab[2]}, t);
            dsdt[3] = a[0];
            dsdt[4] = a[1];
            dsdt[5] = a[2];
            return;
        }
        const auto a = pot.acceleration(s[0], s[1], s[2]);
        dsdt[3] = a[0];
        dsdt[4] = a[1];
        dsdt[5] = a[2];
    }
};

// Streaming accumulator over the sampled (inertial) states. Each sample is
// first transformed to the co-rotating body frame, where every diagnostic is
// measured and the optional trajectory is recorded.
template <class Pot>
struct Accumulator {
    const Pot& pot;
    const FigureRotation& rotation;
    OrbitSummary s;
    std::vector<double>* trajectory;  // optional (x,y,z,vx,vy,vz) per sample
    std::size_t n = 0;
    double e_sum = 0;
    double lx_sum = 0, ly_sum = 0, lz_sum = 0;
    double lx_abs_sum = 0, ly_abs_sum = 0, lz_abs_sum = 0;
    int lx_sign = 0, ly_sign = 0, lz_sign = 0;
    // Inner/outer x-tube morphology: (|x|, |y|) at each z=0 crossing, linearly
    // interpolated to z=0 and reduced in finalise(). The previous sample is kept
    // so the crossing can be interpolated; the crossings are stored only
    // transiently for the orbit being integrated.
    double prev_x = 0, prev_y = 0, prev_z = 0;
    bool have_prev_z = false;
    std::vector<double> zc_abs_x, zc_abs_y;

    static int sgn(double v) { return (v > 0) - (v < 0); }
    void update_sign(double v, int& prev, double& changes) {
        const int cur = sgn(v);
        if (cur != 0) {
            if (prev != 0 && cur != prev) changes += 1;
            prev = cur;
        }
    }

    void operator()(const OrbitState& st, double t) {
        double x = st[0], y = st[1], z = st[2];
        double vx = st[3], vy = st[4], vz = st[5];
        // Jacobi integral 0.5 v^2 + Phi_body(x_b) - Omega . (x x v), from the
        // inertial velocity (any frame's components give the same scalars).
        double jacobi_term = 0.0;
        if (rotation.rotating) {
            const Vec3 xb = rotation.to_body({x, y, z}, t);
            const Vec3 vb = rotation.to_body({vx, vy, vz}, t);
            jacobi_term = dot(rotation.omega, cross(xb, vb));
            const Vec3 wx = cross(rotation.omega, xb);
            x = xb[0]; y = xb[1]; z = xb[2];
            vx = vb[0] - wx[0]; vy = vb[1] - wx[1]; vz = vb[2] - wx[2];
        }
        const double r = std::sqrt(x * x + y * y + z * z);
        // |v_inertial|^2, from the inertial sample (unchanged by the transform).
        const double v2 = st[3] * st[3] + st[4] * st[4] + st[5] * st[5];
        double e = 0.5 * v2 + pot.potential(x, y, z);
        if (rotation.rotating) e -= jacobi_term;
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

        // Record (|x|, |y|) at z=0 crossings for the inner/outer x-tube test,
        // linearly interpolating between the bracketing samples to the exact
        // z=0 point (more robust than taking the post-crossing sample).
        if (have_prev_z && prev_z * z < 0.0) {
            const double t = prev_z / (prev_z - z);  // fraction of the step to z=0
            zc_abs_x.push_back(std::abs(prev_x + t * (x - prev_x)));
            zc_abs_y.push_back(std::abs(prev_y + t * (y - prev_y)));
        }
        prev_x = x;
        prev_y = y;
        prev_z = z;
        have_prev_z = true;

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

        // Inner/outer x-tube morphology (orbit-analysis _is_inner_x_tube): at the
        // z=0 crossings, compare the peak |y| in an |x| centre strip against a
        // border strip near the x-extremes. Inner tubes are pinched at the waist
        // (wider in y at the ends -> ratio < 1); outer tubes are widest at the
        // centre (ratio >= 1). No border data -> large ratio (reads as outer).
        double x_max = 0.0;
        for (double ax : zc_abs_x) x_max = std::max(x_max, ax);
        double y_centre_max = 0.0, y_border_max = 0.0;
        if (x_max > 0.0) {
            const double x_centre = x_max / 5.0;
            const double x_border = x_max - x_centre;
            for (std::size_t k = 0; k < zc_abs_x.size(); ++k) {
                if (zc_abs_x[k] < x_centre)
                    y_centre_max = std::max(y_centre_max, zc_abs_y[k]);
                else if (zc_abs_x[k] > x_border)
                    y_border_max = std::max(y_border_max, zc_abs_y[k]);
            }
        }
        s.x_tube_ratio = (y_border_max > 0.0) ? (y_centre_max / y_border_max) : 1e30;
    }
};

}  // namespace detail

// Integrate a single orbit for n_periods estimated periods, sampling n_samples
// uniformly spaced points. `state` is the inertial initial state; the figure
// rotates at `pattern_speed` (HO units; zero for a static potential). For a
// rotating figure the window is lengthened by body_period_factor() (at most
// max_body_period_factor) so that it also spans n_periods body-frame periods,
// with n_samples raised by the same factor; t_total records the actual length.
// If `trajectory` is non-null it is filled with the (n_samples x 6, after any
// lengthening) co-rotating body-frame states (x_b, v_rot), row-major -- the
// inertial states when static.
template <class Pot>
inline OrbitSummary integrate_orbit(const Pot& pot, OrbitState state,
                                    int n_periods, int n_samples,
                                    double abs_tol, double rel_tol,
                                    const Vec3& pattern_speed,
                                    std::vector<double>* trajectory = nullptr,
                                    int max_body_period_factor = kMaxBodyPeriodFactor) {
    // Nudge exact zeros off the coordinate axes / origin.
    for (double& c : state)
        if (c == 0.0) c = 1e-12;

    OrbitSummary summary;
    // The frames coincide at t = 0, so the period is estimated from the body
    // (i.e. static) force at the initial position.
    const double T = estimate_period(pot, state);
    const double omega_p = std::sqrt(pattern_speed[0] * pattern_speed[0] +
                                     pattern_speed[1] * pattern_speed[1] +
                                     pattern_speed[2] * pattern_speed[2]);
    summary.period = T;
    summary.body_period = body_frame_period(T, omega_p);
    summary.r_peri =
        std::sqrt(state[0] * state[0] + state[1] * state[1] + state[2] * state[2]);
    if (!(T > 0.0) || n_samples < 2) {
        summary.status = 1;
        return summary;
    }
    // Cover n_periods body-frame periods (up to the cap), at the same sampling
    // interval as n_periods inertial periods would have.
    const int factor =
        body_period_factor(T, summary.body_period, max_body_period_factor);
    n_samples *= factor;
    summary.t_total = static_cast<double>(n_periods) * factor * T;
    const double dt_out = summary.t_total / (n_samples - 1);

    namespace ode = boost::numeric::odeint;
    auto stepper = ode::make_dense_output(
        abs_tol, rel_tol, ode::runge_kutta_dopri5<OrbitState>());
    const FigureRotation rotation(pattern_speed);
    detail::EquationsOfMotion<Pot> sys{pot, rotation};
    detail::Accumulator<Pot> acc{pot, rotation, summary, trajectory};
    if (trajectory) {
        trajectory->clear();
        trajectory->reserve(static_cast<std::size_t>(n_samples) * 6);
    }

    try {
        // Exactly n_samples observations (t = 0 plus n_samples - 1 steps).
        // integrate_const(..., t_total, dt_out) can stop one sample short by
        // floating-point rounding, and the FFT then keeps only the largest
        // power-of-two prefix -- half the window.
        ode::integrate_n_steps(stepper, std::ref(sys), state, 0.0, dt_out,
                               static_cast<std::size_t>(n_samples - 1),
                               std::ref(acc));
    } catch (...) {
        acc.s.status = 2;
    }
    acc.finalise();
    OrbitSummary out = acc.s;
    // The sampled minimum is a valid upper bound too (the first sample is the
    // initial state, which the stepper evaluates as well).
    out.r_peri = std::min(sys.r_eval_min, out.r_min);
    if (sys.nan_hit || std::isnan(out.energy_mean)) out.status = 2;
    return out;
}

// Integrate a batch of orbits (OpenMP over orbits). `states` is n_orbits x 6
// row-major (HO units, inertial); `out_summary` is n_orbits x kSummaryCols
// row-major. `pattern_speed` is the figure's angular velocity (HO units).
template <class Pot>
inline void integrate_batch(const Pot& pot, const double* states,
                            std::size_t n_orbits, int n_periods, int n_samples,
                            double abs_tol, double rel_tol,
                            const Vec3& pattern_speed, double* out_summary,
                            bool progress = false,
                            int max_body_period_factor = kMaxBodyPeriodFactor) {
    std::atomic<std::size_t> completed{0};
    #pragma omp parallel for schedule(dynamic, 8)
    for (std::size_t i = 0; i < n_orbits; ++i) {
        OrbitState s;
        for (int j = 0; j < 6; ++j) s[j] = states[i * 6 + j];
        const OrbitSummary summary = integrate_orbit(
            pot, s, n_periods, n_samples, abs_tol, rel_tol, pattern_speed,
            nullptr, max_body_period_factor);
        write_summary(summary, out_summary + i * kSummaryCols);
        if (progress) report_orbit_progress(completed, n_orbits);
    }
}

}  // namespace lanfear
