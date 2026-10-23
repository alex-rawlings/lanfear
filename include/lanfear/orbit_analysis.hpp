#pragma once

// Ties orbit integration (orbit_integrator.hpp) to frequency analysis
// (frequency.hpp): integrate each orbit, then extract the leading spectral
// lines along each principal axis. The per-axis signal is the complex
//   w_k(t) = x_k(t) + i v_k(t) / w0,   w0 = 2*pi / T_period
// whose dominant line is the (signed) fundamental frequency of axis k. Signed
// frequencies distinguish prograde/retrograde loops, and the ratios of the
// three fundamentals drive the resonance-based classification to come.
//
// Alongside, the *planar* position signal about each axis a,
//   p_x = y + i z,   p_y = z + i x,   p_z = x + i y,
// is analysed too. Its lines are signed: a loop circulating about a in the
// sense of positive L_a has a positive leading line, while a box (no net
// circulation) has comparable lines at +w and -w. Rotating the frame about a
// multiplies p_a by a pure phase e^{i Omega t}, so for the plane perpendicular
// to the pattern speed the inertial-frame spectrum is the body-frame one
// shifted by exactly Omega (no sidebands). The radial signal r + i v_r / w0
// gives the frame-independent radial frequency (summary columns
// radial_frequency, radial_cycles).
//
// For a rotating figure the signal is built from the co-rotating body-frame
// trajectory (x_b, v_rot) that integrate_orbit records: that is the frame in
// which the potential is static and a regular orbit is a 3-torus with three
// base frequencies. (In the inertial frame the pattern speed would appear as
// a spurious fourth base frequency.) Resonances found here are therefore
// resonances with the pattern.
//
// As with integrate_batch, trajectories are held only transiently (thread-local)
// so memory stays bounded for >1e6 orbits; only the compact per-orbit summary
// plus the leading frequency lines are returned.

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "frequency.hpp"
#include "orbit_integrator.hpp"
#include "scf_potential.hpp"

namespace lanfear {

// The same line is tracked between halves among the strongest kDiffusionLines
// lines of the second half, and only if its amplitude is at least
// kDiffusionAmpFrac of that half's strongest line.
constexpr int kDiffusionLines = 3;
constexpr double kDiffusionAmpFrac = 0.5;

// Leading `n_lines` NAFF lines of a signal after removing its DC mean.
inline std::vector<SpectralLine> centred_lines(std::vector<cdouble> sig,
                                               double dt, int n_lines) {
    cdouble mean(0.0, 0.0);
    for (const cdouble& v : sig) mean += v;
    mean /= static_cast<double>(sig.size());
    for (cdouble& v : sig) v -= mean;
    return naff(std::move(sig), dt, n_lines);
}

// Frequency drift |w2 - w1| / |w1| between two halves of one axis. w1 is the
// leading line of the first half. The same line is tracked into the second half
// as the near-equal-strength line closest to w1: two comparably strong lines
// can swap rank between halves, and comparing the *leading* lines would then
// report a spurious drift equal to their spacing. NaN if either half is empty.
inline double half_drift(const std::vector<SpectralLine>& first,
                         const std::vector<SpectralLine>& second) {
    if (first.empty() || second.empty() || first[0].frequency == 0.0)
        return std::nan("");
    const double w1 = first[0].frequency;
    const double cut = kDiffusionAmpFrac * second[0].amplitude;
    double w2 = second[0].frequency;
    double best = std::abs(w2 - w1);
    for (std::size_t k = 0; k < second.size() && k < static_cast<std::size_t>(kDiffusionLines); ++k) {
        const SpectralLine& l = second[k];
        if (l.amplitude < cut) continue;
        if (std::abs(l.frequency - w1) < best) {
            best = std::abs(l.frequency - w1);
            w2 = l.frequency;
        }
    }
    return std::abs(w2 - w1) / std::abs(w1);
}

// Radial frequency |w_R| of a body-frame trajectory (n samples of x_b, v_rot):
// the leading NAFF line of r + i v_r / w0. Both r and v_r = x . v / r are the
// same in either frame (x . (Omega x x) = 0). NaN if the radial motion is
// negligible (line amplitude below 1e-6 r_mean); a near-circular orbit's small
// epicycle still registers, giving the epicyclic frequency.
inline double radial_frequency(const std::vector<double>& traj, std::size_t n,
                               double dt, double w0, double r_mean) {
    std::vector<cdouble> sig(n);
    for (std::size_t j = 0; j < n; ++j) {
        const double* s = &traj[j * 6];
        const double r = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
        const double v_r =
            r > 0.0 ? (s[0] * s[3] + s[1] * s[4] + s[2] * s[5]) / r : 0.0;
        sig[j] = cdouble(r, v_r / w0);
    }
    const std::vector<SpectralLine> radial = centred_lines(std::move(sig), dt, 1);
    if (radial.empty() || !(radial[0].amplitude > 1e-6 * r_mean))
        return std::nan("");
    return std::abs(radial[0].frequency);
}

// Analyse one orbit, the figure rotating at `pattern_speed` (HO units; zero
// for a static potential). Fills `fundamental[3]` (leading frequency per axis),
// `lines` (3 * n_lines SpectralLines, axis-major: axis 0 lines, axis 1, axis 2),
// `diffusion[3]` (Laskar frequency-diffusion rate per axis) and `planar_lines`
// (3 * n_lines signed lines of the planar signals p_x, p_y, p_z, same layout)
// and returns the dynamics summary, including the radial frequency. Axes with negligible motion yield lines of near-zero
// amplitude (the caller can treat those as "no oscillation"). The window is
// lengthened for a rotating figure as in integrate_orbit (max_body_period_factor).
//
// The diffusion rate of an axis is |w2 - w1| / |w1| (Laskar), where w1 and w2
// are the leading NAFF frequencies of the first and second half of the
// integration (see half_drift for how the line is tracked between halves). It
// is ~0 for regular orbits and grows for chaotic ones. NaN for a failed orbit or
// when either half yields no frequency.
template <class Pot>
inline OrbitSummary analyse_orbit(const Pot& pot, OrbitState state,
                                  int n_periods, int n_samples, double abs_tol,
                                  double rel_tol, const Vec3& pattern_speed,
                                  int n_lines,
                                  std::array<double, 3>& fundamental,
                                  std::vector<SpectralLine>& lines,
                                  std::array<double, 3>& diffusion,
                                  std::vector<SpectralLine>& planar_lines,
                                  int max_body_period_factor = kMaxBodyPeriodFactor) {
    static thread_local std::vector<double> traj;
    fundamental = {0.0, 0.0, 0.0};
    diffusion = {std::nan(""), std::nan(""), std::nan("")};
    lines.assign(static_cast<std::size_t>(3) * n_lines, SpectralLine{});
    planar_lines.assign(static_cast<std::size_t>(3) * n_lines, SpectralLine{});

    OrbitSummary summary = integrate_orbit(
        pot, state, n_periods, n_samples, abs_tol, rel_tol, pattern_speed,
        &traj, max_body_period_factor);
    if (summary.status != 0 || !(summary.period > 0.0)) return summary;

    const std::size_t n = traj.size() / 6;
    if (n < 4) return summary;
    const double dt = summary.t_total / (n - 1);
    const double w0 = 2.0 * M_PI / summary.period;
    // Equal-length halves (so both share the same frequency resolution).
    const std::size_t half = n / 2;

    std::vector<cdouble> sig(n);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t j = 0; j < n; ++j)
            sig[j] = cdouble(traj[j * 6 + axis], traj[j * 6 + 3 + axis] / w0);
        // Remove the DC component: over a finite window an eccentric/precessing
        // orbit has a non-zero coordinate mean, which is not a physical
        // frequency and would otherwise be reported as a spurious w~0 line.
        cdouble mean(0.0, 0.0);
        for (std::size_t j = 0; j < n; ++j) mean += sig[j];
        mean /= static_cast<double>(n);
        for (std::size_t j = 0; j < n; ++j) sig[j] -= mean;
        std::vector<SpectralLine> ax = naff(sig, dt, n_lines);
        for (std::size_t i = 0; i < ax.size(); ++i)
            lines[static_cast<std::size_t>(axis) * n_lines + i] = ax[i];
        if (!ax.empty()) fundamental[axis] = ax[0].frequency;

        // Each half is re-centred inside centred_lines().
        diffusion[axis] = half_drift(
            centred_lines(std::vector<cdouble>(sig.begin(), sig.begin() + half), dt,
                       kDiffusionLines),
            centred_lines(std::vector<cdouble>(sig.begin() + half,
                                            sig.begin() + 2 * half),
                       dt, kDiffusionLines));
    }

    // Planar signals about x, y, z: (y, z), (z, x), (x, y) as real, imaginary.
    constexpr int kPlane[3][2] = {{1, 2}, {2, 0}, {0, 1}};
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t j = 0; j < n; ++j)
            sig[j] = cdouble(traj[j * 6 + kPlane[axis][0]],
                             traj[j * 6 + kPlane[axis][1]]);
        const std::vector<SpectralLine> pl = centred_lines(sig, dt, n_lines);
        for (std::size_t i = 0; i < pl.size(); ++i)
            planar_lines[static_cast<std::size_t>(axis) * n_lines + i] = pl[i];
    }

    summary.radial_frequency = radial_frequency(traj, n, dt, w0, summary.r_mean);
    summary.radial_cycles = summary.radial_frequency * summary.t_total / (2.0 * M_PI);
    return summary;
}

// Batch analysis (OpenMP over orbits). Output buffers, row-major:
//   out_summary     : n_orbits x kSummaryCols
//   out_fundamental : n_orbits x 3
//   out_lines       : n_orbits x (3 * n_lines * 2)   [freq, amp] per line
//   out_diffusion   : n_orbits x 3   frequency-diffusion rate per axis
//   out_planar_lines: n_orbits x (3 * n_lines * 2)   [freq, amp] per signed
//                     line of the planar signals p_x, p_y, p_z
// `pattern_speed` is the figure's angular velocity (HO units; zero if static).
template <class Pot>
inline void analyse_batch(const Pot& pot, const double* states,
                          std::size_t n_orbits, int n_periods, int n_samples,
                          double abs_tol, double rel_tol,
                          const Vec3& pattern_speed, int n_lines,
                          double* out_summary, double* out_fundamental,
                          double* out_lines, double* out_diffusion,
                          double* out_planar_lines,
                          bool progress = false,
                          int max_body_period_factor = kMaxBodyPeriodFactor) {
    const std::size_t line_stride = static_cast<std::size_t>(3) * n_lines * 2;
    std::atomic<std::size_t> completed{0};
    #pragma omp parallel for schedule(dynamic, 8)
    for (std::size_t i = 0; i < n_orbits; ++i) {
        OrbitState s;
        for (int j = 0; j < 6; ++j) s[j] = states[i * 6 + j];

        std::array<double, 3> fundamental;
        std::array<double, 3> diffusion;
        std::vector<SpectralLine> lines, planar_lines;
        const OrbitSummary summary =
            analyse_orbit(pot, s, n_periods, n_samples, abs_tol, rel_tol,
                          pattern_speed, n_lines, fundamental, lines, diffusion,
                          planar_lines, max_body_period_factor);

        write_summary(summary, out_summary + i * kSummaryCols);
        for (int a = 0; a < 3; ++a)
            out_fundamental[i * 3 + a] = fundamental[a];
        for (int a = 0; a < 3; ++a) out_diffusion[i * 3 + a] = diffusion[a];
        double* lp = out_lines + i * line_stride;
        for (std::size_t k = 0; k < lines.size(); ++k) {
            lp[2 * k] = lines[k].frequency;
            lp[2 * k + 1] = lines[k].amplitude;
        }
        double* pp = out_planar_lines + i * line_stride;
        for (std::size_t k = 0; k < planar_lines.size(); ++k) {
            pp[2 * k] = planar_lines[k].frequency;
            pp[2 * k + 1] = planar_lines[k].amplitude;
        }
        if (progress) report_orbit_progress(completed, n_orbits);
    }
}

}  // namespace lanfear
