#pragma once

// Ties orbit integration (orbit_integrator.hpp) to frequency analysis
// (frequency.hpp): integrate each orbit, then extract the leading spectral
// lines along each principal axis. The per-axis signal is the complex
//   w_k(t) = x_k(t) + i v_k(t) / w0,   w0 = 2*pi / T_period
// whose dominant line is the (signed) fundamental frequency of axis k. Signed
// frequencies distinguish prograde/retrograde loops, and the ratios of the
// three fundamentals drive the resonance-based classification to come.
//
// As with integrate_batch, trajectories are held only transiently (thread-local)
// so memory stays bounded for >1e6 orbits; only the compact per-orbit summary
// plus the leading frequency lines are returned.

#include <array>
#include <cstddef>
#include <vector>

#include "frequency.hpp"
#include "orbit_integrator.hpp"
#include "scf_potential.hpp"

namespace lanfear {

// Analyse one orbit. Fills `fundamental[3]` (leading frequency per axis) and
// `lines` (3 * n_lines SpectralLines, axis-major: axis 0 lines, axis 1, axis 2)
// and returns the dynamics summary. Axes with negligible motion yield lines of
// near-zero amplitude (the caller can treat those as "no oscillation").
template <class Pot>
inline OrbitSummary analyse_orbit(const Pot& pot, OrbitState state,
                                  int n_periods, int n_samples, double abs_tol,
                                  double rel_tol, int n_lines,
                                  std::array<double, 3>& fundamental,
                                  std::vector<SpectralLine>& lines) {
    static thread_local std::vector<double> traj;
    fundamental = {0.0, 0.0, 0.0};
    lines.assign(static_cast<std::size_t>(3) * n_lines, SpectralLine{});

    const OrbitSummary summary = integrate_orbit(
        pot, state, n_periods, n_samples, abs_tol, rel_tol, &traj);
    if (summary.status != 0 || !(summary.period > 0.0)) return summary;

    const std::size_t n = traj.size() / 6;
    if (n < 4) return summary;
    const double dt = summary.t_total / (n - 1);
    const double w0 = 2.0 * M_PI / summary.period;

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
    }
    return summary;
}

// Batch analysis (OpenMP over orbits). Output buffers, row-major:
//   out_summary     : n_orbits x kSummaryCols
//   out_fundamental : n_orbits x 3
//   out_lines       : n_orbits x (3 * n_lines * 2)   [freq, amp] per line
template <class Pot>
inline void analyse_batch(const Pot& pot, const double* states,
                          std::size_t n_orbits, int n_periods, int n_samples,
                          double abs_tol, double rel_tol, int n_lines,
                          double* out_summary, double* out_fundamental,
                          double* out_lines) {
    const std::size_t line_stride = static_cast<std::size_t>(3) * n_lines * 2;
    #pragma omp parallel for schedule(dynamic, 8)
    for (std::size_t i = 0; i < n_orbits; ++i) {
        OrbitState s;
        for (int j = 0; j < 6; ++j) s[j] = states[i * 6 + j];

        std::array<double, 3> fundamental;
        std::vector<SpectralLine> lines;
        const OrbitSummary summary =
            analyse_orbit(pot, s, n_periods, n_samples, abs_tol, rel_tol,
                          n_lines, fundamental, lines);

        write_summary(summary, out_summary + i * kSummaryCols);
        for (int a = 0; a < 3; ++a)
            out_fundamental[i * 3 + a] = fundamental[a];
        double* lp = out_lines + i * line_stride;
        for (std::size_t k = 0; k < lines.size(); ++k) {
            lp[2 * k] = lines[k].frequency;
            lp[2 * k + 1] = lines[k].amplitude;
        }
    }
}

}  // namespace lanfear
