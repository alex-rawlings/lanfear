#pragma once

// Brute-force direct-summation (softened point-mass) potential of a set of
// source particles. This is the "ground truth" reference that the SCF
// expansion is checked against in Potential.validate() /
// Potential.plot_potential_plane() -- an O(n_points * n_sources) sum with no
// approximation beyond the softening itself, so it stays trustworthy as a
// reference regardless of the SCF truncation order under test.

#include <cmath>
#include <cstddef>

namespace lanfear {

// Potential at (x, y, z) from softened point masses at (sx[k], sy[k], sz[k])
// with mass sm[k], k = 0..n_sources-1, Plummer-softened by `softening`.
inline double direct_potential(double x, double y, double z, const double* sx,
                               const double* sy, const double* sz,
                               const double* sm, std::size_t n_sources,
                               double softening) {
    const double eps2 = softening * softening;
    double phi = 0.0;
    for (std::size_t k = 0; k < n_sources; ++k) {
        const double dx = x - sx[k];
        const double dy = y - sy[k];
        const double dz = z - sz[k];
        phi -= sm[k] / std::sqrt(dx * dx + dy * dy + dz * dz + eps2);
    }
    return phi;
}

}  // namespace lanfear
