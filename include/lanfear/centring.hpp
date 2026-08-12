#pragma once

// Shrinking-sphere centring. Starting from the naive (mass-weighted) centre of
// mass and a sphere enclosing a large fraction of the particles, the centre is
// iteratively refined as the mass-weighted COM of the particles inside the
// sphere while the sphere shrinks by a fixed factor each step, until only a
// small fraction of the particles remain inside. This is robust to substructure
// and asymmetric outskirts that bias a single global COM, and is the standard
// way to locate the centre of a simulated halo/galaxy (Power et al. 2003).
//
// The velocity centre (bulk motion) is the mass-weighted mean velocity of the
// particles in the final sphere.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace lanfear {

// Result of shrinking_sphere_centre().
struct CentreResult {
    std::array<double, 3> position{{0.0, 0.0, 0.0}};  // spatial centre
    std::array<double, 3> velocity{{0.0, 0.0, 0.0}};  // bulk (COM) velocity
    double final_radius = 0.0;   // radius of the final sphere
    std::size_t n_final = 0;     // particles in the final sphere
    int n_iterations = 0;        // shrink steps taken
};

// Locate the centre by the shrinking-sphere method.
//
//   pos, vel : N*3 row-major (x0,y0,z0, x1,y1,z1, ...); vel may be null.
//   mass     : N particle masses.
//   n        : particle count.
//   enclose_frac : fraction of particles the initial sphere encloses (0.80).
//   shrink_factor: the sphere radius is multiplied by this each step (0.93).
//   stop_frac    : stop once the sphere holds <= this fraction of particles
//                  (0.01); at least one particle is always required.
//
// Returns the refined spatial centre and the bulk velocity of the final sphere.
inline CentreResult shrinking_sphere_centre(const double* pos, const double* vel,
                                            const double* mass, std::size_t n,
                                            double enclose_frac = 0.80,
                                            double shrink_factor = 0.93,
                                            double stop_frac = 0.01) {
    CentreResult result;
    if (n == 0) return result;

    // Naive mass-weighted centre of mass as the starting guess.
    double cx = 0.0, cy = 0.0, cz = 0.0, m_tot = 0.0;
    #pragma omp parallel for reduction(+ : cx, cy, cz, m_tot) schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        const double mi = mass[i];
        cx += mi * pos[3 * i];
        cy += mi * pos[3 * i + 1];
        cz += mi * pos[3 * i + 2];
        m_tot += mi;
    }
    if (m_tot > 0.0) {
        cx /= m_tot;
        cy /= m_tot;
        cz /= m_tot;
    }
    result.position = {{cx, cy, cz}};

    // Initial radius: the enclose_frac-quantile of the distance from the guess.
    // nth_element gives this in O(N) without a full sort.
    std::vector<double> r2(n);
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        const double dx = pos[3 * i] - cx;
        const double dy = pos[3 * i + 1] - cy;
        const double dz = pos[3 * i + 2] - cz;
        r2[i] = dx * dx + dy * dy + dz * dz;
    }
    std::size_t k = static_cast<std::size_t>(enclose_frac * n);
    if (k >= n) k = n - 1;
    std::nth_element(r2.begin(), r2.begin() + k, r2.end());
    double radius = std::sqrt(r2[k]);

    // Stop once the sphere holds no more than this many particles (>= 1).
    std::size_t stop_count = static_cast<std::size_t>(stop_frac * n);
    if (stop_count < 1) stop_count = 1;

    // Iteratively recompute the COM of the in-sphere particles and shrink.
    while (true) {
        const double r_sq = radius * radius;
        double sx = 0.0, sy = 0.0, sz = 0.0, sm = 0.0;
        double svx = 0.0, svy = 0.0, svz = 0.0;
        std::size_t count = 0;
        #pragma omp parallel for schedule(static) reduction(                    \
                + : sx, sy, sz, sm, svx, svy, svz, count)
        for (std::size_t i = 0; i < n; ++i) {
            const double dx = pos[3 * i] - cx;
            const double dy = pos[3 * i + 1] - cy;
            const double dz = pos[3 * i + 2] - cz;
            if (dx * dx + dy * dy + dz * dz <= r_sq) {
                const double mi = mass[i];
                sx += mi * pos[3 * i];
                sy += mi * pos[3 * i + 1];
                sz += mi * pos[3 * i + 2];
                sm += mi;
                ++count;
                if (vel != nullptr) {
                    svx += mi * vel[3 * i];
                    svy += mi * vel[3 * i + 1];
                    svz += mi * vel[3 * i + 2];
                }
            }
        }

        // An empty sphere cannot refine the centre; keep the last good result.
        if (count == 0 || sm <= 0.0) break;

        cx = sx / sm;
        cy = sy / sm;
        cz = sz / sm;
        result.position = {{cx, cy, cz}};
        if (vel != nullptr)
            result.velocity = {{svx / sm, svy / sm, svz / sm}};
        result.final_radius = radius;
        result.n_final = count;
        ++result.n_iterations;

        // The final sphere is the one at or below the target occupancy.
        if (count <= stop_count) break;
        radius *= shrink_factor;
    }

    return result;
}

}  // namespace lanfear
