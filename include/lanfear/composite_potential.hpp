#pragma once

// Linear superposition of independently-fit potential components (e.g. a
// stellar disc plus a dark-matter halo), each already expressed in its own
// internal HO-style unit system (G = component field mass = component scale
// radius = 1, exactly as SCFPotential/DiscPotential are built). Poisson's
// equation is linear, so summing the physical potentials/accelerations of the
// components is exact; what this class adds is the unit bookkeeping needed to
// combine components built on different physical scales, plus its own set of
// black holes (attached once, at the composite level).
//
// Each component carries three numbers relating it to the *composite's*
// shared HO unit system (length unit L0, mass unit M0 -- see
// lanfear/multi_component_potential.py, which computes them):
//   coord_scale = L0 / a_i           composite-HO coord -> component-HO coord
//   phi_weight  = (M_i/M0) * coord_scale
//   acc_weight  = (M_i/M0) * coord_scale^2
// so that, writing x_h for a composite-HO coordinate,
//   phi_h(x_h)  = sum_i phi_weight_i  * component_i.potential(x_h * coord_scale_i)
//   accel_h(x_h)= sum_i acc_weight_i  * component_i.acceleration(x_h * coord_scale_i)
// reproduce the (composite-HO-normalised) sum of the components' physical
// fields. See the python module docstring for the full derivation.

#include <array>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "black_hole.hpp"
#include "disc_potential.hpp"
#include "scf_potential.hpp"

namespace lanfear {

using PotentialVariant = std::variant<SCFPotential, DiscPotential>;

struct CompositeComponent {
    PotentialVariant pot;
    double coord_scale;  // L0 / a_i
    double phi_weight;   // (M_i/M0) * coord_scale
    double acc_weight;   // (M_i/M0) * coord_scale^2
    std::string label;   // species label (provenance only; not used in maths)
};

class CompositePotential {
public:
    explicit CompositePotential(std::vector<CompositeComponent> components)
        : components_(std::move(components)) {
        if (components_.empty())
            throw std::invalid_argument(
                "CompositePotential needs at least one component");
    }

    std::size_t size() const { return components_.size(); }
    const std::vector<CompositeComponent>& components() const {
        return components_;
    }

    void add_black_hole(double mass, double x, double y, double z,
                        double softening) {
        black_holes_.add_black_hole(mass, x, y, z, softening);
    }
    std::size_t num_black_holes() const { return black_holes_.num_black_holes(); }
    const std::vector<BlackHole>& black_holes() const {
        return black_holes_.black_holes();
    }

    double potential(double x, double y, double z) const {
        double p = 0.0;
        for (const auto& c : components_) {
            const double s = c.coord_scale;
            const double phi = std::visit(
                [&](const auto& pot) { return pot.potential(x * s, y * s, z * s); },
                c.pot);
            p += c.phi_weight * phi;
        }
        return p + black_holes_.potential(x, y, z);
    }

    std::array<double, 3> acceleration(double x, double y, double z) const {
        std::array<double, 3> acc{0.0, 0.0, 0.0};
        for (const auto& c : components_) {
            const double s = c.coord_scale;
            const auto a = std::visit(
                [&](const auto& pot) {
                    return pot.acceleration(x * s, y * s, z * s);
                },
                c.pot);
            acc[0] += c.acc_weight * a[0];
            acc[1] += c.acc_weight * a[1];
            acc[2] += c.acc_weight * a[2];
        }
        black_holes_.add_acceleration(x, y, z, acc);
        return acc;
    }

private:
    std::vector<CompositeComponent> components_;
    BlackHoleSet black_holes_;
};

}  // namespace lanfear
