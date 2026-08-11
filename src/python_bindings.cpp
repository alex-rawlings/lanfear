#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lanfear/disc_potential.hpp"
#include "lanfear/orbit_analysis.hpp"
#include "lanfear/orbit_integrator.hpp"
#include "lanfear/scf_potential.hpp"

namespace py = pybind11;
using lanfear::DiscPotential;
using lanfear::SCFPotential;

namespace {

using CArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Build an SCFPotential from numpy arrays (N,3) positions and (N,) masses.
SCFPotential make_scf(int n_max, int l_max, CArray pos, CArray mass) {
    if (pos.ndim() != 2 || pos.shape(1) != 3)
        throw std::runtime_error("pos must have shape (N, 3)");
    if (mass.ndim() != 1 || mass.shape(0) != pos.shape(0))
        throw std::runtime_error("mass must have shape (N,) matching pos");
    const std::size_t n = static_cast<std::size_t>(pos.shape(0));
    std::vector<double> x(n), y(n), z(n), m(n);
    auto p = pos.unchecked<2>();
    auto mm = mass.unchecked<1>();
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = p(i, 0); y[i] = p(i, 1); z[i] = p(i, 2); m[i] = mm(i);
    }
    return SCFPotential(n_max, l_max, x, y, z, m);
}

// --- generic (potential-type-agnostic) orbit API ----------------------------

template <class Pot>
py::array_t<double> potential_batch(const Pot& self, CArray pts) {
    if (pts.ndim() != 2 || pts.shape(1) != 3)
        throw std::runtime_error("points must have shape (N, 3)");
    const py::ssize_t n = pts.shape(0);
    auto p = pts.unchecked<2>();
    py::array_t<double> out(n);
    auto o = out.mutable_unchecked<1>();
    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(dynamic, 256)
        for (py::ssize_t i = 0; i < n; ++i)
            o(i) = self.potential(p(i, 0), p(i, 1), p(i, 2));
    }
    return out;
}

template <class Pot>
py::array_t<double> acceleration_batch(const Pot& self, CArray pts) {
    if (pts.ndim() != 2 || pts.shape(1) != 3)
        throw std::runtime_error("points must have shape (N, 3)");
    const py::ssize_t n = pts.shape(0);
    auto p = pts.unchecked<2>();
    py::array_t<double> out({n, static_cast<py::ssize_t>(3)});
    auto o = out.mutable_unchecked<2>();
    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(dynamic, 256)
        for (py::ssize_t i = 0; i < n; ++i) {
            const auto a = self.acceleration(p(i, 0), p(i, 1), p(i, 2));
            o(i, 0) = a[0]; o(i, 1) = a[1]; o(i, 2) = a[2];
        }
    }
    return out;
}

template <class Pot>
py::array_t<double> integrate_batch_py(const Pot& self, CArray states,
                                       int n_periods, int n_samples,
                                       double abs_tol, double rel_tol,
                                       bool progress) {
    if (states.ndim() != 2 || states.shape(1) != 6)
        throw std::runtime_error("states must have shape (N, 6)");
    const py::ssize_t n = states.shape(0);
    const double* sp = states.data();
    py::array_t<double> out({n, static_cast<py::ssize_t>(lanfear::kSummaryCols)});
    double* op = out.mutable_data();
    {
        py::gil_scoped_release release;
        lanfear::integrate_batch(self, sp, static_cast<std::size_t>(n),
                                 n_periods, n_samples, abs_tol, rel_tol, op,
                                 progress);
    }
    return out;
}

template <class Pot>
py::tuple integrate_orbit_py(const Pot& self, CArray state, int n_periods,
                             int n_samples, double abs_tol, double rel_tol,
                             bool return_trajectory) {
    if (state.size() != 6)
        throw std::runtime_error("state must have 6 elements");
    lanfear::OrbitState s;
    for (int j = 0; j < 6; ++j) s[j] = state.data()[j];
    std::vector<double> traj;
    const lanfear::OrbitSummary summary = lanfear::integrate_orbit(
        self, s, n_periods, n_samples, abs_tol, rel_tol,
        return_trajectory ? &traj : nullptr);
    py::array_t<double> summ(static_cast<py::ssize_t>(lanfear::kSummaryCols));
    lanfear::write_summary(summary, summ.mutable_data());
    if (!return_trajectory) return py::make_tuple(summ, py::none());
    const py::ssize_t rows = static_cast<py::ssize_t>(traj.size() / 6);
    py::array_t<double> tarr({rows, static_cast<py::ssize_t>(6)});
    std::copy(traj.begin(), traj.end(), tarr.mutable_data());
    return py::make_tuple(summ, tarr);
}

template <class Pot>
py::tuple analyse_batch_py(const Pot& self, CArray states, int n_periods,
                           int n_samples, double abs_tol, double rel_tol,
                           int n_lines, bool progress) {
    if (states.ndim() != 2 || states.shape(1) != 6)
        throw std::runtime_error("states must have shape (N, 6)");
    if (n_lines < 1) throw std::runtime_error("n_lines must be >= 1");
    const py::ssize_t n = states.shape(0);
    const double* sp = states.data();
    py::array_t<double> summary(
        {n, static_cast<py::ssize_t>(lanfear::kSummaryCols)});
    py::array_t<double> fundamental({n, static_cast<py::ssize_t>(3)});
    py::array_t<double> lines({n, static_cast<py::ssize_t>(3),
                              static_cast<py::ssize_t>(n_lines),
                              static_cast<py::ssize_t>(2)});
    {
        py::gil_scoped_release release;
        lanfear::analyse_batch(self, sp, static_cast<std::size_t>(n), n_periods,
                               n_samples, abs_tol, rel_tol, n_lines,
                               summary.mutable_data(), fundamental.mutable_data(),
                               lines.mutable_data(), progress);
    }
    return py::make_tuple(summary, fundamental, lines);
}

template <class Pot>
py::tuple analyse_orbit_py(const Pot& self, CArray state, int n_periods,
                           int n_samples, double abs_tol, double rel_tol,
                           int n_lines) {
    if (state.size() != 6)
        throw std::runtime_error("state must have 6 elements");
    if (n_lines < 1) throw std::runtime_error("n_lines must be >= 1");
    lanfear::OrbitState s;
    for (int j = 0; j < 6; ++j) s[j] = state.data()[j];
    std::array<double, 3> fund;
    std::vector<lanfear::SpectralLine> lines;
    const lanfear::OrbitSummary summary = lanfear::analyse_orbit(
        self, s, n_periods, n_samples, abs_tol, rel_tol, n_lines, fund, lines);
    py::array_t<double> summ(static_cast<py::ssize_t>(lanfear::kSummaryCols));
    lanfear::write_summary(summary, summ.mutable_data());
    py::array_t<double> fundamental(3);
    for (int a = 0; a < 3; ++a) fundamental.mutable_data()[a] = fund[a];
    py::array_t<double> larr({static_cast<py::ssize_t>(3),
                             static_cast<py::ssize_t>(n_lines),
                             static_cast<py::ssize_t>(2)});
    double* lp = larr.mutable_data();
    for (std::size_t k = 0; k < lines.size(); ++k) {
        lp[2 * k] = lines[k].frequency;
        lp[2 * k + 1] = lines[k].amplitude;
    }
    return py::make_tuple(summ, fundamental, larr);
}

// Register the shared orbit/analysis API onto any potential class.
template <class Pot>
void register_orbit_api(py::class_<Pot>& cls) {
    cls.def("potential",
            [](const Pot& s, double x, double y, double z) {
                return s.potential(x, y, z);
            },
            py::arg("x"), py::arg("y"), py::arg("z"),
            "Potential at a single Cartesian point (HO units).")
        .def("acceleration",
             [](const Pot& s, double x, double y, double z) {
                 const auto a = s.acceleration(x, y, z);
                 return std::array<double, 3>{a[0], a[1], a[2]};
             },
             py::arg("x"), py::arg("y"), py::arg("z"),
             "Acceleration at a single Cartesian point (HO units).")
        .def("potential_batch", &potential_batch<Pot>, py::arg("points"),
             "Potential at points (N,3) -> (N,).")
        .def("acceleration_batch", &acceleration_batch<Pot>, py::arg("points"),
             "Acceleration at points (N,3) -> (N,3).")
        .def("integrate_batch", &integrate_batch_py<Pot>, py::arg("states"),
             py::arg("n_periods") = 50, py::arg("n_samples") = 8192,
             py::arg("abs_tol") = 1e-10, py::arg("rel_tol") = 1e-9,
             py::arg("progress") = false,
             "Integrate a batch of orbits (N,6) -> summaries "
             "(N, len(summary_columns)). OpenMP over orbits, GIL released. "
             "Set progress=True to print '<X>% of particles integrated' every "
             "10% of orbits.")
        .def("integrate_orbit", &integrate_orbit_py<Pot>, py::arg("state"),
             py::arg("n_periods") = 50, py::arg("n_samples") = 8192,
             py::arg("abs_tol") = 1e-10, py::arg("rel_tol") = 1e-9,
             py::arg("return_trajectory") = false,
             "Integrate one orbit; returns (summary, trajectory|None).")
        .def("analyse_batch", &analyse_batch_py<Pot>, py::arg("states"),
             py::arg("n_periods") = 50, py::arg("n_samples") = 8192,
             py::arg("abs_tol") = 1e-10, py::arg("rel_tol") = 1e-9,
             py::arg("n_lines") = 4, py::arg("progress") = false,
             "Integrate + frequency-analyse a batch (N,6). Returns "
             "(summary (N,kCols), fundamentals (N,3), lines (N,3,n_lines,2)). "
             "Set progress=True to print '<X>% of particles integrated' every "
             "10% of orbits.")
        .def("analyse_orbit", &analyse_orbit_py<Pot>, py::arg("state"),
             py::arg("n_periods") = 50, py::arg("n_samples") = 8192,
             py::arg("abs_tol") = 1e-10, py::arg("rel_tol") = 1e-9,
             py::arg("n_lines") = 4,
             "Integrate + frequency-analyse one orbit -> "
             "(summary (kCols,), fundamentals (3,), lines (3,n_lines,2)).")
        .def_property_readonly("num_black_holes", &Pot::num_black_holes)
        .def("add_black_hole", &Pot::add_black_hole, py::arg("mass"),
             py::arg("x"), py::arg("y"), py::arg("z"), py::arg("softening"),
             "Add a softened point mass at an arbitrary position (HO units).");
}

// --- DiscPotential construction / SCF sum -----------------------------------

DiscPotential make_disc(CArray a, CArray b) {
    return DiscPotential(
        std::vector<double>(a.data(), a.data() + a.size()),
        std::vector<double>(b.data(), b.data() + b.size()));
}

py::array_t<double> disc_scf_sum(const DiscPotential& self, CArray pos,
                                 CArray mass) {
    if (pos.ndim() != 2 || pos.shape(1) != 3)
        throw std::runtime_error("pos must have shape (N, 3)");
    if (mass.ndim() != 1 || mass.shape(0) != pos.shape(0))
        throw std::runtime_error("mass must have shape (N,) matching pos");
    const std::size_t n = static_cast<std::size_t>(pos.shape(0));
    std::vector<double> x(n), y(n), z(n), m(n);
    auto p = pos.unchecked<2>();
    auto mm = mass.unchecked<1>();
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = p(i, 0); y[i] = p(i, 1); z[i] = p(i, 2); m[i] = mm(i);
    }
    std::vector<double> b;
    {
        py::gil_scoped_release release;
        b = self.scf_sum(x, y, z, m);
    }
    return py::cast(b);
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() =
        "lanfear C++ core: SCF potentials (Hernquist-Ostriker spheroidal and "
        "Miyamoto-Nagai disc bases) with softened black holes, orbit "
        "integration and frequency analysis.";

    // --- SCFPotential (spheroidal HO basis) ---
    py::class_<SCFPotential> scf(m, "SCFPotential");
    scf.def(py::init(&make_scf), py::arg("n_max"), py::arg("l_max"),
            py::arg("pos"), py::arg("mass"),
            "Build the HO/SCF expansion from field particles (BH removed). "
            "pos (N,3) and mass (N,) in HO units.")
        .def_property_readonly("n_max", &SCFPotential::n_max)
        .def_property_readonly("l_max", &SCFPotential::l_max)
        .def(py::pickle(
            [](const SCFPotential& p) {
                py::list bhs;
                for (const auto& bh : p.black_holes())
                    bhs.append(py::make_tuple(bh.mass, bh.pos[0], bh.pos[1],
                                              bh.pos[2], bh.softening));
                return py::make_tuple(p.n_max(), p.l_max(),
                                      py::cast(p.coefficients_cos()),
                                      py::cast(p.coefficients_sin()), bhs);
            },
            [](py::tuple t) {
                if (t.size() != 5)
                    throw std::runtime_error("invalid SCFPotential state");
                auto pot = SCFPotential(t[0].cast<int>(), t[1].cast<int>(),
                                        t[2].cast<std::vector<double>>(),
                                        t[3].cast<std::vector<double>>());
                for (auto item : t[4].cast<py::list>()) {
                    auto bh = item.cast<py::tuple>();
                    pot.add_black_hole(bh[0].cast<double>(), bh[1].cast<double>(),
                                       bh[2].cast<double>(), bh[3].cast<double>(),
                                       bh[4].cast<double>());
                }
                return pot;
            }));
    register_orbit_api(scf);

    // --- DiscPotential (Miyamoto-Nagai disc basis) ---
    py::class_<DiscPotential> disc(m, "DiscPotential");
    disc.def(py::init(&make_disc), py::arg("a"), py::arg("b"),
             "Disc basis with Miyamoto-Nagai scales a[i] (radial) and b[i] "
             "(thickness). Coefficients are set after solving the Galerkin "
             "system (see the python DiscPotential wrapper).")
        .def("scf_sum", &disc_scf_sum, py::arg("pos"), py::arg("mass"),
             "SCF particle sum b_a = sum_k m_k Phi_a(x_k) -> (M,).")
        .def("set_coefficients",
             [](DiscPotential& s, std::vector<double> c) {
                 s.set_coefficients(std::move(c));
             },
             py::arg("coefficients"))
        .def_property_readonly("coefficients", &DiscPotential::coefficients)
        .def_property_readonly("basis_a", &DiscPotential::basis_a)
        .def_property_readonly("basis_b", &DiscPotential::basis_b)
        .def_property_readonly("size", &DiscPotential::size)
        .def(py::pickle(
            [](const DiscPotential& p) {
                py::list bhs;
                for (const auto& bh : p.black_holes())
                    bhs.append(py::make_tuple(bh.mass, bh.pos[0], bh.pos[1],
                                              bh.pos[2], bh.softening));
                return py::make_tuple(py::cast(p.basis_a()), py::cast(p.basis_b()),
                                      py::cast(p.coefficients()), bhs);
            },
            [](py::tuple t) {
                if (t.size() != 4)
                    throw std::runtime_error("invalid DiscPotential state");
                auto pot = DiscPotential(t[0].cast<std::vector<double>>(),
                                         t[1].cast<std::vector<double>>());
                pot.set_coefficients(t[2].cast<std::vector<double>>());
                for (auto item : t[3].cast<py::list>()) {
                    auto bh = item.cast<py::tuple>();
                    pot.add_black_hole(bh[0].cast<double>(), bh[1].cast<double>(),
                                       bh[2].cast<double>(), bh[3].cast<double>(),
                                       bh[4].cast<double>());
                }
                return pot;
            }));
    register_orbit_api(disc);

    // Miyamoto-Nagai primitives (unit mass) for building the Gram matrix.
    m.def("mn_potential", &lanfear::mn_potential, py::arg("x"), py::arg("y"),
          py::arg("z"), py::arg("a"), py::arg("b"));
    m.def("mn_density", &lanfear::mn_density, py::arg("R"), py::arg("z"),
          py::arg("a"), py::arg("b"));

    m.def("summary_columns", []() {
        std::vector<std::string> cols;
        for (std::size_t i = 0; i < lanfear::kSummaryCols; ++i)
            cols.emplace_back(lanfear::summary_columns()[i]);
        return cols;
    }, "Column names for the integrate/analyse summary rows.");
}
