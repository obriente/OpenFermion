#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""tests for sparse_qpe.py"""

import unittest
import numpy

from .sparse_qpe import (_next_alias_number, _alias_region_right_side,
                         _alias_region_left_side, beta_finder, match_phases,
                         abs_phase_difference)


class TestBetaFinder(unittest.TestCase):
    def test_alias_region_generation(self):
        phases = [0.1, 0.5, 0.3, 0.9, 1.1]
        error = 1 / 250
        prev_multiplier = 10
        max_beta = 700
        for j, phase1 in enumerate(phases):
            for phase2 in phases[j + 1:]:
                phase_difference = abs(phase1 - phase2)
                max_alias_number = _next_alias_number(prev_multiplier,
                                                      max_beta, error,
                                                      phase_difference)
                last_rhs = None
                for alias_number in range(
                        int(numpy.floor(max_alias_number - 10)),
                        int(numpy.ceil(max_alias_number + 1))):
                    lhs = _alias_region_left_side(alias_number,
                                                  prev_multiplier, error,
                                                  phase_difference)
                    rhs = _alias_region_right_side(alias_number,
                                                   prev_multiplier, error,
                                                   phase_difference)
                    for test_val in numpy.linspace(lhs + 1e-6, rhs - 1e-6, 10):
                        zoomed_phase1 = (
                            (phase1 * prev_multiplier * test_val) %
                            (2 * numpy.pi))
                        zoomed_phase2 = (
                            (phase2 * prev_multiplier * test_val) %
                            (2 * numpy.pi))
                        phase_diff = abs_phase_difference(
                            zoomed_phase2, zoomed_phase1)
                        self.assertTrue(phase_diff < error + error * test_val)
                    if last_rhs is None:
                        continue
                    for test_val in numpy.linspace(last_rhs + 1e-6, lhs - 1e-6,
                                                   10):
                        zoomed_phase1 = (
                            (phase1 * prev_multiplier * test_val) %
                            (2 * numpy.pi))
                        zoomed_phase2 = (
                            (phase2 * prev_multiplier * test_val) %
                            (2 * numpy.pi))
                        phase_diff = abs_phase_difference(
                            zoomed_phase2, zoomed_phase1)
                        self.assertTrue(phase_diff > error + error * test_val)
                    last_rhs = rhs

    def test_beta_gen(self):
        phases = [0.11, 0.542, 0.33, 0.98, 1.1]
        error = 1 / 250
        prev_multiplier = 10
        beta = beta_finder(phases, error, prev_multiplier)
        resolution = error * (1 + beta)
        for j, phase1 in enumerate(phases):
            for phase2 in phases[j + 1:]:
                zoomed_phase1 = ((phase1 * beta * prev_multiplier) %
                                 (2 * numpy.pi))
                zoomed_phase2 = ((phase2 * beta * prev_multiplier) %
                                 (2 * numpy.pi))
                diff = abs_phase_difference(zoomed_phase1, zoomed_phase2)
                self.assertTrue(diff >= resolution - 1e-6)


class TestMatchPhases(unittest.TestCase):
    def test_matching_full(self):
        phases = [0.11, 0.542, 0.33, 0.98, 1.1]
        delta = 1 / 20
        betas = []
        phase_estimates = []

        phase_estimates = [
            (phase + numpy.random.uniform(-delta / 2, delta / 2)) %
            (2 * numpy.pi) for phase in phases
        ]
        error_estimates = [delta for phase in phases]
        betas.append(beta_finder(phase_estimates, delta, 1))

        for d in range(5):
            multiplier = numpy.prod(betas)
            aliased_phase_estimates = [
                (phase * multiplier +
                 numpy.random.uniform(-delta / 2, delta / 2)) % (2 * numpy.pi)
                for phase in phases
            ]
            aliased_error_estimates = [delta for phase in phases]
            phase_estimates, error_estimates = match_phases(
                phase_estimates, error_estimates, multiplier,
                aliased_phase_estimates, aliased_error_estimates)
            betas.append(
                beta_finder(phase_estimates, delta, multiplier, 1 / delta))

            for phase, error, phase_true in zip(phase_estimates,
                                                error_estimates, phases):
                assert abs(phase - phase_true) < error / 2
