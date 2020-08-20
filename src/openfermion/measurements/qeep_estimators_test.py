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
""" Tests for qeep_estimators.py"""

import unittest
import numpy
from scipy import integrate

from .qeep_estimators import (
    qeep_solve, _bump_function, _bump_fourier_transform,
    weight_function, get_signal_requirements)


class TestQeepSolver(unittest.TestCase):

    def setUp(self):
        self.rng = numpy.random.RandomState(42)

    def add_noise(self, prob, num_samples):
        res = self.rng.binomial(num_samples, prob) / num_samples
        return res

    def get_gk(self, signal_length, phases, amplitudes, num_samples):
        gk_clean = numpy.array([numpy.sum(
            numpy.array(amplitudes) * numpy.exp(1j * numpy.array(phases) * k))
                                for k in range(signal_length+1)])
        if num_samples is None:
            gk_noisy = gk_clean
        else:
            pk_real_clean = 0.5 - 0.5 * numpy.real(gk_clean)
            pk_imag_clean = 0.5 - 0.5 * numpy.imag(gk_clean)
            pk_real_noisy = self.add_noise(pk_real_clean, num_samples)
            pk_imag_noisy = self.add_noise(pk_imag_clean, num_samples)

            gk_noisy = (1 - 2 * pk_real_noisy) + 1j *(1 - 2 * pk_imag_noisy)
        return gk_noisy

    def test_error_bounded(self):
        phases = self.rng.uniform(0, 2*numpy.pi, 5)
        amplitudes = self.rng.uniform(0, 1, 5)
        amplitudes = amplitudes / sum(amplitudes)
        delta = 0.3
        confidence = 0.98
        num_points, signal_length, num_samples = get_signal_requirements(
            confidence, delta)
        signal = self.get_gk(signal_length, phases, amplitudes, num_samples)
        est_probability_vector = numpy.abs(qeep_solve(signal, num_points))
        true_probability_vector = [numpy.sum([weight_function(phase, k, delta) *
                                              amp for phase, amp in
                                              zip(phases, amplitudes)])
                                   for k in range(num_points)]
        error = numpy.sum(numpy.abs(numpy.array(true_probability_vector -
                                                est_probability_vector)))
        self.assertLess(error, delta)

class TestWeightFunctionGeneration(unittest.TestCase):

    def test_bump_function_integral(self):
        self.assertAlmostEqual(integrate.quad(_bump_function, -1, 1)[0], 1)

    def test_bump_function_integration(self):
        self.assertAlmostEqual(_bump_fourier_transform(0),
                               1 / numpy.sqrt(2 * numpy.pi))

    def test_indicator_function_constant(self):
        self.assertAlmostEqual(integrate.quad(_bump_function, 0, 1)[0], 0.5)

    def test_indicator_function_approx_zero(self):
        j = 5
        epsilon = 0.5
        x_vals = numpy.linspace(0, 2*numpy.pi, 500)
        y_vals = [weight_function(k, j, epsilon)
                  for k in x_vals]
        for n in range(500):
            self.assertTrue(((x_vals[n] > (j - 1) * epsilon) &
                             (x_vals[n] < (j + 1) * epsilon)) |
                            (y_vals[n] < 1e-3))
