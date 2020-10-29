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
"""Classes to assist in processing of VPE (ArXiv:2010.02538) signal data"""

import abc
from typing import Tuple, Optional
import numpy

from openfermion.linalg import fit_known_frequencies


class _VPEEstimator(abc.ABCMeta):
    """Generic class for any VPE estimator"""

    @abc.abstractmethod
    def get_simulation_points(self) -> numpy.ndarray:
        """Generates time points for estimation

        VPE requires estimating the phase function g(t) at multiple points t,
        and some care in choosing these points is needed to prevent aliasing.
        This should be taken care of in the estimator.

        Returns:
            times: a set of times t that g(t) should be estimated at.
        """

    @abc.abstractmethod
    def get_expectation_value(self,
                              phase_function: numpy.ndarray) -> numpy.ndarray:
        """Estimates expectation values from an input phase function

        Given a phase function g(t), estimates the expectation value <H> of the
        operator used to generate it on the initial state rho

        $g(t) = Trace[e^{-iHt} (|psi_r><psi_r| + rho) e^{iHt} |0><1|])$

        Arguments:
            phase_function [numpy.ndarray] -- The phase function g(t)

        Returns:
            expectation value [numpy.ndarray] -- <H>.
        """


class PhaseFitEstimator(_VPEEstimator):
    """A VPE estimator that works by fitting a set of known frequencies.

    A Hamiltonian being fast-forwardable is equivalent to its spectral
    decomposition being known. This means that the only information to
    be obtained from QPE is the amplitudes. This estimator proceeds
    by a simple least-squares fit to obtain the amplitudes, and then
    outputs the expectation values.
    """

    def __init__(self, evals: numpy.ndarray, ref_eval: float = 0):
        """
        Arguments:
            evals [numpy.ndarray] -- The (known) eigenvalues of the target
                operator
            ref_eval [numpy.ndarray] -- The eigenvalue of the reference state.
                When using a control qubit for QPE, this should be set to 0.
        """
        self.evals = evals
        self.ref_eval = ref_eval

    def get_simulation_points(self) -> numpy.ndarray:
        """Generates time points for estimation

        VPE requires estimating the phase function g(t) at multiple points t,
        and some care in choosing these points is needed to prevent aliasing.
        This should be taken care of in the estimator.

        In this case, we fit len(self.energies) complex amplitudes to a complex
        valued signal, we need precisely this number of points in the signal.
        Then, to prevent aliasing, we need to make sure that the time step
        dt < 2*pi / (E_max-E_min). Here, we choose dt = pi / (E_max-E_min).
        (Importantly, for Pauli operators this reproduces the H test.)

        Returns:
            times: a set of times t that g(t) should be estimated at.
        """
        numsteps = len(self.evals)
        step_size = numpy.pi / (max(self.evals) - min(self.evals))
        maxtime = step_size * (numsteps - 1)
        times = numpy.linspace(0, maxtime, numsteps)
        return times

    def get_amplitudes(self,
                       phase_function: numpy.ndarray) -> numpy.ndarray:
        """Fits the amplitudes in the phase function to the input signal data.

        Arguments:
            phase_function [numpy.ndarray] -- Phase function input

        Returns:
            amplitudes [numpy.ndarray] -- Fitted estimates of the amplitudes
                of the given frequencies (in the same order as in self.energies)
        """
        times = self.get_simulation_points()
        phase_function_shifted = numpy.array(phase_function) *\
            numpy.exp(1j * times * self.ref_eval)
        amplitudes = fit_known_frequencies(
            phase_function_shifted, times, self.evals)
        return amplitudes

    def get_expectation_value(self,
                              phase_function: numpy.ndarray) -> numpy.ndarray:
        """Estates expectation values via amplitude fitting of known frequencies

        Arguments:
            phase_function [numpy.ndarray] -- The phase function obtained in
                experiment

        Returns:
            expectation_value [float] -- the estimated expectation value
        """
        amplitudes = self.get_amplitudes(phase_function)
        expectation_value = numpy.dot(
            numpy.abs(amplitudes),
            self.evals) / numpy.sum(numpy.abs(amplitudes))
        return expectation_value


def get_phase_function(measurements: Tuple,
                       rotation_set: Optional[Tuple] = None):
    raise NotImplementedError
