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

"""
OpenFermion

For more information, examples, or tutorials visit our website:

www.openfermion.org
"""
from openfermion.chem import (
    make_atom,
    make_atomic_lattice,
    make_atomic_ring,
    angstroms_to_bohr,
    bohr_to_angstroms,
    MolecularData,
    name_molecule,
    geometry_from_file,
    load_molecular_hamiltonian,
    periodic_table,
    periodic_hash_table,
    periodic_polarization,
    geometry_from_pubchem,
    make_reduced_hamiltonian,
)

from openfermion.circuits import (
    FSWAP,
    FSwapPowGate,
    Rxxyy,
    Ryxxy,
    Rzz,
    rot11,
    state_swap_eigen_component,
    fermionic_simulation_gates_from_interaction_operator,
    sum_of_interaction_operator_gate_generators,
    ParityPreservingFermionicGate,
    InteractionOperatorFermionicGate,
    QuadraticFermionicSimulationGate,
    CubicFermionicSimulationGate,
    QuarticFermionicSimulationGate,
    DoubleExcitation,
    DoubleExcitationGate,
    rot111,
    CRxxyy,
    CRyxxy,
    preprocess_lcu_coefficients_for_reversible_sampling,
    lambda_norm,
    get_chemist_two_body_coefficients,
    low_rank_two_body_decomposition,
    prepare_one_body_squared_evolution,
    bogoliubov_transform,
    ffft,
    optimal_givens_decomposition,
    prepare_gaussian_state,
    swap_network,
    gaussian_state_preparation_circuit,
    slater_determinant_preparation_circuit,
    jw_get_gaussian_state,
    jw_slater_determinant,
    trotter_operator_grouping,
    pauli_exp_to_qasm,
    trotterize_exp_qubop_to_qasm,
    uccsd_generator,
    uccsd_convert_amplitude_format,
    uccsd_singlet_paramsize,
    uccsd_singlet_get_packed_amplitudes,
    uccsd_singlet_generator,
    LINEAR_SWAP_NETWORK,
    LinearSwapNetworkTrotterAlgorithm,
    LOW_RANK,
    LowRankTrotterAlgorithm,
    SplitOperatorTrotterAlgorithm,
    SplitOperatorTrotterStep,
    SymmetricSplitOperatorTrotterStep,
    ControlledAsymmetricSplitOperatorTrotterStep,
    AsymmetricSplitOperatorTrotterStep,
    ControlledSymmetricSplitOperatorTrotterStep,
    SPLIT_OPERATOR,
    diagonal_coulomb_potential_and_kinetic_terms_as_arrays,
    bit_mask_of_modes_acted_on_by_fermionic_terms,
    split_operator_trotter_error_operator_diagonal_two_body,
    fermionic_swap_trotter_error_operator_diagonal_two_body,
    simulation_ordered_grouped_hubbard_terms_with_info,
    low_depth_second_order_trotter_error_operator,
    low_depth_second_order_trotter_error_bound,
    simulation_ordered_grouped_low_depth_terms_with_info,
    stagger_with_info,
    simulate_trotter,
    TrotterAlgorithm,
    TrotterStep,
    error_bound,
    error_operator,
    trotter_steps_required,
    vpe_single_circuit,
    vpe_circuits_single_timestep,
    standard_vpe_rotation_set,
)

from openfermion.functionals import contextuality

from openfermion.hamiltonians import (
    FermiHubbardModel,
    rhf_minimization,
    HartreeFockFunctional,
    rhf_params_to_matrix,
    get_matrix_of_eigs,
    generate_hamiltonian,
    bose_hubbard,
    fermi_hubbard,
    dual_basis_kinetic,
    dual_basis_potential,
    dual_basis_jellium_model,
    jellium_model,
    jordan_wigner_dual_basis_jellium,
    hypercube_grid_with_given_wigner_seitz_radius_and_filling,
    plane_wave_kinetic,
    plane_wave_potential,
    wigner_seitz_length_scale,
    hartree_fock_state_jellium,
    mean_field_dwave,
    dual_basis_external_potential,
    plane_wave_external_potential,
    plane_wave_hamiltonian,
    jordan_wigner_dual_basis_hamiltonian,
    s_plus_operator,
    s_squared_operator,
    sx_operator,
    sy_operator,
    sz_operator,
    majorana_operator,
    number_operator,
)

from openfermion.linalg import (
    Davidson,
    DavidsonOptions,
    DavidsonError,
    QubitDavidson,
    SparseDavidson,
    erpa_eom_hamiltonian,
    singlet_erpa,
    givens_decomposition,
    givens_rotate,
    double_givens_rotate,
    givens_decomposition_square,
    givens_matrix_elements,
    fermionic_gaussian_decomposition,
    generate_linear_qubit_operator,
    LinearQubitOperator,
    LinearQubitOperatorOptions,
    ParallelLinearQubitOperator,
    wrapped_kronecker,
    kronecker_operators,
    jordan_wigner_ladder_sparse,
    jordan_wigner_sparse,
    qubit_operator_sparse,
    eigenspectrum,
    get_linear_qubit_operator_diagonal,
    jw_configuration_state,
    jw_hartree_fock_state,
    jw_number_indices,
    jw_sz_indices,
    jw_number_restrict_operator,
    jw_sz_restrict_operator,
    jw_number_restrict_state,
    jw_sz_restrict_state,
    jw_get_ground_state_at_particle_number,
    jw_sparse_givens_rotation,
    jw_sparse_particle_hole_transformation_last_mode,
    get_density_matrix,
    get_ground_state,
    sparse_eigenspectrum,
    expectation,
    variance,
    expectation_computational_basis_state,
    expectation_db_operator_with_pw_basis_state,
    expectation_one_body_db_operator_computational_basis_state,
    expectation_two_body_db_operator_computational_basis_state,
    expectation_three_body_db_operator_computational_basis_state,
    get_gap,
    inner_product,
    boson_ladder_sparse,
    single_quad_op_sparse,
    boson_operator_sparse,
    get_sparse_operator,
    get_number_preserving_sparse_operator,
    generate_parity_permutations,
    fit_known_frequencies,
    prony,
    wedge,
)

from openfermion.measurements import (
    pair_within,
    pair_between,
    pair_within_simultaneously,
    pair_within_simultaneously_binned,
    pair_within_simultaneously_symmetric,
    apply_constraints,
    constraint_matrix,
    linearize_term,
    unlinearize_term,
    get_interaction_rdm,
    one_body_fermion_constraints,
    two_body_fermion_constraints,
    binary_partition_iterator,
    partition_iterator,
    pauli_string_iterator,
    PhaseFitEstimator,
    get_phase_function,
)

from openfermion.ops import (
    BinaryPolynomial,
    BinaryPolynomialError,
    BosonOperator,
    FermionOperator,
    IsingOperator,
    MajoranaOperator,
    QuadOperator,
    QubitOperator,
    SymbolicOperator,
    double_decoding,
    shift_decoder,
    BinaryCode,
    BinaryCodeError,
    PolynomialTensor,
    PolynomialTensorError,
    general_basis_change,
    DiagonalCoulombHamiltonian,
    InteractionOperator,
    InteractionOperatorError,
    InteractionRDM,
    InteractionRDMError,
    QuadraticHamiltonian,
    QuadraticHamiltonianError,
)

from openfermion.testing import (
    random_interaction_operator_term,
    haar_random_vector,
    random_antisymmetric_matrix,
    random_hermitian_matrix,
    random_unitary_matrix,
    random_qubit_operator,
    random_diagonal_coulomb_hamiltonian,
    random_interaction_operator,
    random_quadratic_hamiltonian,
    EqualsTester,
    module_importable,
    assert_equivalent_repr,
    assert_implements_consistent_protocols,
)

from openfermion.third_party import (
    fixed_trace_positive_projection,
    map_to_tensor,
    map_to_matrix,
    higham_polynomial,
    higham_root,
    heaviside,
)

from openfermion.transforms import (
    chemist_ordered,
    normal_ordered,
    normal_ordered_ladder_term,
    normal_ordered_quad_term,
    reorder,
    linearize_decoder,
    checksum_code,
    bravyi_kitaev_code,
    jordan_wigner_code,
    parity_code,
    weight_one_binary_addressing_code,
    weight_one_segment_code,
    weight_two_segment_code,
    interleaved_code,
    binary_code_transform,
    bravyi_kitaev_fast,
    bravyi_kitaev_fast_interaction_op,
    bravyi_kitaev_fast_edge_matrix,
    bravyi_kitaev,
    bravyi_kitaev_tree,
    get_fermion_operator,
    get_boson_operator,
    get_majorana_operator,
    get_quad_operator,
    check_no_sympy,
    FenwickNode,
    FenwickTree,
    jordan_wigner,
    qubit_operator_to_pauli_sum,
    reverse_jordan_wigner,
    symmetry_conserving_bravyi_kitaev,
    verstraete_cirac_2d_square,
    vertical_edges_snake,
    get_interaction_operator,
    get_diagonal_coulomb_hamiltonian,
    get_molecular_data,
    get_quadratic_hamiltonian,
    fourier_transform,
    inverse_fourier_transform,
    freeze_orbitals,
    prune_unused_indices,
    project_onto_sector,
    projection_error,
    rotate_qubit_by_pauli,
    StabilizerError,
    check_commuting_stabilizers,
    check_stabilizer_linearity,
    reduce_number_of_terms,
    taper_off_qubits,
    fix_single_term,
    mccoy,
    weyl_polynomial_quantization,
    symmetric_ordering,
)

from openfermion.utils import (
    bch_expand,
    amplitude_damping_channel,
    dephasing_channel,
    depolarizing_channel,
    HubbardSquareLattice,
    SpinPairs,
    Spin,
    anticommutator,
    commutator,
    double_commutator,
    Grid,
    up_index,
    down_index,
    up_then_down,
    count_qubits,
    get_file_path,
    hermitian_conjugated,
    is_hermitian,
    is_identity,
    load_operator,
    save_operator,
    kronecker_delta,
    map_two_pdm_to_two_hole_dm,
    map_two_pdm_to_one_pdm,
    map_one_pdm_to_one_hole_dm,
    map_one_hole_dm_to_one_pdm,
    map_two_pdm_to_particle_hole_dm,
    map_two_hole_dm_to_two_pdm,
    map_two_hole_dm_to_one_hole_dm,
    map_particle_hole_dm_to_one_pdm,
    map_particle_hole_dm_to_two_pdm,
)

from ._version import __version__
