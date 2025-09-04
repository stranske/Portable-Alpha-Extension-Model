"""Test to demonstrate improved dependency injection pattern."""

from unittest.mock import Mock

import numpy as np

from pa_core.cli import Dependencies


def test_dependencies_with_default_imports():
    """Test that Dependencies class works with default imports."""
    deps = Dependencies()

    # All dependencies should be available
    assert hasattr(deps, "build_from_config")
    assert hasattr(deps, "export_to_excel")
    assert hasattr(deps, "draw_financing_series")
    assert hasattr(deps, "draw_joint_returns")
    assert hasattr(deps, "build_cov_matrix")
    assert hasattr(deps, "simulate_agents")

    # Functions should be callable (not None)
    assert callable(deps.build_from_config)
    assert callable(deps.export_to_excel)
    assert callable(deps.draw_financing_series)
    assert callable(deps.draw_joint_returns)
    assert callable(deps.build_cov_matrix)
    assert callable(deps.simulate_agents)


def test_dependencies_with_explicit_functions():
    """Test that Dependencies class accepts explicit function parameters."""
    # Create mock functions
    mock_build_from_config = Mock(return_value={"test": "agent"})
    mock_export_to_excel = Mock()
    mock_draw_financing_series = Mock(
        return_value=(np.array([[0.001]]), np.array([[0.002]]), np.array([[0.003]]))
    )
    mock_draw_joint_returns = Mock(
        return_value=(
            np.array([[0.01]]),
            np.array([[0.02]]),
            np.array([[0.03]]),
            np.array([[0.04]]),
        )
    )
    mock_build_cov_matrix = Mock(return_value=np.array([[0.01]]))
    mock_simulate_agents = Mock(return_value={"test_agent": np.array([[0.05]])})

    # Create Dependencies with explicit functions
    deps = Dependencies(
        build_from_config=mock_build_from_config,
        export_to_excel=mock_export_to_excel,
        draw_financing_series=mock_draw_financing_series,
        draw_joint_returns=mock_draw_joint_returns,
        build_cov_matrix=mock_build_cov_matrix,
        simulate_agents=mock_simulate_agents,
    )

    # Verify that the provided functions are used
    assert deps.build_from_config is mock_build_from_config
    assert deps.export_to_excel is mock_export_to_excel
    assert deps.draw_financing_series is mock_draw_financing_series
    assert deps.draw_joint_returns is mock_draw_joint_returns
    assert deps.build_cov_matrix is mock_build_cov_matrix
    assert deps.simulate_agents is mock_simulate_agents


def test_dependencies_mixed_default_and_explicit():
    """Test Dependencies with some explicit and some default functions."""
    # Only provide some functions, others should use defaults
    mock_simulate_agents = Mock(return_value={"mock_agent": np.array([[0.05]])})

    deps = Dependencies(simulate_agents=mock_simulate_agents)

    # Provided function should be the mock
    assert deps.simulate_agents is mock_simulate_agents

    # Others should be the default imports (not None and callable)
    assert deps.build_from_config is not None
    assert callable(deps.build_from_config)
    assert deps.export_to_excel is not None
    assert callable(deps.export_to_excel)


def test_improved_testability_example():
    """Demonstrate how the new pattern improves testability."""
    # This shows how a test could mock just the simulation function
    # without needing to mock entire modules

    def mock_simulate_agents(agents, r_beta, r_H, r_E, r_M, f_int, f_ext, f_act):
        """Mock simulation that returns predictable results."""
        return {
            "InternalPA": np.array([[0.01, 0.02]]),
            "ExternalPA": np.array([[0.015, 0.025]]),
            "ActiveExt": np.array([[0.02, 0.03]]),
        }

    # Create a Dependencies instance with just the simulation mocked
    deps = Dependencies(simulate_agents=mock_simulate_agents)

    # The mock is used for simulation
    assert deps.simulate_agents is mock_simulate_agents

    # All other dependencies remain real implementations
    # This allows testing specific components while keeping others real
    assert callable(deps.build_from_config)
    assert callable(deps.draw_joint_returns)

    # Test that the mock works as expected
    mock_result = deps.simulate_agents(
        {},
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
    )

    assert "InternalPA" in mock_result
    assert "ExternalPA" in mock_result
    assert "ActiveExt" in mock_result
    assert mock_result["InternalPA"].shape == (1, 2)


if __name__ == "__main__":
    test_dependencies_with_default_imports()
    test_dependencies_with_explicit_functions()
    test_dependencies_mixed_default_and_explicit()
    test_improved_testability_example()
    print("All dependency injection tests passed!")
