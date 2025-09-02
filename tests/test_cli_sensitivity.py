"""Test sensitivity analysis functionality in CLI."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd

from pa_core.cli import main


def test_sensitivity_flag_added():
    """Test that --sensitivity flag is accepted by argument parser."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = Path(temp_dir) / "config.yml"
        index_file = Path(temp_dir) / "index.csv"
        output_file = Path(temp_dir) / "output.xlsx"
        
        # Create minimal config file (use working config from example)
        config_file.write_text("""
N_SIMULATIONS: 100
N_MONTHS: 12
analysis_mode: returns
external_pa_capital: 100.0
active_ext_capital: 50.0
internal_pa_capital: 150.0
total_fund_capital: 300.0
w_beta_H: 0.5
w_alpha_H: 0.5
theta_extpa: 0.5
active_share: 0.5
mu_H: 0.04
sigma_H: 0.01
mu_E: 0.05
sigma_E: 0.02
mu_M: 0.03
sigma_M: 0.01
""")
        
        # Create minimal index file
        index_data = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=10, freq="ME"),
            "Return": [0.01, -0.02, 0.03, 0.01, -0.01, 0.02, 0.01, 0.00, 0.02, 0.01]
        })
        index_data.to_csv(index_file, index=False)
        
        # Mock the expensive parts of the simulation
        with patch('pa_core.cli.draw_joint_returns') as mock_draws, \
             patch('pa_core.cli.simulate_agents') as mock_simulate, \
             patch('pa_core.cli.create_enhanced_summary') as mock_summary, \
             patch('pa_core.cli.export_to_excel') as mock_export, \
             patch('pa_core.cli.build_from_config') as mock_build_agents, \
             patch('pa_core.cli.build_cov_matrix'), \
             patch('pa_core.cli.draw_financing_series') as mock_financing:

            # Mock the simulation properly
            mock_draws.return_value = ([], [], [], [])  # r_beta, r_H, r_E, r_M
            mock_financing.return_value = ([], [], [])  # f_int, f_ext, f_act
            mock_build_agents.return_value = []
            mock_simulate.return_value = {"Base": [[0.01, 0.02, 0.01]]}
            mock_summary.return_value = pd.DataFrame({
                "Agent": ["Base"],
                "AnnReturn": [8.5],
                "AnnVol": [12.0]
            })
            
            # Test that --sensitivity flag doesn't cause parser error
            try:
                main([
                    "--config", str(config_file),
                    "--index", str(index_file),
                    "--output", str(output_file),
                    "--sensitivity"
                ])
            except SystemExit:
                # CLI should parse arguments without SystemExit due to unknown flag
                pytest.fail("--sensitivity flag was not recognized by argument parser")


def test_sensitivity_analysis_execution():
    """Test that sensitivity analysis actually runs when --sensitivity is provided."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = Path(temp_dir) / "config.yml"
        index_file = Path(temp_dir) / "index.csv"
        output_file = Path(temp_dir) / "output.xlsx"
        
        # Create minimal config file (use working config from example)
        config_file.write_text("""
N_SIMULATIONS: 100
N_MONTHS: 12
analysis_mode: returns
external_pa_capital: 100.0
active_ext_capital: 50.0
internal_pa_capital: 150.0
total_fund_capital: 300.0
w_beta_H: 0.5
w_alpha_H: 0.5
theta_extpa: 0.5
active_share: 0.5
mu_H: 0.04
sigma_H: 0.01
mu_E: 0.05
sigma_E: 0.02
mu_M: 0.03
sigma_M: 0.01
""")
        
        # Create minimal index file
        index_data = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=10, freq="ME"),
            "Return": [0.01, -0.02, 0.03, 0.01, -0.01, 0.02, 0.01, 0.00, 0.02, 0.01]
        })
        index_data.to_csv(index_file, index=False)
        
        with patch('pa_core.cli.draw_joint_returns') as mock_draws, \
             patch('pa_core.cli.simulate_agents') as mock_simulate, \
             patch('pa_core.cli.create_enhanced_summary') as mock_summary, \
             patch('pa_core.cli.export_to_excel') as mock_export, \
             patch('pa_core.cli.build_from_config') as mock_build_agents, \
             patch('pa_core.cli.build_cov_matrix') as mock_cov, \
             patch('builtins.print') as mock_print:
            
            # Mock simulation results
            mock_summary.return_value = pd.DataFrame({
                "Agent": ["Base"],
                "AnnReturn": [8.5],
                "AnnVol": [12.0]
            })
            
            mock_build_agents.return_value = []
            mock_simulate.return_value = {"Base": [[0.01, 0.02, 0.01]]}
            mock_draws.return_value = ([], [], [], [])
            
            main([
                "--config", str(config_file),
                "--index", str(index_file),
                "--output", str(output_file),
                "--sensitivity"
            ])
            
            # Check that sensitivity analysis messages were printed
            print_calls = [str(call) for call in mock_print.call_args_list]
            sensitivity_messages = [call for call in print_calls if "sensitivity" in call.lower() or "🔍" in call]
            
            assert len(sensitivity_messages) > 0, "Sensitivity analysis messages not found in output"


def test_sensitivity_analysis_error_logging():
    """Test that parameter evaluation failures are properly logged."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = Path(temp_dir) / "config.yml"
        index_file = Path(temp_dir) / "index.csv"
        output_file = Path(temp_dir) / "output.xlsx"
        
        # Create config that might cause evaluation errors
        config_file.write_text("""
N_SIMULATIONS: 100
N_MONTHS: 12
analysis_mode: returns
external_pa_capital: 100.0
active_ext_capital: 50.0
internal_pa_capital: 150.0
total_fund_capital: 300.0
w_beta_H: 0.5
w_alpha_H: 0.5
theta_extpa: 0.5
active_share: 0.5
mu_H: 0.04
sigma_H: 0.01
mu_E: 0.05
sigma_E: 0.02
mu_M: 0.03
sigma_M: 0.01
""")
        
        # Create minimal index file
        index_data = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=10, freq="ME"),
            "Return": [0.01, -0.02, 0.03, 0.01, -0.01, 0.02, 0.01, 0.00, 0.02, 0.01]
        })
        index_data.to_csv(index_file, index=False)
        
        with patch('pa_core.cli.create_enhanced_summary') as mock_summary, \
             patch('pa_core.cli.export_to_excel'), \
             patch('pa_core.cli.build_from_config') as mock_build_agents, \
             patch('pa_core.cli.draw_joint_returns') as mock_draws, \
             patch('pa_core.cli.simulate_agents') as mock_simulate, \
             patch('pa_core.cli.build_cov_matrix'), \
             patch('builtins.print') as mock_print:
            
            # Mock main simulation to return valid results
            mock_summary.return_value = pd.DataFrame({
                "Agent": ["Base"],
                "AnnReturn": [8.5],
                "AnnVol": [12.0]
            })
            
            # Mock one of the evaluation steps to raise an exception
            def side_effect_simulate(*args, **kwargs):
                # Raise exception on second and subsequent calls (sensitivity analysis calls)
                if side_effect_simulate.call_count > 1:
                    raise ValueError("Simulated parameter evaluation failure")
                side_effect_simulate.call_count += 1
                return {"Base": [[0.01, 0.02, 0.01]]}
            
            side_effect_simulate.call_count = 0
            mock_simulate.side_effect = side_effect_simulate
            mock_build_agents.return_value = []
            mock_draws.return_value = ([], [], [], [])
            
            main([
                "--config", str(config_file),
                "--index", str(index_file),
                "--output", str(output_file),
                "--sensitivity"
            ])
            
            # Check that failure messages were printed
            print_calls = [str(call) for call in mock_print.call_args_list]
            error_messages = [call for call in print_calls if "failed" in call.lower() or "⚠️" in call]
            
            assert len(error_messages) > 0, "Parameter evaluation failure messages not found in output"