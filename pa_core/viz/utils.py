"""Utility functions for visualization data handling."""

from __future__ import annotations

from typing import Union
import numpy as np
import pandas as pd


def safe_to_numpy(
    data: Union[pd.Series, pd.DataFrame], 
    fillna_value: float = 0.0
) -> np.ndarray:
    """
    Safely convert pandas Series or DataFrame to numpy array with fallback handling.
    
    This helper function handles the common pattern of converting pandas objects
    to numpy arrays while gracefully handling non-numeric data or problematic values
    (NaN, inf) by using fillna as a fallback when issues are detected.
    
    Args:
        data: The pandas Series or DataFrame to convert
        fillna_value: Value to use when filling NaN/inf values in fallback (default: 0.0)
        
    Returns:
        numpy array representation of the data
        
    Raises:
        ValueError: If the conversion fails even after fallback handling
    """
    try:
        result = data.to_numpy()
        # Check for problematic values that might cause issues downstream
        if np.any(np.isinf(result)) or np.any(np.isnan(result)):
            # Use fallback handling
            clean_data = data.fillna(fillna_value).replace([np.inf, -np.inf], fillna_value)
            return clean_data.to_numpy()
        return result
    except (ValueError, TypeError):
        # Handle potential issues with non-numeric data or conversion errors  
        try:
            clean_data = data.fillna(fillna_value).replace([np.inf, -np.inf], fillna_value)
            return clean_data.to_numpy()
        except Exception as e:
            raise ValueError(f"Failed to convert data to numpy array: {e}") from e