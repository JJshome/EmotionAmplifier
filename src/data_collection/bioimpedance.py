"""
Bioimpedance Processing Module

This module handles the acquisition and processing of frequency-scanned bioimpedance 
measurements for emotion detection. It includes specialized signal processing algorithms 
to extract emotional states from subtle changes in bioimpedance patterns.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BioimpedanceProcessor:
    """
    Processes frequency-scanned bioimpedance data to extract emotional information.
    
    The bioimpedance spectroscopy approach measures tissue impedance across multiple 
    frequencies, which correlates with emotional states through autonomic nervous 
    system changes and neural activity patterns.
    """
    
    def __init__(self, 
                 frequency_range: Tuple[float, float] = (1000.0, 100000.0), 
                 frequency_steps: int = 50,
                 sample_rate: float = 100.0,
                 electrode_config: str = 'wrist-wrist',
                 filter_settings: Optional[Dict] = None):
        """
        Initialize the bioimpedance processor.
        
        Args:
            frequency_range: Tuple of (min_freq, max_freq) in Hz
            frequency_steps: Number of frequency steps in the scan
            sample_rate: Sampling rate in Hz
            electrode_config: Electrode configuration (e.g., 'wrist-wrist', 'wrist-chest')
            filter_settings: Optional dictionary of filter parameters
        """
        self.min_freq, self.max_freq = frequency_range
        self.frequency_steps = frequency_steps
        self.sample_rate = sample_rate
        self.electrode_config = electrode_config
        
        # Default filter settings if none provided
        self.filter_settings = filter_settings or {
            'lowpass_cutoff': 10.0,
            'highpass_cutoff': 0.1,
            'notch_freq': 50.0,
            'filter_order': 4
        }
        
        # Calculate the frequency points for scanning
        self.frequencies = np.logspace(
            np.log10(self.min_freq),
            np.log10(self.max_freq),
            num=self.frequency_steps
        )
        
        # Initialize calibration parameters
        self.calibration_data = None
        logger.info(f"Bioimpedance processor initialized with {frequency_steps} frequency steps")
        
    def calibrate(self, baseline_data: np.ndarray) -> bool:
        """
        Calibrate the processor with baseline measurements.
        
        Args:
            baseline_data: Array of baseline impedance measurements
            
        Returns:
            bool: True if calibration was successful
        """
        if baseline_data.shape[0] != len(self.frequencies):
            logger.error(f"Calibration data shape mismatch: got {baseline_data.shape[0]}, expected {len(self.frequencies)}")
            return False
            
        self.calibration_data = baseline_data
        logger.info("Bioimpedance processor calibrated successfully")
        return True
        
    def process_raw_data(self, raw_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process raw bioimpedance data to extract impedance and phase at each frequency.
        
        Args:
            raw_data: Raw bioimpedance measurements (complex values)
            
        Returns:
            Dict containing processed impedance magnitude, phase, and derived parameters
        """
        # Validate input data
        if raw_data.shape[0] != len(self.frequencies):
            raise ValueError(f"Data shape mismatch: got {raw_data.shape[0]}, expected {len(self.frequencies)}")
        
        # Calculate impedance magnitude and phase
        impedance_magnitude = np.abs(raw_data)
        impedance_phase = np.angle(raw_data, deg=True)
        
        # Normalize with calibration data if available
        if self.calibration_data is not None:
            impedance_magnitude = impedance_magnitude / self.calibration_data
        
        # Calculate Cole parameters
        cole_params = self._calculate_cole_parameters(impedance_magnitude, impedance_phase)
        
        # Calculate emotional indices based on impedance patterns
        emotion_indices = self._calculate_emotion_indices(impedance_magnitude, impedance_phase, cole_params)
        
        return {
            'impedance_magnitude': impedance_magnitude,
            'impedance_phase': impedance_phase,
            'cole_parameters': cole_params,
            'emotion_indices': emotion_indices
        }
    
    def _calculate_cole_parameters(self, 
                                   magnitude: np.ndarray, 
                                   phase: np.ndarray) -> Dict[str, float]:
        """
        Calculate Cole model parameters from impedance data.
        
        The Cole model (R_∞, R_0, α, τ) represents tissue electrical properties
        and can be correlated with physiological and emotional states.
        
        Args:
            magnitude: Impedance magnitude array
            phase: Impedance phase array
            
        Returns:
            Dictionary of Cole model parameters
        """
        # Simplified Cole parameter estimation
        # In a full implementation, this would use curve fitting to the Cole model
        
        r_infinity = np.min(magnitude)  # High frequency resistance limit
        r_zero = np.max(magnitude)      # Low frequency resistance limit
        
        # Find frequency with maximum phase angle
        max_phase_idx = np.argmax(np.abs(phase))
        tau = 1.0 / (2.0 * np.pi * self.frequencies[max_phase_idx])
        
        # Alpha parameter (tissue heterogeneity)
        alpha = 0.7  # Simplified - would be fitted in full implementation
        
        return {
            'r_infinity': r_infinity,
            'r_zero': r_zero,
            'tau': tau,
            'alpha': alpha
        }
    
    def _calculate_emotion_indices(self, 
                                  magnitude: np.ndarray, 
                                  phase: np.ndarray,
                                  cole_params: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate emotion indices based on bioimpedance patterns.
        
        Args:
            magnitude: Impedance magnitude array
            phase: Impedance phase array
            cole_params: Cole model parameters
            
        Returns:
            Dictionary of emotion indices
        """
        # These indices are based on research correlating bioimpedance 
        # with autonomic nervous system activity and emotional states

        # Calculate the dispersion width (related to sympathetic activation)
        dispersion_width = cole_params['r_zero'] - cole_params['r_infinity']
        
        # Calculate phase response features
        phase_max = np.max(np.abs(phase))
        phase_variance = np.var(phase)
        
        # Calculate impedance variability (related to emotional arousal)
        magnitude_slope = np.polyfit(
            np.log10(self.frequencies),
            magnitude, 
            deg=1
        )[0]
        
        # Map these features to emotional indices
        # These mappings are simplified; a real implementation would use 
        # validated models based on clinical studies
        
        # Sympathetic activation (related to arousal/excitement)
        sympathetic_index = self._normalize_value(dispersion_width, 0.1, 5.0)
        
        # Parasympathetic activation (related to calmness/relaxation)
        parasympathetic_index = 1.0 - self._normalize_value(magnitude_slope, -1.0, 0.0)
        
        # Emotional intensity
        intensity_index = self._normalize_value(phase_max, 5.0, 30.0)
        
        # Emotional complexity/variability
        complexity_index = self._normalize_value(phase_variance, 0.0, 100.0)
        
        return {
            'sympathetic_activation': sympathetic_index,
            'parasympathetic_activation': parasympathetic_index,
            'emotional_intensity': intensity_index,
            'emotional_complexity': complexity_index
        }
    
    def _normalize_value(self, value: float, min_val: float, max_val: float) -> float:
        """
        Normalize a value to the range [0, 1].
        
        Args:
            value: Value to normalize
            min_val: Minimum expected value
            max_val: Maximum expected value
            
        Returns:
            Normalized value between 0 and 1
        """
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))
    
    def get_frequency_response(self, 
                              data: Dict[str, np.ndarray]) -> Dict[str, List]:
        """
        Get the frequency response data for visualization or further analysis.
        
        Args:
            data: Processed bioimpedance data
            
        Returns:
            Dictionary with frequencies and corresponding parameters
        """
        return {
            'frequencies': self.frequencies.tolist(),
            'magnitude': data['impedance_magnitude'].tolist(),
            'phase': data['impedance_phase'].tolist()
        }
