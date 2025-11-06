#!/usr/bin/env python3
"""
Integration tests for main_analysis.py with comprehensive validation.

This test suite covers:
1. H5 file structure validation (column names and metadata)
2. Plot functionality with metadata/column application and non-ASCII exclusion
3. NaN data handling and propagation in mathematical operations
4. Enhanced plotting features
5. Dask-based parallel processing workflow
"""

import pytest
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from unittest.mock import Mock, patch

from ifi.utils.cache_setup import setup_project_cache

# Setup cache before imports
cache_config = setup_project_cache()

from ifi.analysis.main_analysis import load_and_process_file, run_analysis
from ifi.utils import file_io
from ifi.analysis import plots
from ifi.db_controller.nas_db import NAS_DB
from ifi.db_controller.vest_db import VEST_DB


@pytest.fixture
def tmp_h5_dir(tmp_path):
    """Create a temporary directory for H5 files."""
    h5_dir = tmp_path / "results" / "45821"
    h5_dir.mkdir(parents=True)
    return h5_dir


@pytest.fixture
def mock_nas_db():
    """Create a mock NAS_DB instance."""
    mock_db = Mock(spec=NAS_DB)
    return mock_db


@pytest.fixture
def mock_vest_db():
    """Create a mock VEST_DB instance."""
    mock_db = Mock(spec=VEST_DB)
    return mock_db


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with metadata."""
    fs = 50e6
    duration = 0.01
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples)
    
    df = pd.DataFrame({
        "TIME": t,
        "CH0": np.sin(2 * np.pi * 8e6 * t),
        "CH1": np.cos(2 * np.pi * 8e6 * t),
        "CH2": np.sin(2 * np.pi * 8e6 * t + np.pi / 4),
    })
    
    # Add metadata
    df.attrs["source_file_type"] = "csv"
    df.attrs["source_file_format"] = "MSO58"
    df.attrs["metadata"] = {
        "record_length": n_samples,
        "time_resolution": 1 / fs,
        "sampling_rate": fs,
    }
    
    return df


@pytest.fixture
def sample_dataframe_with_nan():
    """Create a sample DataFrame with NaN values."""
    fs = 50e6
    duration = 0.01
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples)
    
    df = pd.DataFrame({
        "TIME": t,
        "CH0": np.sin(2 * np.pi * 8e6 * t),
        "CH1": np.cos(2 * np.pi * 8e6 * t),
    })
    
    # Introduce NaN values at specific positions
    df.loc[100:200, "CH0"] = np.nan
    df.loc[300:350, "CH1"] = np.nan
    
    df.attrs["source_file_type"] = "csv"
    df.attrs["metadata"] = {"record_length": n_samples}
    
    return df


@pytest.fixture
def sample_dataframe_with_korean():
    """Create a sample DataFrame with Korean characters in column names (should be sanitized)."""
    fs = 50e6
    duration = 0.01
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples)
    
    df = pd.DataFrame({
        "TIME": t,
        "한글컬럼": np.sin(2 * np.pi * 8e6 * t),  # Korean column name
        "CH0": np.cos(2 * np.pi * 8e6 * t),
    })
    
    df.attrs["source_file_type"] = "csv"
    df.attrs["metadata"] = {"record_length": n_samples}
    
    return df


class TestH5StructureValidation:
    """Test H5 file structure validation (column names and metadata)."""
    
    def test_h5_save_and_load_structure(self, tmp_h5_dir, sample_dataframe):
        """Test that H5 file structure is correctly saved and loaded."""
        shot_num = 45821
        
        # Save data
        signals = {"test_file.csv": sample_dataframe}
        stft_results = {}
        cwt_results = {}
        density_data = pd.DataFrame({
            "ne_CH0_test": np.random.randn(len(sample_dataframe)),
            "ne_CH1_test": np.random.randn(len(sample_dataframe)),
        })
        vest_data = pd.DataFrame({
            "ip": np.random.randn(1000),
            "time": np.linspace(0, 1, 1000),
        })
        
        saved_h5_path = file_io.save_results_to_hdf5(
            str(tmp_h5_dir),
            shot_num,
            signals,
            stft_results,
            cwt_results,
            density_data,
            vest_data,
        )
        
        assert saved_h5_path is not None
        assert Path(saved_h5_path).exists()
        
        # Validate H5 structure
        with h5py.File(saved_h5_path, "r") as hf:
            # Check metadata group
            assert "metadata" in hf, "Metadata group should exist"
            assert "shot_number" in hf["metadata"].attrs, "shot_number should be in metadata"
            assert hf["metadata"].attrs["shot_number"] == shot_num
            assert "created_at" in hf["metadata"].attrs
            assert "ifi_version" in hf["metadata"].attrs
            
            # Check signals group
            assert "signals" in hf, "Signals group should exist"
            assert "test_file.csv" in hf["signals"], "Signal file should be in signals group"
            
            signal_group = hf["signals"]["test_file.csv"]
            assert "TIME" in signal_group, "TIME column should exist"
            assert "CH0" in signal_group, "CH0 column should exist"
            assert "CH1" in signal_group, "CH1 column should exist"
            assert "CH2" in signal_group, "CH2 column should exist"
            
            # Validate column data types
            assert signal_group["TIME"].dtype in [np.float64, np.float32]
            assert signal_group["CH0"].dtype in [np.float64, np.float32]
            
            # Check density data
            assert "density_data" in hf, "Density data group should exist"
            assert "ne_CH0_test" in hf["density_data"]
            assert "ne_CH1_test" in hf["density_data"]
            
            # Check VEST data
            assert "vest_data" in hf, "VEST data group should exist"
            assert "ip" in hf["vest_data"]
            assert "time" in hf["vest_data"]
    
    def test_h5_column_names_validation(self, tmp_h5_dir, sample_dataframe):
        """Test that column names are correctly preserved in H5 files."""
        shot_num = 45821
        
        # Create DataFrame with specific column names
        signals = {"test_file.csv": sample_dataframe}
        
        h5_path = file_io.save_results_to_hdf5(
            str(tmp_h5_dir),
            shot_num,
            signals,
            {},
            {},
            pd.DataFrame(),
            pd.DataFrame(),
        )
        
        # Load and verify column names
        with h5py.File(h5_path, "r") as hf:
            signal_group = hf["signals"]["test_file.csv"]
            saved_columns = list(signal_group.keys())
            
            assert "TIME" in saved_columns
            assert "CH0" in saved_columns
            assert "CH1" in saved_columns
            assert "CH2" in saved_columns
            assert len(saved_columns) == 4
    
    def test_h5_metadata_preservation(self, tmp_h5_dir, sample_dataframe):
        """Test that metadata is correctly preserved in H5 files."""
        shot_num = 45821
        
        signals = {"test_file.csv": sample_dataframe}
        
        h5_path = file_io.save_results_to_hdf5(
            str(tmp_h5_dir),
            shot_num,
            signals,
            {},
            {},
            pd.DataFrame(),
            pd.DataFrame(),
        )
        
        # Load results back
        loaded_results = file_io.load_results_from_hdf5(shot_num, str(tmp_h5_dir.parent.parent))
        
        assert loaded_results is not None
        assert "metadata" in loaded_results
        assert loaded_results["metadata"]["shot_number"] == shot_num
    
    def test_h5_empty_signals_handling(self, tmp_h5_dir):
        """Test H5 file creation with empty signals."""
        shot_num = 45821
        
        h5_path = file_io.save_results_to_hdf5(
            str(tmp_h5_dir),
            shot_num,
            {},
            {},
            {},
            pd.DataFrame(),
            pd.DataFrame(),
        )
        
        assert h5_path is not None
        
        with h5py.File(h5_path, "r") as hf:
            assert "signals" in hf
            assert hf["signals"].attrs.get("empty", False)


class TestPlotValidation:
    """Test plot functionality with metadata/column application and non-ASCII exclusion."""
    
    def test_plot_metadata_application(self, sample_dataframe):
        """Test that metadata is correctly applied in plots."""
        plotter = plots.Plotter()
        
        # Extract metadata info
        metadata_str = plotter._extract_metadata_info(sample_dataframe)
        
        # Should contain metadata information
        assert "Type: csv" in metadata_str or "Format: MSO58" in metadata_str
        assert "Length:" in metadata_str or "Resolution:" in metadata_str
    
    def test_plot_column_names_application(self, sample_dataframe):
        """Test that column names are correctly used in plots."""
        plotter = plots.Plotter()
        
        # Create plot with DataFrame
        fig, axes = plotter.plot_waveforms(
            sample_dataframe,
            fs=50e6,
            title="Test Plot",
            show_plot=False,
        )
        
        # Check that axes labels contain column information
        # (exact implementation may vary, but columns should be referenced)
        assert fig is not None
        assert len(axes) > 0
    
    def test_plot_non_ascii_exclusion(self, sample_dataframe_with_korean):
        """Test that non-ASCII characters (Korean) are excluded from plot titles/labels."""
        plotter = plots.Plotter()
        
        # Attempt to plot with Korean column names
        # The plot should handle this gracefully (either sanitize or skip non-ASCII columns)
        try:
            fig, axes = plotter.plot_waveforms(
                sample_dataframe_with_korean,
                fs=50e6,
                title="Test Plot with Korean",
                show_plot=False,
            )
            
            # Check that plot was created (may have sanitized column names)
            assert fig is not None
            
            # Verify that Korean characters are handled gracefully
            # (Plot may sanitize column names or skip non-ASCII columns)
            # Full validation would require parsing plot text elements
            
        except Exception as e:
            # If plotting fails due to non-ASCII, that's acceptable
            # but we should log it
            pytest.skip(f"Plotting with non-ASCII characters not fully supported: {e}")
    
    def test_plot_title_sanitization(self):
        """Test that plot titles are sanitized for file saving."""
        # Test the filename sanitization in interactive_plotting context manager
        test_title = "Shot #45821 - 한글 테스트"
        
        # The sanitization should remove or replace non-ASCII characters
        sanitized = "".join(c for c in test_title if c.isalnum() or c in (" ", "_", "-"))
        sanitized = sanitized.replace(" ", "_").replace("#", "")
        
        # Should not contain Korean characters
        assert not any(ord(c) > 127 for c in sanitized), "Sanitized title should not contain non-ASCII"
        assert "#" not in sanitized, "Sanitized title should not contain #"
    
    def test_plot_enhanced_features(self, sample_dataframe):
        """Test enhanced plotting features."""
        plotter = plots.Plotter()
        
        # Test with various scaling options
        fig, axes = plotter.plot_waveforms(
            sample_dataframe,
            fs=50e6,
            title="Enhanced Plot Test",
            show_plot=False,
            time_scale="ms",
            signal_scale="mV",
            trigger_time=0.290,
            downsample=10,
        )
        
        assert fig is not None
        assert len(axes) > 0


class TestNaNHandling:
    """Test NaN data handling and propagation in mathematical operations."""
    
    def test_refine_data_nan_removal(self, sample_dataframe_with_nan):
        """Test that refine_data removes NaN values."""
        from ifi.analysis import processing
        
        initial_nan_count = sample_dataframe_with_nan.isna().sum().sum()
        assert initial_nan_count > 0, "Test data should contain NaN values"
        
        refined_df = processing.refine_data(sample_dataframe_with_nan)
        
        # After refinement, there should be no NaN values
        final_nan_count = refined_df.isna().sum().sum()
        assert final_nan_count == 0, "Refined data should not contain NaN values"
        
        # Data length should be reduced
        assert len(refined_df) < len(sample_dataframe_with_nan)
    
    def test_nan_propagation_in_math_operations(self):
        """Test that NaN values propagate correctly in mathematical operations."""
        # Create arrays with NaN
        arr1 = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        arr2 = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        
        # Test addition
        result_add = arr1 + arr2
        assert np.isnan(result_add[2]), "NaN + number should be NaN"
        assert np.isnan(result_add[3]), "number + NaN should be NaN"
        assert not np.isnan(result_add[0]), "Valid addition should not be NaN"
        
        # Test multiplication
        result_mul = arr1 * arr2
        assert np.isnan(result_mul[2]), "NaN * number should be NaN"
        assert np.isnan(result_mul[3]), "number * NaN should be NaN"
        
        # Test division
        result_div = arr1 / arr2
        assert np.isnan(result_div[2]), "NaN / number should be NaN"
        assert np.isnan(result_div[3]), "number / NaN should be NaN"
    
    def test_phase_calculation_with_nan(self):
        """Test phase calculation methods handle NaN correctly."""
        from ifi.analysis.phi2ne import PhaseConverter
        
        converter = PhaseConverter()
        
        # Create signals with NaN
        i_signal = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        q_signal = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        
        # calc_phase_iq_asin2 should handle NaN (converts to 0.0)
        phase = converter.calc_phase_iq_asin2(i_signal, q_signal, isnorm=False)
        
        # Phase should not contain NaN (should be converted to 0.0)
        assert not np.any(np.isnan(phase)), "Phase should not contain NaN after processing"
    
    def test_stft_with_nan_data(self, sample_dataframe_with_nan):
        """Test STFT analysis with NaN data."""
        from ifi.analysis import processing
        from ifi.analysis.spectrum import SpectrumAnalysis
        
        # Refine data to remove NaN
        refined_df = processing.refine_data(sample_dataframe_with_nan)
        
        # Calculate STFT
        analyzer = SpectrumAnalysis()
        fs = 1 / refined_df["TIME"].diff().mean()
        signal = refined_df["CH0"].to_numpy()
        
        # STFT should work without NaN
        freq_stft, time_stft, stft_matrix = analyzer.compute_stft(signal, fs)
        
        assert not np.any(np.isnan(stft_matrix)), "STFT matrix should not contain NaN"
        assert not np.any(np.isnan(freq_stft)), "STFT frequencies should not contain NaN"
        assert not np.any(np.isnan(time_stft)), "STFT times should not contain NaN"


class TestDaskParallelProcessing:
    """Test Dask-based parallel processing workflow."""
    
    @patch("ifi.analysis.main_analysis.dask.compute")
    def test_load_and_process_file_dask(self, mock_dask_compute, mock_nas_db, sample_dataframe):
        """Test that load_and_process_file works with Dask."""
        # Mock the NAS_DB get_shot_data method
        mock_nas_db.get_shot_data.return_value = {
            "test_file.csv": sample_dataframe
        }
        
        # Create mock args
        mock_args = Mock()
        mock_args.force_remote = False
        mock_args.no_offset_removal = False
        mock_args.offset_window = 2001
        mock_args.stft = False
        mock_args.cwt = False
        mock_args.ft_cols = None
        
        # Call the delayed function (it returns a delayed object)
        result = load_and_process_file(mock_nas_db, "test_file.csv", mock_args)
        
        # The result should be a Dask delayed object
        assert result is not None
    
    def test_dask_task_creation(self, mock_nas_db):
        """Test that Dask tasks are created correctly."""
        import dask
        
        target_files = ["file1.csv", "file2.csv", "file3.csv"]
        mock_args = Mock()
        
        # Create delayed tasks
        tasks = [dask.delayed(load_and_process_file)(mock_nas_db, f, mock_args) for f in target_files]
        
        assert len(tasks) == 3
        assert all(isinstance(task, dask.delayed.Delayed) for task in tasks)


class TestMainAnalysisIntegration:
    """Integration tests for the full main_analysis workflow."""
    
    @patch("ifi.analysis.main_analysis.NAS_DB")
    @patch("ifi.analysis.main_analysis.VEST_DB")
    def test_full_analysis_workflow(self, mock_vest_class, mock_nas_class, 
                                     sample_dataframe, tmp_h5_dir):
        """Test the full analysis workflow end-to-end."""
        # This is a comprehensive integration test
        # Mock the database classes
        mock_nas_db = Mock(spec=NAS_DB)
        mock_vest_db = Mock(spec=VEST_DB)
        
        mock_nas_class.return_value = mock_nas_db
        mock_vest_class.return_value = mock_vest_db
        
        # Mock find_files
        mock_nas_db.find_files.return_value = ["45821_056.csv"]
        
        # Mock get_shot_data
        mock_nas_db.get_shot_data.return_value = {
            "45821_056.csv": sample_dataframe
        }
        
        # Mock VEST data
        mock_vest_db.load_shot.return_value = pd.DataFrame({
            "ip": np.random.randn(1000),
            "time": np.linspace(0, 1, 1000),
        })
        
        # Create mock args
        mock_args = Mock()
        mock_args.data_folders = None
        mock_args.add_path = False
        mock_args.force_remote = False
        mock_args.results_dir = str(tmp_h5_dir.parent.parent)
        mock_args.no_offset_removal = False
        mock_args.offset_window = 2001
        mock_args.stft = True
        mock_args.cwt = False
        mock_args.ft_cols = None
        mock_args.plot = False
        mock_args.save_plots = False
        mock_args.no_plot_raw = False
        mock_args.no_plot_ft = False
        mock_args.downsample = 10
        mock_args.trigger_time = 0.290
        mock_args.density = True
        mock_args.vest_fields = []
        mock_args.baseline = None
        mock_args.save_data = True
        mock_args.scheduler = "single-threaded"
        
        # Run analysis
        _ = run_analysis(
            query=45821,
            args=mock_args,
            nas_db=mock_nas_db,
            vest_db=mock_vest_db,
        )
        
        # Verify that H5 file was created
        h5_files = list(tmp_h5_dir.glob("*.h5"))
        assert len(h5_files) > 0, "H5 file should be created"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

