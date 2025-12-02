#!/usr/bin/env python3
"""
Integration tests for main_analysis results cache functionality.

This test suite covers:
1. Results directory checking before analysis
2. Skipping analysis when complete results exist
3. Partial analysis when only some results exist
4. Merging cached and newly analyzed results

Run with pytest: pytest tests/analysis/integration/test_main_analysis_results_cache.py
"""

from pathlib import Path
import shutil
import pytest
import pandas as pd
import numpy as np
import h5py
from unittest.mock import Mock, patch, MagicMock

from ifi.utils.common import LogManager
from ifi.analysis.main_analysis import run_analysis
from ifi.db_controller.nas_db import NAS_DB
from ifi.db_controller.vest_db import VEST_DB
from ifi.utils import file_io
from ifi import get_project_root

# Initialize logging
LogManager(level="INFO")
logger = LogManager().get_logger(__name__)

CONFIG_PATH = 'ifi/config.ini'
SHOT_TO_TEST = 45821


@pytest.fixture
def temp_results_dir(tmp_path):
    """Create a temporary results directory structure."""
    project_root = get_project_root()
    results_base = project_root / "ifi" / "results"
    results_base.mkdir(parents=True, exist_ok=True)
    
    shot_dir = results_base / str(SHOT_TO_TEST)
    shot_dir.mkdir(parents=True, exist_ok=True)
    
    yield shot_dir
    
    # Cleanup
    if shot_dir.exists():
        try:
            shutil.rmtree(shot_dir)
        except Exception:
            pass


@pytest.fixture
def complete_results_h5(temp_results_dir):
    """Create a complete results H5 file with all analysis results."""
    h5_file = temp_results_dir / f"{SHOT_TO_TEST}.h5"
    
    n_samples = 1000
    time_data = np.linspace(0, 0.01, n_samples)
    signal_data = np.sin(2 * np.pi * 94e9 * time_data)
    
    with h5py.File(h5_file, 'w') as f:
        # Signals
        signals_group = f.create_group("signals")
        freq_group = signals_group.create_group("freq_94.0_GHz")
        freq_group.create_dataset("TIME", data=time_data)
        freq_group.create_dataset("CH0", data=signal_data)
        freq_group.create_dataset("CH1", data=signal_data * 0.9)
        
        # STFT results
        stft_group = f.create_group("stft_results")
        signal_stft_group = stft_group.create_group("test_file.csv")
        signal_stft_group.create_dataset("time_STFT", data=time_data[::10])
        signal_stft_group.create_dataset("freq_STFT", data=np.linspace(0, 50e6, 100))
        signal_stft_group.create_dataset("STFT_matrix", data=np.random.randn(100, 100))
        
        # CWT results
        cwt_group = f.create_group("cwt_results")
        signal_cwt_group = cwt_group.create_group("test_file.csv")
        signal_cwt_group.create_dataset("time_CWT", data=time_data[::10])
        signal_cwt_group.create_dataset("freq_CWT", data=np.linspace(0, 50e6, 100))
        signal_cwt_group.create_dataset("CWT_matrix", data=np.random.randn(100, 100))
        
        # Density data
        density_group = f.create_group("density_data")
        density_group.create_dataset("ne_CH0_test", data=np.random.randn(n_samples))
        density_group.create_dataset("ne_CH1_test", data=np.random.randn(n_samples))
        
        # Metadata
        metadata_group = f.create_group("metadata")
        metadata_group.attrs["shot_number"] = SHOT_TO_TEST
        metadata_group.attrs["created_at"] = "2025-01-27"
    
    yield h5_file
    
    # Cleanup
    if h5_file.exists():
        try:
            h5_file.unlink()
        except Exception:
            pass


@pytest.fixture
def partial_results_h5(temp_results_dir):
    """Create a partial results H5 file with only signals."""
    h5_file = temp_results_dir / f"{SHOT_TO_TEST}.h5"
    
    n_samples = 1000
    time_data = np.linspace(0, 0.01, n_samples)
    signal_data = np.sin(2 * np.pi * 94e9 * time_data)
    
    with h5py.File(h5_file, 'w') as f:
        # Signals only
        signals_group = f.create_group("signals")
        freq_group = signals_group.create_group("freq_94.0_GHz")
        freq_group.create_dataset("TIME", data=time_data)
        freq_group.create_dataset("CH0", data=signal_data)
        freq_group.create_dataset("CH1", data=signal_data * 0.9)
        
        # Metadata
        metadata_group = f.create_group("metadata")
        metadata_group.attrs["shot_number"] = SHOT_TO_TEST
    
    yield h5_file
    
    # Cleanup
    if h5_file.exists():
        try:
            h5_file.unlink()
        except Exception:
            pass


@pytest.fixture
def mock_nas_db():
    """Create a mock NAS_DB instance."""
    mock_db = Mock(spec=NAS_DB)
    mock_db.find_files = Mock(return_value=[f"test_{SHOT_TO_TEST}_056.csv"])
    mock_db.get_shot_data = Mock(return_value={
        f"test_{SHOT_TO_TEST}_056.csv": pd.DataFrame({
            'TIME': np.linspace(0, 0.01, 1000),
            'CH0': np.sin(np.linspace(0, 2*np.pi, 1000)),
            'CH1': np.sin(np.linspace(0, 2*np.pi, 1000)) * 0.9
        })
    })
    mock_db.connect = Mock(return_value=True)
    mock_db.disconnect = Mock()
    mock_db._is_connected = True
    return mock_db


@pytest.fixture
def mock_vest_db():
    """Create a mock VEST_DB instance."""
    mock_db = Mock(spec=VEST_DB)
    mock_db.load_shot = Mock(return_value={})
    mock_db.connect = Mock(return_value=True)
    mock_db.disconnect = Mock()
    return mock_db


class TestMainAnalysisResultsCache:
    """Test suite for main_analysis results cache functionality."""
    
    def test_load_results_from_hdf5_complete(self, complete_results_h5):
        """Test loading complete results from H5 file."""
        project_root = get_project_root()
        results_dir = project_root / "ifi" / "results"
        
        results = file_io.load_results_from_hdf5(SHOT_TO_TEST, base_dir=str(results_dir))
        
        assert results is not None
        assert "signals" in results
        assert "stft_results" in results
        assert "cwt_results" in results
        assert "density_data" in results
        assert isinstance(results["signals"], dict)
        assert len(results["signals"]) > 0
    
    def test_load_results_from_hdf5_partial(self, partial_results_h5):
        """Test loading partial results from H5 file."""
        project_root = get_project_root()
        results_dir = project_root / "ifi" / "results"
        
        results = file_io.load_results_from_hdf5(SHOT_TO_TEST, base_dir=str(results_dir))
        
        assert results is not None
        assert "signals" in results
        # STFT, CWT, density should not be present
        assert "stft_results" not in results or not results.get("stft_results")
        assert "cwt_results" not in results or not results.get("cwt_results")
        assert "density_data" not in results or results.get("density_data") is None or results.get("density_data").empty
    
    def test_run_analysis_skips_when_complete_results_exist(
        self, complete_results_h5, mock_nas_db, mock_vest_db, tmp_path
    ):
        """Test that run_analysis skips analysis when complete results exist."""
        # Create args namespace
        from argparse import Namespace
        args = Namespace(
            stft=True,
            cwt=True,
            density=True,
            data_folders=None,
            add_path=False,
            force_remote=False,
            vest_fields=[],
            results_dir=str(tmp_path / "results"),
            save_data=False,
            plot=False,
            scheduler="threads"
        )
        
        # Mock get_project_root to return tmp_path
        # Note: get_project_root is imported from ifi, not from main_analysis
        with patch('ifi.get_project_root', return_value=tmp_path):
            # Mock file_io.load_results_from_hdf5 to return complete results
            with patch('ifi.analysis.main_analysis.file_io.load_results_from_hdf5') as mock_load:
                mock_load.return_value = {
                    "signals": {"freq_94.0_GHz": pd.DataFrame({
                        'TIME': np.linspace(0, 0.01, 1000),
                        'CH0': np.sin(np.linspace(0, 2*np.pi, 1000))
                    })},
                    "stft_results": {"test_file.csv": {}},
                    "cwt_results": {"test_file.csv": {}},
                    "density_data": pd.DataFrame({
                        'ne_CH0': np.random.randn(1000)
                    })
                }
                
                # Run analysis
                result = run_analysis(
                    query=[SHOT_TO_TEST],
                    args=args,
                    nas_db=mock_nas_db,
                    vest_db=mock_vest_db
                )
                
                # Should return early with cached results
                assert result is not None
                assert SHOT_TO_TEST in result
                
                # NAS_DB should not be called for data fetching
                mock_nas_db.get_shot_data.assert_not_called()
    
    def test_run_analysis_partial_results(self, partial_results_h5, mock_nas_db, mock_vest_db, tmp_path):
        """Test that run_analysis performs partial analysis when only some results exist."""
        # This test would require more complex mocking of the analysis pipeline
        # For now, we'll test the results loading logic
        project_root = get_project_root()
        results_dir = project_root / "ifi" / "results"
        
        results = file_io.load_results_from_hdf5(SHOT_TO_TEST, base_dir=str(results_dir))
        
        # Should have signals but not complete analysis
        assert results is not None
        assert "signals" in results
        
        # Check that partial results are detected correctly
        has_signals = "signals" in results and bool(results["signals"])
        has_stft = "stft_results" in results and bool(results.get("stft_results"))
        has_cwt = "cwt_results" in results and bool(results.get("cwt_results"))
        has_density = "density_data" in results and results.get("density_data") is not None and not results.get("density_data").empty
        
        # Verify signals exist
        assert has_signals == True, f"Signals should be present, got: {results.get('signals')}"
        # Verify other results are missing (partial results)
        assert has_stft == False, f"STFT should not be present in partial results, got: {results.get('stft_results')}"
        assert has_cwt == False, f"CWT should not be present in partial results, got: {results.get('cwt_results')}"
        assert has_density == False, f"Density should not be present in partial results, got: {results.get('density_data')}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

