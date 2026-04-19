from types import SimpleNamespace

import numpy as np
import pandas as pd

from ifi.analysis import workflow_density


def test_calculate_density_passes_flip_density_to_phase_methods(monkeypatch):
    calls = []

    class FakePhaseConverter:
        def calc_phase_cdm(self, ref_signal, prob_signal, fs, f_center, isflip=False):
            calls.append(("cdm", isflip))
            return np.ones_like(ref_signal, dtype=float), {}

        def calc_phase_fpga(self, ref_phase, probe_phase, time, amp_signal, isflip=False):
            calls.append(("fpga", isflip))
            return np.ones_like(ref_phase, dtype=float)

        def calc_phase_iq(self, i_signal, q_signal, isflip=False):
            calls.append(("iq", isflip))
            return np.ones_like(i_signal, dtype=float), {}

        def phase_to_density(self, phase, analysis_params=None):
            return phase

    class FakeSpectrumAnalysis:
        def find_center_frequency_fft(self, ref_signal, fs):
            return 1.0e6

    monkeypatch.setattr(workflow_density.phi2ne, "PhaseConverter", FakePhaseConverter)
    monkeypatch.setattr(workflow_density.spectrum, "SpectrumAnalysis", FakeSpectrumAnalysis)

    index = pd.Index(np.linspace(0.0, 5.0e-6, 6), name="TIME")
    freq_data = pd.DataFrame(
        {
            "CH0_shot_CDM": np.arange(6, dtype=float),
            "CH1_shot_CDM": np.arange(6, dtype=float) + 1.0,
            "CH0_shot_FPGA": np.arange(6, dtype=float) + 2.0,
            "CH1_shot_FPGA": np.arange(6, dtype=float) + 3.0,
            "CH4_shot_IQ": np.arange(6, dtype=float) + 4.0,
            "CH5_shot_IQ": np.arange(6, dtype=float) + 5.0,
        },
        index=index,
    )
    freq_groups = {
        94.0: {
            "files": ["shot_CDM.csv", "shot_FPGA.csv", "shot_IQ.csv"],
            "params": [
                {"method": "CDM", "ref_col": "CH0", "probe_cols": ["CH1"]},
                {"method": "FPGA", "ref_col": "CH0", "probe_cols": ["CH1"]},
                {"method": "IQ", "probe_cols": [("CH4", "CH5")]},
            ],
        }
    }
    args = SimpleNamespace(density=True, flip_density=True, baseline=None)

    density = workflow_density.calculate_density_data_by_frequency(
        freq_combined_signals={94.0: freq_data},
        freq_groups=freq_groups,
        shot_num=47809,
        args=args,
        vest_ip_data=None,
    )

    assert set(density) == {"freq_94G"}
    assert calls == [("cdm", True), ("fpga", True), ("iq", True)]
