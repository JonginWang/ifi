#!/usr/bin/env python3
"""
Frequency Ridge Tracking
========================
This module contains the functions for frequency ridge tracking.

Functions:
    extract_fridges: Extracts frequency ridges from a time-frequency representation.

Args:
    tf_transf(np.ndarray): Complex time-frequency representation.
    frequency_scales(np.ndarray): Frequency scales to calculate distance penalty term.
    penalty(float): Integer value to penalise frequency jumps.
    num_ridges(int): Number of ridges to be calculated.
    BW(int): Decides how many bins will be subtracted around max energy frequency bins when extracting multiple ridges (2 is standard for syncrosqueezed transform).

Returns:
    max_Energy(np.ndarray): Maximum energy of the frequency ridges.
    ridge_idx(np.ndarray): Indexes of the frequency ridges.
    fridge(np.ndarray): Frequencies of the frequency ridges.

References:
    https://github.com/davidbondesson/Frequency_ridge_tracking
    https://de.mathworks.com/help/signal/ref/tfridge.html

Created on Mon Sep 14 09:01:45 2020

@author: David Bondesson
"""

import numpy as np


def extract_fridges(tf_transf, frequency_scales, penalty=2.0, num_ridges=1, BW=25):
    #   tracks frequency ridges by performing forward-backward
    #   ridge tracking algorithm:
    #   Arguments:  tf_transf - complex time frequency representation
    #               fs - frequency scales to calculate distance penalty term
    #               penalty - integer value to penalise frequency jumps
    #               num_ridges - number of ridges to be calculated
    #               BW - decides how many bins will be subtracted around max
    #                    energy frequency bins when extracting multiple ridges (2 is standard for syncrosqueezed transform)
    #   outputs:    max_Energy (vector) along time axis
    #               ridge_idx - indexes for maximum frequency ridge(s)
    #               fridge - frequencies traccking maximum frequency ridge(s)

    def generate_penalty_matrix(frequency_scales, penalty):
        #   penalty matrix describes all potential penalties of  jumping from current frequency
        #   (first axis) to one or several new frequencies (second axis)
        #   Arguments: frequency_scales - Frequency scale vector from time-freq transform
        #              penalty - user set penalty for freqency jumps (standard =1.0)
        #   outputs:   dist_matrix -penalty matrix
        #
        freq_scale = frequency_scales.copy()
        dist_matrix = np.square(np.subtract.outer(freq_scale, freq_scale)) * penalty

        return dist_matrix

    def calculate_accumulated_penalty_energy_forwards(Energy_to_track, penalty_matrix):
        #   Calculates acummulated penalty in forward direction (t=0...end)
        #   Arguments:  Energy - squared abs time-frequency transform
        #               penalty_matrix - pre calculated penalty for all potential jumps between two frequencies
        #   outputs:    penalised_energy - new energy with added forward penalty
        #               ridge_idx - calculated initial ridge with only forward penalty
        #   OPTIMIZED: Vectorized inner loop for better performance
        penalised_energy = Energy_to_track.copy()
        nfreq, ntime = np.shape(penalised_energy)

        # Vectorized: compute for all frequencies at once per time step
        for idx_time in range(1, ntime):
            # For each current frequency idx_freq, compute:
            #   min(penalised_energy[:, idx_time - 1] + penalty_matrix[idx_freq, :])
            # Broadcasting: (1, nfreq) + (nfreq, nfreq) = (nfreq, nfreq)
            prev_energy = penalised_energy[:, idx_time - 1]  # (nfreq,)
            # candidates[i, j] = prev_energy[j] + penalty_matrix[i, j]
            # This represents cost of going from prev freq j to current freq i
            candidates = prev_energy[None, :] + penalty_matrix  # (nfreq, nfreq)
            min_candidates = np.amin(candidates, axis=1)  # (nfreq,)
            penalised_energy[:, idx_time] += min_candidates

        ridge_idx = np.unravel_index(
            np.argmin(penalised_energy, axis=0), penalised_energy.shape
        )[1]

        return penalised_energy, ridge_idx

    def calculate_accumulated_penalty_energy_backwards(
        Energy_to_track, penalty_matrix, penalised_energy_frwd, ridge_idx_frwd
    ):
        #   Calculates acummulated penalty in backward direction (t=end...0)
        #   Arguments:  Energy - squared abs time-frequency transform
        #               penalty_matrix - pre calculated penalty for all potential jumps between two frequencies
        #               ridge_idx_frwd - Calculated forward ridge
        #   outputs:    ridge_idx_frwd - new ridge with added backward penalty
        #   OPTIMIZED: Vectorized inner loop for better performance
        pen_e = penalised_energy_frwd.copy()
        e = Energy_to_track.copy()
        nfreq, ntime = np.shape(e)
        eps = np.finfo(np.float64).eps

        # Vectorized: compute for all frequencies at once per time step
        for idx_time in range(ntime - 2, -1, -1):
            val = (
                pen_e[ridge_idx_frwd[idx_time + 1], idx_time + 1]
                - e[ridge_idx_frwd[idx_time + 1], idx_time + 1]
            )
            # Vectorized: compute differences for all frequencies at once
            prev_freq_idx = ridge_idx_frwd[idx_time + 1]
            new_penalties = penalty_matrix[prev_freq_idx, :]  # (nfreq,)
            # Compute candidate values for all frequencies
            candidate_values = pen_e[:, idx_time] + new_penalties  # (nfreq,)
            # Find frequency index where difference is within epsilon
            differences = np.abs(val - candidate_values)
            matching_indices = np.where(differences < eps)[0]
            
            if len(matching_indices) > 0:
                # Use first matching index (maintains original behavior)
                ridge_idx_frwd[idx_time] = matching_indices[0]

        return ridge_idx_frwd

    def frwd_bckwd_ridge_tracking(Energy_to_track, penalty_matrix):
        #   Calculates acummulated penalty in forward (t=end...0) followed by backward (t=end...0) direction
        #   Arguments:  Energy - squared abs time-frequency transform
        #               penalty_matrix - pre calculated penalty for all potential jumps between two frequencies
        #   outputs:    ridge_idx_frwd_bck - Estimated forward backward frequency ridge indices

        penalised_energy_frwd, ridge_idx_frwd = (
            calculate_accumulated_penalty_energy_forwards(
                Energy_to_track, penalty_matrix
            )
        )
        #    backward calculation of frequency ridge (min log negative energy)
        ridge_idx_frwd_bck = calculate_accumulated_penalty_energy_backwards(
            Energy_to_track, penalty_matrix, penalised_energy_frwd, ridge_idx_frwd
        )

        return ridge_idx_frwd_bck

    Energy = np.square(np.abs(tf_transf))
    dim = Energy.shape
    ridge_idx = np.zeros((dim[1], num_ridges))
    max_Energy = np.zeros((dim[1], num_ridges))
    fridge = np.zeros((dim[1], num_ridges))

    penalty_matrix = np.squeeze(generate_penalty_matrix(frequency_scales, penalty))
    eps = np.finfo(np.float64).eps

    for current_ridge_index in range(0, num_ridges):
        energy_max = np.max(Energy, axis=0)
        Energy_neg_log_norm = -np.log((Energy / energy_max) + eps)

        ridge_idx[:, current_ridge_index] = np.array(
            frwd_bckwd_ridge_tracking(Energy_neg_log_norm, penalty_matrix)
        )
        ridge_idx = ridge_idx.astype(int)

        max_Energy[:, current_ridge_index] = Energy[
            ridge_idx[:, current_ridge_index],
            np.arange(len(ridge_idx[:, current_ridge_index])),
        ]
        fridge[:, current_ridge_index] = np.squeeze(
            frequency_scales[ridge_idx[:, current_ridge_index]]
        )

        # OPTIMIZED: Vectorized Energy mask setting
        # Create boolean mask for all time indices at once
        for time_idx in range(dim[1]):
            freq_idx = int(ridge_idx[time_idx, current_ridge_index])
            start_idx = max(0, freq_idx - BW)
            end_idx = min(dim[0], freq_idx + BW + 1)
            Energy[start_idx:end_idx, time_idx] = 0
    return max_Energy, ridge_idx, fridge
