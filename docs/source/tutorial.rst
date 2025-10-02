Tutorial
========

This tutorial demonstrates a complete workflow for analyzing interferometer data using IFI Automator.

Complete Analysis Workflow
---------------------------

Step 1: Setup and Data Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from ifi.analysis import PhaseChangeDetector, SignalStacker
    from ifi.db_controller import NAS_DB
    from ifi.utils import LogManager
    
    # Initialize logging
    logger = LogManager()
    logger.info("Starting tutorial analysis")
    
    # Connect to database
    nas_db = NAS_DB()
    
    # Load shot data
    shot_number = 12345
    signal_data = nas_db.get_shot_data(shot_number)
    logger.info(f"Loaded data for shot {shot_number}: {len(signal_data)} samples")

Step 2: Signal Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Initialize signal stacker for preprocessing
    stacker = SignalStacker(fs=1000)
    
    # Find fundamental frequency
    f0 = stacker.find_fundamental_frequency(signal_data)
    logger.info(f"Fundamental frequency: {f0:.2f} Hz")
    
    # Stack signals for better SNR
    stacked_signal = stacker.stack_signals(signal_data, n_stacks=10)
    logger.info(f"Stacked signal length: {len(stacked_signal)}")

Step 3: Phase Change Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Initialize phase change detector
    detector = PhaseChangeDetector(fs=1000)
    
    # Detect phase changes using multiple methods
    results = detector.detect_phase_changes_unified(stacked_signal)
    
    # Print results
    logger.info(f"Analysis completed:")
    logger.info(f"  - Fundamental frequency: {results['fundamental_frequency']:.2f} Hz")
    logger.info(f"  - Phase changes detected: {len(results['phase_changes'])}")
    logger.info(f"  - Analysis methods used: {list(results.keys())}")

Step 4: Results Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Extract phase changes
    phase_changes = results['phase_changes']
    
    if len(phase_changes) > 0:
        logger.info("Phase change analysis:")
        for i, change in enumerate(phase_changes):
            logger.info(f"  Change {i+1}:")
            logger.info(f"    Time: {change['time']:.3f} s")
            logger.info(f"    Magnitude: {change['magnitude']:.3f}")
            logger.info(f"    Method: {change['method']}")
    else:
        logger.info("No significant phase changes detected")

Advanced Analysis
----------------

Multi-method Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Use specific analysis methods
    stft_results = detector.detect_phase_changes_stft(stacked_signal)
    cwt_results = detector.detect_phase_changes_cwt(stacked_signal)
    cdm_results = detector.detect_phase_changes_cdm(stacked_signal)
    
    # Compare results
    logger.info("Method comparison:")
    logger.info(f"  STFT: {len(stft_results['phase_changes'])} changes")
    logger.info(f"  CWT:  {len(cwt_results['phase_changes'])} changes")
    logger.info(f"  CDM:  {len(cdm_results['phase_changes'])} changes")

Custom Analysis Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Custom detector with specific parameters
    custom_detector = PhaseChangeDetector(
        fs=1000,
        f_range=(10, 200),  # Custom frequency range
        threshold=0.1,      # Custom detection threshold
        window_size=1024    # Custom window size
    )
    
    # Run analysis with custom parameters
    custom_results = custom_detector.detect_phase_changes(stacked_signal)

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Process multiple shots
    shot_numbers = [12345, 12346, 12347]
    batch_results = {}
    
    for shot_num in shot_numbers:
        logger.info(f"Processing shot {shot_num}")
        
        # Load data
        signal = nas_db.get_shot_data(shot_num)
        
        # Analyze
        results = detector.detect_phase_changes(signal)
        batch_results[shot_num] = results
        
        logger.info(f"Shot {shot_num}: {len(results['phase_changes'])} changes")

Data Export
~~~~~~~~~~~

.. code-block:: python

    from ifi.utils.file_io import save_results_to_hdf5
    
    # Save results to HDF5 file
    output_file = f"analysis_results_shot_{shot_number}.h5"
    save_results_to_hdf5(
        results=results,
        signals={'stacked_signal': stacked_signal},
        output_path=output_file
    )
    
    logger.info(f"Results saved to {output_file}")

Visualization
~~~~~~~~~~~~~

.. code-block:: python

    import matplotlib.pyplot as plt
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot original signal
    time_axis = np.arange(len(signal_data)) / 1000  # Convert to seconds
    axes[0].plot(time_axis, signal_data)
    axes[0].set_title(f'Original Signal - Shot {shot_number}')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    
    # Plot stacked signal
    axes[1].plot(time_axis[:len(stacked_signal)], stacked_signal)
    axes[1].set_title('Stacked Signal')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    
    # Mark phase changes
    for change in phase_changes:
        axes[1].axvline(x=change['time'], color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'analysis_plot_shot_{shot_number}.png', dpi=300)
    plt.show()

Best Practices
--------------

1. **Always use logging**: Enable logging for debugging and monitoring
2. **Validate input data**: Check data quality before analysis
3. **Use appropriate parameters**: Adjust analysis parameters for your specific use case
4. **Save intermediate results**: Keep track of preprocessing steps
5. **Visualize results**: Always plot and inspect your data

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. **No phase changes detected**: Check frequency range and threshold settings
2. **Poor signal quality**: Try different stacking parameters or preprocessing
3. **Memory issues**: Process data in smaller chunks
4. **Database connection**: Verify credentials and network connectivity

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Use vectorized operations**: Leverage NumPy for better performance
2. **Enable JIT compilation**: Use Numba for computationally intensive tasks
3. **Optimize memory usage**: Process data in chunks for large datasets
4. **Parallel processing**: Use Dask for distributed computing

Next Steps
----------

- Explore the :doc:`api/analysis` for detailed API reference
- Check the :doc:`api/db_controller` for database operations
- Read the :doc:`getting_started` for basic setup instructions
