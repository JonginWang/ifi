Getting Started
===============

This guide will help you get started with IFI Automator for interferometer data analysis.

Installation
------------

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/ifi-team/ifi-automator.git
    cd ifi-automator

2. Install dependencies:

.. code-block:: bash

    pip install -r requirements.txt

3. Set up the conda environment (recommended):

.. code-block:: bash

    conda create -n ifi_py310 python=3.10
    conda activate ifi_py310
    pip install -r requirements.txt

Basic Usage
-----------

Import the necessary modules:

.. code-block:: python

    from ifi.analysis import PhaseChangeDetector, SignalStacker
    from ifi.db_controller import NAS_DB, VEST_DB
    from ifi.utils import LogManager

Initialize the logging system:

.. code-block:: python

    logger = LogManager()
    logger.info("Starting IFI Automator analysis")

Load data from database:

.. code-block:: python

    # Connect to NAS database
    nas_db = NAS_DB()
    signal_data = nas_db.get_shot_data(shot_number=12345)
    
    # Connect to VEST database
    vest_db = VEST_DB()
    shot_info = vest_db.get_shot_info(shot_number=12345)

Perform phase change detection:

.. code-block:: python

    # Initialize detector
    detector = PhaseChangeDetector(fs=1000)
    
    # Detect phase changes
    results = detector.detect_phase_changes(signal_data)
    
    # Print results
    print(f"Fundamental frequency: {results['fundamental_frequency']:.2f} Hz")
    print(f"Phase changes detected: {len(results['phase_changes'])}")

Configuration
-------------

The IFI Automator can be configured through various settings:

Database Configuration
~~~~~~~~~~~~~~~~~~~~~~

NAS Database settings:

.. code-block:: python

    nas_db = NAS_DB(
        host="your-nas-host",
        username="your-username",
        password="your-password"
    )

VEST Database settings:

.. code-block:: python

    vest_db = VEST_DB(
        host="your-vest-host",
        database="vest_db",
        username="your-username",
        password="your-password"
    )

Analysis Parameters
~~~~~~~~~~~~~~~~~~~

Configure analysis parameters:

.. code-block:: python

    detector = PhaseChangeDetector(
        fs=1000,  # Sampling frequency
        f_range=(0.1, 500),  # Frequency range for analysis
        methods=['stft', 'cwt', 'cdm']  # Analysis methods
    )

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. **Import Errors**: Make sure all dependencies are installed correctly
2. **Database Connection**: Verify database credentials and network connectivity
3. **Memory Issues**: For large datasets, consider using chunked processing

Performance Tips
~~~~~~~~~~~~~~~~

1. **Use Numba JIT**: Enable JIT compilation for better performance
2. **Vectorized Operations**: Use NumPy vectorized operations when possible
3. **Memory Management**: Process data in chunks for large datasets

Next Steps
----------

- Read the :doc:`tutorial` for detailed examples
- Check the :doc:`api/analysis` for complete API reference
- Explore the :doc:`api/db_controller` for database operations
