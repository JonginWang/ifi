Welcome to IFI Automator!
==========================

**IFI Automator** is a comprehensive Python package for automated interferometer data analysis and phase change detection in plasma physics research.

Key Features
-------------

* **Automated Data Processing**: Streamlined analysis of interferometer signals
* **Phase Change Detection**: Advanced algorithms for detecting plasma phase changes
* **Database Integration**: Seamless connection to NAS and VEST databases
* **Real-time Analysis**: High-performance signal processing with NumPy and Numba optimization
* **Comprehensive Documentation**: Complete API reference and tutorials

Quick Start
-----------

.. code-block:: python

    from ifi.analysis import PhaseChangeDetector
    from ifi.db_controller import NAS_DB
    
    # Initialize detector
    detector = PhaseChangeDetector(fs=1000)
    
    # Load data from database
    nas_db = NAS_DB()
    signal = nas_db.get_shot_data(shot_number=12345)
    
    # Analyze phase changes
    results = detector.detect_phase_changes(signal)
    print(f"Detected {len(results['phase_changes'])} phase changes")

Installation
------------

.. code-block:: bash

    pip install -r requirements.txt

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide:
   
   getting_started
   tutorial
   data_schema
   
.. toctree::
   :maxdepth: 2
   :caption: API Reference:
   
   api/analysis
   api/db_controller
   api/utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

