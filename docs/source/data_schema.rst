HDF5 Result File Schema
=======================

This page documents the **canonical HDF5 schema** used by IFI to store
analysis results. The schema is derived from:

- ``ifi.utils.file_io.save_results_to_hdf5`` / ``load_results_from_hdf5``
- Integration tests in ``tests/analysis/integration/test_main_analysis_integration.py``

Result files live under ``ifi/results/<shot_num>/*.h5`` and are distinct from
NAS cache files under ``./cache/*.h5``.

File Naming
-----------

- For normal shots (``shot_num != 0``): ``<shot_num>.h5`` (e.g. ``45821.h5``)
- For unknown shots (``shot_num == 0`` and signals present):
  the stem of the first signal key is used: e.g. ``test_file.csv`` → ``test_file.h5``.

Files are typically written to::

    ifi/results/<shot_num>/<shot_num>.h5

Top-Level Layout
----------------

Every result file produced by ``save_results_to_hdf5`` has the following
top-level groups (some may be absent if no data of that type was produced):

.. list-table::
   :header-rows: 1

   * - Path
     - Type
     - Purpose
   * - ``/metadata``
     - Group
     - Shot-level metadata and versioning
   * - ``/signals``
     - Group
     - Raw or combined signal DataFrames
   * - ``/stft_results``
     - Group
     - STFT analysis results per signal
   * - ``/cwt_results``
     - Group
     - CWT analysis results per signal
   * - ``/density_data``
     - Group
     - Phase-to-density results (``ne_*`` columns)
   * - ``/vest_data``
     - Group
     - VEST DB time/current and related signals

``/metadata`` Group
-------------------

The ``/metadata`` group stores minimal but essential shot-level information:

.. list-table::
   :header-rows: 1

   * - Name
     - Location
     - Type
     - Description
   * - ``shot_number``
     - attr
     - int
     - Shot number (0 for unknown or synthetic data)
   * - ``created_at``
     - attr
     - str (ISO)
     - Timestamp when the file was created
   * - ``ifi_version``
     - attr
     - str
     - IFI result schema/version (currently ``\"1.0\"``)

Requirements:

- ``/metadata`` **must exist**.
- All three attributes must be present.

``/signals`` Group
------------------

The ``/signals`` group contains one sub-group **per signal DataFrame**.

- Path pattern: ``/signals/<signal_name>/``
  - Example: ``/signals/test_file.csv/``
  - Example (post-processing): ``/signals/freq_94.0_GHz/``

Each ``<signal_name>`` group corresponds to a pandas :class:`~pandas.DataFrame`.

.. list-table::
   :header-rows: 1

   * - Path
     - Type
     - Description
   * - ``/signals``
     - Group
     - Container for all signals
   * - ``/signals`` attrs[``\"empty\"``]
     - Attr
     - If ``True``, no signals are stored (group is logically empty)
   * - ``/signals/<signal_name>``
     - Group
     - Sub-group for a single DataFrame
   * - ``/signals/<signal_name>/<col_name>``
     - Dataset
     - 1D array for a column (e.g. ``TIME``, ``CH0``, ``CH1``, ...)

Typical dtypes:

- Time and signal columns are stored as ``float32`` or ``float64`` datasets.

Special case: empty signals
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When no signals are provided, ``save_results_to_hdf5``:

- Creates ``/signals``
- Sets ``/signals.attrs[\"empty\"] = True``

If ``empty`` is not set (or False), at least one child group under
``/signals`` is expected.

``/stft_results`` Group
-----------------------

STFT results are stored under ``/stft_results`` as **one group per signal**.

.. list-table::
   :header-rows: 1

   * - Path
     - Type
     - Description
   * - ``/stft_results``
     - Group
     - Container for STFT results
   * - ``/stft_results/<signal_name>``
     - Group
     - STFT results for a specific signal
   * - ``/stft_results/<signal_name>/<key>``
     - Dataset
     - ndarray-valued entries (e.g. frequency, time, matrix)
   * - ``/stft_results/<signal_name>`` attrs[``key``]
     - Attr
     - Scalar metadata (e.g. center frequency)

Pattern:

- For each signal:

  - ndarray values (``np.ndarray``) in the STFT result dict become datasets
    under ``/stft_results/<signal_name>/``.
  - Scalar values (``int``, ``float``, ``str``) become attributes of the
    ``/stft_results/<signal_name>`` group.

``/cwt_results`` Group
----------------------

The ``/cwt_results`` group mirrors the structure of ``/stft_results`` but
for CWT:

.. list-table::
   :header-rows: 1

   * - Path
     - Type
     - Description
   * - ``/cwt_results``
     - Group
     - Container for CWT results
   * - ``/cwt_results/<signal_name>``
     - Group
     - CWT results for a specific signal
   * - ``/cwt_results/<signal_name>/<key>``
     - Dataset
     - ndarray-valued entries
   * - ``/cwt_results/<signal_name>`` attrs[``key``]
     - Attr
     - Scalar metadata

If no CWT analysis is requested, this group may be absent.

``/density_data`` Group
-----------------------

The ``/density_data`` group stores the phase-to-density conversion results as
a DataFrame.

Column names typically follow patterns like:

- ``ne_CH0_test``, ``ne_CH1_test``
- ``ne_CH0_combined``, ``ne_CH1_combined``

.. list-table::
   :header-rows: 1

   * - Path
     - Type
     - Description
   * - ``/density_data``
     - Group
     - Container for density results
   * - ``/density_data/<density_col_name>``
     - Dataset
     - 1D array for a density column

``/vest_data`` Group
--------------------

The ``/vest_data`` group stores VEST DB-derived time/current and related
signals:

.. list-table::
   :header-rows: 1

   * - Path
     - Type
     - Description
   * - ``/vest_data``
     - Group
     - Container for VEST-derived signals
   * - ``/vest_data/ip``
     - Dataset
     - Plasma current or related signal
   * - ``/vest_data/time``
     - Dataset
     - Time axis
   * - ``/vest_data/<other_column>``
     - Dataset
     - Additional VEST fields, if present

Relation to ``./cache`` Files
-----------------------------

Result files under ``ifi/results/<shot_num>/*.h5``:

- Written by ``ifi.utils.file_io.save_results_to_hdf5``.
- Follow the schema described on this page.
- Intended for **analysis results** (signals, STFT/CWT, density, VEST).

Cache files under ``./cache/<shot_num>.h5``:

- Written by :class:`ifi.db_controller.nas_db.NAS_DB` using
  ``pandas.DataFrame.to_hdf(..., format=\"table\")``.
- Keys are sanitized basenames (e.g. ``_45821_056_csv``), not the schema above.
- Include metadata in separate keys (``<key>_metadata``) as table-format HDF.
- Internal to NAS caching and **not** part of the official result schema.

Validator Summary
-----------------

The helper functions in ``ifi.utils.h5_schema`` implement basic schema
validation:

- :func:`ifi.utils.h5_schema.validate_h5_schema`:

  - Ensures ``/metadata`` exists with attributes ``shot_number``,
    ``created_at``, and ``ifi_version``.
  - Ensures ``/signals`` exists; if it is not marked ``empty=True``, at least
    one child group must be present.
  - Confirms that optional groups (``/stft_results``, ``/cwt_results``,
    ``/density_data``, ``/vest_data``) are groups if they exist.

- :func:`ifi.utils.h5_schema.load_h5_data`:

  - Calls ``validate_h5_schema`` first.
  - Then delegates to ``load_results_from_hdf5`` to construct the result
    dictionary.

Future extensions may add deeper checks for dataset dtypes and shapes based on
expected analysis outputs.


