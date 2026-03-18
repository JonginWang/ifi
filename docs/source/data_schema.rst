HDF5 Result File Schema
=======================

This page documents the canonical HDF5 schema used by IFI to store
analysis results.

The schema is implemented by:

- ``ifi.utils.io_utils.save_results_to_hdf5``
- ``ifi.utils.io_utils.load_results_from_hdf5``
- ``ifi.utils.io_h5_inspect.validate_h5_schema``
- ``ifi.utils.io_h5_inspect.load_h5_data``

Result files live under::

    ifi/results/<shot_num>/<shot_num>.h5

These files are distinct from NAS raw-cache files and are intended for
post-processing outputs.

Top-Level Layout
----------------

Every canonical result file uses root attributes plus a fixed set of
top-level groups.

.. list-table::
   :header-rows: 1

   * - Path
     - Type
     - Purpose
   * - ``/`` attrs
     - Attributes
     - Shot-level metadata and schema/version information
   * - ``/rawdata``
     - Group
     - Raw or combined signal DataFrames
   * - ``/stft``
     - Group
     - STFT analysis results per signal
   * - ``/cwt``
     - Group
     - CWT analysis results per signal
   * - ``/density``
     - Group
     - Density results grouped by frequency
   * - ``/vestdata``
     - Group
     - Structured VEST data grouped by sample-rate or source layout

Root Metadata
-------------

Canonical result files store metadata as root attributes, not a
``/metadata`` group.

.. list-table::
   :header-rows: 1

   * - Name
     - Location
     - Type
     - Description
   * - ``shot_number``
     - root attr
     - int
     - Shot number (0 for unknown or synthetic data)
   * - ``created_at``
     - root attr
     - str
     - Timestamp when the file was created
   * - ``ifi_version``
     - root attr
     - str
     - IFI result schema/version

Requirements:

- The file root must contain ``shot_number``, ``created_at``, and
  ``ifi_version``.
- ``validate_h5_schema`` checks these attributes directly from the root.

``/rawdata`` Group
------------------

The ``/rawdata`` group contains one subgroup per source signal DataFrame.

- Path pattern: ``/rawdata/<source_group>/``
- Each subgroup stores one DataFrame worth of columns as datasets.
- The subgroup keeps source metadata in attributes such as
  ``original_name`` and ``canonical_name``.

.. list-table::
   :header-rows: 1

   * - Path
     - Type
     - Description
   * - ``/rawdata``
     - Group
     - Container for all raw signals
   * - ``/rawdata`` attrs[``"empty"``]
     - Attr
     - If ``True``, no raw signals are stored
   * - ``/rawdata/<source_group>``
     - Group
     - One source DataFrame
   * - ``/rawdata/<source_group>/<column>``
     - Dataset
     - One DataFrame column, such as ``TIME`` or ``CH0``

If ``/rawdata`` is not marked ``empty=True``, it must contain at least one
child group.

``/stft`` and ``/cwt`` Groups
-----------------------------

The ``/stft`` and ``/cwt`` groups store transform results per source.

.. list-table::
   :header-rows: 1

   * - Path
     - Type
     - Description
   * - ``/stft/<source_group>``
     - Group
     - STFT results for one source
   * - ``/cwt/<source_group>``
     - Group
     - CWT results for one source
   * - ``/<transform>/<source_group>/<key>``
     - Dataset
     - ndarray-valued result entries
   * - ``/<transform>/<source_group>`` attrs
     - Attrs
     - Scalar metadata for that transform result

The loader reconstructs these groups into ``stft_results`` and
``cwt_results`` entries in the returned Python dictionary.

``/density`` Group
------------------

Density results are stored in structured frequency subgroups.

- Canonical subgroup naming is ``freq_<freq:.0f>G``.
- Examples: ``freq_94G``, ``freq_280G``
- Each subgroup contains one density DataFrame worth of columns.
- Relevant metadata is stored in subgroup attributes, including:
  ``freq``, ``n_ch``, ``n_path``, and ``meas_name``.

.. list-table::
   :header-rows: 1

   * - Path
     - Type
     - Description
   * - ``/density``
     - Group
     - Container for density outputs
   * - ``/density/freq_94G``
     - Group
     - Density data for the 94 GHz branch
   * - ``/density/freq_280G``
     - Group
     - Density data for the 280 GHz branch
   * - ``/density/freq_<...>/<column>``
     - Dataset
     - One density DataFrame column

The current loader expects this structured density layout.

``/vestdata`` Group
-------------------

The ``/vestdata`` group stores structured VEST results.

- Preferred layout is grouped by rate/source structure produced by the
  current writer.
- ``load_results_from_hdf5`` reads this through
  ``load_vest_structured(...)`` and exposes:

  - ``vest_data_by_rate``
  - ``vest_data``

The validator checks that ``/vestdata`` is a group if present, but does not
apply the same natural-name restriction used for IFI analysis groups.

Relation to Cache Files
-----------------------

Result files under ``ifi/results/<shot_num>/<shot_num>.h5``:

- Are written by ``save_results_to_hdf5``
- Follow the canonical schema described on this page
- Are intended for analysis outputs and downstream reuse

NAS raw-cache files:

- Are written separately by NAS DB helpers
- Are not part of this schema
- Should not be confused with canonical result files

Validator Summary
-----------------

Validation and inspection helpers were split out of ``io_h5`` into
``io_h5_inspect``.

- :func:`ifi.utils.io_h5_inspect.validate_h5_schema`

  - Ensures required root metadata attrs exist
  - Ensures ``/rawdata`` exists
  - Ensures non-empty ``/rawdata`` contains child groups
  - Ensures optional ``/stft``, ``/cwt``, and ``/density`` entries are groups
  - Ensures ``/vestdata`` is a group if present

- :func:`ifi.utils.io_h5_inspect.load_h5_data`

  - Calls ``validate_h5_schema`` first
  - Then delegates to ``load_results_from_hdf5``

- :func:`ifi.utils.io_h5_inspect.inspect_h5`

  - Prints a readable tree view of groups, datasets, and attributes

Compatibility Note
------------------

Older result layouts such as ``/metadata``, ``/signals``,
``/stft_results``, ``/cwt_results``, ``/density_data``, and flat legacy
VEST/density branches are no longer part of the supported on-disk schema.
The current loader targets only the canonical structure documented here.
