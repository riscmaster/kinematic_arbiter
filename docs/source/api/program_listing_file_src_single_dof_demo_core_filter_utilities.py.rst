
.. _program_listing_file_src_single_dof_demo_core_filter_utilities.py:

Program Listing for File filter_utilities.py
============================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_single_dof_demo_core_filter_utilities.py>` (``src/single_dof_demo/core/filter_utilities.py``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: py

   # Copyright (c) 2024 Spencer Maughan
   #
   # Permission is hereby granted, free of charge, to any person obtaining a copy
   # of this software and associated documentation files (the "Software"), to deal
   # in the Software without restriction.

   """
   Utility functions for converting between variance and confidence bounds.

   Provides functions to:
   - Convert variance to 3-sigma confidence bounds
   - Convert 3-sigma confidence bounds back to variance
   """

   import math

   THREE_SIGMA_BOUND = 3.0


   def variance_to_bound(variance: float) -> float:
       """Convert variance to 3-sigma confidence bounds."""
       return math.sqrt(variance) * THREE_SIGMA_BOUND


   def bound_to_variance(bound: float) -> float:
       """Convert 3-sigma confidence bounds back to variance."""
       return (bound / THREE_SIGMA_BOUND) ** 2
