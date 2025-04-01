
.. _program_listing_file_src_single_dof_demo_demo.py:

Program Listing for File demo.py
================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_single_dof_demo_demo.py>` (``src/single_dof_demo/demo.py``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: py

   # Copyright (c) 2024 Spencer Maughan
   #
   # Permission is hereby granted, free of charge, to any person obtaining a copy
   # of this software and associated documentation files (the "Software"), to deal
   # in the Software without restriction.

   """Demo application showcasing multiple Kalman filter
   implementations for signal processing.
   """

   import display_gui


   def _main():
       demo = display_gui.DisplayGui()
       demo.run()


   if __name__ == "__main__":
       _main()
