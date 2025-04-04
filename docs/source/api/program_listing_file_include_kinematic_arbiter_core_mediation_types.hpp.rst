
.. _program_listing_file_include_kinematic_arbiter_core_mediation_types.hpp:

Program Listing for File mediation_types.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file_include_kinematic_arbiter_core_mediation_types.hpp>` (``include/kinematic_arbiter/core/mediation_types.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #pragma once

   namespace kinematic_arbiter {
   namespace core {

   enum class MediationAction {
     ForceAccept,     // Proceed with measurement despite validation failure
     Reject,          // Reject the measurement entirely
     AdjustCovariance // Adjust covariance to make measurement valid
   };

   } // namespace core
   } // namespace kinematic_arbiter
