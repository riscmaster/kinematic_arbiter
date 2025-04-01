
.. _program_listing_file_include_kinematic_arbiter_core_state_index.hpp:

Program Listing for File state_index.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file_include_kinematic_arbiter_core_state_index.hpp>` (``include/kinematic_arbiter/core/state_index.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #pragma once

   #include <cstddef>
   #include <vector>
   #include <string>
   #include <unordered_map>
   #include <Eigen/Dense>

   namespace kinematic_arbiter {
   namespace core {

   struct StateIndex {
     // Position indices (3D)
     struct Position {
       static constexpr int X = 0;
       static constexpr int Y = 1;
       static constexpr int Z = 2;

       static constexpr int Begin() { return X; }
       static constexpr int Size() { return 3; }
     };

     // Orientation indices (quaternion)
     struct Quaternion {
       static constexpr int W = 3;
       static constexpr int X = 4;
       static constexpr int Y = 5;
       static constexpr int Z = 6;

       static constexpr int Begin() { return W; }
       static constexpr int Size() { return 4; }
     };

     // Linear velocity indices
     struct LinearVelocity {
       static constexpr int X = 7;
       static constexpr int Y = 8;
       static constexpr int Z = 9;

       static constexpr int Begin() { return X; }
       static constexpr int Size() { return 3; }
     };

     // Angular velocity indices
     struct AngularVelocity {
       static constexpr int X = 10;
       static constexpr int Y = 11;
       static constexpr int Z = 12;

       static constexpr int Begin() { return X; }
       static constexpr int Size() { return 3; }
     };

     // Linear acceleration indices
     struct LinearAcceleration {
       static constexpr int X = 13;
       static constexpr int Y = 14;
       static constexpr int Z = 15;

       static constexpr int Begin() { return X; }
       static constexpr int Size() { return 3; }
     };

     // Angular acceleration indices
     struct AngularAcceleration {
       static constexpr int X = 16;
       static constexpr int Y = 17;
       static constexpr int Z = 18;

       static constexpr int Begin() { return X; }
       static constexpr int Size() { return 3; }
     };

     // Total size of the state vector for a full rigid body state
     static constexpr std::size_t kFullStateSize = 19;
   };

   // Function to get the names of initializable states
   inline std::vector<std::string> GetInitializableStateNames(const Eigen::Array<bool, 1, StateIndex::kFullStateSize>& initializable_states) {
       // Map of state indices to their names
       static const std::unordered_map<int, std::string> index_to_name = {
           {StateIndex::Position::X, "Position X"},
           {StateIndex::Position::Y, "Position Y"},
           {StateIndex::Position::Z, "Position Z"},
           {StateIndex::Quaternion::W, "Quaternion W"},
           {StateIndex::Quaternion::X, "Quaternion X"},
           {StateIndex::Quaternion::Y, "Quaternion Y"},
           {StateIndex::Quaternion::Z, "Quaternion Z"},
           {StateIndex::LinearVelocity::X, "Linear Velocity X"},
           {StateIndex::LinearVelocity::Y, "Linear Velocity Y"},
           {StateIndex::LinearVelocity::Z, "Linear Velocity Z"},
           {StateIndex::AngularVelocity::X, "Angular Velocity X"},
           {StateIndex::AngularVelocity::Y, "Angular Velocity Y"},
           {StateIndex::AngularVelocity::Z, "Angular Velocity Z"},
           {StateIndex::LinearAcceleration::X, "Linear Acceleration X"},
           {StateIndex::LinearAcceleration::Y, "Linear Acceleration Y"},
           {StateIndex::LinearAcceleration::Z, "Linear Acceleration Z"},
           {StateIndex::AngularAcceleration::X, "Angular Acceleration X"},
           {StateIndex::AngularAcceleration::Y, "Angular Acceleration Y"},
           {StateIndex::AngularAcceleration::Z, "Angular Acceleration Z"}
       };

       // Vector to store the names of the initializable states
       std::vector<std::string> state_names;

       // Populate the vector with names based on the indices
       for (int index : initializable_states) {
           auto it = index_to_name.find(index);
           if (it != index_to_name.end()) {
               state_names.push_back(it->second);
           }
       }

       return state_names;
   }

   } // namespace core
   } // namespace kinematic_arbiter
