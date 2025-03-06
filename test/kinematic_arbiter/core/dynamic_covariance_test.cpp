/**
 * @file dynamic_covariance_test.cpp
 * @brief Tests for dynamic covariance estimation in measurement and state models
 */

// Standard includes
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <vector>
#include <deque>
#include <iostream>
#include <sstream>

// Eigen
#include <Eigen/Dense>

// Kinematic arbiter includes
#include "kinematic_arbiter/core/state_index.hpp"
#include "kinematic_arbiter/core/measurement_model_interface.hpp"
#include "kinematic_arbiter/core/state_model_interface.hpp"

// Test utilities
#include "test_utils.hpp"

namespace kinematic_arbiter {
namespace core {
namespace testing {

// Define measurement vector type for tests (2D position measurement)
using TestMeasurementVector = Eigen::Vector2d;

// Simple test measurement model
class TestMeasurementModel : public MeasurementModelInterface<TestMeasurementVector> {
public:
    using Base = MeasurementModelInterface<TestMeasurementVector>;
    using StateVector = typename Base::StateVector;
    using MeasurementVector = typename Base::MeasurementVector;
    using MeasurementJacobian = typename Base::MeasurementJacobian;
    using MeasurementCovariance = typename Base::MeasurementCovariance;

    TestMeasurementModel()
        : Base(Eigen::Isometry3d::Identity()),
          generator_(42) {
        // Initialize measurement matrix to observe X, Y positions
        measurement_jacobian_ = MeasurementJacobian::Zero();
        measurement_jacobian_(0, StateIndex::Position::X) = 1.0;
        measurement_jacobian_(1, StateIndex::Position::Y) = 1.0;

        // Initialize true covariance used for testing
        true_measurement_covariance_ = MeasurementCovariance::Identity();
    }

    // Required interface methods
    MeasurementVector PredictMeasurement(const StateVector& state) const override {
        MeasurementVector measurement;
        measurement(0) = state(StateIndex::Position::X);
        measurement(1) = state(StateIndex::Position::Y);
        return measurement;
    }

    MeasurementJacobian GetMeasurementJacobian(const StateVector&) const override {
        return measurement_jacobian_;
    }

    // For testing - generate noisy measurement using the true covariance
    MeasurementVector GenerateNoisyMeasurement(const StateVector& state) {
        // Get the expected measurement
        MeasurementVector expected = PredictMeasurement(state);

        // Generate noise
        std::normal_distribution<double> dist(0.0, 1.0);
        MeasurementVector noise;
        noise(0) = dist(generator_);
        noise(1) = dist(generator_);

        // Calculate Cholesky decomposition of covariance
        Eigen::LLT<MeasurementCovariance> llt(true_measurement_covariance_);
        MeasurementCovariance L = llt.matrixL();

        // Return noisy measurement
        return expected + L * noise;
    }

    // Get the current estimated measurement covariance
    MeasurementCovariance GetEstimatedCovariance() const {
        return this->measurement_covariance_;
    }

    // Set the true measurement covariance (for test purposes)
    void SetTrueCovariance(const MeasurementCovariance& cov) {
        true_measurement_covariance_ = cov;
    }

    // Expose protected UpdateCovariance method for testing
    using Base::UpdateCovariance;

private:
    MeasurementJacobian measurement_jacobian_;
    MeasurementCovariance true_measurement_covariance_;
    std::mt19937 generator_;
};

// Simple test state model
class TestStateModel : public StateModelInterface {
public:
    using StateVector = StateModelInterface::StateVector;
    using StateMatrix = StateModelInterface::StateMatrix;

    // Default constructor
    TestStateModel()
        : StateModelInterface(),
          generator_(42) {
        // Initialize true process noise used for testing with larger values
        true_process_noise_ = StateMatrix::Identity() * 10.0;
    }

    // Constructor with params
    TestStateModel(const StateModelInterface::Params& params)
        : StateModelInterface(params),
          generator_(42) {
        // Initialize true process noise used for testing with larger values
        true_process_noise_ = StateMatrix::Identity() * 10.0;
    }

    // Required interface methods
    StateVector PredictState(const StateVector& state, double dt) const override {
        StateVector predicted = state;

        // Simple constant velocity model
        predicted(StateIndex::Position::X) += state(StateIndex::LinearVelocity::X) * dt;
        predicted(StateIndex::Position::Y) += state(StateIndex::LinearVelocity::Y) * dt;
        predicted(StateIndex::Position::Z) += state(StateIndex::LinearVelocity::Z) * dt;

        return predicted;
    }

    StateMatrix GetTransitionMatrix(const StateVector&, double dt) const override {
        StateMatrix A = StateMatrix::Identity();

        // Set velocity-to-position transition elements
        A(StateIndex::Position::X, StateIndex::LinearVelocity::X) = dt;
        A(StateIndex::Position::Y, StateIndex::LinearVelocity::Y) = dt;
        A(StateIndex::Position::Z, StateIndex::LinearVelocity::Z) = dt;

        return A;
    }

    // Generate noisy a priori state based on true process noise
    StateVector GenerateNoisyState(const StateVector& state, double dt) {
        // Get the expected next state
        StateVector predicted = PredictState(state, dt);

        // Generate noise vector
        std::normal_distribution<double> dist(0.0, 1.0);
        StateVector noise = StateVector::Zero();
        for (int i = 0; i < static_cast<int>(StateIndex::kFullStateSize); ++i) {
            noise(i) = dist(generator_);
        }

        // Apply noise using the true process noise matrix
        Eigen::LLT<StateMatrix> llt(true_process_noise_);
        StateMatrix L = llt.matrixL();

        return predicted + L * noise;
    }

    // Set the true process noise (for test purposes)
    void SetTrueProcessNoise(const StateMatrix& noise) {
        true_process_noise_ = noise;
    }

    // Set process noise directly (for initialization)
    void SetProcessNoise(const StateMatrix& noise) {
        this->process_noise_ = noise;
    }

    // Get the current estimated process noise
    StateMatrix GetEstimatedProcessNoise() const {
        return this->process_noise_;
    }

    // Expose protected UpdateProcessNoise method for testing
    using StateModelInterface::UpdateProcessNoise;

private:
    StateMatrix true_process_noise_;
    std::mt19937 generator_;
};

// Base test fixture
class DynamicCovarianceTest : public ::testing::Test {
protected:
    void SetUp() override {}

    const int measurementDim_ = 2; // 2D position measurements

    // Helper method to generate a random covariance
    Eigen::MatrixXd generateRandomCovariance(int size) {
        std::mt19937 gen(42);
        return kinematic_arbiter::test::generateDiagonallyDominantCovarianceMatrix(size, gen);
    }
};

// Basic test to verify model functionality
TEST_F(DynamicCovarianceTest, BasicModelTest) {
    TestMeasurementModel meas_model;
    TestStateModel state_model;

    // Create a test state
    StateModelInterface::StateVector state = StateModelInterface::StateVector::Zero();
    state(StateIndex::Position::X) = 1.0;
    state(StateIndex::Position::Y) = 2.0;
    state(StateIndex::LinearVelocity::X) = 0.5;
    state(StateIndex::LinearVelocity::Y) = 0.6;

    // Test measurement prediction
    auto measurement = meas_model.PredictMeasurement(state);
    EXPECT_EQ(measurement.size(), 2);
    EXPECT_DOUBLE_EQ(measurement(0), 1.0);
    EXPECT_DOUBLE_EQ(measurement(1), 2.0);

    // Test state prediction
    double dt = 1.0;
    auto next_state = state_model.PredictState(state, dt);
    EXPECT_DOUBLE_EQ(next_state(StateIndex::Position::X), 1.5);
    EXPECT_DOUBLE_EQ(next_state(StateIndex::Position::Y), 2.6);

    // Test transition matrix
    auto A = state_model.GetTransitionMatrix(state, dt);
    EXPECT_DOUBLE_EQ(A(StateIndex::Position::X, StateIndex::LinearVelocity::X), dt);
    EXPECT_DOUBLE_EQ(A(StateIndex::Position::Y, StateIndex::LinearVelocity::Y), dt);
}

// Test measurement covariance estimation with UpdateCovariance
TEST_F(DynamicCovarianceTest, MeasurementCovarianceEstimation) {
    // Create measurement model
    TestMeasurementModel model;

    // Define true measurement covariance to be estimated
    auto true_measurement_covariance = TestMeasurementModel::MeasurementCovariance::Identity() * 5.0;
    model.SetTrueCovariance(true_measurement_covariance);

    // String stream for detailed output in case of failure
    std::stringstream detailed_output;
    detailed_output << "\nMeasurement Covariance Estimation Error over iterations:" << std::endl;
    detailed_output << "Iteration\tRelative Error" << std::endl;

    // Setup random number generator
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<double> dist(0.0, 1.0);

    // Run test for 100 iterations
    const int iterations = 100;
    std::vector<double> errors;

    for (int i = 1; i <= iterations; ++i) {
        // Generate random measurement from true distribution
        TestMeasurementModel::MeasurementVector noise;
        noise(0) = dist(gen);
        noise(1) = dist(gen);

        // Calculate Cholesky decomposition of covariance to apply proper scaling
        Eigen::LLT<TestMeasurementModel::MeasurementCovariance> llt(true_measurement_covariance);
        TestMeasurementModel::MeasurementCovariance L = llt.matrixL();
        TestMeasurementModel::MeasurementVector scaled_noise = L * noise;

        // Update covariance estimate
        model.UpdateCovariance(scaled_noise);

        // Every 10 iterations, check the accuracy
        if (i % 10 == 0) {
            // Calculate error between estimated and true covariance
            auto estimated_covariance = model.GetEstimatedCovariance();
            double rel_error = (estimated_covariance - true_measurement_covariance).norm() /
                              true_measurement_covariance.norm();
            errors.push_back(rel_error);

            detailed_output << i << "\t" << rel_error << std::endl;
        }
    }

    // Get final error
    auto final_covariance = model.GetEstimatedCovariance();
    double final_rel_error = (final_covariance - true_measurement_covariance).norm() /
                            true_measurement_covariance.norm();

    detailed_output << "\nFinal measurement covariance relative error: " << final_rel_error << std::endl;

    // Test that error decreases
    EXPECT_LT(errors.back(), errors.front()) << "Measurement covariance error should decrease over time";

    // Test final accuracy (30% error acceptable for this test)
    bool passed = final_rel_error < 0.3;
    EXPECT_LT(final_rel_error, 0.3) << "Final measurement covariance should be accurate within 30%";

    // Only print detailed output if test failed
    if (!passed) {
        std::cout << detailed_output.str();
    }
}

// Test process noise estimation with UpdateProcessNoise
TEST_F(DynamicCovarianceTest, ProcessNoiseEstimation) {
    // Create state model
    StateModelInterface::Params params;
    params.process_noise_window = 20;
    TestStateModel model(params);

    // Set up true process noise and initial noise estimate
    double true_position_variance = 10.0;
    StateModelInterface::StateMatrix true_process_noise = StateModelInterface::StateMatrix::Zero();
    true_process_noise(StateIndex::Position::X, StateIndex::Position::X) = true_position_variance;
    true_process_noise(StateIndex::Position::Y, StateIndex::Position::Y) = true_position_variance;
    true_process_noise(StateIndex::Position::Z, StateIndex::Position::Z) = true_position_variance;

    StateModelInterface::StateMatrix initial_noise = StateModelInterface::StateMatrix::Identity() * 5.0;
    model.SetProcessNoise(initial_noise);

    // String stream for detailed output in case of failure
    std::stringstream detailed_output;
    detailed_output << "\nProcess Noise Estimation Error over iterations:" << std::endl;
    detailed_output << "Iteration\tRelative Error" << std::endl;

    // Setup random generator with fixed seed
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, std::sqrt(true_position_variance));

    // Run test for 2000 iterations
    const int iterations = 2000;
    const int print_interval = 200;
    const double dt = 0.1;
    const double process_to_measurement_ratio = 1.0;

    std::vector<double> position_errors;

    for (int i = 1; i <= iterations; ++i) {
        // Generate position noise from true distribution
        StateModelInterface::StateVector a_priori_state = StateModelInterface::StateVector::Zero();
        StateModelInterface::StateVector a_posteriori_state = StateModelInterface::StateVector::Zero();

        // Create innovations for position only
        StateModelInterface::StateVector state_diff = StateModelInterface::StateVector::Zero();
        state_diff(StateIndex::Position::X) = dist(gen);
        state_diff(StateIndex::Position::Y) = dist(gen);
        state_diff(StateIndex::Position::Z) = dist(gen);

        a_posteriori_state = a_priori_state - state_diff;

        // Update process noise
        model.UpdateProcessNoise(a_priori_state, a_posteriori_state,
                                process_to_measurement_ratio, dt);

        // Check accuracy at intervals
        if (i % print_interval == 0) {
            auto process_noise = model.GetEstimatedProcessNoise();
            double pos_x_var = process_noise(StateIndex::Position::X, StateIndex::Position::X);

            double pos_x_error = std::abs(pos_x_var - true_position_variance) / true_position_variance;
            position_errors.push_back(pos_x_error);

            detailed_output << i << "\t" << pos_x_error << std::endl;
            detailed_output << "  X position variance: True=" << true_position_variance
                          << " Estimated=" << pos_x_var << std::endl;
        }
    }

    // Get final process noise and calculate errors
    auto final_process_noise = model.GetEstimatedProcessNoise();
    double pos_x_error = std::abs(final_process_noise(StateIndex::Position::X, StateIndex::Position::X) -
                                true_position_variance) / true_position_variance;
    double pos_y_error = std::abs(final_process_noise(StateIndex::Position::Y, StateIndex::Position::Y) -
                                true_position_variance) / true_position_variance;
    double pos_z_error = std::abs(final_process_noise(StateIndex::Position::Z, StateIndex::Position::Z) -
                                true_position_variance) / true_position_variance;

    double avg_pos_error = (pos_x_error + pos_y_error + pos_z_error) / 3.0;

    detailed_output << "\nFinal position variance errors:" << std::endl;
    detailed_output << "  X: " << pos_x_error << std::endl;
    detailed_output << "  Y: " << pos_y_error << std::endl;
    detailed_output << "  Z: " << pos_z_error << std::endl;
    detailed_output << "Average position error: " << avg_pos_error << std::endl;

    // Test convergence - error should decrease over time
    EXPECT_LT(position_errors.back(), position_errors.front())
        << "Position variance error should decrease over time";

    // Test accuracy - final error should be less than 15%
    bool passed = avg_pos_error < 0.15;
    EXPECT_LT(avg_pos_error, 0.15)
        << "Average position variance error should be less than 15%";

    // Only print detailed output if test failed
    if (!passed) {
        std::cout << detailed_output.str();
    }
}

// Test updating process noise with only partial state updates
TEST_F(DynamicCovarianceTest, PartialStateProcessNoiseEstimation) {
    // Create state model with smaller window for faster convergence
    StateModelInterface::Params params;
    params.process_noise_window = 25; // Smaller window for faster adaptation
    TestStateModel model(params);

    // Define true process noise values for different state components
    const double true_position_variance = 2.0;
    const double true_velocity_variance = 0.5;

    // Set up initial process noise matrix with higher initial values
    StateModelInterface::StateMatrix initial_noise = StateModelInterface::StateMatrix::Identity() * 0.3;
    model.SetProcessNoise(initial_noise);

    // Setup RNG for consistent results
    std::mt19937 gen(42);
    std::normal_distribution<double> pos_dist(0.0, std::sqrt(true_position_variance));
    std::normal_distribution<double> vel_dist(0.0, std::sqrt(true_velocity_variance));

    // Parameters - increased iterations and process_to_measurement_ratio
    const int num_iterations = 2000;
    const double dt = 0.1;
    const double process_to_measurement_ratio = 2.0; // Higher ratio for faster convergence

    // Track position and velocity variance over iterations
    std::vector<double> pos_variance_history;
    std::vector<double> vel_variance_history;

    // Explicitly track cross-correlation to verify selective updates
    std::vector<double> pos_vel_correlation_history;

    // String stream for detailed output in case of failure
    std::stringstream detailed_output;
    detailed_output << "\nPartial State Process Noise Estimation Test:" << std::endl;
    detailed_output << "True position variance: " << true_position_variance << std::endl;
    detailed_output << "True velocity variance: " << true_velocity_variance << std::endl;

    for (int i = 0; i < num_iterations; ++i) {
        StateModelInterface::StateVector a_priori_state = StateModelInterface::StateVector::Zero();
        StateModelInterface::StateVector a_posteriori_state = StateModelInterface::StateVector::Zero();

        if (i % 2 == 0) {  // Even iterations: Position updates only
            // Generate position noise from true distribution
            StateModelInterface::StateVector state_diff = StateModelInterface::StateVector::Zero();
            state_diff(StateIndex::Position::X) = pos_dist(gen);
            state_diff(StateIndex::Position::Y) = pos_dist(gen);
            state_diff(StateIndex::Position::Z) = pos_dist(gen);

            a_posteriori_state = a_priori_state - state_diff;
        } else {  // Odd iterations: Velocity updates only
            // Generate velocity noise from true distribution
            StateModelInterface::StateVector state_diff = StateModelInterface::StateVector::Zero();
            state_diff(StateIndex::LinearVelocity::X) = vel_dist(gen);
            state_diff(StateIndex::LinearVelocity::Y) = vel_dist(gen);
            state_diff(StateIndex::LinearVelocity::Z) = vel_dist(gen);

            a_posteriori_state = a_priori_state - state_diff;
        }

        // Update process noise
        model.UpdateProcessNoise(a_priori_state, a_posteriori_state,
                                process_to_measurement_ratio, dt);

        // Record variance estimates periodically
        if (i % 200 == 0 || i == num_iterations - 1) {
            auto current_noise = model.GetEstimatedProcessNoise();
            double curr_pos_var = current_noise(StateIndex::Position::X, StateIndex::Position::X);
            double curr_vel_var = current_noise(StateIndex::LinearVelocity::X, StateIndex::LinearVelocity::X);
            double curr_corr = current_noise(StateIndex::Position::X, StateIndex::LinearVelocity::X);

            pos_variance_history.push_back(curr_pos_var);
            vel_variance_history.push_back(curr_vel_var);
            pos_vel_correlation_history.push_back(curr_corr);

            // Store current estimates in detailed output
            double pos_error = std::abs(curr_pos_var - true_position_variance) / true_position_variance * 100;
            double vel_error = std::abs(curr_vel_var - true_velocity_variance) / true_velocity_variance * 100;

            detailed_output << "Iteration " << i << ":" << std::endl;
            detailed_output << "  Position X variance: " << curr_pos_var
                      << " (error: " << pos_error << "%)" << std::endl;
            detailed_output << "  Velocity X variance: " << curr_vel_var
                      << " (error: " << vel_error << "%)" << std::endl;
            detailed_output << "  Pos-Vel correlation: " << curr_corr << std::endl;
        }
    }

    // Get final estimates
    auto final_noise = model.GetEstimatedProcessNoise();
    double final_pos_var = final_noise(StateIndex::Position::X, StateIndex::Position::X);
    double final_vel_var = final_noise(StateIndex::LinearVelocity::X, StateIndex::LinearVelocity::X);
    double final_pos_vel_corr = final_noise(StateIndex::Position::X, StateIndex::LinearVelocity::X);

    // Calculate relative errors
    double pos_rel_error = std::abs(final_pos_var - true_position_variance) / true_position_variance;
    double vel_rel_error = std::abs(final_vel_var - true_velocity_variance) / true_velocity_variance;

    detailed_output << "\nFinal Results:" << std::endl;
    detailed_output << "  Position X variance: " << final_pos_var
              << " (True: " << true_position_variance
              << ", Error: " << pos_rel_error * 100 << "%)" << std::endl;
    detailed_output << "  Velocity X variance: " << final_vel_var
              << " (True: " << true_velocity_variance
              << ", Error: " << vel_rel_error * 100 << "%)" << std::endl;
    detailed_output << "  Position-Velocity correlation: " << final_pos_vel_corr << std::endl;

    // Calculate convergence by comparing first and last estimates
    double pos_convergence = (pos_variance_history.back() - pos_variance_history.front()) /
                             pos_variance_history.front() * 100;
    double vel_convergence = (vel_variance_history.back() - vel_variance_history.front()) /
                             vel_variance_history.front() * 100;

    detailed_output << "Position variance convergence: " << pos_convergence << "%" << std::endl;
    detailed_output << "Velocity variance convergence: " << vel_convergence << "%" << std::endl;

    // Explicitly test selective update: check cross-correlations
    double max_corr = 0.0;
    for (double corr : pos_vel_correlation_history) {
        max_corr = std::max(max_corr, std::abs(corr));
    }
    detailed_output << "Maximum Position-Velocity correlation: " << max_corr << std::endl;

    // Test accuracy
    bool accuracy_passed = pos_rel_error < 0.3 && vel_rel_error < 0.3;
    EXPECT_LT(pos_rel_error, 0.3) << "Position variance should converge to true value";
    EXPECT_LT(vel_rel_error, 0.3) << "Velocity variance should converge to true value";

    // Test selective behavior - verify that cross-correlation elements stay zero
    bool selective_passed = max_corr < 0.01;
    EXPECT_LT(max_corr, 0.01)
        << "Position-velocity correlations should remain near zero with selective updates";

    // Test convergence direction
    bool convergence_passed = pos_convergence > 0 && vel_convergence > 0;
    EXPECT_GT(pos_convergence, 0) << "Position variance should increase toward true value";
    EXPECT_GT(vel_convergence, 0) << "Velocity variance should increase toward true value";

    // Only output details if any test failed
    if (!accuracy_passed || !selective_passed || !convergence_passed) {
        std::cout << detailed_output.str();
    }
}

}  // namespace testing
}  // namespace core
}  // namespace kinematic_arbiter
