/**
 * @file test_utils.hpp
 * @brief Common utilities for testing kinematic_arbiter
 */

#pragma once

#include <Eigen/Dense>
#include <random>
#include <vector>

namespace kinematic_arbiter {
namespace test {

// Fixed seed for deterministic tests
constexpr unsigned FIXED_SEED = 42;

// Generate a random diagonally-dominant positive definite matrix
inline Eigen::MatrixXd generateDiagonallyDominantCovarianceMatrix(
    int size, std::mt19937& generator) {
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(size, size);

    // Fill the matrix with random values
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (i != j) {
                matrix(i, j) = dist(generator) * 0.1; // Off-diagonal elements are smaller
            }
        }
        // Make diagonally dominant
        matrix(i, i) = 1.0 + std::abs(dist(generator));
    }

    // Make symmetric
    matrix = 0.5 * (matrix + matrix.transpose());

    // Ensure positive definiteness
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(matrix);
    Eigen::VectorXd eigenvalues = eigenSolver.eigenvalues();
    double minEigenvalue = eigenvalues.minCoeff();

    if (minEigenvalue <= 0) {
        matrix += Eigen::MatrixXd::Identity(size, size) * (std::abs(minEigenvalue) + 0.1);
    }

    return matrix;
}

// Generate random samples from a multivariate normal distribution
inline std::vector<Eigen::VectorXd> generateSamples(
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& covariance,
    int numSamples,
    std::mt19937& generator
) {
    int dim = mean.size();
    std::vector<Eigen::VectorXd> samples;
    samples.reserve(numSamples);

    // Use Eigen's LLT (Cholesky) decomposition for sampling
    Eigen::MatrixXd L = covariance.llt().matrixL();

    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < numSamples; ++i) {
        Eigen::VectorXd z = Eigen::VectorXd::Zero(dim);
        for (int j = 0; j < dim; ++j) {
            z(j) = dist(generator);
        }

        Eigen::VectorXd sample = mean + L * z;
        samples.push_back(sample);
    }

    return samples;
}

// Calculate sample covariance matrix
inline Eigen::MatrixXd calculateSampleCovariance(
    const std::vector<Eigen::VectorXd>& samples,
    const Eigen::VectorXd& mean
) {
    if (samples.empty()) return Eigen::MatrixXd();

    int dim = samples[0].size();
    int n = samples.size();

    Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(dim, dim);

    for (const auto& sample : samples) {
        Eigen::VectorXd diff = sample - mean;
        covariance += diff * diff.transpose();
    }

    return covariance / (n - 1);
}

} // namespace test
} // namespace kinematic_arbiter
