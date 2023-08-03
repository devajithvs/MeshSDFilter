// BSD 3-Clause License
//
// Copyright (c) 2017, Bailin Deng
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef ITERATIVESDFILTER_H_
#define ITERATIVESDFILTER_H_

#include "EigenTypes.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <queue>
#include <set>
#include <utility>
#include <vector>

#ifdef USE_OPENMP
#include <omp.h>
#ifdef USE_MSVC
#define OMP_PARALLEL __pragma(omp parallel)
#define OMP_FOR __pragma(omp for)
#define OMP_SINGLE __pragma(omp single)
#else
#define OMP_PARALLEL _Pragma("omp parallel")
#define OMP_FOR _Pragma("omp for")
#define OMP_SINGLE _Pragma("omp single")
#endif
#else
#include <ctime>
#define OMP_PARALLEL
#define OMP_FOR
#define OMP_SINGLE
#endif

namespace SDFilter {

Eigen::MatrixXd
convertVectorToMatrix(const std::vector<std::vector<double>> &inputVector) {
  Eigen::MatrixXd outputMatrix;
  if (!inputVector.empty()) {
    outputMatrix.resize(inputVector.size(), inputVector[0].size());

    for (size_t i = 0; i < inputVector.size(); ++i) {
      for (size_t j = 0; j < inputVector[i].size(); ++j) {
        outputMatrix(i, j) = inputVector[i][j];
      }
    }
  }
  return outputMatrix;
}

Eigen::Matrix2Xi
convertVectorToMatrix(const std::vector<std::vector<size_s>> &inputVector) {
  Eigen::Matrix2Xi outputMatrix;
  if (!inputVector.empty()) {
    outputMatrix.resize(2, inputVector[0].size());

    for (size_t i = 0; i < inputVector[0].size(); ++i) {
      outputMatrix(0, i) = inputVector[0][i];
      outputMatrix(1, i) = inputVector[1][i];
    }
  }
  return outputMatrix;
}

std::vector<std::vector<double>>
convertMatrixToVector(const Eigen::MatrixXd &inputMatrix) {
  std::vector<std::vector<double>> outputVector(
      inputMatrix.rows(), std::vector<double>(inputMatrix.cols()));

  for (size_s i = 0; i < inputMatrix.rows(); ++i) {
    for (size_s j = 0; j < inputMatrix.cols(); ++j) {
      outputVector[i][j] = inputMatrix(i, j);
    }
  }

  return outputVector;
}

std::vector<std::vector<size_s>>
convertMatrixToVector(const Eigen::Matrix2Xi &inputMatrix) {
  std::vector<std::vector<size_s>> outputVector(
      inputMatrix.rows(), std::vector<size_s>(inputMatrix.cols()));

  for (size_s i = 0; i < inputMatrix.rows(); ++i) {
    for (size_s j = 0; j < inputMatrix.cols(); ++j) {
      outputVector[i][j] = inputMatrix(i, j);
    }
  }

  return outputVector;
}

Eigen::VectorXd
convertVectorToVectorXd(const std::vector<double> &inputVector) {
  Eigen::VectorXd outputVector(inputVector.size());

  for (size_t i = 0; i < inputVector.size(); ++i) {
    outputVector(i) = inputVector[i];
  }

  return outputVector;
}

std::vector<double>
convertVectorXdToVector(const Eigen::VectorXd &inputVector) {
  std::vector<double> outputVector(inputVector.size());

  for (size_s i = 0; i < inputVector.size(); ++i) {
    outputVector[i] = inputVector(i);
  }

  return outputVector;
}

class Parameters {
public:
  Parameters()
      : lambda(10), eta(1.0), mu(1.0), nu(1.0), max_iter(100),
        avg_disp_eps(1e-6), normalize_iterates(true) {}

  virtual ~Parameters() {}

  enum LinearSolverType { CG, LDLT };

  double lambda; // Regularization weight
  double eta; // Gaussian standard deviation for spatial weight, relative to the
              // bounding box diagonal length
  double mu;  // Gaussian standard deviation for guidance weight
  double nu;  // Gaussian standard deviation for signal weight

  // Parameters related to termination criteria
  int max_iter;        // Max number of iterations
  double avg_disp_eps; // Max average per-signal displacement threshold between
                       // two iterations for determining convergence
  bool normalize_iterates; // Normalization of the filtered normals in each
                           // iteration

  // Load options from file
  bool load(const char *filename) {
    std::ifstream ifile(filename);
    if (!ifile.is_open()) {
      std::cerr << "Error while opening file " << filename << std::endl;
      return false;
    }

    std::string line;
    while (std::getline(ifile, line)) {
      std::string::size_type pos = line.find_first_not_of(' ');
      if (pos == std::string::npos) {
        continue;
      }

      // Check for comment line
      else if (line.at(pos) == '#') {
        continue;
      }

      std::string::size_type end_pos = line.find_first_of(' ');
      std::string option_str = line.substr(pos, end_pos - pos);
      std::string value_str = line.substr(end_pos + 1, std::string::npos);
      OptionInterpreter opt(option_str, value_str);

      load_option(opt);
    }

    std::cout << "Successfully loaded options from file " << filename
              << std::endl;

    return true;
  }

  void output() {
    std::cout << std::endl;
    std::cout << "====== Filter parameters =========" << std::endl;
    output_options();
    std::cout << "==================================" << std::endl;
    std::cout << std::endl;
  }

  // Check whether the parameter values are valid
  virtual bool valid_parameters() const {
    if (lambda <= 0.0) {
      std::cerr << "Error: Lambda must be positive" << std::endl;
      return false;
    }

    if (eta <= 0.0) {
      std::cerr << "Error: Eta must be positive" << std::endl;
      return false;
    }

    if (mu <= 0.0) {
      std::cerr << "Error: Mu must be positive" << std::endl;
      return false;
    }

    if (nu <= 0.0) {
      std::cerr << "Error: Nu must be positive" << std::endl;
      return false;
    }

    if (max_iter < 1) {
      std::cerr << "Error: MaxIterations must be at least 1" << std::endl;
      return false;
    }

    if (avg_disp_eps <= 0.0) {
      std::cerr << "Error: average displacement threshold must be positive"
                << std::endl;
      return false;
    }

    return true;
  }

protected:
  class OptionInterpreter {
  public:
    OptionInterpreter(const std::string &option_str,
                      const std::string &value_str)
        : option_str_(option_str), value_str_(value_str) {}

    template <typename T>
    bool load(const std::string &target_option_name,
              T &target_option_value) const {
      if (option_str_ == target_option_name) {
        if (!load_value(value_str_, target_option_value)) {
          std::cerr << "Error loading option: " << target_option_name
                    << std::endl;
          return false;
        }

        return true;
      } else {
        return false;
      }
    }

    template <typename EnumT>
    bool load_enum(const std::string &target_option_name,
                   EnumT enum_value_count, EnumT &value) const {
      if (option_str_ == target_option_name) {
        int enum_int = 0;
        if (load_value(value_str_, enum_int)) {
          if (enum_int >= 0 && enum_int < static_cast<int>(enum_value_count)) {
            value = static_cast<EnumT>(enum_int);
            return true;
          }
        }

        std::cerr << "Error loading option: " << target_option_name
                  << std::endl;
        return false;
      } else {
        return false;
      }
    }

  private:
    std::string option_str_, value_str_;

    bool load_value(const std::string &str, double &value) const {
      try {
        value = std::stod(str);
      } catch (const std::invalid_argument &ia) {
        std::cerr << "Invalid argument: " << ia.what() << std::endl;
        return false;
      } catch (const std::out_of_range &oor) {
        std::cerr << "Out of Range error: " << oor.what() << std::endl;
        return false;
      }

      return true;
    }

    bool load_value(const std::string &str, int &value) const {
      try {
        value = std::stoi(str);
      } catch (const std::invalid_argument &ia) {
        std::cerr << "Invalid argument: " << ia.what() << std::endl;
        return false;
      } catch (const std::out_of_range &oor) {
        std::cerr << "Out of Range error: " << oor.what() << std::endl;
        return false;
      }

      return true;
    }

    bool load_value(const std::string &str, bool &value) const {
      int bool_value = 0;
      if (load_value(str, bool_value)) {
        value = (bool_value != 0);
        return true;
      } else {
        return false;
      }
    }
  };

  virtual bool load_option(const OptionInterpreter &opt) {
    return opt.load("Lambda", lambda) || opt.load("Eta", eta) ||
           opt.load("Mu", mu) || opt.load("Nu", nu) ||
           opt.load("MaxFilterIterations", max_iter);
  }

  virtual void output_options() {
    std::cout << "Lambda: " << lambda << std::endl;
    std::cout << "Eta:" << eta << std::endl;
    std::cout << "Mu:" << mu << std::endl;
    std::cout << "Nu:" << nu << std::endl;
  }
};

class Timer {
public:
  typedef int EventID;

  EventID get_time() {
    EventID id = time_values_.size();

#ifdef USE_OPENMP
    time_values_.push_back(omp_get_wtime());
#else
    time_values_.push_back(clock());
#endif

    return id;
  }

  double elapsed_time(EventID event1, EventID event2) {
    assert(event1 >= 0 && event1 < static_cast<EventID>(time_values_.size()));
    assert(event2 >= 0 && event2 < static_cast<EventID>(time_values_.size()));

#ifdef USE_OPENMP
    return time_values_[event2] - time_values_[event1];
#else
    return double(time_values_[event2] - time_values_[event1]) / CLOCKS_PER_SEC;
#endif
  }

private:
#ifdef USE_OPENMP
  std::vector<double> time_values_;
#else
  std::vector<clock_t> time_values_;
#endif
};

class SDFilter {
protected:
  SDFilter()
      : signal_dim_(-1), signal_count_(-1), print_progress_(true),
        print_timing_(true), print_diagnostic_info_(false) {}

  virtual ~SDFilter() {}

  bool filter(Parameters param) {
    std::cout << "Preprocessing......" << std::endl;

    Timer timer;
    Timer::EventID begin_time = timer.get_time();

    if (!initialize_filter(param)) {
      std::cerr << "Error: unable to initialize filter" << std::endl;
      return false;
    }

    Timer::EventID preprocess_end_time = timer.get_time();

    std::cout << "Filtering......" << std::endl;

    fixedpoint_solver(param);

    Timer::EventID filter_end_time = timer.get_time();

    if (print_timing_) {
      std::cout << "Preprocessing timing: "
                << timer.elapsed_time(begin_time, preprocess_end_time)
                << " secs" << std::endl;
      std::cout << "Filtering timing: "
                << timer.elapsed_time(preprocess_end_time, filter_end_time)
                << " secs" << std::endl;
    }

    return true;
  }

  void fixedpoint_solver(const Parameters &param) {
    // Store signals in the previous iteration
    std::vector<std::vector<double>> d_init_signals = d_signals_;
    std::vector<std::vector<double>> d_prev_signals;

    double weight_scaling_factor = 2 * param.nu * param.nu / param.lambda;
    std::vector<std::vector<double>> d_weighted_init_signals(
        d_signals_.size(), std::vector<double>(d_signals_[0].size(), 0.0));
    for (size_t i = 0; i < d_signals_.size(); ++i) {
      for (size_t j = 0; j < d_signals_[0].size(); ++j) {
        d_weighted_init_signals[i][j] +=
            d_signals_[i][j] * d_area_weights_[j] * weight_scaling_factor;
      }
    }

    std::vector<std::vector<double>> d_filtered_signals;
    double h = -0.5 / (param.nu * param.nu);

    size_s n_neighbor_pairs = d_neighboring_pairs_[0].size();

    // The weights for neighboring pairs that are used for convex combination of
    // neighboring signals in the fixed-point solver
    Eigen::VectorXd neighbor_pair_weights(n_neighbor_pairs);

    // Compute the termination threshold for area weighted squread norm of
    // signal change between two iterations

    double area_weights_sum =
        std::accumulate(d_area_weights_.begin(), d_area_weights_.end(), 0.0);
    double disp_sqr_norm_threshold =
        area_weights_sum * param.avg_disp_eps * param.avg_disp_eps;

    int output_frequency = 10;

    for (int num_iter = 1; num_iter <= param.max_iter; ++num_iter) {
      d_prev_signals = d_signals_;
      d_filtered_signals = d_weighted_init_signals;

      OMP_PARALLEL {
        OMP_FOR
        for (size_s i = 0; i < n_neighbor_pairs; ++i) {
          int idx1 = d_neighboring_pairs_[0][i],
              idx2 = d_neighboring_pairs_[1][i];

          double squaredNorm = 0.0;
          for (size_t k = 0; k < d_signals_.size(); ++k) {
            double diff = d_signals_[k][idx1] - d_signals_[k][idx2];
            squaredNorm += diff * diff;
          }

          neighbor_pair_weights(i) =
              precomputed_area_spatial_guidance_weights_(i) *
              std::exp(h * squaredNorm);
        }

        OMP_FOR
        for (int i = 0; i < signal_count_; ++i) {
          size_s neighbor_info_start_idx = neighborhood_info_boundaries_(i);
          size_s neighbor_info_end_idx = neighborhood_info_boundaries_(i + 1);

          for (size_s j = neighbor_info_start_idx; j < neighbor_info_end_idx;
               ++j) {
            size_s neighbor_idx = neighborhood_info_(0, j);
            size_s coef_idx = neighborhood_info_(1, j);

            for (size_t k = 0; k < d_signals_.size(); ++k) {
              d_filtered_signals[k][i] +=
                  d_signals_[k][neighbor_idx] * neighbor_pair_weights[coef_idx];
            }
          }

          if (param.normalize_iterates) {
            // Normalize column i of d_filtered_signals
            double norm = 0;
            for (size_t k = 0; k < d_filtered_signals.size(); ++k) {
              norm += d_filtered_signals[k][i] * d_filtered_signals[k][i];
            }
            norm = std::sqrt(norm);
            for (size_t k = 0; k < d_filtered_signals.size(); ++k) {
              d_filtered_signals[k][i] /= norm;
            }
          } else {
            // Divide column i of d_filtered_signals by
            // d_filtered_signals[signal_dim_][i]
            double denom = d_filtered_signals[signal_dim_][i];
            for (size_t k = 0; k < d_filtered_signals.size(); ++k) {
              d_filtered_signals[k][i] /= denom;
            }
          }
        }
      }

      d_signals_ = d_filtered_signals;

      double var_disp_sqrnorm = 0.0;
      for (size_t i = 0; i < d_signals_[0].size(); ++i) {
        double column_sum = 0.0;
        for (size_t j = 0; j < d_signals_.size(); ++j) {
          double diff = d_signals_[j][i] - d_prev_signals[j][i];
          column_sum += diff * diff;
        }
        var_disp_sqrnorm += d_area_weights_[i] * column_sum;
      }

      if (print_diagnostic_info_) {
        std::cout << "Iteration " << num_iter << ", Target function value "
                  << target_function(param, convertVectorToMatrix(d_signals_))
                  << std::endl;
      } else if (print_progress_ && num_iter % output_frequency == 0) {
        std::cout << "Iteration " << num_iter << "..." << std::endl;
      }

      if (var_disp_sqrnorm <= disp_sqr_norm_threshold) {
        std::cout << "Solver converged after " << num_iter << " iterations"
                  << std::endl;
        break;
      } else if (num_iter == param.max_iter) {
        std::cout << "Solver terminated after " << param.max_iter
                  << " iterations" << std::endl;
        break;
      }
    }
  }

  // Linear solver for symmetric positive definite matrix,
  class LinearSolver {
  public:
    LinearSolver(Parameters::LinearSolverType solver_type)
        : solver_type_(solver_type), pattern_analyzed(false) {}

    // Initialize the solver with matrix
    bool compute(const SparseMatrixXd &A) {
      if (solver_type_ == Parameters::LDLT) {
        if (!pattern_analyzed) {
          LDLT_solver_.analyzePattern(A);
          if (!check_error(LDLT_solver_, "Cholesky analyzePattern failed")) {
            return false;
          }

          pattern_analyzed = true;
        }

        LDLT_solver_.factorize(A);
        return check_error(LDLT_solver_, "Cholesky factorization failed");
      } else if (solver_type_ == Parameters::CG) {
        CG_solver_.compute(A);
        return check_error(CG_solver_, "CG solver compute failed");
      } else {
        return false;
      }
    }

    template <typename MatrixT> bool solve(const MatrixT &rhs, MatrixT &sol) {
      if (solver_type_ == Parameters::LDLT) {
#ifdef USE_OPENMP
        int n_cols = rhs.cols();

        OMP_PARALLEL {
          OMP_FOR
          for (int i = 0; i < n_cols; ++i) {
            sol.col(i) = LDLT_solver_.solve(rhs.col(i));
          }
        }
#else
        sol = LDLT_solver_.solve(rhs);
#endif

        return check_error(LDLT_solver_, "LDLT solve failed");
      } else if (solver_type_ == Parameters::CG) {
        sol = CG_solver_.solve(rhs);
        return check_error(CG_solver_, "CG solve failed");
      } else {
        return false;
      }
    }

    void reset_pattern() { pattern_analyzed = false; }

    void set_solver_type(Parameters::LinearSolverType type) {
      solver_type_ = type;
      if (solver_type_ == Parameters::LDLT) {
        reset_pattern();
      }
    }

  private:
    Parameters::LinearSolverType solver_type_;
    Eigen::SimplicialLDLT<SparseMatrixXd> LDLT_solver_;
    Eigen::ConjugateGradient<SparseMatrixXd, Eigen::Lower | Eigen::Upper>
        CG_solver_;

    bool pattern_analyzed; // Flag for symbolic factorization

    template <typename SolverT>
    bool check_error(const SolverT &solver, const std::string &error_message) {
      if (solver.info() != Eigen::Success) {
        std::cerr << error_message << std::endl;
      }

      return solver.info() == Eigen::Success;
    }
  };

protected:
  int signal_dim_;   // Dimension of the signals
  int signal_count_; // Number of signals

  std::vector<std::vector<double>>
      d_signals_; // Signals to be filtered. Represented in homogeneous form
                  // when there is no normalization constraint
  std::vector<double> d_area_weights_; // Area weights for each element

  std::vector<std::vector<size_s>> d_neighboring_pairs_;
  Eigen::Matrix2Xi neighboring_pairs_; // Each column stores the indices for a
                                       // pair of neighboring elements
  Eigen::VectorXd
      precomputed_area_spatial_guidance_weights_; // Precomputed weights (area,
                                                  // spatial Gaussian and
                                                  // guidance Gaussian) for
                                                  // neighboring pairs

  // The neighborhood information for each signal element is stored as
  // contiguous columns within the neighborhood_info_ matrix For each column,
  // the first element is the index of a neighboring element, the second one is
  // the corresponding address within array neighboring_pairs_
  Matrix2XIdx neighborhood_info_;
  VectorXIdx neighborhood_info_boundaries_; // Boundary positions for the
                                            // neighborhood information segments

  bool print_progress_;
  bool print_timing_;
  bool print_diagnostic_info_;

  // Overwrite this in a subclass to provide the initial spatial positions,
  // guidance, and signals.
  virtual void get_initial_data(Eigen::MatrixXd &guidance,
                                Eigen::MatrixXd &init_signals,
                                Eigen::VectorXd &area_weights) = 0;

  virtual void
  get_initial_data(std::vector<std::vector<double>> &d_guidance,
                   std::vector<std::vector<double>> &d_init_signals,
                   std::vector<double> &d_area_weights) = 0;

  bool initialize_filter(Parameters &param) {
    // Retrieve input signals and their area weights
    std::vector<std::vector<double>> d_guidance, d_init_signals;

    get_initial_data(d_guidance, d_init_signals, d_area_weights_);

    signal_dim_ = d_init_signals.size();
    signal_count_ = d_init_signals[0].size();
    if (signal_count_ <= 0) {
      return false;
    }

    if (param.normalize_iterates) {
      d_signals_ = d_init_signals;
    } else {
      d_signals_.resize(signal_dim_ + 1, std::vector<double>(signal_count_));
      for (int i = 0; i < signal_dim_; ++i) {
        for (int j = 0; j < signal_count_; ++j) {
          d_signals_[i][j] = d_init_signals[i][j];
        }
      }
      for (int j = 0; j < signal_count_; ++j) {
        d_signals_[signal_dim_][j] = 1.0;
      }
    }

    std::vector<double> d_neighbor_dists;
    if (!get_neighborhood(param, d_neighboring_pairs_, d_neighbor_dists)) {
      std::cerr
          << "Unable to get neighborhood information, no filtering done..."
          << std::endl;
      return false;
    }

    // Pre-compute filtering weights, and rescale the lambda parameter
    int n_neighbor_pairs = d_neighboring_pairs_[0].size();
    if (n_neighbor_pairs <= 0) {
      return false;
    }

    double h_spatial = -0.5 / (param.eta * param.eta);
    double h_guidance = -0.5 / (param.mu * param.mu);

    std::vector<double> d_precomputed_area_spatial_guidance_weights(
        n_neighbor_pairs);
    std::vector<double> d_area_spatial_weights(
        n_neighbor_pairs); // Area-integrated spatial weights, used for
                           // rescaling lambda

    for (int i = 0; i < n_neighbor_pairs; ++i) {
      int idx1 = d_neighboring_pairs_[0][i];
      int idx2 = d_neighboring_pairs_[1][i];

      double d = d_neighbor_dists[i];

      double squaredNorm = 0.0;
      for (size_t k = 0; k < d_guidance.size(); ++k) {
        double diff = d_guidance[k][idx1] - d_guidance[k][idx2];
        squaredNorm += diff * diff;
      }
      double result = std::exp(h_guidance * squaredNorm + h_spatial * d * d);

      d_area_spatial_weights[i] =
          (d_area_weights_[idx1] + d_area_weights_[idx2]) *
          std::exp(h_spatial * d * d);
      d_precomputed_area_spatial_guidance_weights[i] =
          (d_area_weights_[idx1] + d_area_weights_[idx2]) * result;
    }

    // Copy elements
    precomputed_area_spatial_guidance_weights_ =
        convertVectorToVectorXd(d_precomputed_area_spatial_guidance_weights);

    assert(d_neighbor_dists.size() > 0);

    double area_weights_sum =
        std::accumulate(d_area_weights_.begin(), d_area_weights_.end(), 0.0);
    double area_spatial_weights_sum = std::accumulate(
        d_area_spatial_weights.begin(), d_area_spatial_weights.end(), 0.0);

    param.lambda *=
        (area_weights_sum /
         area_spatial_weights_sum); // Rescale lambda to make regularization and
                                    // fidelity terms comparable

    // Pre-compute neighborhood_info_
    std::vector<std::vector<size_s>> neighbors(signal_count_);
    for (size_s i = 0; i < n_neighbor_pairs; ++i) {
      size_s idx1 = d_neighboring_pairs_[0][i];
      size_s idx2 = d_neighboring_pairs_[1][i];

      neighbors[idx1].push_back(idx2);
      neighbors[idx1].push_back(i);

      neighbors[idx2].push_back(idx1);
      neighbors[idx2].push_back(i);
    }

    neighborhood_info_boundaries_.resize(signal_count_ + 1);
    neighborhood_info_boundaries_(0) = 0;
    for (int i = 0; i < signal_count_; ++i) {
      neighborhood_info_boundaries_(i + 1) =
          neighborhood_info_boundaries_(i) + neighbors[i].size() / 2;
    }

    neighborhood_info_.resize(2, 2 * n_neighbor_pairs);

    for (int i = 0; i < signal_count_; ++i) {
      std::vector<size_s> &current_neighbor_info = neighbors[i];

      if (!current_neighbor_info.empty()) {
        size_s n_cols = current_neighbor_info.size() / 2;
        neighborhood_info_.block(0, neighborhood_info_boundaries_(i), 2,
                                 n_cols) =
            Eigen::Map<Matrix2XIdx>(current_neighbor_info.data(), 2, n_cols);
      }
    }

    return true;
  }

  // Find out all neighboring paris, as well as their distance
  virtual bool
  get_neighborhood(const Parameters &param,
                   std::vector<std::vector<size_s>> &d_neighbor_pairs,
                   std::vector<double> &d_neighbor_dist) = 0;

  virtual bool get_neighborhood(const Parameters &param,
                                Eigen::Matrix2Xi &neighbor_pairs,
                                Eigen::VectorXd &neighbor_dist) = 0;

  double target_function(const Parameters &param,
                         const Eigen::MatrixXd &init_signals) {
    // Compute regularizer term, using the contribution from each neighbor pair
    size_s n_neighbor_pairs = d_neighboring_pairs_[0].size();
    Eigen::VectorXd pair_values(n_neighbor_pairs);
    pair_values.setZero();
    double h = -0.5 / (param.nu * param.nu);

    OMP_PARALLEL {
      OMP_FOR
      for (size_s i = 0; i < n_neighbor_pairs; ++i) {
        int idx1 = d_neighboring_pairs_[0][i],
            idx2 = d_neighboring_pairs_[1][i];

        double squaredNorm = 0.0;
        for (size_t k = 0; k < d_signals_.size(); ++k) {
          double diff = d_signals_[k][idx1] - d_signals_[k][idx2];
          squaredNorm += diff * diff;
        }
        pair_values[i] = precomputed_area_spatial_guidance_weights_(i) *
                         std::max(0.0, 1.0 - std::exp(h * squaredNorm));
      }
    }

    double reg = pair_values.sum();

    // Compute the fidelity term, which is the squared difference between
    // current and initial signals, weighted by the areas
    double fid = 0.0;

    for (size_t col = 0; col < d_signals_[0].size(); ++col) {
      double colSum = 0.0;
      for (size_t row = 0; row < d_signals_.size(); ++row) {
        double diff = d_signals_[row][col] - init_signals(row, col);
        colSum += diff * diff;
      }
      fid += d_area_weights_[col] * colSum;
    }

    return fid + reg * param.lambda;
  }
};

} // namespace SDFilter

#endif /* ITERATIVESDFILTER_H_ */
