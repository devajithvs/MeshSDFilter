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

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ float atomicAdd(float *a, float b) { return b; }
#endif

#include "EigenTypes.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <set>
#include <utility>

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

void print_cuda_errors() {
  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(cudaError)
              << std::endl;
  }
}

template <typename EigenType, typename T>
void convert_to_gpu_memory(const EigenType &matrix, T **dev_matrix) {
  // Calculate total size of the matrix
  int totalSize = matrix.size();

  // Allocate memory on the GPU
  cudaMalloc((void **)dev_matrix, totalSize * sizeof(T));

  // Copy data from the CPU to the GPU
  cudaMemcpy(*dev_matrix, matrix.data(), totalSize * sizeof(T),
             cudaMemcpyHostToDevice);
}

template <typename EigenType, typename T>
void convert_from_gpu_memory(T *dev_matrix, EigenType &matrix) {
  // Copy data from the GPU to the CPU
  cudaMemcpy(matrix.data(), dev_matrix, matrix.size() * sizeof(T),
             cudaMemcpyDeviceToHost);

  // Free the GPU memory
  cudaFree(dev_matrix);
}

__global__ void accumulate(int num_elements, const float *array,
                           float *result) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  float sum = 0.0;
  for (int i = tid; i < num_elements; i += stride) {
    sum += array[i];
  }

  // Perform reduction within the block using shared memory
  extern __shared__ float shared_sum[];
  shared_sum[threadIdx.x] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
    }
    __syncthreads();
  }

  // Store the block sum in global memory
  if (threadIdx.x == 0) {
    atomicAdd(result, shared_sum[0]);
  }
}

__global__ void scaleAndWeightInitSignals(int signal_count, int signal_dim,
                                          float *dev_weighted_init_signals,
                                          float *dev_init_signals,
                                          float *dev_area_weights,
                                          float weight_scaling_factor) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < signal_dim * signal_count) {
    int i = idx % signal_dim;
    int j = idx / signal_dim;

    float weighted_value =
        dev_init_signals[idx] * dev_area_weights[j] * weight_scaling_factor;
    atomicAdd(&dev_weighted_init_signals[idx], weighted_value);
  }
}

__global__ void kernel_calculate_neighbor_pair_weights(
    int n_neighbor_pairs, int signal_dim, float h, int *dev_neighboring_pairs,
    float *dev_precomputed_area_spatial_guidance_weights, float *dev_signals,
    float *dev_neighbor_pair_weights) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_neighbor_pairs) {
    int idx1 = dev_neighboring_pairs[2 * i],
        idx2 = dev_neighboring_pairs[2 * i + 1];

    // Calculate squaredNorm of signal difference
    float signal_diff_norm_squared = 0.0;
    for (int j = 0; j < signal_dim; ++j) {
      float diff1 = dev_signals[j + signal_dim * idx1];
      float diff2 = dev_signals[j + signal_dim * idx2];
      float diff = diff1 - diff2;
      signal_diff_norm_squared += diff * diff;
    }

    float exp_term = exp(h * signal_diff_norm_squared);

    dev_neighbor_pair_weights[i] =
        dev_precomputed_area_spatial_guidance_weights[i] * exp_term;
  }
}

// __global__ void kernel_calculate_filtered_signals(
//     int signal_count, Eigen::Index signal_dim,
//     Eigen::Index *dev_neighborhood_info_boundaries,
//     Eigen::Index *dev_neighborhood_info, float *dev_neighbor_pair_weights,
//     float *dev_signals, float *dev_filtered_signals) {
//   int i = blockIdx.x * blockDim.x + threadIdx.x;
//   if (i < signal_count) {
//     int neighbor_info_start_idx = dev_neighborhood_info_boundaries[i];
//     int neighbor_info_end_idx = dev_neighborhood_info_boundaries[i + 1];

//     for (int j = neighbor_info_start_idx; j < neighbor_info_end_idx; ++j) {
//       int neighbor_idx = dev_neighborhood_info[2 * j];
//       int coef_idx = dev_neighborhood_info[2 * j + 1];

//       for (int k = 0; k < signal_dim; ++k) {
//         dev_filtered_signals[k + signal_dim * i] +=
//             dev_signals[k + signal_dim * neighbor_idx] *
//             dev_neighbor_pair_weights[coef_idx];
//       }
//     }

//     float sum = 0.0;
//     for (int k = 0; k < signal_dim; ++k) {
//       sum += dev_filtered_signals[i * signal_dim + k] *
//              dev_filtered_signals[i * signal_dim + k];
//     }
//     sum = sqrt(sum);
//     for (int k = 0; k < signal_dim; ++k) {
//       dev_filtered_signals[i * signal_dim + k] /= sum;
//     }
//     // if (normalize_iterates) {
//     // } else {
//     //   filtered_signals[i * signal_dim + signal_dim] =
//     //       filtered_signals[i * signal_dim + signal_dim] != 0.0f
//     //           ? filtered_signals[i * signal_dim + signal_dim]
//     //           : 1.0f;
//     //   for (int k = 0; k < signal_dim; ++k) {
//     //     filtered_signals[i * signal_dim + k] /=
//     //         filtered_signals[i * signal_dim + signal_dim];
//     //   }
//     // }
//   }
// }

__global__ void kernel_calculate_filtered_signals(
    int signal_count, Eigen::Index signal_dim,
    Eigen::Index *dev_neighborhood_info_boundaries,
    Eigen::Index *dev_neighborhood_info, float *dev_neighbor_pair_weights,
    float *dev_signals, float *dev_filtered_signals,
    float *dev_weighted_init_signals) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < signal_count) {
    int neighbor_info_start_idx = dev_neighborhood_info_boundaries[i];
    int neighbor_info_end_idx = dev_neighborhood_info_boundaries[i + 1];

    // Calculate the maximum range of indices to access
    int max_range = max(neighbor_info_end_idx - neighbor_info_start_idx, 1);

    // Initialize local_data
    float3 local_data = make_float3(0.0f, 0.0f, 0.0f);

    for (int j = 0; j < max_range; ++j) {
      int neighbor_idx =
          dev_neighborhood_info[2 * (neighbor_info_start_idx + j)];
      int coef_idx =
          dev_neighborhood_info[2 * (neighbor_info_start_idx + j) + 1];

      float weight = dev_neighbor_pair_weights[coef_idx];

      float3 neighbor_values;
      neighbor_values.x = dev_signals[3 * neighbor_idx];
      neighbor_values.y = dev_signals[3 * neighbor_idx + 1];
      neighbor_values.z = dev_signals[3 * neighbor_idx + 2];

      local_data.x += neighbor_values.x * weight;
      local_data.y += neighbor_values.y * weight;
      local_data.z += neighbor_values.z * weight;
    }

    dev_filtered_signals[3 * i + 0] = local_data.x;
    dev_filtered_signals[3 * i + 1] = local_data.y;
    dev_filtered_signals[3 * i + 2] = local_data.z;
  }
}

__global__ void kernel_calculate_filtered_signals2(
    int signal_count, Eigen::Index signal_dim,
    Eigen::Index *dev_neighborhood_info_boundaries,
    Eigen::Index *dev_neighborhood_info, float *dev_neighbor_pair_weights,
    float *dev_signals, float *dev_filtered_signals,
    float *dev_weighted_init_signals) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < signal_count) {
    float sum = 0.0;
    for (int k = 0; k < signal_dim; ++k) {
      dev_filtered_signals[i * signal_dim + k] +=
          dev_weighted_init_signals[i * signal_dim + k];
      float curr = dev_filtered_signals[i * signal_dim + k];
      sum += curr * curr;
    }
    float inv_sum_sqrt = 1.0 / sqrt(sum);
    for (int k = 0; k < signal_dim; ++k) {
      dev_filtered_signals[i * signal_dim + k] *= inv_sum_sqrt;
    }
  }
}

__global__ void kernel_calculate_var_disp_sqrnorm(int signal_count,
                                                  Eigen::Index signal_dim,
                                                  float *dev_signals,
                                                  float *dev_prev_signals,
                                                  float *dev_area_weights,
                                                  float *dev_var_disp_sqrnorm) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < signal_count) {
    float sqrnorm = 0.0;

    for (int j = 0; j < signal_dim; ++j) {
      float diff = dev_signals[i * signal_dim + j] -
                   dev_prev_signals[i * signal_dim + j];
      sqrnorm += diff * diff;
    }
    atomicAdd(dev_var_disp_sqrnorm, dev_area_weights[i] * sqrnorm);
  }
}

__global__ void kernel_calculate_var_disp_sqrnorm_opt(
    int signal_count, Eigen::Index signal_dim, float *dev_signals,
    float *dev_prev_signals, float *dev_area_weights,
    float *dev_var_disp_sqrnorm) {
  extern __shared__ float shared_mem[];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < signal_count) {
    float sqrNorm = 0.0;

    for (int j = threadIdx.y; j < signal_dim; j += blockDim.y) {
      float diff = dev_signals[i * signal_dim + j] -
                   dev_prev_signals[i * signal_dim + j];
      sqrNorm += diff * diff;
    }

    // Store intermediate results in shared memory
    shared_mem[threadIdx.y * blockDim.x + threadIdx.x] = sqrNorm;
    __syncthreads();

    // Perform parallel reduction using shared memory
    for (int s = blockDim.y / 2; s > 0; s >>= 1) {
      if (threadIdx.y < s) {
        shared_mem[threadIdx.y * blockDim.x + threadIdx.x] +=
            shared_mem[(threadIdx.y + s) * blockDim.x + threadIdx.x];
      }
      __syncthreads();
    }

    // Write final result to global memory
    if (threadIdx.y == 0) {
      atomicAdd(dev_var_disp_sqrnorm,
                dev_area_weights[i] * shared_mem[threadIdx.x]);
    }
  }
}

class Parameters {
public:
  Parameters()
      : lambda(10), eta(1.0), mu(1.0), nu(1.0), max_iter(100),
        avg_disp_eps(1e-6), normalize_iterates(true) {}

  virtual ~Parameters() {}

  enum LinearSolverType { CGChol, CGBig, CG, LDLT };

  float lambda; // Regularization weight
  float eta; // Gaussian standard deviation for spatial weight, relative to the
             // bounding box diagonal length
  float mu;  // Gaussian standard deviation for guidance weight
  float nu;  // Gaussian standard deviation for signal weight

  // Parameters related to termination criteria
  int max_iter;        // Max number of iterations
  float avg_disp_eps;  // Max average per-signal displacement threshold between
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

    bool load_value(const std::string &str, float &value) const {
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

  float elapsed_time(EventID event1, EventID event2) {
    assert(event1 >= 0 && event1 < static_cast<EventID>(time_values_.size()));
    assert(event2 >= 0 && event2 < static_cast<EventID>(time_values_.size()));

#ifdef USE_OPENMP
    return time_values_[event2] - time_values_[event1];
#else
    return float(time_values_[event2] - time_values_[event1]) / CLOCKS_PER_SEC;
#endif
  }

private:
#ifdef USE_OPENMP
  std::vector<float> time_values_;
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

    std::cout << "Filtering......" << std::endl;

    Timer::EventID preprocess_end_time = timer.get_time();
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
    int block_size = 512;
    Eigen::MatrixXf init_signals = signals_;

    float *dev_signals;
    convert_to_gpu_memory(signals_, &dev_signals);

    float *dev_area_weights;
    convert_to_gpu_memory(area_weights_, &dev_area_weights);

    float *dev_weighted_init_signals;
    cudaMalloc((void **)&dev_weighted_init_signals,
               (signal_dim_ * signal_count_) * sizeof(float));

    // Weighted initial signals, as used in the fixed-point solver
    float weight_scaling_factor = 2 * param.nu * param.nu / param.lambda;
    scaleAndWeightInitSignals<<<(signal_dim_ * signal_count_ + block_size - 1) /
                                    block_size,
                                block_size>>>(
        signal_count_, signal_dim_, dev_weighted_init_signals, dev_signals,
        dev_area_weights, weight_scaling_factor);

    float h = -0.5 / (param.nu * param.nu);

    Eigen::Index n_neighbor_pairs = neighboring_pairs_.cols();

    // Compute the termination threshold for area weighted squread norm of
    // signal change between two iterations
    float *dev_area_weights_sum;
    cudaMalloc(&dev_area_weights_sum, sizeof(float));
    cudaMemset(dev_area_weights_sum, 0, sizeof(float));

    int num_elements = area_weights_.size();
    int grid_size = (num_elements + block_size - 1) / block_size;

    accumulate<<<grid_size, block_size, block_size * sizeof(float)>>>(
        num_elements, dev_area_weights, dev_area_weights_sum);

    float area_weights_sum;
    cudaMemcpy(&area_weights_sum, dev_area_weights_sum, sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(dev_area_weights_sum);

    float disp_sqr_norm_threshold =
        area_weights_sum * param.avg_disp_eps * param.avg_disp_eps;

    int output_frequency = 10;

    int *dev_neighboring_pairs;
    convert_to_gpu_memory(neighboring_pairs_, &dev_neighboring_pairs);

    float *dev_precomputed_area_spatial_guidance_weights;
    convert_to_gpu_memory(precomputed_area_spatial_guidance_weights_,
                          &dev_precomputed_area_spatial_guidance_weights);

    Eigen::Index *dev_neighborhood_info_boundaries;
    convert_to_gpu_memory(neighborhood_info_boundaries_,
                          &dev_neighborhood_info_boundaries);

    Eigen::Index *dev_neighborhood_info;
    convert_to_gpu_memory(neighborhood_info_, &dev_neighborhood_info);

    float *dev_var_disp_sqrnorm;
    cudaMalloc((void **)&dev_var_disp_sqrnorm, sizeof(float));

    // The weights for neighboring pairs that are used for convex combination of
    // neighboring signals in the fixed-point solver
    float *dev_neighbor_pair_weights;
    cudaMalloc((void **)&dev_neighbor_pair_weights,
               n_neighbor_pairs * sizeof(float));

    float *dev_filtered_signals;
    cudaMalloc((void **)&dev_filtered_signals, signals_.size() * sizeof(float));

    int grid_size_weights = (n_neighbor_pairs + block_size - 1) / block_size;
    int grid_size_filtered = (signal_count_ + block_size - 1) / block_size;

    dim3 threads_per_block(
        32,
        8); // You can adjust these values based on your GPU's architecture
    int grid_size_sqrnorm =
        (signals_.cols() + threads_per_block.x - 1) / threads_per_block.x;

    int shared_size = threads_per_block.x * threads_per_block.y * sizeof(float);

    for (int num_iter = 1; num_iter <= param.max_iter; ++num_iter) {
      cudaMemset(dev_var_disp_sqrnorm, 0, sizeof(float));
      cudaMemset(dev_filtered_signals, 0, signals_.size() * sizeof(float));

      // cudaMemcpy(dev_filtered_signals, dev_weighted_init_signals,
      //            signals_.size() * sizeof(float), cudaMemcpyDeviceToDevice);

      kernel_calculate_neighbor_pair_weights<<<grid_size_weights, block_size>>>(
          n_neighbor_pairs, signals_.rows(), h, dev_neighboring_pairs,
          dev_precomputed_area_spatial_guidance_weights, dev_signals,
          dev_neighbor_pair_weights);

      // Calculate the shared memory size per block
      kernel_calculate_filtered_signals<<<grid_size_filtered, block_size>>>(
          signal_count_, signal_dim_, dev_neighborhood_info_boundaries,
          dev_neighborhood_info, dev_neighbor_pair_weights, dev_signals,
          dev_filtered_signals, dev_weighted_init_signals);

      kernel_calculate_filtered_signals2<<<grid_size_filtered, block_size>>>(
          signal_count_, signal_dim_, dev_neighborhood_info_boundaries,
          dev_neighborhood_info, dev_neighbor_pair_weights, dev_signals,
          dev_filtered_signals, dev_weighted_init_signals);

      // kernel_calculate_var_disp_sqrnorm<<<grid_size_filtered, block_size>>>(
      kernel_calculate_var_disp_sqrnorm_opt<<<grid_size_sqrnorm,
                                              threads_per_block, shared_size>>>(
          signal_count_, signal_dim_, dev_filtered_signals, dev_signals,
          dev_area_weights, dev_var_disp_sqrnorm);

      float var_disp_sqrnorm;
      cudaMemcpy(&var_disp_sqrnorm, dev_var_disp_sqrnorm, sizeof(float),
                 cudaMemcpyDeviceToHost);

      print_cuda_errors();
      std::swap(dev_signals, dev_filtered_signals);

      if (print_diagnostic_info_) {
        std::cout << "Iteration " << num_iter << ", Target function value "
                  << target_function(param, init_signals) << std::endl;
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

    cudaFree(dev_neighboring_pairs);
    cudaFree(dev_precomputed_area_spatial_guidance_weights);

    cudaFree(dev_neighborhood_info_boundaries);
    cudaFree(dev_neighborhood_info);

    convert_from_gpu_memory(dev_signals, signals_);
    cudaFree(dev_var_disp_sqrnorm);
    cudaFree(dev_area_weights);

    cudaFree(dev_neighbor_pair_weights);

    cudaFree(dev_weighted_init_signals);
    cudaFree(dev_filtered_signals);
  }

  // Linear solver for symmetric positive definite matrix,
  class LinearSolver {
  public:
    LinearSolver(Parameters::LinearSolverType solver_type)
        : solver_type_(solver_type), pattern_analyzed(false) {}

    // Initialize the solver with matrix
    bool compute(const SparseMatrixXf &A) {
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
    Eigen::SimplicialLDLT<SparseMatrixXf> LDLT_solver_;
    Eigen::ConjugateGradient<SparseMatrixXf, Eigen::Lower | Eigen::Upper>
        CG_solver_;

    bool pattern_analyzed; // Flag for symbolic factorization

    template <typename SolverT>
    bool check_error(const SolverT &solver, const std::string &error_message) {
      if (solver.info() != Eigen::Success) {

        std::cerr << "Error happened" << std::endl;
        std::cerr << error_message << std::endl;
      }

      return solver.info() == Eigen::Success;
    }
  };

protected:
  int signal_dim_;   // Dimension of the signals
  int signal_count_; // Number of signals

  Eigen::MatrixXf
      signals_; // Signals to be filtered. Represented in homogeneous form
                // when there is no normalization constraint
  Eigen::VectorXf area_weights_; // Area weights for each element

  Eigen::Matrix2Xi neighboring_pairs_; // Each column stores the indices for a
                                       // pair of neighboring elements
  Eigen::VectorXf
      precomputed_area_spatial_guidance_weights_; // Precomputed weights
                                                  // (area, spatial Gaussian
                                                  // and guidance Gaussian)
                                                  // for neighboring pairs

  // The neighborhood information for each signal element is stored as
  // contiguous columns within the neighborhood_info_ matrix For each column,
  // the first element is the index of a neighboring element, the second one
  // is the corresponding address within array neighboring_pairs_
  Matrix2XIdx neighborhood_info_;
  VectorXIdx neighborhood_info_boundaries_; // Boundary positions for the
                                            // neighborhood information segments

  bool print_progress_;
  bool print_timing_;
  bool print_diagnostic_info_;

  // Overwrite this in a subclass to provide the initial spatial positions,
  // guidance, and signals.
  virtual void get_initial_data(Eigen::MatrixXf &guidance,
                                Eigen::MatrixXf &init_signals,
                                Eigen::VectorXf &area_weights) = 0;

  bool initialize_filter(Parameters &param) {

    // Retrive input signals and their area weights
    Eigen::MatrixXf guidance, init_signals;
    get_initial_data(guidance, init_signals, area_weights_);

    signal_dim_ = init_signals.rows();
    signal_count_ = init_signals.cols();
    if (signal_count_ <= 0) {
      return false;
    }

    if (param.normalize_iterates) {
      signals_ = init_signals;
    } else {
      signals_.resize(signal_dim_ + 1, signal_count_);
      signals_.block(0, 0, signal_dim_, signal_count_) = init_signals;
      signals_.row(signal_dim_).setOnes();
    }

    Eigen::VectorXf neighbor_dists;
    if (!get_neighborhood(param, neighboring_pairs_, neighbor_dists)) {
      std::cerr
          << "Unable to get neighborhood information, no filtering done..."
          << std::endl;
      return false;
    }

    // Pre-compute filtering weights, and rescale the lambda parameter

    Eigen::Index n_neighbor_pairs = neighboring_pairs_.cols();
    if (n_neighbor_pairs <= 0) {
      return false;
    }

    precomputed_area_spatial_guidance_weights_.resize(n_neighbor_pairs);
    float h_spatial = -0.5 / (param.eta * param.eta);
    float h_guidance = -0.5 / (param.mu * param.mu);
    Eigen::VectorXf area_spatial_weights(
        n_neighbor_pairs); // Area-integrated spatial weights, used for
                           // rescaling lambda

    OMP_PARALLEL {
      OMP_FOR
      for (Eigen::Index i = 0; i < n_neighbor_pairs; ++i) {
        // Read the indices of a neighboring pair, and their distance
        int idx1 = neighboring_pairs_(0, i), idx2 = neighboring_pairs_(1, i);
        float d = neighbor_dists(i);

        // Compute the weights associated with the pair
        area_spatial_weights(i) = (area_weights_(idx1) + area_weights_(idx2)) *
                                  std::exp(h_spatial * d * d);
        precomputed_area_spatial_guidance_weights_(i) =
            (area_weights_(idx1) + area_weights_(idx2)) *
            std::exp(
                h_guidance *
                    (guidance.col(idx1) - guidance.col(idx2)).squaredNorm() +
                h_spatial * d * d);
      }
    }

    assert(neighbor_dists.size() > 0);
    param.lambda *=
        (area_weights_.sum() /
         area_spatial_weights.sum()); // Rescale lambda to make regularization
                                      // and fidelity terms comparable

    // Pre-compute neighborhood_info_
    std::vector<std::vector<Eigen::Index>> neighbors(signal_count_);
    for (Eigen::Index i = 0; i < n_neighbor_pairs; ++i) {
      int idx1 = neighboring_pairs_(0, i), idx2 = neighboring_pairs_(1, i);

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

    OMP_PARALLEL {
      OMP_FOR
      for (int i = 0; i < signal_count_; ++i) {
        std::vector<Eigen::Index> &current_neighbor_info = neighbors[i];

        if (!current_neighbor_info.empty()) {
          Eigen::Index n_cols = current_neighbor_info.size() / 2;
          neighborhood_info_.block(0, neighborhood_info_boundaries_(i), 2,
                                   n_cols) =
              Eigen::Map<Matrix2XIdx>(current_neighbor_info.data(), 2, n_cols);
        }
      }
    }

    return true;
  }

  // Find out all neighboring paris, as well as their distance
  virtual bool get_neighborhood(const Parameters &param,
                                Eigen::Matrix2Xi &neighbor_pairs,
                                Eigen::VectorXf &neighbor_dist) = 0;

  float target_function(const Parameters &param,
                        const Eigen::MatrixXf &init_signals) {
    // Compute regularizer term, using the contribution from each neighbor
    // pair
    Eigen::Index n_neighbor_pairs = neighboring_pairs_.cols();
    Eigen::VectorXf pair_values(n_neighbor_pairs);
    pair_values.setZero();
    float h = -0.5 / (param.nu * param.nu);

    OMP_PARALLEL {
      OMP_FOR
      for (Eigen::Index i = 0; i < n_neighbor_pairs; ++i) {
        int idx1 = neighboring_pairs_(0, i), idx2 = neighboring_pairs_(1, i);
        pair_values[i] = precomputed_area_spatial_guidance_weights_(i) *
                         std::max(0.0, 1.0 - std::exp(h * (signals_.col(idx1) -
                                                           signals_.col(idx2))
                                                              .squaredNorm()));
      }
    }

    float reg = pair_values.sum();

    // Compute the fidelity term, which is the squared difference between
    // current and initial signals, weighted by the areas
    float fid =
        area_weights_.dot((signals_ - init_signals).colwise().squaredNorm());

    return fid + reg * param.lambda;
  }
};

} // namespace SDFilter

#endif /* ITERATIVESDFILTER_H_ */
