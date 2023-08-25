#include "SDFilter.h"

#include "cuda_runtime.h"
#include <cusolverSp.h>
// #include <cusparse.h>
#include <iostream>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t cudaStatus = call;                                             \
    if (cudaStatus != cudaSuccess) {                                           \
      std::cerr << "CUDA Error: " << cudaGetErrorString(cudaStatus) << " in "  \
                << __FILE__ << " at line " << __LINE__ << std::endl;           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Linear solver for symmetric positive definite matrix,
namespace SDFilter {

void solveUsingCusolver(const int n, const int nnz, const int *d_csrRowPtr,
                        const int *d_csrColInd, const double *d_csrVal,
                        const double *d_b, double *d_x) {
  cusolverSpHandle_t handle;
  cudaStream_t stream = nullptr;

  CUDA_CHECK(cudaSetDevice(0)); // Set the CUDA device if necessary
  cusolverSpCreate(&handle);

  CUDA_CHECK(cudaStreamCreate(&stream));
  cusolverSpSetStream(handle, stream);

  cusparseMatDescr_t descrA;
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

  int reorder = 0;     // No reordering
  int singularity = 0; // 0 if the matrix is non-singular

  cusolverStatus_t status = cusolverSpDcsrlsvchol(
      handle, n, nnz, descrA, d_csrVal, d_csrRowPtr, d_csrColInd, d_b, 1e-12,
      reorder, d_x, &singularity);

  if (status != CUSOLVER_STATUS_SUCCESS) {
    std::cerr << "cusolverSpDcsrlsvchol failed" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Clean up
  cusolverSpDestroy(handle);
  cusparseDestroyMatDescr(descrA);
  CUDA_CHECK(cudaStreamDestroy(stream));
}

class LinearSolverGPU {
public:
  LinearSolverGPU(Parameters::LinearSolverType solver_type)
      : solver_type_(solver_type) {}

  // Initialize the solver with matrix
  bool compute(int n1, int nnz1, int *csrRowPtr, int *csrColInd,
               double *csrVal) {
    if (solver_type_ == Parameters::LDLT) {
      std::cerr << "initialization0 working" << std::endl;
      n = n1;
      nnz = nnz1;

      CUDA_CHECK(cudaMalloc((void **)&d_csrVal, nnz * sizeof(double)));
      CUDA_CHECK(cudaMalloc((void **)&d_csrRowPtr, (n + 1) * sizeof(int)));
      CUDA_CHECK(cudaMalloc((void **)&d_csrColInd, nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc((void **)&d_b, n * sizeof(double)));
      CUDA_CHECK(cudaMalloc((void **)&d_x, n * sizeof(double)));

      CUDA_CHECK(cudaMemcpy(d_csrVal, csrVal, nnz * sizeof(double),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_csrRowPtr, csrRowPtr, (n + 1) * sizeof(int),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_csrColInd, csrColInd, nnz * sizeof(int),
                            cudaMemcpyHostToDevice));

      return true;
    } else {
      return false;
    }
  }

  bool solve(const Eigen::MatrixX3d &rhs, Eigen::MatrixX3d &sol) {
    if (solver_type_ == Parameters::LDLT) {

      int n_cols = rhs.cols();

      for (int i = 0; i < n_cols; ++i) {
        const double *b_data = rhs.col(i).data();

        CUDA_CHECK(cudaMemcpy(d_b, b_data, n * sizeof(double),
                              cudaMemcpyHostToDevice));

        solveUsingCusolver(n, nnz, d_csrRowPtr, d_csrColInd, d_csrVal, d_b,
                           d_x);

        CUDA_CHECK(cudaMemcpy(sol.col(i).data(), d_x, n * sizeof(double),
                              cudaMemcpyDeviceToHost));

        return true;
      }

    } else {
      return false;
    }
  }

  void reset_pattern() {
    // CUDA_CHECK(cudaFree(d_csrVal));
    // CUDA_CHECK(cudaFree(d_csrRowPtr));
    // CUDA_CHECK(cudaFree(d_csrColInd));
    // CUDA_CHECK(cudaFree(d_b));
    // CUDA_CHECK(cudaFree(d_x));
  }

  void set_solver_type(Parameters::LinearSolverType type) {
    solver_type_ = type;
    if (solver_type_ == Parameters::LDLT) {
      reset_pattern();
    }
  }

private:
  Parameters::LinearSolverType solver_type_;
  int n, nnz;
  double *d_csrVal, *d_b, *d_x;
  int *d_csrRowPtr, *d_csrColInd;
};

class LinearSolverCPU {
public:
  LinearSolverCPU(Parameters::LinearSolverType solver_type)
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

  bool solve(const Eigen::MatrixX3d &rhs, Eigen::MatrixX3d &sol) {
    if (solver_type_ == Parameters::LDLT) {
      int n_cols = rhs.cols();

      OMP_PARALLEL {
        OMP_FOR
        for (int i = 0; i < n_cols; ++i) {
          sol.col(i) = LDLT_solver_.solve(rhs.col(i));
        }
      }

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
} // namespace SDFilter