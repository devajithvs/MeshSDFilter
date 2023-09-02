#include "SDFilter.h"

#include "cuda_runtime.h"
#include <cublas_v2.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusparse.h>
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
void solveUsingConjugateGradient(int N, int nz, int *d_csrRowPtr,
                                 int *d_csrColInd, double *d_csrVal,
                                 double *d_b, double *d_x,
                                 const double tol = 1e-5f,
                                 const int max_iter = 10000) {

  cublasHandle_t cublasHandle;
  cusparseHandle_t cusparseHandle;

  cublasCreate(&cublasHandle);
  cublasStatus_t blasStatus;
  cusparseCreate(&cusparseHandle);

  double a, b, na, r1, dot;
  double alpha = 1.0;
  double alpham1 = -1.0;
  double beta = 0.0;

  double r0 = 0.0;

  double *d_p, *d_Ax;

  cudaMalloc((void **)&d_p, N * sizeof(double));
  cudaMalloc((void **)&d_Ax, N * sizeof(double));

  /* Wrap raw data into cuSPARSE generic API objects */
  cusparseSpMatDescr_t matA = NULL;
  cusparseCreateCsr(&matA, N, N, nz, d_csrRowPtr, d_csrColInd, d_csrVal,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  cusparseDnVecDescr_t vecx = NULL;
  cusparseCreateDnVec(&vecx, N, d_x, CUDA_R_64F);
  cusparseDnVecDescr_t vecp = NULL;
  cusparseCreateDnVec(&vecp, N, d_p, CUDA_R_64F);
  cusparseDnVecDescr_t vecAx = NULL;
  cusparseCreateDnVec(&vecAx, N, d_Ax, CUDA_R_64F);

  /* Allocate workspace for cuSPARSE */
  size_t bufferSize = 0;
  cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &alpha, matA, vecx, &beta, vecAx, CUDA_R_64F,
                          CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
  void *buffer = NULL;
  cudaMalloc(&buffer, bufferSize);

  // Start CG algorithm
  cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
               vecx, &beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
               buffer);

  cublasDaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_b, 1);
  blasStatus = cublasDdot(cublasHandle, N, d_b, 1, d_b, 1, &r1);

  int k = 1;

  while (r1 > tol * tol && k <= max_iter) {
    if (k > 1) {
      b = r1 / r0;
      blasStatus = cublasDscal(cublasHandle, N, &b, d_p, 1);
      blasStatus = cublasDaxpy(cublasHandle, N, &alpha, d_b, 1, d_p, 1);
    } else {
      blasStatus = cublasDcopy(cublasHandle, N, d_b, 1, d_p, 1);
    }

    cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                 vecp, &beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                 buffer);
    blasStatus = cublasDdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
    a = r1 / dot;

    blasStatus = cublasDaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
    na = -a;
    blasStatus = cublasDaxpy(cublasHandle, N, &na, d_Ax, 1, d_b, 1);

    r0 = r1;
    blasStatus = cublasDdot(cublasHandle, N, d_b, 1, d_b, 1, &r1);
    cudaDeviceSynchronize();
    // printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
    k++;
  }

  if (matA) {
    cusparseDestroySpMat(matA);
  }
  if (vecx) {
    cusparseDestroyDnVec(vecx);
  }
  if (vecAx) {
    cusparseDestroyDnVec(vecAx);
  }
  if (vecp) {
    cusparseDestroyDnVec(vecp);
  }

  // Clean up
  cusparseDestroy(cusparseHandle);
  cublasDestroy(cublasHandle);
  cudaFree(d_p);
  cudaFree(d_Ax);
}

struct CuSolverState {
  cusolverSpHandle_t handle;
  cudaStream_t stream;
  cusparseMatDescr_t descrA;
  csrcholInfo_t info;
  cusparseHandle_t cusparseH;

  CuSolverState() : stream(nullptr), cusparseH(nullptr) {}
};

// Initialize cuSOLVER state
void initializeCuSolverState(CuSolverState &state) {
  CUDA_CHECK(cudaSetDevice(0)); // Set the CUDA device if necessary
  cusolverSpCreate(&state.handle);

  cusparseCreate(&state.cusparseH);
  cusparseSetStream(state.cusparseH, state.stream);

  cudaStreamCreate(&state.stream);
  cusolverSpSetStream(state.handle, state.stream);

  cusparseCreateMatDescr(&state.descrA);
  cusparseSetMatType(state.descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(state.descrA, CUSPARSE_INDEX_BASE_ZERO);

  cusolverSpCreateCsrcholInfo(&state.info);
}

// Clean up cuSOLVER state
void cleanupCuSolverState(CuSolverState &state) {
  cusparseDestroyMatDescr(state.descrA);
  cusolverSpDestroyCsrcholInfo(state.info);

  cusolverSpDestroy(state.handle);
  cudaStreamDestroy(state.stream);
}

// Factorize the given sparse matrix using cuSOLVER
void factorize(const CuSolverState &state, size_t &size_chol, const int n,
               const int nnz, const int *d_csrRowPtr, const int *d_csrColInd,
               const double *d_csrVal) {
  // Analyze Cholesky structure
  cusolverSpXcsrcholAnalysis(state.handle, n, nnz, state.descrA, d_csrRowPtr,
                             d_csrColInd, state.info);

  // Compute workspace size
  size_t size_internal = 0;
  cusolverSpDcsrcholBufferInfo(state.handle, n, nnz, state.descrA, d_csrVal,
                               d_csrRowPtr, d_csrColInd, state.info,
                               &size_internal, &size_chol);

  // Allocate workspace on GPU
  void *d_workspace = nullptr;
  cudaMalloc(&d_workspace, size_chol);

  // Factorize the matrix
  cusolverSpDcsrcholFactor(state.handle, n, nnz, state.descrA, d_csrVal,
                           d_csrRowPtr, d_csrColInd, state.info, d_workspace);

  // Clean up
  cudaFree(d_workspace);
}

// Solve the linear system using cuSOLVER
void solveLDLT(const CuSolverState &state, void *d_workspace, const int n,
               const double *d_b, double *d_x) {
  // Solve the linear system
  cusolverSpDcsrcholSolve(state.handle, n, d_b, d_x, state.info, d_workspace);
}

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
  bool compute(const SparseMatrixXd &M) {
    if (solver_type_ == Parameters::LDLT) {
      n = M.rows();
      nnz = M.nonZeros();

      CUDA_CHECK(cudaMalloc((void **)&d_csrVal, nnz * sizeof(double)));
      CUDA_CHECK(cudaMalloc((void **)&d_csrRowPtr, (n + 1) * sizeof(int)));
      CUDA_CHECK(cudaMalloc((void **)&d_csrColInd, nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc((void **)&d_b, n * sizeof(double)));
      CUDA_CHECK(cudaMalloc((void **)&d_x, n * sizeof(double)));

      CUDA_CHECK(cudaMemcpy(d_csrVal, M.valuePtr(), nnz * sizeof(double),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_csrRowPtr, M.outerIndexPtr(),
                            (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_csrColInd, M.innerIndexPtr(), nnz * sizeof(int),
                            cudaMemcpyHostToDevice));

      initializeCuSolverState(solverState);

      // Create and initialize the input data (d_csrRowPtr, d_csrColInd,
      // d_csrVal, d_b) Factorize the matrix
      size_t size_chol = 0;
      factorize(solverState, size_chol, n, nnz, d_csrRowPtr, d_csrColInd,
                d_csrVal);
      cudaMalloc(&d_workspace, size_chol);
      return true;
    } else if (solver_type_ == Parameters::CG) {

      n = M.rows();
      nnz = M.nonZeros();

      CUDA_CHECK(cudaMalloc((void **)&d_csrVal, nnz * sizeof(double)));
      CUDA_CHECK(cudaMalloc((void **)&d_csrRowPtr, (n + 1) * sizeof(int)));
      CUDA_CHECK(cudaMalloc((void **)&d_csrColInd, nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc((void **)&d_b, n * sizeof(double)));
      CUDA_CHECK(cudaMalloc((void **)&d_x, n * sizeof(double)));

      CUDA_CHECK(cudaMemcpy(d_csrVal, M.valuePtr(), nnz * sizeof(double),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_csrRowPtr, M.outerIndexPtr(),
                            (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_csrColInd, M.innerIndexPtr(), nnz * sizeof(int),
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

        solveLDLT(solverState, d_workspace, n, d_b, d_x);

        CUDA_CHECK(cudaMemcpy(sol.col(i).data(), d_x, n * sizeof(double),
                              cudaMemcpyDeviceToHost));
      }

      return true;
    } else if (solver_type_ == Parameters::CG) {

      int n_cols = rhs.cols();

      for (int i = 0; i < n_cols; ++i) {
        const double *b_data = rhs.col(i).data();

        CUDA_CHECK(cudaMemcpyAsync(d_x, sol.col(i).data(), n * sizeof(double),
                                   cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpyAsync(d_b, b_data, n * sizeof(double),
                                   cudaMemcpyHostToDevice));

        solveUsingConjugateGradient(n, nnz, d_csrRowPtr, d_csrColInd, d_csrVal,
                                    d_b, d_x);

        CUDA_CHECK(cudaMemcpyAsync(sol.col(i).data(), d_x, n * sizeof(double),
                                   cudaMemcpyDeviceToHost));
      }
      return true;

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
    // cudaFree(d_workspace);
    // cleanupCuSolverState(solverState);
  }

  void set_solver_type(Parameters::LinearSolverType type) {
    solver_type_ = type;
    if (solver_type_ == Parameters::LDLT) {
      reset_pattern();
    }
  }

private:
  CuSolverState solverState;
  void *d_workspace = nullptr;

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