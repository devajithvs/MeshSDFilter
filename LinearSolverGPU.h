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

struct PreConjugateState {
  cublasHandle_t cublasHandle;
  cusparseHandle_t cusparseHandle;
  cusolverSpHandle_t cusolverHandle;

  csrcholInfo_t choleskyInfo;

  cusparseMatDescr_t descr = 0;
  cusparseSpMatDescr_t matA = NULL;
  cusparseSpMatDescr_t matM_upper;
  cusparseSpMatDescr_t matM_lower;

  double *d_p, *d_y, *d_zm1, *d_zm2, *d_rm2, *d_omega, *d_valsILU0;

  size_t bufferSizeLU = 0;
  // int bufferSizeLU = 0;
  size_t bufferSizeMV, bufferSizeL, bufferSizeU;
  void *d_bufferLU, *d_bufferMV, *d_bufferL, *d_bufferU;
  cusparseSpSVDescr_t spsvDescrL, spsvDescrU;
  cusparseMatDescr_t matLU;
  csrilu02Info_t infoILU = NULL;

  // Create and initialize factorization (ILU(0)) if needed
  cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
  cusparseDiagType_t diag_unit = CUSPARSE_DIAG_TYPE_UNIT;
  cusparseFillMode_t fill_upper = CUSPARSE_FILL_MODE_UPPER;
  cusparseDiagType_t diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;

  double doubleone = 1.0, doublezero = 0.0;

  /* Wrap raw data into cuSPARSE generic API objects */
  cusparseDnVecDescr_t vecp = NULL, vecY = NULL, vecZM1 = NULL;
  cusparseDnVecDescr_t vecomega = NULL;

  /* Wrap raw data into cuSPARSE generic API objects */
  cusparseDnVecDescr_t vecX = NULL, vecR = NULL;
};

// Function to initialize ConjugateState and perform factorization
void initializePreConjugateState(int N, int nz, int *d_csrRowPtr,
                                 int *d_csrColInd, double *d_csrVal,
                                 PreConjugateState &state) {

  cublasCreate(&state.cublasHandle);
  cusparseCreate(&state.cusparseHandle);

  /* Description of the A matrix */
  cusparseCreateMatDescr(&state.descr);
  cusparseSetMatType(state.descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(state.descr, CUSPARSE_INDEX_BASE_ZERO);

  cusparseCreateCsr(&state.matA, N, N, nz, d_csrRowPtr, d_csrColInd, d_csrVal,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

  //     /* Allocate required memory */
  cudaMalloc((void **)&state.d_y, N * sizeof(double));
  cudaMalloc((void **)&state.d_p, N * sizeof(double));
  cudaMalloc((void **)&state.d_omega, N * sizeof(double));
  cudaMalloc((void **)&state.d_valsILU0, nz * sizeof(double));
  cudaMalloc((void **)&state.d_zm1, (N) * sizeof(double));
  cudaMalloc((void **)&state.d_zm2, (N) * sizeof(double));
  cudaMalloc((void **)&state.d_rm2, (N) * sizeof(double));

  cusparseCreateDnVec(&state.vecp, N, state.d_p, CUDA_R_64F);
  cusparseCreateDnVec(&state.vecY, N, state.d_y, CUDA_R_64F);
  cusparseCreateDnVec(&state.vecZM1, N, state.d_zm1, CUDA_R_64F);

  cusparseCreateDnVec(&state.vecomega, N, state.d_omega, CUDA_R_64F);

  /* Create ILU(0) info object */
  cusparseCreateCsrilu02Info(&state.infoILU);
  cusparseCreateMatDescr(&state.matLU);
  cusparseSetMatType(state.matLU, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(state.matLU, CUSPARSE_INDEX_BASE_ZERO);

  /* Allocate workspace for cuSPARSE */
  cusparseSpMV_bufferSize(
      state.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &state.doubleone,
      state.matA, state.vecp, &state.doublezero, state.vecomega, CUDA_R_64F,
      CUSPARSE_SPMV_ALG_DEFAULT, &state.bufferSizeMV);
  cudaMalloc(&state.d_bufferMV, state.bufferSizeMV);

  // cusparseDcsrilu02_bufferSize(state.cusparseHandle, N, nz, state.matLU,
  //                              d_csrVal, d_csrRowPtr, d_csrColInd,
  //                              state.infoILU, &state.bufferSizeLU);
  // cudaMalloc(&state.d_bufferLU, state.bufferSizeLU);

  cudaMemcpy(state.d_valsILU0, d_csrVal, nz * sizeof(double),
             cudaMemcpyDeviceToDevice);

  // Lower Part
  cusparseCreateCsr(&state.matM_lower, N, N, nz, d_csrRowPtr, d_csrColInd,
                    state.d_valsILU0, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  cusparseSpMatSetAttribute(state.matM_lower, CUSPARSE_SPMAT_FILL_MODE,
                            &state.fill_lower, sizeof(state.fill_lower));
  cusparseSpMatSetAttribute(state.matM_lower, CUSPARSE_SPMAT_DIAG_TYPE,
                            &state.diag_unit, sizeof(state.diag_unit));

  // M_upper
  cusparseCreateCsr(&state.matM_upper, N, N, nz, d_csrRowPtr, d_csrColInd,
                    state.d_valsILU0, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  cusparseSpMatSetAttribute(state.matM_upper, CUSPARSE_SPMAT_FILL_MODE,
                            &state.fill_upper, sizeof(state.fill_upper));
  cusparseSpMatSetAttribute(state.matM_upper, CUSPARSE_SPMAT_DIAG_TYPE,
                            &state.diag_non_unit, sizeof(state.diag_non_unit));
}

void generateILUFactors(PreConjugateState &state, int N, int nz,
                        int *d_csrRowPtr, int *d_csrColInd, double *d_csrVal,
                        double *d_r, double *d_x) {

  cudaMemcpy(state.d_valsILU0, d_csrVal, nz * sizeof(double),
             cudaMemcpyDeviceToDevice);

  // /* Perform analysis for ILU(0) */
  // cusparseDcsrilu02_analysis(state.cusparseHandle, N, nz, state.descr,
  //                            state.d_valsILU0, d_csrRowPtr, d_csrColInd,
  //                            state.infoILU, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
  //                            state.d_bufferLU);

  // /* generate the ILU(0) factors */
  // cusparseDcsrilu02(state.cusparseHandle, N, nz, state.matLU,
  // state.d_valsILU0,
  //                   d_csrRowPtr, d_csrColInd, state.infoILU,
  //                   CUSPARSE_SOLVE_POLICY_USE_LEVEL, state.d_bufferLU);

  cusolverSpXcsrcholAnalysis(state.cusolverHandle, N, nz, state.matLU,
                             d_csrRowPtr, d_csrColInd, state.choleskyInfo);
  // Compute workspace size
  size_t size_internal = 0;
  cusolverSpDcsrcholBufferInfo(
      state.cusolverHandle, N, nz, state.matLU, state.d_valsILU0, d_csrRowPtr,
      d_csrColInd, state.choleskyInfo, &size_internal, &state.bufferSizeLU);

  // Allocate workspace on GPU
  cudaMalloc(&state.d_bufferLU, state.bufferSizeLU);
  cusolverSpDcsrcholFactor(state.cusolverHandle, N, nz, state.matLU,
                           state.d_valsILU0, d_csrRowPtr, d_csrColInd,
                           state.choleskyInfo, &state.bufferSizeLU);

  cusparseCreateDnVec(&state.vecX, N, d_x, CUDA_R_64F);
  cusparseCreateDnVec(&state.vecR, N, d_r, CUDA_R_64F);

  /* Allocate workspace for cuSPARSE */
  cusparseSpSV_createDescr(&state.spsvDescrL);
  cusparseSpSV_bufferSize(
      state.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &state.doubleone,
      state.matM_lower, state.vecR, state.vecX, CUDA_R_64F,
      CUSPARSE_SPSV_ALG_DEFAULT, state.spsvDescrL, &state.bufferSizeL);
  cudaMalloc(&state.d_bufferL, state.bufferSizeL);

  cusparseSpSV_createDescr(&state.spsvDescrU);
  cusparseSpSV_bufferSize(
      state.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &state.doubleone,
      state.matM_upper, state.vecR, state.vecX, CUDA_R_64F,
      CUSPARSE_SPSV_ALG_DEFAULT, state.spsvDescrU, &state.bufferSizeU);
  cudaMalloc(&state.d_bufferU, state.bufferSizeU);

  /* perform triangular solve analysis */
  cusparseSpSV_analysis(state.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &state.doubleone, state.matM_lower, state.vecR,
                        state.vecX, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
                        state.spsvDescrL, state.d_bufferL);

  cusparseSpSV_analysis(state.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &state.doubleone, state.matM_upper, state.vecR,
                        state.vecX, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
                        state.spsvDescrU, state.d_bufferU);
}

void solveUsingPreconditionedConjugateGradient(
    PreConjugateState &state, int N, int nz, int *d_csrRowPtr, int *d_csrColInd,
    double *d_csrVal, double *d_r, double *d_x, const double tol = 1e-5f,
    const int max_iter = 10000) {

  cublasHandle_t cublasHandle;
  cusparseHandle_t cusparseHandle;

  cublasCreate(&cublasHandle);
  cublasStatus_t blasStatus;
  cusparseCreate(&cusparseHandle);

  double alpha, beta, numerator, denominator, nalpha;
  double r1;

  int k;
  double dot;

  k = 0;
  cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
  // preconditioner application: state.d_zm1 = U^-1 L^-1 d_r
  while (r1 > tol * tol && k <= max_iter) {
    cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                       &state.doubleone, state.matM_lower, state.vecR,
                       state.vecY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
                       state.spsvDescrL);

    cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                       &state.doubleone, state.matM_upper, state.vecY,
                       state.vecZM1, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
                       state.spsvDescrU);
    k++;

    if (k == 1) {
      cublasDcopy(cublasHandle, N, state.d_zm1, 1, state.d_p, 1);
    } else {
      cublasDdot(cublasHandle, N, d_r, 1, state.d_zm1, 1, &numerator);
      cublasDdot(cublasHandle, N, state.d_rm2, 1, state.d_zm2, 1, &denominator);
      beta = numerator / denominator;
      cublasDscal(cublasHandle, N, &beta, state.d_p, 1);
      cublasDaxpy(cublasHandle, N, &state.doubleone, state.d_zm1, 1, state.d_p,
                  1);
    }

    cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &state.doubleone, state.matA, state.vecp, &state.doublezero,
                 state.vecomega, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                 state.d_bufferMV);
    cublasDdot(cublasHandle, N, d_r, 1, state.d_zm1, 1, &numerator);
    cublasDdot(cublasHandle, N, state.d_p, 1, state.d_omega, 1, &denominator);
    alpha = numerator / denominator;
    cublasDaxpy(cublasHandle, N, &alpha, state.d_p, 1, d_x, 1);
    cublasDcopy(cublasHandle, N, d_r, 1, state.d_rm2, 1);
    cublasDcopy(cublasHandle, N, state.d_zm1, 1, state.d_zm2, 1);
    nalpha = -alpha;
    cublasDaxpy(cublasHandle, N, &nalpha, state.d_omega, 1, d_r, 1);
    cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
  }
  printf("  iteration = %3d, residual = %e \n", k, sqrt(r1));

  // if (state.matA) {
  //   cusparseDestroySpMat(state.matA);
  // }
  // if (state.vecp) {
  //   cusparseDestroyDnVec(state.vecp);
  // }
  // if (state.vecX) {
  //   cusparseDestroyDnVec(state.vecX);
  // }
  // if (state.vecY) {
  //   cusparseDestroyDnVec(state.vecY);
  // }
  // if (state.vecR) {
  //   cusparseDestroyDnVec(state.vecR);
  // }
  // if (state.vecZM1) {
  //   cusparseDestroyDnVec(state.vecZM1);
  // }
  // if (state.vecomega) {
  //   cusparseDestroyDnVec(state.vecomega);
  // }

  // cudaFree(state.d_y);
  // cudaFree(state.d_p);
  // cudaFree(state.d_omega);
  // cudaFree(state.d_valsILU0);
  // cudaFree(state.d_zm1);
  // cudaFree(state.d_zm2);
  // cudaFree(state.d_rm2);

  cusparseDestroy(cusparseHandle);
  cublasDestroy(cublasHandle);
}

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

  double *d_p, *d_y;

  cudaMalloc((void **)&d_p, N * sizeof(double));
  cudaMalloc((void **)&d_y, N * sizeof(double));

  /* Wrap raw data into cuSPARSE generic API objects */
  cusparseSpMatDescr_t matA = NULL;
  cusparseCreateCsr(&matA, N, N, nz, d_csrRowPtr, d_csrColInd, d_csrVal,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  cusparseDnVecDescr_t vecx = NULL;
  cusparseCreateDnVec(&vecx, N, d_x, CUDA_R_64F);
  cusparseDnVecDescr_t vecp = NULL;
  cusparseCreateDnVec(&vecp, N, d_p, CUDA_R_64F);
  cusparseDnVecDescr_t vecy = NULL;
  cusparseCreateDnVec(&vecy, N, d_y, CUDA_R_64F);

  /* Allocate workspace for cuSPARSE */
  size_t bufferSize = 0;
  cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &alpha, matA, vecx, &beta, vecy, CUDA_R_64F,
                          CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
  void *buffer = NULL;
  cudaMalloc(&buffer, bufferSize);

  // Start CG algorithm
  cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
               vecx, &beta, vecy, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
               buffer);

  cublasDaxpy(cublasHandle, N, &alpham1, d_y, 1, d_b, 1);
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
                 vecp, &beta, vecy, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                 buffer);
    blasStatus = cublasDdot(cublasHandle, N, d_p, 1, d_y, 1, &dot);
    a = r1 / dot;

    blasStatus = cublasDaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
    na = -a;
    blasStatus = cublasDaxpy(cublasHandle, N, &na, d_y, 1, d_b, 1);

    r0 = r1;
    blasStatus = cublasDdot(cublasHandle, N, d_b, 1, d_b, 1, &r1);
    cudaDeviceSynchronize();
    k++;
  }
  printf("iteration = %3d, residual = %e\n", k, sqrt(r1));

  if (matA) {
    cusparseDestroySpMat(matA);
  }
  if (vecx) {
    cusparseDestroyDnVec(vecx);
  }
  if (vecy) {
    cusparseDestroyDnVec(vecy);
  }
  if (vecp) {
    cusparseDestroyDnVec(vecp);
  }

  // Clean up
  cusparseDestroy(cusparseHandle);
  cublasDestroy(cublasHandle);
  cudaFree(d_p);
  cudaFree(d_y);
}

struct CuSolverState {
  cusolverSpHandle_t handle;
  cudaStream_t stream;
  cusparseMatDescr_t descrA;
  csrcholInfo_t info;
  cusparseHandle_t cusparseH;
  void *workspace = nullptr;

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
void factorize(CuSolverState &state, size_t &size_chol, const int n,
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
  cudaMalloc(&state.workspace, size_chol);

  // Factorize the matrix
  cusolverSpDcsrcholFactor(state.handle, n, nnz, state.descrA, d_csrVal,
                           d_csrRowPtr, d_csrColInd, state.info,
                           state.workspace);
}

// Solve the linear system using cuSOLVER
void solveLDLT(CuSolverState &state, const int n, const double *d_b,
               double *d_x) {
  // Solve the linear system
  cusolverSpDcsrcholSolve(state.handle, n, d_b, d_x, state.info,
                          state.workspace);
}

void solveLDLT2(CuSolverState &state, size_t &size_chol, const int n,
                const int nnz, const int *d_csrRowPtr, const int *d_csrColInd,
                const double *d_csrVal, const double *d_b, double *d_x) {
  // Analyze Cholesky structure
  cusolverSpXcsrcholAnalysis(state.handle, n, nnz, state.descrA, d_csrRowPtr,
                             d_csrColInd, state.info);
  // Compute workspace size
  size_t size_internal = 0;
  cusolverSpDcsrcholBufferInfo(state.handle, n, nnz, state.descrA, d_csrVal,
                               d_csrRowPtr, d_csrColInd, state.info,
                               &size_internal, &size_chol);

  // Allocate workspace on GPU
  if (state.workspace) {
    cudaFree(state.workspace);
  }
  cudaMalloc(&state.workspace, size_chol);

  // Factorize the matrix
  cusolverSpDcsrcholFactor(state.handle, n, nnz, state.descrA, d_csrVal,
                           d_csrRowPtr, d_csrColInd, state.info,
                           state.workspace);
  cusolverSpDcsrcholSolve(state.handle, n, d_b, d_x, state.info,
                          state.workspace);
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

      size_t size_chol = 0;
      factorize(solverState, size_chol, n, nnz, d_csrRowPtr, d_csrColInd,
                d_csrVal);
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
    } else if (solver_type_ == Parameters::PRECG) {

      n = M.rows();
      nnz = M.nonZeros();

      CUDA_CHECK(cudaMalloc((void **)&d_csrVal, nnz * sizeof(double)));
      CUDA_CHECK(cudaMalloc((void **)&d_csrRowPtr, (n + 1) * sizeof(int)));
      CUDA_CHECK(cudaMalloc((void **)&d_csrColInd, nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc((void **)&d_b, n * sizeof(double)));
      CUDA_CHECK(cudaMalloc((void **)&d_x, n * sizeof(double)));
      cudaMemset(d_x, 0, n * sizeof(double));

      // Initialize the ConjugateState struct and perform factorization
      initializePreConjugateState(n, nnz, d_csrRowPtr, d_csrColInd, d_csrVal,
                                  preConjugatestate);

      CUDA_CHECK(cudaMemcpy(d_csrVal, M.valuePtr(), nnz * sizeof(double),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_csrRowPtr, M.outerIndexPtr(),
                            (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_csrColInd, M.innerIndexPtr(), nnz * sizeof(int),
                            cudaMemcpyHostToDevice));

      generateILUFactors(preConjugatestate, n, nnz, d_csrRowPtr, d_csrColInd,
                         d_csrVal, d_b, d_x);

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

        CUDA_CHECK(cudaMemcpyAsync(d_b, b_data, n * sizeof(double),
                                   cudaMemcpyHostToDevice));

        solveLDLT(solverState, n, d_b, d_x);
        // solveUsingCusolver(n, nnz, d_csrRowPtr, d_csrColInd, d_csrVal, d_b,
        // d_x);

        CUDA_CHECK(cudaMemcpyAsync(sol.col(i).data(), d_x, n * sizeof(double),
                                   cudaMemcpyDeviceToHost));
      }

      return true;
    } else if (solver_type_ == Parameters::CG) {

      int n_cols = rhs.cols();

      for (int i = 0; i < n_cols; ++i) {
        CUDA_CHECK(cudaMemcpy(d_x, sol.col(i).data(), n * sizeof(double),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(d_b, rhs.col(i).data(), n * sizeof(double),
                              cudaMemcpyHostToDevice));

        solveUsingConjugateGradient(n, nnz, d_csrRowPtr, d_csrColInd, d_csrVal,
                                    d_b, d_x);

        CUDA_CHECK(cudaMemcpy(sol.col(i).data(), d_x, n * sizeof(double),
                              cudaMemcpyDeviceToHost));
      }
      return true;

    } else if (solver_type_ == Parameters::PRECG) {

      int n_cols = rhs.cols();

      for (int i = 0; i < n_cols; ++i) {
        cudaMemset(d_x, 0, n * sizeof(double));

        CUDA_CHECK(cudaMemcpy(d_b, rhs.col(i).data(), n * sizeof(double),
                              cudaMemcpyHostToDevice));

        // Call the PCG solver function with the precomputed factorization
        // generateILUFactors(preConjugatestate, n, nnz, d_csrRowPtr,
        // d_csrColInd,
        //                    d_csrVal, d_b, d_x);

        solveUsingPreconditionedConjugateGradient(preConjugatestate, n, nnz,
                                                  d_csrRowPtr, d_csrColInd,
                                                  d_csrVal, d_b, d_x);

        CUDA_CHECK(cudaMemcpy(sol.col(i).data(), d_x, n * sizeof(double),
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
    // cudaFree(state.workspace);
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
  PreConjugateState preConjugatestate;

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