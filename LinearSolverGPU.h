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

  cusparseMatDescr_t descr = 0;
  cusparseSpMatDescr_t matA = NULL;
  cusparseSpMatDescr_t matM_upper;
  cusparseSpMatDescr_t matM_lower;

  float *d_p, *d_y, *d_zm1, *d_zm2, *d_rm2, *d_omega, *d_valsILU0;

  int bufferSizeLU = 0;
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

  float floatone = 1.0, floatzero = 0.0;

  /* Wrap raw data into cuSPARSE generic API objects */
  cusparseDnVecDescr_t vecp = NULL, vecY = NULL, vecZM1 = NULL;
  cusparseDnVecDescr_t vecomega = NULL;

  /* Wrap raw data into cuSPARSE generic API objects */
  cusparseDnVecDescr_t vecX = NULL, vecR = NULL;
};

// Function to initialize ConjugateState and perform factorization
void initializePreConjugateState(int N, int nz, int *d_csrRowPtr,
                                 int *d_csrColInd, float *d_csrVal,
                                 PreConjugateState &state) {

  cublasCreate(&state.cublasHandle);
  cusparseCreate(&state.cusparseHandle);

  /* Description of the A matrix */
  cusparseCreateMatDescr(&state.descr);
  cusparseSetMatType(state.descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(state.descr, CUSPARSE_INDEX_BASE_ZERO);

  cusparseCreateCsr(&state.matA, N, N, nz, d_csrRowPtr, d_csrColInd, d_csrVal,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  //     /* Allocate required memory */
  cudaMalloc((void **)&state.d_y, N * sizeof(float));
  cudaMalloc((void **)&state.d_p, N * sizeof(float));
  cudaMalloc((void **)&state.d_omega, N * sizeof(float));
  cudaMalloc((void **)&state.d_valsILU0, nz * sizeof(float));
  cudaMalloc((void **)&state.d_zm1, (N) * sizeof(float));
  cudaMalloc((void **)&state.d_zm2, (N) * sizeof(float));
  cudaMalloc((void **)&state.d_rm2, (N) * sizeof(float));

  cusparseCreateDnVec(&state.vecp, N, state.d_p, CUDA_R_32F);
  cusparseCreateDnVec(&state.vecY, N, state.d_y, CUDA_R_32F);
  cusparseCreateDnVec(&state.vecZM1, N, state.d_zm1, CUDA_R_32F);

  cusparseCreateDnVec(&state.vecomega, N, state.d_omega, CUDA_R_32F);

  /* Create ILU(0) info object */
  cusparseCreateCsrilu02Info(&state.infoILU);
  cusparseCreateMatDescr(&state.matLU);
  cusparseSetMatType(state.matLU, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(state.matLU, CUSPARSE_INDEX_BASE_ZERO);

  /* Allocate workspace for cuSPARSE */
  cusparseSpMV_bufferSize(
      state.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &state.floatone,
      state.matA, state.vecp, &state.floatzero, state.vecomega, CUDA_R_32F,
      CUSPARSE_SPMV_ALG_DEFAULT, &state.bufferSizeMV);
  cudaMalloc(&state.d_bufferMV, state.bufferSizeMV);

  cusparseScsrilu02_bufferSize(state.cusparseHandle, N, nz, state.matLU,
                               d_csrVal, d_csrRowPtr, d_csrColInd,
                               state.infoILU, &state.bufferSizeLU);
  cudaMalloc(&state.d_bufferLU, state.bufferSizeLU);

  cudaMemcpy(state.d_valsILU0, d_csrVal, nz * sizeof(float),
             cudaMemcpyDeviceToDevice);

  // Lower Part
  cusparseCreateCsr(&state.matM_lower, N, N, nz, d_csrRowPtr, d_csrColInd,
                    state.d_valsILU0, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseSpMatSetAttribute(state.matM_lower, CUSPARSE_SPMAT_FILL_MODE,
                            &state.fill_lower, sizeof(state.fill_lower));
  cusparseSpMatSetAttribute(state.matM_lower, CUSPARSE_SPMAT_DIAG_TYPE,
                            &state.diag_unit, sizeof(state.diag_unit));

  // M_upper
  cusparseCreateCsr(&state.matM_upper, N, N, nz, d_csrRowPtr, d_csrColInd,
                    state.d_valsILU0, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseSpMatSetAttribute(state.matM_upper, CUSPARSE_SPMAT_FILL_MODE,
                            &state.fill_upper, sizeof(state.fill_upper));
  cusparseSpMatSetAttribute(state.matM_upper, CUSPARSE_SPMAT_DIAG_TYPE,
                            &state.diag_non_unit, sizeof(state.diag_non_unit));
}

void generateILUFactors(PreConjugateState &state, int N, int nz,
                        int *d_csrRowPtr, int *d_csrColInd, float *d_csrVal,
                        float *d_r, float *d_x) {

  cudaMemcpy(state.d_valsILU0, d_csrVal, nz * sizeof(float),
             cudaMemcpyDeviceToDevice);

  /* Perform analysis for ILU(0) */
  cusparseScsrilu02_analysis(state.cusparseHandle, N, nz, state.descr,
                             state.d_valsILU0, d_csrRowPtr, d_csrColInd,
                             state.infoILU, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                             state.d_bufferLU);

  /* generate the ILU(0) factors */
  cusparseScsrilu02(state.cusparseHandle, N, nz, state.matLU, state.d_valsILU0,
                    d_csrRowPtr, d_csrColInd, state.infoILU,
                    CUSPARSE_SOLVE_POLICY_USE_LEVEL, state.d_bufferLU);

  cusparseCreateDnVec(&state.vecX, N, d_x, CUDA_R_32F);
  cusparseCreateDnVec(&state.vecR, N, d_r, CUDA_R_32F);

  /* Allocate workspace for cuSPARSE */
  cusparseSpSV_createDescr(&state.spsvDescrL);
  cusparseSpSV_bufferSize(
      state.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &state.floatone,
      state.matM_lower, state.vecR, state.vecX, CUDA_R_32F,
      CUSPARSE_SPSV_ALG_DEFAULT, state.spsvDescrL, &state.bufferSizeL);
  cudaMalloc(&state.d_bufferL, state.bufferSizeL);

  cusparseSpSV_createDescr(&state.spsvDescrU);
  cusparseSpSV_bufferSize(
      state.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &state.floatone,
      state.matM_upper, state.vecR, state.vecX, CUDA_R_32F,
      CUSPARSE_SPSV_ALG_DEFAULT, state.spsvDescrU, &state.bufferSizeU);
  cudaMalloc(&state.d_bufferU, state.bufferSizeU);

  /* perform triangular solve analysis */
  cusparseSpSV_analysis(state.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &state.floatone, state.matM_lower, state.vecR,
                        state.vecX, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT,
                        state.spsvDescrL, state.d_bufferL);

  cusparseSpSV_analysis(state.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &state.floatone, state.matM_upper, state.vecR,
                        state.vecX, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT,
                        state.spsvDescrU, state.d_bufferU);
}

void solveUsingPreconditionedConjugateGradient(
    PreConjugateState &state, int N, int nz, int *d_csrRowPtr, int *d_csrColInd,
    float *d_csrVal, float *d_r, float *d_x, const float tol = 1e-5f,
    const int max_iter = 10000) {

  cublasHandle_t cublasHandle;
  cusparseHandle_t cusparseHandle;

  cublasCreate(&cublasHandle);
  cublasStatus_t blasStatus;
  cusparseCreate(&cusparseHandle);

  float alpha, beta, numerator, denominator, nalpha;
  float r1;

  int k;
  float dot;

  k = 0;
  cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
  // preconditioner application: state.d_zm1 = U^-1 L^-1 d_r
  while (r1 > tol * tol && k <= max_iter) {
    cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                       &state.floatone, state.matM_lower, state.vecR,
                       state.vecY, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT,
                       state.spsvDescrL);

    cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                       &state.floatone, state.matM_upper, state.vecY,
                       state.vecZM1, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT,
                       state.spsvDescrU);
    k++;

    if (k == 1) {
      cublasScopy(cublasHandle, N, state.d_zm1, 1, state.d_p, 1);
    } else {
      cublasSdot(cublasHandle, N, d_r, 1, state.d_zm1, 1, &numerator);
      cublasSdot(cublasHandle, N, state.d_rm2, 1, state.d_zm2, 1, &denominator);
      beta = numerator / denominator;
      cublasSscal(cublasHandle, N, &beta, state.d_p, 1);
      cublasSaxpy(cublasHandle, N, &state.floatone, state.d_zm1, 1, state.d_p,
                  1);
    }

    cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &state.floatone, state.matA, state.vecp, &state.floatzero,
                 state.vecomega, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                 state.d_bufferMV);
    cublasSdot(cublasHandle, N, d_r, 1, state.d_zm1, 1, &numerator);
    cublasSdot(cublasHandle, N, state.d_p, 1, state.d_omega, 1, &denominator);
    alpha = numerator / denominator;
    cublasSaxpy(cublasHandle, N, &alpha, state.d_p, 1, d_x, 1);
    cublasScopy(cublasHandle, N, d_r, 1, state.d_rm2, 1);
    cublasScopy(cublasHandle, N, state.d_zm1, 1, state.d_zm2, 1);
    nalpha = -alpha;
    cublasSaxpy(cublasHandle, N, &nalpha, state.d_omega, 1, d_r, 1);
    cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
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

void solveUsingPreconditionedConjugateGradient0(
    int N, int nz, int *d_csrRowPtr, int *d_csrColInd, float *d_csrVal,
    float *d_b, float *d_x, int *d_csrRowPtrL, int *d_csrColIndL,
    float *d_csrValL, const float tol = 1e-5f, const int max_iter = 10000) {

  cublasHandle_t cublasHandle;
  cusparseHandle_t cusparseHandle;

  cublasCreate(&cublasHandle);
  cublasStatus_t blasStatus;
  cusparseCreate(&cusparseHandle);

  float a, b, na, rho, temp;
  float alpha = 1.0;
  float alpham1 = -1.0;
  float beta = 0.0;

  float rhop = 0.0;

  float *d_p, *d_y;

  cudaMalloc((void **)&d_p, N * sizeof(float));
  cudaMalloc((void **)&d_y, N * sizeof(float));

  /* Wrap raw data into cuSPARSE generic API objects */
  cusparseSpMatDescr_t matA = NULL;
  cusparseCreateCsr(&matA, N, N, nz, d_csrRowPtr, d_csrColInd, d_csrVal,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseDnVecDescr_t vecx = NULL;
  cusparseCreateDnVec(&vecx, N, d_x, CUDA_R_32F);
  cusparseDnVecDescr_t vecp = NULL;
  cusparseCreateDnVec(&vecp, N, d_p, CUDA_R_32F);
  cusparseDnVecDescr_t vecy = NULL;
  cusparseCreateDnVec(&vecy, N, d_y, CUDA_R_32F);

  cusparseDnVecDescr_t vecR = NULL;
  cusparseCreateDnVec(&vecR, N, d_b, CUDA_R_32F);

  cusparseSpMatDescr_t matM_upper;
  cusparseSpMatDescr_t matM_lower;

  cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
  cusparseDiagType_t diag_unit = CUSPARSE_DIAG_TYPE_UNIT;
  cusparseFillMode_t fill_upper = CUSPARSE_FILL_MODE_UPPER;
  cusparseDiagType_t diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;

  // Lower Part
  cusparseCreateCsr(&matM_lower, N, N, nz, d_csrRowPtrL, d_csrColIndL,
                    d_csrValL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseSpMatSetAttribute(matM_lower, CUSPARSE_SPMAT_FILL_MODE, &fill_lower,
                            sizeof(fill_lower));
  cusparseSpMatSetAttribute(matM_lower, CUSPARSE_SPMAT_DIAG_TYPE, &diag_unit,
                            sizeof(diag_unit));

  // M_upper
  cusparseCreateCsr(&matM_upper, N, N, nz, d_csrRowPtrL, d_csrColIndL,
                    d_csrValL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseSpMatSetAttribute(matM_upper, CUSPARSE_SPMAT_FILL_MODE, &fill_upper,
                            sizeof(fill_upper));
  cusparseSpMatSetAttribute(matM_upper, CUSPARSE_SPMAT_DIAG_TYPE,
                            &diag_non_unit, sizeof(diag_non_unit));

  void *d_bufferL, *d_bufferU;
  size_t bufferSizeL, bufferSizeU;
  cusparseSpSVDescr_t spsvDescrL, spsvDescrU;

  /* Allocate workspace for cuSPARSE */
  cusparseSpSV_createDescr(&spsvDescrL);
  cusparseSpSV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &alpha, matM_lower, vecR, vecx, CUDA_R_32F,
                          CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL);
  cudaMalloc(&d_bufferL, bufferSizeL);

  cusparseSpSV_createDescr(&spsvDescrU);
  cusparseSpSV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &alpha, matM_upper, vecR, vecx, CUDA_R_32F,
                          CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, &bufferSizeU);
  cudaMalloc(&d_bufferU, bufferSizeU);

  /* perform triangular solve analysis */
  cusparseSpSV_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matM_lower, vecR, vecx, CUDA_R_32F,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, d_bufferL);

  cusparseSpSV_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matM_upper, vecR, vecx, CUDA_R_32F,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, d_bufferU);

  /* Allocate workspace for cuSPARSE */
  size_t bufferSize = 0;
  cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &alpha, matA, vecx, &beta, vecy, CUDA_R_32F,
                          CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
  void *buffer = NULL;
  cudaMalloc(&buffer, bufferSize);

  // Start CG algorithm
  cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
               vecx, &beta, vecy, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
               buffer);

  cublasSaxpy(cublasHandle, N, &alpham1, d_y, 1, d_b, 1);
  blasStatus = cublasSdot(cublasHandle, N, d_b, 1, d_b, 1, &rho);

  int k = 0;
  cusparseDnVecDescr_t vecZ = NULL;
  float *d_z;
  cudaMalloc((void **)&d_z, (N) * sizeof(float));
  cusparseCreateDnVec(&vecZ, N, d_z, CUDA_R_32F);

  cusparseDnVecDescr_t vect = NULL;
  float *d_t;
  cudaMalloc((void **)&d_t, (N) * sizeof(float));
  cusparseCreateDnVec(&vect, N, d_t, CUDA_R_32F);

  while (rho > tol * tol && k < max_iter) {

    cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                       matM_lower, vecR, vect, CUDA_R_32F,
                       CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL);

    cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                       matM_upper, vect, vecZ, CUDA_R_32F,
                       CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU);

    rhop = rho;
    blasStatus = cublasSdot(cublasHandle, N, d_b, 1, d_z, 1, &rho);

    if (k == 0) {
      blasStatus = cublasScopy(cublasHandle, N, d_z, 1, d_p, 1);
    } else {
      b = rho / rhop;
      blasStatus = cublasSscal(cublasHandle, N, &b, d_p, 1);
      blasStatus = cublasSaxpy(cublasHandle, N, &alpha, d_z, 1, d_p, 1);
      cublasScopy(cublasHandle, N, d_z, 1, d_p, 1);
    }

    cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                 vecp, &beta, vecy, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                 buffer);
    blasStatus = cublasSdot(cublasHandle, N, d_p, 1, d_y, 1, &temp);
    a = rho / temp;

    blasStatus = cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
    na = -a;
    blasStatus = cublasSaxpy(cublasHandle, N, &na, d_y, 1, d_b, 1);

    cudaDeviceSynchronize();
    k++;
  }
  printf("iteration = %3d, residual = %e\n", k, sqrt(rho));

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

struct ConjugateGradientState {
  cublasHandle_t cublasHandle;
  cusparseHandle_t cusparseHandle;

  float *d_p, *d_y, *d_x;

  cusparseSpMatDescr_t matA = NULL;
  cusparseDnVecDescr_t vecx = NULL;
  cusparseDnVecDescr_t vecp = NULL;
  cusparseDnVecDescr_t vecy = NULL;
};

// Function to initialize ConjugateGradientState
void initializeConjugateGradientState(int N, int nz, int *d_csrRowPtr,
                                      int *d_csrColInd, float *d_csrVal,
                                      float *d_x,
                                      ConjugateGradientState &state) {
  cublasCreate(&state.cublasHandle);
  cusparseCreate(&state.cusparseHandle);

  state.d_x = d_x;

  cudaMalloc((void **)&state.d_p, N * sizeof(float));
  cudaMalloc((void **)&state.d_y, N * sizeof(float));

  /* Wrap raw data into cuSPARSE generic API objects */
  cusparseCreateCsr(&state.matA, N, N, nz, d_csrRowPtr, d_csrColInd, d_csrVal,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  cusparseCreateDnVec(&state.vecx, N, state.d_x, CUDA_R_32F);

  cusparseCreateDnVec(&state.vecp, N, state.d_p, CUDA_R_32F);

  cusparseCreateDnVec(&state.vecy, N, state.d_y, CUDA_R_32F);
}

void destroyConjugateGradientState(ConjugateGradientState &state) {
  cusparseDestroy(state.cusparseHandle);
  cublasDestroy(state.cublasHandle);
  cudaFree(state.d_p);
  cudaFree(state.d_y);

  if (state.matA) {
    cusparseDestroySpMat(state.matA);
  }
  if (state.vecx) {
    cusparseDestroyDnVec(state.vecx);
  }
  if (state.vecy) {
    cusparseDestroyDnVec(state.vecy);
  }
  if (state.vecp) {
    cusparseDestroyDnVec(state.vecp);
  }
}

void solveUsingConjugateGradient(ConjugateGradientState &state, int N, int nz,
                                 int *d_csrRowPtr, int *d_csrColInd,
                                 float *d_csrVal, float *d_b, float *d_x,
                                 const float tol = 1e-5f,
                                 const int max_iter = 10000) {
  cublasStatus_t blasStatus;

  float a, b, na, r1, dot;
  float alpha = 1.0f;
  float alpham1 = -1.0f;
  float beta = 0.0f;

  float r0 = 0.0f;

  /* Allocate workspace for cuSPARSE */
  size_t bufferSize = 0;
  cusparseSpMV_bufferSize(state.cusparseHandle,
                          CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, state.matA,
                          state.vecx, &beta, state.vecy, CUDA_R_32F,
                          CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
  void *buffer = NULL;
  cudaMalloc(&buffer, bufferSize);

  // Start CG algorithm
  cusparseSpMV(state.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
               state.matA, state.vecx, &beta, state.vecy, CUDA_R_32F,
               CUSPARSE_SPMV_ALG_DEFAULT, buffer);

  cublasSaxpy(state.cublasHandle, N, &alpham1, state.d_y, 1, d_b, 1);
  blasStatus = cublasSdot(state.cublasHandle, N, d_b, 1, d_b, 1, &r1);

  int k = 1;

  while (r1 > tol * tol && k <= max_iter) {
    if (k > 1) {
      b = r1 / r0;
      blasStatus = cublasSscal(state.cublasHandle, N, &b, state.d_p, 1);
      blasStatus =
          cublasSaxpy(state.cublasHandle, N, &alpha, d_b, 1, state.d_p, 1);
    } else {
      blasStatus = cublasScopy(state.cublasHandle, N, d_b, 1, state.d_p, 1);
    }

    cusparseSpMV(state.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                 state.matA, state.vecp, &beta, state.vecy, CUDA_R_32F,
                 CUSPARSE_SPMV_ALG_DEFAULT, buffer);
    blasStatus =
        cublasSdot(state.cublasHandle, N, state.d_p, 1, state.d_y, 1, &dot);
    a = r1 / dot;

    blasStatus = cublasSaxpy(state.cublasHandle, N, &a, state.d_p, 1, d_x, 1);
    na = -a;
    blasStatus = cublasSaxpy(state.cublasHandle, N, &na, state.d_y, 1, d_b, 1);

    r0 = r1;
    blasStatus = cublasSdot(state.cublasHandle, N, d_b, 1, d_b, 1, &r1);
    cudaDeviceSynchronize();
    k++;
  }
  // printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
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
               const float *d_csrVal) {
  // Analyze Cholesky structure
  cusolverSpXcsrcholAnalysis(state.handle, n, nnz, state.descrA, d_csrRowPtr,
                             d_csrColInd, state.info);
  // Compute workspace size
  size_t size_internal = 0;
  cusolverSpScsrcholBufferInfo(state.handle, n, nnz, state.descrA, d_csrVal,
                               d_csrRowPtr, d_csrColInd, state.info,
                               &size_internal, &size_chol);

  // Allocate workspace on GPU
  cudaMalloc(&state.workspace, size_chol);

  // Factorize the matrix
  cusolverSpScsrcholFactor(state.handle, n, nnz, state.descrA, d_csrVal,
                           d_csrRowPtr, d_csrColInd, state.info,
                           state.workspace);
}

// Solve the linear system using cuSOLVER
void solveLDLT(CuSolverState &state, const int n, const float *d_b,
               float *d_x) {
  // Solve the linear system
  cusolverSpScsrcholSolve(state.handle, n, d_b, d_x, state.info,
                          state.workspace);
}

void solveLDLT2(CuSolverState &state, size_t &size_chol, const int n,
                const int nnz, const int *d_csrRowPtr, const int *d_csrColInd,
                const float *d_csrVal, const float *d_b, float *d_x) {
  // Analyze Cholesky structure
  cusolverSpXcsrcholAnalysis(state.handle, n, nnz, state.descrA, d_csrRowPtr,
                             d_csrColInd, state.info);
  // Compute workspace size
  size_t size_internal = 0;
  cusolverSpScsrcholBufferInfo(state.handle, n, nnz, state.descrA, d_csrVal,
                               d_csrRowPtr, d_csrColInd, state.info,
                               &size_internal, &size_chol);

  // Allocate workspace on GPU
  if (state.workspace) {
    cudaFree(state.workspace);
  }
  cudaMalloc(&state.workspace, size_chol);

  // Factorize the matrix
  cusolverSpScsrcholFactor(state.handle, n, nnz, state.descrA, d_csrVal,
                           d_csrRowPtr, d_csrColInd, state.info,
                           state.workspace);
  cusolverSpScsrcholSolve(state.handle, n, d_b, d_x, state.info,
                          state.workspace);
}

void solveUsingCusolver(const int n, const int nnz, const int *d_csrRowPtr,
                        const int *d_csrColInd, const float *d_csrVal,
                        const float *d_b, float *d_x) {
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

  cusolverStatus_t status = cusolverSpScsrlsvchol(
      handle, n, nnz, descrA, d_csrVal, d_csrRowPtr, d_csrColInd, d_b, 1e-12,
      reorder, d_x, &singularity);

  if (status != CUSOLVER_STATUS_SUCCESS) {
    std::cerr << "cusolverSpScsrlsvchol failed" << std::endl;
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
  bool compute(const SparseMatrixXf &M) {
    if (solver_type_ == Parameters::LDLT) {
      n = M.rows();
      nnz = M.nonZeros();

      CUDA_CHECK(cudaMalloc((void **)&d_csrVal, nnz * sizeof(float)));
      CUDA_CHECK(cudaMalloc((void **)&d_csrRowPtr, (n + 1) * sizeof(int)));
      CUDA_CHECK(cudaMalloc((void **)&d_csrColInd, nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc((void **)&d_b, n * sizeof(float)));
      CUDA_CHECK(cudaMalloc((void **)&d_x, n * sizeof(float)));

      CUDA_CHECK(cudaMemcpy(d_csrVal, M.valuePtr(), nnz * sizeof(float),
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

      CUDA_CHECK(cudaMalloc((void **)&d_csrVal, nnz * sizeof(float)));
      CUDA_CHECK(cudaMalloc((void **)&d_csrRowPtr, (n + 1) * sizeof(int)));
      CUDA_CHECK(cudaMalloc((void **)&d_csrColInd, nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc((void **)&d_b, n * sizeof(float)));
      CUDA_CHECK(cudaMalloc((void **)&d_x, n * sizeof(float)));

      CUDA_CHECK(cudaMemcpy(d_csrVal, M.valuePtr(), nnz * sizeof(float),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_csrRowPtr, M.outerIndexPtr(),
                            (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_csrColInd, M.innerIndexPtr(), nnz * sizeof(int),
                            cudaMemcpyHostToDevice));

      initializeConjugateGradientState(n, nnz, d_csrRowPtr, d_csrColInd,
                                       d_csrVal, d_x, CGSolverState);

      return true;
    } else if (solver_type_ == Parameters::CGChol) {

      n = M.rows();
      nnz = M.nonZeros();

      CUDA_CHECK(cudaMalloc((void **)&d_csrVal, nnz * sizeof(float)));
      CUDA_CHECK(cudaMalloc((void **)&d_csrRowPtr, (n + 1) * sizeof(int)));
      CUDA_CHECK(cudaMalloc((void **)&d_csrColInd, nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc((void **)&d_b, n * sizeof(float)));
      CUDA_CHECK(cudaMalloc((void **)&d_x, n * sizeof(float)));

      CUDA_CHECK(cudaMemcpy(d_csrVal, M.valuePtr(), nnz * sizeof(float),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_csrRowPtr, M.outerIndexPtr(),
                            (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_csrColInd, M.innerIndexPtr(), nnz * sizeof(int),
                            cudaMemcpyHostToDevice));

      // Create an IncompleteCholesky factorization object
      Eigen::IncompleteCholesky<float> ichol(M);

      // Perform the incomplete Cholesky factorization
      ichol.compute(M);
      Eigen::SparseMatrix<float> symbolicL = ichol.matrixL();
      float *valuePtrL = symbolicL.valuePtr();
      int *csrRowPtrL = symbolicL.outerIndexPtr();
      int *csrColIndL = symbolicL.innerIndexPtr();

      CUDA_CHECK(cudaMalloc((void **)&d_csrValL, nnz * sizeof(float)));
      CUDA_CHECK(cudaMalloc((void **)&d_csrRowPtrL, (n + 1) * sizeof(int)));
      CUDA_CHECK(cudaMalloc((void **)&d_csrColIndL, nnz * sizeof(int)));

      CUDA_CHECK(cudaMemcpy(d_csrValL, valuePtrL, nnz * sizeof(float),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_csrRowPtrL, csrRowPtrL, (n + 1) * sizeof(int),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_csrColIndL, csrColIndL, nnz * sizeof(int),
                            cudaMemcpyHostToDevice));

      return true;
    } else if (solver_type_ == Parameters::CGBig) {

      n = M.rows();
      nnz = M.nonZeros();

      CUDA_CHECK(cudaMalloc((void **)&d_csrVal, nnz * sizeof(float)));
      CUDA_CHECK(cudaMalloc((void **)&d_csrRowPtr, (n + 1) * sizeof(int)));
      CUDA_CHECK(cudaMalloc((void **)&d_csrColInd, nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc((void **)&d_b, n * sizeof(float)));
      CUDA_CHECK(cudaMalloc((void **)&d_x, n * sizeof(float)));
      cudaMemset(d_x, 0, n * sizeof(float));

      // Initialize the ConjugateState struct and perform factorization
      initializePreConjugateState(n, nnz, d_csrRowPtr, d_csrColInd, d_csrVal,
                                  preConjugatestate);

      CUDA_CHECK(cudaMemcpy(d_csrVal, M.valuePtr(), nnz * sizeof(float),
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

  bool solve(const Eigen::MatrixX3f &rhs, Eigen::MatrixX3f &sol) {
    if (solver_type_ == Parameters::LDLT) {

      int n_cols = rhs.cols();

      for (int i = 0; i < n_cols; ++i) {
        const float *b_data = rhs.col(i).data();

        CUDA_CHECK(cudaMemcpyAsync(d_b, b_data, n * sizeof(float),
                                   cudaMemcpyHostToDevice));

        solveLDLT(solverState, n, d_b, d_x);
        // solveUsingCusolver(n, nnz, d_csrRowPtr, d_csrColInd, d_csrVal, d_b,
        // d_x);

        CUDA_CHECK(cudaMemcpyAsync(sol.col(i).data(), d_x, n * sizeof(float),
                                   cudaMemcpyDeviceToHost));
      }

      return true;
    } else if (solver_type_ == Parameters::CG) {

      int n_cols = rhs.cols();

      for (int i = 0; i < n_cols; ++i) {
        CUDA_CHECK(cudaMemcpy(d_x, sol.col(i).data(), n * sizeof(float),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(d_b, rhs.col(i).data(), n * sizeof(float),
                              cudaMemcpyHostToDevice));

        solveUsingConjugateGradient(CGSolverState, n, nnz, d_csrRowPtr,
                                    d_csrColInd, d_csrVal, d_b, d_x);

        CUDA_CHECK(cudaMemcpy(sol.col(i).data(), d_x, n * sizeof(float),
                              cudaMemcpyDeviceToHost));
      }
      return true;

    } else if (solver_type_ == Parameters::CGChol) {

      int n_cols = rhs.cols();

      for (int i = 0; i < n_cols; ++i) {
        CUDA_CHECK(cudaMemcpy(d_x, sol.col(i).data(), n * sizeof(float),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(d_b, rhs.col(i).data(), n * sizeof(float),
                              cudaMemcpyHostToDevice));

        solveUsingPreconditionedConjugateGradient0(
            n, nnz, d_csrRowPtr, d_csrColInd, d_csrVal, d_b, d_x, d_csrRowPtrL,
            d_csrColIndL, d_csrValL);

        CUDA_CHECK(cudaMemcpy(sol.col(i).data(), d_x, n * sizeof(float),
                              cudaMemcpyDeviceToHost));
      }
      return true;

    } else if (solver_type_ == Parameters::CGBig) {

      int n_cols = rhs.cols();

      for (int i = 0; i < n_cols; ++i) {
        cudaMemset(d_x, 0, n * sizeof(float));

        CUDA_CHECK(cudaMemcpy(d_b, rhs.col(i).data(), n * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Call the PCG solver function with the precomputed factorization
        // generateILUFactors(preConjugatestate, n, nnz, d_csrRowPtr,
        // d_csrColInd,
        //                    d_csrVal, d_b, d_x);

        solveUsingPreconditionedConjugateGradient(preConjugatestate, n, nnz,
                                                  d_csrRowPtr, d_csrColInd,
                                                  d_csrVal, d_b, d_x);

        CUDA_CHECK(cudaMemcpy(sol.col(i).data(), d_x, n * sizeof(float),
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

  void destroy() {
    if (solver_type_ == Parameters::LDLT) {

      return;

    } else if (solver_type_ == Parameters::CG) {

      CUDA_CHECK(cudaFree(d_csrVal));
      CUDA_CHECK(cudaFree(d_csrRowPtr));
      CUDA_CHECK(cudaFree(d_csrColInd));
      CUDA_CHECK(cudaFree(d_b));
      CUDA_CHECK(cudaFree(d_x));

      destroyConjugateGradientState(CGSolverState);

      return;

    } else if (solver_type_ == Parameters::CGChol) {

      return;

    } else if (solver_type_ == Parameters::CGBig) {

      return;

    } else {
      return;
    }
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
  ConjugateGradientState CGSolverState;

  Parameters::LinearSolverType solver_type_;
  int n, nnz;
  float *d_csrVal, *d_b, *d_x;
  int *d_csrRowPtr, *d_csrColInd;

  float *d_csrValL;
  int *d_csrRowPtrL, *d_csrColIndL;
};

class LinearSolverCPU {
public:
  LinearSolverCPU(Parameters::LinearSolverType solver_type)
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

  bool solve(const Eigen::MatrixX3f &rhs, Eigen::MatrixX3f &sol) {
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
  Eigen::SimplicialLDLT<SparseMatrixXf> LDLT_solver_;
  Eigen::ConjugateGradient<SparseMatrixXf, Eigen::Lower | Eigen::Upper>
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