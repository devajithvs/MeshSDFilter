#include "SDFilter.h"

#include "cuda_runtime.h"

// Linear solver for symmetric positive definite matrix,
namespace SDFilter {

/******************************************/
/* SET HERMITIAN POSITIVE DEFINITE MATRIX */
/******************************************/
// --- Credit to:
// https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab
void setPDMatrix(double *__restrict h_A, const int N) {

  // --- Initialize random seed
  srand(time(NULL));

  double *h_A_temp = (double *)malloc(N * N * sizeof(double));

  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      h_A_temp[i * N + j] = (float)rand() / (float)RAND_MAX;

  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      h_A[i * N + j] = 0.5 * (h_A_temp[i * N + j] + h_A_temp[j * N + i]);

  for (int i = 0; i < N; i++)
    h_A[i * N + i] = h_A[i * N + i] + N;
}

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