#pragma once

#include <Eigen/Core>

namespace Eigen {

    namespace internal {
        template<typename MatrixType,
            typename Rhs,
            typename Dest,
            typename FilterMatrixType,
            typename Preconditioner>
            EIGEN_DONT_INLINE void constrained_conjugate_gradient(const MatrixType& mat,
                const Rhs& rhs,
                Dest& x,
                const FilterMatrixType& filter,
                const Preconditioner& precond,
                int& iters,
                typename Dest::RealScalar& tol_error)
        {
            using std::abs;
            using std::sqrt;
            typedef typename Dest::RealScalar RealScalar;
            typedef typename Dest::Scalar Scalar;
            typedef Matrix<Scalar, Dynamic, 1> VectorType;

            RealScalar tol = tol_error;
            int maxIters = iters;

            int n = mat.cols();

            VectorType residual = filter * (rhs - mat * x); /* initial residual */

            RealScalar rhsNorm2 = (filter * rhs).squaredNorm();
            if (rhsNorm2 == 0) {
                /* XXX TODO set constrained result here */
                x.setZero();
                iters = 0;
                tol_error = 0;
                return;
            }
            RealScalar threshold = tol * tol * rhsNorm2;
            RealScalar residualNorm2 = residual.squaredNorm();
            if (residualNorm2 < threshold) {
                iters = 0;
                tol_error = sqrt(residualNorm2 / rhsNorm2);
                return;
            }

            VectorType p(n);
            p = filter * precond.solve(residual); /* initial search direction */

            VectorType z(n), tmp(n);
            RealScalar absNew = numext::real(
                residual.dot(p)); /* the square of the absolute value of r scaled by invM */
            int i = 0;
            while (i < maxIters) {
                tmp.noalias() = filter * (mat * p); /* the bottleneck of the algorithm */

                Scalar alpha = absNew / p.dot(tmp); /* the amount we travel on dir */
                x += alpha * p;                     /* update solution */
                residual -= alpha * tmp;            /* update residue */

                residualNorm2 = residual.squaredNorm();
                if (residualNorm2 < threshold) {
                    break;
                }

                z = precond.solve(residual); /* approximately solve for "A z = residual" */

                RealScalar absOld = absNew;
                absNew = numext::real(residual.dot(z)); /* update the absolute value of r */
                RealScalar beta =
                    absNew /
                    absOld; /* calculate the Gram-Schmidt value used to create the new search direction */
                p = filter * (z + beta * p); /* update search direction */
                i++;
            }
            tol_error = sqrt(residualNorm2 / rhsNorm2);
            iters = i;
        }

    }  // namespace internal

    template<typename _MatrixType,
        int _UpLo = Lower,
        typename _FilterMatrixType = _MatrixType,
        typename _Preconditioner = DiagonalPreconditioner<typename _MatrixType::Scalar>>
        class ConstrainedConjugateGradient;

    namespace internal {

        template<typename _MatrixType, int _UpLo, typename _FilterMatrixType, typename _Preconditioner>
        struct traits<
            ConstrainedConjugateGradient<_MatrixType, _UpLo, _FilterMatrixType, _Preconditioner>> {
            typedef _MatrixType MatrixType;
            typedef _FilterMatrixType FilterMatrixType;
            typedef _Preconditioner Preconditioner;
        };

    }  // namespace internal

    template<typename _MatrixType, int _UpLo, typename _FilterMatrixType, typename _Preconditioner>
    class ConstrainedConjugateGradient
        : public IterativeSolverBase<
        ConstrainedConjugateGradient<_MatrixType, _UpLo, _FilterMatrixType, _Preconditioner>> {
        typedef IterativeSolverBase<ConstrainedConjugateGradient> Base;
        using Base::m_error;
        using Base::m_info;
        using Base::m_isInitialized;
        using Base::m_iterations;

    public:
        typedef _MatrixType MatrixType;
        typedef typename MatrixType::Scalar Scalar;
        typedef typename MatrixType::Index Index;
        typedef typename MatrixType::RealScalar RealScalar;
        typedef _FilterMatrixType FilterMatrixType;
        typedef _Preconditioner Preconditioner;

        enum { UpLo = _UpLo };

    public:
        /** Default constructor. */
        ConstrainedConjugateGradient() : Base()
        {
        }

        ConstrainedConjugateGradient(const MatrixType& A) : Base(A)
        {
        }

        ~ConstrainedConjugateGradient()
        {
        }

        FilterMatrixType& filter()
        {
            return m_filter;
        }
        const FilterMatrixType& filter() const
        {
            return m_filter;
        }

        /** \internal */
        template<typename Rhs, typename Dest> void _solve(const Rhs& b, Dest& x) const
        {
            x.setOnes();
            _solveWithGuess(b, x);
        }

    protected:
        FilterMatrixType m_filter;
    };
}