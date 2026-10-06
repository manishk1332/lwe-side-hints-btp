#include <catch2/catch_test_macros.hpp>
#include "dbdd/linalg/matrix.hpp"
#include "dbdd/core/exceptions.hpp"

using namespace dbdd;

TEST_CASE("Matrix basic construction, row-basis access, and operations", "[linalg][matrix]") {
    Matrix<BigInt> M(2, 3, {
        BigInt(1), BigInt(2), BigInt(3),
        BigInt(4), BigInt(5), BigInt(6)
    });

    REQUIRE(M.rows() == 2);
    REQUIRE(M.cols() == 3);
    REQUIRE(!M.is_square());

    Vector<BigInt> r0 = M.row(0);
    REQUIRE(r0.size() == 3);
    REQUIRE(r0[0] == BigInt(1));
    REQUIRE(r0[2] == BigInt(3));

    Vector<BigInt> c1 = M.col(1);
    REQUIRE(c1.size() == 2);
    REQUIRE(c1[0] == BigInt(2));
    REQUIRE(c1[1] == BigInt(5));
}

TEST_CASE("Matrix multiplication and associativity invariant: (A B) C == A (B C)", "[linalg][matrix]") {
    Matrix<BigInt> A(2, 3, {
        BigInt(1), BigInt(2), BigInt(3),
        BigInt(0), BigInt(1), BigInt(4)
    });

    Matrix<BigInt> B(3, 2, {
        BigInt(2), BigInt(-1),
        BigInt(1), BigInt(0),
        BigInt(3), BigInt(2)
    });

    Matrix<BigInt> C(2, 2, {
        BigInt(1), BigInt(2),
        BigInt(3), BigInt(4)
    });

    Matrix<BigInt> AB_C = (A * B) * C;
    Matrix<BigInt> A_BC = A * (B * C);

    REQUIRE(AB_C == A_BC);
}

TEST_CASE("Matrix transpose invariant: (A B)^T == B^T A^T", "[linalg][matrix]") {
    Matrix<BigInt> A(2, 3, {
        BigInt(1), BigInt(2), BigInt(3),
        BigInt(4), BigInt(5), BigInt(6)
    });

    Matrix<BigInt> B(3, 2, {
        BigInt(7), BigInt(8),
        BigInt(9), BigInt(1),
        BigInt(2), BigInt(3)
    });

    Matrix<BigInt> lhs = (A * B).transpose();
    Matrix<BigInt> rhs = B.transpose() * A.transpose();

    REQUIRE(lhs == rhs);
}

TEST_CASE("Matrix row-vector multiplication: (v A) B == v (A B)", "[linalg][matrix]") {
    Matrix<BigInt> A(2, 3, {
        BigInt(1), BigInt(2), BigInt(0),
        BigInt(-1), BigInt(1), BigInt(3)
    });

    Matrix<BigInt> B(3, 2, {
        BigInt(2), BigInt(1),
        BigInt(0), BigInt(2),
        BigInt(-1), BigInt(3)
    });

    Vector<BigInt> v = {BigInt(3), BigInt(-2)};

    Vector<BigInt> lhs = (v * A) * B;
    Vector<BigInt> rhs = v * (A * B);

    REQUIRE(lhs == rhs);
}

TEST_CASE("Matrix exact Bareiss determinant invariants", "[linalg][matrix]") {
    // Identity
    Matrix<BigInt> I3 = Matrix<BigInt>::identity(3);
    REQUIRE(I3.determinant() == BigInt(1));

    // Diagonal
    Vector<BigInt> diag = {BigInt(2), BigInt(3), BigInt(-5)};
    Matrix<BigInt> D = Matrix<BigInt>::diagonal(diag);
    REQUIRE(D.determinant() == BigInt(-30));

    // Upper triangular
    Matrix<BigInt> U(3, 3, {
        BigInt(2), BigInt(7), BigInt(1),
        BigInt(0), BigInt(3), BigInt(9),
        BigInt(0), BigInt(0), BigInt(5)
    });
    REQUIRE(U.determinant() == BigInt(30));

    // Known 3x3 matrix:
    // [ 6,  1,  1 ]
    // [ 4, -2,  5 ]
    // [ 2,  8,  7 ]
    // det = 6*(-14 - 40) - 1*(28 - 10) + 1*(32 - (-4)) = 6*(-54) - 18 + 36 = -324 - 18 + 36 = -306
    Matrix<BigInt> M(3, 3, {
        BigInt(6), BigInt(1), BigInt(1),
        BigInt(4), BigInt(-2), BigInt(5),
        BigInt(2), BigInt(8), BigInt(7)
    });
    REQUIRE(M.determinant() == BigInt(-306));
}

TEST_CASE("Matrix exact rational inverse and linear system solve", "[linalg][matrix]") {
    Matrix<BigRational> A(2, 2, {
        BigRational(4), BigRational(7),
        BigRational(2), BigRational(6)
    });
    // det = 24 - 14 = 10
    REQUIRE(A.determinant() == BigRational(10));

    Matrix<BigRational> A_inv = A.inverse();
    Matrix<BigRational> I2 = Matrix<BigRational>::identity(2);

    REQUIRE(A * A_inv == I2);
    REQUIRE(A_inv * A == I2);

    // Solve x * A = b
    Vector<BigRational> b = {BigRational(1), BigRational(2)};
    Vector<BigRational> x = A.solve_row(b);
    REQUIRE(x * A == b);

    // Singular matrix check
    Matrix<BigRational> singular(2, 2, {
        BigRational(2), BigRational(4),
        BigRational(1), BigRational(2)
    });
    REQUIRE(singular.determinant() == BigRational(0));
    REQUIRE_THROWS_AS(singular.inverse(), SingularMatrixException);
}

TEST_CASE("Matrix rank computation", "[linalg][matrix]") {
    Matrix<BigRational> full_rank(3, 3, {
        BigRational(1), BigRational(0), BigRational(0),
        BigRational(0), BigRational(1), BigRational(0),
        BigRational(0), BigRational(0), BigRational(1)
    });
    REQUIRE(full_rank.rank() == 3);

    Matrix<BigRational> rank_2(3, 3, {
        BigRational(1), BigRational(2), BigRational(3),
        BigRational(2), BigRational(4), BigRational(6),
        BigRational(0), BigRational(1), BigRational(1)
    });
    REQUIRE(rank_2.rank() == 2);
}

TEST_CASE("Matrix block constructions", "[linalg][matrix]") {
    Matrix<BigInt> I2 = Matrix<BigInt>::identity(2);
    Matrix<BigInt> Z2 = Matrix<BigInt>::zero(2, 2);

    Matrix<BigInt> H = Matrix<BigInt>::block_horizontal(I2, Z2);
    REQUIRE(H.rows() == 2);
    REQUIRE(H.cols() == 4);

    Matrix<BigInt> V = Matrix<BigInt>::block_vertical(I2, Z2);
    REQUIRE(V.rows() == 4);
    REQUIRE(V.cols() == 2);

    Matrix<BigInt> D = Matrix<BigInt>::block_diagonal(I2, I2);
    REQUIRE(D.rows() == 4);
    REQUIRE(D.cols() == 4);
    REQUIRE(D.determinant() == BigInt(1));
}

TEST_CASE("Matrix Eigen conversion roundtrip", "[linalg][matrix]") {
    Matrix<double> M(2, 2, {
        1.0, 2.0,
        3.0, 4.0
    });

    Eigen::MatrixXd em = M.to_eigen();
    REQUIRE(em.rows() == 2);
    REQUIRE(em.cols() == 2);
    REQUIRE(em(0, 1) == 2.0);

    Matrix<double> back = Matrix<double>::from_eigen(em);
    REQUIRE(back == M);
}
