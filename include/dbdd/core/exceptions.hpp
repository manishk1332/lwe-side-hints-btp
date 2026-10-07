#pragma once

#include <stdexcept>
#include <string>

namespace dbdd {

class DBDDException : public std::runtime_error {
public:
    explicit DBDDException(const std::string& message);
};

class DimensionMismatchException : public DBDDException {
public:
    explicit DimensionMismatchException(const std::string& message);
};

class SingularMatrixException : public DBDDException {
public:
    explicit SingularMatrixException(const std::string& message);
};

class NonPositiveSemidefiniteException : public DBDDException {
public:
    explicit NonPositiveSemidefiniteException(const std::string& message);
};

class PrimitivityViolationException : public DBDDException {
public:
    explicit PrimitivityViolationException(const std::string& message);
};

class NumericalPrecisionException : public DBDDException {
public:
    explicit NumericalPrecisionException(const std::string& message);
};

class DivisionByZeroException : public DBDDException {
public:
    explicit DivisionByZeroException(const std::string& message);
};

class IndexOutOfBoundsException : public DBDDException {
public:
    explicit IndexOutOfBoundsException(const std::string& message);
};

class EnumerationTimeoutException : public DBDDException {
public:
    explicit EnumerationTimeoutException(const std::string& message);
};

class InvalidParameterException : public DBDDException {
public:
    explicit InvalidParameterException(const std::string& message);
};

} // namespace dbdd
