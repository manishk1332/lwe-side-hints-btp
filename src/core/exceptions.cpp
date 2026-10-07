#include "dbdd/core/exceptions.hpp"

namespace dbdd {

DBDDException::DBDDException(const std::string& message)
    : std::runtime_error(message) {}

DimensionMismatchException::DimensionMismatchException(const std::string& message)
    : DBDDException("Dimension mismatch: " + message) {}

SingularMatrixException::SingularMatrixException(const std::string& message)
    : DBDDException("Singular matrix: " + message) {}

NonPositiveSemidefiniteException::NonPositiveSemidefiniteException(const std::string& message)
    : DBDDException("Non-positive semidefinite matrix: " + message) {}

PrimitivityViolationException::PrimitivityViolationException(const std::string& message)
    : DBDDException("Primitivity violation: " + message) {}

NumericalPrecisionException::NumericalPrecisionException(const std::string& message)
    : DBDDException("Numerical precision error: " + message) {}

DivisionByZeroException::DivisionByZeroException(const std::string& message)
    : DBDDException("Division by zero: " + message) {}

IndexOutOfBoundsException::IndexOutOfBoundsException(const std::string& message)
    : DBDDException("Index out of bounds: " + message) {}

EnumerationTimeoutException::EnumerationTimeoutException(const std::string& message)
    : DBDDException("Enumeration timeout: " + message) {}

InvalidParameterException::InvalidParameterException(const std::string& message)
    : DBDDException("Invalid parameter: " + message) {}

} // namespace dbdd
