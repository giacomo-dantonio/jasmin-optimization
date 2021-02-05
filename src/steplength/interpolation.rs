use argmin::prelude::*;

/// Polynomial interpolations used for computing the step length.
///
/// Reference:
///
/// Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// [page 58]


/// Interpolates the function value at 0, the derivative at 0
/// and the value at another parameter with a quadratic polynomial
/// and returns its minimum point.
pub fn quadratic<F>(
    zero_value: F, zero_derivative: F, parameter: F, parameter_value: F
) -> Result<F, Error>
    where F: ArgminFloat
{
    let a = (
        parameter_value
        - zero_value
        - parameter * zero_derivative
    ) / parameter.powi(2);
    let b = zero_derivative;

    if a > F::from_f64(1E-5).unwrap() {
        Ok (-b / (F::from_f64(2.0).unwrap() * a))
    }
    else {
        Err(Error::msg("Unable to compute quadratic interpolation"))
    }
}

/// Interpolates the function value at 0, the derivative at 0,
/// and the values at two other parameters with a cubic polynomial
/// and returns its minimum point.
pub fn cubic<F>(
    value_zero: F, derivative_zero: F,
    parameter_a: F, value_a: F,
    parameter_b: F, value_b: F,
) -> Result<F, Error>
where F: ArgminFloat
{
    let det = parameter_a.powi(2)
        * parameter_b.powi(2)
        * (parameter_b - parameter_a);

    if det < F::from_f64(1E-5).unwrap() {
        return Err(Error::msg("Unable to compute cubic interpolation"));
    }

    let a = (
        parameter_a.powi(2) * (value_b - value_zero - derivative_zero * parameter_b)
        - parameter_b.powi(2) * (value_a - value_zero - derivative_zero * parameter_a)
    ) / det;

    let b = (
        - parameter_a.powi(3) * (value_b - value_zero - derivative_zero * parameter_b)
        + parameter_b.powi(3) * (value_a - value_zero - derivative_zero * parameter_a)
    ) / det;

    if a < F::from_f64(1E-5).unwrap() {
        return Err(Error::msg("Unable to compute cubic interpolation"));
    }

    let three = F::from_f64(3.0).unwrap();
    let next_value =
        ((b.powi(2) - three * a * derivative_zero).sqrt() - b) / (three * a);

    Ok(next_value)
}