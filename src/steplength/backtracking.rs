use argmin::prelude::*;
use argmin::prelude::ArgminDot;
use serde::{Deserialize, Serialize};
use nalgebra::{Matrix2, Vector2};

use crate::steplength::LineFunc;

/// Implementation of the backtracking line search.
/// Algorithm 3.1 of
/// Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.

pub fn step_length<O, F>(
    op: &mut OpWrapper<O>,
    state: &IterState<O>, 
    descent_dir: &O::Param,
    initial_step_length: F
) -> Result<F, Error>
    where
        F: ArgminFloat,
        O: ArgminOp<Output = F, Float = F>,
        O::Param: ArgminScaledAdd<O::Param, F, O::Param>
            + ArgminDot<O::Param, F>
{
    let param = state.get_param();
    let gradient = state.grad
        .as_ref()
        .ok_or(Error::msg("gradient unavailable"))?;

    let line_cost_func = LineFunc::new(op, descent_dir, &param)?;
    
    // FIXME: avoid magic numbers.
    let linesearch = Backtracking::<F>::new::<O::Param>(
        state.cost,
        F::from_f64(0.7).unwrap(),
        F::from_f64(1E-4).unwrap(),
        gradient,
        &descent_dir,
    );

    let linesearch = Backtracking::<F>::interpolation::<O::Param>(
        state.cost,
        F::from_f64(1E-4).unwrap(),
        gradient,
        &descent_dir,
    );

    let res = Executor::new(line_cost_func, linesearch, initial_step_length)
    .max_iters(10)
    .run()?;

    Ok(res.state.param)
}

#[derive(Serialize, Deserialize)]
enum Strategy {
    Simple,
    Interpolation
}

#[derive(Serialize, Deserialize)]
struct Backtracking<F>
{
    function_value: F,
    contraction_factor : F, // rho
    c: F, 
    slope : F, // φ'(0)
    strategy : Strategy
}

impl<F> Backtracking<F>
where
    F: ArgminFloat
{
    pub fn new<Param>(
        function_value: F,
        contraction_factor: F,
        c: F,
        gradient: &Param,
        descent_dir: &Param) -> Self
    where
        Param : ArgminDot<Param, F>
    {
        let slope = gradient.dot(&descent_dir);

        Backtracking {
            function_value,
            contraction_factor,
            c,
            slope,
            strategy: Strategy::Simple
        }
    }

    pub fn interpolation<Param>(
        function_value: F,
        c: F,
        gradient: &Param,
        descent_dir: &Param) -> Self
    where
        Param : ArgminDot<Param, F>
    {
        let slope = gradient.dot(&descent_dir);

        Backtracking {
            function_value,
            contraction_factor: F::from_f64(0.0).unwrap(),
            c,
            slope,
            strategy: Strategy::Interpolation
        }
    }

    /// Quadratic interpolation of the first two function values and
    /// the derivate at 0.
    /// see page 58.
    fn quadratic_interpolation(&self, current_step_length: F, current_value: F) -> F {
        let a = (
            current_value
            - self.function_value
            - current_step_length * self.slope
        ) / current_step_length.powi(2);
        let b = self.slope;

        -b / (F::from_f64(2.0).unwrap() * a)
    }

    /// Cubic interpolation as in page 58.
    fn cubic_interpolation(
        &self,
        previous_step_length: F, previous_value: F,
        current_step_length: F, current_value: F
    ) -> F
    {
        let function_value = self.function_value;
        let slope = self.slope;
        let det = previous_step_length.powi(2)
            * current_step_length.powi(2)
            * (current_step_length - previous_step_length);

        let a = (
            previous_step_length.powi(2) * (current_value - function_value - slope * current_step_length)
            - current_step_length.powi(2) * (previous_value - function_value - slope * previous_step_length)
        ) / det;

        let b = (
            - previous_step_length.powi(3) * (current_value - function_value - slope * current_step_length)
            + current_step_length.powi(3) * (previous_value - function_value - slope * previous_step_length)
        ) / det;

        let three = F::from_f64(3.0).unwrap();
        let next_step_length =
            ((b.powi(2) - three * a * slope).sqrt() - b) / (three * a);

        next_step_length
    }
}

impl<O, F> Solver<O> for Backtracking<F>
where
    F: ArgminFloat,
    O: ArgminOp<Float = F, Param = F, Output = F>,
{
    const NAME: &'static str = "Jasmin Backtracking";

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>
    ) -> Result<ArgminIterData<O>, Error> {
        let current_step_length = state.param;
        let current_value = state.cost;
        let previous_step_length = state.prev_param;
        let previous_value = state.prev_cost;

        let mut next_step_length = match self.strategy {
            Strategy::Simple => self.contraction_factor * current_step_length,
            Strategy::Interpolation
                if current_step_length == previous_step_length
                => self.quadratic_interpolation(current_step_length, current_value),
            Strategy::Interpolation
                => self.cubic_interpolation(previous_step_length, previous_value, current_step_length, current_value)
        };

        // Safeguard procedure:
        // If any α_i is either too close to its predecessor α_{i-1} or else
        // too much smaller than α_{i-1}, we reset α_i = α_{i-1} / 2.

        // FIXME: Avoid magic numbers
        let step_difference = (next_step_length - current_step_length).abs();
        if step_difference < F::from_f64(1E-4).unwrap()
            || step_difference > F::from_f64(10.0).unwrap() * current_step_length {
            next_step_length = current_step_length / F::from_f64(2.0).unwrap();
        }

        let iter_data = ArgminIterData::new()
            .param(next_step_length);
        
        if let Ok(fk) = op.apply(&current_step_length) {
            Ok(iter_data.cost(fk))
        }
        else {
            Ok(iter_data
                .termination_reason(TerminationReason::Aborted))
        }
    }

    fn terminate(&mut self, state: &IterState<O>) -> TerminationReason
    {
        let alphak = state.get_param();
        let fk = state.get_cost();

        // sufficient decrease condition
        if fk <= self.function_value + alphak * self.c * self.slope {
            TerminationReason::LineSearchConditionMet
        }
        else {
            TerminationReason::NotTerminated
        }
    }
}

// #[cfg(test)]
pub mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use crate::functions::quadratic::Quadratic;
    use crate::steplength::LineFunc;

    struct TestData<'a> {
        cost: LineFunc<'a, Quadratic>,
        gradient: DVector<f64>,
        descent_dir: DVector<f64>,
        solver: Backtracking<f64>
    }

    impl<'a> TestData<'a>
    {
        fn test(callback: Box<dyn Fn(TestData)>)
        {
            let func = Quadratic::new(
                DMatrix::from_row_slice(3, 3, &[
                    2f64, 1f64, 0f64,
                    1f64, 2f64, 0f64,
                    0f64, 0f64, 1f64
                ]),
                DVector::from_row_slice(&[0.0, 1.0, 2.0]),
                0.0,
            );
        
            let x = DVector::from_row_slice(&[1.0, 1.5, -0.5]);
            let gradient = func.gradient_at(&x);
            let value = func.evaluate_at(&x);
            let descent_dir = gradient.mul(&(-1.0));
    
            let cost = LineFunc::new(&func, &descent_dir, &x).unwrap();
    
            let solver = Backtracking::new(
                value,
                0.7,
                1E-4,
                &gradient,
                &descent_dir,
            );
    
            let descent_dir = descent_dir.clone();
            let data = TestData {
                cost,
                gradient,
                descent_dir,
                solver
            };

            callback(data);
        }
    }

    #[test]
    fn test_quadratic_function() {
        TestData::test(Box::new(|data| {
            let res = Executor::new(
                data.cost,
                data.solver,
                1.0
            )
            .max_iters(10)
            .run()
            .unwrap();
    
            assert_eq!(
                TerminationReason::LineSearchConditionMet,
                res.state().termination_reason
            );
        }));
    }

    #[test]
    fn test_quadratic_interpolation() {
        TestData::test(Box::new(|data| {
            let value = data.cost.apply(&0.0).unwrap();

            let step_length = 1f64;
            let next_value = data.cost.apply(&step_length).unwrap();
    
            let b = data.gradient.dot(&data.descent_dir);
            let a = (next_value - value - step_length * b) / step_length.powi(2);
    
            assert!(a > 0.0);
            assert!(b < 0.0);
    
            let next_step_length = data.solver.quadratic_interpolation(step_length, next_value);
            let expected = -b / (2.0 * a);
    
            assert_eq!(expected, next_step_length);
        }));
    }

    // #[test]
    pub fn test_cubic_interpolation() {
        TestData::test(Box::new(|data| {
            let phi_0 = data.cost.apply(&0.0).unwrap();
            let phi_first_0 = data.gradient.dot(&data.descent_dir);
            let alpha_0 = 1f64;
            let phi_alpha_0 = data.cost.apply(&alpha_0).unwrap();
            let alpha_1 = data.solver.quadratic_interpolation(alpha_0, phi_alpha_0);
            let phi_alpha_1 = data.cost.apply(&alpha_1).unwrap();

            let det = alpha_0.powi(2) * alpha_1.powi(2) * (alpha_1 - alpha_0);

            let a = (
                alpha_0.powi(2) * (phi_alpha_1 - phi_0 - phi_first_0 * alpha_1)
                - alpha_1.powi(2) * (phi_alpha_0 - phi_0 - phi_first_0 * alpha_0)
            ) / det;
            let b = (
                -alpha_0.powi(3) * (phi_alpha_1 - phi_0 - phi_first_0 * alpha_1)
                + alpha_1.powi(3) * (phi_alpha_0 - phi_0 - phi_first_0 * alpha_0)
            ) / det;

            let phi_c = |alpha: f64| a * alpha.powi(3) + b * alpha.powi(2) + phi_first_0 * alpha + phi_0;
            let phi_first_c = |alpha: f64| 3.0 * a * alpha.powi(2) + 2.0 * b * alpha + phi_first_0;
            let phi_second_c = |alpha: f64| 6.0 * a * alpha + 2.0 * b;

            // Assert that interpolation is correct
            assert_eq!(phi_0, phi_c(0.0));
            assert_eq!(phi_first_0, phi_first_c(0.0));
            assert!((phi_alpha_0 - phi_c(alpha_0)).abs() < 1E-5);
            assert!((phi_alpha_1 - phi_c(alpha_1)).abs() < 1E-5);

            let next_step_length = data.solver.cubic_interpolation(
                alpha_0, phi_alpha_0,
                alpha_1, phi_alpha_1
            );

            assert_eq!(0.0, phi_first_c(next_step_length));
            assert!(phi_second_c(next_step_length) > 0.0);
        }));
    }
}
