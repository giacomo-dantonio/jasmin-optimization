use argmin::prelude::*;
use argmin::prelude::ArgminDot;
use serde::{Deserialize, Serialize};

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

    let res = Executor::new(line_cost_func, linesearch, initial_step_length)
    .max_iters(50)
    .run()?;

    Ok(res.state.param)
}

#[derive(Serialize, Deserialize)]
struct Backtracking<F>
{
    function_value: F,
    contraction_factor : F, // rho
    c: F, 
    slope : F, // φ'(0)
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
        }
    }
}

impl<O, F> Solver<O> for Backtracking<F>
where
    F: ArgminFloat,
    O: ArgminOp<Float = F, Param = F, Output = F>,
{
    const NAME: &'static str = "Jasmin Backtracking";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        // Compute initial cost, gradient and hessian and set the initial state

        let initial_step_length = state.get_param();
        let initial_value = op.apply(&initial_step_length)?;

        let iter_data = ArgminIterData::<O>::new()
            .param(initial_step_length)
            .cost(initial_value);
        Ok(Some(iter_data))
    }

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>
    ) -> Result<ArgminIterData<O>, Error> {
        let current_step_length = state.param;
        let next_step_length = self.contraction_factor * current_step_length;

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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use crate::functions::{
        quadratic::Quadratic,
    };
    use crate::steplength::LineFunc;

    #[test]
    fn test_quadratic_function() {
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

        let res = Executor::new(cost, solver, 1.0)
        .max_iters(10)
        .run()
        .unwrap();

        assert_eq!(
            TerminationReason::LineSearchConditionMet,
            res.state().termination_reason
        );
    }
}
