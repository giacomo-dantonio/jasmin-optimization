use argmin::prelude::*;
use argmin::prelude::ArgminDot;

use serde::{Deserialize, Serialize};

/// Implementation of the backtracking line search.
/// Algorithm 3.1 of
/// Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.

#[derive(Serialize, Deserialize)]
pub struct Backtracking<F>
{
    function_value: F,
    contraction_factor : F, // rho
    slope : F // slope of the sufficient decrease constant
}

impl<F> Backtracking<F>
where
    F: ArgminFloat
{
    pub fn new<Param>(
        function_value: F,
        contraction_factor: F,
        c: F,
        gradient: Param,
        descent_dir: Param) -> Self
    where
        Param : ArgminDot<Param, F>
    {
        let slope = c * gradient.dot(&descent_dir);

        Backtracking {
            function_value,
            contraction_factor,
            slope
        }
    }
}

impl<O, F> Solver<O> for Backtracking<F>
where
    O: ArgminOp<Float = F, Param = F, Output = F>,
    F: ArgminFloat
{
    const NAME: &'static str = "Jasmin Backtracking";

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>
    ) -> Result<ArgminIterData<O>, Error> {
        let alphak = state.get_param();

        let iter_data = ArgminIterData::new()
            .param(self.contraction_factor * alphak);
        
        if let Ok(fk) = op.apply(&alphak) {
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
        if fk <= self.function_value + alphak * self.slope {
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
    use crate::functions::quadratic::Quadratic;
    use crate::linesearch::LineSearchFunc;

    #[test]
    fn test_quadratic() {
        let func = Quadratic::new(
            DMatrix::from_row_slice(3, 3, &[
                2f64, 1f64, 0f64,
                1f64, 2f64, 0f64,
                0f64, 0f64, 1f64
            ]),
            DVector::from_row_slice(&[0.0, 1.0, 2.0]),
            0.0,
        );
    
        let x
            = DVector::from_row_slice(&[1.0, 1.5, -0.5]);
        let cost
            = LineSearchFunc::new(func.clone(), x.clone()).unwrap();
        let gradient
            = func.gradient_at(&x);
        let value = func.evaluate_at(&x);
    
        let solver = Backtracking::new(
            value,
            0.7,
            1E-4,
            gradient.clone(),
            -gradient.clone(),
        );
    
        let res = Executor::new(
            cost,
            solver,
            1.0
        )
        .max_iters(10)
        .run()
        .unwrap();

        assert_eq!(
            TerminationReason::LineSearchConditionMet,
            res.state().termination_reason
        );
    }
}
