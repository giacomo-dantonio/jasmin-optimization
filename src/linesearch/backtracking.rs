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

