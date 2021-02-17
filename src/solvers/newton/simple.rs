use argmin::prelude::*;
use serde::{Deserialize, Serialize};

use linear_search_solver::Solver;
use crate::steplength::backtracking;
use crate::solvers::linesearch::LineSearch;

#[derive(Serialize, Deserialize, Solver)]
pub struct Newton {
}

impl Newton {
    pub fn new() -> Self {
        Newton {}
    }
}

impl<O, F> LineSearch<O, F> for Newton
where
    F: ArgminFloat,
    O: ArgminOp<Output = F, Float = F>,
    O::Param: ArgminScaledSub<O::Param, F, O::Param>
        + ArgminScaledAdd<O::Param, F, O::Param>
        + ArgminMul<F, O::Param>
        + ArgminDot<O::Param, F>,
    O::Hessian: ArgminInv<O::Hessian>
        + ArgminDot<O::Param, O::Param>
{
    fn descent_dir(
        &mut self,
        _op: &mut OpWrapper<O>,
        state: &IterState<O>
    ) -> Result<O::Param, Error> {
        let gradient = state.grad
            .as_ref()
            .ok_or(Error::msg("gradient unavailable"))?;

        let hessian = state.hessian
            .as_ref()
            .ok_or(Error::msg("hessian unavailable"))?;

        // FIXME: avoid inverting the hessian, solve the linear system instead.
        Ok(hessian
            .inv()?
            .dot(gradient)
            .mul(&F::from_f64(-1.0).unwrap()))
    }

    fn step_length(&self, op: &mut OpWrapper<O>, state: &IterState<O>, descent_dir: &O::Param)
        -> Result<O::Float, Error>
    {
        // The initial step length 1 is fine for newton's methods
        backtracking::step_length(op, state, descent_dir, F::from_f64(1.0).unwrap())
    }
}
