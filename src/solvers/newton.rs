use argmin::prelude::*;
use serde::{Deserialize, Serialize};

use linear_search_solver::Solver;
use crate::solvers::linesearch::LineSearch;
use crate::steplength::{LineFunc, backtracking::Backtracking};

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
    O::Param: ArgminMul<F, O::Param>,
    O::Hessian: ArgminInv<O::Hessian>
        + ArgminDot<O::Param, O::Param>
{
    fn descent_dir(
        &self,
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
}
