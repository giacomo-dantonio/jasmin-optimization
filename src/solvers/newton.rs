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
    O: ArgminOp<Output = F, Float = F> + Clone,
    O::Param: ArgminScaledSub<O::Param, F, O::Param>
        + ArgminScaledAdd<O::Param, F, O::Param>
        + ArgminMul<F, O::Param>
        + ArgminDot<O::Param, F>,
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

    /// Backtracking step length search
    fn step_lengh(&self, op: &mut OpWrapper<O>, state: &IterState<O>, descent_dir: &O::Param)
        -> Result<O::Float, Error>
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

        // The initial step length 1 is fine for newton's methods
        let res = Executor::new(line_cost_func, linesearch, F::from_f64(1.0).unwrap())
        .max_iters(10)
        .run()?;

        Ok(res.state.param)
    }
}
