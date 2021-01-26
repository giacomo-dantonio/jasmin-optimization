use argmin::prelude::*;
use serde::{Deserialize, Serialize};

use crate::linesearch::{LineSearchFunc, backtracking::Backtracking};

#[derive(Serialize, Deserialize)]
pub struct SteepestDescent {
}

impl SteepestDescent {
    pub fn new() -> Self {
        SteepestDescent {}
    }
}

impl<O, F> Solver<O> for SteepestDescent
where
    F: ArgminFloat,
    O: ArgminOp<Output = F, Float = F> + Clone,
    O::Param: ArgminScaledSub<O::Param, F, O::Param>
        + ArgminScaledAdd<O::Param, F, O::Param>,
    O::Param: ArgminMul<F, O::Param>
        + ArgminDot<O::Param, F>
        + ArgminNorm<F>,
{
    const NAME: &'static str = "Jasmin steepest descent";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        let param = state.get_param();
        let initial_cost = op.apply(&param)?;
        let initial_grad = op.gradient(&param)?;

        let iter_data = ArgminIterData::<O>::new()
            .param(param)
            .cost(initial_cost)
            .grad(initial_grad);
        Ok(Some(iter_data))
    }


    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        let param = state.get_param();
        let gradient = state.grad
            .as_ref()
            .ok_or(Error::msg("gradient unavailable"))?;

        let descent_dir = gradient.mul(&F::from_f64(-1.0).unwrap());

        let line_cost_func = LineSearchFunc::new(op, &descent_dir, &param)?;

        // FIXME: avoid magic numbers.
        let linesearch = Backtracking::<F>::new::<O::Param>(
            state.cost,
            F::from_f64(0.7).unwrap(),
            F::from_f64(1E-4).unwrap(),
            gradient,
            &descent_dir,
        );

        // FIXME: use a smart strategy for computing the initial step length.
        let res = Executor::new(line_cost_func, linesearch, F::from_f64(1.0).unwrap())
        .max_iters(10)
        .run()?;

        let step_length = res.state.param;
        let next_param = param.scaled_add(&step_length, &descent_dir);
        let next_cost = op.apply(&next_param)?;
        let next_gradient = op.gradient(&next_param)?;

        Ok(
            ArgminIterData::new()
            .param(next_param)
            .cost(next_cost)
            .grad(next_gradient)
        )
    }

    fn terminate(&mut self, state: &IterState<O>) -> TerminationReason
    {
        if let Some(grad) = state.grad.as_ref() {
            if grad.norm() <= F::from_f64(1E-5).unwrap() {
                TerminationReason::TargetPrecisionReached
            }
            else {
                TerminationReason::NotTerminated
            }
        }
        else {
            TerminationReason::Aborted
        }
    }
}