use argmin::prelude::*;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use linear_search_solver::Solverf64;
use crate::steplength::backtracking;
use crate::solvers::linesearch::LineSearch;

use super::cholesky;

static DELTA : f64 = 1E-4;
static BETA : f64 = 100.0;

#[derive(Serialize, Deserialize, Solverf64)]
pub struct NewtonWithModifications {
}

impl NewtonWithModifications {
    pub fn new() -> Self {
        NewtonWithModifications {}
    }
}

impl<O> LineSearch<O, f64> for NewtonWithModifications
where
    O: ArgminOp<
        Output = f64,
        Float = f64,
        Param = DVector<f64>,
        Hessian = DMatrix<f64>
    >
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

        let (mat_l, vec_d) = cholesky::factorization(&hessian, DELTA, BETA)?;
        let descent_dir = cholesky::solve(&mat_l, &vec_d, &(-gradient))?;

        Ok(descent_dir)
    }

    fn step_length(&self, op: &mut OpWrapper<O>, state: &IterState<O>, descent_dir: &O::Param)
        -> Result<O::Float, Error>
    {
        // The initial step length 1 is fine for newton's methods
        backtracking::step_length(op, state, descent_dir, 1.0)
    }
}
