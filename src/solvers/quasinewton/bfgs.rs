use argmin::prelude::*;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use linear_search_solver::Solverf64;
use crate::solvers::newton::cholesky;
use crate::solvers::linesearch::LineSearch;
use crate::steplength::wolfe;

static DELTA : f64 = 1E-4;
static BETA : f64 = 100.0;
static MAX_WOLFE : u32 = 100;

#[derive(Serialize, Deserialize, Solverf64)]
pub struct Bfgs {
    h_matrix: DMatrix<f64>
}

impl Bfgs {
    pub fn new(initial_hessian: &DMatrix<f64>) -> Result<Self, Error>
    {
        let (mat_l, vec_d) = cholesky::factorization(initial_hessian, DELTA, BETA)?;
        let mat_l_t = mat_l.transpose();

        Ok(Self {
            h_matrix: mat_l * DMatrix::from_diagonal(&vec_d) * mat_l_t
        })
    }

    pub fn solve_direction(&self, grad: &DVector<f64>) -> Result<DVector<f64>, Error>
    {
        if let Some(chol) = self.h_matrix.clone().cholesky() {
            Ok(-chol.solve(&grad))
        }
        else {
            let lu = self.h_matrix.clone().lu();
            let ascent_dir = lu.solve(&grad)
                .ok_or(Error::msg("Cannot solve H-matrix"))?;
            Ok(-ascent_dir)
        }
    }
}

impl<O> LineSearch<O, f64> for Bfgs
where
    O: ArgminOp<
        Output = f64,
        Float = f64,
        Param = DVector<f64>,
        Hessian = DMatrix<f64>
    >
{
    fn descent_dir(
        &mut self,
        _op: &mut OpWrapper<O>,
        state: &IterState<O>
    ) -> Result<O::Param, Error> {
        let grad = state.grad
            .as_ref()
            .ok_or(Error::msg("gradient unavailable"))?;

        if state.iter == 0 {
            let descent_dir = -transform_vec(&self.h_matrix, &grad);
            return Ok(descent_dir);
        }

        let prev_grad = state.prev_grad
            .as_ref()
            .ok_or(Error::msg("gradient unavailable"))?;
        let y = grad - prev_grad;

        let s = state.param.clone().sub(&state.prev_param);

        assert!(y.dot(&s) > 0.0);

        let rho = 1.0 / y.dot(&s);

        let eye = DMatrix::identity(self.h_matrix.nrows(), self.h_matrix.ncols());
        let rank_one = eye - rho * outer_prod(&s, &y)?;
        let next_hessian =
           rank_one.mul(&mul_tr(&self.h_matrix, &rank_one)?)
            + rho * outer_prod(&s, &s)?;
        self.h_matrix = next_hessian;

        self.solve_direction(&grad)
    }

    fn step_length(&self, op: &mut OpWrapper<O>, state: &IterState<O>, descent_dir: &O::Param)
    -> Result<O::Float, Error>
    {
        let grad = state.grad
            .as_ref()
            .ok_or(Error::msg("gradient unavailable"))?;

        assert!(grad.dot(descent_dir) < 0.0);
        wolfe::step_length(op, state, descent_dir, 1.0, MAX_WOLFE)
    }
}

// FIXME: use generics to avoid duplicated code
fn mul_tr(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>) -> Result<DMatrix<f64>, Error>
{
    if lhs.ncols() != rhs.ncols() {
        return Err(Error::msg("Cannot compute outer product of incompatible dimensions."));
    }

    let mut result = DMatrix::zeros(lhs.nrows(), rhs.nrows());
    lhs.mul_to(&rhs.transpose(), &mut result);

    Ok(result)
}

// FIXME: use generics to avoid duplicated code
fn outer_prod(lhs: &DVector<f64>, rhs: &DVector<f64>) -> Result<DMatrix<f64>, Error>
{
    if lhs.ncols() != rhs.ncols() {
        return Err(Error::msg("Cannot compute outer product of incompatible dimensions."));
    }

    let mut result = DMatrix::zeros(lhs.nrows(), rhs.nrows());
    lhs.mul_to(&rhs.transpose(), &mut result);

    Ok(result)
}

fn transform_vec(mat: &DMatrix<f64>, vec: &DVector<f64>) -> DVector<f64> {
    mat * vec
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_bfgs() {

    }
}