use argmin::prelude::*;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use trust_region_solver::Solverf64;
use crate::solvers::newton::cholesky;
use crate::solvers::trustregion::TrustRegion;

static CHOL_DELTA : f64 = 1E-4;
static CHOL_BETA : f64 = 100.0;

// solve the quadratic equation ax² + bx + c = 0
fn solve_quadratic(a: f64, b: f64, c: f64) -> Option<(f64, f64)>
{
    let discriminant = b.powi(2) - 4.0 * a * c;
    if discriminant > 0.0 {
        let d_sqrt = discriminant.sqrt();
        let root1 = (-b - d_sqrt) / (2.0 * a);
        let root2 = (-b + d_sqrt) / (2.0 * a);
        Some((root1, root2))
    }
    else {
        None
    }
}

#[derive(Serialize, Deserialize, Solverf64)]
pub struct NewtonDogleg{
    delta : f64
}

impl NewtonDogleg {
    pub fn new(delta: f64) -> Self
    {
        NewtonDogleg { delta }
    }
}

impl<O> TrustRegion<O, f64> for NewtonDogleg
where
    O: ArgminOp<
        Output = f64,
        Float = f64,
        Param = DVector<f64>,
        Hessian = DMatrix<f64>
    >
{
    fn solve_subproblem(&mut self, _op: &mut OpWrapper<O>, state: &IterState<O>, delta: f64)
        -> Result<O::Param, Error> {
        let grad = state.grad
            .as_ref()
            .ok_or(Error::msg("gradient not available."))?;
        let hessian = state.hessian
            .as_ref()
            .ok_or(Error::msg("hessian not available."))?;

        let u_factor = grad.norm() / (grad.transpose() * (hessian * grad))[(0, 0)];
        let param_u = -u_factor * grad;

        if param_u.norm() > delta {
            let tau = ((grad.transpose() * (hessian * grad))[(0, 0)] * delta) / grad.norm();
            Ok(tau * param_u)
        }
        else {
            let (mat_l, vec_d) = cholesky::factorization(&hessian, CHOL_DELTA, CHOL_BETA)?;
            let param_b = cholesky::solve(&mat_l, &vec_d, &(-grad))?;
            let p_bu = &param_b - &param_u;
    
            // nu == tau - 1
            let (_, nu) = solve_quadratic(
                p_bu.norm_squared(),
                param_u.dot(&p_bu),
                // (&param_u * &p_bu)[(0, 0)],
                &param_u.norm_squared() - delta.powi(2)
            ).ok_or(Error::msg("Cannot compute step length."))?;

            Ok(&param_u + nu * &p_bu)
        }
    }

    fn subproblem(&self, state: &IterState<O>, param: &O::Param)
        -> Result<f64, Error> {
        let grad = state.grad
            .as_ref()
            .ok_or(Error::msg("gradient not available."))?;
        let hessian = state.hessian
            .as_ref()
            .ok_or(Error::msg("hessian not available."))?;

        let value = 0.5 *((hessian * param).transpose() * param)[(0,0)]
            + (grad.transpose() * param)[(0,0)]
            + state.cost;

        Ok(value)
    }
}
