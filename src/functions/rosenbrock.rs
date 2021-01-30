use nalgebra::{DMatrix, DVector};
use argmin::prelude::*;

use super::Function;
#[derive(Clone)]
pub struct Rosenbrock2D {
    a: f64,
    b: f64
}

impl Rosenbrock2D {
    pub fn new(a: f64, b: f64) -> Self {
        Rosenbrock2D { a, b}
    }
}

impl ArgminOp for Rosenbrock2D {
    type Param = DVector<f64>;
    type Output = f64;
    type Hessian = DMatrix<f64>;
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let a = self.a;
        let b = self.b;
        let x1 = param[0];
        let x2 = param[1];

        Ok((a - x1).powi(2) + b * (x2 - x1.powi(2)).powi(2))
    }

    fn gradient(&self, param: &Self::Param) -> Result<Self::Param, Error> {
        let a = self.a;
        let b = self.b;
        let x1 = param[0];
        let x2 = param[1];

        let grad = DVector::from_row_slice(&[
            -2.0 * a + 2.0 * x1 - 4.0 * b * x1 * x2 + 4.0 * b * x1.powi(3),
            2.0 * b * (x2 - x1.powi(2))
        ]);
        Ok(grad)
    }

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, Error> {
        let b = self.b;
        let x1 = param[0];
        let x2 = param[1];

        let mat = DMatrix::from_row_slice(
            2, 2,
            &[
                2.0 - 4.0 * b * x2 + 12.0 * b * x1.powi(2), -4.0 * b * x1,
                -4.0 * b * x1, 2.0 * b
            ]);
        Ok(mat)
    }
}

impl Function for Rosenbrock2D {}
