use nalgebra::{ DMatrix, DVector };
use argmin::prelude::*;

/// Multivariate quadratic function of the form
/// 0.5 x^T Q x + b^T x + c
pub struct Quadratic {
    mat_q: DMatrix<f64>,
    b: DVector<f64>,
    c: f64,
}

impl Quadratic {
    pub fn new(q: DMatrix<f64>, b: DVector<f64>, c: f64) -> Self {
        Quadratic {
            mat_q: q,
            b,
            c
        }
    }

    pub fn evaluate_at(&self, at: &DVector<f64>) -> f64 {
        0.5 *((&self.mat_q * at).transpose() * at)[(0,0)]
        + (&self.b.transpose() * at)[(0,0)]
        + self.c
    }

    pub fn gradient_at(&self, at: &DVector<f64>) -> DVector<f64> {
        &self.mat_q * at + &self.b
    }
}

impl ArgminOp for Quadratic {
    type Param = DVector<f64>;
    type Output = f64;
    type Hessian = DMatrix<f64>;
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(self.evaluate_at(param))
    }

    fn gradient(&self, param: &Self::Param) -> Result<Self::Param, Error> {
        Ok(self.gradient_at(param))
    }

    fn hessian(&self, _param: &Self::Param) -> Result<Self::Hessian, Error> {
        Ok(self.mat_q.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_function() {
        let mut rng = rand::thread_rng();

        let func
        = Quadratic::new(
            DMatrix::from_row_slice(3, 3, &[
                2.0, 0.0, 0.0,
                0.0, 2.0, 0.0,
                0.0, 0.0, 2.0
            ]),
            DVector::from_row_slice(&[1.0, -1.0, 1.0]),
            1.0
        );

        let expected = |x: f64, y: f64, z: f64|
            x.pow(2) + y.pow(2) + z.pow(2)
            + x - y + z + 1f64;

        for _ in 0 .. 10 {
            let (x, y, z) = (rng.gen(), rng.gen(), rng.gen());
            let point
                = DVector::from_row_slice(&[x, y, z]);

            let error: f64 = expected(x, y, z) - func.evaluate_at(&point);
            assert!(error.abs() < 1E-10);
        }
    }

    #[test]
    fn test_gradient()
    {
        let mut rng = rand::thread_rng();

        let func
        = Quadratic::new(
            DMatrix::from_row_slice(3, 3, &[
                2.0, 0.0, 0.0,
                0.0, 2.0, 0.0,
                0.0, 0.0, 2.0
            ]),
            DVector::from_row_slice(&[1.0, -1.0, 1.0]),
            1.0
        );

        let expected = |x: f64, y: f64, z: f64|
            DVector::from_row_slice(&[
                2.0 * x + 1.0,
                2.0 * y - 1.0,
                2.0 * z + 1.0
            ]);

        for _ in 0 .. 10 {
            let (x, y, z) = (rng.gen(), rng.gen(), rng.gen());
            let point
                = DVector::from_row_slice(&[x, y, z]);

            let error: DVector<f64> = expected(x, y, z) - func.gradient_at(&point);
            assert!(error.norm() < 1E-10);
        }
    
    }

}
