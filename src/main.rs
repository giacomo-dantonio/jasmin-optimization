use core::f64;

use jasmin_optimization::linesearch::backtracking::Backtracking;
use jasmin_optimization::functions::quadratic::Quadratic;
use nalgebra::{DMatrix, DVector};
use argmin::prelude::*;

// TODO
// - make LineSearchFunc generic with bounded trait ArgminOp instead of Quadratic
// - move it to a separate module
// - move the content of the main to a test
struct LineSearchFunc {
    func: Quadratic,
    x: DVector<f64>,
    gradient: DVector<f64>,
}

impl LineSearchFunc {
    pub fn new(
        q: DMatrix<f64>,
        b: DVector<f64>,
        c: f64,
        x: DVector<f64>,
    ) -> Self {
        let func
            = Quadratic::new(q, b, c);
        let gradient
            = func.gradient_at(&x);

        LineSearchFunc {
            func,
            x,
            gradient
        }
    }

    pub fn at(&self, x: &DVector<f64>) -> f64 {
        self.func.evaluate_at(x)
    }
}

impl ArgminOp for LineSearchFunc {
    type Param = f64;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let x_next
            = &self.x - *param * &self.gradient;
        Ok(self.func.evaluate_at(&x_next))
    }
}

fn main() -> Result<(), Error>{
    let cost = LineSearchFunc::new(
        DMatrix::from_row_slice(3, 3, &[
            2f64, 1f64, 0f64,
            1f64, 2f64, 0f64,
            0f64, 0f64, 1f64
        ]),
        DVector::from_row_slice(&[0.0, 1.0, 2.0]),
     0.0,
     DVector::from_row_slice(&[1.0, 1.5, -0.5])
    );

    let solver = Backtracking::new(
        cost.at(&cost.x),
        0.7,
        1E-4,
        cost.gradient.clone(),
        -cost.gradient.clone(),
    );

    let res = Executor::new(
        cost,
        solver,
        1.0
    ).add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
    .max_iters(10)
    .run()?;

    println!("{}", res);

    Ok(())
}
