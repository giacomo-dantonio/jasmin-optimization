use jasmin_optimization::linesearch::backtracking::Backtracking;
use nalgebra::{Matrix3, Vector3};
use argmin::prelude::*;

struct LineSearchFunc {
    mat_q: Matrix3<f64>,
    b: Vector3<f64>,
    c: f64,
    x: Vector3<f64>,
    gradient: Vector3<f64>,
}

impl LineSearchFunc {
    pub fn new(
        q: Matrix3<f64>,
        b: Vector3<f64>,
        c: f64,
        x: Vector3<f64>,
    ) -> Self {
        LineSearchFunc {
            mat_q: q,
            b,
            c,
            x,
            gradient: &q * x + b
        }
    }

    pub fn at(&self, at: &Vector3<f64>) -> f64 {
        0.5 * ((&self.mat_q * at).transpose() * at + &self.b.transpose() * at)[(0,0)]
        + self.c
    }
}

impl ArgminOp for LineSearchFunc {
    type Param = f64;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, _param: &Self::Param) -> Result<Self::Output, Error> {
        let x_next
            = &self.x - *_param * self.gradient;
        Ok(self.at(&x_next))
    }
}

fn main() -> Result<(), Error>{
    let cost = LineSearchFunc::new(
        Matrix3::new(
            2f64, 1f64, 0f64,
            1f64, 2f64, 0f64,
            0f64, 0f64, 1f64
        ),
        Vector3::new(0.0, 1.0, 2.0),
     0.0,
     Vector3::new(1.0, 1.5, -0.5)
    );

    let solver = Backtracking::new(
        cost.at(&cost.x),
        0.7,
        1E-4,
        cost.gradient,
        -cost.gradient,
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
