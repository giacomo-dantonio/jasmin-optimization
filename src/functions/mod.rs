use nalgebra::{DMatrix, DVector};
use argmin::prelude::*;

pub mod quadratic;
pub mod rosenbrock;

pub trait Function : ArgminOp<
    Param = DVector<f64>,
    Output = f64,
    Hessian = DMatrix<f64>,
    Jacobian = (),
    Float = f64
> {
    fn solve<S>(self, solver: S, param: Self::Param) -> ArgminResult<Self>
    where
        Self: ArgminOp<Float = f64, Output = f64, Param = DVector<f64>, Hessian = DMatrix<f64>>
            + Clone,
        S: Solver<Self>
    {
        Executor::new(self, solver, param)
        .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
        .max_iters(100)
        .run()
        .unwrap()
    }
}