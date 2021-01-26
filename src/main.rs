use argmin::prelude::*;
use nalgebra::{DMatrix, DVector};
use jasmin_optimization::{
    functions::quadratic::Quadratic,
    solvers::steepest_descent::SteepestDescent,
    solvers::newton::Newton
};

// TODO
// - Implement interpolation strategies for the step length search
// - Avoid duplicated code in line search methods (newton, gradient).
//   Use traits for that.
// - Create issue for ArgminDot which should not be a matrix - vector multiplication

fn main() {
    let cost = Quadratic::new(
        DMatrix::from_row_slice(3, 3, &[
            2f64, 1f64, 0f64,
            1f64, 2f64, 0f64,
            0f64, 0f64, 1f64
        ]),
        DVector::from_row_slice(&[0.0, 1.0, 2.0]),
        0.0,
    );

    let solver = SteepestDescent::new();
    // let solver = Newton::new();
    let res = Executor::new(
        cost,
        solver,
        DVector::from_row_slice(&[-20.0, 12.0, 2.71]),
    )
    .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
    .max_iters(100)
    .run()
    .unwrap();

    println!("{}", res);
}
