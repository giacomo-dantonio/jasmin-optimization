use argmin::prelude::*;
use nalgebra::{DMatrix, DVector};
use jasmin_optimization::{
    functions::quadratic::Quadratic,
    solvers::steepest_descent::SteepestDescent,
    solvers::newton::Newton,
    steplength::backtracking
};

// TODO
// - Log out step length
// - Better initial step length for steepest descent
// - Implement interpolation strategies for the step length search
// - Implement netwon method with modifications
// - Avoid computing the Hessian in the derive macro, if it's not needed.
// - Create issue for ArgminDot which should not be a matrix - vector multiplication
// - Add benchmarks for the implemented methods
// - Test some more functions
// - Add documentation
// - Create github issues

fn main() {
    backtracking::tests::test_cubic_interpolation();

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
