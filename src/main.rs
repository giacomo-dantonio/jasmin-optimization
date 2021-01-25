use jasmin_optimization::{
    linesearch::backtracking::Backtracking,
    linesearch::LineSearchFunc,
    functions::quadratic::Quadratic
};
use nalgebra::{DMatrix, DVector};
use argmin::prelude::*;

// TODO
// - move the content of the main to a test

fn main() -> Result<(), Error>{
    let func = Quadratic::new(
        DMatrix::from_row_slice(3, 3, &[
            2f64, 1f64, 0f64,
            1f64, 2f64, 0f64,
            0f64, 0f64, 1f64
        ]),
        DVector::from_row_slice(&[0.0, 1.0, 2.0]),
        0.0,
    );

    let x = DVector::from_row_slice(&[1.0, 1.5, -0.5]);
    let cost = LineSearchFunc::new(func.clone(), x.clone())?;
    let gradient = func.gradient_at(&x);
    let value = func.evaluate_at(&x);

    let solver = Backtracking::new(
        value,
        0.7,
        1E-4,
        gradient.clone(),
        -gradient.clone(),
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
