use argmin::prelude::*;
use structopt::StructOpt;
use nalgebra::{DMatrix, DVector};
use jasmin_optimization::{
    functions::Function,
    functions::quadratic::Quadratic,
    functions::rosenbrock::Rosenbrock2D,
    solvers::steepest_descent::SteepestDescent,
    solvers::newton::Newton,
    solvers::newton::NewtonWithModifications,
    solvers::quasinewton::Bfgs
};

// TODO
// - Log out step length
// - Fixme allow backtracking contraction factor, for better convergence
//   (test on stepeest descent + Rosenbrock)
// - Avoid computing the Hessian in the derive macro, if it's not needed.
// - Create issue for ArgminDot which should not be a matrix - vector multiplication
// - Add benchmarks for the implemented methods
// - Test some more functions
// - Add documentation
// - Better initial step length for steepest descent

#[derive(Debug, StructOpt)]
#[structopt(name = "jasmin", about = "Some experiments with numerical optimization.")]
struct Opt {
    solver : String,
    function : String
}

macro_rules! solve {
    ($cost:expr, $solver:expr, $x0:expr) => {
        let res = if $solver == "newton" {
            $cost.solve(Newton::new(), $x0)
        }
        else if $solver == "modified" {
            $cost.solve(NewtonWithModifications::new(), $x0)
        }
        else if $solver == "bfgs" {
            let hessian = $cost.hessian(&$x0).unwrap();
            $cost.solve(Bfgs::new(&hessian).unwrap(), $x0)
        }
        else
        {
            $cost.solve(SteepestDescent::new(), $x0)
        };
        println!("{}", res);
    };
}

fn main() {
    let opt = Opt::from_args();
    println!("{:?}", opt);

    if opt.function == "rosenbrock" {
        let cost = Rosenbrock2D::new(1.0, 100.0);
        let x0 = DVector::from_row_slice(&[-1.2, 1.0]);

        solve!(cost, opt.solver, x0);
    }
    else if opt.function == "quadratic" {
        let cost = Quadratic::new(
            DMatrix::from_row_slice(3, 3, &[
                2f64, 1f64, 0f64,
                1f64, 2f64, 0f64,
                0f64, 0f64, 1f64
            ]),
            DVector::from_row_slice(&[0.0, 1.0, 2.0]),
            0.0,
        );
        let x0 = DVector::from_row_slice(&[-20.0, 12.0, 2.71]);

        solve!(cost, opt.solver, x0);
    }
}
