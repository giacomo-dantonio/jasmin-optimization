use argmin::prelude::*;

use crate::steplength::LineFunc;

static C1 : f64 = 1E-4;

// Typical values of c2 are 0.9 when the search direction is chosen
// by a Netwon or quasi-Newton method, and 0.1 when pk is obtained
// from a nonlinear conjugate gradient method.
static C2 : f64 = 0.9;


struct Wolfe<'a, F, O>
where
    F :ArgminFloat,
    O: ArgminOp<Output = F, Float = F>,
    O::Param: ArgminScaledAdd<O::Param, F, O::Param>
        + ArgminDot<O::Param, F>
{
    function_value: F,
    c1: F,
    c2: F,
    slope : F, // φ'(0)
    linefunc: LineFunc<'a, O>, // φ
}

pub fn step_length<O, F>(
    op: &mut OpWrapper<O>,
    state: &IterState<O>, 
    descent_dir: &O::Param,
    initial_step_length: F
) -> Result<F, Error>
    where
        F: ArgminFloat,
        O: ArgminOp<Output = F, Float = F>,
        O::Param: ArgminScaledAdd<O::Param, F, O::Param>
            + ArgminDot<O::Param, F>
{
    let param = state.get_param();
    let gradient = state.grad
        .as_ref()
        .ok_or(Error::msg("gradient unavailable"))?;

    let line_cost_func = LineFunc::new(op, descent_dir, &param)?;
 
    let linesearch = Wolfe::new(
        line_cost_func,
        state.cost,
        F::from_f64(C1).unwrap(),
        F::from_f64(C2).unwrap(),
        gradient,
        &descent_dir,
    );

    let max = F::from_f64(10.0).unwrap();
    let step_length = linesearch.search(initial_step_length, max)?;

    Ok(step_length)
}


impl<'a, F, O> Wolfe<'a, F, O>
where
    F: ArgminFloat,
    O: ArgminOp<Output = F, Float = F>,
    O::Param: ArgminScaledAdd<O::Param, F, O::Param>
        + ArgminDot<O::Param, F>
{
    pub fn new(
        linefunc: LineFunc<'a, O>,
        function_value: F,
        c1: F,
        c2: F,
        gradient: &O::Param,
        descent_dir: &O::Param) -> Self
    {
        let slope = gradient.dot(&descent_dir);

        Wolfe {
            linefunc,
            function_value,
            c1,
            c2,
            slope,
        }
    }

    pub fn search(&self, initial: F, max_step:F) -> Result<F, Error>
    {
        let zero = F::from_f64(0.0).unwrap();
        let two = F::from_f64(2.0).unwrap();

        let mut prev_step = F::from_f64(0.0).unwrap();
        let mut prev_value = self.function_value;
        let mut step_length = initial;

        loop {
            let value = self.linefunc.apply(&step_length)?;

            if !self.sufficient_decrease(value, step_length)
            || (prev_step > zero && value >= prev_value) {
                return self.zoom(prev_step, prev_value, step_length, value);
            }

            let first_derivative = self.linefunc.gradient(&step_length)?;
            if self.curvature_condition(first_derivative) {
                return Ok(step_length);
            }

            if first_derivative >= zero {
                return self.zoom(step_length, value, prev_step, prev_value);
            }

            prev_step = step_length;
            prev_value = value;
            step_length = (step_length + max_step) / two;
        }
    }

    fn zoom(&self, param_low: F, value_low: F, param_high: F, _value_high:F) -> Result<F, Error>
    {
        let zero = F::from_f64(2.0).unwrap();
        let two = F::from_f64(2.0).unwrap();

        let mut param_low = param_low;
        let mut param_high = param_high;

        loop {
            let param_trial = (param_low + param_high) / two;
            let value_trial = self.linefunc.apply(&param_trial)?;

            if !self.sufficient_decrease(value_trial, param_trial)
            || value_trial >= value_low {
                param_high = param_trial
            }
            else {
                let derivative_trial = self.linefunc.gradient(&param_trial)?;

                if self.curvature_condition(derivative_trial) {
                    return Ok(param_trial);
                }

                if derivative_trial * (param_high - param_low) >= zero {
                    param_high = param_low;
                }
                param_low = param_trial;
            }
        }
    }

    fn sufficient_decrease(&self, value: F, step_length: F) -> bool {
        value <= self.function_value + step_length * self.c1 * self.slope
    }

    fn curvature_condition(&self, first_derivative: F) -> bool {
        first_derivative.abs() <= -self.c2 * self.slope
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::DVector;

    use super::*;
    use crate::functions::rosenbrock::Rosenbrock2D;

    #[test]
    fn test_wolfe() {
        let func = Rosenbrock2D::new(1.0, 100.0);
        let param = DVector::from_row_slice(&[-1.2, 1.0]);

        // Newton's method
        let gradient = func.gradient(&param).unwrap();
        let descent_dir = -func.hessian(&param).unwrap() * gradient.clone();

        let line_cost_func = LineFunc::new(&func, &descent_dir, &param).unwrap();
        let cost = line_cost_func.apply(&0.0).unwrap();

        let linesearch = Wolfe::new(
            line_cost_func.clone(),
            cost,
            C1,
            C2,
            &gradient,
            &descent_dir,
        );

        let step_length = linesearch.search(1.0, 10.0).unwrap();
        let value = line_cost_func.apply(&step_length).unwrap();
        let first_derivative = line_cost_func.gradient(&step_length).unwrap();

        assert!(linesearch.sufficient_decrease(value, step_length));
        assert!(linesearch.curvature_condition(first_derivative));
    }
}