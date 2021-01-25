use argmin::prelude::*;

pub mod backtracking;

pub struct LineSearchFunc<O>
where
    O : ArgminOp
{
    func: O,
    gradient: O::Param,
    x: O::Param,
}

impl<O> LineSearchFunc<O>
where
    O : ArgminOp
{
    pub fn new(func: O, x: O::Param) -> Result<Self, Error> {
        let gradient = func.gradient(&x)?;

        Ok(LineSearchFunc {
            func,
            gradient,
            x
        })
    }
}

impl<O> ArgminOp for LineSearchFunc<O>
where
    O : ArgminOp<Output = f64>,
    O::Param : ArgminScaledSub<O::Param, f64, O::Param>,
{
    type Param = f64;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let x_next = self.x.scaled_sub(param, &self.gradient);
        self.func.apply(&x_next)
    }
}
