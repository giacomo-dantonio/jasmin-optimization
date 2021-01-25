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
    // FIXME: avoid cloning the cost function and the parameter
    // FIXME: use an arbitrary descent direction instead of -gradient
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
    O : ArgminOp,
    O::Output : ArgminFloat,
    O::Param : ArgminScaledSub<O::Param, O::Output, O::Param>,
{
    type Param = O::Output;
    type Output = O::Output;
    type Hessian = ();
    type Jacobian = ();
    type Float = O::Output;

    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let x_next = self.x.scaled_sub(param, &self.gradient);
        self.func.apply(&x_next)
    }
}
