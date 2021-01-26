use argmin::prelude::*;

pub mod backtracking;

pub struct LineSearchFunc<O>
where
    O : ArgminOp
{
    func: O,
    descent_dir: O::Param,
    x: O::Param,
}

impl<O> LineSearchFunc<O>
where
    O : ArgminOp
{
    // FIXME: avoid cloning the cost function and the parameter
    pub fn new(func: O, descent_dir: O::Param, x: O::Param) -> Result<Self, Error> {
        Ok(LineSearchFunc {
            func,
            descent_dir,
            x
        })
    }
}

impl<O> ArgminOp for LineSearchFunc<O>
where
    O : ArgminOp,
    O::Output : ArgminFloat,
    O::Param : ArgminScaledAdd<O::Param, O::Output, O::Param>,
{
    type Param = O::Output;
    type Output = O::Output;
    type Hessian = ();
    type Jacobian = ();
    type Float = O::Output;

    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let x_next = self.x.scaled_add(param, &self.descent_dir);
        self.func.apply(&x_next)
    }
}
