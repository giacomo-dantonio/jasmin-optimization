use argmin::prelude::*;

mod interpolation;
pub mod backtracking;
pub mod wolfe;

#[derive(Clone)]
pub struct LineFunc<'a, O>
where
    O : ArgminOp
{
    func: &'a O,
    descent_dir: &'a O::Param,
    x: &'a O::Param,
}

impl<'a, O> LineFunc<'a, O>
where
    O : ArgminOp
{
    pub fn new(func: &'a O, descent_dir: &'a O::Param, x: &'a O::Param) -> Result<Self, Error> {
        Ok(LineFunc {
            func,
            descent_dir,
            x
        })
    }
}

impl<'a, O> ArgminOp for LineFunc<'a, O>
where
    O : ArgminOp,
    O::Output: ArgminFloat,
    O::Param: ArgminScaledAdd<O::Param, O::Output, O::Param>
        + ArgminDot<O::Param, O::Output>
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

    fn gradient(&self, param: &Self::Param) -> Result<Self::Param, Error> {
        let x_next = self.x.scaled_add(param, &self.descent_dir);
        
        let grad = self.func.gradient(&x_next)?;
        Ok(grad.dot(&self.descent_dir))
    }
}
