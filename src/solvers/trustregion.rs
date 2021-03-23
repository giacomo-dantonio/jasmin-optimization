use argmin::prelude::*;

pub trait TrustRegion<O, F>
where
    F: ArgminFloat,
    O: ArgminOp<Output = F, Float = F>
{
    fn solve_subproblem(&mut self, op: &mut OpWrapper<O>, state: &IterState<O>, delta: F)
        -> Result<O::Param, Error>;

    fn subproblem(&self, state: &IterState<O>, param: &O::Param)
        -> Result<F, Error>;
}
