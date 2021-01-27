use argmin::prelude::*;

pub trait LineSearch<O, F>
where
    F: ArgminFloat,
    O: ArgminOp<Output = F, Float = F>
{
    fn descent_dir(&self, op: &mut OpWrapper<O>, state: &IterState<O>)
        -> Result<O::Param, Error>;
    
    fn step_length(&self, op: &mut OpWrapper<O>, state: &IterState<O>, descent_dir: &O::Param)
        -> Result<O::Float, Error>;
}
