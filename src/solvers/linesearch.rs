use argmin::prelude::*;

pub trait LineSearch<O, F>
where
    F: ArgminFloat,
    O: ArgminOp<Output = F, Float = F>
{
    fn descent_dir(&self, op: &mut OpWrapper<O>, state: &IterState<O>)
        -> Result<O::Param, Error>;
}
