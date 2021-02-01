use proc_macro::TokenStream;
use quote::quote;

// FIXME: avoid duplication
pub fn impl_solver(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let gen = quote! {
        impl<O> Solver<O> for #name
        where
            O: ArgminOp<
                Output = f64,
                Float = f64,
                Param = DVector<f64>,
                Hessian = DMatrix<f64>
            >
        {
            const NAME: &'static str = stringify!(#name);

            fn init(
                &mut self,
                op: &mut OpWrapper<O>,
                state: &IterState<O>,
            ) -> Result<Option<ArgminIterData<O>>, Error> {
                // Compute initial cost, gradient and hessian and set the initial state
        
                let param = state.get_param();
                let initial_cost = op.apply(&param)?;
                let initial_grad = op.gradient(&param)?;
                let initial_hessian = op.hessian(&param)?;
        
                let iter_data = ArgminIterData::<O>::new()
                    .param(param)
                    .cost(initial_cost)
                    .grad(initial_grad)
                    .hessian(initial_hessian);
                Ok(Some(iter_data))
            }
        
        
            fn next_iter(
                &mut self,
                op: &mut OpWrapper<O>,
                state: &IterState<O>,
            ) -> Result<ArgminIterData<O>, Error> {
                let param = state.get_param();

                let descent_dir = self.descent_dir(op, state)?;
                let step_length = self.step_length(op, state, &descent_dir)?;

                let next_param = param.scaled_add(&step_length, &descent_dir);
                let next_cost = op.apply(&next_param)?;
                let next_gradient = op.gradient(&next_param)?;
                let next_hessian = op.hessian(&next_param)?;
        
                Ok(
                    ArgminIterData::new()
                    .param(next_param)
                    .cost(next_cost)
                    .grad(next_gradient)
                    .hessian(next_hessian)
                )
            }
        
            fn terminate(&mut self, state: &IterState<O>) -> TerminationReason
            {
                if let Some(grad) = state.grad.as_ref() {
                    if grad.norm() <= 1E-5 {
                        TerminationReason::TargetPrecisionReached
                    }
                    else {
                        TerminationReason::NotTerminated
                    }
                }
                else {
                    TerminationReason::Aborted
                }
            }
        }
    };
    gen.into()
}
