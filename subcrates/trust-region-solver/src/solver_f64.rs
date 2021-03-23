use proc_macro::TokenStream;
use quote::quote;

pub fn impl_solver(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let gen = quote! {
        const MAX_DELTA : f64 = 100f64;
        const ETA : f64 = 0.125;

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

                self.delta = 0.5 * MAX_DELTA;

                Ok(Some(iter_data))
            }
        
        
            fn next_iter(
                &mut self,
                op: &mut OpWrapper<O>,
                state: &IterState<O>,
            ) -> Result<ArgminIterData<O>, Error> {
                let descent_dir = self.solve_subproblem(op, state, self.delta)?;
                let mut next_param = &state.param + &descent_dir;
                let mut next_cost = op.apply(&next_param)?;

                let subproblem_cost = self.subproblem(state, &descent_dir)?;
                let rho = (state.cost - next_cost) / (state.cost - subproblem_cost);

                // update the trust region radius
                if (rho < 0.25)
                {
                    self.delta = 0.25 * self.delta;
                }
                else
                {
                    if (rho > 0.75 && (rho.norm() - self.delta).abs() < 1E-5)
                    {
                        self.delta = MAX_DELTA.min(2.0 * self.delta);
                    }
                }

                let mut next_gradient;
                let mut next_hessian;
                if (rho <= ETA)
                {
                    next_param = state.param.clone();
                    next_cost = state.cost;
                    next_gradient = state.grad.clone();
                    next_hessian = state.hessian.clone();
                }
                else
                {
                    next_gradient = Some(op.gradient(&next_param)?);
                    next_hessian = Some(op.hessian(&next_param)?);
                }
        
                let mut iter_data = ArgminIterData::new()
                    .param(next_param)
                    .cost(next_cost);

                if let Some(gradient) = next_gradient
                {
                    iter_data = iter_data.grad(gradient);
                }
                if let Some(hessian) = next_hessian
                {
                    iter_data = iter_data.hessian(hessian);
                }

                Ok(iter_data)
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
