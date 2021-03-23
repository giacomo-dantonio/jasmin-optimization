extern crate proc_macro;

mod solver_f64;

use proc_macro::TokenStream;
use syn;

#[proc_macro_derive(Solverf64)]
pub fn solver_f64_derive(input: TokenStream) -> TokenStream {
    // Construct a representation of Rust code as a syntax tree
    // that we can manipulate
    let ast = syn::parse(input).unwrap();

    // Build the trait implementation
    solver_f64::impl_solver(&ast)
}
