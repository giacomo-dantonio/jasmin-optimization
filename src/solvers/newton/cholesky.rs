use argmin::prelude::Error;
use nalgebra::{DMatrix, DVector};

static TOLERANCE : f64 = 1E-5;


/// Cholesky LDL factorization, for the moment without modifications
// (Algorithm 3.4 at page 53).
fn factorization(mat_a: &DMatrix<f64>) -> Result<(DMatrix<f64>, DVector<f64>), Error>
{
    let (rows, cols) = mat_a.shape();

    if rows != cols {
        return Err(Error::msg("Unable to compute the Cholesky decomposition: the input matrix is not square."))
    }

    let mut mat_l = DMatrix::<f64>::identity(rows, cols);
    let mut mat_d = DVector::<f64>::repeat(rows, 0.0);

    // Temporary data.
    let mut mat_c = DMatrix::<f64>::repeat(rows, cols, 0.0);

    for j in 0 .. rows {
        mat_c[(j, j)] =
            mat_a[(j, j)] - (0 .. j).map(|s| mat_d[s] * mat_l[(j, s)].powi(2)).sum::<f64>();
        mat_d[j] = mat_c[(j, j)];

        if mat_d[j] < TOLERANCE {
            return Err(Error::msg("Unable to compute the Cholesky decomposition: the input matrix is not positive definite."))
        }

        for i in j+1 .. rows {
            mat_c[(i, j)] =
                mat_a[(i,j)] - (0 .. j).map(|s| mat_d[s] * mat_l[(i,s)] * mat_l[(j,s)]).sum::<f64>();
               
            mat_l[(i, j)] = mat_c[(i, j)] / mat_d[j];
        }
    }

    Ok((mat_l, mat_d))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_definite() {
        let mat_a = DMatrix::from_row_slice(3, 3, &[
            2f64, 1f64, 0f64,
            1f64, 2f64, 0f64,
            0f64, 0f64, 1f64
        ]);

        let (mat_l, vec_d) = factorization(&mat_a).unwrap();
        let mat_d = DMatrix::from_diagonal(&vec_d);

        let mat_l_t = mat_l.transpose();
        let actual = mat_l * mat_d * mat_l_t;
        // let actual = mat_l_t * mat_d * mat_l;
        for entry in (actual - mat_a).iter() {
            assert!(entry.abs() < TOLERANCE);
        }
    }
}