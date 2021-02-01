use std::f64::NEG_INFINITY;

use argmin::prelude::Error;
use nalgebra::{DMatrix, DVector};

static TOLERANCE : f64 = 1E-5;


/// Cholesky LDL factorization, for the moment without modifications
// (Algorithm 3.4 at page 53).
fn factorization(mat_a: &DMatrix<f64>, delta: f64, beta: f64) -> Result<(DMatrix<f64>, DVector<f64>), Error>
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

        let mut theta_j = f64::NEG_INFINITY;
        for i in j+1 .. rows {
            let c_ij =
                mat_a[(i,j)] - (0 .. j).map(|s| mat_d[s] * mat_l[(i,s)] * mat_l[(j,s)]).sum::<f64>();
            mat_c[(i, j)] = c_ij;

            theta_j = if c_ij > theta_j { c_ij } else { theta_j };
        }

        // d_j = max(|c_ij|, (theta_j / beta)^2, delta)
        let mut d_j =  mat_c[(j, j)].abs();
        if theta_j > f64::NEG_INFINITY && d_j < (theta_j / beta).powi(2) {
            d_j = (theta_j / beta).powi(2);
        }
        if d_j < delta {
            d_j = delta;
        }
        mat_d[j] = d_j;

        if mat_d[j] < TOLERANCE {
            return Err(Error::msg("Unable to compute the Cholesky decomposition: the input matrix is not positive definite."))
        }

        for i in j+1 .. rows {
            mat_l[(i, j)] = mat_c[(i, j)] / mat_d[j];
        }
    }

    Ok((mat_l, mat_d))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_positive_definite(mat: &DMatrix<f64>) -> bool {
        mat.clone().cholesky().is_some()
    }

    #[test]
    fn test_positive_definite() {
        let mat_a = DMatrix::from_row_slice(3, 3, &[
            2f64, 1f64, 0f64,
            1f64, 2f64, 0f64,
            0f64, 0f64, 1f64
        ]);

        let (mat_l, vec_d) = factorization(&mat_a, 1E-4, 10.0).unwrap();
        let mat_d = DMatrix::from_diagonal(&vec_d);

        let mat_l_t = mat_l.transpose();
        let actual = mat_l * mat_d * mat_l_t;
        // let actual = mat_l_t * mat_d * mat_l;
        for entry in (actual - mat_a).iter() {
            assert!(entry.abs() < TOLERANCE);
        }
    }

    #[test]
    fn test_indefinite() {
        let mat_a = DMatrix::from_row_slice(3, 3, &[
            1f64, 2f64, 0f64,
            2f64, 1f64, 0f64,
            0f64, 0f64, 1f64
        ]);
        assert!(!is_positive_definite(&mat_a));

        let (mat_l, vec_d) = factorization(&mat_a, 1E-4, 10.0).unwrap();
        let mat_d = DMatrix::from_diagonal(&vec_d);

        let mat_l_t = mat_l.transpose();
        let corrected = mat_l * mat_d * mat_l_t;
        assert!(is_positive_definite(&corrected));
    }
}