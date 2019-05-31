extern crate tch;
use tch::{kind, Tensor};

fn main() {
    let sigma = Tensor::eye(8, kind::FLOAT_CPU);
    let sigma2 = sigma.expand(&[6, 1, 8, 8], true).contiguous();
    sigma2.requires_grad();
    let bx = Tensor::randn(&[8000, 6, 1, 8], kind::FLOAT_CPU);
    let bL = sigma2;

    _batch_mahalanobis(&bL, &bx);
}

fn _batch_mahalanobis(bL: &Tensor, bx: &Tensor) -> Tensor {
    let n = bx.size().last().unwrap().clone();
    let bx_batch_shape = &bx.size()[..(bx.size().len() - 1)];

    // Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    // we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tri.solve
    let bx_batch_dims = bx_batch_shape.len();
    let bL_batch_dims = bL.dim() - 2;
    let outer_batch_dims = bx_batch_dims - bL_batch_dims;
    let old_batch_dims = outer_batch_dims + bL_batch_dims;
    let new_batch_dims = outer_batch_dims + 2 * bL_batch_dims;

    // Reshape bx with the shape (..., 1, i, j, 1, n)
    let mut bx_new_shape = bx.size()[..outer_batch_dims].to_owned();
    for (sL, sx) in bL.size()[..bL_batch_dims].iter()
                    .zip(&bx.size()[outer_batch_dims..bx_batch_dims]) {
        bx_new_shape.extend_from_slice(&[sx / sL, sL.to_owned()]);
    }
    bx_new_shape.push(n);
    let bx = bx.reshape(bx_new_shape.as_slice());

    // Permute bx to make it have shape (..., 1, j, i, 1, n)
    let mut permute_dims = Vec::new();
    permute_dims.extend((0..outer_batch_dims).collect::<Vec<_>>());
    permute_dims.extend((outer_batch_dims..new_batch_dims).step_by(2)
                        .collect::<Vec<_>>());
    permute_dims.extend((outer_batch_dims + 1..new_batch_dims).step_by(2)
                        .collect::<Vec<_>>());
    permute_dims.push(new_batch_dims);
    let permute_dims = 
        permute_dims.iter().map(|x| *x as i64).collect::<Vec<_>>();
    let bx = bx.permute(permute_dims.as_slice());

    let flat_L = bL.reshape(&[-1, n, n]); // shape = b x n x n
    let flat_x = bx.reshape(&[-1, flat_L.size()[0], n]); // shape = c x b x n
    let flat_x_swap = flat_x.permute(&[1, 2, 0]); // shape = b x n x c
    let M_swap = flat_x_swap.triangular_solve(&flat_L, false, false, false);
    let M_swap = M_swap.0.pow(2).sum2(&[-2], false); // shape = b x c
    let M = M_swap.transpose(0, 1);

    // Now we revert the above reshape and permute operators.
    let permuted_M_shape = &bx.size()[0..(bx.size().len() - 1)];
    let permuted_M = M.reshape(permuted_M_shape); // shape = (..., 1, j, i, 1)
    let mut permute_inv_dims = (0..outer_batch_dims).collect::<Vec<_>>();
    for i in (0..bL_batch_dims) {
        permute_inv_dims.extend_from_slice(&[outer_batch_dims + i, old_batch_dims + i]);
    }
    let permute_inv_dims = 
        permute_inv_dims.iter().map(|x| *x as i64).collect::<Vec<_>>();
    let reshaped_M = permuted_M.permute(permute_inv_dims.as_slice()); // shape = (..., 1, i, j, 1)
    let reshaped_M = reshaped_M.reshape(bx_batch_shape);

    println!("n                  = {}", n);
    println!("bx_batch_shape     = {:?}", bx_batch_shape);
    println!("bL_batch_dims      = {}", bL_batch_dims);
    println!("outer_batch_dims   = {}", outer_batch_dims);
    println!("old_batch_dims     = {}", old_batch_dims);
    println!("new_batch_dims     = {}", new_batch_dims);
    println!("bx_new_shape       = {:?}", bx_new_shape);
    println!("permute_dims       = {:?}", permute_dims);
    println!("flat_L.size()      = {:?}", flat_L.size());
    println!("flat_x.size()      = {:?}", flat_x.size());
    println!("flat_x_swap.size() = {:?}", flat_x_swap.size());
    println!("M_swap.size()      = {:?}", M_swap.size());
    println!("M.size()           = {:?}", M.size());
    println!("reshaped_M.size()  = {:?}", reshaped_M.size());

    // return reshaped_M.reshape(bx_batch_shape)
    reshaped_M
}
