use tch::{Tensor, kind};

// struct LidarSimulator {

// }

// impl LidarSimulator {
//     fn 
// }

// struct MySLAM {
//     lsim: &LidarSimulator,

//     /// Step size for scan
//     step_size: f32,

//     /// Maximum scanning range
//     max_step: f32,

//     /// Threshold of probability in (0, 1) for collision check in scan
//     col_threshold: f32,
// }

/// Return scan points given parameters
/// 
/// This function is used to compute simulated observations (distances) and 
/// log probability of observations.
/// Poses `x` is batched for particles. Its size is (n, 3), 
/// where n is the number of particles. The size of scan points, 
/// which is the return value, is (n, n_dirs, max_step, 2).
/// n_dirs is the number of scan directions: length of `dirs`. 
/// The last dimension is for x and y coordinate of the map.
/// 
/// # Arguments
/// * `xs` - Poses of the agent: (x, y, theta)
/// * `step_size` - Step size for scan
/// * `max_range` - Maximum scanning range
/// * `dirs` - Scan directions in radian
pub fn scan_points(xs: &Tensor, step_size: f64, max_range: f64, dirs: &Tensor)
    -> Tensor {
    let n = xs.size()[0];
    let n_dirs = dirs.size()[0];

    // Angles, shape is (n, n_dirs)
    let thetas = xs.slice(1, 2, 3, 1).reshape(&[-1, 1]) + 
        dirs.reshape(&[1, -1]);
    
    // Directional vectors, shape is (n, n_dirs, 2)
    let dxs = thetas.cos();
    let dys = thetas.sin();
    let dxys = Tensor::stack(&[dxs, dys], -1);

    // shape is (n, n_dirs, 1, 2) for broadcasting
    let dxys = dxys.reshape(&[n, n_dirs, 1, 2]);

    // Replicate directional vectors
    let steps = Tensor::arange2(0.0, max_range, step_size, 
                                kind::FLOAT_CPU);
    let steps = steps.reshape(&[1, 1, -1, 1]);
    let dxys = dxys * steps;

    // Reshape xs[:, :2] into (n, 1, 1, 2)
    let xs = xs.slice(1, 0, 2, 1);
    let xs = xs.reshape(&[-1, 1, 1, 2]);

    dxys + xs
}

#[cfg(test)]
mod tests {
    // TODO: is importing here ideomatic?
    use tch::{Tensor};
    use crate::scan_points;

    // This test is intended to run with `cargo test -- --nocapture`, 
    // print debug
    #[test]
    fn test_scan_points() {
        let xs = Tensor::of_slice(
            &[ 0.0, 0.0,  0.0,
              -10.0, 10.0, -0.1 as f32]
        ).reshape(&[2, 3]);
        let step_size = 0.1;
        let max_range = 0.5;
        let dirs = Tensor::of_slice(&[-0.1 as f32, 0.0, 0.1]);
        let xys = scan_points(&xs, step_size, max_range, &dirs);
        xys.print();
        assert_eq!(2 + 2, 4);
    }
}

// impl MySLAM {
//     /// Log probability of Lidar observation 
//     /// 
//     fn logp_obs(z: Tensor&, x: Tensor&) -> Tensor {
//         let n_directions = z.size()[z.dim() - 1];

//         for i in 0..n_directions {


//         }
//     }

// }


