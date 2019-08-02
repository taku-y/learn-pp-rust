extern crate csv;
extern crate ndarray;
extern crate ndarray_csv;

use std::env;
use std::fs::File;
use csv::ReaderBuilder;
use ndarray::s;
use ndarray_csv::Array2Reader;
use tch::{Tensor, kind};

struct SinkhornDistance {
    eps: f32,
    max_iter: usize
}

impl SinkhornDistance {
    pub fn forward_t(&self, xs: &Tensor, ys: &Tensor) -> (Tensor, Tensor, Tensor) {
        let c = self.cost_matrix(xs, ys);
        let x_points = xs.size()[xs.dim() - 2];
        let y_points = ys.size()[ys.dim() - 2];
        let batch_size = if xs.dim() == 2 {
            1
        } else {
            xs.size()[0]
        };

        // both marginals are fixed with equal weights
        let mu = Tensor::empty(&[batch_size as i64, x_points], kind::FLOAT_CPU);
        let nu = Tensor::empty(&[batch_size as i64, y_points], kind::FLOAT_CPU);
        let mut u = mu.zeros_like();
        let mut v = nu.zeros_like();

        // To check if algorithm terminates because of threshold
        // or max iterations reached
        // let mut actual_nits = 0;
        // Stopping criterion
        let thresh = 1e-1;

        // Sinkhorn iterations
        for _i in 0..self.max_iter {
            let u1 = u.shallow_clone();
            let m = self.m(&c, &u, &v);
            let eps = Tensor::from(self.eps as f64);
            let d = m.dim() as i64;
            u = &eps * (1e-8 * &mu).log() - m.logsumexp(&[d - 1], false) + &u;
            v = &eps * (1e-8 * &nu).log()
                - m.transpose(d - 2, d - 1).logsumexp(&[d - 1], false) * &v;
            let err = f32::from((&u - &u1).abs().sum2(&[-1], false).mean());

            // actual_nits += 1;
            if err < thresh {
                break;
            }
        }

        // Transport plan pi = diag(a)*K*diag(b)
        let pi = self.m(&c, &u, &v).exp();
        // Sinkhorn distance
        let cost = (&pi * &c).sum2(&[-2, -1], false);

        (cost, pi, c)
    }

    fn m(&self, c: &Tensor, u: &Tensor, v: &Tensor) -> Tensor {
        // Modified cost for logarithmic updates
        // $M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$
        (-c + u.unsqueeze(-1) + v.unsqueeze(-2)) / Tensor::from(self.eps as f64)
    }

    fn cost_matrix(&self, xs: &Tensor, ys: &Tensor) -> Tensor {
        // Returns the matrix of $|x_i-y_j|^p$.
        let p = 2.0;
        let x_col = xs.unsqueeze(-2);
        let y_lin = ys.unsqueeze(-3);
        let c = (x_col - y_lin).abs().pow(p).sum2(&[-1], false);
        c
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let xs;
    let ys;

    if &args[1] == "moon" {
        println!("Moon");
        // Moon data
        let file = File::open("moon.csv").expect("opening file failed");
        let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
        let data: ndarray::Array2<f32>
            = reader.deserialize_array2((30, 2)).expect("read failed");
        xs = Tensor::of_slice(&data.slice(s![..15, ..]).into_slice().unwrap())
            .reshape(&[15, 2]);
        ys = Tensor::of_slice(&data.slice(s![15.., ..]).into_slice().unwrap())
            .reshape(&[15, 2]);
    }
    else {
        // Simple data
        xs = Tensor::of_slice(&[0.0 as f32, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0])
            .reshape(&[5, 2]);
        ys = Tensor::of_slice(&[0.0 as f32, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0])
            .reshape(&[5, 2]);
    }

    // println!("{}", xs);
    let d = SinkhornDistance {
        eps: 0.1,
        max_iter: 100
    };
    let (cost, _pi, _c) = d.forward_t(&xs, &ys);
    println!("{}", f32::from(cost));
}
