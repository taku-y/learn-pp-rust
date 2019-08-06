extern crate csv;
extern crate ndarray;
extern crate ndarray_csv;

use std::env;
use std::fs::File;
use csv::ReaderBuilder;
use ndarray::s;
use ndarray_csv::Array2Reader;
use tch::{Tensor, kind};

// https://users.rust-lang.org/t/rusts-equivalent-of-cs-system-pause/4494/3
fn pause() {
    use std::io;
    use std::io::prelude::*;

    let mut stdin = io::stdin();
    let mut stdout = io::stdout();

    // We want the cursor to stay at the end of the line, so we print without a newline and flush manually.
    write!(stdout, "Press any key to continue...").unwrap();
    stdout.flush().unwrap();

    // Read a single byte and discard
    let _ = stdin.read(&mut [0u8]).unwrap();
}


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
        let mu = Tensor::empty(&[batch_size as i64, x_points], kind::FLOAT_CPU)
            .fill_(1.0 / x_points as f64).squeeze();
        let nu = Tensor::empty(&[batch_size as i64, y_points], kind::FLOAT_CPU)
            .fill_(1.0 / y_points as f64).squeeze();
        let mut u = mu.zeros_like();
        let mut v = nu.zeros_like();

        // To check if algorithm terminates because of threshold
        // or max iterations reached
        // let mut actual_nits = 0;
        // Stopping criterion
        let thresh = 1e-1;

        // Sinkhorn iterations
        let eps = Tensor::from(self.eps as f64);
        let log_mu = (&mu + 1e-8).log();
        let log_nu = (&nu + 1e-8).log();

        for _i in 0..self.max_iter {
            // v.print();
            // pause();
            let u1 = u.shallow_clone();
            let m = self.m(&c, &u, &v);
            u = &eps * (&log_mu - m.logsumexp(&[-1], false)) + &u;
            let m = self.m(&c, &u, &v).transpose(-2, -1);
            v = &eps * (&log_nu - m.logsumexp(&[-1], false)) + &v;
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
        println!("Moon data");
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
        println!("Simple data");
        xs = Tensor::of_slice(&[0.0 as f32, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0])
            .reshape(&[5, 2]);
        ys = Tensor::of_slice(&[0.0 as f32, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0])
            .reshape(&[5, 2]);
    }

    let d = SinkhornDistance {
        eps: 1.0,
        max_iter: 100
    };
    let (cost, _pi, _c) = d.forward_t(&xs, &ys);
    _c.print();
    _pi.print();
    println!("dist = {}", f32::from(cost));
}
