extern crate tch;
//use std::f64::consts::PI;
use tch::{kind, Tensor, manual_seed, no_grad};
use hello::{Distribution, MultivariateNormal, Scale};

fn main() {
    // let sigma = Tensor::eye(8, kind::FLOAT_CPU);
    // let sigma2 = sigma.expand(&[6, 1, 8, 8], true).contiguous();
    // sigma2.requires_grad();
    // let bx = Tensor::randn(&[8000, 6, 1, 8], kind::FLOAT_CPU);
    // let bl = sigma2;

    // let _ = _batch_mahalanobis(&bl, &bx);
    // let _ = _batch_mv(&bl, &bx);

    _test_rsample1();
    _test_log_prob();
}

fn _test_rsample1() {
    let loc = Tensor::of_slice(&[1.0f32, 2.0]);
    let scale = Tensor::of_slice(&[1.0f32, 0.0, -0.5, 2.0]).reshape(&[2, 2]);
    let dist = MultivariateNormal::new(loc, Scale::ScaleTril(scale));
    let xs = dist.rsample(&[5, 4]);

    xs.print();
}

fn _test_log_prob() {
    let loc = Tensor::of_slice(&[1.0f32, 2.0]);
    let scale = Tensor::of_slice(&[1.0f32, 0.0, -0.5, 2.0]).reshape(&[2, 2]);
    let dist = MultivariateNormal::new(loc, Scale::ScaleTril(scale));

    let s = Tensor::arange2(-8.0, 8.0, 0.05, kind::FLOAT_CPU);
    let xs = s.reshape(&[-1, 1]).ones_like().matmul(&s.reshape(&[1, -1]));
    // let ys = xs.transpose(1, 0); # it causes bug when writing it to npz?
    let ys = s.reshape(&[-1, 1]).matmul(&s.reshape(&[1, -1]).ones_like());
    let xys = Tensor::stack(&[&xs.reshape(&[-1]), &ys.reshape(&[-1])], 1);
    let lp = dist.log_prob(&xys).reshape(&xs.size());
    let f = "test_log_prob.npz";
    Tensor::write_npz(&[("xs", &xs), ("ys", &ys), ("lp", &lp)], f);
}

