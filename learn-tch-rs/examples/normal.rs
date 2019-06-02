extern crate tch;
use tch::{kind, Tensor};

fn main() {
    let t = Tensor::randn(&[5, 4], kind::FLOAT_CPU);
    t.print();
}