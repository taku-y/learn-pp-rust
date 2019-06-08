extern crate tch;
use tch::{kind, Tensor};
use hello::{TensorUtil};

fn main() {
    println!("1 ================");
    let t = Tensor::randn(&[5, 4, 10, 10], kind::FLOAT_CPU);
    println!("{:?}", &t.eshape(2));

    println!("2 ================");
    let t = Tensor::randn(&[4, 4], kind::FLOAT_CPU);
    t.print();
    t.diag(1).print();

    println!("3 ================");
    let t = Tensor::randn(&[4], kind::FLOAT_CPU);
    t.print();
    t.diagflat(0).print();

    println!("4 ================");
    let t = Tensor::randn(&[4, 3, 3], kind::FLOAT_CPU);
    t.slice(0, 2, 3, 1).print();

    println!("5 ================");
    let t = Tensor::randn(&[4, 3, 3], kind::FLOAT_CPU);
    t.slice(0, 2, 3, 1).squeeze().print();
}
