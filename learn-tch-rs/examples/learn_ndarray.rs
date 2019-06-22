#[macro_use]
extern crate ndarray;

use ndarray::Array;

fn main() {
    // let x = array![1.0 as f32, 2.0, 3.0, 4.0, 5.0, 6.0]
    //     .into_shape((3, 1, 2)).unwrap();
    // let y = Array::range(0.0 as f32, 10.0, 1.0)
    //     .into_shape((1, 10, 1)).unwrap();
    // let z = &x * &y;
    // println!("{}", &x);
    // println!("{}", &y);
    // println!("{}", &z);

    // The below broadcast is not supported
    let x = array![[0],[1],[2]] + array![[0, 1, 2]];
    println!("{}", &x);
}