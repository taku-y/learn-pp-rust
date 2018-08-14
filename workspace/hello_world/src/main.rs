extern crate primitiv;
extern crate hello_world;

use std::collections::HashMap;

use primitiv::Graph;
//use primitiv::Optimizer;
use primitiv::Parameter;
use primitiv::Node;

use primitiv::devices as D;
use primitiv::initializers as I;
use primitiv::node_functions as F;
use primitiv::node_functions::random as R;
// use primitiv::optimizers as O;

use std::f32::consts::PI as PI;
//use std::ops::Mul;

use hello_world::{Model, ProcessMode, Normal};

fn main() {
    println!("Hello world!");
    // Device for primitiv
    let mut dev = D::Naive::new();
    D::set_default(&mut dev);

    // Initialize parameters
    let mut p_mean = Parameter::from_initializer([], &I::Constant::new(0.01));
    let mut p_lstd = Parameter::from_initializer([], &I::Constant::new(-0.1));

    // Graph for primitiv
    let mut g = Graph::new();
    Graph::set_default(&mut g);

    // Variational distribution
    let mut guide = | model: &mut Model, mode: ProcessMode | {
        //  Parameter nodes
        let mean = F::parameter(&mut p_mean);
        let lstd = F::parameter(&mut p_lstd);
        let std = F::exp(lstd);

        let _x = model.process(
            "w", &Normal::new(mean, std), mode
        );
    };

    // Inference loop
    for _i in 0..10
    {
        g.clear();

        // Generative model
        let mut model = Model::new();
        guide(&mut model, ProcessMode::SAMPLE);
        guide(&mut model, ProcessMode::LOGP);

        println!("logp = {:?}", model.logp.to_vector());
    }
}

//fn main() {
//    // Device for primitiv
//    let mut dev = D::Naive::new();
//    D::set_default(&mut dev);
//
//    // Initialize parameters
//    let mut p_mean = Parameter::from_initializer([], &I::Constant::new(0.01));
//    let mut p_lstd = Parameter::from_initializer([], &I::Constant::new(0.01));
//
//    // Graph for primitiv
//    let mut g = Graph::new();
//    Graph::set_default(&mut g);
//
//    // Start construction of graph
//    g.clear();
//
//    //  Nodes for parameters
//    let mean = F::parameter(&mut p_mean);
//    let lstd = F::parameter(&mut p_lstd);
//
//    // samples
//    let std = F::exp(lstd);
//    let samples = R::normal(([2], 3), 0.0, 1.0);
//    let samples = F::matmul(samples, &std) + &mean;
//    let result = samples.to_vector();
//
//    // logp
//    let diff = samples - mean;
//    let diff2 = (&diff * &diff);
//
//    let logp1 = -F::log(F::constant([], 2.0 * PI) * &std);
//    let logp2: Node = -0.5 * &diff2;
//    let logp3 = &logp1 + &logp2;
//    let logp4 = -F::log(F::constant([], 2.0 * PI) * &std) - F::constant([], 0.5) * diff2;
//
//    println!("{:?}", logp1.to_vector());
//    println!("{:?}", logp2.to_vector());
//    println!("{:?}", logp3.to_vector());
//    println!("{:?}", logp4.to_vector());
//}