extern crate rand;
extern crate primitiv;

use rand::distributions::Distribution;

use primitiv::Graph;
use primitiv::Optimizer;
use primitiv::{Node, Parameter};

use primitiv::devices as D;
use primitiv::initializers as I;
use primitiv::node_functions as F;
use primitiv::optimizers as O;

fn create_samples(n_samples: u32, lower: &Node) -> Vec<f32> {
    use rand::distributions::normal::Normal;

    let n = Normal::new(0.0, 1.0);
    let mut rng = rand::thread_rng();
    let data = (0..(2 * n_samples))
        .map(|_| n.sample(&mut rng) as f32)
        .collect::<Vec<_>>();
    let data = F::input(([2], n_samples), &data);
    let data = F::matmul(F::transpose(lower), data);
    data.to_vector()
}

fn logp(data: &Node, lower_est: &mut Parameter) -> Node {
    let lower_est = F::parameter(lower_est);
    // let d = F::matmul(F::transpose(lower_est), data);
    let d = F::ltrs(lower_est, data);
    let d = -0.5 * F::batch::mean(F::sum(&d * &d, 0));
    // let i = F::log(F::identity(2) * &lower_est)
    // let logdet = 2 * F::sum(F::sum(i, 0), i);
    d
}

fn main() {
    // Device for primitiv
    let mut dev = D::Naive::new();
    D::set_default(&mut dev);
    let mut g = Graph::new();
    Graph::set_default(&mut g);

    // Create sample data from a 2-dimentional normal distribution
    //
    // `lower` is lower part of the cholesky decomposition of the 
    // covariance matrix of the normal distribution.
    let n_samples = 100;
    let lower = F::input([2, 2], &[1., 2., 0., 5.]);
    let data = create_samples(n_samples, &lower);

    // Initialize parameter of the covariance matrix
    // This value is estimated by optimizer
    let mut lower_est = Parameter::from_initializer(
        [2, 2], &I::Identity::new()
    );

    // Optimizer
    let mut optimizer = O::SGD::new(0.02);
    optimizer.add_parameter(&mut lower_est);
    
    // Training loop
    for i in 0..2000 {
        g.clear();

        // Optimization
        let data_ = F::input(([2], n_samples), &data);
        let loss: primitiv::Node = -1 * logp(&data_, &mut lower_est);
        let loss_val = loss.to_float();
        println!("step {},  loss: {}", i, loss_val);
        optimizer.reset_gradients();
        loss.backward();
        optimizer.update();
    }
    println!("{:#?}", F::parameter(&mut lower_est).to_vector());
}