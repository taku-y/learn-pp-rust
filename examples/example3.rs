extern crate rand;
extern crate primitiv;
extern crate hello_world;
extern crate serde;

use std::fs::File;
use std::io::Write;

#[macro_use]
extern crate serde_derive;
extern crate serde_yaml;

use rand::distributions::Distribution;
use rand::distributions::normal::Normal as NormalInRand;

use primitiv::Graph;
use primitiv::Optimizer;
use primitiv::Parameter;
use primitiv::Node;

use primitiv::devices as D;
use primitiv::initializers as I;
use primitiv::node_functions as F;
use primitiv::optimizers as O;

use hello_world::{RandomVarManager, ProcessMode, Normal, const_node};
use hello_world::utils::{MiniBatchedDataset, MiniBatchIterator};

// Define minibatch
#[derive(Debug)]
struct MyMiniBatch {
    xs: Vec<f32>,
    ys: Vec<f32>,
}

// Define dataset
struct MyDataSet {
    pub xs_all: Vec<f32>,
    pub ys_all: Vec<f32>,
}

// Define outputs
#[derive(Serialize)]
struct AllResults {
    xs: Vec<f32>,
    ys: Vec<f32>,
    ws: Vec<f32>,
    bs: Vec<f32>,
    true_w: f32,
    true_b: f32
}

// Implement a trait for minibatch on dataset
impl MiniBatchedDataset for MyDataSet {
    type MiniBatch = MyMiniBatch;

    fn take_minibatch(&self, ixs: &Vec<usize>) -> Self::MiniBatch {
        Self::MiniBatch {
            xs: ixs.clone().into_iter().
                map(|ix| self.xs_all[ix]).collect::<Vec<_>>(),
            ys: ixs.clone().into_iter().
                map(|ix| self.ys_all[ix]).collect::<Vec<_>>(),
        }
    }

    fn len(&self) -> usize {
        self.xs_all.len() as usize
    }
}

// Create dataset
fn create_dataset(n_samples: i32, true_w: f32, true_b: f32) -> MyDataSet {
    let dist_x = NormalInRand::new(0.0, 10.0);
    let dist_e = NormalInRand::new(0.0, 20.0);
    let mut rng = rand::thread_rng();
    let xs = (0..n_samples).map(|_| dist_x.sample(&mut rng) as f32).collect::<Vec<_>>();
    let es = (0..n_samples).map(|_| dist_e.sample(&mut rng) as f32).collect::<Vec<_>>();
    let ys = xs.iter().zip(es.iter()).map(
        |(x, e)| true_w * x + true_b + e).collect::<Vec<_>>();

    MyDataSet {
        xs_all: xs,
        ys_all: ys
    }
}

// Bayesian linear regression model
fn model(rvm: &mut RandomVarManager, mode: ProcessMode, minibatch: &MyMiniBatch) {
    let xs = &minibatch.xs;
    let ys = &minibatch.ys;

    // Create Nodes for inputs
    let xs = F::input(([], xs.len() as u32), xs);
    let ys = F::input(([], ys.len() as u32), ys);

    // Set namespace when computing log probability
    rvm.namespace_logp = "model/".to_string();

    // Observation
    rvm.add_sample("y".to_string(), ys);

    // Linear regression model
    let w = rvm.process("w".to_string(), &Normal::new(&const_node(0.0), &const_node(1.0)), mode);
    let b = rvm.process("b".to_string(), &Normal::new(&const_node(0.0), &const_node(1.0)), mode);
    let ys_pred = w * xs + b;
    let _ = rvm.process("y".to_string(), &Normal::new(&ys_pred, &const_node(20.0)), mode);
}

// Variational distribution
fn vdist(rvm: &mut RandomVarManager, mode: ProcessMode, params: &[&Node; 4]) {
    let w_m = params[0];
    let w_s = params[1];
    let b_m = params[2];
    let b_s = params[3];

    // Set namespace when computing log probability
    rvm.namespace_logp = "vdist/".to_string();

    let _ = rvm.process("w".to_string(), &Normal::new(w_m, w_s), mode);
    let _ = rvm.process("b".to_string(), &Normal::new(b_m, b_s), mode);
}

fn main() -> Result<(), Box<std::error::Error>> {
    // Create sample data from a linear regression model
    let n_samples = 40;
    let true_w = 5.0 as f32;
    let true_b = -10.0 as f32;
    let dataset = create_dataset(n_samples, true_w, true_b);

    // Device for primitiv
    let mut dev = D::Naive::new();
    D::set_default(&mut dev);

    // Initialize parameters
    let mut p_w_m = Parameter::from_initializer([], &I::Constant::new(0.01));
    let mut p_w_l = Parameter::from_initializer([], &I::Constant::new(0.01));
    let mut p_b_m = Parameter::from_initializer([], &I::Constant::new(0.01));
    let mut p_b_l = Parameter::from_initializer([], &I::Constant::new(0.01));

    // Optimizer
    let mut optimizer = O::SGD::new(0.01);
    optimizer.add_parameter(&mut p_w_m);
    optimizer.add_parameter(&mut p_w_l);
    optimizer.add_parameter(&mut p_b_m);
    optimizer.add_parameter(&mut p_b_l);

    // Graph for primitiv
    let mut g = Graph::new();
    Graph::set_default(&mut g);

    // Inference loop
    for epoch in 0..100 {
        let mut loss_epoch = 0.0 as f32;

        for minibatch in MiniBatchIterator::new(20, &dataset, 0) {
            g.clear();

            // Variational parameters
            let w_m = F::parameter(&mut p_w_m);
            let w_s = F::exp(F::parameter(&mut p_w_l));
            let b_m = F::parameter(&mut p_b_m);
            let b_s = F::exp(F::parameter(&mut p_b_l));
            let params = [&w_m, &w_s, &b_m, &b_s];

            // Random variable manager
            let mut rvm = RandomVarManager::new();

            // Take samples of RVs
            vdist(&mut rvm, ProcessMode::SAMPLE, &params);

            // Compute ELBO
            vdist(&mut rvm, ProcessMode::LOGP, &params);
            model(&mut rvm, ProcessMode::LOGP, &minibatch);
            let elbo = rvm.sum_logps("model/") - rvm.sum_logps("vdist/");

            // Compute negative ELBO as loss function
            let loss: Node = -1 * elbo;

            // Do optimization 1-step
            optimizer.reset_gradients();
            loss.backward();
            optimizer.update();

            loss_epoch = loss.to_float();
        }

        if epoch % 10 == 0 {
            // let w_m = F::parameter(&mut p_w_m);
            // let w_s = F::exp(F::parameter(&mut p_w_l));
            // let b_m = F::parameter(&mut p_b_m);
            // let b_s = F::exp(F::parameter(&mut p_b_l));
            println!("epoch = {}, loss = {:?}", epoch, loss_epoch);
            // println!("w_m = {:?}, w_s = {:?}", w_m.to_float(), w_s.to_float());
            // println!("b_m = {:?}, b_s = {:?}", b_m.to_float(), b_s.to_float());
        }
    }

    // Variational parameters
    let w_m = F::parameter(&mut p_w_m);
    let w_s = F::exp(F::parameter(&mut p_w_l));
    let b_m = F::parameter(&mut p_b_m);
    let b_s = F::exp(F::parameter(&mut p_b_l));
    let params = [&w_m, &w_s, &b_m, &b_s];

    let mut ws = Vec::new();
    let mut bs = Vec::new();

    for _i in 0..50 {
        // Take sample of variational parameters
        let mut rvm = RandomVarManager::new();
        vdist(&mut rvm, ProcessMode::SAMPLE, &params);
        ws.push(rvm.get_sample(&"w".to_string()).to_float());
        bs.push(rvm.get_sample(&"b".to_string()).to_float());
    }

    // All results, output to a file
    let all_results = AllResults {
        xs: dataset.xs_all,
        ys: dataset.ys_all,
        ws,
        bs,
        true_w, 
        true_b, 
    };

    let mut file = File::create("result.yaml")?;
    let data = serde_yaml::to_string(&all_results)?;
    file.write_all(data.as_bytes())?;
    file.flush()?;
    Ok(())
}
