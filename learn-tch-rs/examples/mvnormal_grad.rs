extern crate tch;
use tch::{nn, kind, Tensor, manual_seed, no_grad, nn::OptimizerConfig, 
          Device, nn::Init as Init};
use hello::{Distribution, MultivariateNormal, Scale};

fn _gen_samples() -> Tensor {
    let loc = Tensor::of_slice(&[1.0f32, 2.0]);
    let scale = Tensor::of_slice(&[1.0f32, 0.0, -0.5, 2.0]).reshape(&[2, 2]);
    let dist = MultivariateNormal::new(&loc, &Scale::ScaleTril(scale));
    let xs = dist.rsample(&[5, 4]);

    xs
}

fn main() {
    let xs = _gen_samples();
    let vs = nn::VarStore::new(Device::Cpu);
    let lc = &vs.root().var("loc", &[2], Init::KaimingUniform);
    let sc = &vs.root().var("scale", &[2, 2], Init::KaimingUniform);
    let opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

    for _i in 0..10000000 {
        let p = MultivariateNormal::new(
            lc, &Scale::ScaleTril(sc.shallow_clone())
        );
        let loss = p.log_prob(&xs);
    }
}
