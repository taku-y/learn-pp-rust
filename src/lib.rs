extern crate rand;
extern crate primitiv;

use std::ops;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;
use std::f32::consts::PI as PI;

//use rand::{Rng, SeedableRng, StdRng};

use primitiv::Node;
use primitiv::functions as F_;
use primitiv::node_functions as F;
use primitiv::node_functions::random as R;

pub mod utils;

pub fn const_node(v: f32) -> Node {
    F::constant([], v)
}

#[derive(Clone, Copy)]
pub enum ProcessMode {
    LOGP,
    SAMPLE,
}

#[derive(Debug)]
pub struct RvNode(Rc<Node>);

impl RvNode {
    fn new(node: Node) -> Self {
        RvNode(Rc::new(node))
    }

    fn clone(&self) -> Self {
        RvNode(Rc::clone(&self.0))
    }

    pub fn sum(&self) -> Node {
        F::batch::sum(F::sum(&self.0.as_ref(), 0))
    }

    pub fn to_float(&self) -> f32 {
        self.0.to_float()
    }
}

impl ops::Mul<Node> for RvNode {
    type Output = Node;

    fn mul(self, rhs: Node) -> Node {
        self.0.as_ref() * rhs
    }
}

impl ops::Add<RvNode> for Node {
    type Output = Node;

    fn add(self, rhs: RvNode) -> Node {
        self + rhs.0.as_ref()
    }
}

#[derive(Debug)]
pub struct RandomVarManager {
    pub samples: RefCell<HashMap<String, RvNode>>,
    pub logps: RefCell<HashMap<String, RvNode>>,
    pub namespace_logp: String,
}

impl RandomVarManager {
    pub fn new() -> Self {
        RandomVarManager {
            samples: RefCell::new(HashMap::new()),
            logps: RefCell::new(HashMap::new()),
            namespace_logp: "".to_string(),
        }
    }

    pub fn get_sample<T: Into<String>>(&self, name: T) -> RvNode {
        let name = name.into();
        RvNode::clone(self.samples.borrow().get(&name).
            expect(&format!("Sample of RV '{}' not found", name))
        )
    }

    pub fn add_sample<T: Into<String>>(&self, name: T, sample: Node) {
        self.samples.borrow_mut().insert(name.into(), RvNode::new(sample));
    }

    pub fn sum_logps(&self, prefix: &str) -> Node {
        let mut terms = vec![];

        for k in self.logps.borrow().keys().filter(|x| x.starts_with(prefix)) {
            terms.push(self.logps.borrow().get(k).unwrap().sum());
        }

        F_::sum_vars(&terms)
    }

    pub fn process<'b, T: Into<String>>(&'b self, name: T, 
        dist: &'b (Distribution + 'b), mode: ProcessMode) -> RvNode {
        let name = name.into();
        match mode {
            ProcessMode::SAMPLE => {
                let sample = dist.sample();
                self.add_sample(name.clone(), sample);
                self.get_sample(name.clone())
            }
            ProcessMode::LOGP => {
                let sample = self.get_sample(name.clone());
                let logp = dist.logp(sample.0.as_ref());
                let mut name_ = self.namespace_logp.to_owned();
                name_.push_str(&name);
                self.logps.borrow_mut().insert(name_, RvNode::new(logp));
                sample
            }
        }
    }
}

pub trait Distribution {
    fn logp(&self, sample: &Node) -> Node;
    fn sample(&self) -> Node;
}

pub struct Normal<'a> {
    mean: &'a Node,
    std: &'a Node,
}

impl<'a> Normal<'a> {
    pub fn new(mean: &'a Node, std: &'a Node) -> Self {
        Normal {
            mean: mean,
            std: std
        }
    }
}

impl<'a> Distribution for Normal<'a> {
    fn logp(&self, sample: &Node) -> Node {
        let diff = sample - self.mean;
        let diff2 = &diff * &diff;
        let logp1 = -0.5 * F::log(F::constant([], 2.0 * PI) * self.std * self.std);
        let logp2: Node = -0.5 / (self.std * self.std) * &diff2;

        logp1 + logp2
        //let logp4 = -F::log(F::constant([], 2.0 * PI) * &std) - F::constant([], 0.5) * diff2;
    }

    fn sample(&self) -> Node {
        let sample = R::normal(([], 1), 0.0, 1.0);
        F::matmul(sample, &self.std) + self.mean
    }
}
