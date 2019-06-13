extern crate tch;
use tch::{Tensor, no_grad};

#[macro_use(c)]
extern crate cute;

fn _drop_rightmost_dim(t: Tensor) -> Tensor {
    let t = t.narrow(t.ld() as i64, 0, 1);
    t.reshape(&t.size().as_slice()[..t.ld()])
}

pub trait TensorUtil {
    /// Last dimension
    fn ld(&self) -> usize;
    fn eshape(&self, i: usize) -> Vec<i64>;
}

impl TensorUtil for Tensor {
    fn ld(&self) -> usize {
        self.size().len() - 1
    }

    fn eshape(&self, i: usize) -> Vec<i64> {
        self.size()[&self.dim() - i ..].to_vec()
    }
}

pub trait Distribution {
    /// Generates a sample_shape shaped sample or sample_shape shaped batch of
    /// samples if the distribution parameters are batched.
    fn sample(&self, sample_shape: &[i64]) -> Tensor {
        no_grad(|| {
            return self.rsample(sample_shape)
        })
    }

    /// Generates a sample_shape shaped reparameterized sample or sample_shape
    /// shaped batch of reparameterized samples if the distribution parameters
    /// are batched.
    fn rsample(&self, sample_shape: &[i64]) -> Tensor;

    /// Generates n samples or n batches of samples if the distribution
    /// parameters are batched.
    fn sample_n(&self, _n: i64) -> Tensor { unimplemented!(); }

    /// Returns a new distribution instance (or populates an existing instance
    /// provided by a derived class) with batch dimensions expanded to
    /// `batch_shape`. This method calls :class:`~torch.Tensor.expand` on
    /// the distribution's parameters. As such, this does not allocate new
    /// memory for the expanded distribution instance. Additionally,
    /// this does not repeat any args checking or parameter broadcasting in
    /// `__init__.py`, when an instance is first created.
    /// 
    /// Args:
    ///     batch_shape (torch.Size): the desired expanded size.
    ///     _instance: new instance provided by subclasses that
    ///         need to override `.expand`.
    /// 
    /// Returns:
    ///     New distribution instance with batch dimensions expanded to
    ///     `batch_size`.
    fn expand(&self, batch_shape: &[i64], _instance: bool) -> Self;

    /// Returns the log of the probability density/mass function evaluated at
    /// `value`.
    ///
    /// Args:
    ///    value (Tensor):
    fn log_prob(&self, value: &Tensor) -> Tensor;

    /// Returns the cumulative density/mass function evaluated at
    /// `value`.
    ///
    /// Args:
    ///    value (Tensor):
    fn cdf(&self, _value: &Tensor) -> Tensor { unimplemented!(); }

    /// Returns the inverse cumulative density/mass function evaluated at
    /// `value`.
    ///
    ///Args:
    ///    value (Tensor):
    fn icdf(&self, _value: &Tensor) -> Tensor { unimplemented!(); }

    /// Returns tensor containing all values supported by a discrete
    /// distribution. The result will enumerate over dimension 0, so the shape
    /// of the result will be `(cardinality,) + batch_shape + event_shape`
    /// (where `event_shape = ()` for univariate distributions).
    ///
    /// Note that this enumerates over all batched tensors in lock-step
    /// `[[0, 0], [1, 1], ...]`. With `expand=False`, enumeration happens
    /// along dim 0, but with the remaining batch dimensions being
    /// singleton dimensions, `[[0], [1], ..`.
    ///
    /// To iterate over the full Cartesian product use
    /// `itertools.product(m.enumerate_support())`.
    ///
    /// Args:
    ///     expand (bool): whether to expand the support over the
    ///         batch dims to match the distribution's `batch_shape`.
    ///
    /// Returns:
    ///     Tensor iterating over dimension 0.
    fn enumerate_support(&self, _expand: bool) -> Tensor { unimplemented!(); }

    /// Returns entropy of distribution, batched over batch_shape.
    ///
    /// Returns:
    ///     Tensor of shape batch_shape.
    fn entropy(&self) -> Tensor { unimplemented!(); }

    /// Returns perplexity of distribution, batched over batch_shape.
    ///
    /// Returns:
    ///     Tensor of shape batch_shape.
    fn perplexity(&self) -> Tensor { Tensor::exp(&self.entropy()) }
}

pub mod multivariate_normal;
pub mod plotly_evcxr;
