extern crate tch;
use std::f64::consts::PI;
use tch::{kind, Tensor, manual_seed, no_grad};

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

fn _batch_mv(bmat: &Tensor, bvec: &Tensor) -> Tensor {
    bmat.matmul(&bvec.unsqueeze(-1)).squeeze1(-1)
}

fn _batch_mahalanobis(bl: &Tensor, bx: &Tensor) -> Tensor {
    let n = bx.size().last().unwrap().clone();
    let bx_batch_shape = &bx.size()[..(bx.size().len() - 1)];

    // Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    // we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tri.solve
    let bx_batch_dims = bx_batch_shape.len();
    let bl_batch_dims = bl.dim() - 2;
    let outer_batch_dims = bx_batch_dims - bl_batch_dims;
    let old_batch_dims = outer_batch_dims + bl_batch_dims;
    let new_batch_dims = outer_batch_dims + 2 * bl_batch_dims;

    // Reshape bx with the shape (..., 1, i, j, 1, n)
    let mut bx_new_shape = bx.size()[..outer_batch_dims].to_owned();
    for (sl, sx) in bl.size()[..bl_batch_dims].iter()
                    .zip(&bx.size()[outer_batch_dims..bx_batch_dims]) {
        bx_new_shape.extend_from_slice(&[sx / sl, sl.to_owned()]);
    }
    bx_new_shape.push(n);
    let bx = bx.reshape(bx_new_shape.as_slice());

    // Permute bx to make it have shape (..., 1, j, i, 1, n)
    let mut permute_dims = Vec::new();
    permute_dims.extend((0..outer_batch_dims).collect::<Vec<_>>());
    permute_dims.extend((outer_batch_dims..new_batch_dims).step_by(2)
                        .collect::<Vec<_>>());
    permute_dims.extend((outer_batch_dims + 1..new_batch_dims).step_by(2)
                        .collect::<Vec<_>>());
    permute_dims.push(new_batch_dims);
    let permute_dims = 
        permute_dims.iter().map(|x| *x as i64).collect::<Vec<_>>();
    let bx = bx.permute(permute_dims.as_slice());

    let flat_l = bl.reshape(&[-1, n, n]); // shape = b x n x n
    let flat_x = bx.reshape(&[-1, flat_l.size()[0], n]); // shape = c x b x n
    let flat_x_swap = flat_x.permute(&[1, 2, 0]); // shape = b x n x c
    let m_swap = flat_x_swap.triangular_solve(&flat_l, false, false, false);
    let m_swap = m_swap.0.pow(2).sum2(&[-2], false); // shape = b x c
    let m = m_swap.transpose(0, 1);

    // Now we revert the above reshape and permute operators.
    let permuted_m_shape = &bx.size()[0..(bx.size().len() - 1)];
    let permuted_m = m.reshape(permuted_m_shape); // shape = (..., 1, j, i, 1)
    let mut permute_inv_dims = (0..outer_batch_dims).collect::<Vec<_>>();
    for i in 0..bl_batch_dims {
        permute_inv_dims.extend_from_slice(&[outer_batch_dims + i, old_batch_dims + i]);
    }
    let permute_inv_dims = 
        permute_inv_dims.iter().map(|x| *x as i64).collect::<Vec<_>>();
    let reshaped_m = permuted_m.permute(permute_inv_dims.as_slice()); // shape = (..., 1, i, j, 1)
    let reshaped_m = reshaped_m.reshape(bx_batch_shape);

    println!("n                  = {}", n);
    println!("bx_batch_shape     = {:?}", bx_batch_shape);
    println!("bL_batch_dims      = {}", bl_batch_dims);
    println!("outer_batch_dims   = {}", outer_batch_dims);
    println!("old_batch_dims     = {}", old_batch_dims);
    println!("new_batch_dims     = {}", new_batch_dims);
    println!("bx_new_shape       = {:?}", bx_new_shape);
    println!("permute_dims       = {:?}", permute_dims);
    println!("flat_L.size()      = {:?}", flat_l.size());
    println!("flat_x.size()      = {:?}", flat_x.size());
    println!("flat_x_swap.size() = {:?}", flat_x_swap.size());
    println!("M_swap.size()      = {:?}", m_swap.size());
    println!("M.size()           = {:?}", m.size());
    println!("reshaped_M.size()  = {:?}", reshaped_m.size());

    // return reshaped_M.reshape(bx_batch_shape)
    reshaped_m
}

enum Scale {
    ScaleTril(Tensor),
    Covariance(Tensor),
    Precision(Tensor)
}

fn _drop_rightmost_dim(t: Tensor) -> Tensor {
    let t = t.narrow(t.ld() as i64, 0, 1);
    t.reshape(&t.size().as_slice()[..t.ld()])
}

trait TensorUtil {
    /// Last dimension
    fn ld(&self) -> usize;
}

impl TensorUtil for Tensor {
    fn ld(&self) -> usize {
        self.size().len() - 1
    }
}

struct MultivariateNormal {
    _batch_shape: Vec<i64>,
    _event_shape: Vec<i64>,
    loc: Tensor,
    scale: Scale,
    _unbroadcasted_scale_tril: Tensor
}

trait Distribution {
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

impl Distribution for MultivariateNormal {
    fn expand(&self, batch_shape: &[i64], _instance: bool) -> Self {
        let mut loc_shape = batch_shape.to_vec();
        loc_shape.extend(self._event_shape.clone());
        let mut cov_shape = batch_shape.to_vec();
        cov_shape.extend(self._event_shape.clone());
        cov_shape.extend(self._event_shape.clone());
        let loc = self.loc.expand(loc_shape.as_slice(), false); // TODO: check meaning of implicit
        let _unbroadcasted_scale_tril = 1 * &self._unbroadcasted_scale_tril;
        let scale = match &self.scale {
            Scale::ScaleTril(s) => Scale::ScaleTril(s.expand(cov_shape.as_slice(), false)), // TODO: check meaning of implicit
            _ => unimplemented!()
        };
        MultivariateNormal::new(loc, scale)
    }
    // below code not ported from pytorch:
    //
    // if 'covariance_matrix' in self.__dict__:
    //     new.covariance_matrix = self.covariance_matrix.expand(cov_shape)
    // if 'scale_tril' in self.__dict__:
    //     new.scale_tril = self.scale_tril.expand(cov_shape)
    // if 'precision_matrix' in self.__dict__:
    //     new.precision_matrix = self.precision_matrix.expand(cov_shape)
    // super(MultivariateNormal, new).__init__(batch_shape,
    //                                         self.event_shape,
    //                                         validate_args=False)
    // new._validate_args = self._validate_args
    // return new

    fn rsample(&self, sample_shape: &[i64]) -> Tensor {
        let shape = self._extended_shape(sample_shape);
        let eps = Tensor::randn(shape.as_slice(), kind::FLOAT_CPU);
        // eps.print(); // for debug
        &self.loc + _batch_mv(&self._unbroadcasted_scale_tril, &eps)
    }

    fn log_prob(&self, value: &Tensor) -> Tensor {
        let diff = value - &self.loc;
        let m = _batch_mahalanobis(&self._unbroadcasted_scale_tril, &diff);
        let half_log_det = self._unbroadcasted_scale_tril
            .diagonal(0, -2, -1).log().sum2(&[-1], false);        
        -0.5 * (self._event_shape[0] as f64 * f64::ln(2.0 * PI) + m)
        - half_log_det
    }

    fn entropy(&self) -> Tensor {
        unimplemented!();
    }
}

impl MultivariateNormal {
    pub fn new(loc_: Tensor, scale_: Scale) -> Self {
        let mut loc = loc_.unsqueeze(-1);
        let scale;
        let _unbroadcasted_scale_tril;

        if let Scale::ScaleTril(s) = scale_ {
            let mut tmp = Tensor::broadcast_tensors(&[&s, &loc]);
            loc = _drop_rightmost_dim(tmp.pop().unwrap());
            scale = Scale::ScaleTril(tmp.pop().unwrap());
            _unbroadcasted_scale_tril = s;
        }
        else {
            unimplemented!();
        }

        MultivariateNormal {
            _batch_shape: loc.size()[..loc.ld()].to_vec(),
            _event_shape: [loc.size()[loc.ld()]].to_vec(),
            loc, 
            scale,
            _unbroadcasted_scale_tril
        }
    }

    /// Returns the size of the sample returned by the distribution, given
    /// a `sample_shape`. Note, that the batch and event shapes of a 
    /// distribution instance are fixed at the time of construction.
    /// If this is empty, the returned shape is upcast to (1,).
    ///
    /// # Arguments
    ///
    /// * `sample_shape` - the size of the sample to be drawn.
    fn _extended_shape(&self, sample_shape: &[i64]) -> Vec<i64> {
        let mut sample_shape = sample_shape.to_vec();
        sample_shape.extend(&self._batch_shape);
        sample_shape.extend(&self._event_shape);
        sample_shape
    }
}

fn test_multivariate_normal_sample() {
    manual_seed(0);
    let mean = Tensor::randn(&[3], kind::FLOAT_CPU);
    mean.requires_grad();
    let tmp = Tensor::randn(&[3, 10], kind::FLOAT_CPU);
    let cov = tmp.matmul(&tmp.transpose(0, 1)) / 10;
    cov.requires_grad();
    let prec = cov.inverse();
    prec.requires_grad();
    let scale_tril = cov.cholesky(false);
    scale_tril.requires_grad();
}



    // def test_multivariate_normal_sample(self):
    //     set_rng_seed(0)  # see Note [Randomized statistical tests]
    //     mean = torch.randn(3, requires_grad=True)
    //     tmp = torch.randn(3, 10)
    //     cov = (torch.matmul(tmp, tmp.t()) / tmp.size(-1)).requires_grad_()
    //     prec = cov.inverse().requires_grad_()
    //     scale_tril = torch.cholesky(cov, upper=False).requires_grad_()

    //     self._check_sampler_sampler(MultivariateNormal(mean, cov),
    //                                 scipy.stats.multivariate_normal(mean.detach().numpy(), cov.detach().numpy()),
    //                                 'MultivariateNormal(loc={}, cov={})'.format(mean, cov),
    //                                 multivariate=True)
    //     self._check_sampler_sampler(MultivariateNormal(mean, precision_matrix=prec),
    //                                 scipy.stats.multivariate_normal(mean.detach().numpy(), cov.detach().numpy()),
    //                                 'MultivariateNormal(loc={}, prec={})'.format(mean, prec),
    //                                 multivariate=True)
    //     self._check_sampler_sampler(MultivariateNormal(mean, scale_tril=scale_tril),
    //                                 scipy.stats.multivariate_normal(mean.detach().numpy(), cov.detach().numpy()),
    //                                 'MultivariateNormal(loc={}, scale_tril={})'.format(mean, scale_tril),
    //                                 multivariate=True)
