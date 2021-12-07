import jax.numpy as np
import jax.random as jr
from jax import vmap
from jax.tree_util import register_pytree_node_class

from ssm.distributions.mvn_block_tridiag import MultivariateNormalBlockTridiag
from ssm.inference.em import em
from ssm.inference.laplace_em import laplace_em
from ssm.ctlds.base import CTLDS
from ssm.ctlds.dynamics import StationaryCTDynamics, compute_transition_params
from ssm.ctlds.emissions import GaussianEmissions
from ssm.lds.initial import StandardInitialCondition
from ssm.utils import Verbosity, ensure_has_batch_dim, auto_batch, random_log_rotation

LDSPosterior = MultivariateNormalBlockTridiag


@register_pytree_node_class
class GaussianCTLDS(CTLDS):
    def __init__(self,
                 num_latent_dims: int,
                 num_emission_dims: int,
                 initial_state_mean: np.ndarray=None,
                 initial_state_scale_tril: np.ndarray=None,
                 drift_matrix: np.ndarray=None,
                 drift_bias: np.ndarray=None,
                 diffusion_scale: np.ndarray=None,
                 emission_weights: np.ndarray=None,
                 emission_bias: np.ndarray=None,
                 emission_scale_tril: np.ndarray=None,
                 seed: jr.PRNGKey=None):
        """CTLDS with Gaussian emissions.

        .. math::
            p(y_t | x_t) \sim \mathcal{N}(\mu_{x_t}, \Sigma_{x_t})

        The GaussianCTLDS can be initialized by specifying each parameter explicitly,
        or you can simply specify the ``num_latent_dims``, ``num_emission_dims``, and ``seed``
        to create a GaussianCTLDS with generic, randomly initialized parameters.

        Args:
            num_latent_dims (int): number of latent dims.
            num_emission_dims (int): number of emission dims.
            initial_state_mean (np.ndarray, optional): initial state mean. 
                Defaults to zero vector.
            initial_state_scale_tril (np.ndarray, optional): 
                initial state lower-triangular factor of covariance.
                Defaults to identity matrix.
            dynamics_weights (np.ndarray, optional): weights in dynamics GLM. 
                Defaults to a random rotation.
            dynamics_bias (np.ndarray, optional): bias in dynamics GLM. 
                Defaults to zero vector.
            dynamics_scale_tril (np.ndarray, optional): dynamics GLM lower triangular
                initial state lower-triangular factor of covariance. 
                Defaults to 0.1**2 * identity matrix.
            emission_weights (np.ndarray, optional): weights in emissions GLM. 
                Defaults to a random rotation.
            emission_bias (np.ndarray, optional): bias in emissions GLM. 
                Defaults to zero vector.
            emission_scale_tril (np.ndarray, optional): emissions GLM slower-triangular
                factor of covariance. Defaults to the identity matrix.
            seed (jr.PRNGKey, optional): random seed. Defaults to None.
        """
        # TODO: should this inherit from both CTLDS and GaussianLDS? some repeated code here.
        if initial_state_mean is None:
            initial_state_mean = np.zeros(num_latent_dims)

        if initial_state_scale_tril is None:
            initial_state_scale_tril = np.eye(num_latent_dims)

        if drift_matrix is None:
            seed, rng = jr.split(seed, 2)
            drift_matrix = random_log_rotation(rng, num_latent_dims, theta=np.pi/20)

        if drift_bias is None:
            drift_bias = np.zeros(num_latent_dims)

        if diffusion_scale is None:
            diffusion_scale = 0.1**2 * np.eye(num_latent_dims)

        if emission_weights is None:
            seed, rng = jr.split(seed, 2)
            emission_weights = jr.normal(rng, shape=(num_emission_dims, num_latent_dims))

        if emission_bias is None:
            emission_bias = np.zeros(num_emission_dims)

        if emission_scale_tril is None:
            emission_scale_tril = 1.0**2 * np.eye(num_emission_dims)  # TODO: do we want 0.1**2 here?

        initial_condition = StandardInitialCondition(initial_mean=initial_state_mean,
                                                     initial_scale_tril=initial_state_scale_tril)
        transitions = StationaryCTDynamics(drift_matrix=drift_matrix,
                                         drift_bias=drift_bias,
                                         diffusion_scale=diffusion_scale)
        emissions = GaussianEmissions(weights=emission_weights,
                                         bias=emission_bias,
                                         scale_tril=emission_scale_tril)
        super(GaussianCTLDS, self).__init__(initial_condition,
                                          transitions,
                                          emissions)

    def tree_flatten(self):
        children = (self._initial_condition,
                    self._dynamics,
                    self._emissions)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        super(cls, obj).__init__(*children)
        return obj

    @property
    def emissions_noise_covariance(self):
        R_sqrt = self._emissions.scale_tril
        return R_sqrt @ R_sqrt.T

    # Methods for inference
    @auto_batch(batched_args=("data", "covariates", "metadata"))
    def e_step(self, data, covariates=None, metadata=None) -> LDSPosterior:
        """Compute the exact posterior by extracting the natural parameters
        of the LDS, namely the block tridiagonal precision matrix (J) and
        the linear coefficient (h).

        Args:
            data (np.ndarray): the observed data of shape (B, T, D)
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.
        Returns:
            posterior (LDSPosterior): the exact posterior over the latent states.
        """
        # Shorthand names for parameters
        m1 = self.initial_mean
        Q1 = self.initial_covariance
        C = self.emissions_matrix
        d = self.emissions_bias
        R = self.emissions_noise_covariance
        seq_len = data.shape[0]

        # diagonal blocks of precision matrix
        J_diag = np.dot(C.T, np.linalg.solve(R, C))  # from observations
        J_diag = np.tile(J_diag[None, :, :], (seq_len, 1, 1))
        J_diag = J_diag.at[0].add(np.linalg.inv(Q1))

        # linear potential
        h = np.dot(data - d, np.linalg.solve(R, C))  # from observations
        h = h.at[0].add(np.linalg.solve(Q1, m1))
        
        def _energy_from_transition(covariate):
            A_t, b_t, Q_t = self._dynamics.transition_params(covariate)
            
            J_diag_curr = np.dot(A_t.T, np.linalg.solve(Q_t, A_t)) 
            J_diag_next = np.linalg.inv(Q_t) 
            J_lower_diag_curr = -np.linalg.solve(Q_t, A_t)
            
            h_curr = -np.dot(A_t.T, np.linalg.solve(Q_t, b_t))
            h_next = np.linalg.solve(Q_t, b_t)

            return J_diag_curr, J_diag_next, J_lower_diag_curr, h_curr, h_next

        energy_terms = vmap(_energy_from_transition)(covariates[1:])
        
        J_diag = J_diag.at[:-1].add(energy_terms[0])
        J_diag = J_diag.at[1:].add(energy_terms[1])
        J_lower_diag = energy_terms[2]
        h = h.at[:-1].add(energy_terms[3])
        h = h.at[1:].add(energy_terms[4])
                
        return LDSPosterior.infer(J_diag, J_lower_diag, h)

    @auto_batch(batched_args=("data", "posterior", "covariates", "metadata"))
    def marginal_likelihood(self,
                            data: np.ndarray,
                            posterior: LDSPosterior=None,
                            covariates=None,
                            metadata=None):
        r"""The exact marginal likelihood of the observed data.

        For a Gaussian LDS, we can compute the exact marginal likelihood of
        the data (y) given the posterior p(x | y) via Bayes' rule:

        .. math::
            \log p(y) = \log p(y, x) - \log p(x | y)

        This equality holds for _any_ choice of x. We'll use the posterior mean.

        Args:
            data (np.ndarray): the observed data.
            posterior (LDSPosterior, optional): the posterior distribution
                on latent states. If None, the posterior is computed via
                message passing. Defaults to None.
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.

        Returns:
            - lp (float): The marginal log likelihood of the data.
        """
        if posterior is None:
            posterior = self.e_step(data)
        states = posterior.mean()
        lps = self.log_probability(states, data, covariates) - posterior.log_prob(states)
        return lps

    @ensure_has_batch_dim()
    def fit(self,
            data: np.ndarray,
            covariates=None,
            metadata=None,
            method: str="em",
            key: jr.PRNGKey=None,
            num_iters: int=100,
            tol: float=1e-4,
            verbosity: Verbosity=Verbosity.DEBUG):
        r"""Fit the GaussianLDS to a dataset using the specified method.

        Note: because the observations are Gaussian, we can perform exact EM for a GaussianEM
        (i.e. the model is conjugate).

        Args:
            data (np.ndarray): observed data
                of shape :math:`(\text{[batch]} , \text{num\_timesteps} , \text{emissions\_dim})`
            covariates (PyTreeDef, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTreeDef, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.
            method (str, optional): model fit method. Must be one of ["em", "laplace_em"]. 
                Defaults to "em".
            key (jr.PRNGKey, optional): Random seed.
                Defaults to None.
            num_iters (int, optional): number of fit iterations.
                Defaults to 100.
            tol (float, optional): tolerance in log probability to determine convergence.
                Defaults to 1e-4.
            verbosity (Verbosity, optional): print verbosity.
                Defaults to Verbosity.DEBUG.

        Raises:
            ValueError: if fit method is not reocgnized

        Returns:
            elbos (np.ndarray): elbos at each fit iteration
            model (LDS): the fitted model
            posteriors (LDSPosterior): the fitted posteriors
        """
        model = self
        kwargs = dict(num_iters=num_iters, tol=tol, verbosity=verbosity)

        if method == "em":
            elbos, ctlds, posteriors = em(model, data, covariates, **kwargs)
        elif method == "laplace_em":
            if key is None:
                raise ValueError("Laplace EM requires a PRNGKey. Please provide an rng to fit.")
            # TODO: update laplace_em to use covariates
            raise NotImplementedError
            elbos, ctlds, posteriors = laplace_em(key, model, data, covariates, **kwargs)
        else:
            raise ValueError(f"Method {method} is not recognized/supported.")

        return elbos, ctlds, posteriors