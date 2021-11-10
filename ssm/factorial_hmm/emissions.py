from jax import vmap
import jax.numpy as np
from jax.tree_util import register_pytree_node_class

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import ssm.distributions as ssmd
from ssm.hmm.emissions import Emissions


class FactorialEmissions(Emissions):

    def __init__(self, num_states: tuple):
        super().__init__(num_states)
        self._num_groups = len(num_states)

    @property
    def num_groups(self):
        return self._num_groups


@register_pytree_node_class
class NormalFactorialEmissions(FactorialEmissions):
    """
    x_t | \{z_{tj} \}_{j=1}^J ~ N(\sum_j m_{z_{tj}}, \sigma^2)
    """
    def __init__(self, num_states: tuple,
                 means: (tuple or list)=None,
                 log_scale: float=0.0,
                 emissions_distribution: tfd.Normal=None,
                 emissions_distribution_prior: ssmd.NormalInverseWishart=None) -> None:
        """Normal Emissions for HMM.

        Can be initialized by specifying parameters or by passing in a pre-initialized
        ``emissions_distribution`` object.

        Args:
            num_states (int): number of discrete states
            means (tuple or list, optional): state-dependent and group-dependent emission means. Defaults to None.
            variance (np.ndarray, optional): emission variance shared by all states
            emissions_distribution (ssmd.MultivariateNormalTriL, optional): initialized emissions distribution.
                Defaults to None.
            emissions_distribution_prior (ssmd.NormalInverseWishart, optional): initialized emissions distribution prior.
                Defaults to None.
        """
        super().__init__(num_states)
        if means is not None:
            big_means = np.zeros(num_states)
            for k, m in enumerate(means):
                m_expanded = np.expand_dims(m, axis=range(1, self.num_groups))
                big_means += np.swapaxes(m_expanded, 0, k)
            emissions_distribution = tfd.Normal(big_means, np.exp(log_scale))

        self._means = means
        self._distribution = emissions_distribution
        self._prior = emissions_distribution_prior

    def tree_flatten(self):
        # children = (self._distribution, self._prior)
        # aux_data = self.num_states
        children = (self._means, np.log(self._distribution.scale), self._prior)
        aux_data = self.num_states
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        means, log_scale, prior = children
        return cls(aux_data,
                   means=means, log_scale=log_scale,
                   emissions_distribution_prior=prior)

    @property
    def emissions_shape(self):
        return self._distribution.event_shape

    def distribution(self, state):
        """
        Return the conditional distribution of emission x_t
        given state z_t and (optionally) covariates u_t.
        """
        return self._distribution[state]

    def log_probs(self, data):
        """
        Compute log p(x_t | z_t=(k_1, ..., k_J)) for all t and (k_1,...,k_J).
        """
        return vmap(self._distribution.log_prob)(data)
