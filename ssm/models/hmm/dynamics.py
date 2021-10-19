"""
HMM Dynamics Classes
====================
"""
import jax.numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import jit, tree_util, vmap
from jax.tree_util import register_pytree_node_class
from ssm.distributions import EXPFAM_DISTRIBUTIONS
from ssm.models.base import DiscreteComponent
from ssm.utils import Verbosity, ssm_pbar, sum_tuples


@register_pytree_node_class
class CategoricalDynamics(DiscreteComponent):
    def exact_m_step(self, data, posterior, prior=None):
        expfam = EXPFAM_DISTRIBUTIONS["Categorical"]
        # stats, counts = (posterior.expected_transitions,), 0

        def compute_stats_and_counts(data, posterior):
            stats, counts = (posterior.expected_transitions,), 0
            return stats, counts

        stats, counts = vmap(compute_stats_and_counts)(data, posterior)
        stats = tree_util.tree_map(sum, stats)  # sum out batch for each leaf
        counts = counts.sum(axis=0)

        if prior is not None:
            # Get stats from the prior
            prior_stats, prior_counts = \
                expfam.prior_pseudo_obs_and_counts(prior.transition_prior)
        else:
            # Default to uniform prior (0 stats, 1 counts)
            prior_stats, prior_counts = (np.ones((self.num_states, self.num_states)) + 1e-4,), 0

        stats = sum_tuples(stats, prior_stats)
        counts += prior_counts

        param_posterior = expfam.posterior_from_stats(stats, counts)
        return expfam.from_params(param_posterior.mode())
