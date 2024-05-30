"""
Base class for Autoregressive HMM.
"""
import jax.numpy as np
import jax.random as jr
from jax import vmap, lax
from jax.tree_util import register_pytree_node_class
from functools import partial

import tensorflow_probability.substrates.jax as tfp

from ssm.hmm.base import HMM
from ssm.utils import tree_get, tree_concatenate, auto_batch, tree_map

@register_pytree_node_class
class AutoregressiveHMM(HMM):
    r"""Base class for HMM with autoregressive dependencies.

    Inherits from HMM base class.
    """
    @property
    def num_lags(self):
        dist = self._emissions._distribution
        return dist.covariate_dimension // dist.data_dimension

    def emissions_distribution(self,
                               state: float,
                               covariates=None,
                               metadata=None,
                               history=None) -> tfp.distributions.Distribution:
        """
        The emissions (or observation) distribution conditioned on the current state.

        .. math::
            p(y_t | x_t)

        Args:
            state: The current state on which to condition the emissions.

        Returns:
            emissions_distribution (tfp.distributions.Distribution):
                The emissions distribution conditioned on the provided state.
        """
        return self._emissions.distribution(state, covariates=covariates, metadata=metadata, history=history)

    @auto_batch(batched_args=("states", "data", "covariates", "metadata"))
    def log_probability(self, states, data, covariates=None, metadata=None, history=None):
        r"""
        Computes the log joint probability of a set of states and data (observations).

        .. math::
            \log p(x, y) = \log p(x_1) + \sum_{t=1}^{T-1} \log p(x_{t+1} | x_t) + \sum_{t=1}^{T} \log p(y_t | x_t)

        Args:
            states: latent states :math:`x_{1:T}`
                of shape :math:`(\text{[batch]} , \text{num\_timesteps} , \text{latent\_dim})`
            data: observed data :math:`y_{1:T}`
                of shape :math:`(\text{[batch]} , \text{num\_timesteps} , \text{emissions\_dim})`
            history: previous emissions to condition on
                of shape :math:`(\text{[batch]} , \text{num\_lags} , \text{emissions\_dim})`.
                Default is None which will condition on zeros.

        Returns:
            lp: log joint probability :math:`\log p(x, y)`
                of shape :math:`(\text{batch]},)`
        """

        if history is None:
            history = np.zeros((self.num_lags, *self.emissions_shape))

        lp = 0

        # Get the first timestep probability
        initial_state = tree_get(states, 0)
        initial_data = tree_get(data, 0)
        initial_covariates = tree_get(covariates, 0)

        lp += self.initial_distribution(covariates=initial_covariates,
                                        metadata=metadata).log_prob(initial_state)
        lp += self.emissions_distribution(initial_state,
                                            covariates=initial_covariates,
                                            metadata=metadata,
                                            history=history).log_prob(initial_data)

        def _step(carry, args):
            prev_state, history, lp = carry
            state, emission, covariates = args
            lp += self.dynamics_distribution(prev_state,
                                                covariates=covariates,
                                                metadata=metadata).log_prob(state)
            lp += self.emissions_distribution(state,
                                                covariates=covariates,
                                                metadata=metadata,
                                                history=history).log_prob(emission)
            # new_history = tree_concatenate(tree_get(_history, slice(1, None)), emission[None, ...])
            new_history = np.vstack((history[1:], emission))
            return (state, new_history, lp), None

        initial_carry = (tree_get(states, 0),
                            np.vstack((history[1:], data[0])),
                            lp)
        (_, _, lp), _ = lax.scan(_step, initial_carry,
                                    (tree_get(states, slice(1, None)),
                                    tree_get(data, slice(1, None)),
                                    tree_get(covariates, slice(1, None))))
        return lp


    def sample(self,
               key,
               num_steps: int,
               initial_state=None,
               covariates=None,
               metadata=None,
               num_samples: int=1,
               history=None):
        r"""
        Sample from the joint distribution defined by the state space model.

        .. math::
            x, y \sim p(x, y)

        Args:
            key (jr.PRNGKey): A JAX pseudorandom number generator key.
            num_steps (int): Number of steps for which to sample.
            covariates: Optional covariates that may be needed for sampling.
                Default is None.
            initial_state: Optional state on which to condition the sampled trajectory.
                Default is None which samples the intial state from the initial distribution.
            num_samples (int): Number of indepedent samples (defines a batch dimension).
            prev_emissions: previous emissions to condition on
                of shape :math:`(\text{[batch]} , \text{num\_lags} , \text{emissions\_dim})`.
                Default is None which will condition on zeros.

        Returns:
            states: an array of latent states across time :math:`x_{1:T}`
                of shape :math:`(\text{[batch]} , \text{num\_timesteps} , \text{latent\_dim})`
            emissions: an array of observations across time :math:`y_{1:T}`
                of shape :math:`(\text{[batch]} , \text{num\_lags} , \text{emissions\_dim})`.
        """

        def _sample(key, covariates=None, initial_state=None, history=None):
            if initial_state is None:
                key1, key = jr.split(key, 2)
                initial_covariates = tree_get(covariates, 0)
                initial_state = self.initial_distribution(covariates=initial_covariates,
                                                          metadata=metadata).sample(seed=key1)
            if history is None:
                history = np.zeros((self.num_lags, *self.emissions_shape))
                
            key1, key = jr.split(key, 2)
            initial_emission = self.emissions_distribution(initial_state,
                                                           covariates=initial_covariates,
                                                           metadata=metadata,
                                                           history=history).sample(seed=key1)
            history = np.vstack((history[1:], initial_emission))

            def _step(carry, key_and_covariates):
                history, prev_state = carry
                key, covariates = key_and_covariates
                key1, key2 = jr.split(key, 2)
                state = self.dynamics_distribution(prev_state,
                                                   covariates=covariates,
                                                   metadata=metadata).sample(seed=key1)
                emission = self.emissions_distribution(state,
                                                       covariates=covariates,
                                                       metadata=metadata,
                                                       history=history).sample(seed=key2)
                # next_history = tree_concatenate(tree_get(history, slice(1, None)), emission)
                next_history = np.vstack((history[1:], emission))
                return (next_history, state), (state, emission)

            keys = jr.split(key, num_steps - 1)
            _, (states, emissions) = lax.scan(_step, 
                                              (history, initial_state), 
                                              (keys, tree_get(covariates, slice(1, None))))
            
            expand_dims_fn = partial(np.expand_dims, axis=0)
            states = tree_concatenate(tree_map(expand_dims_fn, initial_state), states)
            emissions = tree_concatenate(tree_map(expand_dims_fn, initial_emission), emissions)
            
            return states, emissions

        if num_samples > 1:
            batch_keys = jr.split(key, num_samples)
            states, emissions = vmap(_sample)(batch_keys, covariates, initial_state, history)
        else:
            states, emissions = _sample(key, covariates, initial_state, history)

        return states, emissions

    def __repr__(self):
        return f"<ssm.hmm.{type(self).__name__} num_states={self.num_states} " \
               f"emissions_shape={self.emissions_shape} num_lags={self.num_lags}>"
