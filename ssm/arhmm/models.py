"""
Model classes for ARHMMs.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as np
import jax.random as jr
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from jax.tree_util import register_pytree_node_class

from ssm.arhmm.base import AutoregressiveHMM
from ssm.hmm.initial import StandardInitialCondition
from ssm.hmm.transitions import StationaryTransitions
from ssm.arhmm.emissions import AutoregressiveEmissions

@register_pytree_node_class
class GaussianARHMM(AutoregressiveHMM):
    def __init__(self,
                 num_states: int,
                 num_emission_dims: int=None,
                 num_lags: int=None,
                 initial_state_probs: np.ndarray=None,
                 transition_matrix: np.ndarray=None,
                 emission_weights: np.ndarray=None,
                 emission_biases: np.ndarray=None,
                 emission_covariances: np.ndarray=None,
                 seed: jr.PRNGKey=None,
                 dtype=np.float64):
        r"""Gaussian autoregressive hidden markov Model (ARHMM).
        
        Let $x_t$ denote the observation at time $t$.  Let $z_t$ denote the corresponding discrete latent state.
        The autoregressive hidden Markov model (with ..math`\text{lag}=1`) has the following likelihood,
        
        .. math::
            x_t \mid x_{t-1}, z_t \sim \mathcal{N}\left(A_{z_t} x_{t-1} + b_{z_t}, Q_{z_t} \right).
            
        The GaussianARHMM can be initialized by specifying each parameter explicitly,
        or you can simply specify the ``num_states``, ``num_emission_dims``, ``num_lags``, and ``seed``
        to create a GaussianARHMM with generic, randomly initialized parameters.

        Args:
            num_states (int): number of discrete latent states
            num_emission_dims (int, optional): number of emission dims. 
                Defaults to None.
            num_lags (int, optional): number of previous timesteps on which to autoregress. 
                Defaults to None.
            initial_state_probs (np.ndarray, optional): initial state probabilities 
                with shape :math:`(\text{num\_states},)`. Defaults to None.
            transition_matrix (np.ndarray, optional): transition matrix
                with shape :math:`(\text{num\_states}, \text{num\_states})`. Defaults to None.
            emission_weights (np.ndarray, optional): emission weights ..math`A_{z_t}` 
                with shape :math:`(\text{num\_states}, \text{emissions\_dim}, \text{emissions\_dim} * \text{num\_lags})`.
                Defaults to None.
            emission_biases (np.ndarray, optional): emission biases ..math`b_{z_t}`
                with shape :math:`(\text{num\_states}, \text{emissions\_dim})`. Defaults to None.
            emission_covariances (np.ndarray, optional): emission covariance ..math`Q_{z_t}`
                with shape :math:`(\text{num\_states}, \text{emissions\_dim}, \text{emissions\_dim})`.
                Defaults to None.
            seed (jr.PRNGKey, optional): random seed. Defaults to None.
        """

        if initial_state_probs is None:
            initial_state_probs = np.ones(num_states) / num_states

        if transition_matrix is None:
            transition_matrix = np.ones((num_states, num_states)) / num_states

        if emission_weights is None:
            assert seed is not None and num_emission_dims is not None and num_lags is not None, \
                "You must either specify the emission_weights or give a dimension, "\
                "number of lags, and seed (PRNGKey) so that they can be initialized randomly."
            this_seed, seed = jr.split(seed, 2)
            emission_weights = tfd.Normal(0, 1).sample(
                seed=this_seed,
                sample_shape=(num_states, num_emission_dims, num_emission_dims * num_lags)).astype(dtype)

        if emission_biases is None:
            assert seed is not None and num_emission_dims is not None, \
                "You must either specify the emission_weights or give a dimension, "\
                "number of lags, and seed (PRNGKey) so that they can be initialized randomly."
            this_seed, seed = jr.split(seed, 2)
            emission_biases = tfd.Normal(0, 1).sample(
                seed=this_seed,
                sample_shape=(num_states, num_emission_dims)).astype(dtype)

        if emission_covariances is None:
            assert num_emission_dims is not None, \
                "You must either specify the emission_covariances or give a dimension "\
                "so that they can be initialized."
            emission_covariances = np.tile(np.eye(num_emission_dims), (num_states, 1, 1)).astype(dtype)

        initial_condition = StandardInitialCondition(num_states, initial_probs=initial_state_probs)
        transitions = StationaryTransitions(num_states, transition_matrix=transition_matrix)
        emissions = AutoregressiveEmissions(num_states,
                                            weights=emission_weights,
                                            biases=emission_biases,
                                            covariances=emission_covariances)
        super(GaussianARHMM, self).__init__(num_states,
                                            initial_condition,
                                            transitions,
                                            emissions)

    @property
    def emission_weights(self):
        return self._emissions._distribution.weights

    @property
    def emission_biases(self):
        return self._emissions._distribution.bias

    @property
    def emission_covariances(self):
        return self._emissions._distribution.scale

    def tree_flatten(self):
        children = (self._initial_condition,
                    self._transitions,
                    self._emissions)
        aux_data = self._num_states
        return children, aux_data

    # directly GaussianHMM using parent (HMM) constructor
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        super(cls, obj).__init__(aux_data, *children)
        return obj