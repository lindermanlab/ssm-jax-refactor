from __future__ import annotations
import jax.numpy as np
import jax.scipy.special as spsp
from jax import vmap
from jax.tree_util import register_pytree_node_class

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import ssm.distributions as ssmd



class Transitions:
    """
    Base class for HMM transitions models,

    .. math::
        p_t(z_t \mid z_{t-1}, u_t)

    where u_t are optional covariates at time t.
    """
    def __init__(self, num_states: int) -> None:
        self._num_states = num_states

    @property
    def num_states(self):
        return self._num_states

    def distribution(self, state, covariates=None, metadata=None):
        """
        Return the conditional distribution of z_t given state z_{t-1}
        
        Args:
            state (int): state z_{t-1}
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.
                
        Returns:
            distribution (tfd.Distribution): conditional distribution of z_t given state z_{t-1}.
        """
        raise NotImplementedError

    def log_transition_matrices(self, data, covariates=None, metadata=None):
        r"""Returns the log probability of data where

        .. math::
            \texttt{log\_P}[i, j] = \log \Pr(z_{t+1} = j | z_t = i)

        if the transition probabilities are stationary or

        .. math::
            \texttt{log\_P}[t, i, j] = \log \Pr(z_{t+1} = j | z_t = i)

        if they are nonstationary.

        Args:
            data (np.ndarray): observed data

        Returns:
            log probs (np.ndarray): log probability as defined above

        """
        # inds = np.arange(self.num_states)
        # return vmap(lambda i: self.distribution(i, covariates=covariates, metadata=metadata).log_prob(inds))(inds)
        raise NotImplementedError

    def m_step(self, dataset, posteriors, covariates=None, metadata=None) -> Transitions:
        """Update the transition parameters in an M step given posteriors
        over the latent states. 

        Args:
            dataset (np.ndarray): the observed dataset with shape (B, T, D)
            posteriors (HMMPosterior): posteriors over the latent states with leaf shape (B, ...)
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.
                
        Returns:
            transitions (Transitions): updated transitions object
        """
        # TODO: implement generic m-step
        raise NotImplementedError


@register_pytree_node_class
class StationaryTransitions(Transitions):
    """
    Basic transition model with a fixed transition matrix.
    """
    def __init__(self,
                 num_states: int,
                 transition_matrix=None,
                 transition_distribution: ssmd.Categorical=None,
                 transition_distribution_prior: ssmd.Dirichlet=None) -> None:
        super(StationaryTransitions, self).__init__(num_states)

        assert transition_matrix is not None or transition_distribution is not None

        if transition_matrix is not None:
            self._distribution = ssmd.Categorical(logits=np.log(transition_matrix))
        else:
            self._distribution = transition_distribution

        if transition_distribution_prior is None:
            num_states = self._distribution.probs_parameter().shape[-1]
            if num_states > 1:
                transition_distribution_prior = \
                    ssmd.Dirichlet(1.1 * np.ones((num_states, num_states)))
            else:
                transition_distribution_prior = None
        self._prior = transition_distribution_prior

    def tree_flatten(self):
        children = (self._distribution, self._prior)
        aux_data = self.num_states
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        distribution, prior = children
        return cls(aux_data,
                   transition_distribution=distribution,
                   transition_distribution_prior=prior)

    @property
    def transition_matrix(self):
        return self._distribution.probs_parameter()

    def distribution(self, state, covariates=None, metadata=None):
        return self._distribution[state]

    def log_transition_matrices(self, data, covariates=None, metadata=None):
        log_P = self._distribution.logits_parameter()
        log_P -= spsp.logsumexp(log_P, axis=1, keepdims=True)
        return log_P

    def m_step(self, dataset, posteriors, covariates=None, metadata=None) -> StationaryTransitions:
        num_states = self._distribution.probs_parameter().shape[-1]
        if num_states > 1:
            #stats = np.sum(np.concatenate([posteriors[i].expected_transitions[None] for i in range(len(posteriors))]), axis=0)
            stats = np.sum(posteriors['expected_transitions'], axis=0)
            stats += self._prior.concentration
            conditional = ssmd.Categorical.compute_conditional_from_stats(stats)
            self._distribution = ssmd.Categorical.from_params(conditional.mode())
        return self


@register_pytree_node_class
class SimpleStickyTransitions(Transitions):
    """
    HMM transition model where diagonal elements are a learned
    parameter (stay_probability) and off-diagonal elements are uniform.
    That is, for a model with n hidden states, the diagonal entries
    of the transition matrix are `stay_probability`, and the off-diagonal
    entries are `(1 - stay_probability) / (n - 1)`.
    """
    def __init__(self,
                 num_states: int,
                 stay_probability: float=None,
                 transition_distribution: ssmd.Categorical=None,
                 transition_distribution_prior: ssmd.Beta=None) -> None:

        super(SimpleStickyTransitions, self).__init__(num_states)

        assert transition_distribution is not None or stay_probability is not None

        # If the transition matrix is given, recompute it given stay_probability.
        if transition_distribution is None:
            assert (stay_probability >= 0) and (stay_probability <= 1)
            self.stay_probability = np.float32(stay_probability)  # prevent weak typing
            self._distribution = ssmd.Categorical(
                logits=self._recompute_log_transition_matrix()
            )

        # Use specified transition matrix.
        else:
            self.stay_probability = transition_distribution.probs_parameter()[0, 0]
            self._distribution = transition_distribution

        # default prior, expected dwell prob = 0.9
        if transition_distribution_prior is None:
            transition_distribution_prior = ssmd.Beta(9, 1)
        self._prior = transition_distribution_prior

    def tree_flatten(self):
        children = (self.stay_probability, self._distribution, self._prior)
        aux_data = self.num_states
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        num_states = aux_data
        stay_probability, distribution, prior = children
        return cls(num_states, stay_probability,
                   transition_distribution=distribution,
                   transition_distribution_prior=prior)

    def _recompute_log_transition_matrix(self):
        T = np.full(
            (self.num_states, self.num_states),
            np.log((1 - self.stay_probability) / (self.num_states - 1))
        )
        return (
            T.at[np.diag_indices_from(T)].set(np.log(self.stay_probability))
        )

    @property
    def transition_matrix(self):
        return self._distribution.probs_parameter()
    
    def log_transition_matrices(self, data, covariates=None, metadata=None):
        log_P = self._distribution.logits_parameter()
        log_P -= spsp.logsumexp(log_P, axis=1, keepdims=True)
        return log_P

    def distribution(self, state, covariates=None, metadata=None):
        return self._distribution[state]

    def m_step(self, dataset, posteriors, covariates=None, metadata=None) -> SimpleStickyTransitions:

        # Sum over trials to compute num_states x num_states matrix
        # where i, j holds the expected number of transitions from
        # state i to state j.
        stats = np.sum(posteriors['expected_transitions'], axis=0)

        # Compute sufficient statistics of posterior.
        #  c1 = number of observed stays + pseudo-observed stays from the prior.
        #  c0 = number of observed jumps + pseudo-observed jumps from the prior.
        c1 = (
            np.sum(stats[np.diag_indices_from(stats)]) +
            self._prior.concentration1
        )
        c1_plus_c0 = (
            np.sum(stats) +
            self._prior.concentration1 +
            self._prior.concentration0
        )
        self.stay_probability = \
            ssmd.Bernoulli.compute_conditional_from_stats((c1, c1_plus_c0 - c1)).mode()

        # Recompute the log transition matrix.
        self._distribution = ssmd.Categorical(
            logits=self._recompute_log_transition_matrix()
        )

        return self