from __future__ import annotations
import jax.numpy as np
import jax.scipy.special as spsp
from jax.tree_util import register_pytree_node_class

import ssm.distributions as ssmd


class InitialCondition:
    """
    Base class for initial state distributions of an HMM.

    .. math::
        p(z_1 \mid u_t)

    where u_t are optional covariates at time t.
    """
    def __init__(self, num_states: int) -> None:
        self._num_states = num_states

    @property
    def num_states(self):
        return self._num_states

    def distribution(self, covariates=None, metadata=None):
        """
        Return the distribution of z_1.

        Args:
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.

        Returns:
            distribution (tfd.Distribution): distribution of z_1
        """
        raise NotImplementedError

    def log_initial_probs(self, data, covariates=None, metadata=None):
        """
        Return [log Pr(z_1 = k) for k in range(num_states)]
        """
        return self.distribution(covariates=covariates, metadata=metadata).log_prob(np.arange(self.num_states))

    def m_step(self, dataset, posteriors, covariates=None, metadata=None) -> InitialCondition:
        """Update the initial distribution in an M step given posteriors over the latent states.

        Args:
            dataset (np.ndarray): the observed dataset with shape (B, T, D)
            posteriors (HMMPosterior): posteriors over the latent states with leaf shape (B, ...)
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.
                
        Returns:
            initial_condition (InitialCondition): updated initial condition object
        """
        # TODO: implement generic m-step
        raise NotImplementedError


@register_pytree_node_class
class StandardInitialCondition(InitialCondition):
    """
    The standard model is a categorical distribution.
    """
    def __init__(self,
                 num_states: int,
                 initial_probs=None,
                 initial_distribution: ssmd.Categorical=None,
                 initial_distribution_prior: ssmd.Dirichlet=None) -> None:
        super(StandardInitialCondition, self).__init__(num_states)

        assert initial_probs is not None or initial_distribution is not None

        if initial_probs is not None:
            self._distribution = ssmd.Categorical(logits=np.log(initial_probs))
        else:
            self._distribution = initial_distribution
        num_states = self._distribution.probs_parameter().shape[-1]

        if initial_distribution_prior is None:
            if num_states > 1:
                initial_distribution_prior = ssmd.Dirichlet(1.1 * np.ones(num_states))
            else:
                initial_distribution_prior = None
        self._distribution_prior = initial_distribution_prior

    def tree_flatten(self):
        children = (self._distribution, self._distribution_prior)
        aux_data = self.num_states
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        distribution, prior = children
        return cls(aux_data,
                   initial_distribution=distribution,
                   initial_distribution_prior=prior)

    def distribution(self, covariates=None, metadata=None):
       return self._distribution

    def log_initial_probs(self, data, covariates=None, metadata=None):
        """
        Return [log Pr(z_1 = k) for k in range(num_states)]
        """
        lps = self._distribution.logits_parameter()
        return lps - spsp.logsumexp(lps)

    def m_step(self, dataset, posteriors, covariates=None, metadata=None) -> StandardInitialCondition:
        """Update the initial distribution in an M step given posteriors over the latent states.

        Here, an exact M-step is performed.

        Args:
            dataset (np.ndarray): the observed dataset with shape (B, T, D)
            posteriors (HMMPosterior): posteriors over the latent states with leaf shape (B, ...)
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.
                
        Returns:
            initial_condition (StandardInitialCondition): updated initial condition object
        """
        num_states = self._distribution.probs_parameter().shape[-1]
        if num_states > 1:
            #stats = np.sum(np.concatenate([posteriors[i].expected_initial_states[None] for i in range(len(posteriors))]), axis=0)
            stats = np.sum(posteriors['expected_initial_states'], axis=0)
            #stats = np.sum(posteriors.expected_states[:, 0, :], axis=0)
            stats += self._distribution_prior.concentration
            conditional = ssmd.Categorical.compute_conditional_from_stats(stats)
            self._distribution = ssmd.Categorical.from_params(conditional.mode())
        return self
