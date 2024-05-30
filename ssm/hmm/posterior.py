"""

"""
import jax.numpy as np
import jax.scipy.special as spsp
from jax import lax, value_and_grad, vmap

from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.python.internal import reparameterization


### Core message passing routines
def hmm_log_normalizer(log_initial_state_probs,
                       log_transition_matrix,
                       log_likelihoods):
    """
    Compute the marginal likelihood (i.e. log normalizer) under
    a Hidden Markov Model (HMM) with the specified natural parameters.
    The normalizer is computed via the _forward message passing_ recursion,
    .. math::
        \log \\alpha_{t+1,j} =
            \log \sum_i \exp \{ \log \\alpha_{t,i} +
                                \log p(x_t | z_t = i) +
                                \log p(z_{t+1} = j | z_t = i) \}
    where,
    .. math::
        \\alpha_{t+1} \propto p(z_{t+1} | x_{1:t}).
    Args:
        log_initial_state_probs: Log of the initial state probabilities.
            A shape ``(K,)`` array where ``K`` is the number of states.
        log_transition_matrix: Log transition matrix or matrices. The shape
            must either be ``(K, K)`` where ``K`` is the number of states
            or ``(T-1, K, K)`` where ``T`` is the length of the sequence.
            In the former case, the entry ``[i,j]`` specifies the log
            probability of transitioning from state ``i`` to state ``j``.
            In the latter case, the ``[t,i,j]`` entry gives the log probability
            of transitioning from state ``z_t = i`` to state ``z_{t+1} = j``
            for ``t=1,...,T-1``.
        log_likelihoods: Log likelihoods combined in a shape ``(T, K)``
            array where ``T`` is the length of the sequence and ``K`` is
            the number of states.  The ``[t, i]`` entry specifies the log
            likelihood of observation ``x[t]`` given state ``z[t] = i``.
    Returns:
        The log probability of the sequence, summing out the discrete states.
    """
    assert log_initial_state_probs.ndim == 1 and log_likelihoods.ndim == 2
    num_states = len(log_initial_state_probs)
    num_timesteps = len(log_likelihoods)
    assert log_likelihoods.shape[1] == num_states

    if log_transition_matrix.ndim == 2:
        # Stationary (fixed) transition probabilities
        assert log_transition_matrix.shape == (num_states, num_states)
        return _stationary_hmm_log_normalizer(
            log_initial_state_probs, log_transition_matrix, log_likelihoods
        )

    elif log_transition_matrix.ndim == 3:
        # Time-varying transition probabilities
        assert log_transition_matrix.shape == (
            num_timesteps - 1,
            num_states,
            num_states,
        )
        return _nonstationary_hmm_log_normalizer(
            log_initial_state_probs, log_transition_matrix, log_likelihoods
        )

    else:
        raise Exception("`log_transition_matrix` must be either 2d or 3d.")


def _stationary_hmm_log_normalizer(log_initial_state_probs,
                                   log_transition_matrix,
                                   log_likelihoods):
    # alpha = log p(z_t | x_{1:t-1})
    # then: 1) condition on x_t to get p(z_t | x_{1:t})
    #       2) marginalize to get p(z_{t+1} | x_{1:t})
    def marginalize(alpha, ll):
        alpha = spsp.logsumexp(alpha + ll + log_transition_matrix.T, axis=1)
        return alpha, alpha

    alpha_T, alphas = lax.scan(
        marginalize, log_initial_state_probs, log_likelihoods[:-1]
    )

    # Include the initial potentials to get log Pr(z_t | x_{1:t-1})
    # for all time steps. These are the "filtered potentials".
    filtered_potentials = np.vstack([log_initial_state_probs, alphas])

    # Account for the last timestep when computing marginal lkhd
    return spsp.logsumexp(alpha_T + log_likelihoods[-1]), filtered_potentials


def _nonstationary_hmm_log_normalizer(log_initial_state_probs,
                                      log_transition_matrices,
                                      log_likelihoods):
    def marginalize(alpha, prms):
        log_P, ll = prms
        alpha = spsp.logsumexp(alpha + ll + log_P.T, axis=1)
        return alpha, alpha

    alpha_T, alphas = lax.scan(
        marginalize,
        log_initial_state_probs,
        (log_transition_matrices, log_likelihoods[:-1]),
    )

    # Include the initial potentials to get log Pr(z_t | x_{1:t-1})
    # for all time steps. These are the "filtered potentials".
    filtered_potentials = np.vstack([log_initial_state_probs, alphas])

    # Account for the last timestep when computing marginal lkhd
    return spsp.logsumexp(alpha_T + log_likelihoods[-1]), filtered_potentials


class _HMMPosterior(tfp.distributions.Distribution):
    """
    TODO
    """
    def __init__(self,
                 log_initial_state_probs,
                 log_likelihoods,
                 log_transition_matrix,
                 log_normalizer,
                 filtered_potentials,
                 expected_initial_states,
                 expected_states,
                 expected_transitions,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="HMMPosterior",
             ) -> None:
        self._log_initial_state_probs = log_initial_state_probs
        self._log_transition_matrix = log_transition_matrix
        self._log_likelihoods = log_likelihoods
        self._log_normalizer = log_normalizer
        self._filtered_potentials = filtered_potentials
        self._expected_initial_states = expected_initial_states
        self._expected_states = expected_states
        self._expected_transitions = expected_transitions

        # We would detect the dtype dynamically but that would break vmap
        # see https://github.com/tensorflow/probability/issues/1271
        dtype = np.int32
        super(_HMMPosterior, self).__init__(
            dtype=dtype,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            parameters=dict(log_initial_state_probs=self._log_initial_state_probs,
                            log_likelihoods=self._log_likelihoods,
                            log_transition_matrix=self._log_transition_matrix,
                            log_normalizer=self._log_normalizer,
                            filtered_potentials=self._filtered_potentials,
                            expected_initial_states=self._expected_initial_states,
                            expected_states=self._expected_states,
                            expected_transitions=self._expected_transitions),
            name=name,
        )

    @classmethod
    def infer(cls,
              log_initial_state_probs,
              log_likelihoods,
              log_transition_matrix):
        """
        Run message passing code to get the log normalizer, the filtered potentials,
        and the expected states and transitions. Then return a posterior with all
        of these parameters cached.

        Note: We assume that the given potentials are for a single time series!
        """
        # Check shapes
        assert log_likelihoods is not None and log_likelihoods.ndim == 2
        assert log_initial_state_probs is not None and log_initial_state_probs.ndim == 1
        assert log_transition_matrix is not None and log_transition_matrix.ndim in (2, 3)

        num_timesteps, num_states = log_likelihoods.shape
        assert log_initial_state_probs.shape[-1] == num_states
        if log_transition_matrix.ndim == 2:
            assert log_transition_matrix.shape[:-2] == log_likelihoods.shape[:-2]
            assert log_transition_matrix.shape[-2:] == (num_states, num_states)
        else:
            assert log_transition_matrix.shape[:-3] == log_likelihoods.shape[:-2]
            assert log_transition_matrix.shape[-3:] == (num_timesteps - 1, num_states, num_states)

        # Since this is a natural exponential family, the expected states and transitions
        # are given by gradients of the log normalizer.
        f = value_and_grad(hmm_log_normalizer, argnums=(1, 2), has_aux=True)
        (log_normalizer, filtered_potentials), (expected_transitions, expected_states) = \
            f(log_initial_state_probs, log_transition_matrix, log_likelihoods)

        # The expected initial states are part of expected states
        expected_initial_states = expected_states[0]

        return cls(log_initial_state_probs,
                   log_likelihoods,
                   log_transition_matrix,
                   log_normalizer,
                   filtered_potentials,
                   expected_initial_states,
                   expected_states,
                   expected_transitions)

    @property
    def num_timesteps(self):
        return self._log_likelihoods.shape[-2]

    @property
    def num_states(self):
        return self._log_likelihoods.shape[-1]

    @property
    def is_stationary(self):
        """
        If the HMM is stationary (i.e. the transition matrix is the same
        at all timesteps), then we should have:
          self._log_transition_matrix.shape == (..., K, K)
        If it is nonstationary, it should be
          self._log_transition_matrix.shape == (..., T-1, K, K)
        In the former case, the log_transition_matrix has the same number
        of dimensions as the log_likelihoods; in the latter, it has one more.
        """
        return self._log_transition_matrix.ndim == self._log_likelihoods.ndim

    @property
    def log_normalizer(self):
        return self._log_normalizer

    def _mean(self):
        return self._expected_states

    @property
    def expected_initial_states(self):
        return self._expected_initial_states

    @property
    def expected_states(self):
        return self._expected_states

    @property
    def expected_transitions(self):
        return self._expected_transitions

    def _log_prob(self, data, **kwargs):
        def _log_prob_single(z, initial, unary, pair):
            T = len(z)
            lp = initial[z[0]]
            lp += unary[np.arange(T), z].sum()
            if self.is_stationary:
                lp += pair[z[:-1], z[1:]].sum()
            else:
                lp += pair[np.arange(T-1), z[:-1], z[1:]].sum()
            lp -= self.log_normalizer
            return lp

        flatten = lambda x: x.reshape((-1,) + x.shape[len(self.batch_shape):])
        lps = vmap(_log_prob_single)(flatten(data),
                                     flatten(self._log_initial_state_probs),
                                     flatten(self._log_likelihoods),
                                     flatten(self._log_transition_matrix))
        return lps.reshape(self.batch_shape)

    def _sample_n(self, n, seed=None):
        if not self.ran_message_passing:
            self.message_passing()

        # TODO make these parameters w/ lazy op?
        alphas = self._alphas

        raise NotImplementedError

    def _entropy(self):
        """
        Compute the entropy

            H[X] = -E[\log p(x)]
                 = -E[-1/2 x^T J x + x^T h - log Z(J, h)]
                 = 1/2 <J, E[x x^T] - <h, E[x]> + log Z(J, h)
        """
        # Ez = self.expected_states
        # Ezzp1 = self.expected_transitions
        # h = self._unary_potentials
        # J = self._pairwise_potentials

        # entropy = -np.sum(h * Ez, axis=(-2, -1))
        # entropy += -np.sum(J * Ezzp1, axis=(-2, -1))
        # entropy += self.log_normalizer
        # return entropy
        raise NotImplementedError


class StationaryHMMPosterior(_HMMPosterior):

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        return dict(
            log_initial_state_probs=tfp.internal.parameter_properties.ParameterProperties(event_ndims=1),
            log_likelihoods=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
            log_transition_matrix=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
            log_normalizer=tfp.internal.parameter_properties.ParameterProperties(event_ndims=0),
            filtered_potentials=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
            expected_initial_states=tfp.internal.parameter_properties.ParameterProperties(event_ndims=1),
            expected_states=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
            expected_transitions=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
        )


class NonstationaryHMMPosterior(_HMMPosterior):
    """
    TODO
    """
    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        return dict(
            log_initial_state_probs=tfp.internal.parameter_properties.ParameterProperties(event_ndims=1),
            log_likelihoods=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
            log_transition_matrix=tfp.internal.parameter_properties.ParameterProperties(event_ndims=3),
            log_normalizer=tfp.internal.parameter_properties.ParameterProperties(event_ndims=0),
            filtered_potentials=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
            expected_initial_states=tfp.internal.parameter_properties.ParameterProperties(event_ndims=1),
            expected_states=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
            expected_transitions=tfp.internal.parameter_properties.ParameterProperties(event_ndims=3),
        )

