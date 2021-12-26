"""
General EM routines
"""
import warnings
import jax.numpy as np
from jax import jit, vmap
from ssm.utils import Verbosity, ensure_has_batch_dim, ssm_pbar


@ensure_has_batch_dim(model_arg="model")
def em(model,
       data,
       covariates=None,
       metadata=None,
       num_iters=100,
       tol=1e-4,
       verbosity=Verbosity.DEBUG,
    ):
    """Fit a model using EM.

    Assumes the model has the following methods for EM:

        - `model.e_step(data)` (i.e. E-step)
        - `model.m_step(dataset, posteriors)`
        - `model.marginal_likelihood(data, posterior)`

    Args:
        model (ssm.base.SSM): the model to be fit
        data (PyTree): the observed data with leaf shape (B, T, D).
        covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
            Defaults to None.
        metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
            Defaults to None.
        num_iters (int, optional): number of iterations of EM fit. Defaults to 100.
        tol (float, optional): tolerance in marginal lp to declare convergence. Defaults to 1e-4.
        verbosity (ssm.utils.Verbosity, optional): verbosity of fit. Defaults to Verbosity.DEBUG.

    Returns:
        log_probs: log probabilities across EM iterations
        model: the fitted model
        posterior: the posterior over the inferred latent states
    """

    @jit
    def update(parameters):
        with model.inject(parameters):
            posterior = model.e_step(data, covariates=covariates, metadata=metadata)
            lp = model.marginal_likelihood(data, posterior, covariates=covariates, metadata=metadata).sum()
            
            # should this return parameters?
            new_model = model.m_step(data, posterior, covariates=covariates, metadata=metadata)
            
            # either way, we can extract updated parameters here and rely on context manager
            # to reset any side effects
            parameters = new_model._parameters
            
        return parameters, posterior, lp

    # Run the EM algorithm to convergence
    log_probs = []
    pbar = ssm_pbar(num_iters, verbosity, "Iter {} LP: {:.3f}", 0, np.nan)

    if verbosity > Verbosity.OFF:
        pbar.set_description("[jit compiling...]")

    # pull parameters out of the model
    parameters = model._parameters
    
    for itr in pbar:
        parameters, posterior, lp = update(parameters)
        assert np.isfinite(lp), "NaNs in marginal log probability"

        log_probs.append(lp)
        if verbosity > Verbosity.OFF:
            pbar.set_description("LP: {:.3f}".format(lp))

        # Check for convergence
        if itr > 1:
            if log_probs[-1] < log_probs[-2]:
                pass # warnings.warn(UserWarning("LP is decreasing in EM fit!"))

            if abs(log_probs[-1] - log_probs[-2]) < tol and verbosity > Verbosity.OFF:
                pbar.set_description("[converged] LP: {:.3f}".format(lp))
                pbar.refresh()
                break

    # update the model object with our new parameters
    model._parameters = parameters

    return np.array(log_probs), model, posterior
