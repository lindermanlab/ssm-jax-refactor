import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from jax import random as jr
import flax.linen as nn
from typing import NamedTuple
from copy import deepcopy as dc

# Import some ssm stuff.
from ssm.utils import Verbosity, random_rotation, possibly_disable_jit
from ssm.lds.models import GaussianLDS
import ssm.nn_util as nn_util
import ssm.utils as utils
import ssm.inference.fivo as fivo
import ssm.inference.proposals as proposals
import ssm.inference.tilts as tilts


def gdm_define_test(key):

    proposal_structure = 'DIRECT'   # {None/'BOOTSTRAP', 'RESQ', 'DIRECT', }
    tilt_structure = 'DIRECT'     # {'DIRECT', 'NONE'/None/'BOOTSTRAP'}

    # Define the parameter names that we are going to learn.
    # This has to be a tuple of strings that index which args we will pull out.
    free_parameters = ('dynamics_bias', )

    # Define the true model.
    key, subkey = jr.split(key)
    true_model, true_states, dataset = gdm_define_true_model_and_data(subkey)

    # Now define a model to test.
    key, subkey = jax.random.split(key)
    model, get_model_params, rebuild_model_fn = gdm_define_test_model(subkey, true_model, free_parameters)

    # Define the proposal.
    key, subkey = jr.split(key)
    proposal, proposal_params, rebuild_prop_fn = gdm_define_proposal(subkey, model, dataset, proposal_structure)

    # Define the tilt.
    key, subkey = jr.split(key)
    tilt, tilt_params, rebuild_tilt_fn = gdm_define_tilt(subkey, model, dataset, tilt_structure)

    # Return this big pile of stuff.
    ret_model = (true_model, true_states, dataset)
    ret_test = (model, get_model_params, rebuild_model_fn)
    ret_prop = (proposal, proposal_params, rebuild_prop_fn)
    ret_tilt = (tilt, tilt_params, rebuild_tilt_fn)
    return ret_model, ret_test, ret_prop, ret_tilt


def gdm_define_test_model(key, true_model, free_parameters):
    """

    :param subkey:
    :param true_model:
    :return:
    """
    key, subkey = jr.split(key)

    # # Define the parameter names that we are going to learn.
    # # This has to be a tuple of strings that index which args we will pull out.
    # free_parameters = ()  # 'dynamics_weights', )

    # Close over the free parameters we have elected to learn.
    get_free_model_params_fn = lambda _model: fivo.get_model_params_fn(_model, free_parameters)

    if len(free_parameters) > 0:

        # Get the default parameters from the true model.
        true_params = fivo.get_model_params_fn(true_model)

        # Generate a model to use.  NOTE - this will generate a new model, and we will
        # overwrite any of the free parameters of interest into the true model.
        tmp_model = true_model.__class__(num_latent_dims=true_model.latent_dim,
                                         num_emission_dims=true_model.emissions_shape[0],
                                         seed=subkey)

        # Dig out the free parameters.
        init_free_params = get_free_model_params_fn(tmp_model)

        # Overwrite all the params with the new values.
        default_params = utils.mutate_named_tuple_by_key(true_params, init_free_params)

        # TODO - force mutate some of the parameters.
        # Mutate the free parameters.
        for _k in free_parameters:
            _base = getattr(default_params, _k)
            key, subkey = jr.split(key)
            new_val = {_k: _base + (2.0 * jr.normal(key=subkey, shape=_base.shape))}
            default_params = utils.mutate_named_tuple_by_key(default_params, new_val)

        # Build out a new model using these values.
        default_model = fivo.rebuild_model_fn(default_params, tmp_model)

    else:

        # If there are no free parameters then just use the true model.
        default_model = dc(true_model)

    # Close over rebuilding the model.
    rebuild_model_fn = lambda _params: fivo.rebuild_model_fn(_params, default_model)

    return default_model, get_free_model_params_fn, rebuild_model_fn


def gdm_define_tilt(subkey, model, dataset, tilt_structure):
    """

    Args:
        subkey:
        model:
        dataset:

    Returns:

    """

    if (tilt_structure is None) or (tilt_structure == 'NONE'):
        _empty_rebuild = lambda *args: None
        return None, None, _empty_rebuild

    # Tilt functions take in (dataset, model, particles, t-1).
    dummy_particles = model.initial_distribution().sample(seed=jr.PRNGKey(0), sample_shape=(2,), )
    stock_tilt_input = (dataset[-1], model, dummy_particles[0], 0)
    dummy_tilt_output = nn_util.vectorize_pytree(dataset[0][-1], )

    # Define a more conservative initialization.
    w_init = lambda *args: (0.01 * jax.nn.initializers.normal()(*args))  # TODO - 0.1 *
    b_init = lambda *args: (0.1 * jax.nn.initializers.normal()(*args))  # TODO - 0.1 *
    head_mean_fn = nn.Dense(dummy_tilt_output.shape[0], kernel_init=w_init, bias_init=b_init, use_bias=False)  # TODO - not using bias.

    # head_log_var_fn = nn.Dense(dummy_tilt_output.shape[0], kernel_init=w_init, bias_init=b_init)
    b_init = lambda *args: (1.0 + (0.1 * jax.nn.initializers.normal()(*args)))
    head_log_var_fn = nn_util.Static(dummy_tilt_output.shape[0], bias_init=b_init)

    # Check whether we have a valid number of tilts.
    n_tilts = len(dataset[0]) - 1

    # Define the proposal itself.
    tilt = tilts.IndependentGaussianTilt(n_tilts=n_tilts,
                                         tilt_input=stock_tilt_input,
                                         head_mean_fn=head_mean_fn,
                                         head_log_var_fn=head_log_var_fn)
    tilt_params = tilt.init(subkey)

    # Return a function that we can call with just the parameters as an argument to return a new closed proposal.
    rebuild_tilt_fn = tilts.rebuild_tilt(tilt, tilt_structure)
    return tilt, tilt_params, rebuild_tilt_fn


def gdm_define_proposal(subkey, model, dataset, proposal_structure):
    """

    :param subkey:
    :param model:
    :param dataset:
    :return:
    """

    if (proposal_structure is None) or (proposal_structure == 'BOOTSTRAP'):
        _empty_rebuild = lambda *args: None
        return None, None, _empty_rebuild

    # Define the proposal that we will use.
    # Stock proposal input form is (dataset, model, particles, t, p_dist, q_state).
    dummy_particles = model.initial_distribution().sample(seed=jr.PRNGKey(0), sample_shape=(2,), )
    dummy_p_dist = model.dynamics_distribution(dummy_particles)
    stock_proposal_input_without_q_state = (dataset[0], model, dummy_particles[0], 0, dummy_p_dist)
    dummy_proposal_output = nn_util.vectorize_pytree(np.ones((model.latent_dim,)), )

    # Define a more conservative initialization.
    w_init = lambda *args: (0.01 * jax.nn.initializers.normal()(*args))  # TODO - 0.1 *
    b_init = lambda *args: (0.1 * jax.nn.initializers.normal()(*args))  # TODO - 0.1 *
    head_mean_fn = nn.Dense(dummy_proposal_output.shape[0], kernel_init=w_init, bias_init=b_init, use_bias=False)  # TODO - not using bias.

    # head_log_var_fn = nn.Dense(dummy_proposal_output.shape[0], kernel_init=w_init, bias_init=b_init)
    b_init = lambda *args: (1.0 + (0.1 * jax.nn.initializers.normal()(*args)))
    head_log_var_fn = nn_util.Static(dummy_proposal_output.shape[0], bias_init=b_init)

    # Check whether we have a valid number of proposals.
    n_props = len(dataset[0])

    # Define the proposal itself.
    proposal = proposals.IndependentGaussianProposal(n_proposals=n_props,
                                                     stock_proposal_input_without_q_state=stock_proposal_input_without_q_state,
                                                     dummy_output=dummy_proposal_output,
                                                     head_mean_fn=head_mean_fn,
                                                     head_log_var_fn=head_log_var_fn, )
    proposal_params = proposal.init(subkey)

    # Return a function that we can call with just the parameters as an argument to return a new closed proposal.
    rebuild_prop_fn = proposals.rebuild_proposal(proposal, proposal_structure)
    return proposal, proposal_params, rebuild_prop_fn


def gdm_define_true_model_and_data(key):
    """

    :param key:
    :return:
    """
    latent_dim = 1
    emissions_dim = 1
    num_trials = 10000
    num_timesteps = 10

    # Create a more reasonable emission scale.
    dynamics_scale_tril = 1.0 * np.eye(latent_dim)
    emission_scale_tril = 1.0 * np.eye(emissions_dim)
    true_dynamics_weights = np.eye(latent_dim)
    true_emission_weights = np.eye(emissions_dim)

    # Create the true model.
    key, subkey = jr.split(key)

    true_model = GaussianLDS(num_latent_dims=latent_dim,
                             num_emission_dims=emissions_dim,
                             seed=subkey,
                             dynamics_scale_tril=dynamics_scale_tril,
                             dynamics_weights=true_dynamics_weights,
                             emission_weights=true_emission_weights,
                             emission_scale_tril=emission_scale_tril)

    # Sample some data.
    key, subkey = jr.split(key)
    true_states, dataset = true_model.sample(key=subkey, num_steps=num_timesteps, num_samples=num_trials)

    # For the GDM example we zero out all but the last elements.
    dataset = dataset.at[:, :-1].set(np.nan)

    return true_model, true_states, dataset


def gdm_do_plot(_param_hist, _loss_hist, _true_loss_em, _true_loss_smc, _true_params,
                param_figs):

    fsize = (12, 8)
    idx_to_str = lambda _idx: ['Model (p): ', 'Proposal (q): ', 'Tilt (r): '][_idx]

    for _p, _hist in enumerate(_param_hist):

        if _hist[0] is not None:
            if len(_hist[0]) > 0:

                n_param = len(_hist[0].keys())

                if param_figs[_p] is None:
                    param_figs[_p] = plt.subplots(n_param, 1, figsize=fsize, sharex=True, squeeze=True)

                for _i, _k in enumerate(_hist[0].keys()):
                    to_plot = []
                    for _pt in _param_hist[_p]:
                        to_plot.append(_pt[_k].flatten())
                    to_plot = np.array(to_plot)

                    if hasattr(param_figs[_p][1], '__len__'):
                        plt.sca(param_figs[_p][1][_i])
                    else:
                        plt.sca(param_figs[_p][1])
                    plt.cla()
                    plt.plot(to_plot)
                    plt.title(idx_to_str(_p) + str(_k))
                    plt.grid(True)
                    plt.tight_layout()
                    plt.pause(0.00001)

    return param_figs


def gdm_do_print(_step, true_model, opt, true_lml, pred_lml, pred_fivo_bound, em_log_marginal_likelihood=None):
    """
    Do menial print stuff.
    :param _step:
    :param pred_lml:
    :param true_model:
    :param true_lml:
    :param opt:
    :param em_log_marginal_likelihood:
    :return:
    """

    _str = 'Step: {: >5d},  True Neg-LML: {: >8.3f},  Pred Neg-LML: {: >8.3f},  Pred FIVO bound {: >8.3f}'.\
        format(_step, true_lml, pred_lml, pred_fivo_bound)
    if em_log_marginal_likelihood is not None:
        _str += '  EM Neg-LML: {: >8.3f}'.format(em_log_marginal_likelihood)

    print(_str)
    if opt[0] is not None:
        if len(opt[0].target) > 0:
            # print()
            print('\tModel')
            true_bias = true_model.dynamics_bias.flatten()
            pred_bias = opt[0].target[0].flatten()
            print('\t\tTrue: dynamics bias:     ', '  '.join(['{: >9.3f}'.format(_s) for _s in true_bias]))
            print('\t\tPred: dynamics bias:     ', '  '.join(['{: >9.3f}'.format(_s) for _s in pred_bias]))

    if opt[2] is not None:
        r_param = opt[2].target._dict['params']
        print('\tTilt:')

        r_mean_w = r_param['head_mean_fn']['kernel']
        print('\t\tR mean weight:           ', '  '.join(['{: >9.3f}'.format(_s) for _s in r_mean_w.flatten()]))

        try:
            r_mean_b = r_param['head_mean_fn']['bias']  # ADD THIS BACK IF WE ARE USING THE BIAS
            print('\t\tR mean bias       (->0): ', '  '.join(['{: >9.3f}'.format(_s) for _s in r_mean_b.flatten()]))
        except:
            pass

        try:
            r_lvar_w = r_param['head_log_var_fn']['kernel']
            print('\t\tR var(log) weight (->0): ', '  '.join(['{: >9.3f}'.format(_s) for _s in r_lvar_w.flatten()]))
        except:
            pass

        r_lvar_b = r_param['head_log_var_fn']['bias']
        print('\t\tR var bias:              ', '  '.join(['{: >9.3f}'.format(_s) for _s in r_lvar_b.flatten()]))

    if opt[1] is not None:
        q_param = opt[1].target._dict['params']
        print('\tProposal')

        q_mean_w = q_param['head_mean_fn']['kernel']
        print('\t\tQ mean weight:           ', '  '.join(['{: >9.3f}'.format(_s) for _s in q_mean_w.flatten()]))

        try:
            q_mean_b = q_param['head_mean_fn']['bias']  # ADD THIS BACK IF WE ARE USING THE BIAS
            print('\t\tQ mean bias       (->0): ', '  '.join(['{: >9.3f}'.format(_s) for _s in q_mean_b.flatten()]))
        except:
            pass

        try:
            q_lvar_w = q_param['head_log_var_fn']['kernel']
            print('\t\tQ var weight      (->0): ', '  '.join(['{: >9.3f}'.format(_s) for _s in q_lvar_w.flatten()]))
        except:
            pass

        q_lvar_b = q_param['head_log_var_fn']['bias']
        print('\t\tQ var bias:              ', '  '.join(['{: >9.3f}'.format(_s) for _s in q_lvar_b.flatten()]))

    print()
    print()
    print()
