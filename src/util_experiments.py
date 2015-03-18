import matplotlib.pylab as plt
import copy
import prepare_from_data_synthetic
import kernels
import numpy as np
import os
import prepare_from_data_synthetic
import prepare_from_data_chain

def run_experiment_sequential(completed_arguments_dict, console_log, prepare_from_data_type):
    if prepare_from_data_type == 'synthetic':
        prepare_from_data_type = prepare_from_data_synthetic
    elif prepare_from_data_type == 'chain':
        prepare_from_data_type = prepare_from_data_chain
    prepare_from_data_type.learn_predict_gpstruct_wrapper(
        console_log=console_log, 
        hp_debug=True,
        **completed_arguments_dict)
    
def run_experiment_parallel(completed_arguments_dict, console_log, prepare_from_data_type):
    if prepare_from_data_type == 'synthetic':
        prepare_from_data_type = prepare_from_data_synthetic
    elif prepare_from_data_type == 'chain':
        prepare_from_data_type = prepare_from_data_chain
    prepare_from_data_type.learn_predict_gpstruct_wrapper(
        console_log=console_log, 
        hp_debug=True,
        **completed_arguments_dict)

import time
    
def run_experiments(lbview=None, prepare_from_data_type = 'synthetic', variable_arguments_list = [], common_arguments = {}, result_prefix = '/tmp/pygpstruct_', console_log = False):
    default_common_arguments = {'n_samples' : 2001, 'prediction_verbosity':2}
    default_common_arguments.update(common_arguments)
    
    completed_arguments_list = []
    for variable_arguments_dict in variable_arguments_list:
        # define result_prefix for this experiment
        job_hash = time.time().__hash__()
        variable_arguments_dict["result_prefix"] = result_prefix + str(job_hash) + "/"

        completed_arguments_dict = copy.deepcopy(default_common_arguments)
        completed_arguments_dict.update(variable_arguments_dict)
        completed_arguments_list.append(completed_arguments_dict)
        
    if lbview != None:
        # need to pass all args in lambda cos otherwise "ValueError: sorry, can't pickle functions with closures"
        lbview.map_sync(lambda completed_arguments_dict, console_log=console_log, prepare_from_data_type=prepare_from_data_type: 
            run_experiment_parallel(completed_arguments_dict, console_log, prepare_from_data_type),
                                       completed_arguments_list)
    else:
        for completed_arguments_dict in completed_arguments_list:
            run_experiment_sequential(completed_arguments_dict, console_log, prepare_from_data_type) 
    
    # read files to obtain history_hp and history_ll
    history_list = []
    for variable_arguments_dict in variable_arguments_list:
        # determine #MCMC steps for this experiment from default and experiment arguments
        if "n_samples" in default_common_arguments:
            n_mcmc_steps = default_common_arguments["n_samples"]
        elif "n_samples" in variable_arguments_dict:
            n_mcmc_steps = variable_arguments_dict["n_samples"] # corresponding var args dict
        else:
            raise("Error: should have found n_samples either in default_common_arguments or in variable_arguments_list[i].")
        import learn_predict
        path = variable_arguments_dict["result_prefix"]
        history_ll = np.fromfile(os.path.join(path, 'results.bin'), dtype=learn_predict.dtype_for_arrays).reshape((n_mcmc_steps, 5))[:,0] # extract only LL history_hp
        history_hp = np.fromfile(os.path.join(path, 'history_hp.bin'), dtype=learn_predict.dtype_for_arrays).reshape((n_mcmc_steps, -1))
        history_list.append((history_ll, history_hp))
        # once history retrieved, delete folder
        import shutil
        #shutil.rmtree(path)

    return history_list, default_common_arguments, variable_arguments_list

def plot_experiments(history_list, default_common_arguments, variable_arguments_list, y_min=None):
        plt.figure(0, figsize=(20,7))
        ax = plt.plot(np.array([history_ll for (history_ll, __) in history_list]).T)
        # TODO remove result_prefix and kernel from legend?
        keys_to_remove =[
            'result_prefix',
            'data_indices_train',
            'data_indices_test',
            'data_folder',
            'native_implementation',
            'prediction_verbosity'
        ]
        default_common_arguments = copy.deepcopy(default_common_arguments)
        for k in keys_to_remove:
            default_common_arguments.pop(k, None)
        variable_arguments_list = copy.deepcopy(variable_arguments_list)
        for d in variable_arguments_list:
            for k in keys_to_remove:
                d.pop(k, None)
        legend = plt.legend(tuple(['%s' % str(d) for d in variable_arguments_list]), loc='lower left', 
                            title = 'Common arguments: %s' % str(default_common_arguments))
        plt.ylabel('training data log-likelihood')
        plt.xlabel('MCMC step')
        plt.ylim(bottom=y_min)
