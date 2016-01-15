import matplotlib.pylab as plt
import copy
import prepare_from_data_synthetic
import kernels
import numpy as np
import os
import prepare_from_data_synthetic
import prepare_from_data_chain

def run_experiment_sequential(completed_arguments_dict, prepare_from_data_type):
    if prepare_from_data_type == 'synthetic':
        prepare_from_data_type = prepare_from_data_synthetic
    elif prepare_from_data_type == 'chain':
        prepare_from_data_type = prepare_from_data_chain
    prepare_from_data_type.learn_predict_gpstruct_wrapper(
        **completed_arguments_dict)
    
def run_experiment_parallel(completed_arguments_dict, prepare_from_data_type):
    if prepare_from_data_type == 'synthetic':
        prepare_from_data_type = prepare_from_data_synthetic
    elif prepare_from_data_type == 'chain':
        prepare_from_data_type = prepare_from_data_chain
    completed_arguments_dict["console_log"]=False # no need for that in parallel case
    prepare_from_data_type.learn_predict_gpstruct_wrapper(
        **completed_arguments_dict)

import time
    
# TODO should always return a JSON file+ stdout recap of call jobs launched, to coordinate hashes with expt parameters
def run_experiments(lbview=None, prepare_from_data_type = 'synthetic', variable_arguments_list = [], common_arguments = {}, result_prefix = '/tmp/pygpstruct_',require_output=False):

    default_common_arguments = {'n_samples' : 2001}
    default_common_arguments.update(common_arguments)
    
    completed_arguments_list = []
    for variable_arguments_dict in variable_arguments_list:
        completed_arguments_dict = copy.deepcopy(default_common_arguments)
        completed_arguments_dict.update(variable_arguments_dict)
        if not "result_prefix" in completed_arguments_dict:
            # define result_prefix for this experiment
            job_hash = time.time().__hash__()
            completed_arguments_dict["result_prefix"] = result_prefix + str(job_hash) + "/"
        # else reuse job hash, maybe so as to hot-start

        completed_arguments_list.append(completed_arguments_dict)
    if lbview != None:
        # need to pass all args in lambda cos otherwise "ValueError: sorry, can't pickle functions with closures"
        asr = lbview.map_async(lambda completed_arguments_dict, prepare_from_data_type=prepare_from_data_type: 
            run_experiment_parallel(completed_arguments_dict, prepare_from_data_type),
                                       completed_arguments_list)
        return asr # list(zip(asr, completed_arguments_list))
    else:
        for completed_arguments_dict in completed_arguments_list:
            run_experiment_sequential(completed_arguments_dict, prepare_from_data_type) 
        print(completed_arguments_list)
    
    if require_output:
        # read files to obtain history_hp and history_ll
        history_list = []
        for completed_arguments_dict in completed_arguments_list:
            # determine #MCMC steps for this experiment from default and experiment arguments
            n_mcmc_steps = completed_arguments_dict["n_samples"]
            import learn_predict
            path = completed_arguments_dict["result_prefix"]
            history_ll = np.fromfile(os.path.join(path, 'results.bin'), dtype=util.dtype_for_arrays).reshape((n_mcmc_steps, 5))[:,0] # extract only LL history_hp
            history_hp = np.fromfile(os.path.join(path, 'history_hp.bin'), dtype=util.dtype_for_arrays).reshape((n_mcmc_steps, -1))
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


# code to plot figures from file results.txt

import numpy as np
import matplotlib.pyplot as plt
import glob

def read_data(file_pattern, data_col, max_display_length):

    data_sets = []
    #print("file_pattern: " + file_pattern )
    files = glob.glob( file_pattern )
    if (files == []):
        print("Error: no matching files for pattern %s !" % file_pattern)
    else:
        #print("matching files: " + str(files))
        pass
    for file_name in files:
        if data_col == None:  # matlab
            data_from_file = np.loadtxt(file_name)
        else: # python
            if (file_pattern.endswith(".txt")):
                data_from_file = np.loadtxt(file_name)[:,data_col]
            else:
                with open(file_name, 'rb') as f:
                    data_from_file = np.fromfile(f, dtype=np.float32)
                data_from_file = data_from_file.reshape((-1,5))[:,data_col]
        if (data_from_file.shape[0] < max_display_length):
            max_display_length = data_from_file.shape[0]
        data_sets.append(data_from_file)
    data_sets = [data_from_file[:max_display_length] for data_from_file in data_sets]
    data = np.vstack(data_sets).T
    return data
    
def plot_data(data, label, ax):#, linestyle):
    iterations_per_log_line = 1
    t = np.arange(0,data.shape[0] * iterations_per_log_line, iterations_per_log_line)
    ax.plot(t, data.mean(axis=1), lw=1, label='%s' % label)#, linestyle=linestyle)#, color='black') 
    #ax.set_xticks(np.arange(0,50000+1,1000))
    ax.xaxis.grid(True)

def make_figure(data_col_list, file_pattern_list, bottom=None, top=None, max_display_length=1e6, pdf_filename=False, title=False):
    fig, axarr = plt.subplots(len(data_col_list), 1, squeeze=False)
    fig.set_size_inches(20, len(data_col_list)*3)
    if title:
        fig.suptitle(title, fontsize=20)
    for (data_col_id, data_col) in enumerate(data_col_list): # new figure for each data type/ column
       
        #linestyles = ['-', '--', '-.', ':']
        #linestyle_index = 0
        data_bottom = np.inf
        data_top = -np.inf
        for (file_pattern_legend, file_pattern) in file_pattern_list: # new curve for each file group
            data = read_data(file_pattern, data_col, max_display_length)
            plot_data(data, file_pattern_legend, axarr[data_col_id, 0])#, linestyles[linestyle_index])
            plotted_data = data.mean(axis=1)
            data_bottom = min(plotted_data[(plotted_data.shape[0]/5):].min(), data_bottom)
            data_top = max(plotted_data[(plotted_data.shape[0]/5):].max(), data_top)
            #print(data[-1,:].mean())
            #linestyle_index += 1
    
        axarr[data_col_id, 0].set_xlabel('MCMC iterations')
        data_col_legend = {None: 'Matlab error rate', 
                           4: 'per-atom average negative log marginal',
                           3: 'test set error rate, marginalized over f''s', 
                           2: 'current LL test set',
                           1: 'current error rate on test set',
                           0: 'current LL train set'}
        axarr[data_col_id, 0].set_ylabel(data_col_legend[data_col])
        if bottom == None:
            axarr[data_col_id, 0].set_ylim(bottom=data_bottom)
        else:
            axarr[data_col_id, 0].set_ylim(bottom=bottom)
        if top == None:
            axarr[data_col_id, 0].set_ylim(top=data_top)
        else:
            axarr[data_col_id, 0].set_ylim(top=top)
            
        axarr[data_col_id, 0].legend() #loc='upper right')
        
    if pdf_filename:
        import matplotlib
        matplotlib.rcParams['pdf.fonttype'] = 42 # to avoid PDF Type 3 fonts in resulting plots, cf http://www.phyletica.com/?p=308
        fig.savefig(pdf_filename,bbox_inches='tight');