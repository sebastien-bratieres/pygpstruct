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

# WASDOING for fear:
# insert run_experiment_fear into if lbview==None block
# test, debug
def run_experiment_fear(completed_arguments_dict, prepare_from_data_type):
    if prepare_from_data_type == 'synthetic':
        raise("not implemented yet: prepare_from_data_synthetic")
    elif prepare_from_data_type == 'chain':
        prepare_from_data_type = prepare_from_data_chain
    completed_arguments_dict["console_log"]='False' # no need for that in Fear case
    result_prefix = completed_arguments_dict["result_prefix"]
    
    import subprocess
    import os
    import sys
    
    try:
        print('making path %s' % result_prefix)
        os.makedirs(result_prefix)
    except OSError as err:
        if err.errno!=17:
            raise


    python_script_name = os.path.join(result_prefix, "qsub.py")
    with open(python_script_name, "w") as f:
        f.write('import sys; sys.path.append(\'/home/mlg/sb358/pygpstruct/src\'); ' + 
                'import numpy as np; import prepare_from_data_chain; ' + 
                'import util; util.stop_check((12-0.1)*3600); ' + # stop 360 sec before SGE job will be killed (which is after 12h) to reduce chances of being killed in the middle of a state save operation
                'prepare_from_data_chain.learn_predict_gpstruct_wrapper(%s, stop_check=util.stop_check);\n'
                % (', '.join(['%s=%s' % (k, params[k]) for k in sorted(params.keys())]))) 
    
    shell_script_name = os.path.join(result_prefix, "qsub.sh")
    with open(shell_script_name, "w") as f:
        f.write("#!sh\n")
        # f.write("source activate py3")
        f.write('python %s >%s.out 2>%s.err\n' # should replace > by >> 
                % (python_script_name, python_job_log_file_prefix, python_job_log_file_prefix))            

    for rerun in range(repeat_runs):
        ssh_command_as_list = ["ssh", "fear", "qsub", "-o", os.path.join(result_prefix, "qsub.out"), "-e", os.path.join(result_prefix, "qsub.err")]
        if (rerun > 0):
            ssh_command_as_list.extend(['-hold_jid', str(job_id)]) # if second or higher in a sequence of runs, wait for the previous job to finish
        ssh_command_as_list.append(shell_script_name)
        
        process = subprocess.Popen(ssh_command_as_list, 
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       cwd= files_path,
                                       )
                # qsub -o $model.fold_$fold.out -e $model.fold_$fold.err $model.sh $fold /bigscratch/sb358/image_segmentation/model_$model.train50/fold_$fold/
        
        # Poll process for new output until finished # from # http://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
        while True:
            nextline = process.stdout.readline().decode('utf-8')
            import re
            match = re.match('Your job (\d+) \(".+"\) has been submitted', nextline)
            if (match != None):
                job_id = match.group(1)
                with open("%s.rerun_%s.qsub.jobid" % (log_files_prefix, str(rerun)), "w") as f:
                    f.write(job_id)
            if nextline == '' and process.poll() != None:
                break
            sys.stdout.write(nextline)
            sys.stdout.flush()

        output = process.communicate()[0] # communicate returns (stdout, stderr)
        exitCode = process.returncode

        if (exitCode == 0):
            pass # return output
        else:
            print(output)
            raise subprocess.CalledProcessError(exitCode, cmd='TODO', output=output)
    

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
    print(completed_arguments_list)
    import sys
    sys.stdout.flush()
    if lbview != None:
        # need to pass all args in lambda cos otherwise "ValueError: sorry, can't pickle functions with closures"
        asr = lbview.map_async(lambda completed_arguments_dict, prepare_from_data_type=prepare_from_data_type: 
            run_experiment_parallel(completed_arguments_dict, prepare_from_data_type),
                                       completed_arguments_list)
        return asr
    else:
        for completed_arguments_dict in completed_arguments_list:
            run_experiment_sequential(completed_arguments_dict, prepare_from_data_type) 
    
    if require_output:
        # read files to obtain history_hp and history_ll
        history_list = []
        for completed_arguments_dict in completed_arguments_list:
            # determine #MCMC steps for this experiment from default and experiment arguments
            n_mcmc_steps = completed_arguments_dict["n_samples"]
            import learn_predict
            path = completed_arguments_dict["result_prefix"]
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
