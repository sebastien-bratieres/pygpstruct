import sys

# NB if started with job_hash which doesnt exist, will restart from scratch and not warn !!
def launch_qsub_job(params, 
    files_prefix = '/bigscratch/sb358/pygpstruct-chain/results/afac/',
    job_hash = None, 
    file_prefix_without_data_indices=True):
    """
    params: dict of parameters to pass to prepare_from_data_chain.learn_predict_gpstruct_wrapper
    files_prefix : where all the files are put. Can end with / (directory) or not (common file prefix)
    job_hash : string or None. if None, creates new job folder. if string, reuse what's in that folder.
    file_prefix_without_data_indices : whether or not the file_prefix should be made to contain info on data (train, test) indices
    """
    import hashlib
    import time
    import subprocess
    import os
    
    files_path = os.path.split(files_prefix)[0]
    try:
        print('making path %s' % files_path)
        os.makedirs(files_path)
    except OSError as err:
        if err.errno!=17:
            raise

    if (job_hash == None):
        hash_string = hashlib.sha1()
        hash_string.update(str(time.time()))
        job_hash = hash_string.hexdigest()[:10]
    
    log_files_prefix = files_prefix + 'qsub_' + job_hash
    
    # turn params to a filename
    
    if file_prefix_without_data_indices:
        params_for_filename = params.copy()
        params_for_filename.pop('data_indices_test')
        params_for_filename.pop('data_indices_train')
        params_for_filename.pop('data_folder')
        param_as_filename = '++'.join(['%s=%s' % (k,v) for (k,v) in params_for_filename.items()])
    else:
        params_for_filename = params
        if params_for_filename.has_key('hp_bin_init'):
            param_as_filename = '++'.join(['%s=%s' % (k,params_for_filename[k]) for k in ['data_indices_test', 'data_indices_train', 'hp_bin_init', 'prediction_thinning', 'n_samples']])
        else:
            param_as_filename = '++'.join(['%s=%s' % (k,params_for_filename[k]) for k in ['data_indices_test', 'data_indices_train', 'prediction_thinning', 'n_samples']])
    param_as_filename = param_as_filename.replace(' ', '').replace('(','').replace(')','').replace(',','-').replace("'", '').replace('{', '').replace('}','')
    python_job_log_file_prefix = log_files_prefix + "." + param_as_filename

    params['result_prefix'] = "'" + python_job_log_file_prefix + ".'" # will be passed inside string command line
    params['console_log'] = 'False'
    params_string = ''
    for (k,v) in params.items():
        params_string += "%s=%s, " % (k, str(v))

    python_script_name = "%s.py" % log_files_prefix
    with open(python_script_name, "w") as f:
        f.write('import sys; sys.path.append(\'/home/mlg/sb358/pygpstruct/src\'); ' + 
                'import numpy as np; import prepare_from_data_chain; ' + 
                'import util; util.stop_check((12-0.1)*3600); ' + # stop 360 sec before SGE job will be killed (which is after 12h) to reduce chances of being killed in the middle of a state save operation
                'prepare_from_data_chain.learn_predict_gpstruct_wrapper(%s stop_check=util.stop_check);\n'
                % (params_string)) # NB params_string ends with a , 
    
    shell_script_name = "%s.sh" % log_files_prefix
    with open(shell_script_name, "w") as f:
        f.write("#!sh\n")
        f.write('python %s >%s.out 2>%s.err\n' # should replace > by >> 
                % (python_script_name, python_job_log_file_prefix, python_job_log_file_prefix))            

    process = subprocess.Popen(["ssh", "fear", "qsub", "-o", "%s.qsub.out" % (log_files_prefix), "-e", 
                                "%s.qsub.err" % (log_files_prefix), shell_script_name], 
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
            with open("%s.qsub.jobid" % (log_files_prefix), "w") as f:
                f.write(match.group(1))
        if nextline == '' and process.poll() != None:
            break
        sys.stdout.write(nextline)
        sys.stdout.flush()

    output = process.communicate()[0] # communicate returns (stdout, stderr)
    exitCode = process.returncode

    if (exitCode == 0):
        return output
    else:
        print(output)
        raise subprocess.CalledProcessError(exitCode, cmd='TODO', output=output)