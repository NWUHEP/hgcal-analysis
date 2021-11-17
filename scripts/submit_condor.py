#!/usr/bin/env python
import os
import datetime
import argparse, optparse
import yaml
from pathlib import Path

def get_current_time():
    now = datetime.datetime.now()
    currentTime = '{0:02d}{1:02d}{2:02d}_{3:02d}{4:02d}{5:02d}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    return currentTime

def make_directory(file_path, clear=True):
    if not file_path.exists():
        os.system(f'mkdir -p {file_path}')

    if clear and len(os.listdir(file_path)) != 0:
        os.system(f'rm -rf {file_path}/*')

def make_file_batches(process, batch_config):
    file_location = batch_config['location']
    files_per_job = batch_config['files_per_job']

    file_list = [f'{file_location}/{f}' for f in os.listdir(file_location)]
    if len(file_list) == 0:
        return None

    n_files = len(file_list)
    file_split = [file_list[i:i + files_per_job] for i in range(0, n_files, files_per_job)]

    return file_split

def prepare_submit(process, batches, output_dir, executable):

    # Writing the batch config file
    batch_filename = f'.batch_tmp_{process}'
    batch_tmp = open(batch_filename, 'w')
    batch_tmp.write('''\
    Universe              = vanilla
    Should_Transfer_Files = YES
    WhenToTransferOutput  = ON_EXIT
    want_graceful_removal = true
    Notification          = Never
    Requirements          = OpSys == "LINUX"&& (Arch != "DUMMY" )
    request_disk          = 2000000
    request_memory        = 4096
    \n
    ''')

    for i, batch in enumerate(batches):

        ## make file with list of inputs ntuples for the analyzer
        input_file = open(f'input_{process}_{i}.txt', 'w')
        input_file.writelines(f'{f}\n' for f in batch)
        input_file.close()

        ### set output directory
        batch_tmp.write(f'''
        Executable            = {executable}
        Arguments             = {i} input_{process}_{i}.txt
        Transfer_Input_Files  = source.tar.gz, input_{process}_{i}.txt
        Output                = reports/{process}_{i}_$(Cluster)_$(Process).stdout
        Error                 = reports/{process}_{i}_$(Cluster)_$(Process).stderr
        Log                   = reports/{process}_{i}_$(Cluster)_$(Process).log
        Queue
        \n
        ''')

    batch_tmp.close()

    cmd = f'condor_submit {batch_filename}'
    return cmd
        
def prepare_output(output_dir, prefix):
    time_string = get_current_time()
    output_dir = Path(f'{output_dir}/{prefix}_{time_string}')
    output_dir.mkdir(parents=True)

    stage_dir = Path(f'batch/{prefix}_{time_string}')
    stage_dir.mkdir(parents=True)

    return output_dir, stage_dir
    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', 
                        type=str,
                        help="Provide configuration file for submission of batch jobs."
                      )
    args = parser.parse_args()

    # Load configuration file
    with open(args.config, 'r') as config_file: 
        config = yaml.safe_load(config_file)

    current_dir = os.getcwd()
    executable = config['executable']
    output_dir, stage_dir = prepare_output(config['output_dir'], config['prefix'])
    os.system(f'tar czf {stage_dir}/source.tar.gz {current_dir} --exclude="*.hdf5" --exclude="batch" --exclude-vcs')
    os.system(f'cp {current_dir}/{executable} {stage_dir}/.')
    os.chdir(stage_dir)

    print('Preparing to submit jobs:\n')
    for process, batch_config in config['inputs'].items():
        batches = make_file_batches(process, batch_config)
        if not batches:
            print('No files found for process {process}!')
            continue

        cmd = prepare_submit(process, batches, output_dir, executable)
        print(f'{cmd}')
    
    os.chdir(current_dir)
