import os
import shutil

def remove_all_files(dirpath):
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

def generate_random_search_bash(rand_seed=0):
    bash_save_path = '../gat_jobs/'
    data_names = ['cora', 'citeseer', 'pubmed']
    if os.path.exists(bash_save_path):
        remove_all_files(bash_save_path)
    if bash_save_path and not os.path.exists(bash_save_path):
        os.makedirs(bash_save_path)
    for data_name in data_names:
        task_id = str(rand_seed)
        with open(bash_save_path + 'gat_run_' + task_id +'.sh', 'w') as rsh_i:
            command_i = 'bash gatrun.sh ' + task_id + ' ' + data_name
            rsh_i.write(command_i)
    print('{} jobs have been generated'.format(len(data_names)))


if __name__ == '__main__':
    generate_random_search_bash(rand_seed=0)