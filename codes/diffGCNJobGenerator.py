import os
import shutil

def remove_all_files(dirpath):
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

def generate_random_search_bash(task_num, data_name):
    bash_save_path = '../' + data_name + '_jobs/'
    if os.path.exists(bash_save_path):
        remove_all_files(bash_save_path)
    if bash_save_path and not os.path.exists(bash_save_path):
        os.makedirs(bash_save_path)
    for i in range(task_num):
        task_id = str(i+1)
        with open(bash_save_path + 'diff_gcn_run_' + task_id +'.sh', 'w') as rsh_i:
            command_i = 'bash diffgcnrun.sh ' + task_id + ' ' + data_name
            rsh_i.write(command_i)
    print('{} jobs have been generated'.format(task_num))


if __name__ == '__main__':
    generate_random_search_bash(task_num=10, data_name='cora')