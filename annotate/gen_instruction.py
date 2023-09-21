import json
import os
import yaml

def gen_action_inst(instruct_dir, traj_num):
    
    with open(instruct_dir + 'action_deps.json', 'r') as file:
        action_deps = json.load(file)

    action_inst_dir = instruct_dir + 'action_inst/'
    action_dir = instruct_dir + 'action/'

    for i in range(traj_num):
        sub_action_dir = action_dir + str(i) + '.json'
        with open(sub_action_dir, 'r') as file:
            action_i= json.load(file)
        for key in action_i:
            a = action_i[key]
            a_i = action_deps[str(a)]
            action_i[key] = a_i
        sub_action_inst_dir = action_inst_dir + str(i) + '.json'
        with open( sub_action_inst_dir, 'w') as file:
            json.dump(action_i, file)


if __name__ == '__main__': 

    current_file = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file)
    parent_directory = os.path.dirname(current_directory)
    config_path = parent_directory + '/config/config_main/main.yaml'

    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    game = config['annotate']['game_annonate']
    traj_num = config['annotate']['traj_num']
    instruct_dir = config['instruct_dir'] + game + '/'
    action_inst_dir = instruct_dir + 'action_inst/'
    if not os.path.exists(action_inst_dir):
        os.makedirs(action_inst_dir)
    
    gen_action_inst(instruct_dir, traj_num)
    


