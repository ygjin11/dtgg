import numpy as np
import random
# our package
from tool.fixed_replay_buffer import FixedReplayBuffer
from tool.utils import sample
import os
import cv2
from PIL import Image
import json
import sys
import argparse
import yaml

# get best trajectory
def obtain_best_traj(num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer):
    # -- load data from memory (make more efficient)
    
    # return info
    obss = []
    actions = []
    returns = [0]
    done_idxs = []
    stepwise_returns = []

    # transitions_per_buffer: number of transitions each buffer already loaded
    # trasition: state_a->state_b 
    transitions_per_buffer = np.zeros(50, dtype=int)

    # num_trajectories: number of trajectories
    num_trajectories = 0
    
    # num_steps: number of <s, a ,r>
    while len(obss) < num_steps:

        # choose one from 50 buffers 
        buffer_num = np.random.choice(np.arange(50 - num_buffers, 50), 1)[0]
        i = transitions_per_buffer[buffer_num]

        # load i buffer
        frb = FixedReplayBuffer(
            data_dir=data_dir_prefix + game + '/1/replay_logs',
            replay_suffix=buffer_num,
            observation_shape=(84, 84),
            stack_size=4,
            update_horizon=1,
            gamma=0.99,
            observation_dtype=np.uint8,
            batch_size=32,
            replay_capacity=100000)
        
        if frb._loaded_buffers:
            done = False
            curr_num_transitions = len(obss)
            trajectories_to_load = trajectories_per_buffer

            while not done:
                states, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(batch_size=1, indices=[i])
                states = states.transpose((0, 3, 1, 2))[0] # (1, 84, 84, 4) --> (4, 84, 84)
                obss += [states]
                actions += [ac[0]]
                stepwise_returns += [ret[0]]
                if terminal[0]:
                    done_idxs += [len(obss)]
                    returns += [0]
                    if trajectories_to_load == 0:
                        done = True
                    else:
                        trajectories_to_load -= 1
                returns[-1] += ret[0]
                i += 1
                if i >= 100000:
                    obss = obss[:curr_num_transitions]
                    actions = actions[:curr_num_transitions]
                    stepwise_returns = stepwise_returns[:curr_num_transitions]
                    returns[-1] = 0
                    i = transitions_per_buffer[buffer_num]
                    done = True
            num_trajectories += (trajectories_per_buffer - trajectories_to_load)
            transitions_per_buffer[buffer_num] = i

    actions = np.array(actions)
    returns = np.array(returns)
    stepwise_returns = np.array(stepwise_returns)
    done_idxs = np.array(done_idxs)

    # -- create reward-to-go dataset
    start_index = 0
    rtg = np.zeros_like(stepwise_returns)
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = stepwise_returns[start_index:i]
        for j in range(i-1, start_index-1, -1): # start from i-1
            rtg_j = curr_traj_returns[j-start_index:i-start_index]
            rtg[j] = sum(rtg_j)
        start_index = i
    print('max rtg is %d' % max(rtg))

    # -- create timestep dataset
    start_index = 0
    timesteps = np.zeros(len(actions)+1, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1
    print('max timestep is %d' % max(timesteps))


    max_rtg_index = np.argsort(rtg)[-1]
    max_indices = np.where(rtg == rtg[max_rtg_index])[0]  
    index_max = np.min(max_indices)
    index_max_last = len(obss)
    for i in range(len(done_idxs)):
        if done_idxs[i] == index_max:
            if i!=len(done_idxs)-1:
                index_max_last = done_idxs[i+1]
    index_max_last = index_max_last

    obss = obss[index_max: index_max_last]
    actions = actions[index_max: index_max_last]
    done_idxs = done_idxs[index_max: index_max_last]
    rtg = rtg[index_max: index_max_last]
    timesteps = timesteps[index_max: index_max_last]

    return obss, actions, actions, done_idxs, rtg, timesteps

# get change point
def find_change_points(sequence):
    change_points = []
    in_decreasing_sequence = False

    for i in range(len(sequence) - 1):
        if sequence[i] > sequence[i + 1]:
            if not in_decreasing_sequence:
                in_decreasing_sequence = True
                change_points.append(i)
        else:
            in_decreasing_sequence = False

    return change_points

# get little segments
def get_segment_traj(obss, actions, rtg, num_segment):
    
    index_change =  find_change_points(rtg)
    segment_all_traj, segment_all_action, segment_all_rtg = [], [], []
    point_pair = []
    for i in range(len(index_change)-1):
        pair = []
        pair.append(index_change[i])
        pair.append(index_change[i+1])
        point_pair.append(pair)
    
    random_pair = random.sample(point_pair, num_segment)
    for i in range(len(random_pair)):
        start = random_pair[i][0]
        end = random_pair[i][1]
        if end-start < 20:
            start  = 20 - (end-start)
        if end-start > 20:
            start = end - 20

        segment_traj, segment_action, segment_rtg  = obss[start:end], actions[start:end], rtg[start:end]

        segment_all_traj.append(segment_traj)
        segment_all_action.append(segment_action)
        segment_all_rtg.append(segment_rtg)

    return segment_all_traj, segment_all_action, segment_all_rtg

# save video and frame
def get_video_frame(dir, segment_traj):
    for i in range(len(segment_traj)):
        segment = segment_traj[i]
        frames = []
        for state in segment:
            frames.append(state[0])
        
        # save ivdeo
        video_dir = dir + 'video/'
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        video_filename = os.path.join(video_dir, f'{i}.mp4')
        video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 10, (84, 84), isColor=False)
        for frame in frames:
            video.write(frame)
        
        # save picture
        frame_dir = dir + 'frame/' + str(i) + '/'
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        for k in range(len(frames)):
            frame_k_dir = frame_dir + f'{k}.png'
            gray_image = Image.fromarray(frames[k].astype(np.uint8))
            gray_image.save(frame_k_dir)

        video.release()

# save action and rtg
def get_action_rtg(dir, segment_action, segment_rtg):

    root_action_dir = dir + 'action/'
    root_rtg_dir = dir + 'rtg/'

    if not os.path.exists(root_action_dir):
        os.makedirs(root_action_dir)
    if not os.path.exists(root_rtg_dir):
        os.makedirs(root_rtg_dir)

    for i in range(len(segment_action)):
        segment_action_i = segment_action[i]
        segment_rtg_i = segment_rtg[i]
        Dict_action = dict()
        Dict_rtg = dict()
        for j in range(len(segment_action_i)):
            Dict_action[j] = int(segment_action_i[j])
            Dict_rtg[j] = int(segment_rtg_i[j])
        with open(root_action_dir+str(i)+'.json','w') as file:
            json.dump(Dict_action, file)    
        with open(root_rtg_dir+str(i)+'.json','w') as file:
            json.dump(Dict_rtg, file)  

if __name__ == '__main__': 

    current_file = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file)
    parent_directory = os.path.dirname(current_directory)
    config_path = parent_directory + '/config/config_para/para.yaml'

    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    game = config['annotate']['game_annonate']
    data_dir_prefix = config['dataset_dir'] 
    out_putdir = config['instruct_dir'] + game + '/'
    num_segment = config['annotate']['traj_num']

    num_steps = 1000000
    num_buffers = 50
    trajectories_per_buffer = 20

    obss, actions, returns, done_idxs, rtg, timesteps = obtain_best_traj(num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer)
    if not os.path.exists(out_putdir):
        os.makedirs(out_putdir)
    segment_traj, segment_action, segment_rtg  = get_segment_traj(obss, actions, rtg, num_segment)
    get_video_frame(out_putdir, segment_traj)
    get_action_rtg(out_putdir, segment_action, segment_rtg)


                                                                  
