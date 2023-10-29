'''our package'''
from tool.clip_extract_vision import get_vision_clip
'''external package'''
import numpy as np
import os
from rich.console import Console
console = Console()
import torch.nn.functional as F
import torch
from concurrent.futures import ThreadPoolExecutor

#@ parameters
## imgs: img sequence(length is not certain)
#! note: last img is current img
## guide_loaded: every guide features 
def cal_game_rel(imgs, guide_loaded, alpha = 0.99):
    rel = []
    for i in range(len(imgs)):
        rel_i = []
        ## cal current img feature
        cur_img_feature = get_vision_clip(imgs[i])
        ## cla rel between current image_i and every guide
        for j in range(len(guide_loaded)):
            rel_i_guide_j = 0
            for k in range(len(guide_loaded[j])):
                rel_i_guide_j += F.cosine_similarity(guide_loaded[j][k],\
                                                     cur_img_feature)
            rel_i.append(rel_i_guide_j)
        rel.append(rel_i)
    result = []
    for i in range(len(rel[0])):
        result_i = 0
        for j in range(len(rel)):
            result_i = rel[len(rel)-1-j][i] + alpha * result_i
        result.append(result_i)

    oncatenated_result = torch.cat(result, dim=0)
    res = F.softmax(oncatenated_result, dim=0)      
    return res

        
# @ load guide 
def load_guide(game):
    instruct_dir = f'/home/Userlist/jinyg/dtmi/instruct/{game}/frame/'
    len_guide = 5
    guide_loaded = []
    def load_guide_images(i):
        guide_i_loaded = []
        guide_i_dir = f'{instruct_dir}{i}/'
        for j in range(20):
            guide_i_dir_j = f'{guide_i_dir}{j}.png'
            guide_i_loaded.append(get_vision_clip(guide_i_dir_j))
        return guide_i_loaded
    with ThreadPoolExecutor(max_workers=5) as executor:
        guide_loaded = list(executor.map(load_guide_images, range(len_guide)))
    return guide_loaded

if __name__ == '__main__':
    #@ state
    ## x = np.zeros((84, 84))
    ## img_features = get_vision_clip(x)
    ## print(img_features.shape)

    imgs_0 = ['/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/0/0.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/0/1.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/0/2.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/0/3.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/0/4.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/0/5.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/0/6.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/0/7.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/0/8.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/0/9.png']
    
    imgs_1 = ['/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/1/0.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/1/1.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/1/2.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/1/3.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/1/4.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/1/5.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/1/6.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/1/7.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/1/8.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/1/9.png']
    
    imgs_2 = ['/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/2/0.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/2/1.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/2/2.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/2/3.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/2/4.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/2/5.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/2/6.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/2/7.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/2/8.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/2/9.png']
    
    imgs_3 = ['/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/3/0.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/3/1.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/3/2.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/3/3.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/3/4.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/3/5.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/3/6.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/3/7.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/3/8.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/3/9.png']
    
    
    imgs_4 = ['/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/4/0.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/4/1.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/4/2.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/4/3.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/4/4.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/4/5.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/4/6.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/4/7.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/4/8.png',\
            '/home/Userlist/jinyg/dtmi/instruct/Air_Raid/frame/4/9.png']
  
    game = 'Air_Raid'

    #@ load guide
    console.log("load guide")
    guide_loaded = load_guide(game)
    print(len(guide_loaded),len(guide_loaded[0]))

    #@ test similarity
    console.log("calcualte relevance")
    rel = cal_game_rel(imgs_0, guide_loaded)
    console.log(rel)
    rel = cal_game_rel(imgs_1, guide_loaded)
    console.log(rel)
    rel = cal_game_rel(imgs_2, guide_loaded)
    console.log(rel)
    rel = cal_game_rel(imgs_3, guide_loaded)
    console.log(rel)
    rel = cal_game_rel(imgs_4, guide_loaded)
    console.log(rel)

    #@ alpha = 0.9
    # # tensor([0.2834, 0.3420, 0.3018, 0.0408, 0.0318], device='cuda:0',
    # #        dtype=torch.float16)
    # # tensor([0.0330, 0.0842, 0.0257, 0.4019, 0.4553], device='cuda:0',
    # #        dtype=torch.float16)
    # # tensor([0.0389, 0.0641, 0.8848, 0.0068, 0.0053], device='cuda:0',
    # #        dtype=torch.float16)
    # # tensor([0.0412, 0.0819, 0.0819, 0.4712, 0.3240], device='cuda:0',
    # #        dtype=torch.float16)
    # # tensor([0.0095, 0.0292, 0.0089, 0.2446, 0.7080], device='cuda:0',
    # #        dtype=torch.float16)

    #@ alpha = 0.95
    # # tensor([0.2952, 0.3345, 0.3345, 0.0189, 0.0167], device='cuda:0',
    # #    dtype=torch.float16)
    # # tensor([0.0240, 0.0949, 0.0240, 0.3752, 0.4819], device='cuda:0',
    # #     dtype=torch.float16)
    # # tensor([0.0248, 0.0409, 0.9302, 0.0026, 0.0018], device='cuda:0',
    # #     dtype=torch.float16)
    # # tensor([0.0247, 0.0593, 0.0523, 0.5625, 0.3010], device='cuda:0',
    # #     dtype=torch.float16)
    # # tensor([0.0039, 0.0173, 0.0034, 0.2390, 0.7363], device='cuda:0',
    # #     dtype=torch.float16)

