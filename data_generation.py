import visualization.plot_3d_global as plot_3d
from Motion.InverseKinematics import animation_from_positions
from Motion import BVH
# from render_final import *
import numpy as np
from utils.rotation_conversions import *
from utils.motion_process import recover_from_ric
import visualization.plot_3d_global as plot_3d
import os
import torch
# import cv2
import sys
import options.option_transformer as option_trans

import clip
import models.vqvae as vqvae
import models.t2m_trans as trans

import openai
import json
from prompt_generation import *

def motion2bvh_ik(motions, bvh_path):
    frames, njoints, nfeats = motions.shape
    MINS = motions.min(axis=0).min(axis=0)
    MAXS = motions.max(axis=0).max(axis=0)

    height_offset = MINS[1]
    motions[:, :, 1] -= height_offset

    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    SMPL_JOINT_NAMES = [
        "Pelvis",
        "Left_hip",
        "Right_hip",
        "Spine1",
        "Left_knee",
        "Right_knee",
        "Spine2",
        "Left_ankle",
        "Right_ankle",
        "Spine3",
        "Left_foot",
        "Right_foot",
        "Neck",
        "Left_collar",
        "Right_collar",
        "Head",
        "Left_shoulder",
        "Right_shoulder",
        "Left_elbow",
        "Right_elbow",
        "Left_wrist",
        "Right_wrist",
    ]
    anim, sorted_order, _ = animation_from_positions(motions, parents)
    BVH.save(bvh_path, anim, names=np.array(SMPL_JOINT_NAMES)[sorted_order], frametime= 1/20)

# rescale bvh according to scaling_factor
def rescale_bvh(input_file_path, output_file_path, scaling_factor):
    
    # Open the input BVH file
    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()

        # Open the output BVH file
        with open(output_file_path, 'w') as output_file:

            # Iterate through each line in the input file
            index = 0 
            for line in lines:
                
                if line.strip() == "MOTION":
                    print(line)
                    break

                # Check if the line starts with "OFFSET"
                if line.strip().startswith('OFFSET'):

                    # Split the line into individual values
                    values = line.strip().split()

                    # Extract the "OFFSET" values as a list
                    offset = [float(v) for v in values[1:]]

                    # Multiply the "OFFSET" values by the scale factor
                    offset = [v * scaling_factor for v in offset]

                    offset_index = line.find('OFFSET')
                    before_offset = line[:offset_index]
                    after_offset = line[offset_index:]

                    # Write the modified "OFFSET" values to the output file, preserving any leading tabs
                    output_file.write(before_offset + 'OFFSET {:.6f} {:.6f} {:.6f}\n'.format(*offset))

                else:
                    # Write all other lines to the output file unchanged
                    output_file.write(line)
                    
                index += 1
                
            print(index)

            # Handle motion part
            output_file.write(lines[index])  # Append "MOTION"
            index += 1
            output_file.write(lines[index])  # Append "Frames: ..."
            index += 1
            output_file.write(lines[index])  # Append "Frame Time: ..."
            index += 1

            # Apply scaling factor to root translation in each frame
            for line in lines[index:]:
                values = line.strip().split()
                values[0] = str(float(values[0]) * scaling_factor)
                values[1] = str(float(values[1]) * scaling_factor)
                values[2] = str(float(values[2]) * scaling_factor)
                output_file.write(" ".join(values) + "\n")


#clip_text is a list with 1 text description 
def text_to_bvh(clip_text, name, npy_dir, bvh_dir, bvh_resize_dir, vis_file, clip_model, trans_encoder,net, std, mean,   
                 vis = False, npy_file = None):
    
    if npy_file is not None:
        pose = np.load(npy_file)
    else:
        text = clip.tokenize(clip_text, truncate=True).cuda()
        feat_clip_text = clip_model.encode_text(text).float()
        index_motion = trans_encoder.sample(feat_clip_text[0:1], False)
        pred_pose = net.forward_decoder(index_motion)

        from utils.motion_process import recover_from_ric
        pred_xyz = recover_from_ric((pred_pose*std+mean).float(), 22)
        xyz = pred_xyz.reshape(1, -1, 22, 3)

        np.save(f'{npy_dir}/{name}.npy', xyz.detach().cpu().numpy())
        pose = xyz.detach().cpu().numpy()
        
    
    if vis:
        import visualization.plot_3d_global as plot_3d
        pose_vis = plot_3d.draw_to_batch(pose,clip_text, [vis_file])

    #pose to bvh using inverse kinematics     
    bvh_path = f'{bvh_dir}/{name}.bvh'
    bvh_resize_path = f'{bvh_resize_dir}/{name}.bvh'
    motion2bvh_ik(pose[0], bvh_path)
    #resize bvh
    rescale_bvh(bvh_path, bvh_resize_path, 1.7/1.55)
    return 


# generate textual prompts for activities using ChatGPT, activities are lists of strings, motion_descriptions is a list of strings
# where each string provide a short description of the corresponding activity, total is the number of prompts per activity
def generate_prompt(activities, motion_descriptions, prompt_dir, total):
    activity_string = ', '.join(activities)

    num_prompts = 25
    if num_prompts >total:
        num_prompts = total

    os.makedirs(prompt_dir, exist_ok = True)
    for j in range(len(activities)):
        act = activities[j]
        motion_des = motion_descriptions[j]

        #move to next activity if prompt exist        
        if os.path.exists(f"{prompt_dir}/{act}.txt"):
            print(act)
            continue
        
        # read from the text file if exists
        if os.path.exists(f"{prompt_dir}/{act}.txt"):
            with open(f"{prompt_dir}/{act}.txt", 'r') as f:
                my_list = []
                for line in f:
                    my_list.append(line.strip())
            print(len(my_list))
        else:
            my_list = []

        while len(my_list) < total:
            
            try:
                response = generate_activity_description(act, num_prompts, activity_string, motion_des)
                print(response.choices[0]['message']['content'])
                prompts = response.choices[0]['message']['content']
        #         print(prompts)

                mylist = prompts.split('\n')

                #postprocess prompts
                print(mylist[-num_prompts:])
                mylist = mylist[-num_prompts:]
                for i in range(len(mylist)):
                    line = mylist[i]
                    if line[0].isdigit():
                        # Remove the number and period
                        line = line.split(". ", 1)[1]
                    print(line)
                    mylist[i] = line
                my_list.extend(mylist)
                print(len(my_list))


                # write to text files without overwriting    
                with open(f"{prompt_dir}/{act}.txt", "a") as file:
                    # Write each string in the list to the file
                    for string in mylist:
                        file.write(string + "\n")
            except:
                print("error in parsing prompts")



# sys.argv = ['GPT_eval_multi.py']
# args = option_trans.get_args_parser()
# args.dataname = 't2m'
# args.resume_pth = 'pretrained/VQVAE/net_last.pth'
# args.resume_trans = 'pretrained/VQTransformer_corruption05/net_best_fid.pth'
# args.down_t = 2
# args.depth = 3
# args.block_size = 51
# ## load clip model and datasets
# clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False, download_root='./')  # Must set jit=False for training
# clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
# clip_model.eval()
# for p in clip_model.parameters():
#     p.requires_grad = False

# net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
#                        args.nb_code,
#                        args.code_dim,
#                        args.output_emb_width,
#                        args.down_t,
#                        args.stride_t,
#                        args.width,
#                        args.depth,
#                        args.dilation_growth_rate)

# trans_encoder = trans.Text2Motion_Transformer(num_vq=args.nb_code, 
#                                 embed_dim=1024, 
#                                 clip_dim=args.clip_dim, 
#                                 block_size=args.block_size, 
#                                 num_layers=9, 
#                                 n_head=16, 
#                                 drop_out_rate=args.drop_out_rate, 
#                                 fc_rate=args.ff_rate)


# print ('loading checkpoint from {}'.format(args.resume_pth))
# ckpt = torch.load(args.resume_pth, map_location='cpu')
# net.load_state_dict(ckpt['net'], strict=True)
# net.eval()
# net.cuda()

# print ('loading transformer checkpoint from {}'.format(args.resume_trans))
# ckpt = torch.load(args.resume_trans, map_location='cpu')
# trans_encoder.load_state_dict(ckpt['trans'], strict=True)
# trans_encoder.eval()
# trans_encoder.cuda()

# mean = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')).cuda()
# std = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')).cuda()



# activities = ['walking forward', 'walking counter-clockwise', 'walking clockwise', 'climbing up stairs', 
#               'climbing down stairs', 'running', 'jumping', 'sitting', 'standing', 'lying', 'riding up elevator',
#              'riding down elevator']

# motion_descriptions = ['The subject walks forward in a straight line',
#                       'walking counter-clockwise in a full circle',
#                       'walking clockwise in a full circle',
#                       None,
#                       None,
#                       'The subject runs forward in a straight line',
#                       'jummping rope',
#                       'The subject sits on a chair either working or resting.',
#                       'The subject stands and talks to someone',
#                       None,
#                       'The subject rides in an ascending elevator',
#                       'The subject rides in a descending elevator']


# dir_name = 'test_dir'
# vis_dir = f'{dir_name}/vis_result'
# prompt_dir = f'{dir_name}/prompts'
# vis = True
# total = 10
# os.makedirs(dir_name, exist_ok= True)
# generate_prompt(activities, motion_descriptions, prompt_dir, total)



# num_repeats = 1
# for act in activities:
#     npy_dir = f'{dir_name}/npy/{act}'
#     bvh_dir = f'{dir_name}/bvh/ik/{act}'
#     bvh_resize_dir = f'{dir_name}/bvh_resized/ik/{act}'
    
#     vis_skeleton_dir = f'{vis_dir}/{act}'
#     os.makedirs(vis_skeleton_dir, exist_ok = True)
#     os.makedirs(npy_dir, exist_ok = True)
# #     os.makedirs(smpl_dir, exist_ok = True)
#     os.makedirs(bvh_dir, exist_ok = True)
#     os.makedirs(bvh_resize_dir, exist_ok = True)
    
#     # open the file for reading
#     with open(f'{prompt_dir}/{act}.txt', 'r', errors='replace') as f:
#       mylist = []
#       for line in f:
#         #remove faulty lines
#         if '\ufffd' in line:
#             print(line)
#         else:
#             mylist.append(line.strip())

#     print(mylist)
#     for i in range(len(mylist)):
#         print(mylist[i])
#         clip_text = [mylist[i]]
#         print(clip_text)
#         for j in range(num_repeats):
#             name = f'{act}_{i}_{j}'
#             print(name)
            
#             if os.path.exists(f'{bvh_dir}/{name}.bvh'):
# #                 print(f'{bvh_dir}/{name}.bvh')
#                 continue 
#             text_to_bvh(clip_text, name, npy_dir,  bvh_dir, bvh_resize_dir, 
#                         vis_file = f'{vis_skeleton_dir}/{name}.gif', vis = vis)