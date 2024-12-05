from pyquaternion import Quaternion
import math
import numpy as np
import pickle as pkl
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
import os
import tqdm
from argparse import ArgumentParser

def convert_pose(quaternion,translation):
    # Convert rotation matrix to quaternion
    quaternion = Quaternion(quaternion)
    rotation_matrix=quaternion.rotation_matrix

    rotation_matrix=np.linalg.inv(rotation_matrix)
    translation=-np.dot(rotation_matrix,translation)

    return rotation_matrix,translation

parser = ArgumentParser("maskrcnn")
parser.add_argument("--meta_folder", default="output", type=str)
args = parser.parse_args()


meta_folder=args.meta_folder
# scenes=['061400_s20-122_1710115970.0_1710115986.0']
imageroot = os.path.join(meta_folder,'images')
motionroot = os.path.join(meta_folder,'motions')
scenes=os.listdir(imageroot)


# import pdb
# pdb.set_trace()

# scenes=os.listdir('./')
scenes=[temp for temp in scenes if ((not '.pkl' in temp) and (not '.py' in temp) and (not '.bash' in temp) and (not 'trajs' in temp))]
scenes.sort()
# import pdb
# pdb.set_trace()
# scenes=['070823_s20-302_1701219822.0_1701219840.0']
count=0
diff_errors=[]
for scene in tqdm.tqdm(scenes):
    # import pdb
    # pdb.set_trace()
    if not os.path.exists(os.path.join(imageroot,scene,'sparse/0/images.txt')):
        continue
        
    count+=1
    txtpath=os.path.join(imageroot,scene,'sparse/0/images.txt')
    with open(txtpath, 'r') as file:
        lines = file.readlines()
    lines = [line for idx, line in enumerate(lines) if ('.jpg' in line or '.png' in line)]

    try:
        inds=[int(temp.split('.jpg')[0].split('/')[-1]) for temp in lines]
    except:
        inds=[int(temp.split('.jpg')[0].split(' ')[-1]) for temp in lines]
        # import pdb
        # pdb.set_trace()

    combined = list(zip(inds, lines))
    sorted_combined = sorted(combined)
    lines = [x for _, x in sorted_combined]

    poses={}
    converted_poses=[]
    for line in lines:
        items=line.split(' ')
        quat=items[1:5]
        translation=items[5:8]
        quat=np.array([float(q) for q in quat])
        translation=np.array([float(q) for q in translation])
        filename=items[-1]
        # import pdb
        # pdb.set_trace()
        # try:
        converted_pose=convert_pose(quat,translation)
        poses[filename]=converted_pose[1]
        converted_poses.append(converted_pose)
        # except:
        #     import pdb
        #     pdb.set_trace()
    
    pred_poses=np.array(list(poses.values()))
    with open(os.path.join('/mnt/nvme1n1p1/test_navi_oodmotion_maskcomlmap',scene+'.pkl'),'rb') as f:
        delta_gt_poses=pkl.load(f)
    # poses
    pred_poses=pred_poses[:,:2]
    delta_gt_poses=delta_gt_poses.numpy()

    gt_poses=delta_gt_poses.cumsum(axis=0)

    # plt.cla()
    # plt.axis('equal')
    # plt.scatter(pred_poses[:, 0], pred_poses[:, 1], color='red', label='Aligned Trajectory 2')
    # plt.savefig('./trajs/'+scene+'.jpg', format='jpg')

    # import pdb
    # pdb.set_trace()
    try:
        aligned_traj1, aligned_traj2, disparity = procrustes(pred_poses, gt_poses)
    except:
        continue
    scale_pred2uni=np.linalg.norm(aligned_traj1[-1]-aligned_traj1[0])/np.linalg.norm(pred_poses[-1]-pred_poses[0])
    scale_uni2gt=np.linalg.norm(gt_poses[-1]-gt_poses[0])/np.linalg.norm(aligned_traj2[-1]-aligned_traj2[0])

    totoalscale=scale_pred2uni*scale_uni2gt

    deltaposes=[]
    for ind in range(1,len(converted_poses)):
        prev_pose=converted_poses[ind-1]
        curr_pose=converted_poses[ind]

        # import pdb
        # pdb.set_trace()

        prev_pose_mat=np.eye(4)
        prev_pose_mat[:3,:3]=prev_pose[0]
        prev_pose_mat[:3, 3]=prev_pose[1]
        curr_pose_mat=np.eye(4)
        curr_pose_mat[:3,:3]=curr_pose[0]
        curr_pose_mat[:3, 3]=curr_pose[1]

        T_current_to_prev = np.dot(np.linalg.inv(prev_pose_mat), curr_pose_mat)#transformation_matrix(prev_rot,curr_rot)

        deltapose = T_current_to_prev[:3, 3]

        deltaposes.append([deltapose[2],deltapose[0]])

    deltaposes=np.stack(deltaposes)#*totoalscale

    ratios=[]
    for i in range(5):
        ratios.append(np.linalg.norm(delta_gt_poses[i])/np.linalg.norm(deltaposes[i]))

    # import pdb
    # pdb.set_trace()

    ratio=np.mean(ratios)
    deltaposes*=ratio

    deltapose_error=deltaposes[:,:2]-delta_gt_poses[:len(deltaposes)]

    # import pdb
    # pdb.set_trace()

    diff_errors.append(deltapose_error)

    # deltapose_error=np.abs(deltapose_error)
    # deltapose_error[np.where(deltapose_error>10)]=0
    # print(deltapose_error.mean(axis=0))
    # if max(deltapose_error.mean(axis=0))<0.20:
    #     diff_errors.append(deltapose_error)

    # import pdb
    # pdb.set_trace()

        # # # 计算当前帧的pose在上一帧ego坐标系下的位置
        # current_pose_homogeneous = np.append(np.zeros(3),1)#np.append(current_pose, 1)
        # current_pose_ego_frame_homogeneous = np.dot(T_current_to_prev, current_pose_homogeneous)

    # dist_pred0togt0=np.linalg.norm(aligned_traj1[0]-aligned_traj2[0])
    # dist_pred0togtend=np.linalg.norm(aligned_traj1[0]-aligned_traj2[-1])

    # if dist_pred0togtend<dist_pred0togt0:
    #     aligned_traj1*=-1

    # pred_diff=np.stack([aligned_traj1[i+1]-aligned_traj1[i] for i in range(0,len(aligned_traj1)-1)])
    # gt_diff=np.stack([aligned_traj2[i+1]-aligned_traj2[i] for i in range(0,len(aligned_traj2)-1)])

    # # import pdb
    # # pdb.set_trace()

    # diff_error=(np.linalg.norm(pred_diff-gt_diff,axis=1)*scale).mean()
    # diff_errors.append(diff_error)

    # # import pdb
    # # pdb.set_trace()

    # plt.cla()
    # plt.scatter(aligned_traj1[:, 0], aligned_traj1[:, 1], color='blue', label='Aligned Trajectory 1')
    # plt.scatter(aligned_traj2[:, 0], aligned_traj2[:, 1], color='red', label='Aligned Trajectory 2')
    # plt.axis('equal')
    # # plt.legend()
    # # plt.title('Aligned Trajectories')
    # # plt.xlabel('X')
    # # plt.ylabel('Y')
    # # plt.grid(True)
    # plt.savefig('./trajs/aligned_trajs/'+scene+'.jpg', format='jpg')

    # plt.cla()
print(len(diff_errors))
diff_errors=np.concatenate(diff_errors,axis=0)
diff_errors=np.abs(diff_errors)
# diff_errors[np.where(diff_errors>5)]=0
diff_errors = diff_errors[np.where(diff_errors.sum(axis=1)<10)]
print('[E_AIF_x,  E_AIF_y]:')
print(diff_errors.mean(axis=0))
# import pdb
# pdb.set_trace()