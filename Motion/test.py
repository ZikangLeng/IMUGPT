import numpy as np
from Quaternions import Quaternions
import myBVH
import BVH
import myAnimation
import Animation
import myInverseKinematics as myIK
import InverseKinematics as IK
import math
import sys
from model_zoo import fk_layer
import torch
import copy
import os.path as osp
from scipy.spatial.transform import Rotation as R
# from Auto_Conditioned_RNN_motion.src.read_bvh import write_traindata_to_bvh


# orients: inital rotation (e.g., T pose). In our settings should be [1,0,0,0] (=zero rotation)
# offsets: initial skeleton joints position

# rotations: rotations in frames through time. relative to initial offsets
# positions: joint position in frames through time


def joint_angle_error(pred_mat, gt_mat):
    """
    Compute the geodesic distance between the two input matrices.
    :param pred_mat: predicted rotation matrices. Shape: ( Seq, 9, 3, 3)
    :param gt_mat: ground truth rotation matrices. Shape: ( Seq, 24, 3, 3)
    :return: Mean geodesic distance between input matrices.
    """

    gt_mat = gt_mat[:, SMPL_OR_JOINTS, :, :]

    # Reshape the matrices into B x 3 x 3 arrays
    r1 = np.reshape(pred_mat, [-1, 3, 3])
    r2 = np.reshape(gt_mat, [-1, 3, 3])

    # Transpose gt matrices
    r2t = np.transpose(r2, [0, 2, 1])

    # compute R1 * R2.T, if prediction and target match, this will be the identity matrix
    r = np.matmul(r1, r2t)

    angles = []
    # Convert rotation matrix to axis angle representation and find the angle
    for i in range(r1.shape[0]):
        aa, _ = cv2.Rodrigues(r[i])
        angles.append(np.linalg.norm(aa))

    return np.mean(np.array(angles))

def parents_20_totem():
    parents = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 10, 8, 12, 13, 14, 8, 16, 17, 18])
    return parents

def offsets_20_totem():
    offsets = np.array([
       [ 0.      ,  0.      ,  0.      ],
       [-2.726159,  0.      ,  0.      ],
       [ 0.      , -9.081711,  0.      ],
       [ 0.      , -9.313666,  0.      ],
       [ 2.726159,  0.      ,  0.      ],
       [ 0.      , -9.081711,  0.      ],
       [ 0.      , -9.313666,  0.      ],
       [ 0.      ,  0.      ,  0.      ],
       [ 0.      ,  4.787401,  0.      ],
       [ 0.      ,  5.271469,  0.      ],
       [ 0.      ,  2.483916,  0.      ],
       [ 0.      ,  2.358166,  0.      ],
       [ 0.      ,  5.271469,  0.      ],
       [ 0.      ,  3.096982,  0.      ],
       [ 0.      ,  5.718695,  0.      ],
       [ 0.      ,  5.161836,  0.      ],
       [ 0.      ,  5.271469,  0.      ],
       [ 0.      ,  3.096982,  0.      ],
       [ 0.      ,  5.718695,  0.      ],
       [ 0.      ,  5.161836,  0.      ]
    ])
    return offsets


def IK_test_consistency():
    """ read a bvh, run FK and then IK and expect what we had in the bvh """

    # bvh_path = osp.expanduser('~/tmp/from_MotioNet.bvh')
    # bvh_path = osp.expanduser('~/tmp/20 Joints GT.bvh')
    bvh_path = osp.expanduser('~/tmp/generated_35.bvh')
    bvh_path_out = bvh_path.replace('.bvh', '_fk_ik.bvh')
    anim, names, _ = BVH.load(bvh_path, world=True)
    # bvh_path_out = bvh_path.replace('.bvh', '_test.bvh')
    # BVH.save(bvh_path_out, anim, names=names)
    # return

    # convert nan rotataions to Id rotations
    anim.rotations.qs[np.isnan(anim.rotations.qs).all(axis=2)] = Quaternions.id(1)

    # forward kinematics: input rotations to positions
    fk_transforms = Animation.transforms_global(anim)
    fk_positions = fk_transforms[:, :, :3, 3]

    # inverse kinematics: positions to rotations
    # anim_ik, _ = IK.animation_from_positions(fk_positions, parents=parents_20_totem(), offsets=offsets_20_totem())
    anim_ik, _ = IK.animation_from_positions(fk_positions, parents=anim.parents)

    # mpjpe: confirm that requested positions is the same as the ones obtained by anim_ik
    fk_transforms_from_ik = Animation.transforms_global(anim_ik)
    fk_positions_from_ik = fk_transforms_from_ik[:, :, :3, 3]
    mpjpe = np.linalg.norm((fk_positions - fk_positions_from_ik), axis=2)
    print('max location error = {}'.format(mpjpe.max().round(2)))
    print('mean location error = {}'.format(mpjpe.mean().round(2)))

    # mpjae
    mpjae = np.linalg.norm((-anim.rotations * anim_ik.rotations).log(), axis=2)
    print('max angle error = {}'.format(mpjae.max().round(2)))
    print('mean angle error = {}'.format(mpjae.mean().round(2)))

    # save output of IK. you should compare it to the input and make sure they look the same
    BVH.save(bvh_path_out, anim_ik, names=names)


def IK_test_consistency_Quaternions():

    """ read a bvh, run FK and then IK and expect what we had in the bvh """
    bvh_path = osp.expanduser('~/tmp/S9_Discussion 1_0_1_2_3_view_0_totem_up.bvh')
    # bvh_path = osp.expanduser('~/tmp/minimal3_out.bvh')
    bvh_path = osp.expanduser('~/tmp/20 joints GT.bvh')
    bvh_path_out = bvh_path.replace('.bvh', '_fk_ik.bvh')
    anim, names, _ = BVH.load(bvh_path, world=True)

    # test that BVH.save(BVH.load(file)) produces identical file
    bvh_path_same = bvh_path.replace('.bvh', '_same.bvh')
    BVH.save(bvh_path_same, anim, names=names)

    anim.positions[:,0] = anim.offsets[0] # SIGAL: just for now put pelvis in 0,0,0 for all frames
    anim.rotations.qs[np.isnan(anim.rotations.qs).all(axis=2)] = Quaternions.id(1)
    anim_original = copy.deepcopy(anim)

    fk_transforms = Animation.transforms_global(anim)
    fk_positions = fk_transforms[:, :, :3, 3]

    anim.rotations = Quaternions.id(anim.shape)
    ik = IK.BasicInverseKinematics(anim, fk_positions, silent=False, iterations=1)
    # ik = IK.BasicJacobianIK(anim, fk_positions, silent=False, iterations=100)
    import time
    start = time.time()
    ik()
    end = time.time()
    print('ik time = {}'.format(end-start))

    # ensure rotation are the same as ik(fk(rotation))
    # if original rotation is same as rotation obtained by ik, then the multiplication of one by the inverse of the
    # other would be a zero rotation, or a matrix to which the input joint is an eigen vector with eigen value of 1
    rot_mult_by_inverse = -anim_original.rotations * anim.rotations

    # hack to overcome a bug in Quaternions.__neg__
    anim_original_euler = anim_original.rotations.euler(order='xyz').reshape(-1,3)
    anim_original_scipy_rotations = R.from_euler(angles=anim_original_euler, seq='xyz')
    anim_original_scipy_rotations_inv = anim_original_scipy_rotations.inv()
    anim_original_euler_inv = anim_original_scipy_rotations_inv.as_euler(seq='xyz').reshape(anim.shape+(3,))
    anim_original_inv = Quaternions.from_euler(anim_original_euler_inv, order='zyx', world=True)
    # rot_mult_by_inverse = anim_original_inv * anim.rotations
    anim_scipy = R.from_euler(angles=anim.rotations.euler().reshape(-1,3), seq='xyz')
    rot_mult_by_inverse_scipy = anim_original_scipy_rotations_inv * anim_scipy
    rot_mult_by_inverse = Quaternions.from_euler(rot_mult_by_inverse_scipy.as_euler(seq='xyz').reshape(anim.shape+(3,)),
                                                 order='zyx', world=True)

    angle, _ = rot_mult_by_inverse.angle_axis()
    print('max angle error = {}'.format(angle.max().round(5)))
    print('mean angle error = {}'.format(angle.mean().round(5)))

    rotvec = rot_mult_by_inverse_scipy.as_rotvec()
    print('max angle error (scipy) = {}'.format(np.linalg.norm(rotvec, axis=1).max().round(5)))
    print('mean angle error (scipy) = {}'.format(np.linalg.norm(rotvec, axis=1).mean().round(5)))

    BVH.save(bvh_path_out, anim, names=names)

    rot = rotations_from_positions(fk_positions)

    pass


def IK_test_consistency_scipy():
    """ read a bvh, run FK and then IK and expect what we had in the bvh """
    bvh_path = osp.expanduser('~/tmp/from_MotioNet.bvh')
    bvh_path = osp.expanduser('~/tmp/S9_Discussion 1_0_1_2_3_view_0_totem_up.bvh')
    # bvh_path = osp.expanduser('~/tmp/minimal3_out.bvh')
    bvh_path_out = bvh_path.replace('.bvh', '_fk_ik.bvh')
    anim, names, _ = myBVH.load(bvh_path, world=True)

    # test that BVH.save(BVH.load(file)) produces identical file
    bvh_path_same = bvh_path.replace('.bvh', '_same.bvh')
    myBVH.save(bvh_path_same, anim, names=names)

    anim.positions[:,0] = anim.offsets[0] # SIGAL: just for now put pelvis in 0,0,0 for all frames
    anim_original = copy.deepcopy(anim)

    fk_transforms = myAnimation.transforms_global(anim)
    fk_positions = fk_transforms[:, :, :3, 3]

    # anim.rotations = Quaternions.id(anim.shape)
    anim.rotations = R.identity(anim.shape[0]*anim.shape[1])
    ik = myIK.BasicInverseKinematics(anim, fk_positions, silent=False, iterations=1)
    # ik = IK.BasicJacobianIK(anim, fk_positions, silent=False, iterations=100)
    # ik(align_method='scipy')
    import time
    start = time.time()
    ik(align_method='axis_angle')
    end = time.time()
    print('ik time = {}'.format(end-start))

    # ensure rotation are the same as ik(fk(rotation))
    # if original rotation is same as rotation obtained by ik, then the multiplication of one by the inverse of the
    # other would be a zero rotation, or a matrix to which the input joint is an eigen vector with eigen value of 1
    rot_mult_by_inverse = anim_original.rotations.inv() * anim.rotations
    rotvec = rot_mult_by_inverse.as_rotvec()
    print('max angle error = {}'.format(np.linalg.norm(rotvec, axis=1).max().round(5)))
    print('mean angle error = {}'.format(np.linalg.norm(rotvec, axis=1).mean().round(5)))
    myBVH.save(bvh_path_out, anim, names=names)

    pass


def IK_test_minimal():
    # bvh_path = osp.expanduser('~/tmp/minimal3_net_rot.bvh')
    bvh_path = osp.expanduser('~/tmp/minimal5.bvh')
    # bvh_path = osp.expanduser('~/tmp/from_MotioNet.bvh')
    bvh_path_out = bvh_path.replace('.bvh', '_out.bvh')

    anim, names, _ = BVH.load(bvh_path, world=True)
    if False:
        bvh_path_out = bvh_path.replace('.bvh', '_test.bvh')
        BVH.save(bvh_path_out, anim, names=names)
        return

    # target_positions = np.array([[[0,0,0],[0,0,1]], [[0,0,0],[0,0,-1]]])
    # target_positions = np.array([[[0,0,0],[0,0,1],[1,0,1]], [[0,0,0],[1,0,0],[1/math.sqrt(2),0,-1/math.sqrt(2)]]])
    # target_positions = np.array([[[0,0,0],[0,0,-1],[1,0,-1]], [[0,0,0],[1,0,0],[2,0,0]]]) # minimal_net_rot
    target_positions = np.array([[[0,0,0],[0,0,1],[1,0,1],[1,0,0],[2,0,0]], [[0,0,0],[1,0,0],[2,0,0],[0,0,-1],[0,0,-2]]]) # minimal5
    # n_frames = target_positions.shape[0]
    anim.rotations = Quaternions.id(target_positions.shape[:2]) # R.identity(anim.shape[1]*n_frames)
    anim.positions = np.repeat(anim.positions, target_positions.shape[0], axis=0)

    ik = IK.BasicInverseKinematics(anim, target_positions, silent=False) #  BasicJacobianIK
    ik()
    # ik('axis_angle')
    # ik.call_axis_angle()
    # euler_angels =[[[math.degrees(angle) for angle in frame] for frame in joint_frames] for joint_frames in Quaternions.euler(anim.rotations)]
    # euler = anim.rotations.as_euler(seq='xyz', degrees=True).reshape(anim.shape+(3,))
    euler = np.degrees(anim.rotations.euler(order='xyz'))
    # print(euler_angels)
    BVH.save(bvh_path_out, anim, names=names)
    pass


def test_anim_from_pose():
    bvh_path = osp.expanduser('~/tmp/minimal5_out.bvh')
    anim, names, _ = BVH.load(bvh_path, world=True)

    target_positions = np.array([[[0,0,0],[0,0,1],[1,0,1],[1,0,0],[2,0,0]], [[0,0,0],[1,0,0],[2,0,0],[0,0,-1],[0,0,-2]]]) # minimal5
    anim, sorted_order = Animation.animation_from_positions(target_positions, anim.parents)

    bvh_path_out = bvh_path.replace('.bvh', '_pos.bvh')
    BVH.save(bvh_path_out, anim, names, positions=True)


def test_Quat():
    from Quaternions import Quaternions
    order = 'xyz'
    for t in np.arange(100):

        # degrees_from = np.random.random(3)*360-180 # sample degrees in [-180,180]
        degrees_from = np.random.random(3)*180-90 # sample degrees in [-90,90]
        # reconstruction using Quaternions class. fails for the commented example
        # degrees_from = np.array([174,147,-56]) # breaks the reconstruction retrievs [-6, 33,124] which is the complement to 180 (ignoring signs)
        radians_from = np.radians(degrees_from)
        '''
        q1_from = Quaternions.from_euler(radians_from, world=True, order=order[::-1])
        q2_from = Quaternions.from_euler(radians_from[::-1], world=False, order=order)
        radians_to = [q1_from.euler(order=order), q2_from.euler(order=order)]
        degrees_to = np.degrees(radians_to)
        if not np.allclose(degrees_to, degrees_from):
            print('Quaternions (reverse source) failed at index {}: {}->{}'.format(t, degrees_from, degrees_to[0]))
        '''

        q_from = Quaternions.from_euler(radians_from, world=True, order=order)
        degrees2_to = np.degrees(q_from.euler(order=order))
        if not np.allclose(degrees2_to, degrees_from):
            print('Quaternions failed at index {}: {}->{}'.format(t, degrees_from, degrees2_to))

        # reconstruction using scipy rotations
        r = R.from_euler(seq=order, angles=degrees_from, degrees=True)
        # rotvec = r.as_rotvec()
        # angle = np.linalg.norm(rotvec) if np.abs(np.linalg.norm(rotvec)) > 1e-10 else 1
        # axis = rotvec/angle
        degrees_to_scipy = r.as_euler(seq=order, degrees=True)
        if not np.allclose(degrees_to_scipy, degrees_from):
            print('scipy failed at index {}: {}->{}'.format(t, degrees_from, degrees_to_scipy))

        assert np.isclose(q_from.qs[0,0], r.as_quat()[-1])
        assert np.allclose(q_from.qs[0,1:], r.as_quat()[:3])
        assert np.allclose(degrees2_to, degrees_to_scipy)
        print(degrees_from)
    pass


if __name__ == '__main__':
    test_Quat()
    # IK_test_minimal()
    # IK_test_consistency_scipy()
    # IK_test_consistency_Quaternions()
    # IK_test_consistency() # clean
    # test_anim_from_pose()
    # pose_2d = np.array([[473.68356, 444.9424],
    #                       [500.9961, 448.02988],
    #                            [479.83926, 530.78564],
    #  [506.21838, 622.56885],
    #  [493.66083, 621.9954],
    #  [488.23514, 616.77313],
    #  [445.9001, 441.81586],
    #  [456.18906, 537.1581],
    #  [467.30923, 633.76935],
    #  [452.63992, 627.30396],
    #  [445.83035, 621.7957],
    #  [473.68616, 444.9206],
    #  [488.18674, 397.43405],
    #  [481.02847, 340.39694],
    #  [478.51755, 318.808],
    #  [485.76895, 297.57162],
    #  [481.02847, 340.39694],
    #  [454.01608, 359.75955],
    #  [430.05878, 415.7349],
    #  [412.99722, 452.88666],
    #  [412.99722, 452.88666],
    #  [423.38077, 446.13205],
    #  [402.1603, 466.74966],
    #  [402.1603, 466.74966],
    #  [481.02847, 340.39694],
    #  [515.4715, 456.42984],
    #  [515.4715, 456.42984],
    #  [499.2511, 448.2281],
    #  [515.06067, 479.12094],
    #  [515.06067, 479.12094]]);
    # pose_3d_world = np.array([[ -91.67900085,154.40400696,907.26098633],  [-223.23565632,163.80551039,890.53418376],
    #                           [-188.4702927,  14.07710873,475.16878684],  [-261.84053303,186.55287012, 61.4389037 ],
    #                           [-264.62785761, 28.95641816, 20.83458371],  [-266.93123238,-45.76369506, 26.87732168],
    #                           [  39.87788703,145.00248692,923.98781844],   [-11.67599155,160.89920349,484.39146568],
    #                           [-51.55029059,220.14625426, 35.83438562],   [-40.52277848, 58.26784588, 22.91115888],
    #                           [ -33.55923988,-16.02682627, 30.44793846],   [-91.69201499,154.39797455,907.35995219],
    #                           [-132.34780685,215.73018703, 1128.83955387],   [-97.16740453,202.34434815, 1383.14663731],
    #                           [-112.97074557,127.96944763, 1477.44563894],  [-120.03290535,190.96475412, 1573.39994835],
    #                           [ -97.16740453,202.34434815, 1383.14663731],  [  25.89544902,192.35947612, 1296.15713367],
    #                           [ 107.10582385,116.05029919, 1040.5062281 ],  [ 129.83814556,-48.02490094,850.94805608],
    #                           [ 129.83814556,-48.02490094,850.94805608],  [  56.46490551 -112.51779935,872.32465706],
    #                           [ 162.02074895, -108.72366914,778.28460466],  [ 162.02074895, -108.72366914,778.28460466],
    #                           [ -97.16740453,202.34434815, 1383.14663731],  [-230.36956092,203.17922166, 1311.96389029],
    #                           [-315.40537031,164.55285512, 1049.17466342],  [-350.77134811, 43.44216846,831.34726005],
    #                           [-350.77134811, 43.44216846,831.34726005],  [-301.10484019,-37.9455738, 861.50105071],
    #                           [-379.28613625,-18.24484144,711.81547212],  [-379.28613625,-18.24484144,711.81547212]])
    # rotation_GT =  np.array([[-0.91536173,0.40180837,0.02574754],  [ 0.05154812,0.18037357, -0.98224649],
    #                          [-0.39931903, -0.89778361, -0.18581953]])
    # translation_GT = np.array([[1841.10702775],  [4955.28462345],  [1563.4453959 ]])
    #
    # R.from_matrix(rotation_GT).as_euler(seq='xyz', degrees=True)  # result: array([-101.6937263 ,   23.53561481,  176.77682207])
    pass
