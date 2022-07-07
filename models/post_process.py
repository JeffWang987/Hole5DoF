import numpy as np
import cv2
import copy


def geometric_filter(left_pts, right_pts, extrin=None, intrin=None, resize_fac=2, thres=0.01, image_left=None, image_right=None, show=False, id=0, log_dir='./logs'):
    """ use geometirc principles (epipolar constraints to filter noisy holes)

    Args:
        left_pts (numpy.array): N, 2
        right_pts (numpy.array): N, 2
        extrin (numpy.array): 4, 4
        intrin (numpy.array): 3, 3 original downscaled 4x
        resize_fac (int, optional): _description_. Defaults to 2.
        thres (int, optional): _description_. Defaults to 1.
    """
    if extrin is None:
        # resize_fac = 1
        # step1. preprocess cam parameters
        L_to_R = np.load('./cam_param/L_TO_R.npy')
        KR = np.load('./cam_param/KR.npy')
        KL = np.load('./cam_param/KL.npy')
        
        RT_R = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
        PR = KR @ RT_R
        RT_L = L_to_R[:3,:]  # 3, 4
        PL = KL @ RT_L 
    # step2. compute fundamental matrix
    # t1, t2, t3 = R_to_L[:3, 3]
    # R = R_to_L[:3, :3]
    # t_ = np.array([
    #     [0, -t3, t2],
    #     [t3, 0, -t1],
    #     [-t2, t1, 0]
    # ])
    # TODO check fundamental matrix accuracy
    F = np.load('./cam_param/F.npy')  # cord_R @ F @ cord_L.T
    crop_bias = int((2448-2048) / 2 / resize_fac)

    # F = np.linalg.inv(KL).T @ t_ @ R @ np.linalg.inv(KR)  # 3x3
    # print(F)
    # print(F_)

    # step2. compute matching pts
    left_pts = np.concatenate([left_pts, np.ones((left_pts.shape[0], 1))], axis=1)  # TODO check x,y,1
    right_pts = np.concatenate([right_pts, np.ones((right_pts.shape[0], 1))], axis=1)
    
    # step2.1 compute epipolar line
    left_pts[:, 0] = left_pts[:, 0] + crop_bias
    right_eps = F @ left_pts.T  # 3xN
    # step2.2 compute distance
    right_pts[:, 0] = right_pts[:, 0] + crop_bias
    dists = right_pts @ right_eps / np.sqrt(right_eps[0:1]**2+right_eps[1:2]**2)  # NxN, 每一列对应的是right_pt对应的N个left_pt的距离
    # print(np.abs(dists).min())
    # step2.3 filter
    geo_mask = np.abs(dists)  # row : right,  colume: left

    # step3. show
    key_pts_imgR = []
    key_pts_imgL = []
    
    if show and image_left is not None:
        N = geo_mask.shape[0]
        l_img = np.pad(image_left, ((0,0), (crop_bias,crop_bias), (0,0)), 'constant')
        r_img = np.pad(image_right, ((0,0), (crop_bias,crop_bias), (0,0)), 'constant')
        for i in range(N):
            left_center = left_pts[i, :2]
            right_center = right_pts[i, :2]
            cv2.circle(l_img, (int(left_center[0]), int(left_center[1])), radius=5, color=(255,0,0), thickness=-1)
            cv2.circle(r_img, (int(right_center[0]), int(right_center[1])), radius=5, color=(255,0,0), thickness=-1)
        ori_det_img = np.hstack((l_img, r_img))

        # img_l = cv2.imread('./cam_param/chessboardL.jpg')
        # img_r = cv2.imread('./cam_param/chessboardR.jpg')
        l_img = np.pad(image_left, ((0,0), (crop_bias,crop_bias), (0,0)), 'constant')
        r_img = np.pad(image_right, ((0,0), (crop_bias,crop_bias), (0,0)), 'constant')

        for i in range(N):
            min_val = geo_mask[:, i].min()  # 对于左边的一个点，右边所有点距离最小值
            # 如果满足条件，则说明左边这个点在右边有匹配，可以画图，并寻找匹配点
            if min_val < thres:
                color = tuple(np.random.randint(0, 255, 3).tolist())
                # 画ref点（左图）
                left_center = left_pts[i, :2]
                cv2.circle(l_img, (int(left_center[0]), int(left_center[1])), radius=5, color=color, thickness=-1)
                # cv2.circle(img_l, (int(left_center[0]), int(left_center[1])), radius=5, color=color, thickness=-1)
                # 画match点 (右图)
                matched_right_pts = right_pts[geo_mask[:, i]<thres][:, :2]  # N2, 2
                for match_i, matched_pt in enumerate(matched_right_pts):
                    this_err = geo_mask[:, i][geo_mask[:, i]<thres][match_i]
                    r_img = cv2.putText(r_img, '{:.2f}'.format(this_err), (int(matched_pt[0]), int(matched_pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.circle(r_img, (int(matched_pt[0]), int(matched_pt[1])), radius=5, color=color, thickness=-1)
                    # cv2.circle(img_r, (int(matched_pt[0]), int(matched_pt[1])), radius=5, color=color, thickness=-1)
                # 画极线
                ep_a, ep_b, ep_c = right_eps[:, i]
                start_x = 0
                start_y = -ep_c / ep_b
                end_x = r_img.shape[1]  # W
                end_y = -(end_x * ep_a + ep_c) / ep_b
                x0, y0 = map(int, [start_x, start_y])
                x1, y1 = map(int, [end_x, end_y])
                r_img = cv2.line(r_img, (x0, y0), (x1, y1), color, 2)
                # img_r = cv2.line(img_r, (x0, y0), (x1, y1), color, 2)


        this_show_img = np.hstack((l_img, r_img))
        this_show_img = np.vstack([ori_det_img, this_show_img])
        # this_show_chess = np.hstack((img_l, img_r))
        # this_show_img = np.vstack([ori_det_img, this_show_img, this_show_chess])
        cv2.imwrite(log_dir+'/geo_filter/'+ '/{}.jpg'.format(id), this_show_img)
                
        #         key_pts_imgR.append(right_center)
        #         key_pts_imgL.append(left_center)

        # # 3D pts
        # p3ds = []
        # for uvR, uvL in zip(key_pts_imgR, key_pts_imgL):
        #     _p3d = DLT(PR, PL, uvR, uvL)
        #     p3ds.append(_p3d)
        # p3ds = np.array(p3ds)
        # cam_dist = np.sqrt((p3ds**2).sum(1))

    return geo_mask  # NxN for photometric filter


def DLT(P1, P2, point1, point2):
    A = [point1[1] * P1[2, :] - P1[1, :],
         P1[0, :] - point1[0] * P1[2, :],
         point2[1] * P2[2, :] - P2[1, :],
         P2[0, :] - point2[0] * P2[2, :]
         ]
    A = np.array(A).reshape((4, 4))
    # print('A: ')
    # print(A)

    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices=False)

    # print('Triangulated point: ')
    # print(Vh[3, 0:3] / Vh[3, 3])
    return Vh[3, 0:3] / Vh[3, 3]


def DLT_plane(pts):
    """
    plane func: z = ax + by + c
    Args:
        pts : [N, 3]
    """
    N = pts.shape[0]
    # 创建系数矩阵A
    A = np.ones((N, 3))
    A[:, :2] = pts[:, :2]
    # 创建矩阵b
    B = pts[:, 2:3]
    # 通过X=(AT*A)-1*AT*b直接求解
    A_T = A.T
    A1 = np.dot(A_T,A)
    A2 = np.linalg.inv(A1)
    A3 = np.dot(A2,A_T)
    X= np.dot(A3, B)
    a, b, c = X[0,0], X[1,0], X[2,0]
    print('The plane function is : z = %.3f * x + %.3f * y + %.3f'%(a, b, c))
    #计算标准差
    sig = np.sqrt(np.mean(((a * pts[:,0] + b * pts[:,1] + c - b[:,0])**2)))
    print ('The MSE is {}'.format(sig))
    return a, b, c

def LS_plane(pts):
    """
    plane func: z = ax + by + c
    Args:
        pts : [N, 3]
    """
    N = pts.shape[0]
    # 创建系数矩阵A
    A = np.zeros((3,3))
    A[0,0] = (pts[:, 0]**2).sum()
    A[0,1] = (pts[:, 0]*pts[:, 1]).sum()
    A[0,2] = (pts[:, 0]).sum()
    A[1,0] = A[0,1]
    A[1,1] = (pts[:, 1]**2).sum()
    A[1,2] = (pts[:, 1]).sum()
    A[2,0] = A[0, 2]
    A[2,1] = A[1, 2]
    A[2,2] = N
    # 创建矩阵b
    B = np.zeros((3,1))
    B[0,0] = pts[:,0]*pts[:,2]
    B[1,0] = pts[:,1]*pts[:,2]
    B[2,0] = pts[:,2]

    #求解X
    A_inv=np.linalg.inv(A)
    X = np.dot(A_inv, B)
    a, b, c = X[0,0], X[1,0], X[2,0]
    print('The plane function is : z = %.3f * x + %.3f * y + %.3f'%(a, b, c))
    #计算标准差
    sig = np.sqrt(np.mean(((a * pts[:,0] + b * pts[:,1] + c - b[:,0])**2)))
    print ('The MSE is {}'.format(sig))
    return a, b, c
