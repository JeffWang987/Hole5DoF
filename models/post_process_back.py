import numpy as np
import cv2
import copy
def geometric_filter(left_pts, right_pts, extrin=None, intrin=None, resize_fac=2, thres=1, image_left=None, image_right=None, show=False, id=0):
    """ use geometirc principles (epipolar constraints to filter noisy holes)

    Args:
        left_pts (numpy.array): N, 2
        right_pts (numpy.array): N, 2
        extrin (numpy.array): 4, 4
        intrin (numpy.array): 3, 3 original downscaled 4x
        resize_fac (int, optional): _description_. Defaults to 2.
        thres (int, optional): _description_. Defaults to 1.
    """
    # if extrin is None:
    #     # resize_fac = 1
    #     # step1. preprocess cam parameters
    #     R_to_L = np.load('./cam_param/C1_TO_C2.npy')
    #     KR = np.load('./cam_param/K1.npy') * 4
    #     KL = np.load('./cam_param/K2.npy') * 4  # ori downscaled 4x
        
    #     crop_bias = (2448-2048) / 2.
    #     KR[0, 2] -= crop_bias
    #     KL[0, 2] -= crop_bias

    #     KR /= resize_fac
    #     KL /= resize_fac
    #     KR[-1, -1] = 1
    #     KL[-1, -1] = 1

    # step2. compute fundamental matrix
    # t1, t2, t3 = R_to_L[:3, 3]
    # R = R_to_L[:3, :3]
    # t_ = np.array([
    #     [0, -t3, t2],
    #     [t3, 0, -t1],
    #     [-t2, t1, 0]
    # ])
    # TODO check fundamental matrix accuracy
    F = np.load('./cam_param/F.npy')[0]
    crop_bias = int((2448-2048) / 2 / resize_fac)

    # F = np.linalg.inv(KL).T @ t_ @ R @ np.linalg.inv(KR)  # 3x3
    # print(F)
    # print(F_)

    # step2. compute matching pts
    left_pts = np.concatenate([left_pts, np.ones((left_pts.shape[0], 1))], axis=1)  # TODO check x,y,1
    right_pts = np.concatenate([right_pts, np.ones((right_pts.shape[0], 1))], axis=1)
    
    # step2.1 compute epipolar line
    right_pts[:, 0] = right_pts[:, 0] + crop_bias
    left_eps = F @ right_pts.T  # 3xN
    # step2.2 compute distance
    left_pts[:, 0] = left_pts[:, 0] + crop_bias
    dists = left_pts @ left_eps / np.sqrt(left_eps[0:1]**2+left_eps[1:2]**2)  # NxN, 每一列对应的是right_pt对应的N个left_pt的距离
    # print(np.abs(dists).min())
    # step2.3 filter
    geo_mask = np.abs(dists) < thres

    # step3. show
    if show and image_left is not None:
        N = geo_mask.shape[0]
        for i in range(N):
            if geo_mask[:, i].sum() > 0:
                l_img = np.pad(image_left, ((0,0), (crop_bias,crop_bias), (0,0)), 'constant')
                r_img = np.pad(image_right, ((0,0), (crop_bias,crop_bias), (0,0)), 'constant')
                img_l = cv2.imread('./cam_param/chessboard2.jpg')
                img_r = cv2.imread('./cam_param/chessboard1.jpg')
                # 画ref点
                right_center = right_pts[i, :2]
                # color = tuple(np.random.randint(0, 255, 3).tolist())
                color = (0,255,0)
                cv2.circle(r_img, (int(right_center[0]), int(right_center[1])), radius=5, color=color, thickness=-1)
                cv2.circle(img_r, (int(right_center[0]), int(right_center[1])), radius=5, color=color, thickness=-1)
                # 画match点
                matched_left_pts = left_pts[geo_mask[:, i]!=0][:, :2]  # N2, 2
                for matched_pt in matched_left_pts:
                    cv2.circle(l_img, (int(matched_pt[0]), int(matched_pt[1])), radius=5, color=color, thickness=-1)
                    cv2.circle(img_l, (int(matched_pt[0]), int(matched_pt[1])), radius=5, color=color, thickness=-1)

                mis_matched_left_pts = left_pts[geo_mask[:, i]==0][:, :2]  # N2, 2
                for mis_matched_pt in mis_matched_left_pts:
                    cv2.circle(l_img, (int(mis_matched_pt[0]), int(mis_matched_pt[1])), radius=5, color=(0,0,255), thickness=-1)

                # 画极线
                ep_a, ep_b, ep_c = left_eps[:, i]
                start_x = 0
                start_y = -ep_c / ep_b
                end_x = l_img.shape[1]
                end_y = -(end_x*ep_a + ep_c)/ep_b

                x0, y0 = map(int, [start_x, start_y])
                x1, y1 = map(int, [end_x, end_y])
                l_img= cv2.line(l_img, (x0, y0), (x1, y1), color, 2)
                img_l= cv2.line(img_l, (x0, y0), (x1, y1), color, 2)
                this_show_img = np.hstack((l_img, r_img))
                this_show_chess = np.hstack((img_l, img_r))
                this_show_img = np.vstack([this_show_img, this_show_chess])
                cv2.imwrite('./logs/geo_filter/'+ '/{}_{}.jpg'.format(id, i), this_show_img)

                
        # mis_matched_left_pts = left_pts[geo_mask.sum(1)==0][:, :2]  # N2, 2
        # for mis_matched_pt in mis_matched_left_pts:
        #     cv2.circle(image_left, (int(mis_matched_pt[0]), int(mis_matched_pt[1])), radius=10, color=(0,0,0), thickness=1)
            
        # cv2.imwrite('./logs/geo_filter/'+ '/left{}.jpg'.format(id), image_left)
        # cv2.imwrite('./logs/geo_filter/'+ '/right{}.jpg'.format(id), image_right)


    return geo_mask  # NxN for photometric filter