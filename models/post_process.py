import numpy as np
import cv2

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
    if extrin is None:
        # resize_fac = 1
        # step1. preprocess cam parameters
        C1_to_C2 = np.load('./cam_param/C1_TO_C2.npy')
        K1 = np.load('./cam_param/K1.npy') * 4
        K2 = np.load('./cam_param/K2.npy') * 4  # ori downscaled 4x
        
        crop_bias = (2448-2048) / 2.
        K1[0, 2] -= crop_bias
        K2[0, 2] -= crop_bias

        K1 /= resize_fac
        K2 /= resize_fac
        K1[-1, -1] = 1
        K2[-1, -1] = 1

    # step2. compute fundamental matrix
    t1, t2, t3 = C1_to_C2[:3, 3]
    R = C1_to_C2[:3, :3]
    t_ = np.array([
        [0, -t3, t2],
        [t3, 0, -t1],
        [-t2, t1, 0]
    ])
    # TODO check fundamental matrix accuracy
    F_ = np.load('./cam_param/F.npy')
    F = np.linalg.inv(K2).T @ t_ @ R @ np.linalg.inv(K1)  # 3x3
    # print(F)
    # print(F_)

    # step2. compute matching pts
    left_pts = np.concatenate([left_pts, np.ones((left_pts.shape[0], 1))], axis=1)  # TODO check x,y,1
    right_pts = np.concatenate([right_pts, np.ones((right_pts.shape[0], 1))], axis=1)
    
    # step2.1 compute epipolar line
    left_eps = F @ right_pts.T  # 3xN
    # step2.2 compute distance
    dists = left_pts @ left_eps  # NxN, 每一列对应的是right_pt对应的N个left_pt的距离
    # print(np.abs(dists).min())
    # step2.3 filter
    geo_mask = np.abs(dists) < thres

    # step3. show
    if show and image_left is not None:
        N = geo_mask.shape[0]
        for i in range(N):
            if geo_mask[:, i].sum() > 0:
                # 画ref点
                right_center = right_pts[i, :2]
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.circle(image_right, (int(right_center[0]), int(right_center[1])), radius=5, color=color, thickness=-1)
                # 画match点
                matched_left_pts = left_pts[geo_mask[:, i]!=0][:, :2]  # N2, 2
                for matched_pt in matched_left_pts:
                    cv2.circle(image_left, (int(matched_pt[0]), int(matched_pt[1])), radius=5, color=color, thickness=-1)
                # 画极线
                line = cv2.computeCorrespondEpilines(right_pts[i, :2].reshape(1, 1, 2), 2, F)
                line = line.reshape(3)

                x0, y0 = map(int, [0, -line[2]/line[1]])
                x1, y1 = map(int, [image_left.shape[-1], -(line[2]+line[0]*image_right.shape[-1])/line[1]])
                image_left= cv2.line(image_left, (x0, y0), (x1, y1), color, 2)
        cv2.imwrite('./logs/geo_filter/'+ '/left{}.jpg'.format(id), image_left)
        cv2.imwrite('./logs/geo_filter/'+ '/right{}.jpg'.format(id), image_right)


    return geo_mask  # NxN for photometric filter