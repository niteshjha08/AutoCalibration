from re import S
import cv2
import numpy as np
import os
from scipy.optimize import least_squares
import glob
# from utils import estimateHomographies

def get_homographies(camera_cal_imgs_path):
    # imgs_names = os.listdir(camera_cal_imgs_path)
    # cv2.namedWindow('img',cv2.WINDOW_NORMAL)
    # cv2.namedWindow('warped',cv2.WINDOW_NORMAL)
    
    # objpoints = []
    rows = 6
    columns = 9
    # for i in range(columns):
    #         for j in range(rows):
    #             objpoints.append([[i*21.5,j*21.5]])
    # objpoints = np.array(objpoints)
    x, y = np.meshgrid(range(9), range(6))
    objpoints = np.hstack((x.reshape(54, 1), y.reshape(
        54, 1))).astype(np.float32)
    objpoints = objpoints*21.5
    objpoints = np.asarray(objpoints)

    imgpoints = []
    homographies = []
    imgs_names = sorted(glob.glob('./../data/Calibration_Imgs/*.jpg'))
    images = []
    for name in imgs_names:

        im_pt = []
        img = cv2.imread(name)
        images.append(img)
        ret,corners = cv2.findChessboardCorners(img,(columns,rows))
        
        for corner in corners:
            im_pt.append([[corner[0,0],corner[0,1]]])

        corners = corners.reshape(-1,2)
        imgpoints.append(corners)
        im_pt = np.array(im_pt)


        H,mask = cv2.findHomography(objpoints,corners)
        homographies.append(H)
    imgpoints = np.array(imgpoints)

    return homographies, imgpoints, objpoints, images

def v_ij(H,i,j):
    v = np.array([H[0,i] * H[0,j], H[0,i] * H[1,j] + H[1,i] * H[0,j], H[1,i] * H[1,j],\
            H[2,i] * H[0,j] + H[0,i] * H[2,j], H[2,i] * H[1,j] + H[1,i] * H[2,j], H[2,i] * H[2,j]])
    return v

def get_v_matrix(homographies):
    v = []
    for H in homographies:
        v_12 = v_ij(H,0,1)
        v_11 = v_ij(H,0,0)
        v_22 = v_ij(H,1,1)
        v.extend([v_12,np.subtract(v_11,v_22)])

    return v

def get_intrinsic_matrix(V):
    U,S,Vt = np.linalg.svd(V)
    B = Vt[np.argmin(S)]
    print("B:",B)
    # exit()
    B_11 = B[0]
    B_12 = B[1]
    B_22 = B[2]
    B_13 = B[3]
    B_23 = B[4]
    B_33 = B[5]
    
    v0 = (B_12* B_13 - B_11*B_23)/(B_11*B_22 - B_12**2)
    lamda = B_33 - (B_13**2 + v0*(B_12*B_13 - B_11*B_23))/B_11
    alpha = np.sqrt(lamda/B_11)
    beta = np.sqrt(lamda*B_11/(B_11*B_22 - B_12**2))
    gamma = -1*B_12*(alpha**2)*beta/lamda
    u0 = gamma*v0/beta - (B_13*alpha**2)/lamda
    K = np.array([[alpha, gamma, u0],[0, beta, v0],[0, 0, 1]])
    return K

def get_extrinsic_matrix(H, K):
    Kinv = np.linalg.inv(K)
    lamda = 1/np.linalg.norm(np.dot(Kinv,H[:,0]))
    r1 = lamda * np.dot(Kinv,H[:,0])
    r2 = lamda * np.dot(Kinv,H[:,1])
    r3 = np.cross(r1,r2)
    t = lamda * np.dot(Kinv,H[:,2])
    E = np.array([r1,r2,r3,t])
    return E.T

def error_func(params, objpoints, imgpoints, homographies):
    K = np.array([[params[0], params[4], params[2]],[0, params[1], params[3]],[0, 0, 1]])
    k1 = params[5]
    k2 = params[6]
    u0 = params[2]
    v0 = params[3]

    error = []
    # objpoints = np.hstack((objpoints[:,:,0], objpoints[:,:,1], np.zeros((54,1)), np.ones((54,1))))

    for img_points, homography in zip(imgpoints, homographies):
        # print("loop: ", np.shape(img_points), np.shape(homography))
        E = get_extrinsic_matrix(homography, K)

        for im_pt, obj_pt in zip(img_points, objpoints):
            u_actual = im_pt[0]
            v_actual = im_pt[1]
            obj_pt_hom = np.array([[obj_pt[0]],[obj_pt[1]], [0], [1]])
            cam_points = np.dot(E,obj_pt_hom)
            cam_points = cam_points/cam_points[2]
            x, y = cam_points[0], cam_points[1]

            img_coord = np.dot(K, cam_points)
            img_coord = img_coord/img_coord[2]
            u, v = img_coord[0], img_coord[1]

            sq_term = x**2 + y**2
            u_ideal = u + (u-u0)*((k1*sq_term) + k2*sq_term**2)
            v_ideal = v + (v-v0)*((k1*sq_term) + k2*sq_term**2)

            error.append(u_actual - u_ideal)
            error.append(v_actual - v_ideal)
    error =np.float64(error).flatten()
    return error

def optimize(K, objpoints, imgpoints, homographies):
    k1 = 0
    k2 = 0
    params = [K[0,0], K[1,1],K[0,2],K[1,2],K[0,1],k1,k2]
    optimized_params = least_squares(fun=error_func, x0 = params, method='lm', args=[objpoints,imgpoints,homographies])
    [alpha, beta, u0, v0, gamma, k1, k2] = optimized_params.x
    K = np.zeros((3,3))
    K[0,0] = alpha
    K[0,1] = gamma
    K[0,2] = u0
    K[1,1] = beta
    K[1,2] = v0
    K[2,2] = 1
    return K, k1, k2

def get_reprojection_points(K, objpoints, imgpoints, homographies, k1, k2):
    reproj_points = []
    u0 = K[0,2]
    v0 = K[1,2]
    for img_pts,homography in zip(imgpoints, homographies):
        E = get_extrinsic_matrix(homography, K)
        img_reproj_points = []
        for obj_pt in objpoints:
            M = np.array([[obj_pt[0]], [obj_pt[1]], [0], [1]])
            cam_points = np.dot(E,M)
            cam_points = cam_points/cam_points[2]
            x, y = cam_points[0], cam_points[1]

            proj_pt = np.dot(K, cam_points)
            proj_pt = proj_pt/proj_pt[2]
            u, v = proj_pt[0], proj_pt[1]

            sq_term = x**2 + y**2
            u_ideal = u + (u-u0)*((k1*sq_term) + k2*sq_term**2)
            v_ideal = v + (v-v0)*((k1*sq_term) + k2*sq_term**2)
            img_reproj_points.append([u_ideal,v_ideal])
        reproj_points.append(img_reproj_points)
    return reproj_points

def draw_reprojection_img(img, imgpoints, reproj_points):
    for reproj_pt, imgpt in zip(reproj_points, imgpoints):
        # print('centers: ', imgpt, reproj_pt)
        cv2.circle(img,(int(reproj_pt[0]),int(reproj_pt[1])),4,(0,255,0),-1)
        cv2.circle(img,(int(imgpt[0]),int(imgpt[1])),10,(0,0,255),2)
    cv2.imshow('reprojection', img)
    cv2.waitKey(0)

def reproject(images, imgpoints, reproj_points):
    for image, imgpt, reproj_pt in zip(images, imgpoints, reproj_points):
        draw_reprojection_img(image,imgpt, reproj_pt)
    

def main(camera_cal_imgs_path):

    homographies, imgpoints, objpoints, images = get_homographies(camera_cal_imgs_path)

    v_matrix = get_v_matrix(homographies)

    V = np.array(v_matrix)
    K = get_intrinsic_matrix(V)
    print("K initial:\n", K)

    K_final, k1, k2 = optimize(K, objpoints, imgpoints, homographies)
    print("final K:",K_final)
    print("k1, k2:",k1,k2)

    reproj_points = get_reprojection_points(K_final, objpoints, imgpoints, homographies, k1, k2)
    print("shape of reprojection points:",np.shape(reproj_points))
    print("shape of image points:",np.shape(imgpoints))
    reproj_points = np.squeeze(reproj_points)
    print("shape of reprojection points:",np.shape(reproj_points))
    reproject(images, imgpoints, reproj_points)




if __name__=="__main__":
    camera_cal_imgs_path = r'./../data/Calibration_Imgs/'
    main(camera_cal_imgs_path)