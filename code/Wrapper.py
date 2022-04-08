import cv2
import numpy as np
import os

def get_homographies(camera_cal_imgs_path):
    imgs_names = os.listdir(camera_cal_imgs_path)
    cv2.namedWindow('img',cv2.WINDOW_NORMAL)
    cv2.namedWindow('warped',cv2.WINDOW_NORMAL)
    
    objpoints = []
    rows = 6
    columns = 9
    for i in range(columns):
            for j in range(rows):
                objpoints.append([[i*100,j*100]])
    objpoints = np.array(objpoints)

    imgpoints = []
    homographies = []
    for name in imgs_names:
        # print('loading image')
        im_pt = []
        img_path = os.path.join(camera_cal_imgs_path,name)
        img = cv2.imread(img_path)
        
        ret,corners = cv2.findChessboardCorners(img,(rows,columns))
        
       
        for corner in corners:
            cv2.circle(img,(int(corner[0,0]),int(corner[0,1])),5,(0,0,255),-1)
            # imgpoints.append([[int(corner[0,0]),int(corner[0,1])]])
            im_pt.append([[int(corner[0,0]),int(corner[0,1])]])
        imgpoints.extend(im_pt)
        im_pt = np.array(im_pt)
        # print(np.shape(objpoints))
        # print(np.shape(im_pt))

        H,mask = cv2.findHomography(im_pt,objpoints)
        homographies.append(H)
    imgpoints = np.array(imgpoints)

    return homographies, imgpoints, objpoints

def v_ij(H,i,j):
    v = [H[0,i] * H[0,j], H[0,0] * H[1,j] + H[1,i] * H[0,j], H[1,i] * H[1,j],\
            H[2,i] * H[0,j] + H[0,i] * H[2,j], H[2,i] * H[1,j] + H[1,i] * H[2,j], H[2,i] * H[2,j]]
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
    gamma = B_12*(alpha**2)*beta/lamda
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

def main(camera_cal_imgs_path):

    homographies, imgpoints, objpoints = get_homographies(camera_cal_imgs_path)
    print('imgpoints shape:', imgpoints.shape)
    print('objpoints shape:', objpoints.shape)
    exit()
    v_matrix = get_v_matrix(homographies)
    # print(np.shape(v_matrix))
    V = np.array(v_matrix)
    K = get_intrinsic_matrix(V)
    H_test = homographies[0]
    E = get_extrinsic_matrix(H_test, K)
    print(E.shape)


if __name__=="__main__":
    camera_cal_imgs_path = r'./../data/Calibration_Imgs/'
    main(camera_cal_imgs_path)