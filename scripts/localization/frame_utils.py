import numpy as np
import cv2
import matplotlib.pyplot as plt

class MonoFrame:
    def __init__(self):
        self.K = np.array([ [405.64, 0, 189.91],
                            [0, 405.59, 139.91],
                            [0, 0, 1]], dtype = "double")
        self.k1: -0.37 
        self.k2: 0.2 
        self.p1: 0.0 
        self.p2: 0.0 
        self.width: 384 
        self.height: 288 
        self.orb = cv2.ORB_create()

def GetImage(data, i_frame):
    image_filename = data.img_folder + 'frame_'+str(i_frame).zfill(6)+'.png'
    img = cv2.imread(image_filename, 1)[..., ::-1]
    return img

def CompareTwoFrames(img_prev, img_curr):
    img_prev_gray = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
    img_curr_gray = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    img_prev_kp, img_prev_des = orb.detectAndCompute(img_prev_gray, None)
    img_curr_kp, img_curr_des = orb.detectAndCompute(img_curr_gray, None)
    matcher = cv2.BFMatcher()
    matches = matcher.match(img_prev_des, img_curr_des)
    
    min_distance = 1e4
    max_distance = 0
    for x in matches:
        if x.distance < min_distance: min_distance = x.distance
        if x.distance > max_distance: max_distance = x.distance
    #print('min distance: %f' % min_distance)
    #print('max distance: %f' % max_distance)
    good_match = []
    for x in matches:
        if x.distance <= max(3 * min_distance, 30):
            good_match.append(x)
            
    final_img = cv2.drawMatches(img_prev, img_prev_kp, img_curr, img_curr_kp, good_match,None)
    print(len(good_match))
    # Show the final image
    cv2.imshow("Matches", final_img)
    cv2.imwrite("matchings.png", final_img)
    cv2.waitKey(0)

def PoseEstimate2D2D(img_prev, img_curr, orb, K):
    img_prev_kp, img_prev_des = orb.detectAndCompute(img_prev, None)
    img_curr_kp, img_curr_des = orb.detectAndCompute(img_curr, None)
    #distort = np.zeros((4,1))

    '''
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    #matches = flann.knnMatch(img_prev_des, img_curr_des, k=2)
    matches = flann.knnMatch(   np.asarray(img_prev_des, np.float32), 
                                np.asarray(img_curr_des, np.float32), k=2)
    '''

    if ( (img_prev_des is not None) and (img_curr_des is not None) ):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING) # crossCheck=True
        matches = bf.match(img_prev_des, img_curr_des) 
    
        min_distance = 1e4
        max_distance = 0
        for x in matches:
            if x.distance < min_distance: min_distance = x.distance
            if x.distance > max_distance: max_distance = x.distance
        #print('min distance: %f' % min_distance)
        #print('max distance: %f' % max_distance)
        good_match = []
        for x in matches:
            if x.distance <= max(4 * min_distance, 30):
                good_match.append(x)
        n_matches = len(good_match)
        #print('n_good_matches: %d' % len(good_match))
        #img_result = cv2.drawMatches(img_prev, img_prev_kp, img_curr, img_curr_kp, good_match, outImg=None)
        #plt.imshow(img_result[:,:,::-1])
        #plt.show()
    else:
        good_match = []
        n_matches = 0
    
    points1 = []
    points2 = []
    
    if n_matches > 420: 
        print(n_matches)
        for i in good_match:
            points1.append(list(img_prev_kp[i.queryIdx].pt))
            points2.append(list(img_curr_kp[i.trainIdx].pt))
        points1 = np.array(points1)
        points2 = np.array(points2)
        
        E,mask = cv2.findEssentialMat(points1, points2, K) #opencv4        
        num,R,t,mask = cv2.recoverPose(E, points1, points2, K)
        #print(R)
        #print(t)
        
    else:
        R = None
        t = None
    
    return n_matches, R, t, points1, points2
        


if __name__ == "__main__":
    frame = MonoFrame()
    
    img_prev_filename = '../../data/girona/test_frames/frame_000081.png'
    img_curr_filename = '../../data/girona/test_frames/frame_000083.png'

    img_prev = cv2.imread(img_prev_filename)
    img_curr = cv2.imread(img_curr_filename)

    CompareTwoFrames(img_prev, img_curr)

    #PoseEstimate2D2D(img_prev, img_curr, frame.K)
