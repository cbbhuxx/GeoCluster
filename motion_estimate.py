import cv2
import numpy as np 
import math

class Homography():
    def __init__(self, first_frame, verbose=True):
        """
            Implements a class that estimates the Homography using point correspondances. 
        """
        self.old = first_frame
        self.old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.orb = cv2.ORB_create()
        self.old_kp, self.old_des = self.orb.detectAndCompute(self.old_gray, None)
        self.old_frame = first_frame
        self.old_yaw = 0.0
        self.old_t = np.array([0.0,0.0])
        self.verbose = verbose

    def estimate(self, frame):
        """ This method estimates translation and yaw from the homography
        """
        new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_kp, new_des = self.orb.detectAndCompute(new_gray, None)
        
        if new_des is not None and self.old_des is not None:
            matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
            matches = matcher.match(self.old_des, new_des)
            matches = sorted(matches, key= lambda x:x.distance)
            
            npoints = len(matches)
            
            if npoints > 100:
                matches = matches[0:30]
                points1 = np.zeros((len(matches),2), dtype=np.float32)
                points2 = np.zeros((len(matches),2), dtype=np.float32)

                for i,match in enumerate(matches):
                    points1[i, :] = self.old_kp[match.queryIdx].pt
                    points2[i, :] = new_kp[match.trainIdx].pt
            
                h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
                nransac = (mask==1).sum()
                
                if nransac > 10:
                    matchesMask = mask.ravel().tolist()
                    img = cv2.drawMatches(self.old_frame, self.old_kp, frame, new_kp, matches, None, matchesMask=matchesMask)
                    hn = h / h[2,2]
                    t = hn[:2,2]

                    sy = math.sqrt(hn[0,0]*hn[0,0] + hn[1,0]*hn[1,0]) 
                    singular = sy < 1e-6

                    if not singular:
                        yaw = math.atan2(hn[1,0],hn[0,0])
                    else:
                        yaw = 0
    
                    self.old_yaw = yaw
                    self.old_t = t
                    
                else:
                    if self.verbose:
                        print("Warning! not enough points in ransac")
                    yaw = self.old_yaw
                    t = self.old_t
                    img = frame
            else: 
                if self.verbose:
                    print("Warning! not enough matched points")
                yaw = self.old_yaw 
                t = self.old_t 
                img = frame

        else:                  
            yaw = self.old_yaw
            t = self.old_t 
            img = frame

        self.old_frame = frame
        self.old_kp = new_kp
        self.old_des = new_des
        
        return yaw, t, img


