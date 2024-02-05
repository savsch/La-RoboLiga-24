import gym
import os
import time as t
import LaRoboLiga24
import cv2
import pybullet as p
import numpy as np

CAR_LOCATION = [0,0,1.5]

BALLS_LOCATION = dict({
    'red': [7, 4, 1.5],
    'blue': [2, -6, 1.5],
    'yellow': [-6, -3, 1.5],
    'maroon': [-5, 9, 1.5]
})
BALLS_LOCATION_BONOUS = dict({
    'red': [9, 10, 1.5],
    'blue': [10, -8, 1.5],
    'yellow': [-10, 10, 1.5],
    'maroon': [-10, -9, 1.5]
})

HUMANOIDS_LOCATION = dict({
    'red': [11, 1.5, 1],
    'blue': [-11, -1.5, 1],
    'yellow': [-1.5, 11, 1],
    'maroon': [-1.5, -11, 1]
})

VISUAL_CAM_SETTINGS = dict({
    'cam_dist'       : 13,
    'cam_yaw'        : 0,
    'cam_pitch'      : -110,
    'cam_target_pos' : [0,4,0]
})


os.chdir(os.path.dirname(os.getcwd()))
env = gym.make('LaRoboLiga24',
    arena = "arena2",
    car_location=CAR_LOCATION,
    ball_location=BALLS_LOCATION,  # toggle this to BALLS_LOCATION_BONOUS to load bonous arena
    humanoid_location=HUMANOIDS_LOCATION,
    visual_cam_settings=VISUAL_CAM_SETTINGS
)

"""
CODE AFTER THIS
"""
forwardSpeed = 6.5
def calculateAngle(line):
    x1, y1, x2, y2 = line
    angle_rad = np.arctan2(y2 - y1, x2 - x1)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def findContours(img, lowerRange, UpperRange):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerRange, UpperRange)
    res = cv2.bitwise_and(img, img, mask=mask)
    imgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(imgray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours, imgray

env.open_grip()
def scpp(lower_color, upper_color):
    debt = 0
    holding = False
    ballInSearch = False
    goalpostInSearch = False
    yetToShoot = True
    while yetToShoot:
        img = env.get_image(cam_height=0, dims=[600, 600])
        goalBounds = None
        contours, gray = findContours(img, lower_color, upper_color)
        ballDetectedInLoop = False
        for c in contours:
            ballDetectedInContour = False
            area = cv2.contourArea(c)
            if area < 700: continue
            x,y,w,h = cv2.boundingRect(c)
            if (True if (1.1 < w * h / area < 1.5) else False) if (0.85 < w / h < 1.2) else False:
                # ball
                ballDetectedInContour = ballDetectedInLoop = True
                if ballInSearch:
                    ballInSearch=False
                    env.move([0,0,0,0])
                    t.sleep(0.25)
                if not holding:
                    centre = x + w / 2
                    TOLERANCE = 10
                    if centre > (300+TOLERANCE):
                        S = 0.25 + (centre - 300 - TOLERANCE) / 20
                        env.move([S,-S,S,-S])
                    elif centre < (300-TOLERANCE):
                        S = 0.25 + (300 - TOLERANCE - centre) / 20
                        env.move([-S,S,-S,S])
                    else:
                        s = forwardSpeed # + (TOLERANCE-np.abs(300-centre))/2
                        env.move([s,s,s,s])
                        debt+=1
                        if area>40700:
                            env.move([0,0,0,0])
                            t.sleep(.2)
                            env.close_grip()
                            t.sleep(.25)
                            holding = True
                            debt*=5.5
                            while debt > 0:
                                env.move([-forwardSpeed,-forwardSpeed,-forwardSpeed,-forwardSpeed])
                                debt-=1
                                p.stepSimulation()
                            
                            env.move([0,0,0,0])
                            t.sleep(.2)

                

            if holding and (not ballDetectedInContour) and y<350 and h>25 and w > 80 and (y+h)<450:
                # candidate goalpost
                
                goalpostImg = gray[y:y+h,x:x+w]
                goalpostImg[goalpostImg>0] = 200  # this line is needed only for blue colour (which is way darker than the rest)
                edges = cv2.Canny(goalpostImg,50,150,apertureSize = 3)
                lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=20,lines=np.array([]), minLineLength=25,maxLineGap=80)
                if lines is not None:
                    a,b,c = lines.shape
                    vertLines=[]
                    for i in range(a):
                        if not (85 < np.abs(calculateAngle(lines[i][0])) < 95): continue
                        vertLines.append(lines[i][0][0])
                    if len(vertLines)>1 and (max(vertLines)-min(vertLines))>100:
                        # goalpost confirmed

                        # accounting for the human within:
                        nimg = img[y+h//2:y+h,x:x+w]
                        cnt,_ = findContours(nimg, np.array([8, 100, 130], dtype=np.uint8), np.array([28, 255, 255], dtype=np.uint8))
                        x1,y1,x2,y2=1000,1000,0,0
                        for human in cnt:
                            X,Y,W,H = cv2.boundingRect(human)
                            x1=min(x1,X)
                            x2=max(x2,X+W)
                        
                        if      (x1+x2)/2    <   w - (x1+x2)/2:
                            goalBounds = (x+x2, x+w)
                        else:
                            goalBounds = (x, x+x1)

                        if goalpostInSearch:
                            goalpostInSearch = False
                            env.move([0,0,0,0])
                            t.sleep(0.25)
                        centre = (goalBounds[0] + goalBounds[1]) / 2
                        if centre > 315:
                            S = 0.05 + (centre - 315) / 20
                            env.move([S,-S,S,-S])
                        elif centre < 285:
                            S = 0.05 + (285 - centre) / 20
                            env.move([-S,S,-S,S])
                        else:
                            env.move([0,0,0,0])
                            t.sleep(.2)
                            env.open_grip()
                            t.sleep(.25)
                            yetToShoot = False
                            env.shoot(100)

        if (not holding) and (not ballDetectedInLoop):
            env.move([-12,12,-12,12])
            ballInSearch=True
        if (holding) and (goalBounds is None):
            env.move([12,-12,12,-12])
            goalpostInSearch = True

        p.stepSimulation()



scpp(np.array([101, 100, 100], dtype=np.uint8), np.array([121, 255, 255], dtype=np.uint8))
t.sleep(0.2)
scpp(np.array([25, 100, 100], dtype=np.uint8), np.array([35, 255, 255], dtype=np.uint8))
t.sleep(0.2)
scpp(np.array([150, 50, 50], dtype=np.uint8), np.array([170, 255, 255], dtype=np.uint8))
t.sleep(0.2)
scpp(np.array([0, 50, 50], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8))

#done
t.sleep(5)


env.close()
