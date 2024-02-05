import gym
import os
import time as t
import LaRoboLiga24
import cv2
import pybullet as p
import numpy as np

CAR_LOCATION = [-25.5,0,1.5]


VISUAL_CAM_SETTINGS = dict({
    'cam_dist'       : 30,      #cam_dist changed from 13 to 30 for better view
    'cam_yaw'        : 0,
    'cam_pitch'      : -90.01,
    'cam_target_pos' : [0,4,0]
})


os.chdir(os.path.dirname(os.getcwd()))
env = gym.make('LaRoboLiga24',
    arena = "arena1",
    car_location=CAR_LOCATION,
    visual_cam_settings=VISUAL_CAM_SETTINGS
)

"""
FULLMETAL GENESIS
{Code after this}
"""
timeStep = 1.0 / 19.0
p.setPhysicsEngineParameter(fixedTimeStep=timeStep)
# Changed the timeStep for optimization

p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,0)
# Disabled shadows for optimization

FINISHED = False   # FINISHED is True once bot detects the Finish Line
turnMultiplier=0
p1 = (256, 121)
p2 = (256, 60)
turnMultiplier = 0

def DistPointToLine(point):
    global p1, p2
    x0, y0 = point
    x1, y1 = p1
    x2, y2 = p2
    midX, midY = (x1+x2)/2, (y1+y2)/2
    distance = np.sqrt((midX-x0)**2 * (midY-y0)**2)
    return distance
# Function to find Distance from point to line
    

def AvgDistances(contours):
    global p1, p2
    totalLeft, meanLeft = 0, 0.0
    totalRight, meanRight = 0, 0.0
    for contour in contours:
        for point in contour[:,0,:]:
            dist = DistPointToLine(point)
            side = np.sign((p2[0] - p1[0]) * (point[1] - p1[1]) -
                           (p2[1] - p1[1]) * (point[0] - p1[0]))
            if side == 1:
                totalRight+=1
                meanRight += (dist - meanRight)/totalRight
            else:
                totalLeft+=1
                meanLeft += (dist - meanLeft)/totalLeft
    return meanLeft, meanRight
# Averages distances of the edge contours of left and right


def CalculateAngle(line):
    x1, y1, x2, y2 = line[0]
    angle_rad = np.arctan2(y2 - y1, x2 - x1)
    angle_deg = np.degrees(angle_rad)
    return angle_deg
# Calculates angle for lines to help detect the Finish (the only straight) line.


def FindFinishLine(gray):
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=120, maxLineGap=10)
    if lines is not None and len(lines) > 0:
        for l in lines:
            angle=CalculateAngle(l)
            if np.abs(angle)<0.0290:
                global FINISHED
                FINISHED = True
# Detects the finsh line


def FindContours(img):
    global turnMultiplier
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(imgray, 190, 255)
    res = cv2.bitwise_and(imgray, imgray, mask=mask)
    FindFinishLine(imgray)   #detection of finish line
    contours, _ = cv2.findContours(res, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
        key=lambda b:(b[1][0]+b[1][3]), reverse=False))
    if len(boundingBoxes)>1:
        if (np.abs(boundingBoxes[-2][3]-boundingBoxes[-1][3])) > 30 and (np.abs(boundingBoxes[-2][3]-boundingBoxes[-1][3])) < 35:
            turnMultiplier = -0.7
        else:
            turnMultiplier += (0.2 if turnMultiplier<1 else 0)
    else:
        turnMultiplier = 1
    return contours[-2:]
#FindContours returns the filtered 2 contours of left and right edges



#PID Control Variables
kp,ki,kd=21.89,-7.130,28
error=0
P,i,d=0,0,0
v=0
prevError, prevI=0,0
nerf=0
startTime = t.time()

# main loop from here

while not FINISHED:
    img = env.get_image(cam_height=0, dims=[512,512])[:128,]
    contours = FindContours(img)
    left_dist, right_dist = AvgDistances(contours)
    
    error = left_dist - right_dist
    P=error
    i=i+prevI
    d=error - prevError
    v= (kp*P + ki*i + kd*d)/8000
    prevI = i
    prevError = error
    
    if v<2 and v>-2:
        nerf=0
    elif nerf<40:
        nerf+=1
    bs=15.9-nerf/2
    if v>7: v=7
    if v<-7: v=-7
    v=v*turnMultiplier
    env.move(vels = [bs-v,bs+v,bs-v,bs+v])
    if FINISHED and np.abs(error)>1000: FINISHED=False
    p.stepSimulation()
    if np.abs(error)<500:     #giving a boost if error is less than 500
        for i in range(np.int32(  (500-np.abs(error))/260 )):
            env.move(vels=[20,20,20,20])
            p.stepSimulation()
    


# Breaks out of loop on detection of the finish line
# Gives a boost on detection of finish line

for i in range(20):
    env.move([20,20,20,20])
    t.sleep(0.01)
    p.stepSimulation()

timeTaken = (t.time() - startTime)
env.move([0,0,0,0])
def showText():
    global timeTaken
    img = np.zeros((300,600,3), np.uint8)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(img,'FullMetal Genesis', 
        (35,118), 
        font, 
        2,
        fontColor,
        2,
        lineType)
    cv2.putText(img,"{:.2f} s".format(timeTaken), 
        (110,250), 
        font, 
        3,
        fontColor,
        3,
        lineType)
    cv2.imshow("Lap Complete", img)
# Shows the team name and time taken 
showText()

cv2.waitKey(15000)

'''
FINISHED
'''

cv2.destroyAllWindows()
env.close()
p.disconnect()
