import numpy as np
import cv2
import mediapipe as mp 

# here we use the mediapipe to detect the pose of the person present in the camera
mPose=mp.solutions.pose
# this drwaing utils is for drawing the lines
mDraw=mp.solutions.drawing_utils
pose=mPose.Pose()

cap=cv2.VideoCapture(0)

# inorder to increase the thickness and the radius of the connections and landmarks
draw_spec1=mDraw.DrawingSpec(thickness=2,circle_radius=3,color=(0,0,255))
draw_spec2=mDraw.DrawingSpec(thickness=2,circle_radius=8,color=(0,255,0))

while True:
    success,video=cap.read()
    cv2.resize(video,(700,800))
    video=cv2.flip(video,1)
# here we process the pose
    results=pose.process(video)
    # for showing all the land_marks on the video landmarks is actually the dots in the video 
    # so it only shows the dots not conneted these dots
    # so in next line we add the one mofre parameter that provide the connections
    # add also the draw spec 1
    mDraw.draw_landmarks(video,results.pose_landmarks,mPose.POSE_CONNECTIONS,draw_spec1,draw_spec2)

    # here we make a blanl screen where our detection will be display
    # so this blank screen have same height width and channel that a videos have
    h,w,chanel=video.shape
    blank_screen=np.zeros([h,w,chanel])
    # so here we fill the blank screen 
    blank_screen.fill(255)
    
    # siinorder to draw the connections and landmarks we copied the above statement
    mDraw.draw_landmarks(blank_screen,results.pose_landmarks,mPose.POSE_CONNECTIONS,draw_spec1,draw_spec2)


    cv2.imshow("PoseDetection",video)
    cv2.imshow("BlankScreen",blank_screen)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        
        break