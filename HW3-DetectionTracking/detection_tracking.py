import os
import sys
import cv2
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import numpy as np
from math import cos, sin, sqrt


face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

def help_message():
   print("Usage: [Question_Number] [Input_Video] [Output_Directory]")
   print("[Question Number]")
   print("1 Camshift")
   print("2 Particle Filter")
   print("3 Kalman Filter")
   print("4 Optical Flow")
   print("[Input_Video]")
   print("Path to the input video")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "02-1.avi " + "./")

def CamShift(frame,roi_hist,track_window):
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)
    return ret

    # output.write("%d,%d,%d\n" % (i, pts[0][0], pts[0][1]))  # Write as frame_index,pt_x,pt_y

    return pts

def KalmanFilter(c,w,r,h,img_width,img_height):
    def calc_point(angle):
        return (np.around(img_width / 2 + img_width / 3 * cos(angle), 0).astype(int),
                np.around(img_height / 2 - img_width / 3 * sin(angle), 1).astype(int))

    # --- init
    kalman = cv2.KalmanFilter(4, 2, 0)
    # state = 0.1 * np.random.randn(2, 1)
    state = np.array([c + w / 2, r + h / 2, 0, 0], dtype='float64')  # initial position
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                        [0., 1., 0., .1],
                                        [0., 0., 1., 0.],
                                        [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)
    kalman.processNoiseCov = 1e-5 * np.eye(4)
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state
    print(state)
    # --- tracking
    state_angle = state[0]
    state_pt = calc_point(state_angle)

    prediction = kalman.predict()
    #return prediction
    # print(prediction)
    # ...
    # obtain measurement
    '''measurement = kalman.measurementNoiseCov * np.random.randn(1, 1)
    # print(measurement)
    # generate measurement
    measurement = np.dot(kalman.measurementMatrix, state) + measurement
    print(measurement)
    # if measurement_valid:  # e.g. face found
    # ...
    # kalman.correct(measurement)
    '''

    predict_angle = prediction[0, 0]
    predict_pt = calc_point(predict_angle)

    measurement = kalman.measurementNoiseCov * np.random.randn(1, 1)

    # generate measurement


    measurement = np.dot(kalman.measurementMatrix, state) + measurement

    measurement_angle = measurement[0, 0]
    measurement_pt = calc_point(measurement_angle)


    # plot points
    def draw_cross(center, color, d):
        cv2.line(img,
                 (center[0] - d, center[1] - d), (center[0] + d, center[1] + d),
                 color, 1, cv2.LINE_AA, 0)
        cv2.line(img,
                 (center[0] + d, center[1] - d), (center[0] - d, center[1] + d),
                 color, 1, cv2.LINE_AA, 0)


    img = np.zeros((img_height, img_width, 3), np.uint8)
    draw_cross(np.int32(state_pt), (255, 255, 255), 3)
    draw_cross(np.int32(measurement_pt), (0, 0, 255), 3)
    draw_cross(np.int32(predict_pt), (0, 255, 0), 3)

    cv2.line(img, state_pt, measurement_pt, (0, 0, 255), 3, cv2.LINE_AA, 0)
    cv2.line(img, state_pt, predict_pt, (0, 255, 255), 3, cv2.LINE_AA, 0)

    kalman.correct(measurement)

    process_noise = sqrt(kalman.processNoiseCov[0, 0]) * np.random.randn(2, 1)
    state = np.dot(kalman.transitionMatrix, state) + process_noise

    cv2.imshow("Kalman", img)


# use prediction or posterior as your tracking result

def particleevaluator(back_proj, particle):
    return back_proj[particle[1], particle[0]]

def ParticleFilter(c,w,r,h,frame,hist_bp,particles,n_particles):
    # --- init

    # a function that, given a particle position, will return the particle's "fitness"


    # hist_bp: obtain using cv2.calcBackProject and the HSV histogram

    # c,r,w,h: obtain using detect_one_face()

    # --- tracking
    stepsize = 8
    # Particle motion model: uniform step (TODO: find a better motion model)
    np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")

    # Clip out-of-bounds particles
    particles = particles.clip(np.zeros(2), np.array((frame.shape[0], frame.shape[1])) - 1).astype(int)

    f = particleevaluator(hist_bp, particles.T)  # Evaluate particles
    weights = np.float32(f.clip(1))  # Weight ~ histogram response
    weights /= np.sum(weights)  # Normalize w
    pos = np.sum(particles.T * weights, axis=1).astype(int)  # expected position: weighted average

    if 1. / np.sum(weights ** 2) < n_particles / 2.:  # If particle cloud degenerate:
        particles = particles[resample(weights), :]  # Resample particles according to weights

    print(particles)


def detect_one_face(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0,0,0,0)
    return faces[0]

def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c,r,w,h = window
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist


def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
      while u > C[j]:
          j+=1
      indices.append(j-1)
    return indices

def skeleton_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    print(frame.shape)
    # detect face in first frame
    c,r,w,h = detect_one_face(frame)
    print(c,r,w,h)
    # Write track point for first frame
    output.write("%d,%d,%d\n" % (r,w,h)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)


    # calculate the HSV histogram in the window
    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you

    hsvt = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    hist_bp = cv2.calcBackProject([hsvt], [0, 1], roi_hist, [0, 180, 0, 256], 1)
    n_particles = 200

    init_pos = np.array([c + w / 2.0, r + h / 2.0], int)  # Initial position
    particles = np.ones((n_particles, 2), int) * init_pos  # Init particles to init position
    f0 = particleevaluator(hist_bp, init_pos) * np.ones(n_particles)  # Evaluate appearance model
    weights = np.ones(n_particles) / n_particles  # weights are uniform (at first)

    # initialize the tracker
    # e.g. kf = cv2.KalmanFilter(4,2,0)
    # or: particles = np.ones((n_particles, 2), int) * initial_pos

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        # perform the tracking
        # ret = CamShift(frame,roi_hist,track_window)
        # KalmanFilter(c,w,r,h,frame.shape[0],frame.shape[1])
        ParticleFilter(c,w,r,h,frame,hist_bp,particles,n_particles)
        # e.g. cv2.meanShift, cv2.CamShift, or kalman.predict(), kalman.correct()
        # print(ret)
        # Draw it on image



        # pts = cv2.boxPoints(ret)
        # print(pts)
        # pts = np.int0(pts)
        #
        # img2 = cv2.polylines(frame, [pts], True, 255, 2)
        # cv2.imshow('img2', img2)
        # k = cv2.waitKey(60) & 0xff
        # if k == 27:
        #     break
        # else:
        #     cv2.imwrite(chr(k) + ".jpg", img2)
        #


        # use the tracking result to get the tracking point (pt):
        # if you track a rect (e.g. face detector) take the mid point,
        # if you track particles - take the weighted average
        # the Kalman filter already has the tracking point in the state vector

        # write the result to the output file
        # output.write("%d,%d,%d\n" % (frameCounter,pts[0][0],pts[0][1])) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()

if __name__ == '__main__':
    question_number = -1
   
    # Validate the input arguments
    if (len(sys.argv) != 4):
        help_message()
        sys.exit()
    else: 
        question_number = int(sys.argv[1])
        if (question_number > 4 or question_number < 1):
            print("Input parameters out of bound ...")
            sys.exit()

    # read video file
    video = cv2.VideoCapture(sys.argv[2]);

    if (question_number == 1):
        skeleton_tracker(video, "output_camshift.txt")
    elif (question_number == 2):
        skeleton_tracker(video, "output_particle.txt")
    elif (question_number == 3):
        skeleton_tracker(video, "output_kalman.txt")
    elif (question_number == 4):
        skeleton_tracker(video, "output_of.txt")


'''
For Kalman Filter:

# --- init
        kalman = cv2.KalmanFilter(2, 1, 0)

state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                    [0., 1., 0., .1],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
kalman.measurementMatrix = 1. * np.eye(2, 4)
kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
kalman.errorCovPost = 1e-1 * np.eye(4, 4)
kalman.statePost = state


# --- tracking

prediction = kalman.predict()

# ...
# obtain measurement

if measurement_valid: # e.g. face found
    # ...
    posterior = kalman.correct(measurement)

# use prediction or posterior as your tracking result
'''

'''
For Particle Filter:

# --- init

# a function that, given a particle position, will return the particle's "fitness"
def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]

# hist_bp: obtain using cv2.calcBackProject and the HSV histogram
# c,r,w,h: obtain using detect_one_face()
n_particles = 200

init_pos = np.array([c + w/2.0,r + h/2.0], int) # Initial position
particles = np.ones((n_particles, 2), int) * init_pos # Init particles to init position
f0 = particleevaluator(hist_bp, pos) * np.ones(n_particles) # Evaluate appearance model
weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)


# --- tracking

# Particle motion model: uniform step (TODO: find a better motion model)
np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")

# Clip out-of-bounds particles
particles = particles.clip(np.zeros(2), np.array((im_w,im_h))-1).astype(int)

f = particleevaluator(hist_bp, particles.T) # Evaluate particles
weights = np.float32(f.clip(1))             # Weight ~ histogram response
weights /= np.sum(weights)                  # Normalize w
pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average

if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
    particles = particles[resample(weights),:]  # Resample particles according to weights
# resample() function is provided for you
'''
