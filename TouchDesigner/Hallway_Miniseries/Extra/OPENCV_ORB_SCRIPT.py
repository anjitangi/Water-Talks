# me - this DAT
# scriptOp - the OP which is cooking

import numpy as np
import cv2 as cv

# press 'Setup Parameters' in the OP to call this function to re-create the parameters.
def onSetupParameters(scriptOp):

	return

# called whenever custom pulse parameter is pushed
def onPulse(par):
	return


def onCook(scriptOp):
	img = op('null2').numpyArray(delayed=True)

	if img.dtype == np.float32 or img.dtype == np.float64:
		img = np.clip(img * 255, 0, 255).astype(np.uint8)
		
	rgb_img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)

	# Initiate ORB detector
	orb = cv.ORB_create()
	# find the keypoints with ORB
	kp = orb.detect(rgb_img,None)
	# compute the descriptors with ORB
	kp, des = orb.compute(rgb_img, kp)
	# draw only keypoints location,not size and orientation
	img2 = cv.drawKeypoints(rgb_img, kp, None, color=(0,255,0), flags=0)

	
	scriptOp.copyNumpyArray(img2)
	
	return
