# Import necessary packages for robotic arm
import time
from xarm import version
from xarm.wrapper import XArmAPI
from matplotlib import pyplot as plt
import cv2
import numpy as np

print('xArm-Python-SDK Version:{}'.format(version.__version__))

arm = XArmAPI('192.168.1.233')
arm.clean_warn()
arm.clean_error()
arm.motion_enable(True)
arm.set_mode(0)
arm.set_state(0)
time.sleep(1)

params = {'speed': 100,
          'acc': 2000,
          'angle_speed': 20,
          'angle_acc': 500,
          'events': {},
          'variables': {},
          'quit': False}

# Register error/warn changed callback
def error_warn_change_callback(data):
    if data and data['error_code'] != 0:
        arm.set_state(4)
        params['quit'] = True
        print('err={}, quit'.format(data['error_code']))
        arm.release_error_warn_changed_callback(error_warn_change_callback)


arm.register_error_warn_changed_callback(error_warn_change_callback)


# Register state changed callback
def state_changed_callback(data):
    if data and data['state'] == 4:
        if arm.version_number[0] >= 1 and arm.version_number[1] >= 1 and arm.version_number[2] > 0:
            params['quit'] = True
            print('state=4, quit')
            arm.release_state_changed_callback(state_changed_callback)


arm.register_state_changed_callback(state_changed_callback)

# Register counter value changed callback
if hasattr(arm, 'register_count_changed_callback'):
    def count_changed_callback(data):
        print('counter val: {}'.format(data['count']))


    arm.register_count_changed_callback(count_changed_callback)

# Capture Image with Connected Camera
imageRealTimeCapturing = cv2.VideoCapture(0)
imageRealTimeCapturing.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
imageRealTimeCapturing.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# Define the count for the Saved Images
imageCounter = 1
imageProcessCounter = 1

params['speed'] = 400
params['acc'] = 200
# Arm Movement Code , Initial Position
if arm.error_code == 0 and not params['quit']:
    if arm.set_servo_angle(angle=[0.0, 0.0, 0.0, 0.0, -90.0, 0.0],
                           speed=params['angle_speed'],
                           mvacc=params['angle_acc'], wait=True) != 0:
        params['quit'] = True

# First Position
if arm.error_code == 0 and not params['quit']:
    if arm.set_servo_angle(angle=[135, 0, -110, 0, 105, 90],
                           speed=params['angle_speed'],
                           mvacc=params['angle_acc'], wait=True) != 0:
        params['quit'] = True

# RealTimeCapturing
_, frame = imageRealTimeCapturing.read(0)
# Conversion from BGR to HSV values
hsvFrameConvert_1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# lower_red_1 = np.array([0, 100, 20])
# upper_red_1 = np.array([10, 255, 255])
# lower_red_2 = np.array([160, 100, 20])
# upper_red_2 = np.array([179, 255, 255])
lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])
# Setup of Mask for the Range
# lower_mask = cv2.inRange(hsvFrameConvert_1, lower_red_1, upper_red_1)
# upper_mask = cv2.inRange(hsvFrameConvert_1, lower_red_2, upper_red_2)
# mask = cv2.inRange(hsvFrameConvert_1, lower_green, upper_green)
mask = cv2.inRange(hsvFrameConvert_1, lower_green, upper_green)
# mask = lower_mask + upper_mask
# result = cv2.bitwise_and(frame, frame, mask=mask)
# Show Images
cv2.imshow('Original Image', frame)
cv2.imshow('Processed Image', mask)
key = cv2.waitKey(1)
# Saving Images
imageName = "PictureTaken{}.jpg".format(imageCounter)
cv2.imwrite(imageName, frame)
print("\nPicture_{} is Saved".format(imageCounter))
imageCounter += 1
imageProcessedName = "ProcessImage{}.jpg".format(imageProcessCounter)
cv2.imwrite(imageProcessedName, mask)
print("Processed_Picture_{} is Saved".format(imageProcessCounter))
imageProcessCounter += 1

# 2nd Position
if arm.error_code == 0 and not params['quit']:
    if arm.set_servo_angle(angle=[45, 0, -110, 0, 105, 25],
                           speed=params['angle_speed'],
                           mvacc=params['angle_acc'], wait=True) != 0:
        params['quit'] = True
# RealTimeCapturing
_, frame = imageRealTimeCapturing.read(0)
# Conversion from BGR to HSV values
hsvFrameConvert_1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# lower_red_1 = np.array([0, 100, 20])
# upper_red_1 = np.array([10, 255, 255])
# lower_red_2 = np.array([160, 100, 20])
# upper_red_2 = np.array([179, 255, 255])
lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])
# Setup of Mask for the Range
# lower_mask = cv2.inRange(hsvFrameConvert_1, lower_red_1, upper_red_1)
# upper_mask = cv2.inRange(hsvFrameConvert_1, lower_red_2, upper_red_2)
# mask = cv2.inRange(hsvFrameConvert_1, lower_green, upper_green)
# mask = lower_mask + upper_mask
mask = cv2.inRange(hsvFrameConvert_1, lower_green, upper_green)
result = cv2.bitwise_and(frame, frame, mask=mask)
# Show Images
cv2.imshow('Original Image', frame)
cv2.imshow('Processed Image', mask)
key = cv2.waitKey(1)
# Saving Images
imageName = "PictureTaken{}.jpg".format(imageCounter)
cv2.imwrite(imageName, frame)
print("\nPicture_{} is Saved".format(imageCounter))
imageCounter += 1
imageProcessedName = "ProcessImage{}.jpg".format(imageProcessCounter)
cv2.imwrite(imageProcessedName, mask)
print("Processed_Picture_{} is Saved".format(imageProcessCounter))
imageProcessCounter += 1

# 3rd Position
if arm.error_code == 0 and not params['quit']:
    if arm.set_servo_angle(angle=[-45, 0, -110, 0, 105, 25],
                           speed=params['angle_speed'],
                           mvacc=params['angle_acc'], wait=True) != 0:
        params['quit'] = True
# RealTimeCapturing
_, frame = imageRealTimeCapturing.read(0)
# Conversion from BGR to HSV values
hsvFrameConvert_1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# lower_red_1 = np.array([0, 100, 20])
# upper_red_1 = np.array([10, 255, 255])
# lower_red_2 = np.array([160, 100, 20])
# upper_red_2 = np.array([179, 255, 255])
lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])
# Setup of Mask for the Range
# lower_mask = cv2.inRange(hsvFrameConvert_1, lower_red_1, upper_red_1)
# upper_mask = cv2.inRange(hsvFrameConvert_1, lower_red_2, upper_red_2)
# mask = cv2.inRange(hsvFrameConvert_1, lower_green, upper_green)
# mask = lower_mask + upper_mask
mask = cv2.inRange(hsvFrameConvert_1, lower_green, upper_green)
result = cv2.bitwise_and(frame, frame, mask=mask)
# Show Images
cv2.imshow('Original Image', frame)
cv2.imshow('Processed Image', mask)
key = cv2.waitKey(1)
# Saving Images
imageName = "PictureTaken{}.jpg".format(imageCounter)
cv2.imwrite(imageName, frame)
print("\nPicture_{} is Saved".format(imageCounter))
imageCounter += 1
imageProcessedName = "ProcessImage{}.jpg".format(imageProcessCounter)
cv2.imwrite(imageProcessedName, mask)
print("Processed_Picture_{} is Saved".format(imageProcessCounter))
imageProcessCounter += 1

# 4th Position
if arm.error_code == 0 and not params['quit']:
    if arm.set_servo_angle(angle=[-135.0, 0.0, -110.0, 0.0, 105.0, 25.0],
                           speed=params['angle_speed'],
                           mvacc=params['angle_acc'], wait=True) != 0:
        params['quit'] = True
# RealTimeCapturing
_, frame = imageRealTimeCapturing.read(0)
# Conversion from BGR to HSV values
hsvFrameConvert_1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# lower_red_1 = np.array([0, 100, 20])
# upper_red_1 = np.array([10, 255, 255])
# lower_red_2 = np.array([160, 100, 20])
# upper_red_2 = np.array([179, 255, 255])
lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])
# Setup of Mask for the Range
# lower_mask = cv2.inRange(hsvFrameConvert_1, lower_red_1, upper_red_1)
# upper_mask = cv2.inRange(hsvFrameConvert_1, lower_red_2, upper_red_2)
# mask = cv2.inRange(hsvFrameConvert_1, lower_green, upper_green)
# mask = lower_mask + upper_mask
mask = cv2.inRange(hsvFrameConvert_1, lower_green, upper_green)
# Show Images
cv2.imshow('Original Image', frame)
cv2.imshow('Processed Image', mask)
key = cv2.waitKey(1)
# Saving Images
imageName = "PictureTaken{}.jpg".format(imageCounter)
cv2.imwrite(imageName, frame)
print("\nPicture_{} is Saved".format(imageCounter))
imageCounter += 1
imageProcessedName = "ProcessImage{}.jpg".format(imageProcessCounter)
cv2.imwrite(imageProcessedName, mask)
print("Processed_Picture_{} is Saved".format(imageProcessCounter))
imageProcessCounter += 1

# Initial Position
if arm.error_code == 0 and not params['quit']:
    if arm.set_servo_angle(angle=[0.0, 0.0, 0.0, 0.0, -90.0, 0.0],
                           speed=params['angle_speed'],
                           mvacc=params['angle_acc'], wait=True) != 0:
        params['quit'] = True

# Read Image
img1 = cv2.imread('ProcessImage1.jpg')
img2 = cv2.imread('ProcessImage2.jpg')
img3 = cv2.imread('ProcessImage3.jpg')
img4 = cv2.imread('ProcessImage4.jpg')
# Standby register
selected_segment = 0
white_count1 = 0
white_count2 = 0
white_count3 = 0
white_count4 = 0
Target = 0

# Comparison
for y in range(img1.shape[0]):
    for x in range(img1.shape[1]):
        if img1[y,x][0]==255 and img1[y,x][1]==255 and img1[y,x][2]==255:
            white_count1 = white_count1 + 1
for y in range(img2.shape[0]):
    for x in range(img2.shape[1]):
        if img2[y,x][0]==255 and img2[y,x][1]==255 and img2[y,x][2]==255:
            white_count2 = white_count2 + 1
for y in range(img3.shape[0]):
    for x in range(img3.shape[1]):
        if img3[y,x][0]==255 and img3[y,x][1]==255 and img3[y,x][2]==255:
            white_count3 = white_count3 + 1
for y in range(img4.shape[0]):
    for x in range(img4.shape[1]):
        if img4[y,x][0]==255 and img4[y,x][1]==255 and img4[y,x][2]==255:
            white_count4 = white_count4 + 1

if white_count1 > selected_segment:
    selected_segment = white_count1
    finalimg = cv2.imread('ProcessImage1.jpg')
    finalnormalimg = cv2.imread('PictureTaken1.jpg')
    Target = 1
if white_count2 > selected_segment:
    selected_segment = white_count2
    finalimg = cv2.imread('ProcessImage2.jpg')
    finalnormalimg = cv2.imread('PictureTaken2.jpg')
    Target = 2
if white_count3 > selected_segment:
    selected_segment = white_count3
    finalimg = cv2.imread('ProcessImage3.jpg')
    finalnormalimg = cv2.imread('PictureTaken3.jpg')
    Target = 3
if white_count4 > selected_segment:
    selected_segment = white_count4
    finalimg = cv2.imread('ProcessImage4.jpg')
    finalnormalimg = cv2.imread('PictureTaken4.jpg')
    Target = 4

# Load Target Image
load_Target = finalimg
load_Target2 = finalnormalimg

Save_ImageForGray1 = "FinalPicture.jpg".format()
cv2.imwrite(Save_ImageForGray1, load_Target)

# Saving Images
Load_ImageForGray1 = cv2.imread("FinalPicture.jpg")

# Convert to HSV
hsvFrameConvert_1 = cv2.cvtColor(Load_ImageForGray1, cv2.COLOR_BGR2HSV)
# lower_red_1 = np.array([0, 100, 20])
# upper_red_1 = np.array([10, 255, 255])
# lower_red_2 = np.array([160, 100, 20])
# upper_red_2 = np.array([179, 255, 255])
lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])
# Setup of Mask for the Range
# lower_mask = cv2.inRange(hsvFrameConvert_1, lower_red_1, upper_red_1)
# upper_mask = cv2.inRange(hsvFrameConvert_1, lower_red_2, upper_red_2)
# mask = cv2.inRange(hsvFrameConvert_1, lower_green, upper_green)
# mask = lower_mask + upper_mask
mask = cv2.inRange(hsvFrameConvert_1, lower_green, upper_green)

# Contours Drawing
contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
ret, thresh = cv2.threshold(mask, 75, 255, 0, cv2.THRESH_BINARY)

# Save Target Image for conversion
finalImgName = "FinalImage_1.jpg".format()
cv2.imwrite(finalImgName, load_Target2)
ToLoadGray = cv2.imread("FinalImage_1.jpg")
# Convert Loaded Image intro GrayScale
grayLoad = cv2.cvtColor(ToLoadGray, cv2.COLOR_BGR2GRAY)
# Convert to Binary by Thresholding
ret, binary_map = cv2.threshold(grayLoad, 127, 255, 0)
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
# Get CC_STAT_AREA component as Stats(label, COLUMN)
areas = stats[1:,cv2.CC_STAT_AREA]
result = np.zeros((labels.shape), np.uint8)

for i in range(0, nlabels - 1):
    if (areas[i] >= 3000):   #keep
        result[labels == i + 1] = 255

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(mask, contours, -1, (0,255,255), 2)

for i, cnt in enumerate(contours):
    if hierarchy[0][i][3] != -1:        # basically look for holes
        # if the size of the contour is less than a threshold (noise)
        if cv2.contourArea(cnt) < 70:
            # Fill the holes in the original image
            cv2.drawContours(load_Target, [cnt], 0, (255), -1)

# highlight required size masked shape
for c in contours:
    area = cv2.contourArea(c)
    # Remove noise , small sections
    if area > 3000:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(load_Target, (x,y), (x+w, y+h), (0, 0, 255), 2)
        cv2.drawContours(load_Target, c, -1, (255, 0, 0), 2)
    print(area)

# Center of BasePic1
center_Base1X = 620
center_Base1Y = 420
# Center of BasePic2
center_Base2X = 620
center_Base2Y = 420
# Center of BasePic3
center_Base3X = 620
center_Base3Y = 420
# Center of BasePic4
center_Base4X = 600
center_Base4Y = 510

pic_frameX = 0
pic_frameY = 0
pic_frameXMin = load_Target.shape[1]
pic_frameYMin = load_Target.shape[0]

for y in range(load_Target.shape[0]):
    for x in range(load_Target.shape[1]):
        if (load_Target[y,x]).all():
            if x >  pic_frameX:
                pic_frameX = x
                if y > pic_frameY :
                    pic_frameY = y
                    if x < pic_frameXMin :
                        pic_frameXMin = x
                        if y < pic_frameYMin :
                            pic_frameYMin = y

highestx = 0
highesty = 0
lowestx = load_Target.shape[1]
lowesty = load_Target.shape[0]

for y in range(load_Target.shape[0]):
    for x in range(load_Target.shape[1]):
        if load_Target[y,x][0]==255 and load_Target[y,x][1]==255 and load_Target[y,x][2]==255:
            if x >  highestx:
                highestx = x
            if y > highesty :
                highesty = y
            if x < lowestx :
                lowestx = x
            if y < lowesty :
                lowesty = y

# Picture Frame Print
print('Frame Length: ', pic_frameX)
print('Frame Height: ', pic_frameYMin)
# First Point Taken , Upper Left
print('x1: ', lowestx)
print('x2: ', highestx)
# Second Point Taken . Lower Right
print('y1: ', lowesty)
print('y2: ', highesty)
# Length of the Cube in the Picture
length_X = highestx - lowestx
length_Y = highesty - lowesty
# Center Point
center_x = lowestx + (highestx-lowestx)/2
center_y = lowesty + (highesty-lowesty)/2
# Center of Picture
centre_pointX = load_Target.shape[1] / 2
centre_pointY = load_Target.shape[0] / 2

cv2.imshow('RedBlockDetection', load_Target)

print('\ndistance between first 2 point in x: ', length_X)
print('distance between first 2 point in y: ', length_Y)
print('Center of X coordinate: ', center_x)
print('Center of Y coordinate: ', center_y)

# Sector Coordination Positive Negative
if Target == 1:
    center_BaseFinalX = center_Base1X
    center_BaseFinalY = center_Base1Y
    center_FinalX = -(center_Base1X-center_x)
    center_FinalY = center_Base1Y-center_y
if Target == 2:
    center_BaseFinalX = center_Base2X
    center_BaseFinalY = center_Base2Y
    center_FinalX = center_Base2X-center_x
    center_FinalY = center_Base2Y-center_y
if Target == 3:
    center_BaseFinalX = center_Base3X
    center_BaseFinalY = center_Base3Y
    center_FinalX = center_Base3X-center_x
    center_FinalY = -(center_Base3Y-center_y)
if Target == 4:
    center_BaseFinalX = center_Base4X
    center_BaseFinalY = center_Base4Y
    center_FinalX = -(center_Base4X-center_x)
    center_FinalY = -(center_Base4Y-center_y)

print('\nCenter of X coordinate From Base of Robot: ', center_FinalX)
print('Center of Y coordinate From Base of Robot: ', center_FinalY)

# 1st Circle , Upper Left
plt.plot(lowestx, lowesty, marker='o', color="white")
# 2nd Circle . Lower Right
plt.plot(highestx, highesty, marker='o', color="white")
# 3rd Circle , Upper Right
plt.plot(highestx, lowesty, marker='o', color="white")
# 4th Circle , Lower Left
plt.plot(lowestx, highesty, marker='o', color="white")

# Using Ratio to calculate the accurate Coordinate Based on the End-Factor from own derivation
# ValueX_1stSectorFrom_EndFactor = -235 / -216
# ValueY_1stSectorFrom_EndFactor = 345 / 305
RatioXValue_XCord_to_Pixel = 1.088  # 1.088
RatioYValue_YCord_to_Pixel = 1.131  # 1.131
print('\nRatio for conversion for X-Cord: ', RatioXValue_XCord_to_Pixel)
print('Ratio for conversion for Y-Cord: ', RatioYValue_YCord_to_Pixel)
final_ConversionCenterX = center_FinalX * RatioXValue_XCord_to_Pixel
final_ConversionCenterY = center_FinalY * RatioYValue_YCord_to_Pixel
final_ConversionCenterX_1 = '{0:.3f}'.format(final_ConversionCenterX)
final_ConversionCenterY_1 = '{0:.3f}'.format(final_ConversionCenterY)
print('\nX-Cord after conversion: ', final_ConversionCenterX)
print('Y-Cord after conversion: ', final_ConversionCenterY)

# Center of the Square
plt.plot(center_x, center_y, marker='+', color="black")
# Center of the Picture
plt.plot(centre_pointX, centre_pointY, marker='+', color="white")
plt.plot(center_BaseFinalX, center_BaseFinalY, marker='+', color="white")
plt.text(0, 0, (final_ConversionCenterX_1,final_ConversionCenterY_1,'0.000'), fontsize=20, bbox=dict(facecolor='red', alpha=0.5))

# Compare which length calculated to be the length of the side of the cube
if length_X > length_Y:
    side = length_X
else:
    side = length_Y

print('\nThe number of pixel used for the length of the cube in the picture is: ',side)
print('The actual length of the cube in cm: 5cm')
# currently 1mm represent how many pixel
represent_side = side / 50
RatioOfConversionOfImage = 1 / represent_side
focal_length = 65                      # focal length assumed to be 30mm
# focal_lengthRatio = focal_length /10
print('As ', represent_side, ' pixel is equal to 1mm')
# Formulae of conversion : distance = size_obj * focal length / size_img
distance_webcam = (((50)*(focal_length) / (side*RatioOfConversionOfImage)) - 6)
distance_webcam1 = '{0:.3f}'.format(distance_webcam)
print('\nDistance from webcam is estimated to be', distance_webcam1, 'cm ')
# Plot the graph
plt.imshow(load_Target)
plt.show()

# Upon Breaking the loop
imageRealTimeCapturing.release()
cv2.destoryAllWindows()