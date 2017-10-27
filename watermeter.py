__author__ = 'bjorn@haxx.se'
#import picamera
import sys
import math
import cv2
import numpy as np
import argparse

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def create_hue_mask(image, lower_color, upper_color):
    lower = np.array(lower_color, np.uint8)
    upper = np.array(upper_color, np.uint8)
 
    # Create a mask from the colors
    mask = cv2.inRange(image, lower, upper)
    return mask

def findRed(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red_hue = create_hue_mask(hsv, [0, 100, 100], [10, 255, 255])
    higher_red_hue = create_hue_mask(hsv, [170, 100, 100], [179, 255, 255])

    mask = cv2.bitwise_or(lower_red_hue, higher_red_hue)
    red = cv2.GaussianBlur(mask,(5,5),0)
    return red

def findCircles(img):
    ### find dials by circles
    gradient = 0
    if hasattr(cv2, 'HOUGH_GRADIENT'):
        # OpenCV v3
        gradient = cv2.HOUGH_GRADIENT
    elif hasattr(cv2.cv, 'CV_HOUGH_GRADIENT'):
        # OpenCV v2
        gradient = cv2.cv.CV_HOUGH_GRADIENT
    else:
        print "Unsupported OpenCV version %s" % cv2.__version__
        exit(1)

    circles = cv2.HoughCircles(img, gradient, 1, 50,
                               param1=20,param2=15,minRadius=20,maxRadius=50)

    if circles is None:
        print "No circles found!"
        if args.show_error:
            cv2.imshow('error', image)
            cv2.waitKey()
            cv2.destroyAllWindows()
        exit(1)

    center = None
    dials = []
    if circles is not None:
        numcirc = len(circles[0,:])

        index = 0
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            #cv2.circle(image,(i[0],i[1]),2,(0,0,255),3)
            dials.append( [i[0], i[1], i[2]] )
            index += 1

    if numcirc != 4:
        print "%d circles found but expected 4!" % numcirc
        if args.show_error:
            cv2.imshow('error', image)
            cv2.waitKey()
            cv2.destroyAllWindows()
        exit(1)

    def getX(item):
        return item[0]
    fromleft = sorted(dials, key=getX)
    return fromleft

# cut out one dial at a time and find needle angle
def findAngle(img, redimg, center, width):
### lines
    edges = cv2.Canny(redimg,100,200,apertureSize = 3)
    lines = cv2.HoughLinesP(image=edges,
                            rho=1,
                            theta=np.pi/90,
                            threshold=15,
                            minLineLength=width/4,
                            maxLineGap=50)

    tip = None
    maxlen = 0
    if lines is None:
        print "No lines found"
        if args.show_error:
            cv2.imshow('error',img)
            cv2.waitKey()
            cv2.destroyAllWindows()
        exit(2)
    else:
        pl = None
        #print "%d lines" % len(lines)
        for x in range(0, len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                l = line([x1,y1], [x2,y2]);
                cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
                #print "line: %d,%d %d,%d" % (x1, y1, x2, y2)

                # see if it intersects any other line
                for y in range(0, len(lines)):
                    for xx1,yy1,xx2,yy2 in lines[y]:
                        l2 = line([xx1,yy1], [xx2,yy2]);
                        if l2 is l:
                            continue
                        r = intersection(l, l2)
                        if r and r[0] > 0 and r[1] > 0:
                            dist = math.sqrt( (r[0] - center[0])**2 + (r[1] - center[1])**2 )
                            #print "intersection %d,%d at distance %d" % ( r[0], r[1], dist)
                            #cv2.circle(img,(r[0],r[1]),2,(0,0,255),2)
                            if dist > maxlen and dist < width/2:
                                tip = r
                                maxlen = dist

    if tip is None:
        print "No tip found!"
        if args.show_error:
            cv2.imshow('error',img)
            cv2.waitKey()
            cv2.destroyAllWindows()
        exit(2)

    #print "chosen intersection: %d,%d at distance %d" % ( tip[0], tip[1], maxlen)
    cv2.line(img, (tip[0], tip[1]), (center[0], center[1]), (255,0,255), 2)

    xlen = tip[0] - center[0]
    ylen = center[1] - tip[1]
    rad = math.atan2(ylen, xlen)
    deg = math.degrees(rad)
    #print "angle deg:", deg
    #print "angle rad:", rad

    if deg < 0:
        percent = (90 + abs(deg)) / 360
    elif deg < 90:
        percent = (90 - deg) / 360
    else:
        percent = (450 - deg) / 360

    #print "percent", math.trunc(percent * 100)
    string = "%d%%" % math.trunc(percent * 100)
    cv2.putText(img, string, (center[0] - width/5, center[1] - width/3),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                              (255,255,255), 2)

    return math.trunc(percent * 100)

##########################################################3

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--show',
                    action='store_true',
                    help='Show processed image')
parser.add_argument('-e', '--show-error',
                    action='store_true',
                    help='Show image in case of error')
parser.add_argument('-c', '--camera',
                    action='store_true',
                    help='Read image from attached camera')
parser.add_argument('file', nargs='?',
                    help='Read image from this file')
args = parser.parse_args()

if args.camera:
    # Take a picture using webcam
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()
    if ret:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    else:
        print "Failed to read webcam image!"
        exit(0)
elif args.file:
    image = cv2.imread(args.file)
else:
    parser.print_help()
    exit(0)

height, width, channels = image.shape
if width > 1500 or width < 600:
    scale = 1000.0 / width
    print "Image is %dx%d, resizing to %dx%d" % ( width, height, width * scale, height * scale)
    scaled = cv2.resize(image, (0,0), fx=scale, fy=scale)
    image = scaled

red = findRed(image)
fromleft = findCircles(red)

# rotate image so dial 1 and 3 are level
xlen = fromleft[2][0] - fromleft[0][0];
ylen = fromleft[2][1] - fromleft[0][1];
#pripnt "xlen %d ylen %d" % (xlen, ylen)
rad = math.atan2(ylen, xlen)
deg = math.degrees(rad)
#print "image is rotated %.2f degrees" % deg

image_center = tuple(np.array(image.shape)[:2]/2)
rot_mat = cv2.getRotationMatrix2D(image_center, deg, 1)
image = cv2.warpAffine(image, rot_mat, image.shape[:2], flags=cv2.INTER_LINEAR)
red = cv2.warpAffine(red, rot_mat, image.shape[:2], flags=cv2.INTER_LINEAR)

# find dials again in rotated image
fromleft = findCircles(red)
angles = []

for d in fromleft:
    ulx = int(d[0])
    uly = int(d[1])
    radius = int(d[2])*3
    cut = image[uly-radius : uly+radius,
                ulx-radius : ulx+radius];
    redcut = red[uly-radius : uly+radius,
                 ulx-radius : ulx+radius];
    val = findAngle(cut, redcut, (radius, radius), radius*2)
    angles.append(val)
    #break

liter = ((angles[3] / 10) * 100 +
         (angles[2] / 10) * 10 +
         (angles[1] / 10) +
         float(angles[0]) / 100)

string = "%.2f liter" % liter
cv2.putText(image, string,
            (image_center[0]/2, image_center[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 2,
            (255,255,255), 5)


if args.show:
    cv2.imshow('image',image)
    cv2.waitKey()
    cv2.destroyAllWindows()

print "%.2f liter" % liter
