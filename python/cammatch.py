#!/usr/bin/env python

import glob
import numpy as np
import cv2
import cv2.cv as cv
import pygame
import sys

import pygdisp
import scenefinder
import signdetector

import time
def average_since(timevalues, since):
    values = [tv.value for tv in filter( lambda tv: tv.time >= since , timevalues)]
    if(len(values)== 0):
        return -1
    else:
        return np.average( np.ma.masked_array(values, np.isnan(values) ) )

def show_detection(ascene, print_data, detected, rect):
    fonty = 0
    ystep = height / len( scnfinder.query_image.keys())

    top = 99999
    for k in print_data.keys():
        if top > print_data[k]:
            top = print_data[k]

    for k in print_data.keys():
        fonty += ystep
        if( k == detected ):
            cv2.putText(ascene, str(print_data[k]) + " " + k, (0, fonty), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #rect = scnfinder.findRect(k, scn)
            cv2.polylines(ascene, [np.int32(rect)],True,(0, 255, 0), 3)
        elif( print_data[k] < 0 ):
            cv2.putText(ascene, k, (0, fonty), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        else:
            cv2.putText(ascene, str(print_data[k]) + " " + k, (0, fonty), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,0), 2)
            sz = cv2.getTextSize(k, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)

#import smbus
#import time

def control(reg, value):
#        i2c = smbus.SMBus(1)
        adr = 0x4
#        i2c.write_byte_data(adr, 1, 0x8E)

def on_detect(detected, rect, ascene):
    print detected, rect

    if detected == '10':
        print '-- 10 --'
    elif detected == '30':
        print '-- 30 --'
    elif detected == '60':
        print detected
    elif detected == '90':
        print detected
    elif detected == 'craction':
        print detected
    elif detected == 'dear':
        print detected
    elif detected == 'dontenter':
        print detected
    elif detected == 'exclaimation':
        print detected
    elif detected == 'forward':
        print detected
    elif detected == 'lane':
        print detected
    elif detected == 'left':
        print detected
    elif detected == 'parking':
        print detected
    elif detected == 'right':
        print detected
    elif detected == 'rotate':
        print detected
    elif detected == 'roundabout':
        print detected
    elif detected == 'slow':
        print detected
    elif detected == 'straight':
        print detected
    elif detected == 'tomare':
        print detected
    elif detected == 'tuukoudome':
        print detected

global detect_history

if __name__ == '__main__':


    debug = None

    brightness = None
    contrast = None
    saturation = None
    gain = None

    for arg in sys.argv[1:]:
        if arg == 'debug':
            debug = True
        elif arg.startswith('brightness='):
            brightness = float( arg.replace('brightness=','') )
        elif arg.startswith('contrast='):
            contrast = float( arg.replace('contrast=','') )
        elif arg.startswith('saturation='):
            saturation = float( arg.replace('saturation=','') )
        elif arg.startswith('gain='):
            gain = float( arg.replace('gain=','') )

    def nothing(*arg):
        pass

    width = 320
    height = 240
    fps = 2

    pygame.init()
    disp = pygdisp.PygDisp(pygame, 640, 480)

    capture = cv.CaptureFromCAM(0)

    if debug == None:
        _img = cv.QueryFrame(capture)

        if _img == None:
            print 'capture failed'
            sys.exit(-1)

    print '     width = ', cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH)
    print '    heihgt = ', cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT)
    print '       fps = ', cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FPS)
    print 'brightness = ', cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_BRIGHTNESS)
    print '  contrast = ', cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_CONTRAST)
    print 'saturation = ', cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_SATURATION)
    print '       hue = ', cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_HUE)
    print '      gain = ', cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_GAIN)
    cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_FRAME_WIDTH,width)
    cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_FRAME_HEIGHT,height)
    #cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_FPS, fps)

    if brightness != None:
        print 'set brightness = ', brightness
        cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_BRIGHTNESS, brightness)
        print 'brightness = ', cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_BRIGHTNESS)
    if contrast != None:
        print 'set contrast = ', contrast 
        cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_CONTRAST, contrast)
        print '  contrast = ', cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_CONTRAST)
    if saturation != None:
        print 'set saturation = ', saturation 
        cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_SATURATION, saturation)
        print 'saturation = ', cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_SATURATION)
    if gain != None:
        print 'set gain = ', gain
        cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_GAIN, gain) 
        print '      gain = ', cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_GAIN)

    def configure_brightness(brightness):
        cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_BRIGHTNESS, brightness)
        print 'brightness = ', cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_BRIGHTNESS)

    def configure_contrast(contrast):
        cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_CONTRAST, contrast)
        print '  contrast = ', cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_CONTRAST)

    def configure_saturation(saturation):
        cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_SATURATION, saturation)
        print ' saturation = ', cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_SATURATION)

    def configure_hue(hue):
        cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_HUE, hue)
        print '       hue = ', cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_HUE)

    def configure_gain(gain):
            cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_GAIN, gain) 
            print '      gain = ', cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_GAIN)

    prop_brightness = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_BRIGHTNESS)
    prop_contrast = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_CONTRAST)
    prop_saturation = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_SATURATION)
    prop_hue = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_HUE)
    prop_gain = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_GAIN)
    
    print 'brightness = ', prop_brightness
    print '  contrast = ', prop_contrast
    print 'saturation = ', prop_saturation
    print '      gain = ', prop_gain

    if debug != None:
        cv2.namedWindow('config')
        cv2.createTrackbar("brightness", "config", int(prop_brightness), 255, configure_brightness)
        cv2.createTrackbar("contrast", "config", int(prop_contrast), 255, configure_contrast)
        cv2.createTrackbar("saturation", "config", int(prop_saturation), 255, configure_saturation)
        cv2.createTrackbar("gain", "config", int(prop_gain), 255, configure_gain)

    scnfinder = scenefinder.SceneFinder()

    signdet = signdetector.SignDetector()

    pnglist = glob.glob('./*.png')
    for f in pnglist:
        print "image: " + f
        scnfinder.setQueryImageFile(f)

    prev_time = 0

    while True:
        section0 = time.time()

        _img = cv.QueryFrame(capture)
        img = np.asarray(_img[:,:])
        h, w = img.shape[:2]

        section1 = time.time()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

        debug_images = None

        if debug != None:
            annotated_img = img.copy()
            annotated_img /= 2
            contour_img = np.zeros((h, w, 3), np.uint8)

            maskedcontour_img = gray.copy()

            debug_images = {}
            debug_images['contour'] = contour_img
            debug_images['annotated']  = annotated_img
            debug_images['maskedcontour']  = maskedcontour_img


        kpmask, kprects = signdet.create_keypoint_mask(gray, debug_images)

        if debug != None:
            cv2.imshow('contour', contour_img)
            cv2.imshow('annotated', annotated_img)
            cv2.imshow('maskedcontour', maskedcontour_img)

        hintcolor = None
        if len(kprects) > 0: 
            hintrect = img[kprects[0][0]:kprects[0][2], kprects[0][1]:kprects[0][3]]
            if hintrect.shape[0] > 0 and hintrect.shape[1] >0:
                hintcolor = cv2.mean( hintrect )
                if debug != None:
                    cv2.imshow('hint', hintrect)

        
        section2 = time.time()

        result = None
        scn = None
        if hintcolor != None:
            result, scn = scnfinder.findInScene(gray, kpmask, hintcolor)

        section3 = time.time()

        fonty = 0
        ystep = height / len( scnfinder.query_image.keys())

        print_data = {}

        top = 999999
        detected = None
        rect = None

        top_name = "none"

        if result != None:
            for k in scnfinder.query_image.keys():
                if result.has_key(k) and len(result[k].matches) != 0:
                    dist_ave = result[k].dist_ave #average_since(result[k].stats, time.time() - 2) 
                    print_data[k] = dist_ave
                    if top > dist_ave:
                        top = dist_ave
                        top_name = k
                    #if dist_ave < 65:
                    #if True:
                    #    detected = k
                    #    rect = scnfinder.findRect(k, scn)
                else:
                    print_data[k] = -1

            print top_name

            #detected = TimeValue(time.time(), (top_name, kprects[0]) )

	    #detect_history.append(detected)

            #if detected != None:
            #    on_detect(detected, rect, ascene)

            # display
            show_detection(img, print_data, detected, rect)

        #alphamask = np.zeros((h, w, 3), np.uint8)
        #alphamask[kpmask != 0] = (255, 255, 255)
        #masked = cv2.bitwise_and(img, alphamask)
        #disp.blit(masked)

        for r in kprects:
            cv2.rectangle(img, (r[0],r[1]), (r[2],r[3]), (0,255,0), 5)

        disp.blit(img)

        now = time.time()
        #print (now-section0), (now-section1), (now-section2), (now-section3)
        prev_time = now

 
        if debug != None:
            ch = cv2.waitKey(5)
            if ch == 27:
                break


    cv2.destroyAllWindows()
