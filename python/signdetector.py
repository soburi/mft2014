#!/usr/bin/env python

import cv2

import cv2.cv as cv
import numpy as np
import sys
import time

from collections import namedtuple

global TimeValue
TimeValue = namedtuple('TimeValue', 'time value')

def color_table(x):
    r=0
    g=0
    b=0

    for i in range(0,x):
        if i%3 == 0:
            r += 64
        elif i%3 == 1:
            g += 64
        else:
            b += 64

    return (b%255,g%255,r%255)

def drop_before(timevalues, before):
    return filter( lambda tv: tv.time >= before, timevalues)


class SignDetector:
    def __init__(self):
        self.lines_stats = []
        self.votes = 100


    def find_circle_sign(self, gray, annotated):
        circle_rects = [] 

        circles = cv2.HoughCircles(gray, cv.CV_HOUGH_GRADIENT,1, 30
            ,param1=100, param2=90, minRadius=20, maxRadius=100) 

        
        if circles != None:
            circles = np.round(circles[0, :]).astype("int") 
            for (x, y, r) in circles:
                circle_rects.append( (x-r, y-r, x+r, y+r) )
                if annotated != None:
                    cv2.circle(annotated, (x, y), r, (255, 255, 0), 4)
                    cv2.rectangle(annotated, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1) 

        return circle_rects

    def find_rect_sign(self, edge, annotated, contour_img, maskedcontour_img):
        found_rects = []
        lines = cv2.HoughLines(edge, 1, np.pi/180, 80) 
        mask = np.zeros(edge.shape, np.uint8)

        self.lines_stats.append( TimeValue(time.time(), lines) ) 
        self.lines_stats = drop_before(self.lines_stats, time.time() -1) 

        for t_lines in self.lines_stats:
            lines = t_lines.value

            if lines != None:
                for rho,theta in lines[0]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a)) 
                    
                    cv2.line(mask,(int(x1),int(y1)),(int(x2),int(y2)),(255,255,255),10)
        
        masked = cv2.bitwise_and(edge, mask)

        if maskedcontour_img != None:
            maskedcontour_img.data = masked

        contours0, hierarchy = cv2.findContours( masked, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        def contourBounding(cntr):
            minx = 999999999
            miny = 999999999
            maxx = 0
            maxy = 0

            for c in cntr:
                if minx >= c[0][0]:
                    minx = c[0][0]
                if miny >= c[0][1]:
                    miny = c[0][1]
                if maxx < c[0][0]:
                    maxx = c[0][0]
                if maxy < c[0][1]:
                    maxy = c[0][1]

            #minx = min(cntr, key=lambda x: x[0][0])
            #miny = min(cntr, key=lambda x: x[0][1])
            #maxx = max(cntr, key=lambda x: x[0][0])
            #maxy = max(cntr, key=lambda x: x[0][1])
            #return (minx[0][0], miny[0][1], maxx[0][0], maxy[0][1])
            return (minx, miny, maxx, maxy)

        def aspect_filter(r):
            aspect = abs(float(r[0])-float(r[2]))/(abs(float(r[1])-float(r[3])) + 0.00001)
            return ((0.8 < aspect) and (aspect < 1.2) )

        def rect_diagonal_sq(r):
            return (float(r[0])-float(r[2])) * (float(r[0])-float(r[2])) + (float(r[1])-float(r[3])) * (float(r[1])-float(r[3]))

        rects = [contourBounding(cnt) for cnt in contours0]


        for i in xrange(0, len(contours0) ):
            rect2 = filter(aspect_filter, rects)
            if len(rect2) != 0:
                rmax = max(rect2, key=rect_diagonal_sq)
                if rect_diagonal_sq(rmax) > 30*30:
                    found_rects = [rmax]

            if contour_img != None:
                cv2.drawContours(contour_img, contours0, i, color_table(i),-1, cv2.CV_AA, hierarchy, 3)
                for r in rect2:
                    cv2.rectangle(annotated, (r[0],r[1]), (r[2],r[3]), color_table(i))

        return found_rects
            
    def create_keypoint_mask(self, gray, debug_images=None):

        blur = cv2.GaussianBlur(gray ,(5,5),1,1)

        edge = cv2.Canny(blur, 2500, 5000, apertureSize=5)
        kpmask = np.zeros(edge.shape, np.uint8)

        annotated_img = None
        contour_img = None
        maskedcontour_img = None

        if debug_images != None:
            annotated_img = debug_images['annotated']
            contour_img = debug_images['contour']
            maskedcontour_img = debug_images['maskedcontour']

        if annotated_img != None:
            annotated_img[edge != 0] = (0, 255, 0)
                        
        #founds = self.find_circle_sign(blur, annotated_img)
        #for r in founds:
        #    cv2.rectangle(kpmask, (r[0],r[1]), (r[2],r[3]), (255,255,255), -1)
        #    cv2.rectangle(kpmask, (r[0],r[1]), (r[2],r[3]), (255,255,255), 10)

        founds = self.find_rect_sign(edge, annotated_img, contour_img, maskedcontour_img)
        for r in founds:
            cv2.rectangle(kpmask, (r[0],r[1]), (r[2],r[3]), (255,255,255), -1)
            cv2.rectangle(kpmask, (r[0],r[1]), (r[2],r[3]), (255,255,255), 10)

        return kpmask, founds

