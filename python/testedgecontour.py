#!/usr/bin/env python

'''
This sample demonstrates Canny edge detection.

Usage:
  edge.py [<video source>]

  Trackbars control edge thresholds.

'''

import cv2

import cv2.cv as cv
import numpy as np
import video
import sys
import time

from collections import namedtuple

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


global TimeValue
TimeValue = namedtuple('TimeValue', 'time value')

def drop_before(timevalues, before):
    return filter( lambda tv: tv.time >= before, timevalues)

if __name__ == '__main__':
    print __doc__

    try: fn = sys.argv[1]
    except: fn = 0

    def nothing(*arg):
        pass

    cv2.namedWindow('edge')
    cv2.createTrackbar('thrs1', 'edge', 2500, 5000, nothing)
    cv2.createTrackbar('thrs2', 'edge', 5000, 5000, nothing)
    cv2.createTrackbar('HL votes', 'edge', 100, 200, nothing)
    cv2.createTrackbar( "levels", "edge", 3, 7, nothing)

    cap = video.create_capture(fn)

    lines_stats = []


    while True:
        flag, img = cap.read()
        h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thrs1 = cv2.getTrackbarPos('thrs1', 'edge')
        thrs2 = cv2.getTrackbarPos('thrs2', 'edge')
        edge = cv2.Canny(gray, thrs1, thrs2, apertureSize=5) 

        votes = cv2.getTrackbarPos('HL votes', 'edge')

        circles = None
        circles = cv2.HoughCircles(edge, cv.CV_HOUGH_GRADIENT,1, 30#)
                            ,param1=100,param2=votes,minRadius=20)#,maxRadius=0) 




        lines = cv2.HoughLines(edge, 1, np.pi/180, votes)

        lines_stats.append( TimeValue(time.time(), lines) )

        lines_stats = drop_before(lines_stats, time.time() -1)


        circle_rects = []

        vis = img.copy()
        if circles != None:
            circles = np.round(circles[0, :]).astype("int") 
            for (x, y, r) in circles:
                cv2.circle(vis, (x, y), r, (255, 255, 0), 4)
                cv2.rectangle(vis, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1) 
                cv2.rectangle(vis, (x-r, y-r), (x+r, y+r) , (255,255,255), 10)

                circle_rects.append( (x-r, y-r, x+r, y+r) )

        mask = np.zeros(edge.shape, np.uint8)


        for t_lines in lines_stats:
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
                    
                    cv2.line(mask,(x1,y1),(x2,y2),(255,255,255),10)



        vis /= 2
        vis[edge != 0] = (0, 255, 0)

        masked = cv2.bitwise_and(edge, mask)

        contours0, hierarchy = cv2.findContours( masked.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        def contourBounding(cntr):
            minx = min(cntr, key=lambda x: x[0][0])
            miny = min(cntr, key=lambda x: x[0][1])
            maxx = max(cntr, key=lambda x: x[0][0])
            maxy = max(cntr, key=lambda x: x[0][1])

            return (minx[0][0], miny[0][1], maxx[0][0], maxy[0][1])

        tmp = []
        contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]
        rects    = [contourBounding(cnt) for cnt in contours]

        vis2 = np.zeros((h, w, 3), np.uint8)

        kpmask = np.zeros(edge.shape, np.uint8)

        def aspect_filter(r):
            aspect = abs(float(r[0])-float(r[2]))/(abs(float(r[1])-float(r[3])) + 0.00001)
            return ((0.8 < aspect) and (aspect < 1.2) )

        def rect_diagonal(r):
            return pow( pow(abs(float(r[0])-float(r[2])), 2) + pow(abs(float(r[1])-float(r[3])),2), 0.5)

        for i in xrange(0, len(contours) ):
            cv2.drawContours(vis2, contours, i, color_table(i),-1, cv2.CV_AA, hierarchy, 3)
            rect2 = filter(aspect_filter, rects)

            print circle_rects

            print rect2

            rect2 += circle_rects

            for r in rect2:
                cv2.rectangle(vis2, (r[0],r[1]), (r[2],r[3]), color_table(i))

            if len(rect2) != 0:
                rmax = max(rect2, key=rect_diagonal)
                if rect_diagonal(rmax) > 60:
                    cv2.rectangle(kpmask, (rmax[0],rmax[1]), (rmax[2],rmax[3]), (255,255,255), 10)

        vis[masked != 0] = (0,0,255)

        cv2.imshow('edge', masked)
        cv2.imshow('vis', vis)
        cv2.imshow('contours', vis2)
        cv2.imshow('kpmask', kpmask)

        
        ch = cv2.waitKey(5)
        if ch == 27:
            break
    cv2.destroyAllWindows()
