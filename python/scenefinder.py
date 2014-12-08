import re
import os
import cv2
import numpy as np

from collections import namedtuple
import time

global TimeValue
TimeValue = namedtuple('TimeValue', 'time value')

class SceneFinder:
    def __init__(self):
        self.detector = cv2.ORB(100, 1.2, 4, 31, 0, 2)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.query_image = {}
        self.stats = {}

    def filterMatches(self, matches):
        matches = sorted(matches, key = lambda x:x.distance)
        good = filter(lambda x: x.distance < 60, matches)
        if len(good) < 10:
            raise RuntimeException('len(good) < 10')
        return good

    def detectObject(self, objname, scndesc):
        obj = self.query_image[objname]
        try:
            #print 'begin match ', objname
            #print obj.desc.type
            #print scn.desc.type
            #print obj.desc.cols(), scn.desc.cols()
            detected = self.matcher.match(obj.desc,scndesc)
            #print 'end match'

            return detected
        except Exception as e:
            print e
            return []

    def findRect(self, objname, scn):
        try:
            obj = self.query_image[objname]
            matches = self.matcher.match(obj.desc,scn.desc)
            good = self.filterMatches(matches)
            obj_points = []
            scn_points = []

            for m in good:
                #print m.distance
                obj_points.append( obj.kp[ m.queryIdx ].pt )
                scn_points.append( scn.kp[ m.trainIdx ].pt )

            H, status = cv2.findHomography(np.float32(obj_points), np.float32(scn_points), cv2.RANSAC )
            #print H, status

            obj_corners = np.float32(
                [[0, 0],
                [obj.width-1, 0],
                [obj.width-1, obj.height-1],
                [0, obj.height-1]]
            ).reshape(-1,1,2)

            #print np.float32(obj_corners)

            scn_corners = cv2.perspectiveTransform(np.float32(obj_corners), np.float32(H) )
            #print scn_corners

            return scn_corners
        except:
            return [[0,0], [0,0]]

    def computeImage(self, image, mask=None):
        kp, desc = self.detector.detectAndCompute(image, mask)
        #print image
        #print image.shape
        height, width  = image.shape

        ImageInfo = namedtuple('ImageInfo', 'image width height kp desc')
        return ImageInfo(image, width, height, kp, desc)

    def setQueryImage(self, name, image):
        imginfo = self.computeImage(image)
        root, ext = os.path.splitext( os.path.basename(name) )
        self.query_image[root] = imginfo
        self.stats[root] = []

    def setQueryImageFile(self, filename):
        img = cv2.imread(filename, 0)
        name = re.sub('\.*$', '', filename)
        self.setQueryImage(name, img)

    def drop_before(self, timevalues, before):
        return filter( lambda tv: tv.time >= before, timevalues)

    def findInScene(self, scene, mask=None, hint=None):
        scn = self.computeImage(scene, mask)
        result = {}
        QueryResult = namedtuple('QueryResult', 'matches stats dist_ave')

        keys = self.query_image.keys()

        blue_sign = [ 'honk', 'haead', 'lane', 'roundabout', 'parking' ]
        red_sign = [ '30', '60', '90', 'slow', 'tomare', 'tuukoudome' ]
        yellow_sign = [ 'dear', 'exclaimation', 'rotate' ]
        if hint != None:
            print hint
            if hint[0] > hint[1] and hint[0] > hint[2]:
                # blue
                keys = list(set(blue_sign) & set(keys) )
                print keys
            elif hint[2] > hint[0] and hint[2] - hint[1] >= 20:
                # red
                keys = list(set(red_sign) & set(keys) )
                print keys
            elif hint[2] > hint[0]:
                keys = list(set(red_sign) & set(yellow_sign) & set(keys) )
                print keys
            else:
                keys = []
        else:
            keys = []

        print keys
        if scn != None and scn.desc != None:
            for k in keys:
                #matches = self.detectObject(self.query_image[k], scn)
                matches = self.detectObject(k, scn.desc)

                dist_ave = np.average( [m.distance for m in matches] )
                old_stats = self.stats[k]
                old_stats.append( TimeValue(time.time(), dist_ave) )
                new_stats = self.drop_before(old_stats, (time.time() - 2) )

                self.stats[k] = new_stats

                result[k] = QueryResult(matches, new_stats, dist_ave)

        return result, scn

