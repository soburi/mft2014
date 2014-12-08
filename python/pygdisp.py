#!/usr/bin/python

import numpy
import cv2
import cv2.cv as cv

class PygDisp:
	def __init__(self, pyg, w, h):
		self.pyg = pyg
		size = (pyg.display.Info().current_w, pyg.display.Info().current_h)

		scr_width = w
		scr_height  = h

		if (w < 0):
			scr_width = float(size[0])
			scr_height = float(size[1])

		if( scr_width/w >= scr_height/h):
			scale = scr_height/h
			woff = (scr_width-(w*scale))/2
			self.dist_rect = (woff, 0, scr_width-woff*2, scr_height)
			print self.dist_rect
			self.surf = pyg.Surface((scr_width,scr_height))
		else:
			scale = scr_width/w
			hoff = (scr_height-(h*scale))/2
			self.dist_rect = (0, hoff, scr_width, scr_height-hoff*2)
			print self.dist_rect
			self.surf = pyg.Surface((scr_width,scr_height))

		size = (int(scr_width), int(scr_height))
		print size
		self.screen = pyg.display.set_mode(size)

	def cvimage_to_pygame(self, image):
		"""Convert cvimage into a pygame image"""
		return self.pyg.image.frombuffer(image.tostring(), image.shape[1::-1], "RGB")

	def blit(self, img):
		npimg = numpy.asarray(img[:,:])
		(h,w, depth) = img.shape
		offscr = self.pyg.Surface((w,h))
		offscr.blit(self.cvimage_to_pygame(cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB) ), offscr.get_rect() )
		scaled = self.pyg.transform.scale(offscr, (int(self.dist_rect[2]), int(self.dist_rect[3])) )
		#self.surf.blit(self.cvimage_to_pygame(cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB) ), self.surf.get_rect() )
		#scaled = self.pyg.transform.scale(self.surf, (int(self.dist_rect[2]), int(self.dist_rect[3])) )
		#scaled = self.pyg.transform.scale2x(self.surf)
		self.screen.blit(scaled, (0,0) ) #, self.dist_rect )
		self.pyg.display.flip()
