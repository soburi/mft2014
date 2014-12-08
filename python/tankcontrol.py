import smbus
import time


class TankControl:
	def __init__(self):
		self.speed = 0
		self.i2c = smbus.SMBus(1)

	def control(self, cmd, values):
		adr = 0x4
		self.i2c.write_i2c_block_data(adr, cmd, values)

	def speed_control(self, left, right):
		cmd = 1
		print left, right
		return self.control(cmd, [0, left, 1, right, 2, 0])

	def rotate_control(self, rot):
		cmd = 1
		print left, right
		return self.control(cmd, [0, 0, 1, 0, 2, rot])

	def special_control(self, param):
		cmd = 1
		print param
		return self.control(cmd, [3, param])

	def ahead(self, marker):
		#say_text(funcname)
		self.direction = 0
		self.speed_control(self.speed, self.speed)

	def right(self, marker):
		#say_text(funcname)
		self.direction = 1
		self.speed_control((self.speed*0.6).round, self.speed)

	def left(self, marker):
		#say_text(funcname)
		self.direction = -1
		self.speed_control(self.speed, (self.speed*0.6).round)

	def rotate(self, marker):
		#say_text(funcname)
		self.speed_control(self.speed, -self.speed)

	def stop(self, marker):
		#say_text(funcname)
		self.speed_control(0, 0)

	def slow(self, arker):
		#say_text(funcname)
		self.speed -= 30
		self.speed_control(self.speed, self.speed)

	def spd100(self, marker):
		#say_text(funcname)
		self.speed = 100
		self.speed_control(self.speed, self.speed)

	def spd50(self, marker):
		#say_text(funcname)
		self.speed = 50
		self.speed_control(self.speed, self.speed)

	def honk(self, marker):
		self.speed_control(0, 124)
		#if(self.stop_beep == nil):
		#	 self.stop_beep = ASync.new { speed_control(0,123) }

		self.stop_beep.start(1)

	def tomare(self, marker):
		self.speed_control(0, 0)

	def dear(self, marker):
		rotate_control(100)

	def train(self, marker):
		print 'train'

	def back(self, marker):
		self.speed_control(-100, -100)

	def special1(self, marker):
		self.special_control(1)

	def special2(self, marker):
		self.special_control(2)


