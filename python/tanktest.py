import tankcontrol

try:
	tc = tankcontrol.TankControl()
	#print tc.ahead(None)
	#print tc.spd100(None)
	#print tc.back(None)
	print tc.special2(None)
except Exception as e:
	print e
