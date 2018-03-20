import matplotlib.pyplot as plt

def make_arr(filename):
	file = open(filename, "r+")
	arr = []	
	liness = []
	line = file.readlines()

	for j in range(len(line)):
		lines = line[j].split(" ")
		liness.append(lines)

	for i in range(len(liness)):
		arr.append(liness[i][1].strip('\n'))
	return arr

def plot(array):
	plt.plot(array)
	plt.show()


a = make_arr("data_vae.txt")
plot(a)