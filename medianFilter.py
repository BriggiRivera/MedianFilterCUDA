import math
import numpy as np
import os
import PIL
import sys
import time

from numba import cuda, int32
from os import listdir
from os.path import join
from PIL import Image

@cuda.jit(device = True)
def sort(array, size):
	for i in range(size):
		for j in range(i+1, size):
			if array[i] > array[j]:
				array[i], array[j] = array[j], array[i]

@cuda.jit
def medianFilter(image, processed, n):
	x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	middle = int(n/2)

	if x >= middle and y >= middle and x < image.shape[0] - middle and y < image.shape[1] - middle: 
		size = 256 #max
		r = cuda.local.array(size, int32)
		g = cuda.local.array(size, int32)
		b = cuda.local.array(size, int32)

		middle = int(n/2)
		cont = 0
		for i in range(x - middle, x + middle + 1):
			for j in range(y - middle, y + middle + 1):
				r[cont] = image[i][j][0]
				g[cont] = image[i][j][1]
				b[cont] = image[i][j][2]
				cont += 1

		sort(r, n*n)
		sort(g, n*n)
		sort(b, n*n)
	
		processed[x][y][0] = r[int(n*n/2)]
		processed[x][y][1] = g[int(n*n/2)]
		processed[x][y][2] = b[int(n*n/2)]

def getImages(path):
	return [join(path, x) for x in os.listdir(path)]

def processImageCUDA(image, n):
	image_HOST = np.array(image, copy=True)
	image_in_DEVICE = cuda.to_device(image_HOST)
	image_ou_DEVICE = cuda.device_array(image_HOST.shape)

	blocksInX = math.ceil(image_HOST.shape[0] / 32.0)
	blocksInY = math.ceil(image_HOST.shape[1] / 32.0)
	blocksPerGrid = (blocksInX, blocksInY)
	threadsPerBlock = (32, 32)
	
	medianFilter[blocksPerGrid, threadsPerBlock](image_in_DEVICE, image_ou_DEVICE, n)
	processed_HOST = image_ou_DEVICE.copy_to_host()
	return processed_HOST

def processImageCPU(img, n):
	image = np.array(img, copy=True)
	processed = np.zeros(image.shape)
	middle = int(n/2)
	r = np.zeros(n*n)
	g = np.zeros(n*n)
	b = np.zeros(n*n)
	for x in range(middle, image.shape[0] - middle):
		for y in range(middle, image.shape[1] - middle):
			cont = 0
			for i in range(x - middle, x + middle + 1):
				for j in range(y - middle, y + middle + 1):
					r[cont] = image[i][j][0]
					g[cont] = image[i][j][1]
					b[cont] = image[i][j][2]
					cont += 1
			r.sort()
			g.sort()
			b.sort()
			processed[x][y][0] = r[int(n*n/2)]
			processed[x][y][1] = g[int(n*n/2)]
			processed[x][y][2] = b[int(n*n/2)]

	return processed

def main(argv):

	if len(argv) > 2:
		print("Scanning folder: ", argv[1])
		paths = getImages(argv[1])
		cont = 1
		for path in paths:
			image = np.asarray(Image.open(path))
			print(path)
			# ---------------------------------------------------------
			start = time.time()
			result = processImageCUDA(image, int(argv[3]))
			end = time.time()
			print("CUDA\t", argv[3], "\t" , path, "\t", end - start, "\t", result.shape)
			# ---------------------------------------------------------
			start = time.time()
			result = processImageCPU(image, int(argv[3]))
			end = time.time()
			print("CPU\t", argv[3], "\t" , path, "\t", end - start, "\t", result.shape)
			# ---------------------------------------------------------
			result = Image.fromarray(result.astype('uint8'))
			result.save(join(argv[2], "{0}.jpg".format(cont)))
			cont = cont + 1
	pass

def main1(argv):
	image = np.asarray(Image.open('Thor.png'))
	image_HOST = np.array(image, copy=True)
	image_in_DEVICE = cuda.to_device(image_HOST)
	image_ou_DEVICE = cuda.device_array(image_HOST.shape)

	blocksInX = math.ceil(image_HOST.shape[0] / 32.0)
	blocksInY = math.ceil(image_HOST.shape[1] / 32.0)
	blocksPerGrid = (blocksInX, blocksInY)
	threadsPerBlock = (32, 32)

	medianFilter[blocksPerGrid, threadsPerBlock](image_in_DEVICE, image_ou_DEVICE, 10)

	processed_HOST = image_ou_DEVICE.copy_to_host()
	result = Image.fromarray(processed_HOST.astype('uint8'))
	result.save('Thor2.jpeg')

if __name__ == "__main__":
	main(sys.argv)
