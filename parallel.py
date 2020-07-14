import copy
import sys
from timeit import default_timer as timer

import cv2
import numpy as np
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from skimage import io


def cluster_affiliation (pixels_line, cluster_centers, x):
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	N = len(cluster_centers)
	cluster_dic = [[] for _ in range(0, N)]
	new_img_line = [[0] * 4 for _ in range(len(pixels_line))]
	for y in range(len(pixels_line)):
		pixel = pixels_line[y].astype(np.int32)
		min_distance = sys.maxsize
		cluster_id = 0
		for i in range(N):
			current_center = cluster_centers[i].astype(np.int32)
			distance = ((current_center[0] - pixel[0]) ** 2
				+ (current_center[1] - pixel[1]) ** 2
				+ (current_center[2] - pixel[2]) ** 2
				+ (current_center[3] - pixel[3]) ** 2) ** (1 / 2)
			if distance < min_distance:
				min_distance = distance
				cluster_id = i
		cluster_dic[cluster_id].append(pixel)
		new_img_line[y] = cluster_centers[cluster_id]
	return cluster_dic, new_img_line, x

def recalculate_centroids(current_cluster, centers, index):
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	cluster_center = centers[index]
	if len(current_cluster) > 0:
		summ_a = summ_r = summ_g = summ_b = 0
		for pixel in current_cluster:
			summ_r += pixel[0]
			summ_g += pixel[1]
			summ_b += pixel[2]
			summ_a += pixel[3]
		cluster_center[0] = summ_r // len(current_cluster)
		cluster_center[1] = summ_g // len(current_cluster)
		cluster_center[2] = summ_b // len(current_cluster)
		cluster_center[3] = summ_a // len(current_cluster)
	return cluster_center, index

def is_equal(list_1, list_2):
    flag = True
    for i in range(len(list_1)):
        if list_1[i] != list_2[i]:
            flag = False
            break
    return flag


if __name__ == '__main__':
	original_image = io.imread('img.jpg')

	scale_percent = 60
	#width = int(original_image.shape[1] * scale_percent / 100)
	#height = int(original_image.shape[0] * scale_percent / 100)
	width = 100
	height = 100
	dimensions = (width, height)

	resized = cv2.resize(original_image, dimensions, interpolation=cv2.INTER_AREA)
	rgba_image = cv2.cvtColor(resized, cv2.COLOR_RGB2RGBA)

	# количество кластеров
	N = 10
	# первичная инициализация центров кластеров рандомными значениями
	cluster_centers = []
	for i in range(0, N):
		x_rand = np.random.randint(low=0, high=dimensions[1])
		y_rand = np.random.randint(low=0, high=dimensions[0])
		cluster_centers.append(rgba_image[x_rand, y_rand])
	cluster_centers = np.asarray(cluster_centers)

	cluster_dic = [[] for _ in range(0, N)]
	cluster_dic_prev = []
	itr = 0
	new_img = [[[0] * 4 for _ in range(dimensions[0])] for _ in range(dimensions[1])]
	np.asarray(new_img)
	start = timer()
	while (True):
		steps = 0
		with MPIPoolExecutor() as executor:
			for result in executor.map(cluster_affiliation, rgba_image, [cluster_centers] * height, range(height)):
				steps += 1
				for i in range(N):
					for elem in result[0][i]:
						cluster_dic[i].append(elem)
				new_img[result[2]] = result[1]
		finished = True
		while finished == True:
			if steps == height:
				finished = False
		steps = 0
		with MPIPoolExecutor() as executor:
			results = []
			for result in executor.map(recalculate_centroids, cluster_dic, [cluster_centers] * N, range(N)):
				steps += 1
				centroid_index = result[1]
				cluster_centers[centroid_index] = result[0]
		finished = True
		while finished == True:
			if steps == N:
				finished = False

		equality = True
		if len(cluster_dic_prev) > 0:
			for i in range(0, N):
				if cluster_dic_prev[i] != cluster_dic[i]:
					equality = False
					break
		else:
			equality = False
		cluster_dic_prev = copy.deepcopy(cluster_dic)
		print(itr)
		itr += 1
		if equality == True or itr > 3:
			break
	end = timer()
	print(cluster_dic[0])
	out = cv2.cvtColor(np.asarray(new_img), cv2.COLOR_RGBA2BGR)
	print('Time - ', end - start)
	#cv2.imshow("cont", out)
	cv2.waitKey(0)