import math
import sys
from enum import Enum
import cv2
import numpy as np
import vectorization
import svgwrite
from skimage import io


class PolygonStatus(Enum):
    recounted = 0
    too_small = 1
    not_counted = 2


def benchmark(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('[*] Время выполнения {}: {} секунд.'.format(func.__name__, end - start))
        return result

    return wrapper


def is_equal(list_1, list_2):
    if list_1[0] != list_2[0]:
        return False
    if list_1[1] != list_2[1]:
        return False
    if list_1[2] != list_2[2]:
        return False
    if list_1[3] != list_2[3]:
        return False
    return True


def get_polygon(image, curr_x, curr_y, status_matrix):
    def compare(x, y):
        if is_equal(color, image[x][y]) and status_matrix[x][y] == PolygonStatus.not_counted.value:
            polygon.append((x, y))
            status_matrix[x][y] = PolygonStatus.recounted.value

    polygon = [(curr_x, curr_y)]
    color = image[curr_x][curr_y]
    status_matrix[curr_x][curr_y] = PolygonStatus.recounted.value
    x_len = len(image)
    y_len = len(image[0])
    index = 0
    while index < len(polygon):
        curr_x, curr_y = polygon[index]
        if curr_y - 1 >= 0:
            compare(curr_x, curr_y - 1)
        if curr_y + 1 < y_len:
            compare(curr_x, curr_y + 1)
        if curr_x - 1 >= 0:
            compare(curr_x - 1, curr_y)
        if curr_x + 1 < x_len:
            compare(curr_x + 1, curr_y)
        index += 1
    return polygon


@benchmark
def polygon_recount(image_matrix):
    x_len = len(image_matrix)
    y_len = len(image_matrix[0])
    min_limit = int(x_len * y_len / 100 * 0.005)
    print('min lim - ', min_limit)
    is_counted = [[PolygonStatus.not_counted.value] * y_len for _ in range(x_len)]
    for x in range(x_len):
        for y in range(y_len):
            if is_counted[x][y] == PolygonStatus.not_counted.value:
                current_polygon = get_polygon(image_matrix, x, y, is_counted)
                if len(current_polygon) <= min_limit:
                    for x_curr, y_curr in current_polygon:
                        is_counted[x_curr][y_curr] = PolygonStatus.too_small.value
    return is_counted


def pixel_difference(center, pixel):
    center = center.astype(np.int32)
    pixel = pixel.astype(np.int32)
    distance = ((center[0] - pixel[0]) ** 2
                + (center[1] - pixel[1]) ** 2
                + (center[2] - pixel[2]) ** 2
                + (center[3] - pixel[3]) ** 2) ** (1 / 2)
    return distance


@benchmark
def polygon_merge(image_matrix, polygon_status):
    x_len = len(image_matrix)
    y_len = len(image_matrix[0])
    for x in range(x_len):
        for y in range(y_len):
            if polygon_status[x][y] == PolygonStatus.too_small.value:
                min_distance = sys.maxsize
                color_difference = 0
                new_color = []
                if y > 0:
                    color_difference = pixel_difference(image_matrix[x][y], image_matrix[x][y - 1])
                    if color_difference < min_distance:
                        min_distance = color_difference
                        new_color = image_matrix[x][y - 1]
                if x > 0:
                    color_difference = pixel_difference(image_matrix[x][y], image_matrix[x - 1][y])
                    if color_difference < min_distance:
                        min_distance = color_difference
                        new_color = image_matrix[x - 1][y]
                if y < y_len - 1:
                    if polygon_status[x][y + 1] != PolygonStatus.too_small.value:
                        color_difference = pixel_difference(image_matrix[x][y], image_matrix[x][y + 1])
                        if color_difference < min_distance:
                            min_distance = color_difference
                            new_color = image_matrix[x][y + 1]
                if x < x_len - 1:
                    if polygon_status[x + 1][y] != PolygonStatus.too_small.value:
                        color_difference = pixel_difference(image_matrix[x][y], image_matrix[x + 1][y])
                        if color_difference < min_distance:
                            min_distance = color_difference
                            new_color = image_matrix[x + 1][y]
                image_matrix[x][y] = new_color
                polygon_status[x][y] = PolygonStatus.recounted.value
    return image_matrix


def contouring(image_matrix):
    final_img = [[[0] * 4 for _ in range(dimensions[0])] for _ in range(dimensions[1])]
    black = np.array([80, 80, 80, 255]).astype(np.uint8)
    white = np.array([255, 255, 255, 255]).astype(np.uint8)
    x_len = len(image_matrix)
    y_len = len(image_matrix[0])
    for x in range(x_len):
        for y in range(y_len):
            if x != (x_len - 1) and y != (y_len - 1):
                if not is_equal(image_matrix[x][y], image_matrix[x][y + 1]) \
                        or not is_equal(image_matrix[x][y], image_matrix[x + 1][y]):
                    final_img[x][y] = black
                else:
                    final_img[x][y] = white
            elif x == (height - 1) and y != (width - 1):
                if not is_equal(image_matrix[x][y], image_matrix[x][y + 1]):
                    final_img[x][y] = black
                else:
                    final_img[x][y] = white
            elif y == (width - 1) and x != (height - 1):
                if not is_equal(image_matrix[x][y], image_matrix[x + 1][y]):
                    final_img[x][y] = black
                else:
                    final_img[x][y] = white
            else:
                final_img[x][y] = white
    return final_img


@benchmark
def clustering(rgba, cluster_centers):
    x_len = len(rgba)
    y_len = len(rgba[0])
    cluster_dic = [[] for _ in range(0, N)]
    itr = 0
    new_image = [[[0] * 4 for _ in range(y_len)] for _ in range(x_len)]
    while True:
        for x in range(x_len):
            for y in range(y_len):
                pixel = rgba[x, y].astype(np.int32)
                min_distance = sys.maxsize
                cluster_id = 0
                for i in range(0, N):
                    current_center = cluster_centers[i].astype(np.int32)
                    distance = pixel_difference(current_center, pixel)
                    if distance < min_distance:
                        min_distance = distance
                        cluster_id = i
                cluster_dic[cluster_id].append(pixel)
                new_image[x][y] = cluster_centers[cluster_id]
        for i in range(0, N):
            current_cluster = cluster_dic[i]
            if len(current_cluster) > 0:
                summ_a = summ_r = summ_g = summ_b = 0
                for pixel in current_cluster:
                    summ_r += pixel[0]
                    summ_g += pixel[1]
                    summ_b += pixel[2]
                    summ_a += pixel[3]
                cluster_centers[i][0] = summ_r // len(current_cluster)
                cluster_centers[i][1] = summ_g // len(current_cluster)
                cluster_centers[i][2] = summ_b // len(current_cluster)
                cluster_centers[i][3] = summ_a // len(current_cluster)
        print(itr)
        itr += 1
        if itr > 3:
            break
    return new_image


def coloring(image, image_contours):
    x_len = len(image)
    y_len = len(image[0])
    white = np.array([255, 255, 255, 255]).astype(np.uint8)
    colored_image = [[[0] * 4 for _ in range(y_len)] for _ in range(x_len)]
    for x in range(x_len):
        for y in range(y_len):
            if is_equal(image_contours[x][y], white):
                colored_image[x][y] = image[x][y]
            else:
                colored_image[x][y] = image_contours[x][y]
    return colored_image


@benchmark
def image_to_svg(image):
    x_len = len(image)
    y_len = len(image[0])
    is_counted = [[PolygonStatus.not_counted.value] * y_len for _ in range(x_len)]
    svg_image = svgwrite.Drawing('vectorized_image.svg', profile='tiny')
    for x in range(x_len):
        for y in range(y_len):
            color = []
            if is_counted[x][y] == PolygonStatus.not_counted.value:
                current_polygon = get_polygon(image, x, y, is_counted)
                for ind in range(3):
                    color.append(image[x][y][ind])
                edges = vectorization.start_vectorization(current_polygon)
                polygon_path = svgwrite.shapes.Polyline(points=edges, fill=svgwrite.rgb(color[0], color[1], color[2]), stroke="#000", stroke_width=0.5)
                polygon_path.translate(1)
                svg_image.add(polygon_path)
    svg_image.save()


original_image = io.imread('img_1.jpg')

max_dimension = 200
original_dimensions = [original_image.shape[0], original_image.shape[1]]
if max(original_dimensions[0], original_dimensions[1]) == original_image.shape[0]:
    height = max_dimension
    scale_percent = int(max_dimension / (original_image.shape[0] / 100))
    width = int(original_image.shape[1] * scale_percent / 100)
else:
    width = max_dimension
    scale_percent = int(max_dimension / (original_image.shape[1] / 100))
    height = int(original_image.shape[0] * scale_percent / 100)
# width = height = 200
dimensions = (width, height)
sys.setrecursionlimit(width * height)

resized = cv2.resize(original_image, dimensions, interpolation=cv2.INTER_AREA)
rgba_image = cv2.cvtColor(resized, cv2.COLOR_RGB2RGBA)

# amount of clusters
N = math.ceil(max(width, height) / 100 * 15)
# N = 20
print('clusters - ', N)
# random initialization of clusters' centroids
cluster_centers = []
for i in range(0, N):
    x_rand = np.random.randint(low=0, high=dimensions[1])
    y_rand = np.random.randint(low=0, high=dimensions[0])
    cluster_centers.append(rgba_image[x_rand, y_rand])
cluster_centers = np.asarray(cluster_centers)

new_img = [[[0] * 4 for _ in range(dimensions[0])] for _ in range(dimensions[1])]

# clustering k-means
new_img = clustering(rgba_image, cluster_centers)

polygon_status = polygon_recount(new_img)
image_recounted = polygon_merge(new_img, polygon_status)

clustered_image = cv2.cvtColor(np.asarray(image_recounted), cv2.COLOR_RGBA2BGR)

# contouring for black-white image
contours = contouring(image_recounted)
result = cv2.cvtColor(np.asarray(coloring(image_recounted, contours)), cv2.COLOR_RGBA2BGR)

contoured_image = cv2.cvtColor(np.asarray(contours), cv2.COLOR_RGBA2BGR)
image_to_svg(image_recounted)

# cv2.imshow("result", result)
# cv2.imshow("contours", contoured_image)
# cv2.imshow("clusters", clustered_image)
# out = contouring(new_img)
# cv2.imshow("clust", cv2.cvtColor(np.asarray(new_img), cv2.COLOR_RGBA2BGR))
# cv2.imshow("clusters", cv2.cvtColor(np.asarray(out), cv2.COLOR_RGBA2BGR))

cv2.waitKey(0)
