import svgwrite
import sys
from enum import Enum


class CheckingDepth(Enum):
	first_stage = 1
	larger_stage = 2


class DirectionPriority(Enum):
	left = 0
	up = 1
	right = 2
	down = 3


def outer_remover(list):
	outers = []
	for elem in list:
		x = elem[0]
		y = elem[1]
		if (x - 1, y) in list and (x + 1, y) in list:
			outers.append(elem)
	for elem in outers:
		list.remove(elem)


def is_all_used(list):
	for elem in list:
		if elem[2] == 0:
			return False
	return True


def get_left_side(x, y, polygon):
	if (x, y - 1) in polygon:
		return True
	else:
		return False


def get_upper_side(x, y, polygon):
	if (x - 1, y) in polygon:
		return True
	else:
		return False


def get_right_side(x, y, polygon):
	if (x, y + 1) in polygon:
		return True
	else:
		return False


def get_down_side(x, y, polygon):
	if (x + 1, y) in polygon:
		return True
	else:
		return False


def try_right_side(x, y, polygon):
	index = 1
	while True:
		if (x, y + index) in polygon:
			if (x + 1, y + index) in polygon:
				return True
			else:
				index += 1
		else:
			return False


def try_left_side(x, y, polygon):
	index = 1
	while True:
		if (x, y - index) in polygon:
			if (x - 1, y - index) in polygon:
				return True
			else:
				index += 1
		else:
			return False


def move_to_right(x, y, polygon):
	index = 1
	while True:
		if (x, y + index) in polygon:
			if (x + 1, y + index) in polygon:
				return x + 1, y + index
			else:
				index += 1


def move_to_left(x, y, polygon):
	index = 1
	while True:
		if (x, y - index) in polygon:
			if (x - 1, y - index) in polygon:
				return x - 1, y - index
			else:
				index += 1


current_polygon = [(2, 7), (1, 9), (1, 10), (2, 8), (2, 9), (2, 10), (3, 8), (3, 9), (3, 10), (2, 16), (3, 16), (4, 3),
				   (4, 4), (4, 5), (4, 8), (4, 9), (4, 10), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (5, 3), (5, 4),
				   (5, 5), (5, 8), (5, 9), (5, 10), (5, 14), (5, 15), (5, 16), (6, 5), (6, 6), (6, 7), (6, 8), (6, 13),
				   (6, 14), (6, 15), (7, 2), (7, 3), (7, 5), (7, 6), (7, 7), (7, 8), (7, 13), (7, 14), (7, 15), (8, 2),
				   (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14),
				   (8, 15), (8, 16), (9, 2), (9, 3), (9, 4), (9, 7), (9, 8), (9, 9), (9, 10), (9, 11), (9, 12), (10, 2),
				   (10, 3), (10, 4), (10, 8), (10, 9), (10, 10), (11, 9), (11, 10), (4, 6), (3, 4), (4, 2)]

current_polygon.sort()
print(current_polygon)
test = svgwrite.Drawing('test.svg', profile='tiny')

edge_x = current_polygon[0][0]
edge_y = current_polygon[0][1]

string = "M%d,%d " % (edge_x, edge_y)
ordered_edges = [(edge_x, edge_y)]

is_edge = False
priority = DirectionPriority.down.value
while True:
	if priority == DirectionPriority.down.value:
		if get_down_side(edge_x, edge_y, current_polygon):
			edge_x += 1
			priority = DirectionPriority.left.value
		elif try_right_side(edge_x, edge_y, current_polygon):
			edge_x, edge_y = move_to_right(edge_x, edge_y, current_polygon)
			is_edge = True
		if is_edge:
			ordered_edges.append((edge_x, edge_y))
			is_edge = False
	if priority == DirectionPriority.left.value:
		if get_left_side(edge_x, edge_y, current_polygon):
			edge_y -= 1
			if get_upper_side(edge_x, edge_y, current_polygon) and not get_upper_side(edge_x, edge_y + 1,
																					  current_polygon):
				edge_x -= 1
				priority = DirectionPriority.up.value
				is_edge = True
			elif get_down_side(edge_x, edge_y, current_polygon) and not get_down_side(edge_x, edge_y + 1,
																					  current_polygon):
				edge_x += 1
				priority = DirectionPriority.down.value
				is_edge = True
		else:
			is_edge = True
			priority = DirectionPriority.down.value
		if is_edge:
			ordered_edges.append((edge_x, edge_y))
			is_edge = False
	if priority == DirectionPriority.up.value:
		if get_upper_side(edge_x, edge_y, current_polygon):
			edge_x -= 1
			priority = DirectionPriority.right.value
		elif try_left_side(edge_x, edge_y, current_polygon):
			edge_x, edge_y = move_to_left(edge_x, edge_y, current_polygon)
			is_edge = True
		if is_edge:
			ordered_edges.append((edge_x, edge_y))
			is_edge = False
	if priority == DirectionPriority.right.value:
		if get_right_side(edge_x, edge_y, current_polygon):
			edge_y += 1
			if get_upper_side(edge_x, edge_y, current_polygon) and not get_upper_side(edge_x, edge_y - 1,
																					  current_polygon):
				edge_x -= 1
				priority = DirectionPriority.up.value
				is_edge = True
			elif get_down_side(edge_x, edge_y, current_polygon) and not get_down_side(edge_x, edge_y - 1,
																					  current_polygon):
				edge_x += 1
				priority = DirectionPriority.down.value
				is_edge = True
		else:
			is_edge = True
			priority = DirectionPriority.up.value
		if is_edge:
			ordered_edges.append((edge_x, edge_y))
			is_edge = False

string += "Z"
print(string)
test_path = test.path(d=string,
					  stroke="#000",
					  fill=svgwrite.rgb(59, 18, 133),
					  stroke_width=0.1)
test.add(test_path)
test.save()
