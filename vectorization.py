import svgwrite
from enum import Enum


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


def is_start(x, y, edges):
	if edges[0][1] == x and edges[0][0] == y:
		return True
	else:
		return False


# current_polygon = [(2, 7), (1, 9), (1, 10), (2, 8), (2, 9), (2, 10), (3, 8), (3, 9), (3, 10), (2, 16), (3, 16), (4, 3),
# 				   (4, 4), (4, 5), (4, 8), (4, 9), (4, 10), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (5, 3), (5, 4),
# 				   (5, 5), (5, 8), (5, 9), (5, 10), (5, 14), (5, 15), (5, 16), (6, 5), (6, 6), (6, 7), (6, 8), (6, 13),
# 				   (6, 14), (6, 15), (7, 2), (7, 3), (7, 5), (7, 6), (7, 7), (7, 8), (7, 13), (7, 14), (7, 15), (8, 2),
# 				   (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14),
# 				   (8, 15), (8, 16), (9, 2), (9, 3), (9, 4), (9, 7), (9, 8), (9, 9), (9, 10), (9, 11), (9, 12), (10, 2),
# 				   (10, 3), (10, 4), (10, 8), (10, 9), (10, 10), (11, 9), (11, 10), (4, 6), (3, 4), (4, 2)]

current_polygon = [(1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 21), (1, 22), (1, 24),
				   (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (2, 15), (2, 21), (2, 22), (2, 23), (2, 24),
				   (3, 9), (3, 10), (3, 11), (3, 12), (3, 18), (3, 19), (3, 21), (3, 22),
				   (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 17), (4, 18), (4, 19), (4, 20),
				   (4, 21), (4, 22), (4, 23), (5, 9), (5, 10), (5, 11), (5, 12), (5, 13), (5, 14), (5, 15),
				   (5, 18), (5, 19), (5, 21), (5, 23), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6),
				   (6, 9), (6, 10), (6, 11), (6, 12), (6, 18), (6, 19), (5, 25), (6, 25), (7, 4), (7, 5), (7, 6),
				   (7, 9), (7, 10), (7, 11), (7, 12), (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (7, 18), (7, 19),
				   (7, 20), (7, 21), (7, 22), (7, 23), (7, 24), (7, 25), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8),
				   (8, 9), (8, 10), (8, 13), (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (8, 19), (9, 6), (9, 7), (9, 8),
				   (9, 15), (9, 16), (9, 17), (9, 18), (9, 19), (9, 20), (9, 21), (10, 6), (10, 7), (10, 8),
				   (10, 13), (10, 14), (10, 15), (10, 16), (10, 17), (10, 18), (10, 19), (10, 20), (10, 21), (11, 6),
				   (11, 7), (11, 8), (11, 9), (11, 13), (11, 14), (11, 15), (11, 18), (11, 19), (12, 6), (12, 7),
				   (12, 13), (12, 14), (12, 15), (12, 18), (12, 19), (12, 20)]

current_polygon.sort()
print(current_polygon)
test = svgwrite.Drawing('test2.svg', profile='tiny')

edge_x = current_polygon[0][0]
edge_y = current_polygon[0][1]

string = "M%d,%d " % (edge_y, edge_x)
ordered_edges = [(edge_y, edge_x)]

is_edge = False
priority = DirectionPriority.down.value
while True:
	if is_start(edge_x, edge_y, ordered_edges) and len(ordered_edges) > 1:
		break
	if priority == DirectionPriority.down.value:
		if get_down_side(edge_x, edge_y, current_polygon):
			edge_x += 1
			priority = DirectionPriority.left.value
		elif get_left_side(edge_x, edge_y, current_polygon):
			edge_y -= 1
			priority = DirectionPriority.left.value
		elif try_right_side(edge_x, edge_y, current_polygon):
			edge_x, edge_y = move_to_right(edge_x, edge_y, current_polygon)
			is_edge = True
		elif get_right_side(edge_x, edge_y, current_polygon):
			edge_y += 1
			is_edge = True
			priority = DirectionPriority.right.value
		elif get_upper_side(edge_x, edge_y, current_polygon):
			edge_x -= 1
			priority = DirectionPriority.right.value
		if is_edge:
			if (edge_y, edge_x) not in ordered_edges:
				ordered_edges.append((edge_y, edge_x))
			is_edge = False
	if priority == DirectionPriority.left.value:
		if get_left_side(edge_x, edge_y, current_polygon):
			edge_y -= 1
			if get_upper_side(edge_x, edge_y, current_polygon) and not get_upper_side(edge_x, edge_y + 1,
																					  current_polygon):
				edge_x -= 1
				priority = DirectionPriority.up.value
				is_edge = True
		else:
			is_edge = True
			priority = DirectionPriority.down.value
		if is_edge:
			if (edge_y, edge_x) not in ordered_edges:
				ordered_edges.append((edge_y, edge_x))
			is_edge = False
	if priority == DirectionPriority.up.value:
		if get_upper_side(edge_x, edge_y, current_polygon):
			edge_x -= 1
			priority = DirectionPriority.right.value
		elif get_right_side(edge_x, edge_y, current_polygon):
			edge_y += 1
			priority = DirectionPriority.right.value
		elif try_left_side(edge_x, edge_y, current_polygon):
			edge_x, edge_y = move_to_left(edge_x, edge_y, current_polygon)
			is_edge = True
		elif get_left_side(edge_x, edge_y, current_polygon):
			edge_y -= 1
			is_edge = True
			priority = DirectionPriority.left.value
		else:
			priority = DirectionPriority.down.value
		if is_edge:
			if (edge_y, edge_x) not in ordered_edges:
				ordered_edges.append((edge_y, edge_x))
			is_edge = False
	if priority == DirectionPriority.right.value:
		if get_right_side(edge_x, edge_y, current_polygon):
			edge_y += 1
			if get_down_side(edge_x, edge_y, current_polygon) and not get_down_side(edge_x, edge_y - 1,
																					current_polygon):
				edge_x += 1
				priority = DirectionPriority.down.value
				is_edge = True
		else:
			is_edge = True
			priority = DirectionPriority.up.value
		if is_edge:
			if (edge_y, edge_x) not in ordered_edges:
				ordered_edges.append((edge_y, edge_x))
			is_edge = False

print(ordered_edges)
x_prev = edge_x
y_prev = edge_y
for y, x in ordered_edges:
	if y != edge_y and x != edge_x:
		if x == x_prev and y_prev > y and y_prev - y > 1:
			x1 = x + 1
			y1 = int((y_prev + y) / 2)
			string += "Q%d,%d %d,%d " % (y1, x1, y, x)
		if x == x_prev and y > y_prev and y - y_prev > 1:
			x1 = x - 1
			y1 = int((y_prev + y) / 2)
			string += "Q%d,%d %d,%d " % (y1, x1, y, x)
		else:
			string += "L%d,%d " % (y, x)
		x_prev = x
		y_prev = y
string += "Z"
print(string)
test_path = test.path(d=string,
					  stroke="#000",
					  fill=svgwrite.rgb(59, 18, 133),
					  stroke_width=0.1)
test.add(test_path)
test.save()
