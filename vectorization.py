import svgwrite
import sys
from enum import Enum


class CheckingDepth(Enum):
	first_stage = 1
	larger_stage = 2


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


def check_upside(x, y, polygon, edges, stage, start):
	curr_x = x - 1
	curr_y = y
	if (curr_x, curr_y) in polygon:
		if stage == CheckingDepth.larger_stage.value:
			index = 1
			while True:
				if (curr_x, curr_y + index) in polygon:
					index += 1
				else:
					curr_y = curr_y + index - 1
					break
		edges.append((curr_x, curr_y))
		check_upside(curr_x, curr_y, polygon, edges, CheckingDepth.larger_stage.value, start)
	else:
		if stage == CheckingDepth.larger_stage.value:
			index = 1
			is_end = False
			while True:
				if (x, y - index) in polygon:
					if (curr_x, y - index) in polygon:
						curr_y = y - index
						edges.append((curr_x, curr_y))
						check_upside(curr_x, curr_y, polygon, edges, CheckingDepth.larger_stage.value, start)
						break
					else:
						index += 1
				else:
					is_end = True
					break
			if is_end:
				down_x = x + 1
				down_y = curr_y
				# while (_x, down_y) in polygon:
				# 	down_y -= 1
				# if down_y != curr_y:
				# 	edges += "L%d,%d " % (curr_x, down_y)
				while down_x < start:
					if (down_x, down_y) in polygon:
						while (down_x, down_y - 1) in polygon:
							down_y -= 1
						edges.append((down_x, down_y))
						down_x += 1
					else:
						while True:
							if (down_x - 1, down_y + 1) in polygon:
								if (down_x, down_y + 1) in polygon:
									down_y += 1
									edges.append((down_x, down_y))
									down_x += 1
									break
								else:
									down_y += 1
							else:
								break


def get_left_start(x, y, polygon, edges):
	index = 1
	while True:
		if (x, y - index) in polygon:
			check_upside(x, y - index, polygon, edges, CheckingDepth.first_stage.value, x)
			index += 1
		else:
			return x, y - index + 1


def get_right_start(x, y, polygon):
	index = 1
	while True:
		if (x, y + index) in polygon:
			if (x + 1, y + index) in polygon:
				return x + 1, y + index
			else:
				index += 1
		else:
			return None


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
x_start = current_polygon[0][0]
y_start = current_polygon[0][1]
string = "M%d,%d " % (x_start, y_start)
ordered_edges = [(x_start, y_start)]

while True:
	edge_x = x_start + 1
	edge_y = y_start
	if (edge_x, edge_y) in current_polygon:
		edge_x, edge_y = get_left_start(edge_x, edge_y, current_polygon, ordered_edges)
		ordered_edges.append((edge_x, edge_y))
		x_start = edge_x
		y_start = edge_y
	else:
		edge_x, edge_y = get_right_start(x_start, y_start, current_polygon)
		ordered_edges.append((edge_x, edge_y))
		x_start = edge_x
		y_start = edge_y

# starts = [(x_start, y_start)]
# ends = []
# position = 1
# while position < len(current_polygon):
#     current_pix = current_polygon[position]
#     prev_pix = current_polygon[position - 1]
#     if current_pix[0] != x_start or (current_pix[1] != prev_pix[1] + 1 and current_pix[0] == x_start):
#         x_start = current_pix[0]
#         y_start = current_pix[1]
#         if position < len(current_polygon) - 1 and current_polygon[position + 1][0] == x_start:
#             starts.append((x_start, y_start))
#         ends.append((prev_pix[0], prev_pix[1]))
#     position += 1
#     if position >= len(current_polygon):
#         if (current_pix[0], current_pix[1]) not in ends:
#             ends.append((current_pix[0], current_pix[1]))
# outer_remover(starts)
# outer_remover(ends)
# for x, y in edges:
# 	edges += "L%d,%d " % (x, y)


string += "Z"
print(string)
test_path = test.path(d=string,
					  stroke="#000",
					  fill=svgwrite.rgb(59, 18, 133),
					  stroke_width=0.1)
test.add(test_path)
test.save()
