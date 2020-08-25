from enum import Enum


class DirectionPriority(Enum):
    left = 0
    up = 1
    right = 2
    down = 3


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


def is_start(x, y, edges):
    if edges[0][1] == x and edges[0][0] == y:
        return True
    else:
        return False


def start_vectorization(current_polygon):
    current_polygon.sort()
    edge_x = current_polygon[0][0]
    edge_y = current_polygon[0][1]
    ordered_edges = [(edge_y, edge_x)]
    is_edge = False
    priority = DirectionPriority.down.value
    while True:
        if len(current_polygon) < 2:
            break
        if is_start(edge_x, edge_y, ordered_edges) and len(ordered_edges) > 1:
            break
        if priority == DirectionPriority.down.value:
            if get_down_side(edge_x, edge_y, current_polygon):
                edge_x += 1
                is_edge = True
                if get_left_side(edge_x, edge_y, current_polygon):
                    priority = DirectionPriority.left.value
            elif get_right_side(edge_x, edge_y, current_polygon):
                priority = DirectionPriority.right.value
            elif get_upper_side(edge_x, edge_y, current_polygon):
                priority = DirectionPriority.up.value
            if is_edge:
                ordered_edges.append((edge_y, edge_x))
                is_edge = False
        if priority == DirectionPriority.left.value:
            if get_left_side(edge_x, edge_y, current_polygon):
                edge_y -= 1
                is_edge = True
                if get_upper_side(edge_x, edge_y, current_polygon):
                    priority = DirectionPriority.up.value
            elif get_down_side(edge_x, edge_y, current_polygon):
                priority = DirectionPriority.down.value
            elif get_right_side(edge_x, edge_y, current_polygon):
                priority = DirectionPriority.right.value
            if is_edge:
                ordered_edges.append((edge_y, edge_x))
                is_edge = False
        if priority == DirectionPriority.up.value:
            if get_upper_side(edge_x, edge_y, current_polygon):
                edge_x -= 1
                is_edge = True
                if get_right_side(edge_x, edge_y, current_polygon):
                    priority = DirectionPriority.right.value
            elif get_left_side(edge_x, edge_y, current_polygon):
                priority = DirectionPriority.left.value
            elif get_down_side(edge_x, edge_y, current_polygon):
                priority = DirectionPriority.down.value
            if is_edge:
                ordered_edges.append((edge_y, edge_x))
                is_edge = False
        if priority == DirectionPriority.right.value:
            if get_right_side(edge_x, edge_y, current_polygon):
                edge_y += 1
                is_edge = True
                if get_down_side(edge_x, edge_y, current_polygon):
                    priority = DirectionPriority.down.value
            elif get_upper_side(edge_x, edge_y, current_polygon):
                priority = DirectionPriority.up.value
            elif get_left_side(edge_x, edge_y, current_polygon):
                priority = DirectionPriority.left.value
            else:
                is_edge = True
                priority = DirectionPriority.up.value
            if is_edge:
                ordered_edges.append((edge_y, edge_x))
                is_edge = False

    return ordered_edges
