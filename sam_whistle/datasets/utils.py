from shapely.geometry import LineString
import numpy as np
import networkx as nx
import cv2

def simplify_path(pts, tolerance=0.5):
    """Simplify the sequence of points between nodes using Shapely."""
    line = LineString(pts)
    simplified_line = line.simplify(tolerance)
    return np.array(simplified_line.coords)

def simplify_graph(graph, tolerance=0.5, multi_edge=False, return_keys = True):
    """Simplify the graph by removing intermediate points between nodes."""
    simplified_graph = nx.Graph()
    pre_id = -1
    cur_id = 0
    key_nodes_ids = []
    for (s,e) in graph.edges():
        if multi_edge:
            for i in range(len(graph[s][e])):
                ps = graph[s][e][i]['pts']
                ps  = simplify_path(ps, tolerance=tolerance)

                for j, coord in enumerate(ps):
                    if j == 0 or j == len(ps) - 1:
                        key_nodes_ids.append(cur_id)
                    simplified_graph.add_node(cur_id, pos=coord)
                    if j > 0:
                        simplified_graph.add_edge(pre_id, cur_id)
                    pre_id = cur_id
                    cur_id += 1
        else:
            ps = graph[s][e]['pts']
            ps  = simplify_path(ps, tolerance=tolerance)

            for j, coord in enumerate(ps):
                if j == 0 or j == len(ps) - 1:
                    key_nodes_ids.append(cur_id)
                simplified_graph.add_node(cur_id, pos=coord)
                if j > 0:
                    simplified_graph.add_edge(pre_id, cur_id)
                pre_id = cur_id
                cur_id += 1
    
    if return_keys:
        simplified_graph.graph['key_nodes'] = key_nodes_ids
    return simplified_graph

def graph_to_mask(graph, img_shape, width=3):
    assert len(img_shape) == 2, 'img_shape must be 2D'
    mask = np.zeros(img_shape, dtype=np.uint8)
    positions = nx.get_node_attributes(graph, 'pos')
    for edge in graph.edges():
        pt1 = tuple(map(int, positions[edge[0]]))
        pt2 = tuple(map(int, positions[edge[1]]))
        cv2.line(mask, pt1[::-1], pt2[::-1], 1, width)  # need coord be (col, row)
    return mask

def graph_to_keymask(graph, img_shape, radius=3):
    assert len(img_shape) == 2, 'img_shape must be 2D'
    mask = np.zeros(img_shape, dtype=np.uint8)
    key_nodes = graph.graph['key_nodes']
    positions = nx.get_node_attributes(graph, 'pos')
    for node in key_nodes:
        pt = tuple(map(int, positions[node]))
        cv2.circle(mask, pt[::-1], radius, 1, -1)
    return mask


def sknw_graph_2_key_mask(graph, img_shape, radius=3):
    """Convert sknw graph nodes to keypoints on a mask."""
    assert len(img_shape) == 2, 'img_shape must be 2D'
    mask = np.zeros(img_shape, dtype=np.uint8)
    nodes = graph.nodes()
    keypoints = np.array([nodes[i]['o'] for i in nodes])
    for kp in keypoints:
        pt = tuple(map(int, kp))
        cv2.circle(mask, pt[::-1], radius, 1, -1)

    return mask