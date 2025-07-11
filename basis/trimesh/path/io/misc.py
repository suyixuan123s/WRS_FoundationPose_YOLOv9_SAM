import numpy as np
from ..entities import Line, Arc
from ...util import is_shape
from ...geometry import faces_to_edges
from ...grouping import group_rows
from collections import deque


def dict_to_path(drawing_obj):
    '''
    将字典对象转换为路径对象

    :param drawing_obj: 包含 'entities' 和 'vertices' 的字典对象
    :return: 包含实体和顶点的字典
    '''
    loaders = {'Arc': Arc, 'Line': Line}
    vertices = np.array(drawing_obj['vertices'])
    entities = [None] * len(drawing_obj['entities'])
    for entity_index, entity in enumerate(drawing_obj['entities']):
        entities[entity_index] = loaders[entity['type']](points=entity['points'],
                                                         closed=entity['closed'])
    return {'entities': entities,
            'vertices': vertices}


def lines_to_path(lines):
    '''
    将一组线段转换为路径对象

    :param lines: 线段数组,形状为 (n, 2, [2|3])
    :return: 包含实体和顶点的字典
    '''
    lines = np.asanyarray(lines)
    if is_shape(lines, (-1, (2, 3))):
        result = {'entities': np.array([Line(np.arange(len(lines)))]),
                  'vertices': lines}
        return result
    elif is_shape(lines, (-1, 2, (2, 3))):
        entities = [Line([i, i + 1]) for i in range(0, (lines.shape[0] * 2) - 1, 2)]
        vertices = lines.reshape((-1, lines.shape[2]))
        result = {'entities': entities,
                  'vertices': vertices}
    else:
        raise ValueError('线段必须是 (n,(2|3)) or (n,2,(2|3))')
    return result


def polygon_to_path(polygon):
    '''
    将 shapely.geometry.Polygon 转换为一组 (n,2,2) 线段

    :param polygon: shapely.geometry.Polygon 对象
    :return: 包含实体和顶点的字典
    '''

    def append_boundary(boundary):
        entities.append(Line(np.arange(len(boundary.coords)) + len(vertices)))
        vertices.extend(boundary.coords)

    entities = deque()
    vertices = deque()

    append_boundary(polygon.exterior)
    for interior in polygon.interiors:
        append_boundary(interior)

    return {'entities': np.array(entities),
            'vertices': np.array(vertices)}


def faces_to_path(mesh, face_ids=None):
    '''
    从给定的网格和面索引中找到轮廓边缘,并将其转换为 Path3D

    :param mesh: Trimesh 对象
    :param face_ids: (n) mesh.faces 的索引列表
    :return: 包含实体和顶点的字典
    '''
    if face_ids is None:
        faces = mesh.faces
    else:
        faces = mesh.faces[face_ids]
    edges = np.sort(faces_to_edges(faces), axis=1)
    unique_edges = group_rows(edges, require_count=1)
    segments = mesh.vertices[edges[unique_edges]]
    return lines_to_path(segments)
