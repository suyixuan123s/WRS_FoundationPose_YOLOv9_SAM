import copy
import math
import visualization.panda.world as wd
import modeling.collision_model as cm
import humath
import robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper as yg
import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as hnde
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode
import numpy as np
import basis.robot_math as rm
import modeling.geometric_model as gm
import os
import pickle
import basis.data_adapter as da

import basis.trimesh as trimeshWan
from direct.task.TaskManagerGlobal import taskMgr

from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from descartes import PolygonPatch
import shapely
from shapely.geometry import MultiPoint
from shapely.geometry import Point


# 全部在z正半轴: 
def placeGround(obj, displacement_homo_list):
    for homo in displacement_homo_list:
        # print("init homo ", homo)
        # obj_copy = copy.deepcopy(obj)
        obj_copy = obj.copy()
        z_max = 0
        for vertice in obj_copy.objtrm.vertices:

            # print("检测顶点", vertice[2])
            wd_vertice = [vertice[0], vertice[1], vertice[2], 1]
            # print("pos", wd_vertice)
            wd_vertice = np.dot(homo, wd_vertice)
            # print("wd_pos", wd_vertice)
            # gm.gen_sphere(pos=np.array(wd_vertice[:3]),radius=0.0005).attach_to(base)

            # if wd_vertice[2]<-6.57870183e-19:
            distance = -wd_vertice[2]
            # print("距离？", distance)
            if distance >= z_max:
                z_max = distance
                # print("最远点的z？", z_max)

        homo[2, 3] = homo[2, 3] + z_max  # i=3时  设为0.05
    return displacement_homo_list


def getFacetsCenter(obj_trimesh, facets):
    '''
    get the coordinate of large facet center in self.largefacetscenter
    get the normal vecter of large facet center in self.largefacet_normals
    get the vertices coordinate of large facet in self.largefacet_vertices
    :return:
    '''
    vertices = obj_trimesh.vertices
    faces = obj_trimesh.faces
    facets = facets
    smallfacesarea = obj_trimesh.area_faces
    smallface_normals = obj_trimesh.face_normals
    smallfacecenter = []
    smallfacectlist = []
    smallfacectlist_area = []
    largefacet_normals = []
    largefacet_vertices = []
    largefacetscenter = []
    for i, smallface in enumerate(faces):
        smallfacecenter.append(humath.centerPoint(np.array([vertices[smallface[0]],
                                                                 vertices[smallface[1]],
                                                                 vertices[smallface[2]]])))
    for facet in facets:
        b = []
        b_area = []
        temlargefaceVerticesid = []
        temlargefaceVertices = []
        for face in facet:
            b.append(smallfacecenter[face])
            b_area.append(smallfacesarea[face])
            temlargefaceVerticesid.extend(faces[face])
            # print("temlargefaceVerticesid", temlargefaceVerticesid)
        smallfacectlist.append(b)
        smallfacectlist_area.append(b_area)
        smallfacenomallist = [smallface_normals[facet[j]] for j in range(len(facet))]
        largefacet_normals.append(np.average(smallfacenomallist, axis=0))
        # self.largefacet_normals.append(self.smallface_normals[facet[0]]) #TODO an average normal
        temlargefaceVerticesid = list(set(temlargefaceVerticesid))  # remove repeating vertices ID
        for id in temlargefaceVerticesid:
            temlargefaceVertices.append(vertices[id])
        largefacet_vertices.append(temlargefaceVertices)
    for i, largeface in enumerate(smallfacectlist):
        largefacetscenter.append(humath.centerPointwithArea(largeface, smallfacectlist_area[i]))
    return largefacetscenter, largefacet_normals, largefacet_vertices

if __name__ == '__main__':
    base = wd.World(cam_pos=[0.501557, 0.137317, 0.48133], w=960,
                    h=540, lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    this_dir, this_filename = os.path.split(__file__)
    obj = cm.CollisionModel(r'E:\ABB-Project\ABB_wrs\0000_placementplanner\objects\test_long_small.stl')
    obj.set_rgba([1, 1, 0, 0.5])
    # obj.attach_to(base)
    obj_ch= obj.objtrm.convex_hull
    vertices = obj_ch.vertices
    faces = obj_ch.faces
    # print("faces: ",faces)   ##3个顶点序号组成的面
    com = obj.objtrm.center_mass

    # base.run()
    convex_obj = trimeshWan.Trimesh(vertices=vertices, faces=faces)
    convex_obj_gm = gm.GeometricModel(convex_obj)
    convex_obj_gm.set_rgba([1, 1, 0, 1])
    # convex_obj_gm.attach_to(base)
    facets, facetnormals, facetcurvatures = convex_obj.facets_noover(faceangle=0.99)  ##.99 可更改
    # print("facets:",facets) ## 同一个面的面序号
    # for id, item in enumerate(facets_center):
    #     gm.gen_sphere(item, radius=0.005).attach_to(base)
    #     gm.gen_arrow(item, item+facet_normal[id]*0.05).attach_to(base)

    facets_center, facet_normal, facet_vertices = getFacetsCenter(convex_obj, facets)  ##加权思想求4/5顶点构成的面 的 坐标
    pos_list = [np.array([0, 0, 0]) + center for center in facets_center]   ## 随着物体走
    rotmat_list = [rm.rotmat_between_vectors(normal, np.array([0, 0, 1])) for normal in facet_normal]
    ##以上实现   所有面放置地上
    # for id, item in enumerate(facets_center):
    #     gm.gen_sphere(item, radius=0.005).attach_to(base)
    #     gm.gen_arrow(item, item+facet_normal[id]*0.05).attach_to(base)
    # base.run()
    facet_project_list = []
    com_project_list = []
    rgba_list = []
    stable_ids=[]
    for id, facet in enumerate(facet_vertices):   ##facet: 是 顶点坐标 构成的  面  .num(id)=7   基于原点的坐标
        # print("see facet",facet)
        facet_project = [rotmat_list[id].dot(vertex)[:2] for vertex in facet]   ## 物体发生旋转之后的 顶点坐标  在投影
        facet_project_list.append(facet_project)
        com_project_list.append(rotmat_list[id].dot(com)[:2])    ## com点也要投影

    for id, facet_project in enumerate(facet_project_list):   ##每个面  发生旋转之后的顶点坐标【x,y】
        # print("facet_project",facet_project)
        contact_polygon = Polygon(facet_project)   ## polygon构建多边形  储存的是多边形的坐标
        # print("contact_polygon",contact_polygon)
        convex_hull_polygon=contact_polygon.convex_hull   ##  利用坐标  构建凸包多边形！！
        print(convex_hull_polygon)
        #可视化的过程  监测
        contact_patch = PolygonPatch(convex_hull_polygon, fc='yellow', ec='black', alpha=0.5)   ##可视化多边形的接触块.
        ##alpha 是透明度.   ec是边框颜色
        fig = plt.figure(figsize=(5, 5), dpi=100)
        plt.axis('on')
        # plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_ylim(-0.2, 0.2)
        ax1.set_xlim(-0.2, 0.2)
        # ring_patch2 = PolygonPatch(contact_patch, color="yellow", alpha=0.5)
        ax1.add_patch(contact_patch)   ##将凸包多边形 增加到图中

        if shapely.within(Point(com_project_list[id][0], com_project_list[id][1]), convex_hull_polygon):
            ax1.scatter(x=com_project_list[id][0], y=com_project_list[id][1], color="green")
            rgba_list.append([0,1,0,1])    ##满足 稳定关系  com设为green
            # stable_ids.append(id)
        else:
            ax1.scatter(x=com_project_list[id][0], y=com_project_list[id][1], color="red")
            rgba_list.append([1, 0, 0, 1])
    plt.show()
    # print("稳定序号: ",stable_ids)
    # base.run()
    ##检验稳定


    # print("see old pos_list",pos_list)
    # pos_list=[pos_list[i] for i in stable_ids]
    # print("see new pos_list",pos_list)
    # base.run()
    homo_list = [rm.homomat_from_posrot(pos_list[i], rotmat_list[i]) for i in range(len(pos_list))]
    # print("")
    homo_inv_list = [np.linalg.inv(homo) for homo in homo_list]

    displacement_list = list(np.linspace([0, 0, 0], [1.2, 0, 0], len(homo_list)))
    displacement_homo_list = [rm.homomat_from_posrot(displacement_list[i]).dot(homo_inv_list[i]) for i in range(len(displacement_list))]

    ##全部在z正半轴: 
    new_displacement_homo_list=placeGround(obj,displacement_homo_list)


    for id,homo in enumerate(new_displacement_homo_list):
        obj_copy = copy.deepcopy(obj)
        obj_copy.set_homomat(homo)
        obj_copy.set_rgba(rgba_list[id])
        obj_copy.attach_to(base)

    # base.run()

    def update(textNode, count, task):

        if textNode[0] is not None:
            textNode[0].detachNode()
            textNode[1].detachNode()
            textNode[2].detachNode()
        cam_pos = base.cam.getPos()
        textNode[0] = OnscreenText(
            text=str(cam_pos[0])[0:5],
            fg=(1, 0, 0, 1),
            pos=(1.0, 0.8),
            align=TextNode.ALeft)
        textNode[1] = OnscreenText(
            text=str(cam_pos[1])[0:5],
            fg=(0, 1, 0, 1),
            pos=(1.3, 0.8),
            align=TextNode.ALeft)
        textNode[2] = OnscreenText(
            text=str(cam_pos[2])[0:5],
            fg=(0, 0, 1, 1),
            pos=(1.6, 0.8),
            align=TextNode.ALeft)
        return task.again


    cam_view_text = OnscreenText(
        text="Camera View: ",
        fg=(0, 0, 0, 1),
        pos=(1.15, 0.9),
        align=TextNode.ALeft)
    testNode = [None, None, None]
    count = [0]
    taskMgr.doMethodLater(0.01, update, "addobject", extraArgs=[testNode, count],
                          appendTask=True)

    base.run()
