import numpy as np
import gimpact as gi
from panda3d.bullet import BulletRigidBodyNode, BulletPlaneShape
from panda3d.core import Vec3


def gen_cdmesh_vvnf(vertices, vertex_normals, faces):
    """
    生成一个 cdmesh 给定顶点、顶点法线和面

    :param vertices: 顶点列表
    :param vertex_normals: 顶点法线列表
    :param faces: 面列表
    :return: gimpact.TriMesh (需要安装 gimpact)

    author: weiwei
    date: 20210118
    """
    return gi.TriMesh(vertices, faces.flatten())


def is_collided(objcm0, objcm1):
    """
    检查两个 objcm 在转换为指定的 cdmesh 类型后是否发生碰撞

    :param objcm0: 第一个对象
    :param objcm1: 第二个对象
    :return: (是否碰撞, 碰撞点列表)

    author: weiwei
    date: 20210117
    """
    obj0 = gen_cdmesh_vvnf(*objcm0.extract_rotated_vvnf())
    obj1 = gen_cdmesh_vvnf(*objcm1.extract_rotated_vvnf())
    contacts = gi.trimesh_trimesh_collision(obj0, obj1)
    contact_points = [ct.point for ct in contacts]
    return (True, contact_points) if len(contact_points) > 0 else (False, contact_points)


def gen_plane_cdmesh(updirection=np.array([0, 0, 1]), offset=0, name='autogen'):
    """
    生成一个平面 bulletrigidbody 状态

    :param updirection: bulletplaneshape 在 panda3d 中的法线参数
    :param offset: bulletplaneshape 在 panda3d 中的 d 参数
    :param name: 节点名称
    :return: bulletrigidbody

    author: weiwei
    date: 20170202, tsukuba
    """
    bulletplnode = BulletRigidBodyNode(name)
    bulletplshape = BulletPlaneShape(Vec3(updirection[0], updirection[1], updirection[2]), offset)
    bulletplshape.setMargin(0)
    bulletplnode.addShape(bulletplshape)
    return bulletplnode


if __name__ == '__main__':
    import os, math, basis
    import numpy as np
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import modeling.collision_model as cm
    import basis.robot_math as rm

    # base = wd.World(cam_pos=[1.0, 1, .0, 1.0], lookat_pos=[0, 0, 0])
    # objpath = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    # objcm1 = cm.CollisionModel(objpath)
    #
    # homomat = np.eye(4)
    # homomat[:3, :3] = rm.rotmat_from_axangle([0, 0, 1], math.pi / 2)
    # homomat[:3, 3] = np.array([0.02, 0.02, 0])
    # objcm1.set_homomat(homomat)
    # objcm1.set_rgba([1, 1, .3, .2])
    #
    # objcm2 = objcm1.copy()
    # objcm2.set_pos(objcm1.get_pos() + np.array([.05, .02, .0]))
    # objcm1.change_cdmesh_type('convex_hull')
    # objcm2.change_cdmesh_type('obb')
    # iscollided, contacts = is_collided(objcm1, objcm2)
    # # objcm1.show_cdmesh(type='box')
    # # show_triangles_cdmesh(objcm1)
    # # show_triangles_cdmesh(objcm2)
    # objcm1.show_cdmesh()
    # objcm2.show_cdmesh()
    # # objcm1.show_cdmesh(type='box')
    # # objcm2.show_cdmesh(type='triangles')
    # objcm1.attach_to(base)
    # objcm2.attach_to(base)
    # print(iscollided)
    #
    # for ct in contacts:
    #     gm.gen_sphere(ct, radius=.01).attach_to(base)
    # # pfrom = np.array([0, 0, 0]) + np.array([1.0, 1.0, 1.0])
    # # pto = np.array([0, 0, 0]) + np.array([-1.0, -1.0, -0.9])
    # # hitpos, hitnrml = rayhit_triangles_closet(pfrom=pfrom, pto=pto, objcm=objcm)
    # # objcm.attach_to(base)
    # # objcm.show_cdmesh(type='box')
    # # objcm.show_cdmesh(type='convex_hull')
    # # gm.gen_sphere(hitpos, radius=.003, rgba=np.array([0, 1, 1, 1])).attach_to(base)
    # # gm.gen_stick(spos=pfrom, epos=pto, thickness=.002).attach_to(base)
    # # gm.gen_arrow(spos=hitpos, epos=hitpos + hitnrml * .07, thickness=.002, rgba=np.array([0, 1, 0, 1])).attach_to(base)
    # base.run()



    base = wd.World(cam_pos=[1.0, 1, .0, 1.0], lookat_pos=[0, 0, 0])
    objpath = os.path.join(basis.__path__[0], 'objects', 'yumifinger.stl')
    objcm1 = cm.CollisionModel(objpath, cdmesh_type='triangles')

    homomat = np.array([[5.00000060e-01, 7.00629234e-01, 5.09036899e-01, -3.43725011e-02],
                        [8.66025329e-01, -4.04508471e-01, -2.93892622e-01, 5.41121606e-03],
                        [-2.98023224e-08, 5.87785244e-01, -8.09016943e-01, 1.13636881e-01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # homomat = np.array([[1.00000000e+00, 2.38935501e-16, 3.78436685e-17, -7.49999983e-03],
    #                     [2.38935501e-16, -9.51056600e-01, -3.09017003e-01, 2.04893537e-02],
    #                     [-3.78436685e-17, 3.09017003e-01, -9.51056600e-01, 1.22025304e-01],
    #                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    objcm1.set_homomat(homomat)
    objcm1.set_rgba([1, 1, .3, .2])

    objpath = os.path.join(basis.__path__[0], 'objects', 'tubebig.stl')
    objcm2 = cm.CollisionModel(objpath, cdmesh_type='triangles')
    iscollided, contact_points = is_collided(objcm1, objcm2)

    # objcm1.show_cdmesh(type='box')
    # show_triangles_cdmesh(objcm1)
    # show_triangles_cdmesh(objcm2)

    objcm1.show_cdmesh()
    objcm2.show_cdmesh()

    # objcm1.show_cdmesh(type='box')
    # objcm2.show_cdmesh(type='triangles')
    objcm1.attach_to(base)
    objcm2.attach_to(base)
    print(iscollided)

    for ctpt in contact_points:
        gm.gen_sphere(ctpt, radius=.01).attach_to(base)

    # pfrom = np.array([0, 0, 0]) + np.array([1.0, 1.0, 1.0])
    # pto = np.array([0, 0, 0]) + np.array([-1.0, -1.0, -0.9])
    # hitpos, hitnrml = rayhit_triangles_closet(pfrom=pfrom, pto=pto, objcm=objcm)
    # objcm.attach_to(base)
    # objcm.show_cdmesh(type='box')
    # objcm.show_cdmesh(type='convex_hull')
    # gm.gen_sphere(hitpos, radius=.003, rgba=np.array([0, 1, 1, 1])).attach_to(base)
    # gm.gen_stick(spos=pfrom, epos=pto, thickness=.002).attach_to(base)
    # gm.gen_arrow(spos=hitpos, epos=hitpos + hitnrml * .07, thickness=.002, rgba=np.array([0, 1, 0, 1])).attach_to(base)
    base.run()
