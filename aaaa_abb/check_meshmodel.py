import open3d
from direct.task.TaskManagerGlobal import taskMgr
import visualization.panda.world as wd
import modeling.collision_model as cm
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode
import modeling.geometric_model as gm
import os
import open3d as o3d
import vision.depth_camera.pcd_data_adapter as vdda

if __name__ == '__main__':
    base = wd.World(cam_pos=[2.01557, 0.637317, 1.88133], w=960, h=540, lookat_pos=[0, 0, 0])

    # this_dir, this_filename = os.path.split(__file__)
    # obj = cm.CollisionModel('rack_10ml_green.STL')
    # obj.set_rgba([1, 1, 0, 0.5])
    # obj.attach_to(base)
    # gm.gen_frame().attach_to(base)
    # base.run()

    Mesh = o3d.io.read_triangle_mesh("rack_5ml_new.STL")
    Mesh.paint_uniform_color([1, 0, 0])
    Mesh.compute_vertex_normals()
    pcd1 = Mesh.sample_points_poisson_disk(number_of_points=5000)
    pcd1, ind = pcd1.remove_radius_outlier(nb_points=15, radius=5)
    pcd1_np = vdda.o3dpcd_to_parray(pcd1)
    center1 = pcd1_np.mean(axis=0)
    gm.gen_sphere(center1, radius=0.01).attach_to(base)
    gm.gen_pointcloud(pcd1_np, pntsize=2).attach_to(base)
    gm.gen_frame().attach_to(base)
    # base.run()


    # 屏幕上显示相机的位置信息,并定期更新这些信息
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
