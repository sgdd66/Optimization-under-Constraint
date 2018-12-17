# ***************************************************************************
# Copyright (c) 2018 西安交通大学
# All rights reserved
# 
# 文件名称：Mayavi_test.py
# 
# 摘    要：数据可视化mayavi库测试文件
# 
# 创 建 者：上官栋栋
# 
# 创建日期：2018年11月9日
#
# 修改记录
# 日期  修改者   		版本     修改内容
# ------------- 		-------  ------------------------  
# ***************************************************************************

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tvtk.tools import tvtk_doc,ivtk
from tvtk.api import tvtk
import numpy as np
from mayavi import mlab
from traits.api import HasTraits ,Delegate,Instance,Int,Str



class DataVisual(object):
    def test1(self):
        '''调用帮助文档T'''
        tvtk_doc.main()

    def test2(self):
        '''展开交互图像窗口的常用流程'''
        s=tvtk.CubeSource(x_length=1.0,y_length=2.0,z_length=3.0)
        m=tvtk.PolyDataMapper(input_connection=s.output_port)
        a=tvtk.Actor(mapper=m)
        r=tvtk.Renderer(background=(0.5,0.5,0.5))
        r.add_actor(a)
        w=tvtk.RenderWindow(size=(500,500))
        w.add_renderer(r)
        i=tvtk.RenderWindowInteractor(render_window=w)
        i.initialize()
        i.start()

    def test3(self):
        plot3d=tvtk.MultiBlockPLOT3DReader(xyz_file_name="combxyz.bin",
                                           q_file_name="combq,bin",
                                           scalar_function_number=100,
                                           vector_function_number=200)
        plot3d.update()
        grid=plot3d.output.get_block(0)

        con=tvtk.ContourFilter()
        con.set_input_data(grid)
        con.generate_values(10,grid.point_data.scalars.range)

        m=tvtk.PolyDataMapper(scalar_range=grid.point_data.scalars.range,
                              input_connection=con.output_port)
        a=tvtk.Actor(mapper=m)
        a.property.opacity=0.5

        win=self.ivtk_scene(a)
        win.scene.isometric_view()
        self.event_loop()

    # 命名为tvtkfunc.py
    def ivtk_scene(self,actors):
        from tvtk.tools import ivtk
        # 创建一个带Crust（Python Shell）的窗口
        win = ivtk.IVTKWithCrustAndBrowser()
        win.open()
        win.scene.add_actor(actors)
        # 修正窗口错误
        dialog = win.control.centralWidget().widget(0).widget(0)
        from pyface.qt import QtCore
        dialog.setWindowFlags(QtCore.Qt.WindowFlags(0x00000000))
        dialog.show()
        return win

    def event_loop(self):
        from pyface.api import GUI
        gui = GUI()
        gui.start_event_loop()

    def test4(self):
        s=tvtk.CubeSource(x_length=1.0,y_length=2.0,z_length=3.0)
        m=tvtk.PolyDataMapper(input_connection=s.output_port)
        a=tvtk.Actor(mapper=m)

        gui=GUI()
        # win=ivtk.IVTKWithCrust()
        # win=ivtk.IVTK()
        win=ivtk.IVTKWithCrustAndBrowser()
        win.open()
        win.scene.add_actor(a)

        dialog=win.control.centralWidget().widget(0).widget(0)
        from pyface.qt import QtCore
        dialog.setWindowFlags(QtCore.Qt.WindowFlags(0x00000000))
        dialog.show()

        gui.start_event_loop()

    # imageData数据类型 and RectilinearGrid数据类型
    def test5(self):
        # imageData数据类型
        img=tvtk.ImageData(spacing=(1,1,1),origin=(0,0,0),dimensions=(3,4,5))

        #RectilinearGrid数据类型
        x=np.array([0,3,5,6])
        y=np.array([3,5,9,12])
        z=np.array([4,5,7,9])
        r=tvtk.RectilinearGrid()
        r.x_coordinates=x
        r.y_coordinates=y
        r.z_coordinates=z
        r.dimensions=len(x),len(y),len(z)

        for i in range(64):
            print(r.get_point(i))

    #StructureGrid数据类
    def test6(self):

        def generate_annulus(r, theta, z):
            """ Generate points for structured grid for a cylindrical annular
                volume.  This method is useful for generating a unstructured
                cylindrical mesh for VTK.
            """
            # Find the x values and y values for each plane.
            x_plane = (np.cos(theta) * r[:, None]).ravel()
            y_plane = (np.sin(theta) * r[:, None]).ravel()

            # Allocate an array for all the points.  We'll have len(x_plane)
            # points on each plane, and we have a plane for each z value, so
            # we need len(x_plane)*len(z) points.
            points = np.empty([len(x_plane) * len(z), 3])

            # Loop through the points for each plane and fill them with the
            # correct x,y,z values.
            start = 0
            for z_plane in z:
                end = start + len(x_plane)
                # slice out a plane of the output points and fill it
                # with the x,y, and z values for this plane.  The x,y
                # values are the same for every plane.  The z value
                # is set to the current z
                plane_points = points[start:end]
                plane_points[:, 0] = x_plane
                plane_points[:, 1] = y_plane
                plane_points[:, 2] = z_plane
                start = end

            return points

        dims = (3, 4, 3)
        r = np.linspace(5, 15, dims[0])
        theta = np.linspace(0, 0.5 * np.pi, dims[1])
        z = np.linspace(0, 10, dims[2])
        pts = generate_annulus(r, theta, z)
        sgrid = tvtk.StructuredGrid(dimensions=(dims[1], dims[0], dims[2]))
        sgrid.points = pts
        s = np.random.random((dims[0] * dims[1] * dims[2]))
        sgrid.point_data.scalars = np.ravel(s.copy())
        sgrid.point_data.scalars.name = 'scalars'

    #读取STL文件
    def test7(self):
        s = tvtk.STLReader(file_name="E:/mesh.stl")  # 调用stl文件
        m = tvtk.PolyDataMapper(input_connection=s.output_port)
        a = tvtk.Actor(mapper=m)

        # win=ivtk.IVTKWithCrust()
        # win=ivtk.IVTK()
        win = ivtk.IVTKWithCrustAndBrowser()
        win.open()
        win.scene.add_actor(a)
        win.scene.isometric_view()

        dialog = win.control.centralWidget().widget(0).widget(0)
        from pyface.qt import QtCore
        dialog.setWindowFlags(QtCore.Qt.WindowFlags(0x00000000))
        dialog.show()

        gui = GUI()
        gui.start_event_loop()

    def test8(self):

        plot3d = tvtk.MultiBlockPLOT3DReader(
            xyz_file_name="combxyz.bin",  # 网格文件
            q_file_name="combq.bin",  # 空气动力学结果文件
            scalar_function_number=100,  # 设置标量数据数量
            vector_function_number=200  # 设置矢量数据数量
        )
        plot3d.update()

        grid = plot3d.output.get_block(0)
        con=tvtk.ContourFilter()
        con.set_input_data(grid)
        con.generate_values(10,grid.point_data.scalars.range)

        m=tvtk.PolyDataMapper(scalar_range=grid.point_data.scalars.range,
                              input_connection=con.output_port)
        a=tvtk.Actor(mapper=m)
        a.property.opacity=0.5

        win=self.ivtk_scene(a)
        win.scene.isometric_view()
        self.event_loop()

    #mayavi mlab example
    def test9(self):
        # x=[[-1,1,1,-1,-1],[-1,1,1,-1,-1]]
        # y=[[-1,-1,-1,-1,-1],[2,2,2,2,2]]
        # z=[[1,1,-1,-1,1],[1,1,-1,-1,1]]
        x=[[0,0],[0,1]]
        y=[[0,0],[1,1]]
        z=[[0,1],[0,0]]
        mlab.mesh(x,y,z)
        mlab.show()

    def test10(self):
        from numpy import pi,sin,cos
        dphi,dtheta=pi/30,pi/30
        [phi,theta]=np.mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
        r=7
        x=r*sin(phi)*cos(theta)
        y=r*cos(phi)
        z=r*sin(phi)*sin(theta)
        mlab.mesh(x,y,z,representation='wireframe')
        mlab.show()

    def test11(self):
        t=np.linspace(0,4*np.pi,20)
        x=np.sin(2*t)
        y=np.cos(t)
        z=np.cos(2*t)
        s=np.zeros(20)+1
        mlab.points3d(x, y, z, s, colormap='Spectral', scale_factor=0.025)
        mlab.plot3d(x,y,z,s,colormap='Spectral',tube_radius=0.025)
        mlab.show()

    #imshow函数
    def test12(self):

        data = np.loadtxt('C:\\Users\\Administrator\\Documents\\GitHub\\DifferentialEvolution\\map.csv',
                          delimiter=',')
        mlab.imshow(data,colormap='gist_earth',interpolate=False)
        mlab.show()

    #surf函数
    def test13(self):

        data=np.loadtxt('./Data/map.csv', delimiter=',')
        row=data.shape[0]
        column=data.shape[1]
        x,y=np.mgrid[0:row:1,0:column:1]


        surf=mlab.surf(x,y,data)
        lut=surf.module_manager.scalar_lut_manager.lut.table.to_array()
        lut[:,-1]=np.linspace(0,125,256)
        surf.module_manager.scalar_lut_manager.lut.table=lut
        mlab.show()

    #contour函数
    def test14(self):
        data=np.loadtxt('./Data/map.csv', delimiter=',')
        row=data.shape[0]
        column=data.shape[1]
        x,y=np.mgrid[0:row:1,0:column:1]

        x,y=np.mgrid[0:row:1,0:column:1]
        mlab.contour_surf(x,y,data,contours=100)
        mlab.show()

    #向量场绘制
    def test15(self):
        from numpy import mgrid,sqrt,sin,zeros_like
        x, y, z = mgrid[-0:3:0.6, -0:3:0.6, 0:3:0.3]
        r = sqrt(x ** 2 + y ** 2 + z ** 4)
        u = y * sin(r) / (r + 0.001)
        v = -x * sin(r) / (r + 0.001)
        w = zeros_like(r)
        mlab.quiver3d(x, y, z, u, v, w)
        mlab.colorbar()
        mlab.show()

    #鼠标选取物体
    def test16(self):
        figure=mlab.gcf()
        figure.scene.disable_render=True
        x1,y1,z1=np.random.random((3,10))
        red_glyphs=mlab.points3d(x1,y1,z1,color=(1,0,0),resolution=10)

        x2,y2,z2=np.random.random((3,10))
        white_glyphs=mlab.points3d(x2,y2,z2,color=(0.9,0.9,0.9),resolution=10)

        outline=mlab.outline(line_width=3)
        outline.outline_mode='cornered'
        outline.bounds=(x1[0]-0.1,x1[0]+0.1,
                        y1[0] - 0.1, y1[0] + 0.1,
                        z1[0] - 0.1, z1[0] + 0.1)
        figure.scene.disable_render=False

        glyph_points=red_glyphs.glyph.glyph_source.glyph_source.output.points.to_array()

        def picker_callback(picker):
            if picker.actor in red_glyphs.actor.actors:
                point_id =int(picker.point_id/glyph_points.shape[0])
                if point_id!=-1:
                    x,y,z=x1[point_id],y1[point_id],z1[point_id]
                    outline.bounds=(x-0.1,x+0.1,
                        y - 0.1, y + 0.1,
                        z - 0.1, z + 0.1)

        picker=figure.on_mouse_pick(picker_callback)
        picker.tolerance=0.01
        mlab.title('Click on red balls')
        mlab.show()

    #标量数据可视化
    def test17(self):
        x,y,z=np.ogrid[-10:10:20j,-10:10:20j,-10:10:20j]
        s=np.sin(x*y*z)/(x*y*z)
        src=mlab.pipeline.scalar_field(s)
        # mlab.pipeline.image_plane_widget(src,plane_orientation='x_axes',slice_index=10)
        # mlab.pipeline.image_plane_widget(src, plane_orientation='y_axes', slice_index=10)
        # mlab.pipeline.iso_surface(src,contours=[s.min()+0.1*s.ptp()],opacity=0.1)
        # mlab.pipeline.iso_surface(src,contours=[s.max()-0.1*s.ptp()])
        # mlab.contour3d(s,contours=5,transparent=True)
        mlab.pipeline.volume(src)

        mlab.outline()
        mlab.show()

    #向量数据可视化
    def test18(self):
        x, y, z = np.mgrid[0:1:20j, 0:1:20j, 0:1:20j]
        u = np.sin(np.pi * x) * np.cos(np.pi * z)
        v = -2 * np.sin(np.pi * y) * np.cos(2 * np.pi * z)
        w = np.cos(np.pi * x) * np.sin(np.pi * z) + np.cos(np.pi * y) * np.sin(2 * np.pi * z)

        #以下是四种不同的向量场绘制函数，mlab.pipeline.vector_field是对原向量场的降采样

        # #1.绘制等势面，即曲面向量范数相同
        # src=mlab.pipeline.vector_field(u,v,w)
        # magnitude=mlab.pipeline.extract_vector_norm(src)
        # mlab.pipeline.iso_surface(magnitude,contours=[1.9,1.8])

        # #2.绘制整个向量场
        # src=mlab.pipeline.vector_field(u,v,w)        
        # mlab.pipeline.vectors(src,mask_points=10,scale_factor=1)

        # #3.仅绘制一个平面中的向量，且这个平面可以移动
        # src=mlab.pipeline.vector_field(u,v,w)        
        # mlab.pipeline.vector_cut_plane(src,mask_points=10,scale_factor=1)

        # #4.绘制流线图
        # src=mlab.pipeline.vector_field(u,v,w)
        # magnitude=mlab.pipeline.extract_vector_norm(src)
        # flow = mlab.pipeline.streamline(magnitude, seedtype='point',
        #                                 seed_visible=False,
        #                                 seed_scale=0.5,
        #                                 seed_resolution=5 )

        #以下的方法针对所有数据

        #显示所有数据
        # mlab.quiver3d(u,v,w)

        #绘制流线图
        mlab.flow(u,v,w,seed_scale=1,seed_resolution=3,integration_direction='both',seedtype='sphere')


        mlab.show()


def showData(path):
        
    graphData = []  
    pointData = []  

    with open(path,'r') as file:
        texts = file.readlines()
        pos = 0 
        import re
        reg_int = re.compile(r'-?\d+')
        reg_float = re.compile(r'-?\d\.\d+e[\+,-]\d+')   
     
        num = int(reg_int.search(texts[pos]).group(0))
        pos+=1

        for k in range(num//3):
            row = int(reg_int.search(texts[pos]).group(0))
            col = int(reg_int.search(texts[pos+1]).group(0))
            pos+=3

            x = np.zeros((row,col))
            for i in range(row):
                text_list = reg_float.findall(texts[pos+i])
                for j in range(col):
                    x[i,j] = float(text_list[j])
            graphData.append(x)
            pos+=row+1

            y = np.zeros((row,col))
            for i in range(row):
                text_list = reg_float.findall(texts[pos+i])
                for j in range(col):
                    y[i,j] = float(text_list[j])
            graphData.append(y)              
            pos+=row+1

            v = np.zeros((row,col))
            for i in range(row):
                text_list = reg_float.findall(texts[pos+i])
                for j in range(col):
                    v[i,j] = float(text_list[j])
            graphData.append(v)   
            pos+=row              

        num = int(reg_int.search(texts[pos]).group(0))
        pos+=1

        for k in range(num//2):
            row = int(reg_int.search(texts[pos]).group(0))
            col = int(reg_int.search(texts[pos+1]).group(0))
            pos+=3

            x = np.zeros((row,col))
            for i in range(row):
                text_list = reg_float.findall(texts[pos+i])
                for j in range(col):
                    x[i,j] = float(text_list[j])
            pointData.append(x)
            pos+=row+1

            y = np.zeros(row)
            for i in range(row):
                y[i] = float(reg_float.search(text_list[j]).group(0))
            pointData.append(y)              
            pos+=row+1
  
    #当前只针对len(graphData)==3&len(pointData)==2的情况，多余的数据不显示：
    mlab.figure(size=[1024,800])
    graph = mlab.imshow(graphData[0],graphData[1],graphData[2])
    # mlab.surf(x,y,s)
    point = mlab.points3d(pointData[0][:,0],pointData[0][:,1],np.zeros(pointData[0].shape[0]),scale_factor=0.5)
    # mlab.outline()
    # mlab.axes(xlabel='x', ylabel='y', zlabel='z')
    mlab.colorbar(graph,'value',orientation='vertical',label_fmt='%.2f')
    mlab.view(0,0)
    photoPath = re.sub(r'.txt','.png',path)
    mlab.savefig(photoPath)
    mlab.show()


if __name__=='__main__':
    # test=DataVisual()
    # test.test18()
    # showData('./Data/Kriging_Predicte_Model.txt')
    # showData('./Data/Kriging_True_Model.txt')
    showData('./Data/Kriging_Varience_Model.txt')