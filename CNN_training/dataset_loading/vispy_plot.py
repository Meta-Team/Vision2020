# generate 3d plot in vispy
import sys
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vispy import app, visuals, scene

#########################################################################################################################
# set generic plotting parameters
Xmin, Xmax, Xcenter = 0, 640, 320
Ymin, Ymax, Ycenter = 0, 480, 240
Zmin, Zmax, Zcenter = -10, 10, 0
XaxisColor, YaxisColor, ZaxisColor = [1,0,0,1], [0,1,0,1], [1,1,1,1]
XaxisMarginRatio, YaxisMarginRatio, ZaxisMarginRatio = 0.1, 0.1, 0.1
XaxisScale, YaxisScale, ZaxisScale = 4/5/(Xmax-Xmin), 3/5/(Ymax-Ymin), 1/(Zmax-Zmin)

PLOT_Z_FUNC = False

#########################################################################################################################
# set Z funcs plotting parameters, used when PLOT_Z_FUNC is True
if PLOT_Z_FUNC:
    def zfunc1(x, y):
        return 3*x**2 + 6*y**2 + 8*x*y

    def zfunc2(x, y):
        return x + y

    zfuncs = [zfunc1, zfunc2]
    zfuncColors = [[0,1,1,1], [1,0,0,1]]
    Xprec, Yprec = 300, 300
    XYmesh = np.meshgrid(np.linspace(start=Xmin, stop=Xmax, num=Xprec), np.linspace(start=Ymin, stop=Ymax, num=Yprec))
    Xs, Ys = XYmesh[0].flatten(), XYmesh[1].flatten()
    Zs = [zfuncs[i](Xs, Ys) for i in range(len(zfuncs))]
    Zmin, Zmax = min([np.min(i) for i in Zs]), max([np.max(i) for i in Zs]) # rewrite Zmin and Zmax
    ZaxisScale = 1/(Zmax-Zmin)

#########################################################################################################################
# set non-Z-funcs plot parameters, obtain [Xs, Ys, Zs] as a list of 3 1-D numpy.ndarrays, used when PLOT_Z_FUNC is False
else:
    # load data
    all_2_class_bboxes = np.load("all_bboxes.npy")
    all_x_means = all_2_class_bboxes[:,0] / 2 + all_2_class_bboxes[:,2] / 2
    all_y_means = all_2_class_bboxes[:,1] / 2 + all_2_class_bboxes[:,3] / 2
    all_size_ratios = (all_2_class_bboxes[:,2] - all_2_class_bboxes[:,0]) * (all_2_class_bboxes[:,3] - all_2_class_bboxes[:,1]) / (640*480)
    all_aspect_ratios = (all_2_class_bboxes[:,2] - all_2_class_bboxes[:,0]) / (all_2_class_bboxes[:,3] - all_2_class_bboxes[:,1]) # x / y
    all_log_aspect_ratios = np.log(all_aspect_ratios)
    all_colors = np.vstack([np.array([1,0,0]) if i == 1 else np.array([0,0,1]) for i in all_2_class_bboxes[:,-1]])
    
    Xs = all_x_means
    Ys = all_y_means
    Zs = all_log_aspect_ratios
    scatterColors = all_colors
    Zmin, Zmax, Zcenter = min(all_log_aspect_ratios), max(all_log_aspect_ratios), 0
    ZaxisScale = 1/(Zmax-Zmin)

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################

# build visuals
visual3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
# plot using scene
# build canvas
canvas = scene.SceneCanvas(keys='interactive', show=True,  size=(2880, 1800))
# Add a ViewBox to let the user zoom/rotate
view = canvas.central_widget.add_view()
view.bgcolor = 'black'
view.camera = 'turntable'
view.camera.fov = 0
view.camera.distance = 100

# plot first curve
visual3Ds = []
if PLOT_Z_FUNC:
    Xs, Ys = (Xs-Xcenter)*XaxisScale, (Ys-Ycenter)*YaxisScale
    for i in range(len(zfuncs)):
        Zs[i] = (Zs[i]-Zcenter)*ZaxisScale
        points = np.array([Xs, Ys, Zs[i]]).T
        # plot scatter surface
        visual3Ds.append(visual3D(parent=view.scene))
        visual3Ds[i].set_gl_state('opaque', blend=True, depth_test=True) # translucent, additive
        visual3Ds[i].set_data(points, face_color=zfuncColors[i], symbol='o', size=2, edge_width=0.5, edge_color=zfuncColors[i])
        # plot surface
        #Xranges, Yranges = np.linspace(start=Xmin, stop=Xmax, num=Xprec), np.linspace(start=Ymin, stop=Ymax, num=Yprec)
        #visual3Ds.append(scene.visuals.SurfacePlot(x=Xranges,y=Yranges, z=Zs[i].reshape(len(Xranges), len(Yranges)), color=zfuncColors[i], parent=view.scene))
else:
    Xs, Ys, Zs = (Xs-Xcenter)*XaxisScale, (Ys-Ycenter)*YaxisScale, (Zs-Zcenter)*ZaxisScale
    points = np.array([Xs, Ys, Zs]).T
    # plot scatter surface
    visual3Ds = visual3D(parent=view.scene)
    visual3Ds.set_gl_state('opaque', blend=True, depth_test=True) # translucent, additive
    visual3Ds.set_data(points, face_color=scatterColors, symbol='o', size=5, edge_width=0.5, edge_color=scatterColors)
    # plot surface
    #Xranges, Yranges = np.linspace(start=Xmin, stop=Xmax, num=Xprec), np.linspace(start=Ymin, stop=Ymax, num=Yprec)
    #visual3Ds.append(scene.visuals.SurfacePlot(x=Xranges,y=Yranges, z=Zs[i].reshape(len(Xranges), len(Yranges)), color=zfuncColors[i], parent=view.scene))

# plot X,Y,Z axes.
Xmargin, Ymargin, Zmargin = XaxisMarginRatio*(Xmax-Xmin), YaxisMarginRatio*(Ymax-Ymin), ZaxisMarginRatio*(Zmax-Zmin)
xax = scene.Axis(domain=(Xmin-Xmargin, Xmax+Xmargin), pos=[[(Xmin-Xmargin-Xcenter)*XaxisScale, 0], [(Xmax+Xmargin-Xcenter)*XaxisScale, 0]], tick_direction=(0, -1), 
                    axis_color=XaxisColor, tick_color=XaxisColor, text_color=XaxisColor, font_size=5, parent=view.scene)
yax = scene.Axis(domain=(Ymin-Ymargin, Ymax+Ymargin), pos=[[0, (Ymin-Ymargin-Ycenter)*YaxisScale], [0, (Ymax+Ymargin-Ycenter)*YaxisScale]], tick_direction=(-1, 0), 
                    axis_color=YaxisColor, tick_color=YaxisColor, text_color=YaxisColor, font_size=5, parent=view.scene)
zax = scene.Axis(domain=(Zmin-Zmargin, Zmax+Zmargin), pos=[[(Zmin-Zmargin-Zcenter)*ZaxisScale, 0], [(Zmax+Zmargin-Zcenter)*ZaxisScale, 0]], tick_direction=(0, -1), 
                    axis_color=ZaxisColor, tick_color=ZaxisColor, text_color=ZaxisColor, font_size=5, parent=view.scene)
zax.transform = scene.transforms.MatrixTransform()  # scene has no 3d support, so rotate zax in space to its correct direction
zax.transform.rotate(-90, (0, 1, 0))  # rotate around yaxis
zax.transform.rotate(-45, (0, 0, 1))  # tick direction towards (-1,-1)

# run vispy plotting
if __name__ == '__main__':
    if sys.flags.interactive != 1:
        app.run()
