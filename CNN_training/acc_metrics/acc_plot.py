# generate 3d plot in vispy
import sys
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vispy import app, visuals, scene

# Xaxis: red, predicted value of relative position in image in [0,1]
# Yaxis: green, true value of relative position in image in [0,1]
# Zaxis: blue, accuracy in [0,1]

#########################################################################################################################
# set generic plotting parameters
Xmin, Xmax, Xcenter = 0, 1, 0.5
Ymin, Ymax, Ycenter = 0, 1, 0.5
XaxisColor, YaxisColor, ZaxisColor = [1,0,0,1], [0,1,0,1], [0,0,1,1]
XaxisMarginRatio, YaxisMarginRatio, ZaxisMarginRatio = 0.1, 0.1, 0.1
XaxisScale, YaxisScale = 1/(Xmax-Xmin), 1/(Ymax-Ymin)


#########################################################################################################################
# set Z funcs plotting parameters
def relu(x):
    return x * (x > 0)

def acc_aid(x, y):
    return 1 - abs(x-y)**(1/2) * (1 - 4*relu((x-0.5)*(y-0.5)))**2

def acc_auto(x, y):
    return (1 - abs(x-y)*(1 - 4*relu((x-0.5)*(y-0.5)))) * (1-relu(np.tanh(-100*(x-0.5)*(y-0.5))))

# draw acc_aid and acc_auto
# zfuncs = [acc_aid, acc_auto]
# zfuncColors = [[1,0,0,1],[0,1,1,1]] # scatter plot dot colors for all zfuncs: red for acc_aid, blue for acc_auto

# only draw acc_aid
# zfuncs = [acc_aid]
# zfuncColors = [[1,0,0,1]] # scatter plot dot colors for all zfuncs: red for acc_aid

# # only draw acc_auto
zfuncs = [acc_auto]
zfuncColors = [[0,0,1,1]] # scatter plot dot colors for all zfuncs: blue for acc_auto

Xprec, Yprec = 500, 500
XYmesh = np.meshgrid(np.linspace(start=Xmin, stop=Xmax, num=Xprec), np.linspace(start=Ymin, stop=Ymax, num=Yprec))
Xs, Ys = XYmesh[0].flatten(), XYmesh[1].flatten()
Zs = [zfuncs[i](Xs, Ys) for i in range(len(zfuncs))]

# # filter points with Z at least 0.5
# tolerance = 0.01
# target_Z = 0.5
# target_Z_idx = 0
# idx_list = []
# for i in range(len(Zs[target_Z_idx])):
#     if  Zs[target_Z_idx][i] > target_Z:
#         idx_list.append(i)
# Xs = Xs[idx_list]
# Ys = Ys[idx_list]
# for i in range(len(Zs)):
#     Zs[i] = Zs[i][idx_list]

# rewrite Zmin and Zmax for axis scale
Zmin, Zmax = min([np.min(i) for i in Zs]), max([np.max(i) for i in Zs]) 
ZaxisScale = 1/(Zmax-Zmin)
Zcenter = (Zmax+Zmin)/2

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
view.bgcolor = 'white'
view.camera = 'turntable'
view.camera.fov = 0
view.camera.distance = 100

# plot first curve
visual3Ds = []
Xs, Ys = (Xs-Xcenter)*XaxisScale, (Ys-Ycenter)*YaxisScale
for i in range(len(zfuncs)):
    Zs[i] = (Zs[i]-Zcenter)*ZaxisScale
    points = np.array([Xs, Ys, Zs[i]]).T
    # plot scatter surface
    visual3Ds.append(visual3D(parent=view.scene))
    visual3Ds[i].set_gl_state('translucent', blend=True, depth_test=True) # opaque, translucent, additive
    visual3Ds[i].set_data(points, face_color=zfuncColors[i], symbol='o', size=5, edge_width=0.5, edge_color=zfuncColors[i])
    # plot surface
    # Xranges, Yranges = np.linspace(start=np.min(Xs), stop=np.max(Xs), num=Xprec), np.linspace(start=np.min(Ys), stop=np.max(Ys), num=Yprec)
    # visual3Ds.append(scene.visuals.SurfacePlot(x=Xranges,y=Yranges, z=Zs[i].reshape(len(Xranges), len(Yranges)), color=zfuncColors[i], parent=view.scene))

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
zax.transform.rotate(-90, (0, 0, 1))  # tick direction 

# run vispy plotting
if __name__ == '__main__':
    if sys.flags.interactive != 1:
        app.run()
