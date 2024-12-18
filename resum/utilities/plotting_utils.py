# https://community.plotly.com/t/rotating-3d-plots-with-plotly/34776/2
# https://community.plotly.com/t/how-to-export-animation-and-save-it-in-a-video-format-like-mp4-mpeg-or/64621/2
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
from scipy.optimize import minimize

def get_points_xy(x):
    l = x[4]/2
    # Convert theta to radians
    theta = x[3] * np.pi/180.
    phi = 360./float(x[2])
    phi = phi* np.pi/180.
    x=[]
    y=[]
    for i in range(x[2]):
        x.append(0)
        y.append(0)
        x.append(x[0]*np.cos(i*phi))
        y.append(x[0]*np.sin(i*phi))
        x.append(x[0]*np.cos(i*phi) + rotate(0.,x[4]/2, np.pi + i*phi + theta)[0])
        y.append(x[0]*np.sin(i*phi) + rotate(0.,x[4]/2, np.pi + i*phi + theta)[1])
        x.append(x[0]*np.cos(i*phi) + rotate(0.,-x[4]/2, np.pi + i*phi + theta)[0])
        y.append(x[0]*np.sin(i*phi) + rotate(0.,-x[4]/2, np.pi + i*phi + theta)[1])

        x.append(x[0]*np.cos(i*phi) + rotate(x[1]/2,x[4]/2, np.pi + i*phi + theta)[0])
        y.append(x[0]*np.sin(i*phi) + rotate(x[1]/2,x[4]/2, np.pi + i*phi + theta)[1])
        x.append(x[0]*np.cos(i*phi) + rotate(x[1]/2,-x[4]/2, np.pi + i*phi + theta)[0])
        y.append(x[0]*np.sin(i*phi) + rotate(x[1]/2,-x[4]/2, np.pi + i*phi + theta)[1])

        x.append(x[0]*np.cos(i*phi) + rotate(-x[1]/2,x[4]/2, np.pi + i*phi + theta)[0])
        y.append(x[0]*np.sin(i*phi) + rotate(-x[1]/2,x[4]/2, np.pi + i*phi + theta)[1])
        x.append(x[0]*np.cos(i*phi) + rotate(-x[1]/2,-x[4]/2, np.pi + i*phi + theta)[0])
        y.append(x[0]*np.sin(i*phi) + rotate(-x[1]/2,-x[4]/2, np.pi + i*phi + theta)[1])
    return x, y

def get_outer_radius(x):
    theta=x[3] * np.pi/180.
    l = x[4]/2

    Ax = x[0] + (l*np.cos(np.pi/2 - theta)) + np.cos(theta*np.pi/180.)* x[1]/2.
    Ay = l*np.sin(np.pi/2 - theta)          + np.sin(theta*np.pi/180.)* x[1]/2.
    Bx = x[0] - (l*np.cos(np.pi/2 - theta)) + np.cos(theta*np.pi/180.)* x[1]/2.
    By = -l*np.sin(np.pi/2 - theta)         + np.sin(theta*np.pi/180.)* x[1]/2.

    def outer_radius(x, Ax, Bx, Ay, By):
        y1=(x-Ax)*(By-Ay)/(Bx-Ax)+Ay
        if (Bx == Ax):
            y1 = 0.
        return -1.*np.sqrt(x**2+y1**2)

    x_ini = Ax
    bounds=[[np.min([Ax,Bx]),np.max([Ax,Bx])]]
    res = minimize(outer_radius, x_ini, args=(Ax,Bx,Ay,By), bounds=bounds)
    return np.abs(res.fun)

def get_inner_radius(x):
    theta=x[3] * np.pi/180.
    l = x[4]/2

    Bx = x[0] + (l*np.cos(np.pi/2 - theta)) - np.cos(theta*np.pi/180.)* x[1]/2.
    By = l*np.sin(np.pi/2 - theta)          - np.sin(theta*np.pi/180.)* x[1]/2.
    Ax = x[0] - (l * np.cos(np.pi/2 - theta)) - np.cos(theta*np.pi/180.)* x[1]/2.
    Ay = -l*np.sin(np.pi/2 - theta)         - np.sin(theta*np.pi/180.)* x[1]/2.

    def inner_radius(x, Ax, Bx, Ay, By):
        y1=(x-Ax)*(By-Ay)/(Bx-Ax)+Ay
        if Bx==Ax:
            y1=0.
        #print("x", (Ax,Ay), (Bx, By), x, (x-Ax),(By-Ay),(Bx-Ax))
        #print("y1",y1,np.sqrt(x**2+y1**2))
        return np.sqrt(x**2+y1**2)

    x_ini = Ax
    bounds=[[np.min([Ax,Bx]),np.max([Ax,Bx])]]
    res = minimize(inner_radius, x_ini, args=(Ax,Bx,Ay,By), bounds=bounds)
    #print("result",res)
    return res.fun

def get_formated(x_train_l, index_x,index_y,index_z,y_train_l):
    xtmp=np.array([i for i in np.atleast_2d((x_train_l[:].T[index_x]))])
    ytmp=[i for i in np.atleast_2d((x_train_l[:].T[index_y]))]
    ztmp=[i for i in np.atleast_2d((x_train_l[:].T[index_z]))]

    for i in xtmp:
        x=i
    for i in ytmp:
        y=i
    for i in ztmp:
        z=i
    c=y_train_l.reshape((len(y_train_l),))
    return x,y,z,c

def draw_samples_distribution_3D(x,y,z,c):
    fig = go.Figure(go.Scatter3d(x=x, y=y, z=z, mode="markers",marker=dict(
        size=8,
        color=c,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8,
        showscale=True
    )))
    x_eye = -1.25
    y_eye = 2
    z_eye = 1.0


    fig.show()

def draw_samples_distribution_3D_rotating(x,y,z,c):
    fig = go.Figure(go.Scatter3d(x=x, y=y, z=z,mode="markers",
        marker=dict(
            size=8,
            color=c,                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8,
            showscale=True
        )
    ))
    x_eye = 0
    y_eye = 2
    z_eye = 1.0

    fig.update_layout(
            title="",
            width=600,
            height=600,
            scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
            updatemenus=[
                dict(
                    type="buttons",
                showactive=False,
                y=1,
                x=0.8,
                xanchor="left",
                yanchor="bottom",
                pad=dict(t=45, r=10),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=5, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    )
                ],
            )
        ],
    )


    def rotate_z(x, y, z, theta):
        w = x + 1j * y
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    frames = []
    pil_frames = []
    for t in np.arange(0, 3.14, 0.025):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze))))
    fig.frames = frames

    fig.show()

def draw_parameter_dependencies(ax, x_train_l, y_train_l, x_label='Radius', y_label='Thickness',labels=['Radius','Thickness','Phi', 'Theta', 'Length']):
    ltmp=np.array(labels)
    x_idx=np.where(ltmp==x_label)[0][0]
    y_idx=np.where(ltmp==y_label)[0][0]
    colors = cm.hsv(y_train_l/max(y_train_l))
    x,y,z,c = get_formated(x_train_l,x_idx,y_idx,2,y_train_l)
    c=1
    ax.grid(True,linestyle='-',color='0.75')
    # scatter with colormap mapping to z value
    a = ax.scatter(x,y,s=20,c=colors, marker = 'o', cmap = cm.jet );
    
    #fig.colorbar(a)
    #plt.show()

def draw_parameter_corr(x_train_l,y_train_l,labels):
    fig, axs = plt.subplots(len(labels), len(labels),figsize=(9,9), layout="constrained")

    colmap = cm.ScalarMappable(cmap=cm.hsv)
    colmap.set_array(y_train_l)

    for i in range(len(labels)):
        for j in range(len(labels)):
            DrawParameterDependencies(axs[j,i], x_train_l, y_train_l,labels[i],labels[j],labels)
            if i==0:
                axs[j,i].set_ylabel(labels[j])
            if j==len(labels)-1:
                axs[j,i].set_xlabel(labels[i])

    fig.colorbar(colmap)

def draw_moderator_config(radius, thickness, npanels, theta, length):
    figure, axes = plt.subplots( 1 )
    axes.set_aspect( 1 )
    axes.set_facecolor([135./265.,151./265.,154./265.])
    axes.set_xlim(-300,300)
    axes.set_ylim(-300,300)
    theta2 = np.linspace( 0 , 2 * np.pi , 150 )
    r = 265
    a = r * np.cos( theta2 )
    b = r * np.sin( theta2 )
    axes.plot( a, b, color=[182./265., 182./265., 182./265.] )
    axes.fill_between(a,b,color=[182./265., 182./265., 182./265.])
    
    r2 = 90
    a2 = r2 * np.cos( theta2 )
    b2 = r2 * np.sin( theta2 )
    axes.plot( a2, b2, color=[172./265., 165./265., 162./265.])
    axes.fill_between(a2,b2,color=[128./265., 155./265., 151./265.])
    npanels=int(np.round(npanels))
    phi = 360./float(npanels)
    #axes.grid(color='lightgray', linestyle='-', linewidth=1)
    for i in range(npanels):
        ang=180.+i*phi+theta
        x=-thickness/2+radius*np.cos(i* phi * np.pi/180)
        y=-length/2+ radius*np.sin(i*phi* np.pi/180)
        axes.add_patch(Rectangle((x,y),thickness,length, rotation_point='center',
                    angle= ang,
                    edgecolor='none',
                    facecolor=[0./265., 125./265., 115./265.],
                    lw=4))
    axes.text(-200,-380, f'r= {round(radius,1)}, d={round(thickness,1)}, N={round(npanels,0)}, '+r'$\theta$'+f'={round(theta,1)}, L={round(length,1)}', fontsize=8)

def draw_moderator_configuration(x, axes=None):
    if axes == None:
        figure, axes = plt.subplots( 1 )
    axes.set_aspect( 1 )
    axes.set_facecolor([135./265.,151./265.,154./265.])
    axes.set_xlim(-300,300)
    axes.set_ylim(-300,300)
    theta2 = np.linspace( 0 , 2 * np.pi , 150 )
    r = 265
    a = r * np.cos( theta2 )
    b = r * np.sin( theta2 )
    axes.plot( a, b, color=[182./265., 182./265., 182./265.] )
    axes.fill_between(a,b,color=[182./265., 182./265., 182./265.])
    
    r2 = 90
    a2 = r2 * np.cos( theta2 )
    b2 = r2 * np.sin( theta2 )
    axes.plot( a2, b2, color=[172./265., 165./265., 162./265.])
    axes.fill_between(a2,b2,color=[128./265., 155./265., 151./265.])
    x[2]=int(np.round(x[2]))
    phi = 360./float(x[2])
    #axes.grid(color='lightgray', linestyle='-', linewidth=1)
    for i in range(int(x[2])):
        ang=180.+i*phi-x[3]
        x1=-x[1]/2+x[0]*np.cos(i* phi * np.pi/180.)
        y1=-x[4]/2+x[0]*np.sin(i*phi* np.pi/180.)
        axes.add_patch(Rectangle((x1,y1),x[1],x[4], rotation_point='center',
                    angle= ang,
                    edgecolor='none',
                    facecolor=[0./265., 125./265., 115./265.],
                    lw=4))
    axes.text(-200,-380, f'r= {round(x[0],1)}, d={round(x[1],1)}, N={round(x[2],0)}, '+r'$\theta$'+f'={round(x[3],1)}, L={round(x[4],1)}', fontsize=8)

def draw_moderator(x,draw_radius=0):

    figure, axes = plt.subplots( 1 )
    axes.set_aspect( 1 )
    alpha = np.linspace( 0 , 2 * np.pi , 150 )
    r = 265
    a = r * np.cos( alpha )
    b = r * np.sin( alpha )
    axes.plot( a, b, color='gray' )
    
    r2 = 90
    a2 = r2 * np.cos( alpha )
    b2 = r2 * np.sin( alpha )
    axes.plot( a2, b2, color='gray')

    if draw_radius >0:
        r2 = draw_radius
        a2 = r2 * np.cos( alpha )
        b2 = r2 * np.sin( alpha )
        axes.plot( a2, b2, color='blue')
    
    phi = 2*np.pi/x[2]
    for i in range(x[2]):
        center_x = np.cos(phi*i)*x[0]
        center_y = np.sin(phi*i)*x[0]

        #plt.gca().add_patch(Rectangle((center_x-thickness/2,center_y-length/2),thickness, length, color='gray'))
        plt.gca().add_patch(Rectangle((center_x-x[1]/2,center_y-x[4]),x[1], x[4]*2, angle=-x[3], rotation_point='center'))

    return [figure, axes]

def rotate(x,y,theta):
    x_new=x * np.cos(theta) - y * np.sin(theta)
    y_new=x * np.sin(theta) + y * np.cos(theta)
    return x_new, y_new


def get_points(x):
    l = x[4]/2
    # Convert theta to radians
    theta = x[3] * np.pi/180.
    phi = 360./np.round(x[2]) * np.pi/180.
    
    A0x = x[0] + (l*np.cos(np.pi/2 - theta)) + np.cos(theta*np.pi/180.)* x[1]/2.
    A0y = l*np.sin(np.pi/2 - theta)          + np.sin(theta*np.pi/180.)* x[1]/2.
    B0x = x[0] - (l*np.cos(np.pi/2 - theta)) + np.cos(theta*np.pi/180.)* x[1]/2.
    B0y = -l*np.sin(np.pi/2 - theta)         + np.sin(theta*np.pi/180.)* x[1]/2.
    C0x = x[0] + (l*np.cos(np.pi/2 - theta)) - np.cos(theta*np.pi/180.)* x[1]/2.
    C0y = l*np.sin(np.pi/2 - theta)          - np.sin(theta*np.pi/180.)* x[1]/2.
    D0x = x[0] - (l*np.cos(np.pi/2 - theta)) - np.cos(theta*np.pi/180.)* x[1]/2.
    D0y = -l*np.sin(np.pi/2 - theta)         - np.sin(theta*np.pi/180.)* x[1]/2.

    A1x = x[0]*np.cos(phi) - (l*np.cos(np.pi/2 - phi + theta)) - np.cos(theta*np.pi/180.)* x[1]/2.
    A1y = x[0]*np.sin(phi) + (l*np.sin(np.pi/2 - phi + theta)) - np.sin(theta*np.pi/180.)* x[1]/2.
    B1x = x[0]*np.cos(phi) + (l*np.cos(np.pi/2 - phi + theta)) - np.cos(theta*np.pi/180.)* x[1]/2.
    B1y = x[0]*np.sin(phi) - (l*np.sin(np.pi/2 - phi + theta)) - np.sin(theta*np.pi/180.)* x[1]/2.
    C1x = x[0]*np.cos(phi) - (l*np.cos(np.pi/2 - phi + theta)) + np.cos(theta*np.pi/180.)* x[1]/2.
    C1y = x[0]*np.sin(phi) + (l*np.sin(np.pi/2 - phi + theta)) + np.sin(theta*np.pi/180.)* x[1]/2.
    D1x = x[0]*np.cos(phi) + (l*np.cos(np.pi/2 - phi + theta)) + np.cos(theta*np.pi/180.)* x[1]/2.
    D1y = x[0]*np.sin(phi) - (l*np.sin(np.pi/2 - phi + theta)) + np.sin(theta*np.pi/180.)* x[1]/2.
    
    return [[A0x,A0y],[B0x,B0y],[C0x,C0y],[D0x,D0y],[A1x,A1y],[B1x,B1y],[C1x,C1y],[D1x,D1y]]

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A,B,C,D):
    # Return true if line segments AB and CD intersect
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def is_crossed(x):
    _,_,A,B,_,_,C,D = get_points(x)
    return intersect(A,B,C,D)

def draw_panel_border(x,radius=0):
    points=get_points(x)
    figure, axes = plt.subplots( 1 )
    axes.set_aspect( 1 )
    theta = np.linspace( 0 , 2 * np.pi , 150 )
    r = 265
    a = r * np.cos( theta )
    b = r * np.sin( theta )
    axes.plot( a, b, color='gray' )
    
    r2 = 90
    a2 = r2 * np.cos( theta )
    b2 = r2 * np.sin( theta )
    axes.plot( a2, b2, color='gray')

    if radius >0:
        r2 = radius
        a2 = r2 * np.cos( theta )
        b2 = r2 * np.sin( theta )
        axes.plot( a2, b2, color='blue')
    xs = [x[0] for x in points]
    ys = [x[1] for x in points]

    for i in range(0,8,2):
        axes.plot(xs[i:i+2], ys[i:i+2])
   
    return [figure, axes]

def get_subplot_moderator(ax, x):

    ax.set_aspect( 1 )
    phi = 2.*np.pi/np.round(x[2])
    for i in range(int(x[2])):
        center_x = np.cos(phi*i)*x[0]
        center_y = np.sin(phi*i)*x[0]
        plt.gca().add_patch(Rectangle((center_x-x[1]/2,center_y-x[4]/2),x[1], x[4], angle=i*(360./x[2])-x[3], color="teal", rotation_point='center'))

    alpha = np.linspace( 0 , 2 * np.pi , 150 )
    r = 265
    a = r * np.cos( alpha )
    b = r * np.sin( alpha )
    ax.plot( a, b, color='gray' )
    
    r2 = 90
    a2 = r2 * np.cos( alpha )
    b2 = r2 * np.sin( alpha )
    ax.plot( a2, b2, color='gray')

def parameter_constraints(x):
        if any(i < 0 for i in x) is True:
            return 0
        elif get_inner_radius(x) < 90.:
            return 0
        elif get_outer_radius(x) > 265.:
            return 0
        elif get_outer_radius(x)-get_inner_radius(x)  > 20.:
            return 0
        elif x[2]*x[1]*x[4] > np.pi*(get_outer_radius(x)**2-get_inner_radius(x)**2):
                return 0
        else:
            return 1