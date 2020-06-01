#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import scipy as sp
from scipy import ndimage
from scipy import stats
import pandas
import copy
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm
import pylab as plt

#from mpl_toolkits import mplot3d
get_ipython().run_line_magic('matplotlib', 'auto')

from mpl_toolkits.mplot3d import Axes3D
#import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

n_tic = time.process_time()
# img = cv2.imread( r'm31_24proj.png', 0)
# x,y = [], []
# for i in range(277):          #select pixels based on intensity, over range of image
#     for j in range(306):
#         if img[i, j] >= 75:
#             x.append(i)
#             y.append(j)
# xx, yy=[], []
# for i in x:
#     i=-(i-138.5)
#     xx.append(i)
# for j in y:
#     j=j-153
#     yy.append(y)
# IR=np.vstack((yy,xx))

# Y=IR[1]
# X=IR[0]
# inject = np.vstack([X, Y])
# IR = np.append(CR, inject, axis=1)
# Basic cmap b from ds9
cdict= {'red':  ((0., 0.25, 0.25 ),
                 (0.25, 0, 0 ),
                 (0.5, 1, 1),
                 (1, 1, 1)),
 
        'green': ((0., 0, 0 ),
                 (0.5, 0, 0 ),
                 (0.75, 1, 1),
                 (1, 1, 1)),
 
        'blue': ((0, 0.25, 0.25),
                 (0.25, 1, 1),
                 (0.5, 0, 0),
                 (0.75, 0, 0),
                 (1, 1, 1)),
                 }

b_ds9 = LinearSegmentedColormap('b_ds9', cdict)

def build3d(num, theme='dark', grid=False, panel=0.0,boxsize=250, figsize=(8,6), dpi=150):
    ''' 
        Sets basic parameters for 3d plotting 
        and returns the figure and axis object.
        
        Inputs
        -----------
        theme: When set to 'dark' uses a black background
               any other setting will produce a typical white ackground
               
        grid: When true, includes the grid and ticks colored opposite of the background
        
        panel: 0-1, sets the opacity of the grid panels
        
        boxsize: Determines the -/+ boundaries of each axis
        
        figsize, dpi: matplotlib figure parameters
        
    '''
    
    fig = plt.figure(num = num, figsize=figsize, dpi=dpi)
    ax = Axes3D(fig)
    
    if(grid and theme=='dark'):
        color='white'
    else:
        color='black'
    
    if not(grid):
        ax.grid(False)
        
    ax.set_zlim3d([-boxsize, boxsize])
    ax.set_zlabel('Z', color=color)
    ax.set_ylim3d([-boxsize, boxsize])
    ax.set_ylabel('Y', color=color)
    ax.set_xlim3d([-boxsize, boxsize])
    ax.set_xlabel('X', color=color)
    ax.tick_params(axis='both', colors=color)
    
    # Sets pane color transparent
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, panel))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, panel))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, panel))

    # For all black black, no tick
    if(theme == 'dark'):
        fig.set_facecolor('black')
        ax.set_facecolor('black')
    
    return fig, ax

def reset3d(ax, theme='dark', grid=False, panel=0.0,boxsize=250, figsize=(8,6), dpi=150):
    ''' 
        resets basic parameters for 3d plotting 
        but returns only modified axis.
        
        Inputs
        -----------
        axis: current axis
        
        theme: When set to 'dark' uses a black background
               any other setting will produce a typical white ackground
               
        grid: When true, includes the grid and ticks colored opposite of the background
        
        panel: 0-1, sets the opacity of the grid panels
        
        boxsize: Determines the -/+ boundaries of each axis
        
    '''

    
    if(grid and theme=='dark'):
        color='white'
    else:
        color='black'
    
    if not(grid):
        ax.grid(False)
        
    ax.set_zlim3d([-boxsize, boxsize])
    ax.set_zlabel('Z', color=color)
    ax.set_ylim3d([-boxsize, boxsize])
    ax.set_ylabel('Y', color=color)
    ax.set_xlim3d([-boxsize, boxsize])
    ax.set_xlabel('X', color=color)
    ax.tick_params(axis='both', colors=color)
    
    # Sets pane color transparent
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, panel))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, panel))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, panel))

    # For all black black, no tick
    if(theme == 'dark'):
        ax.set_facecolor('black')
    
    return ax

def get_color(coordinates, kde=True, color='b'):
    ''' Calculates the KDE for the coordinates array
        If all particles leave the boundary, (ie empty coordinate array)
        a single color is returned
        
        Note KDE calculation is a bottleneck, 
        so avoid for large N particles by setting kde==False
    '''
    if(coordinates.shape[1]!=0 and kde):
        kde = stats.gaussian_kde(coordinates)
        color = kde(coordinates)
    
    else:
        color = color
    
    return color

############################################## Spawn Definitions #################################################


def spawn_sphere(CR, particles, rmax ,x0, y0, z0, shell=False):
    ''' 
    generates a spherically uniform distribuiton of particles 
    and append them to the provided CR array
        
    Inputs
    --------
    CR: the array to add particles to (TODO make condition so if CR=None, create sphere)
        
    particles: integer number of particles to produce
        
    rmax: max radius of sphere 
        
    x0,y0,z0: intital coordinates
        
    shell: if True, produces points on a spherical shell instead of solid sphere

    Return
    ---------
    the new particle coordinates array
    '''
    
    phi = np.random.uniform(0, 2*np.pi, particles)
    theta = np.arccos(np.random.uniform(-1, 1, particles))
    
    if shell==True:
        r=rmax
    else:
        r = rmax*pow(np.random.uniform(0, 1, particles), 1./3.)
    X = r*np.cos(phi)*np.sin(theta) + x0
    Y = r*np.sin(phi)*np.sin(theta) + y0
    Z = r*np.cos(theta) + z0
    
    inject = np.vstack([X,Y,Z])
    CR = np.append(CR, inject, axis=1)

    return CR

def spawn_sphere_ring(CR, particles, rmin, rmax ,x0, y0, z0, shell=False):
    ''' 
    generates a spherically uniform distribuiton of particles 
    and append them to the provided CR array
        
    Inputs
    --------
    CR: the array to add particles to (TODO make condition so if CR=None, create sphere)
        
    particles: integer number of particles to produce
        
    rmax: max radius of sphere 
        
    x0,y0,z0: intital coordinates
        
    shell: if True, produces points on a spherical shell instead of solid sphere

    Return
    ---------
    the new particle coordinates array
    '''
    
    phi = np.random.uniform(0, 2*np.pi, particles)
    theta = np.arccos(np.random.uniform(-1, 1, particles))
    
    if shell==True:
        r=rmax
    else:
        r = rmax*np.sqrt(np.random.uniform( pow(rmin/rmax, 2), 1, particles))
    X = r*np.cos(phi)*np.sin(theta) + x0
    Y = r*np.sin(phi)*np.sin(theta) + y0
    Z = r*np.cos(theta) + z0
    
    inject = np.vstack([X,Y,Z])
    CR = np.append(CR, inject, axis=1)

    return CR


def spawn_ring(CR, particles=10, rmin=15, rmax=15, thickness=10, x0=0,y0=0,z0=0, shell=False):
    ''' generates an annular uniform distribuiton of particles 
        and appends them to the provided CR array
        
        Inputs
        --------
        CR: the array to add particles to (TODO make condition so if CR=None, create sphere)
        
        particles: integer number of particles to produce
        
        rmin, rmax: min/max radius of the ring (note, rmin=0 produces a cylinder)
        
        x0,y0,z0: intital coordinates
        
        Return
        ---------
        the new particle coordinates array
    '''
# Note that r here means the radius of the cylinder of the spawn ring
    phi = np.random.uniform(0, 2*np.pi, particles)
    if shell==True:
        r = rmax
    else:
        r = rmax*np.sqrt(np.random.uniform( (pow(rmin/rmax, 2)), 1, particles))
    X = r*np.cos(phi) + x0
    Y = r*np.sin(phi) + y0
    Z = thickness * np.random.uniform(-1, 1, particles) + z0
    
    inject = np.vstack([X,Y,Z])

    CR = np.append(CR, inject, axis=1)
    return CR


def spawn_IR(CR, particles=36057, x0=0, y0=0):
    img = cv2.imread( r'm31_24proj.png', 0)
    X,Y = [], []

    for i in range(277):          #select pixels based on intensity, over range of image
        for j in range(306):
            if img[i, j] >= 75: #75 is intensity
                y=-(i-138.5)# image is flipped wrt y axis so we need to multiply by negative one. The 138.5 comes from needing to center the image at the origin
                x=j-153 #centers image at center
                X.append(x)
                Y.append(y)

            if img[i, j] >= 100:
                y=-(i-138.5)
                x=j-153
                X.append(x)
                Y.append(y)

            if img[i, j] >= 150: 
                y=-(i-138.5)
                x=j-153
                X.append(x)
                Y.append(y)

            if img[i, j] >= 200:
                y=-(i-138.5)
                x=j-153
                X.append(x)
                Y.append(y)


    IR=np.vstack((X,Y))
    X=IR[0]
    Y=IR[1]
    Z=np.random.uniform(0, 1,  len(IR[0])) #Since we don't have z axis just put them random for now
    inject = np.vstack([X, Y, Z])
    CR = np.append(CR, inject, axis=1)
                                   
    return CR

def spawn_H(CR, particles=13955, x0=0, y0=0):
    og_img = plt.imread(r'm31_HIproj.png')
    #Load Image in greyscale using OpenCV:
    img = cv2.imread( r'm31_HIproj.png', 0)
    img_dim = img.shape
    
    X, Y = [], []


    for i in range(img_dim[0]):     #select pixels based on intensity, over x,y range of image
        for j in range(img_dim[1]):
            if img[i, j] >= 125:
                y=-(i-(img_dim[0]*0.5)) #Center image
                x=j-(img_dim[1]*0.5)
                X.append(x)
                Y.append(y)

            if img[i, j] >= 175:
                y=-(i-(img_dim[0]*0.5))
                x=j-(img_dim[1]*0.5)
                X.append(x)
                Y.append(y)

            if img[i, j] >= 200:
                y=-(i-(img_dim[0]*0.5))
                x=j-(img_dim[1]*0.5)
                X.append(x)
                Y.append(y)

            if img[i, j] >= 250:
                y=-(i-(img_dim[0]*0.5))
                x=j-(img_dim[1]*0.5)
                X.append(x)
                Y.append(y)
                
    H=np.vstack((X,Y))
    Y=H[1]
    X=H[0]
    Z=np.random.uniform(0, 1,  len(H[0])) #Since we don't have z axis just put them random for now
    inject = np.vstack([X, Y, Z])
    CR = np.append(CR, inject, axis=1)
    
    return CR
    

####################################################### CR Position definitions #######################################

def initial_CR(particles=100, kde=False):
    ''' Sets the initial particles to be injected
        
        Inputs
        --------
        kde: use kde as density (kde==True) or solid color (kde==False)
        
        Outputs
        --------
        CR, CR_esc, and density arrays
        
        TODO: give acess to initial parameters, 
        either through class or function args
    
    '''
    
    
    CR = np.zeros((3,particles))

    # Samples a uniformly distributed Cylinder
    ''' max_z0 = 10
    max_r0 = 15
    phi = np.random.uniform(0,2*np.pi, particles)
    r = rmax*np.sqrt(np.random.uniform(0, 1, particles))
    X0 = r*np.cos(phi)
    Y0 = r*np.sin(phi)
    Z0 = np.random.uniform(-max_z0, max_z0, particles)
    '''
    # uniform sphere
    # CR = np.random.normal(0,spread, (pieces, 3))
    rmax = 15
    phi = np.random.uniform(0, 2*np.pi, particles)
    theta = np.arccos(np.random.uniform(-1, 1, particles))
    r = rmax*pow(np.random.uniform(0, 1, particles), 1./3.)
    X0 = r*np.cos(phi)*np.sin(theta)
    Y0 = r*np.sin(phi)*np.sin(theta)
    Z0 = r*np.cos(theta)
    
    
    CR[0] = X0
    CR[1] = Y0
    CR[2] = Z0

    # For Normal spherical Gaussian
    # spread = 15
    # CR = np.random.normal(0,spread, (pieces, 3))
    
    CR_esc = np.empty((3,0))    
    density = get_color(CR, kde)
    
    return CR, CR_esc, density

def run_step(CR, CR_esc, rstep, zstep):
    ''' perform one step iteration on the CR density
    
        Inputs
        -------
        CR: array of confined particle coordinates
        CR_esc: array of current escaped particles
        rstep,
        zstep: callable functions for the r and z steps respectively
                      currently, these are the maximum values to draw from 
                      a uniform distribution.
        Outputs
        --------
        updated arrays CR, r, z, CR_esc
    
    '''

    r = np.sqrt(CR[0]**2 + CR[1]**2) 
    z = CR[2]
    
    particles = CR.shape[1]
    
    #r_stepsize = rstep(r,z)
    #r_stepsize = rstep(CR[0],CR[1])
    #z_stepsize = zstep(z,r)
    
    #r_step = np.random.uniform(0, r_stepsize, particles)
    #phi = np.random.uniform(0,2*np.pi, particles)
    
    #Xstep = r_step*np.cos(phi)
    #Ystep = r_step*np.sin(phi)
    #Zstep = np.random.uniform(-z_stepsize, z_stepsize, particles)
            #z_stepsize*np.random.choice([-1,1], size=z.shape)
    
    ################### In progress ##############
    r_stepsize = rstep(CR[0],CR[1], CR[2])
    r_step = np.random.uniform(0, r_stepsize, particles)
    phi = np.random.uniform(0,2*np.pi, particles)
    theta = np.arccos(np.random.uniform(-1, 1, particles))
    
    Xstep = r_step*np.cos(phi)*np.sin(theta)
    Ystep = r_step*np.sin(phi)*np.sin(theta)
    Zstep = .5*r_step*np.cos(theta) #.1 is just to make it so the particle diffusion forms a more disk like shape (obviously just a working parameter for now. still looking for good paper describing step size in z direction)
            #z_stepsize*np.random.choice([-1,1], size=z.shape)
    
    ###############################################

    CR[0] += Xstep
    CR[1] += Ystep
    CR[2] += Zstep
    
    r_free = r > 300 #boundary limits
    z_free  = abs(z) > 200
    
    iter_CR_esc = CR.T[np.logical_or(r_free, z_free )].T
    CR = CR.T[np.logical_not(np.logical_or(r_free, z_free))].T

    CR_esc = np.append(CR_esc, iter_CR_esc, axis=1)
    
    r = np.sqrt(CR[0]**2 + CR[1]**2) 
    z = CR[2]
    

    return CR, r, z, CR_esc, r_step

def step_size(x,y, z):
    x0 = 0
    y0 = 0
    z0 = 0
    rmax = 135
    r = np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2) #inefficient diffusion bubble
    rmask = (r<rmax).astype(int) #.astype() turns numbers in to ones and zeros
    
    x1 = -155
    y1 = 165
    z1 = 0
    rmax = 35
    r = np.sqrt((x-x1)**2 + (y-y1)**2 + (z-z1)**2) #inefficient diffusion bubble
    rmask1 = (r<rmax).astype(int) #r<rmax since in return it is being subtracted
    
    x2 = 0
    y2 = 0
    z2 = 0
    rmax2 = 20 
    zmax2  = 10
    r2 = np.sqrt((x-x2)**2 + (y-y2)**2)
    Z2 = abs(z-z2)
    cylindrical_halo = (np.logical_and(r2<rmax2, Z2<zmax2 )).astype(int)
    
    x3 = 0
    y3 = 0
    z3 =0
    rmax3 = 100
    zmax = 1
    r3 = np.sqrt((x-x3)**2 + (y-y3)**2 + (z-z3)**2) #inefficient diffusion bubble
    spherical_halo = (r3<rmax3).astype(int) #r<rmax since in return it is being subtracted
    
    return 100*(1 - .6*(spherical_halo) - 0.36*(cylindrical_halo) -0*rmask - 0*rmask1)

# def save_steps(label='gal_CR', kde=True):
#     ''' Runs the diffusion with default parameters
#         and saves some iterations as png images
        
#         Inputs
#         -------
#         label: string that form the output image file name
#                in the form label + iteration +.png
               
#         kde: whether to use kde as density (kde==True) or solid color (kde==False)
        
#         Outputs
#         -------
#         CR, CR_esc, and density arrays
        
#         TODO: Make it easier to change parameters
        
#     '''
    
#     fig, ax = build3d(grid=True, panel=0.3, boxsize=175)
    
#     rstep = step_size
#     zstep = lambda z,r : 0.5

#     CR, CR_esc, density = initial_CR(particles=0)
#     CR = spawn_ring(CR, particles=1000,rmax=65, rmin=55, x0=0, y0=0, z0=0)
#     CR = spawn_sphere(CR, particles=500,rmax=15, x0=0, y0=0, z0=0)

#     density = get_color(CR, kde=True)
#     ax.set_title('M31 - CR Diffusion', color='white')
#     ax.scatter( CR[0],CR[1],CR[2], zdir='z', c=density,cmap='viridis',s=1, alpha=0.75)
#     ax.scatter( CR_esc[0],CR_esc[1],CR_esc[2], zdir='z', c='r',s=1, alpha=0.5)
#     plt.savefig(label + '0.png')
#     for i in range(0,10001):
#         CR, r,z, CR_esc = run_step(CR, CR_esc, rstep, zstep)
#         if(i%10==0):
#             CR = spawn_ring(CR, particles=10,rmax=65, rmin=55, x0=0, y0=0, z0=0)
#             CR = spawn_sphere(CR, particles=10,rmax=10, x0=0, y0=0, z0=0)
#             CR = spawn_sphere(CR, particles=10, rmax=5, x0=55, y0=55, z0=0, shell=True)


#         if (i%100==0):
#             ax.clear()
#             ax = reset3d(ax, grid=True, panel=0.3, boxsize=175)
#             ax.text(75,-70,-70, 'iter: ' + str(i+1), color='white')
#             ax.set_title("M31 - CR Diffusion", color='white')
            
#             density = get_color(CR, kde)
            
#             ax.scatter( CR[0],CR[1],CR[2], zdir='z', c=density,cmap='viridis',s=1, alpha=0.75)
#             ax.scatter( CR_esc[0],CR_esc[1],CR_esc[2], zdir='z', c='r',s=1, alpha=0.5)
#             plt.savefig(label + str(i+1)+'.png')
        
#     return CR, CR_esc, density


################################################ Column Density ##########################################################

def points_in_cylinder(pt1, pt2, r, q):
    ''''    pt1: the center of your first endpoint in your cylinder (need to input a 3d array)
            pt2: the center of your second endpoint in your cylinder (need to input a 3d array)
            r: radius of your cylinder for your line of sight
            q: this is the 3d point you are checking to see if it is inside the cylinder
            returns: if point is inside the volume it returns the tuple (array([0], dtype=int64),) and if it is outside the 
                     volume it returns (array([], dtype=int64),)'''''
    #math for what's below can be checked on https://math.stackexchange.com/questions/3518495/check-if-a-general-point-is-inside-a-given-cylinder
    
    vec = np.subtract(pt2,pt1)
    const = r * np.linalg.norm(vec)
    return np.where(np.dot(np.subtract(q, pt1), vec) >= 0 and np.dot(np.subtract(q, pt2), vec) <= 0 
                    and np.linalg.norm(np.cross(np.subtract(q, pt1), vec)) <= const)[0] #notice the [0] at the end gives us only the list

def truncated_cone(p0, p1, R0, R1, CR):
    
    v = p1 - p0  # vector in direction of axis that'll give us the height of the cone
    h = np.sqrt(v[0]**2 +v[1]**2 + v[2]**2) #height of cone
    mag = norm(v) # find magnitude of vector
    v = v / mag  # unit vector in direction of axis
    not_v = np.array([1, 1, 0]) # make some vector not in the same direction as v
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])
    n1 = np.cross(v, not_v) # make vector perpendicular to v
    # print n1,'\t',norm(n1)
    n1 /= norm(n1)# normalize n1
    n2 = np.cross(v, n1) # make unit vector perpendicular to v and n1
    
    # surface ranges over t from 0 to length of axis and 0 to 2*pi
    n = 80
    t = np.linspace(0, mag, n)
    theta = np.linspace(0, 2 * np.pi, n)
    # use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    R = np.linspace(R0, R1, n)
    # generate coordinates for surface
    X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    
    z = CR[2]
    permRadius = R1 - (R1 - R0) * (z /h) #boundary of cone
    pointRadius = (np.sqrt(np.add(np.subtract(CR[0], p0[0])**2, np.subtract(CR[1], p0[1])**2)))
#     print(permRadius)
#     print(pointRadius)
#     print(z, h)
    param1 = np.logical_and(z <= h, z >= p1[2]) #note: if cone is facing down on the origin use p1[0], if facing up, use p0[0]
    params = (np.where(np.logical_and(param1, pointRadius<= permRadius), True, False)) #checks to see if it satisfies both the radius and z parameters
#     print(param1)
#     print(params)
    n_CR = sum(params.astype(int)) #Total number of particles in cone
    
    return n_CR, X,Y,Z




#--------------------------------FIT FUNCTIONS-----------------------------------------------------------------------------            
def model_func(t, N_0, t_0): #N_0 is amplitude, t is number of steps, and t_0 is a function of step size and boundary size
    return N_0 * np.exp(-1*t/t_0)

def fit_exp_nonlinear(t, y):
    opt_parms, parm_cov = sp.optimize.curve_fit(model_func, t, y, p0=[initial, t00], maxfev=initial)
    N_0, t_0 = opt_parms
    return N_0, t_0
################################################ Application #############################################################

nsteps =1000000
rstep = step_size #see return from step_size def (this line though is only swapping the title of the def)
zstep = lambda z,r: 1  #not used in spherical coordinate case

CR, CR_esc, density = initial_CR(particles=0)
CR = spawn_sphere(CR, particles=3000, rmax=5.5, x0=0, y0=0, z0=0, shell=False)
# CR = spawn_sphere_ring(CR, particles = 6666, rmax=117, rmin=5.5, x0=0, y0=0, z0=0, shell=False)
# CR = spawn_ring(CR, particles=3333, rmax=200, rmin=117, x0=0, y0=0, z0=0, shell=False)
# CR = spawn_IR(CR)
# CR = spawn_H(CR)
initial = CR.shape[1]

# audrey's code
particle_array = []
t_start = []
left_bound = .9*initial
right_bound = 0.10*initial
tt=[]
cr=[]
n00 = []
too = []
for n in range(0,nsteps):
    CR, r,z, CR_esc, r_step = run_step(CR, CR_esc, rstep, zstep)
    t = n #edit this if you want to say something like 1 step = 1/3 seconds ie set t = t/3
    tt.append(t)
    cr.append(CR.shape[1])
    if CR.shape[1]<= left_bound:
        particle_array = np.append(CR.shape[1], particle_array)
        t_start = np.append(t, t_start)
    if CR.shape[1]<=right_bound:
#         t_start = np.append(t, t_start)
#         particle_array = np.append(CR.shape[1], particle_array)
        break
# for if (t%#1 == #2):, #1 is every number of steps when the if statement activates, #2 is the initial step. 
#     if(t%100==99):
#         CR = spawn_ring(CR, particles=0,rmax=65, rmin=55, x0=0, y0=0, z0=0)
#         CR = spawn_sphere(CR, particles=0, rmax=15, x0=0, y0=0, z0=0)
#         CR = spawn_sphere(CR, particles=8, rmax=15, x0=55, y0=55, z0=0, shell=True)
    if CR.shape[1]<=(np.exp(-1))*initial:   #number of steps to transient
        tt0 = -t/(np.log(CR.shape[1]/initial))
        n00.append(CR.shape[1])
        too.append(t)
#         rate=(initial)*np.exp(-t/t0)
# #         N0, t0 = fit_exp_nonlinear(t, particle_array[left_index:right_index+1]) #gives optimized variables
#         print("Analytical t0 =", t0)
#         print("Number of Seconds to get to Steady State =" , t+1, "seconds")
#         break
#     if CR.shape[1] == 0:
#         break

particle_array = np.flip(np.array(particle_array))
t_start = (np.flip(np.array(t_start))).astype(int)
ttt=t_start
t_start = np.subtract(t_start, t_start[0])
left_index = np.where(particle_array== max(particle_array))[0][0]
right_index = np.where(particle_array== min(particle_array))[0][0]
bound_range = right_index - left_index + 1
t00 =-too[0]/(np.log(n00[0]/initial))

t = np.linspace(left_index, right_index, bound_range)
N_0, t_0 = fit_exp_nonlinear(t, particle_array[left_index:right_index+1]) #gives optimized variables
fit_y = model_func(t, N_0, t_0) #Note: t must have more than 100 steps to give accurate output



# # inside = []   
# # N = 0 
# # r_cylinder = 100 #radius of desired cross section
# # L = 1500 #how far out you want to see
# # V_cylinder = np.pi*(r_cylinder**2)*L
# # for i in range(0, len(CR[0])):
# #     in_or_out = points_in_cylinder(np.array([-750,0,0]),np.array([750,0,0]),r_cylinder,
# #                             np.array([CR[0][i],CR[1][i],CR[2][i]]))
# #     if np.size(in_or_out)==1: #if its one it means its inside, zero if it's outside
# #         inside.append(np.size(in_or_out))
# #         N += 1 #number of particles inside the cross section
# # n_CR = N/V_cylinder #column density

# # print('Average Column Density of Cosmic Rays:', n_CR, 'particles/kpc^3')

# A0 = np.array([0, 0, 750])
# A1 = np.array([0, 0, 0])
# n_CR, X,Y,Z = truncated_cone(A0, A1, 0, 200, CR)
# print(n_CR,'particles in cone')
 
get_ipython().run_line_magic('matplotlib', 'inline')


plt.figure(1)
ax1 = plt.subplot(111)
#plot the remaining particles vs. time
ax1.plot(t, particle_array[left_index:right_index+1], linewidth=2.0)
# plt.plot(tt, cr)

#plot fitted line
ax1.plot(t, fit_y, color='orange', 
        label='Fitted Function:\n $y = %0.2f e^{-t/%0.2f}$' % (N_0, t_0), linewidth=3.0)
# plt.plot(t+ttt[0], fit_y, color='orange', label='Fitted Function:\n $y = %0.2f e^{-t/%0.2f}$' % (N_0, t_0), linewidth=3.0)
#add plot lables
left_percentage = left_bound/initial * 100
right_percentage = right_bound/initial * 100
plt.title('%.f to %.f percent of remaining particles' % (left_percentage, right_percentage))
plt.ylabel('number of particles')
plt.xlabel('time-step')
plt.legend(loc='best')
plt.show()

############################################# transient run ############################################################
                

CR, CR_esc, density = initial_CR(particles=0)
CR = spawn_sphere(CR, particles=10**4, rmax=5.5, x0=0, y0=0, z0=0, shell=False)
# CR = spawn_sphere_ring(CR, particles = 6666, rmax=117, rmin=5.5, x0=0, y0=0, z0=0, shell=False)
# CR = spawn_ring(CR, particles=3333, rmax=200, rmin=117, x0=0, y0=0, z0=0, shell=False)
# CR = spawn_IR(CR)
# CR = spawn_H(CR)
initial = CR.shape[1]

for n in range(0,nsteps): 
    t = n #number of steps in one second
    CR, r,z, CR_esc, r_step = run_step(CR, CR_esc, rstep, zstep)
    t = t #edit this if you want to say something like 1 step = 1/3 seconds ie set t = t/3
# for if (t%#1 == #2):, #1 is every number of steps when the if statement activates, #2 is the initial step. 
#     if(t%100==99):
#         CR = spawn_ring(CR, particles=0,rmax=65, rmin=55, x0=0, y0=0, z0=0)
#         CR = spawn_sphere(CR, particles=0, rmax=15, x0=0, y0=0, z0=0)
#         CR = spawn_sphere(CR, particles=8, rmax=15, x0=55, y0=55, z0=0, shell=True)
    if t==round(t_0+ttt[0]):   #number of steps to transient
        print("t0 =", t_0 + ttt[0])
        break

# [(print("t0 =", -t/(np.log(CR.shape[1]/initial))),break) for t in range(0,nsteps) if (CR.shape[1]<=(np.exp(-1))*initial)]
# [((CR, r,z, CR_esc, r_step == run_step(CR, CR_esc, rstep, zstep)),print("t0 =", t_0 + ttt[0])) for t in range(0,round(t_0 + ttt[0]))]
        
escaped = CR_esc.shape[1]
print('Initial Particles that Escaped Fraction = {:.3f}% or {:} total'.format( (escaped/initial)*100, escaped))
print('Particles Remaining:', CR.shape[1])
print('total particles', CR.shape[1]+CR_esc.shape[1])


 
get_ipython().run_line_magic('matplotlib', 'auto')


density = get_color(CR, kde=True)    
fig, ax = build3d(num=2, grid=True, panel=0.5, boxsize=350)
ax.set_title('M31 - CR Diffusion', color='white')

ax.scatter( CR[0],CR[1],CR[2], zdir='z', c=density,cmap='viridis',s=1, alpha=0.75)
# ax.scatter( CR_esc[0],CR_esc[1],CR_esc[2], zdir='z', c='r',s=1, alpha=0.5)

plt.show()

# # inside = []   
# # N = 0 
# # r_cylinder = 100 #radius of desired cross section
# # L = 1500 #how far out you want to see
# # V_cylinder = np.pi*(r_cylinder**2)*L
# # for i in range(0, len(CR[0])):
# #     in_or_out = points_in_cylinder(np.array([-750,0,0]),np.array([750,0,0]),r_cylinder,
# #                             np.array([CR[0][i],CR[1][i],CR[2][i]]))
# #     if np.size(in_or_out)==1: #if its one it means its inside, zero if it's outside
# #         inside.append(np.size(in_or_out))
# #         N += 1 #number of particles inside the cross section
# # n_CR = N/V_cylinder #column density

# # print('Average Column Density of Cosmic Rays:', n_CR, 'particles/kpc^3')

# A0 = np.array([0, 0, 750])
# A1 = np.array([0, 0, 0])
# n_CR, X,Y,Z = truncated_cone(A0, A1, 0, 200, CR)
# print(n_CR,'particles in cone')

# fig = plt.figure(2)
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.plot_surface(X, Y, Z, color='b', linewidth=0, antialiased=False)
# ax.scatter(CR[0],CR[1],CR[2], color='r')


n_toc = time.process_time() 
print("\nComputation time = "+str((n_toc - n_tic ))+"seconds") 


# In[ ]:


x=np.linspace(0,5,10)
y=np.linspace(0,5,10)
def pp(x,y):
    return(x+y)

def get_weather_data():
     return np.random.randrange(90, 110)
hot_temps = [temp for _ in range(20) if (temp := get_weather_data()) >= 100]
hot_temps


# In[135]:


nsteps =1000000
rstep = step_size #see return from step_size def (this line though is only swapping the title of the def)
# zstep = lambda z,r: 1  #not used in spherical coordinate case
zstep = .2

CR, CR_esc, density = initial_CR(particles=0)
CR = spawn_sphere(CR, particles=1000, rmax=5.5, x0=0, y0=0, z0=0, shell=False)
# CR = spawn_sphere_ring(CR, particles = 6666, rmax=117, rmin=5.5, x0=0, y0=0, z0=0, shell=False)
# CR = spawn_ring(CR, particles=3333, rmax=200, rmin=117, x0=0, y0=0, z0=0, shell=False)
# CR = spawn_IR(CR)
# CR = spawn_H(CR)
initial = CR.shape[1]

for n in range(0,nsteps): 
    t = n #number of steps in one second
    CR, r,z, CR_esc, r_step = run_step(CR, CR_esc, rstep, zstep)
    t = t #edit this if you want to say something like 1 step = 1/3 seconds ie set t = t/3
# for if (t%#1 == #2):, #1 is every number of steps when the if statement activates, #2 is the initial step. 
#     if(t%100==99):
#         CR = spawn_ring(CR, particles=0,rmax=65, rmin=55, x0=0, y0=0, z0=0)
#         CR = spawn_sphere(CR, particles=0, rmax=15, x0=0, y0=0, z0=0)
#         CR = spawn_sphere(CR, particles=8, rmax=15, x0=55, y0=55, z0=0, shell=True)
    if t==round(t_0+ttt[0]):   #number of steps to transient
        print("t0 =", t_0 + ttt[0])
        break


escaped = CR_esc.shape[1]
print('Initial Particles that Escaped Fraction = {:.3f}% or {:} total'.format( (escaped/initial)*100, escaped))
print('Particles Remaining:', CR.shape[1])
print('total particles', CR.shape[1]+CR_esc.shape[1])
 
get_ipython().run_line_magic('matplotlib', 'auto')

density = get_color(CR, kde=True)    
fig, ax = build3d(grid=True, panel=0.5, boxsize=350)
ax.set_title('M31 - CR Diffusion', color='white')

ax.scatter( CR[0],CR[1],CR[2], zdir='z', c=density,cmap='viridis',s=1, alpha=0.75)
ax.scatter( CR_esc[0],CR_esc[1],CR_esc[2], zdir='z', c='r',s=1, alpha=0.5)

plt.show()

# # inside = []   
# # N = 0 
# # r_cylinder = 100 #radius of desired cross section
# # L = 1500 #how far out you want to see
# # V_cylinder = np.pi*(r_cylinder**2)*L
# # for i in range(0, len(CR[0])):
# #     in_or_out = points_in_cylinder(np.array([-750,0,0]),np.array([750,0,0]),r_cylinder,
# #                             np.array([CR[0][i],CR[1][i],CR[2][i]]))
# #     if np.size(in_or_out)==1: #if its one it means its inside, zero if it's outside
# #         inside.append(np.size(in_or_out))
# #         N += 1 #number of particles inside the cross section
# # n_CR = N/V_cylinder #column density

# # print('Average Column Density of Cosmic Rays:', n_CR, 'particles/kpc^3')

# A0 = np.array([0, 0, 750])
# A1 = np.array([0, 0, 0])
# n_CR, X,Y,Z = truncated_cone(A0, A1, 0, 200, CR)
# print(n_CR,'particles in cone')

# fig = plt.figure(2)
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.plot_surface(X, Y, Z, color='b', linewidth=0, antialiased=False)
# ax.scatter(CR[0],CR[1],CR[2], color='r')


#---------------------PRINT VALUES-----------------------------------------------------------------------------------
print ('N_0 = ', N_0)
print ('t_0 = ', t_0)
slope, intercept, r_value, p_value, std_err = stats.linregress(fit_y, particle_array[left_index:right_index+1])
print ("r-squared: ", r_value**2)


n_toc = time.process_time() 
print("\nComputation time = "+str((n_toc - n_tic ))+"seconds")  


# In[120]:


round(6.7)


# In[322]:


nsteps =1000000
rstep = step_size #see return from step_size def (this line though is only swapping the title of the def)
# zstep = lambda z,r : 0.02 #not used in spherical coordinate case

CR, CR_esc, density = initial_CR(particles=0)
# CR = spawn_sphere(CR, particles=1000, rmax=5.5, x0=0, y0=0, z0=0, shell=False)
CR = spawn_sphere_ring(CR, particles = 6666, rmax=117, rmin=5.5, x0=0, y0=0, z0=0, shell=False)
# CR = spawn_ring(CR, particles=3333, rmax=200, rmin=117, x0=0, y0=0, z0=0, shell=False)
# CR = spawn_IR(CR)
# CR = spawn_H(CR)
initial = CR.shape[1]
print('initial', initial)

particle_array = np.array(initial)
for n in range(0,nsteps): 
    t = n #number of steps in one second
    CR, r,z, CR_esc, r_step, particle_array = run_step(CR, CR_esc, rstep, zstep)
    t = t #edit this if you want to say something like 1 step = 1/3 seconds ie set t = t/3
    particle_array = np.append(CR.shape[1], particle_array)
    
# for if (t%#1 == #2):, #1 is every number of steps when the if statement activates, #2 is the initial step. 
#     if(t%100==99):
#         CR = spawn_ring(CR, particles=0,rmax=65, rmin=55, x0=0, y0=0, z0=0)
#         CR = spawn_sphere(CR, particles=0, rmax=15, x0=0, y0=0, z0=0)
# #         CR = spawn_sphere(CR, particles=8, rmax=15, x0=55, y0=55, z0=0, shell=True)
    if CR.shape[1]<=(np.exp(-1))*initial:   #number of steps to transient
        print(CR.shape[1], (np.exp(-1))*initial)
        t0 = -t/(np.log(CR.shape[1]/initial))
#         N0, t0 = fit_exp_nonlinear(t, particle_array[left_index:right_index+1]) #gives optimized variables
        print("Analytical t0 =", t0)
        print("Number of Seconds to get to Steady State =" , t+1, "seconds")
        break
particle_array = np.flip(particle_array[:-1])


escaped = CR_esc.shape[1]
print('Initial Particles that Escaped Fraction = {:.3f}% or {:} total'.format( (escaped/initial)*100, escaped))
print('Particles Remaining:', CR.shape[1])
print('total particles', CR.shape[1]+CR_esc.shape[1])


# # inside = []   
# # N = 0 
# # r_cylinder = 100 #radius of desired cross section
# # L = 1500 #how far out you want to see
# # V_cylinder = np.pi*(r_cylinder**2)*L
# # for i in range(0, len(CR[0])):
# #     in_or_out = points_in_cylinder(np.array([-750,0,0]),np.array([750,0,0]),r_cylinder,
# #                             np.array([CR[0][i],CR[1][i],CR[2][i]]))
# #     if np.size(in_or_out)==1: #if its one it means its inside, zero if it's outside
# #         inside.append(np.size(in_or_out))
# #         N += 1 #number of particles inside the cross section
# # n_CR = N/V_cylinder #column density

# # print('Column Density of Cosmic Rays:', n_CR, 'particles/kpc^3')


get_ipython().run_line_magic('matplotlib', 'auto')

density = get_color(CR, kde=True)    
fig, ax = build3d(grid=True, panel=0.5, boxsize=350)
ax.set_title('M31 - CR Diffusion', color='white')

ax.scatter( CR[0],CR[1],CR[2], zdir='z', c=density,cmap='viridis',s=1, alpha=0.75)
ax.scatter( CR_esc[0],CR_esc[1],CR_esc[2], zdir='z', c='r',s=1, alpha=0.5)

plt.show()

t = np.linspace(0,t, t+1)

plt.figure(2)
plt.plot(t, particle_array)
plt.plot(t, .9*initial*np.exp(-t/t0), c = 'orange')


# In[57]:


def initial_e(particles=100, kde=False):
    ''' Sets the initial particles to be injected
        
        Inputs
        --------
        kde: use kde as density (kde==True) or solid color (kde==False)
        
        Outputs
        --------
        CR, CR_esc, and density arrays
        
        TODO: give acess to initial parameters, 
        either through class or function args
    
    '''
    
    
    e = np.zeros((3,particles))

    # Samples a uniformly distributed Cylinder
    ''' max_z0 = 10
    max_r0 = 15
    phi = np.random.uniform(0,2*np.pi, particles)
    r = rmax*np.sqrt(np.random.uniform(0, 1, particles))
    X0 = r*np.cos(phi)
    Y0 = r*np.sin(phi)
    Z0 = np.random.uniform(-max_z0, max_z0, particles)
    '''
    # uniform sphere
    # CR = np.random.normal(0,spread, (pieces, 3))
    rmax = 15
    phi = np.random.uniform(0, 2*np.pi, particles)
    theta = np.arccos(np.random.uniform(-1, 1, particles))
    r = rmax*pow(np.random.uniform(0, 1, particles), 1./3.)
    X0 = r*np.cos(phi)*np.sin(theta)
    Y0 = r*np.sin(phi)*np.sin(theta)
    Z0 = r*np.cos(theta)
    
    
    e[0] = X0
    e[1] = Y0
    e[2] = Z0

    # For Normal spherical Gaussian
    # spread = 15
    # CR = np.random.normal(0,spread, (pieces, 3))
    
    e_esc = np.empty((3,0))    
    density = get_color(e, kde)
    
    return e, e_esc, density

def e_run_step(e, e_esc, rstep, zstep):

    r = np.sqrt(e[0]**2 + e[1]**2) 
    z = e[2]
    
    particles = e.shape[1]
    
    r_stepsize = rstep(e[0],e[1], e[2])
    r_step = np.random.uniform(0, r_stepsize, particles)
    phi = np.random.uniform(0,2*np.pi, particles)
    theta = np.arccos(np.random.uniform(-1, 1, particles))
    
    Xstep = r_step*np.cos(phi)*np.sin(theta)
    Ystep = r_step*np.sin(phi)*np.sin(theta)
    Zstep = r_step*np.cos(theta) #.4 is just to make it so the particle diffusion forms a more disk like shape (obviously just a working parameter for now. still looking for good paper describing step size in z direction)
            #z_stepsize*np.random.choice([-1,1], size=z.shape)

    e[0] += Xstep
    e[1] += Ystep
    e[2] += Zstep
    
    r_free = r > 1500 #boundary limits
    z_free  = abs(z) > 150
    
#     iter_e_esc = e.T[np.logical_or(r_free, z_free )].T
#     e = e.T[np.logical_not(np.logical_or(r_free, z_free))].T

#     e_esc = np.append(e_esc, iter_e_esc, axis=1)
    
    r = np.sqrt(e[0]**2 + e[1]**2) 
    z = e[2]

    return e, r, z, r_step



nsteps =10000
rstep = step_size #see return from step_size def (this line though is only swapping the title of the def)
zstep = lambda z,r : 0.5 #not used in spherical coordinate case

e, e_esc, density = initial_e(particles=0)
e = spawn_sphere(e, particles=100, rmax=5.5, x0=0, y0=0, z0=0, shell=False)
e_initial = e.shape[1]
e_initial_NRG = np.add(np.zeros(e_initial), 1000)


c = 9.72*10**(-12) #speed of light in kpc/s
b = 10**(-16) #GeV^-1*sec^-1

for i in range(0,nsteps+1):
    t = i
    e, r,z, r_step = e_run_step(e, e_esc, rstep, zstep)
    e_final = e_initial_NRG/(1+(b/c)*r_step*e_initial_NRG)  #energy loss equation for electorns
    e_initial_NRG = e_final #converts efinal back to enitial for the next run in the loop
    e_min = e_final < 1
    e_stop = e_final.T[np.logical_not(e_min)].T
    if e_stop.shape[0]<=(1/np.exp(1))*e_initial_NRG.shape[0]:  #just setting this condition for now so that the simulation doesn't take too long. will delete later once able to track when the electorns leave
        print("Number of steps to get to Steady State =" , t+1, "seconds")
        break 
        

        
print(e_stop.shape[0])
print(e_final)


density = get_color(CR, kde=True)    
fig, ax = build3d(grid=True, panel=0.5, boxsize=5500)
ax.set_title('M31 - Electron Diffusion', color='white')

ax.scatter( e[0],e[1],e[2], zdir='z', cmap='viridis',s=1, alpha=0.75)


# In[201]:


w = np.array([[1,2,3,4,5],[6,7,8,9,10],[13,7,8,68,47]])
w = np.array([1,2,3,4,5,6,7,8,9,10,13,7,8,68,47])

# r = np.sqrt(w[0]**2 + w[1]**2) 
# z = w[2]
r_free = w > 30 #boundary limits
# z_free  = abs(z) > 30

# iter_w_esc = [np.logical_or(r_free)]
w = w.T[np.logical_not((r_free))].T
# print(iter_w_esc)
print(w)


# In[ ]:





# In[170]:


######## Imports

import matplotlib.pyplot as plt
import matplotlib.image as mpim
import cv2
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
og_img = plt.imread(r'm31_24proj.png')
img = cv2.imread( r'm31_24proj.png', 0)

###### Histogram
plt.imshow(og_img)
plt.title('Original IR Projection')
plt.show()


plt.imshow(img, extent=[0,250,0,250])
plt.title('Intensity of M31 IR Image')
plt.colorbar()
plt.show()

imhist = cv2.calcHist([img],[0],None,[256],[0,256])

plt.hist(img.ravel(), 256, [0, 256])
plt.title('Pixel Frequency vs Intensity')
plt.xlabel('Pixel Intensity [0,250]')
plt.ylabel('Frequency')
plt.show()

####### Record pixels above certain intensity

xpix, ypix   = [], []  #initialize arrays
xpix1, ypix1 = [], []
xpix2, ypix2 = [], []
xpix3, ypix3 = [], []
X,Y = [], []

for i in range(277):          #select pixels based on intensity, over range of image
    for j in range(306):
        if img[i, j] >= 75:
            y=-(i-138.5) #center coordinates
            x=j-153
            xpix.append(x)
            ypix.append(y)
            X.append(x)
            Y.append(y)
            
        if img[i, j] >= 100:
            y=-(i-138.5)
            x=j-153
            xpix1.append(x)
            ypix1.append(y)
            X.append(x)
            Y.append(y)
            
        if img[i, j] >= 150: 
            y=-(i-138.5)
            x=j-153
            xpix2.append(x)
            ypix2.append(y)
            X.append(x)
            Y.append(y)
            
        if img[i, j] >= 200:
            y=-(i-138.5)
            x=j-153
            xpix3.append(x)
            ypix3.append(y)
            X.append(x)
            Y.append(y)
            
# for i in range(277):          #select pixels based on intensity, over range of image
#     for j in range(306):
#         if img[i, j] >= 75:
#             y=-(i-138.5)
#             x=j-153
#             X.append(x)
#             Y.append(y)
            


#print('x:', xpix, 'y:', ypix)
pixc  = np.vstack((xpix, ypix))     #format x,y into single coordinate array
pixc1 = np.vstack((xpix1, ypix1))
pixc2 = np.vstack((xpix2, ypix2))
pixc3 = np.vstack((xpix3, ypix3))


IR=np.vstack((X,Y))
Y=IR[1]
X=IR[0]
print(len(X))
print(min(X), min(Y), max(X), max(Y))

plt.scatter(pixc[0], pixc[1], marker='+', label='intensity >= 75')
plt.scatter(pixc1[0], pixc1[1], marker='+', label='intensity >= 100')
plt.scatter(pixc2[0], pixc2[1], marker='+', label='intensity >= 150')
plt.scatter(pixc3[0], pixc3[1], marker='+', label='intensity >= 200')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
plt.title('Pixel Coordinates by Intensity')
plt.show()

plt.figure(1)
plt.scatter(X, Y, marker='+', s=1)


# In[169]:


import matplotlib.pyplot as plt
import matplotlib.image as mpim
import cv2
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
#Load Original image:
og_img = plt.imread(r'm31_HIproj.png')

#Load Image in greyscale using OpenCV:
img = cv2.imread( r'm31_HIproj.png', 0)
cv2.imwrite(r'C:\Users\conma\OneDrive\Documents\Research\M31 Image Processing\m31_HIproj_greyscale.png', img)

img_dim = img.shape
####### Histogram
plt.imshow(og_img)
plt.title('Original HI Projection')
plt.axis('off')
plt.show()


#plt.imshow(img, extent=[0,img.shape[0],0,img.shape[1]])
#plt.title('Intensity of M31 HI Image')
#plt.xlabel('$x$')
#plt.ylabel('$y$')
#plt.colorbar()
#plt.show()

imhist = cv2.calcHist([img],[0],None,[256],[0,256])

#plt.hist(img.ravel(), 256, [0, 256])
#plt.title('Pixel Frequency vs Intensity')
#plt.xlabel('Pixel Intensity [0,255]')
#plt.ylabel('Frequency')
#plt.show()

####### Record pixels above certain intensity

xpix,  ypix  = [], []  #initialize arrays
xpix1, ypix1 = [], []
xpix2, ypix2 = [], []
xpix3, ypix3 = [], []
X, Y = [], []


for i in range(img_dim[0]):     #select pixels based on intensity, over x,y range of image
    for j in range(img_dim[1]):
        if img[i, j] >= 125:
            y=-(i-(img_dim[0]*0.5)) #Center image
            x=j-(img_dim[1]*0.5)
            xpix.append(x)
            ypix.append(y)
            X.append(x)
            Y.append(y)

        if img[i, j] >= 175:
            y=-(i-(img_dim[0]*0.5))
            x=j-(img_dim[1]*0.5)
            xpix1.append(x)
            ypix1.append(y)
            X.append(x)
            Y.append(y)

        if img[i, j] >= 200:
            y=-(i-(img_dim[0]*0.5))
            x=j-(img_dim[1]*0.5)
            xpix2.append(x)
            ypix2.append(y)
            X.append(x)
            Y.append(y)

        if img[i, j] >= 250:
            y=-(i-(img_dim[0]*0.5))
            x=j-(img_dim[1]*0.5)
            xpix3.append(x)
            ypix3.append(y)
            X.append(x)
            Y.append(y)

pixc  = np.vstack((xpix, ypix))     #format x,y into array of coordinates
pixc1 = np.vstack((xpix1, ypix1))
pixc2 = np.vstack((xpix2, ypix2))
pixc3 = np.vstack((xpix3, ypix3))

H=np.vstack((X,Y))
Y=H[1]
X=H[0]
print(len(X))

#plot coordinates. Note: image is read from top to bottom, so invert_yaxis is used to orient image
plt.scatter( pixc[0],  pixc[1], marker='+', label=r'Intensity $\geq$ 125')
plt.scatter(pixc1[0], pixc1[1], marker='+', label=r'Intensity $\geq$ 175')
plt.scatter(pixc2[0], pixc2[1], marker='+', label=r'Intensity $\geq$ 200')
plt.scatter(pixc3[0], pixc3[1], marker='+', label=r'Intensity $\geq$ 250')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
plt.title('M31 HI Intensity')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.savefig('m31_HIMAP.png')
plt.show()

plt.figure(1)
plt.scatter(X, Y, marker='+', s=1)


# In[6]:


e_initial=np.add(np.zeros(63), 1000)
x = []
c = 9.72*10**(-12) #speed of light in kpc/s
b = 10**(-16) #GeV^-1*sec^-1
for i in range(0,len(e_initial+1)):
    CR, r,z, CR_esc, r_step = run_step(CR, CR_esc, rstep, zstep)
    print(r_step)
    e_final = e_initial/(1+(b/c)*r_stepsize*e_initial)
    e_initial = e_final
print(e_final)


# e_free = e_final < 1 #boundary limits

# iter_CR_esc = CR.T[np.logical_or(r_free, z_free )].T
# CR = CR.T[np.logical_not(np.logical_or(r_free, z_free))].T

# CR_esc = np.append(CR_esc, iter_CR_esc, axis=1)


# In[130]:


a=[1,2,3,4]
b=[22,3,4,5]
np.multiply(a,b)
print(rstep)


# In[ ]:





# In[ ]:





# In[114]:


a =[[10,1,2,13,24,45,69,9,30],[10,1,2,13,24,45,69,9,30],[10,1,2,13,24,45,69,9,30]]
def points_in_cylinder(pt1, pt2, r, q):
    
    vec = np.subtract(pt2,pt1)
    const = r * np.linalg.norm(vec)
    return np.where(np.dot(np.subtract(q, pt1), vec) >= 0 and np.dot(np.subtract(q, pt2), vec) <= 0 
                    and np.linalg.norm(np.cross(np.subtract(q, pt1), vec)) <= const)
b = points_in_cylinder(np.array([0,0,0]),np.array([10,0,0]),3,np.array([69,69,69]))
print(b)
h = []
g = 0
for i in range(0, len(a[0])):
    c = points_in_cylinder(np.array([0,0,0]),np.array([100,0,0]),3,np.array([a[0][i],a[1][i],a[2][i]]))[0]
    h.append(np.size(c))
    if np.size(c)==1:
        g += 1
print(h)
print(g)


# In[87]:


x = np.zeros(1)
if x.size: 
    print("x")
else:
    print("No x")


# In[69]:


x= np.array([0,0,0], dtype = np.object)
print(x)


# In[32]:


def points_in_cylinder(pt1, pt2, r, q):
    ''''    pt1: the center of your first endpoint in your cylinder (need to input a 3d array)
            pt2: the center of your second endpoint in your cylinder (need to input a 3d array)
            r: radius of your cylinder for your line of sight
            q: this is the 3d point you are checking to see if it is inside the cylinder
            returns: if point is inside the volume it returns the tuple (array([0], dtype=int64),) and if it is outside the 
                     volume it returns (array([], dtype=int64),)'''''
    
    #math for what's below can be checked on https://math.stackexchange.com/questions/3518495/check-if-a-general-point-is-inside-a-given-cylinder
    
    vec = np.subtract(pt2,pt1)  
    const = r * np.linalg.norm(vec)
    return np.where(np.dot(np.subtract(q, pt1), vec) >= 0 and np.dot(np.subtract(q, pt2), vec) <= 0 
                    and np.linalg.norm(np.cross(np.subtract(q, pt1), vec)) <= const)[0] #notice the [0] at the end gives us only the list

inside = []   
N = 0 
r_cylinder = 100 #radius of desired cross section
L = 1500 #how far out you want to see
V_cylinder = np.pi*(r_cylinder**2)*L
for i in range(0, len(CR[0])):
    in_or_out = points_in_cylinder(np.array([-750,0,0]),np.array([750,0,0]),r_cylinder,
                            np.array([CR[0][i],CR[1][i],CR[2][i]]))
    if np.size(in_or_out)==1: #if its one it means its inside, zero if it's outside
        inside.append(np.size(in_or_out))
        N += 1 #number of particles inside the cross section
n_CR = N/V_cylinder #column density

print('Column Density of Cosmic Rays:', n_CR, 'particles/kpc^3')


# In[22]:


nsteps =10000000
rstep = step_size #see return from step_size def (this line though is only swapping the title of the def)
zstep = lambda z,r : 0.5 #not used in spherical coordinate case

CR, CR_esc, density = initial_CR(particles=0)
# CR = spawn_sphere(CR, particles=100, rmax=5.5, x0=0, y0=0, z0=0, shell=False)
# CR = spawn_sphere_ring(CR, particles = 6666, rmax=117, rmin=5.5, x0=0, y0=0, z0=0, shell=False)
# CR = spawn_ring(CR, particles=3333, rmax=200, rmin=117, x0=0, y0=0, z0=0, shell=False)
# CR = spawn_IR(CR)
CR = spawn_H(CR)
initial = CR.shape[1]

# e = spawn_sphere(e, particles=1000, rmax=5.5, x0=0, y0=0, z0=0, shell=False)
# e_initial = e.shape[1]
# e_initial_NRG = np.add(np.zeros(e_initial), 1000)
# b = 10**(-16) #GeV^-1*sec^-1

for n in range(0,nsteps): 
    t = n #number of steps in one second
    CR, r,z, CR_esc, r_step, particle_array = run_step(CR, CR_esc, rstep, zstep)
#     e, r,z, e_esc = run_step(e, e_esc, rstep, zstep)
    t = t #edit this if you want to say something like 1 step = 1/3 seconds ie set t = t/3
#     e_final = e_initial/(1+b*t*e_initial)
# for if (t%#1 == #2):, #1 is every number of steps when the if statement activates, #2 is the initial step. 
#     if(t%100==99):
#         CR = spawn_ring(CR, particles=0,rmax=65, rmin=55, x0=0, y0=0, z0=0)
#         CR = spawn_sphere(CR, particles=0, rmax=15, x0=0, y0=0, z0=0)
# #         CR = spawn_sphere(CR, particles=8, rmax=15, x0=55, y0=55, z0=0, shell=True)
    if CR.shape[1]<=(np.exp(-1))*initial:   #number of steps to transient
        t0 = -t/(np.log(CR.shape[1]/initial))
        rate=(initial)*np.exp(-t/t0)
#         N0, t0 = fit_exp_nonlinear(t, particle_array[left_index:right_index+1]) #gives optimized variables
        print("Analytical t0 =", t0)
        print("Number of Seconds to get to Steady State =" , t+1, "seconds")
        break
        

escaped = CR_esc.shape[1]
print('Initial Particles that Escaped Fraction = {:.3f}% or {:} total'.format( (escaped/initial)*100, escaped))
print('total particles', CR.shape[1]+CR_esc.shape[1])

# inside = []   
# N = 0 
# r_cylinder = 100 #radius of desired cross section
# L = 1500 #how far out you want to see
# V_cylinder = np.pi*(r_cylinder**2)*L
# for i in range(0, len(CR[0])):
#     in_or_out = points_in_cylinder(np.array([-750,0,0]),np.array([750,0,0]),r_cylinder,
#                             np.array([CR[0][i],CR[1][i],CR[2][i]]))
#     if np.size(in_or_out)==1: #if its one it means its inside, zero if it's outside
#         inside.append(np.size(in_or_out))
#         N += 1 #number of particles inside the cross section
# n_CR = N/V_cylinder #column density

# print('Column Density of Cosmic Rays:', n_CR, 'particles/kpc^3')
 
get_ipython().run_line_magic('matplotlib', 'auto')
 
density = get_color(CR, kde=True)    
fig, ax = build3d(grid=True, panel=0.5, boxsize=450)
ax.set_title('M31 - CR Diffusion', color='white')

ax.scatter( CR[0],CR[1],CR[2], zdir='z', c=density,cmap='viridis',s=1, alpha=0.75)
ax.scatter( CR_esc[0],CR_esc[1],CR_esc[2], zdir='z', c='r',s=1, alpha=0.5)

plt.show

# plt.figure(figsize = (8,6))
# plt.scatter( CR[0],CR[1],cmap='viridis', alpha=0.75)
# plt.title('M31 - CR Diffusion', color='white')
# plt.show()


# In[533]:


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm
import pylab as plt

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# CR=np.array([[0,1,2,3,23],[0,1,2,3,460], [0,1,2,3,0]])

def truncated_cone(p0, p1, R0, R1, CR):
    
    v = p1 - p0  # vector in direction of axis that'll give us the height of the cone
    h = np.sqrt(v[0]**2 +v[1]**2 + v[2]**2) #height of cone
    mag = norm(v) # find magnitude of vector
    v = v / mag  # unit vector in direction of axis
    not_v = np.array([1, 1, 0]) # make some vector not in the same direction as v
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])
    n1 = np.cross(v, not_v) # make vector perpendicular to v
    # print n1,'\t',norm(n1)
    n1 /= norm(n1)# normalize n1
    n2 = np.cross(v, n1) # make unit vector perpendicular to v and n1
    
    # surface ranges over t from 0 to length of axis and 0 to 2*pi
    n = 80
    t = np.linspace(0, mag, n)
    theta = np.linspace(0, 2 * np.pi, n)
    # use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    R = np.linspace(R0, R1, n)
    # generate coordinates for surface
#     X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1, projection='3d')
#     ax.plot_surface(X, Y, Z, color='b', linewidth=0, antialiased=False)
#     ax.scatter(CR[0],CR[1],CR[2], color='r')
    z = CR[2]
    permRadius = R1 - (R1 - R0) * (z /h) #boundary of cone
    pointRadius = (np.sqrt(np.add(np.subtract(CR[0], p0[0])**2, np.subtract(CR[1], p0[1])**2)))
#     print(permRadius)
#     print(pointRadius)
#     print(z, h)
    param1 = np.logical_and(z <= h, z >= p1[2]) #checks if it satisfies the z parameters
    params = (np.where(np.logical_and(param1, pointRadius<= permRadius), True, False)) #checks to see if it satisfies both the radius and z parameters
#     print(param1)
#     print(params)
    n_CR = sum(params.astype(int)) #Total number of particles in cone
    
    return n_CR

# print(CR)
A0 = np.array([0, 0, 150])
A1 = np.array([0, 0, 0])
print(truncated_cone(A0, A1, 0,300, CR))

n_bins = 10
axis = A0 - A1 #order depends on which angle you want to view the galaxy at (ie. above or belox xy plane)
length = np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
bins = np.linspace(0, length, n_bins+1)
# print(bins)
spacing = axis/(n_bins) #average space between rings
print(spacing)
# s = np.zeros(3)
for i in bins:
    i = np.int(i)
    A1 = spacing + A0
    V = (1/3)*length*np.pi*(np.dot(A1,A1) + np.dot(A1,A0) + np.dot(A0,A0))  #Volume of a Frustum of a Right Circular Cone
    truncated_cone(A0, A1, 0,300, CR)
    break
# #         print('between', c, 'and', i)
#         n_particles = sum(rad[c:i])
#         row = n_particles/area
#         density.append(row)
# #     print(a, s)
#     s=a
#     c=i
# r = np.sqrt(CR[0]**2 + CR[1]**2 +CR[2]**2)
# print(max(r))
# print(rad)
# print(r)
# r=(r.astype(np.int))
# # print(r)
# # print(np.bincount(r))
# rad = np.bincount(r)
# # print(rad)
# # print(len(rad))
# # print(len(rad))
# # print(sum(rad))

# n_bins = 10
# bins = np.linspace(0, len(rad), n_bins+1)
# r = np.linspace(0,300,n_bins)
# # print(r)
# # print(bins)
# spacing = max(r)/(n_bins-1) #average space between rings
# # print(spacing)
# s = 0
# density = []
# for i in bins:
#     i = np.int(i)
#     a = spacing + s
#     area = np.pi*(a**2 - s**2)
#     if i > 0:
# #         print('between', c, 'and', i)
#         n_particles = sum(rad[c:i])
#         row = n_particles/area
#         density.append(row)
# #     print(a, s)
#     s=a
#     c=i
        
# # print(x, figsize=(8,6))    
# # print(sum(x))
# # area = np.pi
# # print(area)
# plt.figure(3 , figsize=(8,6))
# plt.plot(r, density)
# plt.title('Density Profile')
# plt.xlabel('Radius (kpc)')
# plt.ylabel('Density')

# r = np.sqrt(CR[0]**2 + CR[1]**2)
# plt.figure(4, figsize=(8,6))
# plt.hist(r, bins=100)
# plt.xlabel('Radius (kpc)')
# plt.ylabel('Number of Particles')
# plt.title('Radial Profile')


# In[532]:


np.dot(np.array([2,2,2]), np.array([3,3,3]))


# In[23]:


r = np.sqrt(CR[0]**2 + CR[1]**2)
# print(max(r))
# print(rad)
# print(r)
r=(r.astype(np.int))
# print(r)
# print(np.bincount(r))
rad = np.bincount(r)
# print(rad)
# print(len(rad))
# print(len(rad))
# print(sum(rad))

n_bins = 100
bins = np.linspace(0, len(rad), n_bins+1)
r = np.linspace(0,300,n_bins)
# print(r)
# print(bins)
spacing = max(r)/(n_bins-1) #average space between rings
# print(spacing)
s = 0
density = []
for i in bins:
    i = np.int(i)
    a = spacing + s
    area = np.pi*(a**2 - s**2)
    if i > 0:
#         print('between', c, 'and', i)
        n_particles = sum(rad[c:i])
        row = n_particles/area
        density.append(row)
#     print(a, s)
    s=a
    c=i
        
# print(x, figsize=(8,6))    
# print(sum(x))
# area = np.pi
# print(area)
plt.figure(3 , figsize=(8,6))
plt.plot(r, density)
plt.title('Density Profile')
plt.xlabel('Radius (kpc)')
plt.ylabel('Density')

r = np.sqrt(CR[0]**2 + CR[1]**2)
plt.figure(4, figsize=(8,6))
plt.hist(r, bins=100)
plt.xlabel('Radius (kpc)')
plt.ylabel('Number of Particles')
plt.title('Radial Profile')


# In[314]:


import math

class Vector:
    def __init__(self, pos):
        self.X = pos[0]
        self.Y = pos[1]
        self.Z = pos[2]

class ConeShape:
    def __init__(self, pos, height, bRadius, tRadius):
        self.X = pos[0]
        self.Y = pos[1]
        self.Z = pos[2]

        self.baseRadius = bRadius
        self.topRadius = tRadius

        self.Height = height

    def Position(self):
        pos = [self.X, self.Y, self.Z]
        return Vector(pos)

    def ContainsPoint(self, x, y, z):
        z -= self.Z

        permRadius = self.baseRadius - (self.baseRadius - self.topRadius) * (z /self.Height)
        print(("Test: {0:0.2f}").format(permRadius))

        pointRadius = (math.sqrt((x - self.X) ** 2 + (y - self.Y) ** 2))

        if (z <= self.Height and z >= 0.0) and (pointRadius <= permRadius):
            return True
        return False

def main():
    cone = ConeShape([0.0, 0.1, 0.0], 10.0, 5.0, 0.1)
    if (cone.ContainsPoint(0.0, 0.1, 0.0)):
        print("Yes!")

    cone = ConeShape([0.0, 0.0, 0.0], 10.0, 5.0, 0.0)
    if (cone.ContainsPoint(1, 1, 1)):
        print("Yes!")
    else:
        print('NO!')

main()


# In[34]:


import numpy as np
import random
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
import scipy.optimize
import builtins

#----------------------------------PARAMETERS-------------------------------------------------------------------------------
n = 1000 #number of particles to run
size = 20 #radius of the circle boundary
width = 4 #thickness of the plane in the z-direction

#----------------------------------CYLINDRICAL BOUNDARY - if reached, particle dies-------------------------------------------
r_max = size
z_top = width/2
z_bottom = -1*(width/2)

dead_to_z = 0 #this tracks the number of particles that escape through top or bottom of the "box"
dead_to_r = 0 #this tracks the number of particles that escape through the sides

#this array keeps track of number of particles remaining at each step, index = time-step
particle_array = np.zeros(10000)

range = builtins.range

for i in range(1,n):
    #particle starts at origin
    x = 0   
    y = 0
    z = 0
    
    r_mag = 0
    z_mag = 0
    step_count = 0
    
    while (r_mag < r_max) and (z_mag < z_top) and (z_mag > z_bottom):
        #pick random angle
        theta = random.uniform(0, 2*np.pi)
        
        #pick random step size
        s = random.uniform(0,1)
        x_step = s*np.cos(theta)
        y_step = s*np.sin(theta)
        z_step = 0.02*random.uniform(-1,1) #less efficient diffusion in z-direction
        
        x += x_step
        y += y_step
        z += z_step
        
        r_mag = np.sqrt(x**2+y**2)
        z_mag = z #can be positive or negative
        
        step_count += 1   #keeps track of number of steps
        
        if (z_mag > z_top or z_mag < z_bottom):
            dead_to_z += 1
        if(r_mag > r_max):
            dead_to_r += 1
        
        #when the particle dies, add it to particle_array up to step index it died in
        if (r_mag > r_max) or (z_mag > z_top) or (z_mag < z_bottom):
            for j in range(0, step_count):
                particle_array[j] += 1
                
print("# of particles that escape through top or bottom of the box = ", dead_to_z)
print("# of particles that escape through the sides = ", dead_to_r)





# get rid of elements of particle_array where n > 0.9n
particle_array = np.delete(particle_array, np.argwhere(particle_array > 0.9*n))
print('Modified Numpy Array :')
print(particle_array)
print(len(particle_array))

#-------------------------DETERMINE WHAT RANGE YOU WANT: __% to __% of particles remaining-----------------------------------
left_bound = 0.899
right_bound = 0.0

#find indices corresponding to boundaries
print(np.where(particle_array == left_bound*n))
lb_arr = np.where(particle_array == left_bound*n)
left_index = lb_arr[0][0]
print("left index = ", left_index)

print(np.where(particle_array == right_bound*n))
rb_arr = np.where(particle_array == right_bound*n)
right_index = rb_arr[0][0]
print("right index = ", right_index)

bound_range = right_index - left_index + 1
print("RANGE = ", bound_range)

#--------------------------------FIT FUNCTIONS-----------------------------------------------------------------------------            
def model_func(t, N_0, t_0): #N_0 is amplitude, t is number of steps, and t_0 is a function of step size and boundary size
    return N_0 * np.exp(-1*t/t_0)

def fit_exp_nonlinear(t, y):
    opt_parms, parm_cov = sp.optimize.curve_fit(model_func, t, y, p0=[1000, 900], maxfev=1000)
    N_0, t_0 = opt_parms
    return N_0, t_0

#-----------------------------------PLOTS-----------------------------------------------------------------------------------
plt.figure(figsize=(10,7))
t = np.linspace(left_index, right_index, bound_range)

N_0, t_0 = fit_exp_nonlinear(t, particle_array[left_index:right_index+1]) #gives optimized variables
fit_y = model_func(t, N_0, t_0) #put optimized variables into fit line

#plot the remaining particles vs. time
plt.plot(t, particle_array[left_index:right_index+1], linewidth=2.0)

#plot guess
#guess_y = model_func(t, 1300, 900)
#plt.plot(t, guess_y, color='red', label='Guess Function', linewidth=3.0)

#plot fitted line
plt.plot(t, fit_y, color='orange', 
        label='Fitted Function:\n $y = %0.2f e^{-t/%0.2f}$' % (N_0, t_0), linewidth=3.0)

#add plot lables
left_percentage = left_bound * 100
right_percentage = right_bound * 100
plt.title('%.f to %.f percent of remaining particles' % (left_percentage, right_percentage))
plt.ylabel('number of particles')
plt.xlabel('time-step')
plt.legend(loc='best')
plt.show()


#---------------------PRINT VALUES-----------------------------------------------------------------------------------
print ('N_0 = ', N_0)
print ('t_0 = ', t_0)
slope, intercept, r_value, p_value, std_err = stats.linregress(fit_y, particle_array[left_index:right_index+1])
print ("r-squared: ", r_value**2)


# In[42]:


np.column_stack([[1,1,1],[2,5,2],[3,3,3]])


# In[138]:


import numpy as np
import scipy.spatial as spatial
points = np.array([(1, 2), (3, 4), (4, 5), (100,100)])
tree = spatial.KDTree(np.array(points))
radius = 3.0

neighbors = tree.query_ball_tree(tree, radius)
print(neighbors)
# [[0, 1], [0, 1, 2], [1, 2], [3]]
 
frequency = np.array(map(len, neighbors))
print(frequency)
# [2 3 2 1]
density = frequency/radius**2
print(density)
# [ 0.22222222  0.33333333  0.22222222  0.11111111]


# In[360]:


nsteps =10000000
rstep = step_size #see return from step_size def (this line though is only swapping the title of the def)
zstep = lambda z,r : 0.5 #not used in spherical coordinate case

CR, CR_esc, density = initial_CR(particles=0)
# CR = spawn_sphere(CR, particles=100, rmax=5.5, x0=0, y0=0, z0=0, shell=False)
# CR = spawn_sphere_ring(CR, particles = 6666, rmax=117, rmin=5.5, x0=0, y0=0, z0=0, shell=False)
# CR = spawn_ring(CR, particles=3333, rmax=200, rmin=117, x0=0, y0=0, z0=0, shell=False)
# CR = spawn_IR(CR)
CR = spawn_H(CR)
initial = CR.shape[1]

# e = spawn_sphere(e, particles=1000, rmax=5.5, x0=0, y0=0, z0=0, shell=False)
# e_initial = e.shape[1]
# e_initial_NRG = np.add(np.zeros(e_initial), 1000)
# b = 10**(-16) #GeV^-1*sec^-1

for n in range(0,nsteps): 
    t = n #number of steps in one second
    CR, r,z, CR_esc, r_step, particle_array = run_step(CR, CR_esc, rstep, zstep)
#     e, r,z, e_esc = run_step(e, e_esc, rstep, zstep)
    t = t #edit this if you want to say something like 1 step = 1/3 seconds ie set t = t/3
#     e_final = e_initial/(1+b*t*e_initial)
# for if (t%#1 == #2):, #1 is every number of steps when the if statement activates, #2 is the initial step. 
#     if(t%100==99):
#         CR = spawn_ring(CR, particles=0,rmax=65, rmin=55, x0=0, y0=0, z0=0)
#         CR = spawn_sphere(CR, particles=0, rmax=15, x0=0, y0=0, z0=0)
# #         CR = spawn_sphere(CR, particles=8, rmax=15, x0=55, y0=55, z0=0, shell=True)
    if CR.shape[1]<=(np.exp(-1))*initial:   #number of steps to transient
        t0 = -t/(np.log(CR.shape[1]/initial))
        rate=(initial)*np.exp(-t/t0)
#         N0, t0 = fit_exp_nonlinear(t, particle_array[left_index:right_index+1]) #gives optimized variables
        print("Analytical t0 =", t0)
        print("Number of Seconds to get to Steady State =" , t+1, "seconds")
        break
        

escaped = CR_esc.shape[1]
print('Initial Particles that Escaped Fraction = {:.3f}% or {:} total'.format( (escaped/initial)*100, escaped))
print('total particles', CR.shape[1]+CR_esc.shape[1])

# inside = []   
# N = 0 
# r_cylinder = 100 #radius of desired cross section
# L = 1500 #how far out you want to see
# V_cylinder = np.pi*(r_cylinder**2)*L
# for i in range(0, len(CR[0])):
#     in_or_out = points_in_cylinder(np.array([-750,0,0]),np.array([750,0,0]),r_cylinder,
#                             np.array([CR[0][i],CR[1][i],CR[2][i]]))
#     if np.size(in_or_out)==1: #if its one it means its inside, zero if it's outside
#         inside.append(np.size(in_or_out))
#         N += 1 #number of particles inside the cross section
# n_CR = N/V_cylinder #column density

# print('Column Density of Cosmic Rays:', n_CR, 'particles/kpc^3')
 
get_ipython().run_line_magic('matplotlib', 'auto')
 
density = get_color(CR, kde=True)    
# fig, ax = build3d(grid=True, panel=0.5, boxsize=800)
plt.figure(figsize = (8,6))
ax = plt.subplot()
ax.set_title('M31 - CR Diffusion', color='white')

ax.scatter( CR[0],CR[1],cmap='viridis',s=1)
ax.scatter( CR_esc[0],CR_esc[1], c='r',s=1)
plt.show()

plt.figure(figsize = (8,6))
ax = plt.subplot()
ax.set_title('M31 - CR Diffusion', color='white')

ax.scatter(CR[0],CR[1], c=density,cmap='viridis',s=1)
plt.show()
# %matplotlib inline
# plt.figure(figsize=(8,6))
# plt.hist(parallax, bins=100, range=(1,1000), label='Parllaxes')


# In[470]:





# In[435]:


round(5.1)


# In[ ]:




