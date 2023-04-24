import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from skimage import io

def generate_pixel_coordinate_tensor(image):
    #since we try to mak this function general, but the return of io.imread can be very different
    #we have to adress all possible cases. The first one being that is a greyscale.
    #Here we have made the requisite for the image to be more than 4 pixels wide in its last dimension.
    if image.shape[-1] > 4:
        dims = image.shape
        ncolors = 1
    elif image.shape[-1] == 4:
        dims = image.shape[:-1]
        ncolors = 4
    elif image.shape[-1] == 3:
        dims = image.shape[:-1]
        ncolors = 3
    else:
        raise Exception('Wrong image format')

    if len(dims) == 2:
        output_grid = np.mgrid[0:dims[0],
                               0:dims[1]].reshape((2, -1))
    else:
        output_grid = np.mgrid[0:dims[0],
                               0:dims[1],
                               0:dims[2]].reshape((3, -1))

    output_colors = np.reshape(image, (-1, ncolors))
    return torch.tensor(output_grid, dtype=torch.float), torch.tensor(output_colors)

def affine_rotation_tensor_2d(theta):
    rot_matrix = torch.zeros((2,2), dtype=torch.float)
    rot_matrix[0, 0] = np.cos(theta)
    rot_matrix[0, 1] = - np.sin(theta)
    rot_matrix[1, 0] = np.sin(theta)
    rot_matrix[1, 1] = np.cos(theta)
    output = torch.row_stack((torch.column_stack((rot_matrix, torch.tensor([0,0]))),
                              torch.tensor([0,0,1])))
    return output

def affine_rotation_given_center_2d(vector, angle):
    rot_matrix = torch.zeros((2,2), dtype=torch.float)
    rot_matrix[0, 0] = np.cos(angle)
    rot_matrix[0, 1] = - np.sin(angle)
    rot_matrix[1, 0] = np.sin(angle)
    rot_matrix[1, 1] = np.cos(angle)
    
    affine_portion = torch.tensor([vector[0]*(1-math.cos(angle))
                                   + vector[1]*math.sin(angle),
                                   vector[1]*(1-math.cos(angle))
                                   - vector[0]*math.sin(angle)])
    output = torch.row_stack((torch.column_stack((rot_matrix, affine_portion)),
                              torch.tensor([0,0,1])))
    return output


def affine_translation_tensor_2d(vector):
    output = torch.row_stack((torch.column_stack((torch.eye(2), vector)),
                              torch.tensor([0,0,1])))
    return output

def AnimationFunction2D(frame, figure, input_grid, translation_vector):
    output_grid = affine_rotation_given_center_2d(translation_vector, frame) @ input_grid
    figure.set_offsets(output_grid[:-1,:].permute(1,0))

if __name__=="__main__":
    img = io.imread('arbol.png')

    grid, colors = generate_pixel_coordinate_tensor(img)

    centroid = torch.tensor([torch.mean(grid[i,:][colors[:,2] < 240], dtype=torch.float)
                         for i in (0,1)]).reshape(-1,1)
    diameter = torch.max(torch.cdist(grid[:, colors[:,2] < 240].permute(1,0),
                                     grid[:, colors[:,2] < 240].permute(1,0), p=2))
    translation_vector = torch.tensor([ diameter, diameter]).reshape(-1,1)

    affine_grid = torch.row_stack((grid, torch.ones((1,len(grid[0])))))
    fig = plt.figure()
    figure = plt.scatter(grid[1], grid[0], c=colors/255.0)
    plt.xlim(0, 2.2*diameter)
    plt.ylim(0, 2.2*diameter)

    anim_created = ani.FuncAnimation(fig,
                                     AnimationFunction2D,
                                     fargs=(figure, affine_grid, translation_vector),
                                     frames=np.linspace(0, 3* math.pi ,100))
    anim_created.save('apartado2.gif')
 
