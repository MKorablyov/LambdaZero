import torch
import torch.nn as nn
import torch.nn.functional as F


from se3cnn.image.convolution import SE3Convolution
from lie_learn.representations.SO3.wigner_d import wigner_D_matrix
from se3cnn.image.gated_block import GatedBlock
from se3cnn.image.utils import rotate_scalar, rotate_field
import numpy as np
from scipy.ndimage import zoom

from TorchProteinLibrary.FullAtomModel.CoordsTransform import getRandomRotation
from TorchProteinLibrary.Volume import VolumeRotation
from _Volume import Volume2Xplor

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

from SE3Vector import generateScalarField, rotateScalarField
from SE3Rotations import generateVectorField

size = 8  # space size
x = torch.zeros(size, size, size, dtype=torch.float).numpy()
y = torch.zeros(size, size, size, dtype=torch.float).numpy()
z = torch.zeros(size, size, size, dtype=torch.float).numpy()
for i in range(size):
	x[i,:,:] = i
	y[:,i,:] = i
	z[:,:,i] = i

if __name__=='__main__':
	

	model = nn.Sequential(
		SE3Convolution([(1,0)], [(1,1)], size=5, dyn_iso=False, padding=2)
	)

	with torch.no_grad():
		for parameter in model.parameters():
			parameter.fill_(1.0)

	scalar_field = generateScalarField(size)
	with torch.no_grad():
		output_field = model(scalar_field)/5.0

	print(output_field.size())
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	Q = ax.quiver(  x, y, z, output_field[0,0,:,:,:].numpy(), output_field[0,1,:,:,:].numpy(), output_field[0,2,:,:,:].numpy(), 
					normalize=False)

	def update(frame):
		global Q
		Q.remove()
		scalar_field_rot, R = rotateScalarField(frame, scalar_field)
		with torch.no_grad():
			output_field_rot = model(scalar_field_rot)/5.0
		Q = ax.quiver(  x, y, z, output_field_rot[0,0,:,:,:].numpy(), output_field_rot[0,1,:,:,:].numpy(), output_field_rot[0,2,:,:,:].numpy(), 
						normalize=False)
		return Q,
	
	ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128))
	
	plt.show()