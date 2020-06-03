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
from math import *

def generateScalarField(size):
	scalar_field = torch.zeros(1, 1, size, size, size, dtype=torch.float)
	for i in range(size):
		scalar_field[0,:,i,:,:] = 3.0*(float(i)/float(size) - 0.5)

	for i in range(size):
		dx = float(i) - float(size)/2.0
		dx *= dx
		for j in range(size):
			dy = float(j) - float(size)/2.0
			dy *= dy
			for k in range(size):
				dz = float(k) - float(size)/2.0
				dz *= dz
				if dx + dy + dz > float(size*size)/4.0:
					scalar_field[0,0,i,j,k] = 0.0
	
	return scalar_field

def rotateScalarField(phi, scalar_field):
	R = torch.tensor([[ [cos(phi), -sin(phi), 0.0],
						[sin(phi), cos(phi), 0.0],
						[0.0, 0.0, 1.0]]],
						dtype=torch.float, device='cuda')
	rotate = VolumeRotation(mode='bilinear')
	scalar_field_rot = rotate(scalar_field.cuda(), R).cpu()
	# Volume2Xplor(scalar_field_rot[0,0,:,:,:].to(device='cpu'), "%.1f.xplor"%phi, 1.0)
	# scalar_field_rot = scalar_field_rot.permute(0, 1, 4, 3, 2).contiguous()
	return scalar_field_rot, R.to(dtype=torch.double, device='cpu').numpy()

if __name__=='__main__':

	size = 30  # space size

	model = nn.Sequential(
		SE3Convolution([(1,0)], [(1,1)], size=5, dyn_iso=False, padding=5),
		nn.AvgPool3d(kernel_size=size)
	)

	with torch.no_grad():
		for parameter in model.parameters():
			parameter.fill_(1.0)
	
	scalar_field = generateScalarField(size)

	with torch.no_grad():
		output_field = model(scalar_field).squeeze()

	vecs = []
	rot_vec = []
	x = []
	y = []
	z = []
	with torch.no_grad():
		for i in range(0, 10):
			phi = float(i)*np.pi/10.0
			input_field_rot, R = rotateScalarField(phi, scalar_field)
			output_field_rot = model(input_field_rot).squeeze()
			vecs.append(output_field_rot)

			D = wigner_D_matrix(1, 0.0, 0.0, phi)
			v = D.dot(output_field.numpy())
			print("F[Rx]", output_field_rot)
			print("RF[x]", v)
			rot_vec.append(torch.from_numpy(v).squeeze())
			x.append(float(i/10.0))
			y.append(0)
			z.append(0)


	vecs = torch.stack(vecs, dim=0)
	rot_vecs = torch.stack(rot_vec, dim=0)

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.quiver(x, y, z, vecs[:,0].numpy(), vecs[:,1].numpy(), vecs[:,2].numpy(), color='blue')
	ax.quiver(x, y, z, rot_vecs[:,0].numpy(), rot_vecs[:,1].numpy(), rot_vecs[:,2].numpy(), color='red')

	ax.set_xlim(0, 1)
	ax.set_ylim(-1, 1)
	ax.set_zlim(-1, 1)

	plt.show()
	# ax = fig.gca(projection='3d')
	# ax.quiver(x.numpy(), y.numpy(), z.numpy(), vector_field[0,0,:,:,:].numpy(), vector_field[0,1,:,:,:].numpy(), vector_field[0,2,:,:,:].numpy(), length=1.0, normalize=True)
	# plt.show()