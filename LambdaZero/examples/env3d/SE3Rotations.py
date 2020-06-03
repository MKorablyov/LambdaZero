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

from SE3Vector import generateScalarField, rotateScalarField

from se3cnn.SE3 import rotate_scalar
from se3cnn.SO3 import rot
from se3cnn.SO3 import irr_repr

def generateVectorField(size):
	vector_field = torch.zeros(1, 3, size, size, size, dtype=torch.float)
	for i in range(size):
		for j in range(size):
			v = np.array([float(i) - float(size)/2.0, 0, float(j) - float(size)/2.0])
			r = np.linalg.norm(v)
			v = 1.0*v/(r+0.000001)
			v_tangent = np.array([-v[2], 0, v[0]])
			vector_field[0,0,i,:,j] = float(v_tangent[0])
			vector_field[0,1,i,:,j] = float(v_tangent[1])
			vector_field[0,2,i,:,j] = float(v_tangent[2])
	
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
					vector_field[0,:,i,j,k] = 0.0
	
	return vector_field

def rotate(x, alpha, beta, gamma):
	y = x.numpy()
	R = rot(alpha, beta, gamma)
	for i in range(y.shape[0]):
		y[i] = rotate_scalar(y[i], R)
	x = x.new_tensor(y)
	rep = irr_repr(1, alpha, beta, gamma, x.dtype).to(x.device)
	print(rep)
	x = torch.einsum("ij,jxyz->ixyz", (rep, x))
	return x


if __name__=='__main__':
	size = 8  # space size

	model = nn.Sequential(
		SE3Convolution([(1,1)], [(1,1)], size=5, dyn_iso=False, padding=5),
		nn.AvgPool3d(kernel_size=size)
	)

	with torch.no_grad():
		for parameter in model.parameters():
			parameter.fill_(1.0)

	scalar_field = generateScalarField(size)
	scalar_field_rot, R = rotateScalarField(np.pi/2.0, scalar_field)

	R = R[0,:,:]
	
	x = torch.zeros(size, size, size, dtype=torch.float)
	y = torch.zeros(size, size, size, dtype=torch.float)
	z = torch.zeros(size, size, size, dtype=torch.float)
	for i in range(size):
		x[i,:,:] = i
		y[:,i,:] = i
		z[:,:,i] = i

	vector_field = generateVectorField(size)
	vector_field_rot = torch.zeros_like(vector_field).copy_(vector_field)
	vector_field_rot = rotate(vector_field_rot[0,:,:,:,:], 0, 0, np.pi/2.0).unsqueeze(dim=0).contiguous()

	# fig = plt.figure()
	# ax = fig.gca(projection='3d')
	# ax.quiver(x.numpy(), y.numpy(), z.numpy(), vector_field[0,0,:,:,:].numpy(), vector_field[0,1,:,:,:].numpy(), vector_field[0,2,:,:,:].numpy(), length=1.0, normalize=True)
	# plt.show()

	# fig = plt.figure()
	# ax = fig.gca(projection='3d')
	# ax.quiver(x.numpy(), y.numpy(), z.numpy(), vector_field_rot[0,0,:,:,:].numpy(), vector_field_rot[0,1,:,:,:].numpy(), vector_field_rot[0,2,:,:,:].numpy())
	# plt.show()

	input_field = vector_field
	input_field_rot = vector_field_rot

	with torch.no_grad():
		output_field = model(input_field).squeeze()
		output_field_rot = model(input_field_rot).squeeze()

	print("|F[Rx]|=", torch.sqrt(torch.sum(output_field_rot*output_field_rot)).item())
	print("|F[x]|=", torch.sqrt(torch.sum(output_field*output_field)).item())

	fields = [0, 1]
	idx = 0
	for l, mult in enumerate(fields):
		if l == 0: 
			idx += mult*(2*l+1)
			continue
		
		D = wigner_D_matrix(l, 0.0, 0.0, np.pi/2.0)

		for n in range(mult):
			print(idx, idx+2*l+1)
			print(D)
			rot_output_field = D.dot(output_field[idx:idx+2*l+1].numpy())
			print("F[x](l=%d)"%l, output_field[idx:idx+2*l+1])
			print("F[Rx](l=%d)"%l, output_field_rot[idx:idx+2*l+1])
			print("RF[x](l=%d)"%l, rot_output_field)
			
			x = output_field_rot[idx:idx+2*l+1]
			y = torch.from_numpy(rot_output_field).to(dtype=torch.float)
			norm = max([torch.sqrt(torch.sum(x*x)).item(), torch.sqrt(torch.sum(y*y)).item()])

			print("Err=",torch.sqrt(torch.sum((x - y)*(x-y))).item() /  norm)

			idx += 2*l+1