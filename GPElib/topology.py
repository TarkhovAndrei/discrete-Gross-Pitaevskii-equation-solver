'''
Copyright <2019> <Andrei E. Tarkhov, Skolkovo Institute of Science and Technology, https://github.com/TarkhovAndrei/DGPE>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following 2 conditions:

1) If any part of the present source code is used for any purposes with subsequent publication of obtained results,
the GitHub repository shall be cited in all publications, according to the citation rule:
	"Andrei E. Tarkhov, Skolkovo Institute of Science and Technology,
	 source code from the GitHub repository https://github.com/TarkhovAndrei/DGPE, 2019."

2) The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from mayavi.mlab import orientation_axes
class Topology(object):
	def __init__(self, **kwargs):
		if 'PSI' in kwargs:
			self.PSI = kwargs.get('PSI')
			self.X = np.real(self.PSI)
			self.Y = np.imag(self.PSI)
			self.Z = np.zeros_like(self.X)
		elif 'X' in kwargs and 'Y' in kwargs:
			self.X = np.array(kwargs.get('X'),dtype=np.float64)
			self.Y = np.array(kwargs.get('Y'), dtype=np.float64)
			self.Z = np.zeros_like(self.X)
			self.PSI = self.X + 1j * self.Y
		else:
			self.PSI = np.ones((10,10,10), dtype=np.complex64)
			self.X = np.real(self.PSI)
			self.Y = np.imag(self.PSI)
			self.Z = np.zeros_like(self.X)

		self.X_norm = self.X / np.sqrt(np.power(self.X,2) + np.power(self.Y,2))
		self.Y_norm = self.Y / np.sqrt(np.power(self.X,2) + np.power(self.Y,2))

		self.current_X, self.current_Y, self.current_Z = self.current(self.X, self.Y, self.Z)
		self.rot_current_vector_X, self.rot_current_vector_Y, self.rot_current_vector_Z  = self.rot(self.current_X,
																									self.current_Y,
																									self.current_Z)
		self.div_rot_current_vector = self.div(self.rot_current_vector_X, self.rot_current_vector_Y, self.rot_current_vector_Z)

		self.Theta = np.arctan2(self.Y_norm, self.X_norm)

		self.N_tuple = self.PSI.shape
		self.Nx, self.Ny, self.Nz = self.N_tuple

		self.N_wells = self.Nx * self.Ny * self.Nz
		self.wells_indices = [(i,j,k) for i in range(self.Nx) for j in range(self.Ny) for k in range(self.Nz)]

		self.IX = np.array([ix[0] for ix in self.wells_indices], dtype=np.float64).reshape(self.N_tuple)
		self.IY = np.array([ix[1] for ix in self.wells_indices], dtype=np.float64).reshape(self.N_tuple)
		self.IZ = np.array([ix[2] for ix in self.wells_indices], dtype=np.float64).reshape(self.N_tuple)

		self.wells_index_tuple_to_num = dict()
		for i in range(self.Nx):
			for j in range(self.Ny):
				for k in range(self.Nz):
					# self.wells_index_tuple_to_num[(i,j,k)] = i + self.Nx * (j + self.Ny * k)
					self.wells_index_tuple_to_num[(i,j,k)] = k + self.Nz * (j + self.Ny * i)

		self.cube_sides = ['x_low', 'x_high', 'y_low', 'y_high', 'z_low','z_high']
		self.topological_charges_of_cube_sides = np.zeros((len(self.cube_sides),)+ self.N_tuple)
		self.cummulative_topcharges_of_cubes = np.zeros(self.N_tuple)
		self.to_plot_cubes = np.zeros(self.N_tuple) == 1
		self.plot_cube_sides = np.zeros((len(self.cube_sides),) + self.N_tuple) == 1
		self.calculate_charges_of_cube_sides()

	def norm_angle(self, phi):
		phi_new = np.zeros((3,) + phi.shape)
		for i in range(phi_new.shape[0]):
			phi_new[i] = phi + (2. * np.pi * (i - 1.))
		ai = np.expand_dims(np.argmin(np.abs(phi_new), axis=0), axis=0)
		phi_new = np.take_along_axis(phi_new, ai, axis=0)[0]
		return phi_new

	def get_shift_and_axis(self, i, contour):
		corner = contour[i]
		corner = np.array(corner, dtype=np.int8)
		axis = np.where(1.*corner > 0.5)[0]
		if axis.shape[0] > 0:
			shift = -corner[np.where(1.*corner > 0.5)[0]]
		else:
			shift = 0
		corner_next = np.roll(contour, -1, axis=0)[i]
		axis_next = np.where(1.*corner_next > 0.5)[0]
		if axis_next.shape[0] > 0:
			shift_next = -corner_next[np.where(1.*corner_next > 0.5)[0]]
		else:
			shift_next = 0
		return shift_next, axis_next, shift, axis

	def get_total_angle_of_contour(self, contour):
		angle = np.zeros_like(self.Theta, dtype=np.float64)
		for i, corner in enumerate(contour):
			shift_next, axis_next, shift, axis = self.get_shift_and_axis(i, contour)
			angle += self.norm_angle(np.roll(self.Theta,shift_next,axis=axis_next)- np.roll(self.Theta,shift,axis=axis))
			# theta_next = self.get_rolled_array(self.Theta, shift_next, axis_next)
			# theta = self.get_rolled_array(self.Theta, shift, axis)
			# angle += self.norm_angle(theta_next - theta)
		return angle / (2. * np.pi)

	def calculate_charges_of_cube_sides(self):
		for i, cube_side in enumerate(self.cube_sides):
			self.topological_charges_of_cube_sides[i,:,:,:] = self.get_total_angle_of_contour(self.get_contour(i))
		self.cummulative_topcharges_of_cubes = np.sum(np.abs(self.topological_charges_of_cube_sides), axis=0)
		self.to_plot_cubes = 1.*self.cummulative_topcharges_of_cubes > 0.1
		self.plot_cube_sides[0,:,:,:] = np.logical_and(self.to_plot_cubes, ~np.roll(self.to_plot_cubes, 1, axis=0))
		self.plot_cube_sides[1,:,:,:] = np.logical_and(self.to_plot_cubes, ~np.roll(self.to_plot_cubes, -1, axis=0))
		self.plot_cube_sides[2,:,:,:] = np.logical_and(self.to_plot_cubes, ~np.roll(self.to_plot_cubes, 1, axis=1))
		self.plot_cube_sides[3,:,:,:] = np.logical_and(self.to_plot_cubes, ~np.roll(self.to_plot_cubes, -1, axis=1))
		self.plot_cube_sides[4,:,:,:] = np.logical_and(self.to_plot_cubes, ~np.roll(self.to_plot_cubes, 1, axis=2))
		self.plot_cube_sides[5,:,:,:] = np.logical_and(self.to_plot_cubes, ~np.roll(self.to_plot_cubes, -1, axis=2))


	def get_contour(self, i):
		# x = 0 cube side
		if i == 0:
			#(x,y,z)
			return [(0,0,0),(0,0,1),(0,1,1),(0,1,0)]
		# x = 1 cube side
		elif i == 1:
			#(x,y,z)
			return [(1,0,0), (1,1,0),(1,1,1),(1,0,1)]
		# y = 0 cube side
		elif i == 2:
			#(x,y,z)
			return [(0,0,0),(1,0,0),(1,0,1),(0,0,1)]
		# y = 1 cube side
		elif i == 3:
			#(x,y,z)
			return [(0,1,0),(0,1,1),(1,1,1),(1,1,0)]
		# z = 0 cube side
		elif i == 4:
			#(x,y,z)
			return [(0,0,0),(0,1,0),(1,1,0),(1,0,0)]
		# z = 1 cube side
		elif i == 5:
			return [(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
		else:
			return 0

	def get_contour_center_coords(self, i):
		return np.mean(np.array(self.get_contour(i), dtype=np.float64), axis=0)

	def get_contour_center_normal(self, i):
		# x = 0 cube side
		if i == 0:
			return (-1.,0.,0.)
		# x = 1 cube side
		elif i == 1:
			return (1.,0.,0.)
		# y = 0 cube side
		elif i == 2:
			return (0.,-1.,0.)
		# y = 1 cube side
		elif i == 3:
			return (0.,1.,0.)
		# z = 0 cube side
		elif i == 4:
			return (0.,0.,-1.)
		# z = 1 cube side
		elif i == 5:
			return (0.,0.,1.)
		else:
			return 0

	def get_topological_quiver3d_vortices(self):
		for i in range(len(self.cube_sides)):
			# IX, IY, IZ = np.meshgrid(np.arange(self.Nx),np.arange(self.Ny),np.arange(self.Nz))
			IX = np.array([ix[0] for ix in self.wells_indices], dtype=np.float64).reshape(self.N_tuple)
			IY = np.array([ix[1] for ix in self.wells_indices], dtype=np.float64).reshape(self.N_tuple)
			IZ = np.array([ix[2] for ix in self.wells_indices], dtype=np.float64).reshape(self.N_tuple)
			# IX *= 1.
			# IY *= 1.
			# IZ *= 1.
			idx_to_plot = np.abs(1.*self.topological_charges_of_cube_sides[i,:,:,:].flatten()) > 0.5

			IU = 0.5 * self.get_contour_center_normal(i)[0] * self.topological_charges_of_cube_sides[i,:,:,:]
			IV = 0.5 * self.get_contour_center_normal(i)[1] * self.topological_charges_of_cube_sides[i,:,:,:]
			IW = 0.5 * self.get_contour_center_normal(i)[2] * self.topological_charges_of_cube_sides[i,:,:,:]

			idx_positive_charge = (1.*self.topological_charges_of_cube_sides[i,:,:,:]) < 0.5
			idx_negative_charge = (1.*self.topological_charges_of_cube_sides[i,:,:,:]) > -0.5

			IX[idx_negative_charge] += 0.5
			IY[idx_negative_charge] += 0.5
			IZ[idx_negative_charge] += 0.5

			IX[idx_positive_charge] += self.get_contour_center_coords(i)[0]
			IY[idx_positive_charge] += self.get_contour_center_coords(i)[1]
			IZ[idx_positive_charge] += self.get_contour_center_coords(i)[2]

			if i == 0:
				self.qX = IX.flatten().copy()[idx_to_plot]
				self.qY = IY.flatten().copy()[idx_to_plot]
				self.qZ = IZ.flatten().copy()[idx_to_plot]
				self.qU = IU.flatten().copy()[idx_to_plot]
				self.qV = IV.flatten().copy()[idx_to_plot]
				self.qW = IW.flatten().copy()[idx_to_plot]
			else:
				self.qX = np.hstack((self.qX, IX.flatten().copy()[idx_to_plot]))
				self.qY = np.hstack((self.qY, IY.flatten().copy()[idx_to_plot]))
				self.qZ = np.hstack((self.qZ, IZ.flatten().copy()[idx_to_plot]))
				self.qU = np.hstack((self.qU, IU.flatten().copy()[idx_to_plot]))
				self.qV = np.hstack((self.qV, IV.flatten().copy()[idx_to_plot]))
				self.qW = np.hstack((self.qW, IW.flatten().copy()[idx_to_plot]))
		return self.qX, self.qY, self.qZ, self.qU, self.qV, self.qW


	def rot(self, X0, Y0, Z0, normalized=False):

		if normalized == True:
			norm = np.sqrt(X0 ** 2 + Y0 ** 2)
			X0 /= norm
			Y0 /= norm

		rot_X = (	self.first_derivative_4th_order(Z0, axis=1) -
					self.first_derivative_4th_order(Y0, axis=2)
				 )

		rot_Y = (	self.first_derivative_4th_order(X0, axis=2) -
				   self.first_derivative_4th_order(Z0, axis=0)
				)

		rot_Z = ( self.first_derivative_4th_order(Y0, axis=0)  -
				  self.first_derivative_4th_order(X0, axis=1)
				)

		return rot_X, rot_Y, rot_Z

	def div(self, X0, Y0, Z0):

		div_Z = (   self.first_derivative_4th_order(X0, axis=0) +
					self.first_derivative_4th_order(Y0, axis=1) +
					self.first_derivative_4th_order(Z0, axis=2)
				)

		return div_Z

	def first_derivative_4th_order(self, arr, axis=0):
		return ((-np.roll(arr, -2, axis=axis) + 8. * np.roll(arr, -1, axis=axis) -
				   8. * np.roll(arr, 1, axis=axis) + np.roll(arr, 2, axis=axis)) / 12.)

	def current(self, X0, Y0, Z0):
		flow_X = 2. * (X0 * self.first_derivative_4th_order(Y0, axis=0)  -
					   Y0 * self.first_derivative_4th_order(X0, axis=0))

		flow_Y = 2. * (X0 * self.first_derivative_4th_order(Y0, axis=1) -
					   Y0 * self.first_derivative_4th_order(X0, axis=1))

		flow_Z = 2. * (X0 * self.first_derivative_4th_order(Y0, axis=2) -
					   Y0 * self.first_derivative_4th_order(X0, axis=2))
		return flow_X, flow_Y, flow_Z

	# def get_topological_quiver3d(self):
	# 	for i in range(len(self.cube_sides)):
	# 		# IX, IY, IZ = np.meshgrid(np.arange(self.Nx),np.arange(self.Ny),np.arange(self.Nz))
	# 		IX = np.array([ix[0] for ix in self.wells_indices]).reshape(self.N_tuple)
	# 		IY = np.array([ix[1] for ix in self.wells_indices]).reshape(self.N_tuple)
	# 		IZ = np.array([ix[2] for ix in self.wells_indices]).reshape(self.N_tuple)
	# 		IX = IX.astype(float)
	# 		IY = IY.astype(float)
	# 		IZ = IZ.astype(float)
	# 		IX += self.get_contour_center_coords(i)[0]
	# 		IY += self.get_contour_center_coords(i)[1]
	# 		IZ += self.get_contour_center_coords(i)[2]
	# 		idx_to_plot = np.abs(self.topological_charges_of_cube_sides[i,:,:,:].flatten()) > 0.5
	# 		IU = self.get_contour_center_normal(i)[0] * self.topological_charges_of_cube_sides[i,:,:,:]
	# 		IV = self.get_contour_center_normal(i)[1] * self.topological_charges_of_cube_sides[i,:,:,:]
	# 		IW = self.get_contour_center_normal(i)[2] * self.topological_charges_of_cube_sides[i,:,:,:]
	# 		if i == 0:
	# 			self.qX = IX.flatten()[idx_to_plot]
	# 			self.qY = IY.flatten()[idx_to_plot]
	# 			self.qZ = IZ.flatten()[idx_to_plot]
	# 			self.qU = IU.flatten()[idx_to_plot]
	# 			self.qV = IV.flatten()[idx_to_plot]
	# 			self.qW = IW.flatten()[idx_to_plot]
	# 		else:
	# 			self.qX = np.hstack((self.qX, IX.flatten()[idx_to_plot]))
	# 			self.qY = np.hstack((self.qY, IY.flatten()[idx_to_plot]))
	# 			self.qZ = np.hstack((self.qZ, IZ.flatten()[idx_to_plot]))
	# 			self.qU = np.hstack((self.qU, IU.flatten()[idx_to_plot]))
	# 			self.qV = np.hstack((self.qV, IV.flatten()[idx_to_plot]))
	# 			self.qW = np.hstack((self.qW, IW.flatten()[idx_to_plot]))
	# 	return self.qX, self.qY, self.qZ, self.qU, self.qV, self.qW
	#

	# #
	# # def flow_XYZ(X0, Y0, Z0):
	# # 	N = X0.shape[0]
	# # 	Nx = X0.shape[0]
	# # 	Ny = X0.shape[1]
	# # 	Nz = X0.shape[2]
	# # 	flow_X = np.zeros((Nx,Ny,Nz))
	# # 	flow_Y = np.zeros((Nx,Ny,Nz))
	# # 	flow_Z = np.zeros((Nx,Ny,Nz))
	# #
	# # #     idx = np.arange(N)
	# # 	idx = np.arange(Nx)
	# # 	idy = np.arange(Ny)
	# # 	idz = np.arange(Nz)
	# # 	X = X0.copy()
	# # 	Y = Y0.copy()
	# # 	Z = Z0.copy()
	# #
	# # 	flow_X = 2. * ((Y[NN_arr(Nx, idx + 1),:,:] * X[NN_arr(Nx, idx),:,:] -
	# # 				 Y[NN_arr(Nx, idx),:,:] * X[NN_arr(Nx, idx + 1),:,:]) +
	# # 			  (Y[NN_arr(Nx, idx - 1),:,:] * X[NN_arr(Nx, idx),:,:] -
	# # 				 Y[NN_arr(Nx, idx),:,:] * X[NN_arr(Nx, idx - 1),:,:]))
	# #
	# # 	flow_Y = 2. * ( (Y[:,NN_arr(Ny, idy + 1),:] * X[:,NN_arr(Ny, idy),:] -
	# # 				 Y[:,NN_arr(Ny, idy),:] * X[:,NN_arr(Ny, idy + 1),:]) +
	# # 			  (Y[:,NN_arr(Ny, idy - 1),:] * X[:,NN_arr(Ny, idy),:] -
	# # 				 Y[:,NN_arr(Ny, idy),:] * X[:,NN_arr(Ny, idy - 1),:]))
	# #
	# # 	flow_Z = 2. * ( (Y[:,:,NN_arr(Nz, idz + 1)] * X[:,:,NN_arr(Nz, idz)] -
	# # 				 Y[:,:,NN_arr(Nz, idz)] * X[:,:,NN_arr(Nz, idz + 1)]) +
	# # 			   (Y[:,:,NN_arr(Nz, idz - 1)] * X[:,:,NN_arr(Nz, idz)] -
	# # 				 Y[:,:,NN_arr(Nz, idz)] * X[:,:,NN_arr(Nz, idz - 1)]))
	# #
	# # 	return flow_X, flow_Y, flow_Z
#


	# def NN_arr(self, idx, axis=0):
	# 	jdx = idx.copy()
	# 	jdx[idx < 0] = self.N_tuple[axis] + jdx[idx < 0]
	# 	jdx[idx > self.N_tuple[axis] - 1] = jdx[idx > self.N_tuple[axis] - 1] - self.N_tuple[axis]
	# 	return jdx

	# def get_rolled_array(self, arr, shift, axis):
	# 	arr1 = arr.copy()
	# 	for iax, ax in enumerate(axis):
	# 		idx = np.arange(arr.shape[ax])
	# 		idx += shift[iax]
	# 		idx = self.NN_arr(idx, axis=ax)
	# 		if ax == 0:
	# 			arr1 = arr1[idx]
	# 		elif ax == 1:
	# 			arr1 = arr1[:,idx]
	# 		elif ax == 2:
	# 			arr1 = arr1[:,:,idx]
	# 	return arr1



	# def norm_angle(self, phi):
	# 	phi_new = np.zeros_like(phi, dtype=np.float64)
	# 	for i in range(phi.shape[0]):
	# 		for j in range(phi.shape[1]):
	# 			for k in range(phi.shape[2]):
	# 				tmp = np.argmin(np.abs(np.array([phi[i,j,k], -2 * np.pi + phi[i,j,k], phi[i,j,k] + 2 * np.pi])))
	# 				phi_new[i,j,k] = np.array([phi[i,j,k], (-2*np.pi + phi[i,j,k]), (phi[i,j,k] + 2 * np.pi)])[tmp]
	# 	return phi_new
