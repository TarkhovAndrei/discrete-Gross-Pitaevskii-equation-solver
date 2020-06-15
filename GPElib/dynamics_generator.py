'''
Copyright <2017> <Andrei E. Tarkhov, Skolkovo Institute of Science and Technology,
https://github.com/TarkhovAndrei/DGPE_ergodization_time>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following 2 conditions:

1) If any part of the present source code is used for any purposes followed by publication of obtained results,
the citation of the present code shall be provided according to the rule:

	"Andrei E. Tarkhov, Skolkovo Institute of Science and Technology,
	source code from the GitHub repository https://github.com/TarkhovAndrei/DGPE_ergodization_time
	was used to obtain the presented results, 2017."

2) The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.integrate as intgr
from scipy.integrate import odeint, solve_ivp
from scipy.sparse import dok_matrix
import multiprocessing as mp
from time import time

import torch
import torchdiffeq

from .gpu_dgpe_conservative import DGPE_ODE
from .gpu_dgpe_relaxation import DGPE_ODE_RELAXATION

class DynamicsGenerator(object):
	def __init__(self, **kwargs):
		#Hamiltonian parameters

		self.FloatPrecision = kwargs.get('FloatPrecision', np.float64)
		self.torch_FloatPrecision = kwargs.get('torch_FloatPrecision', torch.float64)
		torch.set_default_dtype(self.torch_FloatPrecision)
		self.torch_gpu_id = kwargs.get('gpu_id', 0)
		self.torch_device = torch.device('cuda:' + str(self.torch_gpu_id)
										 if torch.cuda.is_available() else 'cpu')

		self.gpu_integrator = kwargs.get('gpu_integrator', 'None')

		self.J = kwargs.get('J', 1.0)
		self.anisotropy = kwargs.get('anisotropy', 1.0)

		self.beta_amplitude = kwargs.get('beta', 0.01)
		self.gamma = kwargs.get('gamma', 0.01)

		self.use_matrix_operations = kwargs.get('use_matrix_operations', True)
		self.use_matrix_operations_for_energy = kwargs.get('use_matrix_operations_for_energy', True)
		self.h_ext_x = kwargs.get('h_ext_x', 0.)
		self.h_ext_y = kwargs.get('h_ext_y', 0.)
		self.lam1 = kwargs.get('lam1', 1.)
		self.lam2 = kwargs.get('lam2', 0.3)

		self.W = kwargs.get('W', 0.)
		self.N_tuple = kwargs.get('N_wells', 10)
		self.dimensionality = kwargs.get('dimensionality', 1)
		self.step = kwargs.get('step', 5.7e-05)
		self.time = kwargs.get('time', 1.4 * 50)

		self.n_steps = kwargs.get('n_steps', int(self.time / self.step))
		self.integrator = kwargs.get('integrator', 'personal')
		self.calculation_type = kwargs.get('calculation_type', 'lyap')
		self.integration_method = kwargs.get('intergration_method', 'RK45')
		self.smooth_quench = kwargs.get('smooth_quench', False)

		self.rtol = kwargs.get('rtol', 1e-6)
		self.atol = kwargs.get('atol', 1e-6)
		if self.calculation_type == 'lyap':
			self.n_steps_savings = 2
		else:
			self.n_steps_savings = self.n_steps

		self.Nx = 1
		self.Ny = 1
		self.Nz = 1
		if type(self.N_tuple) == type(5):
			self.Nx = self.N_tuple
			self.N_tuple = (self.Nx, self.Ny, self.Nz)
		elif len(self.N_tuple) == 2:
			self.Nx = self.N_tuple[0]
			self.Ny = self.N_tuple[1]
			self.N_tuple = (self.Nx, self.Ny, self.Nz)
		elif len(self.N_tuple) == 3:
			self.Nx = self.N_tuple[0]
			self.Ny = self.N_tuple[1]
			self.Nz = self.N_tuple[2]
		if self.Ny > 1:
			self.dimensionality = 2
		if self.Nz > 1:
			self.dimensionality = 3
		print("Geometry: ", self.N_tuple)
		self.N_wells = self.Nx * self.Ny * self.Nz

		self.wells_indices = [(i,j,k) for i in range(self.Nx) for j in range(self.Ny) for k in range(self.Nz)]

		self.wells_enumeration = np.arange(self.N_wells).reshape(self.N_tuple)

		self.nn_idx_1 = np.roll(self.wells_enumeration, -1, axis=0).flatten()
		self.nn_idx_2 = np.roll(self.wells_enumeration, 1, axis=0).flatten()
		self.nn_idy_1 = np.roll(self.wells_enumeration, -1, axis=1).flatten()
		self.nn_idy_2 = np.roll(self.wells_enumeration, 1, axis=1).flatten()
		self.nn_idz_1 = np.roll(self.wells_enumeration, -1, axis=2).flatten()
		self.nn_idz_2 = np.roll(self.wells_enumeration, 1, axis=2).flatten()

		self.wells_index_tuple_to_num = dict()
		for i in range(self.Nx):
			for j in range(self.Ny):
				for k in range(self.Nz):
					# self.wells_index_tuple_to_num[(i,j,k)] = i + self.Nx * (j + self.Ny * k)
					self.wells_index_tuple_to_num[(i,j,k)] = k + self.Nz * (j + self.Ny * i)
		#Seeds
		self.disorder_seed = kwargs.get('disorder_seed', 78)
		self.traj_seed = kwargs.get('traj_seed', 78)
		self.pert_seed = kwargs.get('pert_seed', 123)

		self.N_part = kwargs.get('N_part_per_well', 100000)
		self.N_part *= self.N_wells
		self.tau_char = kwargs.get('tau_char', 1.0 / np.sqrt(3. * self.beta_amplitude * self.J * self.N_part/self.N_wells))

		self.E_calibr = kwargs.get('E_calibr', 0)
		if self.dimensionality == 1:
			self.threshold_XY_to_polar = kwargs.get('threshold_XY_to_polar', 1.)
		else:
			self.threshold_XY_to_polar = kwargs.get('threshold_XY_to_polar', 0.5)

		self.energy = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.participation_rate = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.effective_nonlinearity = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.angular_momentum = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.number_of_particles = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.distance = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.histograms = {}
		self.rho_histograms = {}

		self.T = np.linspace(0, self.time, self.n_steps)
		self.RHO = np.zeros(self.N_tuple + (self.n_steps_savings,), dtype=self.FloatPrecision)
		self.THETA = np.zeros(self.N_tuple + (self.n_steps_savings,), dtype=self.FloatPrecision)
		self.X = np.zeros(self.N_tuple + (self.n_steps_savings,), dtype=self.FloatPrecision)
		self.Y = np.zeros(self.N_tuple + (self.n_steps_savings,), dtype=self.FloatPrecision)

		self.psi = np.zeros(2 * self.N_wells, dtype=self.FloatPrecision)

		self.psiNext = np.zeros(2 * self.N_wells, dtype=self.FloatPrecision)
		self.psiNextXY = np.zeros(2 * self.N_wells, dtype=self.FloatPrecision)
		self.psiJac = np.zeros(2 * self.N_wells, dtype=self.FloatPrecision)

		self.dpsi = np.zeros(2 * self.N_wells, dtype=self.FloatPrecision)

		# self.dFdXY = np.zeros((2 * self.N_wells, 2 * self.N_wells), dtype=self.FloatPrecision)
		self.dFdXY = dok_matrix((2 * self.N_wells, 2 * self.N_wells), dtype=self.FloatPrecision)

		self.dX = np.zeros(self.N_wells, dtype=self.FloatPrecision)
		self.dY = np.zeros(self.N_wells, dtype=self.FloatPrecision)

		self.xL = np.zeros(self.N_wells, dtype=self.FloatPrecision)
		self.yL = np.zeros(self.N_wells, dtype=self.FloatPrecision)

		self.disorder_in_interactions = False
		self.beta_disorder_amplitude = kwargs.get('beta_disorder_amplitude', 0.)
		self.beta_disorder_seed = kwargs.get('beta_disorder_seed', 1531)
		self.beta_disorder_array_flattened = np.zeros(self.N_wells, dtype=self.FloatPrecision)
		self.beta_disorder_array_volume = self.beta_disorder_array_flattened.reshape(self.N_tuple)

		self.local_disorder = False
		self.local_disorder_amplitude = kwargs.get('local_disorder_amplitude', 0.)
		self.local_disorder_seed = kwargs.get('local_disorder_seed', 1531)
		self.h_dis_x_flat = np.zeros(self.N_wells, dtype=self.FloatPrecision)
		self.h_dis_y_flat = np.zeros(self.N_wells, dtype=self.FloatPrecision)
		self.h_dis_x_volume = self.h_dis_x_flat.reshape(self.N_tuple)
		self.h_dis_y_volume = self.h_dis_y_flat.reshape(self.N_tuple)

		if 'local_disorder_amplitude' in kwargs:
			self.local_disorder = True
			np.random.seed(self.local_disorder_seed)
			self.h_dis_x_flat = self.local_disorder_amplitude * np.random.randn(self.N_wells)
			self.h_dis_y_flat = self.local_disorder_amplitude * np.random.randn(self.N_wells)
			self.h_dis_x_volume = self.h_dis_x_flat.reshape(self.N_tuple)
			self.h_dis_y_volume = self.h_dis_y_flat.reshape(self.N_tuple)
		else:
			self.local_disorder = False
			np.random.seed(self.local_disorder_seed)
			self.h_dis_x_flat = np.zeros(self.N_wells, dtype=self.FloatPrecision)
			self.h_dis_y_flat = np.zeros(self.N_wells, dtype=self.FloatPrecision)
			self.h_dis_x_volume = self.h_dis_x_flat.reshape(self.N_tuple)
			self.h_dis_y_volume = self.h_dis_y_flat.reshape(self.N_tuple)

		if 'beta_disorder_amplitude' in kwargs:
			self.disorder_in_interactions = True
			np.random.seed(self.beta_disorder_seed)
			self.beta_disorder_array_flattened = self.beta_disorder_amplitude * np.random.randn(self.N_wells)
			if 'beta_strong_disorder' in kwargs:
				self.beta_disorder_array_flattened = np.zeros(self.N_wells)
				idx_disorder = np.arange(self.N_wells, dtype=np.int64)
				np.random.shuffle(idx_disorder)
				self.idx_high_disorder = idx_disorder[:self.N_wells/100]
				self.idx_low_disorder = idx_disorder[self.N_wells/100:2*self.N_wells/100]
				self.beta_disorder_array_flattened[self.idx_high_disorder] = self.beta_amplitude * 0.9
				self.beta_disorder_array_flattened[self.idx_low_disorder] = - self.beta_amplitude * 0.9
			self.beta_disorder_array = self.beta_disorder_array_flattened.reshape(self.N_tuple)
		else:
			self.disorder_in_interactions = False
			np.random.seed(self.beta_disorder_seed)
			self.beta_disorder_array_flattened = np.zeros(self.N_wells)
			self.beta_disorder_array = self.beta_disorder_array_flattened.reshape(self.N_tuple)


		np.random.seed()

		self.beta_volume = self.beta_disorder_array + self.beta_amplitude
		self.beta_flat = self.beta_volume.flatten()
		self.beta = self.beta_flat.copy()


		self.tempered_glass_cooling = kwargs.get('tempered', False)
		if self.tempered_glass_cooling == True:
			self.gamma_slow = kwargs.get('gamma_slow', 0.01)
			self.gamma_fast = kwargs.get('gamma_fast', 100.)
			self.gamma_tempered = np.zeros(self.N_wells, dtype=self.FloatPrecision) + self.gamma_slow
			self.idx_borders = np.zeros(self.N_wells, dtype=np.bool)
			for i in range(self.Nx):
				for j in range(self.Ny):
					for k in range(self.Nz):
						curr_idx = self.wells_index_tuple_to_num[(i,j,k)]
						if (i == 0) or (i == self.Nx - 1):
							self.idx_borders[curr_idx] = True
						if (j == 0) or (j == self.Ny - 1):
							self.idx_borders[curr_idx] = True
						if (k == 0) or (k == self.Nz - 1):
							self.idx_borders[curr_idx] = True
			self.gamma_tempered[np.nonzero(self.idx_borders)[0]] = self.gamma_fast

		self.temperature_dependent_rate = False

		self.consistency_checksum = 0
		self.error_code = ""
		self.configure(kwargs)
		self.generate_disorder()

	def configure(self, kwargs):
		self.PERT_EPS = 1e-8
		self.FTOL = kwargs.get('FTOL', 1e-14)
		self.E_eps = kwargs.get('E_eps', 1e-2)
		self.singular_eps = 1e-8

	def set_pert_seed(self, pert_seed):
		self.pert_seed = pert_seed

	def generate_disorder(self):
		np.random.seed(self.disorder_seed)
		self.e_disorder = -self.W  + 2. * self.W * np.random.rand(self.N_tuple[0], self.N_tuple[1], self.N_tuple[2])
		self.e_disorder_flat = self.e_disorder.flatten()
		np.random.seed()

	def set_init_XY(self, x, y):
		self.X[:,:,:,0] = x.reshape(self.N_tuple)
		self.Y[:,:,:,0] = y.reshape(self.N_tuple)
		self.RHO[:,:,:,0], self.THETA[:,:,:,0] = self.from_XY_to_polar(self.X[:,:,:,0], self.Y[:,:,:,0])

	def from_polar_to_XY(self, rho, theta):
		rho = np.abs(rho)
		return rho * np.cos(theta), rho * np.sin(theta)

	def from_XY_to_polar(self, x, y):
		rho = np.sqrt((x ** 2) + (y ** 2))
		theta = np.arctan2(y, x)
		return rho, theta

	def constant_perturbation_XY(self, x0, y0):
		# np.random.seed(self.pert_seed)
		np.random.seed()
		# print "Seed: ", self.pert_seed
		eps = 1e-1
		x1 = x0 + eps * np.random.randn(self.N_tuple[0],self.N_tuple[1], self.N_tuple[2])
		y1 = y0 + eps * np.random.randn(self.N_tuple[0],self.N_tuple[1], self.N_tuple[2])
		dist = self.calc_traj_shift_XY(x0, y0, x1, y1)
		x1 = x0 + (x1 - x0) * self.PERT_EPS /dist
		y1 = y0 + (y1 - y0) * self.PERT_EPS /dist
		return x1, y1

	def generate_init(self, traj_seed, energy_per_site, kind='random'):
		np.random.seed(traj_seed)
		rho = np.array(np.sqrt(1.0 * self.N_part/self.N_wells) * np.ones(self.N_tuple))
		theta = np.zeros(self.N_tuple, dtype=self.FloatPrecision)
		if kind == 'random':
			print("random")
			theta += 2. * np.pi * np.random.rand(self.N_tuple[0], self.N_tuple[1], self.N_tuple[2])
		elif kind == 'random_population_and_phase':
			print("random_population_and_phase")
			theta += 2. * np.pi * np.random.rand(self.N_tuple[0], self.N_tuple[1], self.N_tuple[2])
			rho = np.random.rand(self.N_tuple[0], self.N_tuple[1], self.N_tuple[2])
			rho /= np.sqrt(np.sum(rho ** 2))
			rho *= np.sqrt(self.N_part)
		elif kind =='AF':
			for i in self.N_tuple:
				if i % 2 == 1:
					theta[i] = np.pi/2
				else:
					theta[i] = 0
			theta += 0.1 * np.pi * np.random.randn(self.N_tuple[0], self.N_tuple[1], self.N_tuple[2])
		elif kind =='FM':
			theta += 0.1 * np.pi * np.random.randn(self.N_tuple[0], self.N_tuple[1], self.N_tuple[2])

		self.RHO[:,:,:,0] = rho.reshape(self.N_tuple)
		self.THETA[:,:,:,0] = theta.reshape(self.N_tuple)
		self.X[:,:,:,0], self.Y[:,:,:,0] = self.from_polar_to_XY(self.RHO[:,:,:,0], self.THETA[:,:,:,0])
		self.E_calibr = 1.0 * energy_per_site * self.N_wells

	def rk4_step_exp(self, y0, *args):
		h = self.step
		self.psi = y0
		k1 = h * self.Hamiltonian_fast()

		y2 = y0 + (k1/2.)
		self.psi = y2
		k2 = h * self.Hamiltonian_fast()

		y3 = y0 + (k2/2.)
		self.psi = y3
		k3 = h * self.Hamiltonian_fast()

		y4 = y0 + k3
		self.psi = y4
		k4 = h * self.Hamiltonian_fast()

		yi = y0 + (k1 + 2.*k2 + 2.*k3 + k4)/6.
		return yi

	def rk4_step_exp_XY(self, y0, *args):
		h = self.step
		self.psi = y0
		k1 = h * self.HamiltonianXY_fast()

		y2 = y0 + (k1/2.)
		self.psi = y2
		k2 = h * self.HamiltonianXY_fast()

		y3 = y0 + (k2/2.)
		self.psi = y3
		k3 = h * self.HamiltonianXY_fast()

		y4 = y0 + k3
		self.psi = y4
		k4 = h * self.HamiltonianXY_fast()

		yi = y0 + (k1 + 2.*k2 + 2.*k3 + k4)/6.
		return yi

	def rk4_relax_step_exp(self, y0, *args):
		h = self.step
		self.psi = y0
		k1 = h * self.Relaxation_fast()

		y2 = y0 + (k1/2.)
		self.psi = y2
		k2 = h * self.Relaxation_fast()

		y3 = y0 + (k2/2.)
		self.psi = y3
		k3 = h * self.Relaxation_fast()

		y4 = y0 + k3
		self.psi = y4
		k4 = h * self.Relaxation_fast()

		yi = y0 + (k1 + 2.*k2 + 2.*k3 + k4)/6.
		return yi

	def rk4_relax_step_exp_old_slow(self, y0, *args):
		h = self.step
		k1 = h * self.Relaxation(y0)

		y2 = y0 + (k1/2.)
		k2 = h * self.Relaxation(y2)

		y3 = y0 + (k2/2.)
		k3 = h * self.Relaxation(y3)

		y4 = y0 + k3
		k4 = h * self.Relaxation(y4)

		yi = y0 + (k1 + 2.*k2 + 2.*k3 + k4)/6.
		return yi

	def rk4_slow_relax_step_exp(self, y0, *args):
		h = self.step
		self.psi = y0
		k1 = h * (self.Relaxation_fast() + self.Hamiltonian_fast())

		y2 = y0 + (k1/2.)
		self.psi = y2
		k2 = h * (self.Relaxation_fast() + self.Hamiltonian_fast())

		y3 = y0 + (k2/2.)
		self.psi = y3
		k3 = h * (self.Relaxation_fast() + self.Hamiltonian_fast())

		y4 = y0 + k3
		self.psi = y4
		k4 = h * (self.Relaxation_fast() + self.Hamiltonian_fast())

		yi = y0 + (k1 + 2.*k2 + 2.*k3 + k4)/6.
		return yi

	def rk4_relax_step_exp_XY(self, y0, *args):

		h = self.step
		self.psi = y0
		k1 = h * self.RelaxationXY_fast()

		y2 = y0 + (k1/2.)
		self.psi = y2
		k2 = h * self.RelaxationXY_fast()

		y3 = y0 + (k2/2.)
		self.psi = y3
		k3 = h * self.RelaxationXY_fast()

		y4 = y0 + k3
		self.psi = y4
		k4 = h * self.RelaxationXY_fast()

		yi = y0 + (k1 + 2.*k2 + 2.*k3 + k4)/6.
		return yi

	def rk4_slow_relax_step_exp_XY(self, y0, *args):

		h = self.step
		self.psi = y0
		k1 = h * (self.RelaxationXY_fast() + self.HamiltonianXY_fast())

		y2 = y0 + (k1 / 2.)
		self.psi = y2
		k2 = h * (self.RelaxationXY_fast() + self.HamiltonianXY_fast())

		y3 = y0 + (k2 / 2.)
		self.psi = y3
		k3 = h * (self.RelaxationXY_fast() + self.HamiltonianXY_fast())

		y4 = y0 + k3
		self.psi = y4
		k4 = h * (self.RelaxationXY_fast() + self.HamiltonianXY_fast())

		yi = y0 + (k1 + 2. * k2 + 2. * k3 + k4) / 6.
		return yi

	def run_dynamics_old(self):
		for i in range(1, self.n_steps):
			if (np.any(self.RHO[:,:,:,i-1] ** 2 < self.threshold_XY_to_polar)):
				psi = self.rk4_step_exp_XY(np.hstack((self.X[:,:,:,i-1].flatten(), self.Y[:,:,:,i-1].flatten())))
				self.X[:,:,:,i] = psi[:self.N_wells].reshape(self.N_tuple)
				self.Y[:,:,:,i] = psi[self.N_wells:].reshape(self.N_tuple)
				self.RHO[:,:,:,i], self.THETA[:,:,:,i] = self.from_XY_to_polar(self.X[:,:,:,i], self.Y[:,:,:,i])
			else:
				psi = self.rk4_step_exp(np.hstack((self.RHO[:,:,:,i-1].flatten(), self.THETA[:,:,:,i-1].flatten())))
				self.RHO[:,:,:,i] = psi[:self.N_wells].reshape(self.N_tuple)
				self.THETA[:,:,:,i] = psi[self.N_wells:].reshape(self.N_tuple)
				self.X[:,:,:,i], self.Y[:,:,:,i] = self.from_polar_to_XY(self.RHO[:,:,:,i], self.THETA[:,:,:,i])
		self.energy, self.number_of_particles, self.angular_momentum = self.calc_constants_of_motion(self.RHO, self.THETA, self.X, self.Y)

	def full_eq_of_motion(self, ts, y0):#y0, ts):
		self.psi = y0
		return self.RelaxationXY_fast(time=ts) + self.HamiltonianXY_fast(time=ts)

	def J_func_full_eq_of_motion(self, ts, y0):#y0, ts):
		self.psiJac = y0
		self.FullJacobianWithRelaxXY_fast()
		return self.dFdXY

	def full_eq_of_motion_conservative(self, ts, y0):#y0, ts):
		self.psi = y0
		return self.HamiltonianXY_fast()

	def J_func_full_eq_of_motion_conservative(self, ts, y0):#y0, ts):
		self.psiJac = y0
		gamma_tmp = self.gamma
		self.gamma = 0
		self.FullJacobianWithRelaxXY_fast()
		self.gamma = gamma_tmp
		return self.dFdXY

	def run_dynamics(self, no_pert=False):
		if self.gpu_integrator == 'torch':
			print('Running torch')

			psi0 = np.hstack((self.X[:, :, :, 0].flatten(), self.Y[:, :, :, 0].flatten()))
			ts = np.arange(self.n_steps, dtype=self.FloatPrecision) * self.step
			self.T[:self.n_steps] = ts

			conservative_ODE = DGPE_ODE(self.torch_device, self.N_wells, self.J, self.anisotropy, self.gamma,
				 self.nn_idx_1, self.nn_idx_2, self.nn_idy_1, self.nn_idy_2, self.nn_idz_1, self.nn_idz_2,
				 self.h_dis_x_flat, self.h_dis_y_flat,
				 self.beta_disorder_array_flattened, self.beta_flat, self.e_disorder_flat)

			ODE_result_object = torchdiffeq.odeint(
										  # self.torch_HamiltonianXY_fast,
										  conservative_ODE,
										  torch.from_numpy(psi0).type(self.torch_FloatPrecision).to(self.torch_device),#, dtype=self.torch_FloatPrecision),
									      torch.from_numpy(ts).type(self.torch_FloatPrecision).to(self.torch_device),#, dtype=self.torch_FloatPrecision),
										  rtol=self.rtol,
										  atol=self.atol
										)

			ODE_result = ODE_result_object.detach().cpu().numpy()

		elif self.integrator == 'scipy':
			print('Running scipy')
			psi0 = np.hstack((self.X[:,:,:,0].flatten(), self.Y[:,:,:,0].flatten()))
			ts = np.arange(self.n_steps, dtype=self.FloatPrecision) * self.step
			self.T[:self.n_steps] = ts

			ODE_result_object = solve_ivp(self.full_eq_of_motion_conservative, (np.min(ts), np.max(ts)+0.01), psi0,
										  method=self.integration_method,
										  rtol=self.rtol,
										  atol=self.atol,
										  # method='RK45',
										  # method='RK23',
										  # method='Radau',
										  # method='BDF',
										  # method='LSODA',
										  # rtol=1e-6, atol=1e-6,
										  # rtol=1e-6, atol=1e-6,
										  t_eval=ts, jac=self.J_func_full_eq_of_motion_conservative)

			ODE_result = ODE_result_object.y.T

			# ODE_result = odeint(self.full_eq_of_motion, psi0, ts, Dfun=self.J_func_full_eq_of_motion)
			# ODE_result = odeint(self.full_eq_of_motion, psi0, ts)#, Dfun=self.J_func_full_eq_of_motion)

			# ODE_result = odeint(self.full_eq_of_motion, psi0, ts, Dfun=self.J_func_full_eq_of_motion,
			#                     h0=1e-4, hmin=1e-5, hmax=1e-3)

		# ODE_result = odeint(self.full_eq_of_motion, psi0, ts, Dfun=self.J_func_full_eq_of_motion)
		# ODE_result = odeint(self.full_eq_of_motion, psi0, ts)#, Dfun=self.J_func_full_eq_of_motion)

		# ODE_result = odeint(self.full_eq_of_motion, psi0, ts, Dfun=self.J_func_full_eq_of_motion,
		#                     h0=1e-4, hmin=1e-5, hmax=1e-3)

		self.set_constants_of_motion_local(0, 0)


		if self.use_matrix_operations_for_energy:
			self.X = np.moveaxis(ODE_result[:,:self.N_wells], 0, -1).reshape(self.N_tuple + (ODE_result.shape[0],))
			self.Y = np.moveaxis(ODE_result[:,self.N_wells:], 0, -1).reshape(self.N_tuple + (ODE_result.shape[0],))
			self.icurr = self.n_steps - 1
			self.inext = self.n_steps
			self.energy = self.calc_energy_XY_global(ODE_result)
			self.number_of_particles = self.calc_nop_XY_global(ODE_result)
		else:
			icurr = 0
			inext = 1
			self.icurr = 0
			self.inext = 1
			for i in range(1, self.n_steps):
				if self.integrator == 'scipy':
					psi = ODE_result[i,:]
					self.X[:,:,:,inext] = psi[:self.N_wells].reshape(self.N_tuple)
					self.Y[:,:,:,inext] = psi[self.N_wells:].reshape(self.N_tuple)
					self.RHO[:,:,:,inext], self.THETA[:,:,:,inext] = self.from_XY_to_polar(self.X[:,:,:,inext], self.Y[:,:,:,inext])
				elif self.integrator == 'personal':
					if (np.any(self.RHO[:,:,:,icurr] ** 2 < self.threshold_XY_to_polar)):
						if (i == 1):
							self.psiNextXY = np.hstack((self.X[:, :, :, 0].flatten(), self.Y[:, :, :, 0].flatten()))
						psi = self.rk4_step_exp_XY(self.psiNextXY)
						self.psiNextXY = psi
						self.X[:,:,:,inext] = psi[:self.N_wells].reshape(self.N_tuple)
						self.Y[:,:,:,inext] = psi[self.N_wells:].reshape(self.N_tuple)
						self.RHO[:,:,:,inext], self.THETA[:,:,:,inext] = self.from_XY_to_polar(self.X[:,:,:,inext], self.Y[:,:,:,inext])
						self.psiNext = np.hstack((self.RHO[:,:,:,inext].flatten(), self.THETA[:,:,:,inext].flatten()))
					else:
						if (i == 1):
							self.psiNext = np.hstack((self.RHO[:,:,:,0].flatten(), self.THETA[:,:,:,0].flatten()))
						psi = self.rk4_step_exp(self.psiNext)
						self.psiNext = psi
						self.RHO[:,:,:,inext] = psi[:self.N_wells].reshape(self.N_tuple)
						self.THETA[:,:,:,inext] = psi[self.N_wells:].reshape(self.N_tuple)
						self.X[:,:,:,inext], self.Y[:,:,:,inext] = self.from_polar_to_XY(self.RHO[:,:,:,inext], self.THETA[:,:,:,inext])
						self.psiNextXY = np.hstack((self.X[:,:,:,inext].flatten(), self.Y[:,:,:,inext].flatten()))

				self.set_constants_of_motion_local(i, inext)

				if self.calculation_type == 'lyap':
					icurr = 1 - icurr
					inext = 1 - inext
					self.icurr = 1 - self.icurr
					self.inext = 1 - self.inext
				else:
					icurr = icurr + 1
					inext = inext + 1

					self.icurr = icurr + 1
					self.inext = inext + 1


	def run_quench(self, no_pert=False, E_desired=0,temperature_dependent_rate=False, N_max=1e+7):
		self.set_constants_of_motion_local(0, 0)

		Ecurr = self.energy[0]
		Enext = self.energy[0]

		if Ecurr < E_desired:
			if self.gamma > 0:
				self.gamma = - self.gamma
		else:
			if self.gamma < 0:
				self.gamma = - self.gamma
		icurr = 0
		inext = 1
		self.icurr = 0
		self.inext = 1

		i = 1
		while ((Ecurr - E_desired) * (Enext - E_desired) > 0) and (i < N_max):

			self.set_constants_of_motion_local(i - 1, icurr)
			Ecurr = self.energy[i-1]

			if (np.any(self.RHO[:, :, :, icurr] ** 2 < self.threshold_XY_to_polar)):
				if (i == 1):
					self.psiNextXY = np.hstack((self.X[:, :, :, 0].flatten(), self.Y[:, :, :, 0].flatten()))
				psi = self.rk4_relax_step_exp_XY(self.psiNextXY)
				self.psiNextXY = psi
				self.X[:, :, :, inext] = psi[:self.N_wells].reshape(self.N_tuple)
				self.Y[:, :, :, inext] = psi[self.N_wells:].reshape(self.N_tuple)
				self.RHO[:, :, :, inext], self.THETA[:, :, :, inext] = self.from_XY_to_polar(self.X[:, :, :, inext],
																							 self.Y[:, :, :, inext])
				self.psiNext = np.hstack((self.RHO[:, :, :, inext].flatten(), self.THETA[:, :, :, inext].flatten()))
			else:
				if (i == 1):
					self.psiNext = np.hstack((self.RHO[:, :, :, 0].flatten(), self.THETA[:, :, :, 0].flatten()))
				psi = self.rk4_relax_step_exp(self.psiNext)
				self.psiNext = psi
				self.RHO[:, :, :, inext] = psi[:self.N_wells].reshape(self.N_tuple)
				self.THETA[:, :, :, inext] = psi[self.N_wells:].reshape(self.N_tuple)
				self.X[:, :, :, inext], self.Y[:, :, :, inext] = self.from_polar_to_XY(self.RHO[:, :, :, inext],
																					   self.THETA[:, :, :, inext])
				self.psiNextXY = np.hstack((self.X[:, :, :, inext].flatten(), self.Y[:, :, :, inext].flatten()))


			self.set_constants_of_motion_local(i, inext)
			Enext = self.energy[i]

			if self.calculation_type == 'lyap':
				icurr = 1 - icurr
				inext = 1 - inext
				self.icurr = 1 - self.icurr
				self.inext = 1 - self.inext

			else:
				icurr = icurr + 1
				inext = inext + 1

				self.icurr = icurr + 1
				self.inext = inext + 1
			i += 1

	def quenching_profile(self, time=0.):
		return -self.gamma * 1./(self.lam1-self.lam2) * (self.lam1 * np.exp(-self.lam1 * time) - self.lam2 * np.exp(-self.lam2 * time))

	def get_gamma_reduction(self, psi, time=0.):
		if self.temperature_dependent_rate:
			if self.use_matrix_operations:
				return self.gamma_reduction * (self.calc_energy_XY(psi[:self.N_wells],psi[self.N_wells:],0) - self.E_desired)
			else:
				return self.gamma_reduction * (self.calc_energy_XY(psi[:self.N_wells].reshape(self.N_tuple),psi[self.N_wells:].reshape(self.N_tuple),0) - self.E_desired)
		else:
			return 1.

	def run_relaxation(self, no_pert=False, E_desired=0, temperature_dependent_rate=False, N_max=1e+7):

		self.set_constants_of_motion_local(0, 0)

		Ecurr = self.energy[0]
		Enext = self.energy[0]

		# self.temperature_dependent_rate = False

		self.E_desired = E_desired
		self.temperature_dependent_rate = temperature_dependent_rate
		self.gamma_reduction = 1./(Ecurr - self.E_desired)

		if (E_desired - Ecurr) * self.gamma > 0:
			self.gamma = -self.gamma

		# if Ecurr < E_desired:
		# 	if self.gamma > 0:
		# 		self.gamma = - self.gamma
		# else:
		# 	if self.gamma < 0:
		# 		self.gamma = - self.gamma

		if self.gpu_integrator == 'torch':
			print('Running torch')
			psi0 = np.hstack((self.X[:, :, :, 0].flatten(), self.Y[:, :, :, 0].flatten()))
			ts = np.arange(N_max, dtype=self.FloatPrecision) * self.step
			self.T[:N_max] = ts

			relaxational_ODE = DGPE_ODE_RELAXATION(self.torch_device, self.N_wells, self.J, self.anisotropy, self.gamma,
										self.nn_idx_1, self.nn_idx_2, self.nn_idy_1, self.nn_idy_2, self.nn_idz_1,
										self.nn_idz_2,
										self.h_dis_x_flat, self.h_dis_y_flat,
										self.beta_disorder_array_flattened, self.beta_flat, self.e_disorder_flat,
										self.E_desired, self.gamma_reduction, self.lam1, self.lam2, self.smooth_quench, self.temperature_dependent_rate)

			ODE_result_object = torchdiffeq.odeint(relaxational_ODE,
											# self.torch_Hamiltonian_with_Relaxation_XY_fast,
												   torch.from_numpy(psi0).type(self.torch_FloatPrecision).to(self.torch_device),#, dtype=self.torch_FloatPrecision),
												   torch.from_numpy(ts).type(self.torch_FloatPrecision).to(self.torch_device),# dtype=self.torch_FloatPrecision),
												   rtol=self.rtol,
												   atol=self.atol
												   )

			ODE_result = ODE_result_object.detach().cpu().numpy()

		elif self.integrator == 'scipy':
			print('Running scipy')
			psi0 = np.hstack((self.X[:, :, :, 0].flatten(), self.Y[:, :, :, 0].flatten()))
			ts = np.arange(N_max, dtype=self.FloatPrecision) * self.step
			self.T[:N_max] = ts
			# ODE_result = odeint(self.full_eq_of_motion, psi0, ts, Dfun=self.J_func_full_eq_of_motion,
			#                     h0=0.001, hmin=1e-5, hmax=5e-2)
			ODE_result_object = solve_ivp(self.full_eq_of_motion, (np.min(ts), np.max(ts)+0.01), psi0,
										  method=self.integration_method,
										  rtol=self.rtol,
										  atol=self.atol,
										  # method='RK45',
										  # method='RK23',
										  # method='Radau',
										  # method='BDF',
										  # method='LSODA',
										  # rtol=1e-6, atol=1e-6,
										  # rtol=1e-6, atol=1e-6,
										  t_eval=ts, jac=self.J_func_full_eq_of_motion)
			ODE_result = ODE_result_object.y.T

		if self.use_matrix_operations_for_energy:
			self.X = np.moveaxis(ODE_result[:,:self.N_wells], 0, -1).reshape(self.N_tuple + (ODE_result.shape[0],))
			self.Y = np.moveaxis(ODE_result[:,self.N_wells:], 0, -1).reshape(self.N_tuple + (ODE_result.shape[0],))
			self.energy = self.calc_energy_XY_global(ODE_result)
			self.number_of_particles = self.calc_nop_XY_global(ODE_result)
			idx_desired = np.nonzero((self.energy[:-1] - self.E_desired) * (self.energy[1:] - self.E_desired) < 0)[0]
			try:
				self.n_steps = idx_desired[0]
			except:
				self.n_steps = 1
			self.icurr = self.n_steps - 1
			self.inext = self.n_steps
		else:
			icurr = 0
			inext = 1
			self.icurr = 0
			self.inext = 1

			if self.integrator == 'scipy':
				i = 1
				while ((Ecurr - E_desired) * (Enext - E_desired) > 0) and (i < N_max):
					self.set_constants_of_motion_local(i - 1, icurr)
					Ecurr = self.energy[i - 1]
					psi = ODE_result[i, :]
					self.X[:, :, :, inext] = psi[:self.N_wells].reshape(self.N_tuple)
					self.Y[:, :, :, inext] = psi[self.N_wells:].reshape(self.N_tuple)
					self.RHO[:, :, :, inext], self.THETA[:, :, :, inext] = self.from_XY_to_polar(self.X[:, :, :, inext],
																								 self.Y[:, :, :, inext])

					self.set_constants_of_motion_local(i, inext)
					Enext = self.energy[i]

					if self.calculation_type == 'lyap':
						icurr = 1 - icurr
						inext = 1 - inext
						self.icurr = 1 - self.icurr
						self.inext = 1 - self.inext
					else:
						icurr = icurr + 1
						inext = inext + 1

						self.icurr = icurr + 1
						self.inext = inext + 1
					i += 1
				self.n_steps = i

			elif self.integrator == 'personal':
				i = 1
				while ((Ecurr - E_desired) * (Enext - E_desired) > 0) and (i < N_max):
					self.set_constants_of_motion_local(i - 1, icurr)
					Ecurr = self.energy[i-1]

					if (np.any(self.RHO[:, :, :, icurr] ** 2 < self.threshold_XY_to_polar)):
						if (i == 1):
							self.psiNextXY = np.hstack((self.X[:, :, :, 0].flatten(), self.Y[:, :, :, 0].flatten()))
						psi = self.rk4_slow_relax_step_exp_XY(self.psiNextXY)
						self.psiNextXY = psi
						self.X[:, :, :, inext] = psi[:self.N_wells].reshape(self.N_tuple)
						self.Y[:, :, :, inext] = psi[self.N_wells:].reshape(self.N_tuple)
						self.RHO[:, :, :, inext], self.THETA[:, :, :, inext] = self.from_XY_to_polar(self.X[:, :, :, inext],
																									 self.Y[:, :, :, inext])
						self.psiNext = np.hstack((self.RHO[:, :, :, inext].flatten(), self.THETA[:, :, :, inext].flatten()))
					else:
						if (i == 1):
							self.psiNext = np.hstack((self.RHO[:, :, :, 0].flatten(), self.THETA[:, :, :, 0].flatten()))
						psi = self.rk4_slow_relax_step_exp(self.psiNext)
						self.psiNext = psi
						self.RHO[:, :, :, inext] = psi[:self.N_wells].reshape(self.N_tuple)
						self.THETA[:, :, :, inext] = psi[self.N_wells:].reshape(self.N_tuple)
						self.X[:, :, :, inext], self.Y[:, :, :, inext] = self.from_polar_to_XY(self.RHO[:, :, :, inext],
																							   self.THETA[:, :, :, inext])
						self.psiNextXY = np.hstack((self.X[:, :, :, inext].flatten(), self.Y[:, :, :, inext].flatten()))

					self.set_constants_of_motion_local(i, inext)
					Enext = self.energy[i]

					if self.calculation_type == 'lyap':
						icurr = 1 - icurr
						inext = 1 - inext
						self.icurr = 1 - self.icurr
						self.inext = 1 - self.inext

					else:
						icurr = icurr + 1
						inext = inext + 1

						self.icurr = icurr + 1
						self.inext = inext + 1
					i += 1

				self.n_steps = i

		self.temperature_dependent_rate = False

	def reverse_hamiltonian(self, error_J, error_beta, error_disorder):
		self.J = -1. * self.J * (1.0 + error_J * np.random.randn())
		self.beta = -1. * self.beta * (1.0 + error_beta * np.random.randn())
		self.e_disorder = -1. * self.e_disorder * (1.0 + error_disorder * np.random.randn())
		self.e_disorder_flat = self.e_disorder.flatten()

	def Hamiltonian_fast(self):
		self.dpsi *= 0

		for itup in self.wells_indices:
			i = self.wells_index_tuple_to_num[itup]
			self.dpsi[i + self.N_wells] += - self.beta_flat[i] * (self.psi[i]**2) - self.e_disorder[itup]
			for idx, jtup in enumerate(self.nearest_neighbours(itup)):
				j = self.wells_index_tuple_to_num[jtup]
				# Introduce anisotropy of J for the 3rd axis
				if idx > 3:
					self.dpsi[i] -= self.anisotropy * self.J * (self.psi[j] * np.sin(self.psi[j + self.N_wells] - self.psi[i + self.N_wells]))
					self.dpsi[i + self.N_wells] += self.anisotropy * self.J  * (self.psi[j] * np.cos(self.psi[j + self.N_wells] - self.psi[i + self.N_wells])) / self.psi[i]
				else:
					self.dpsi[i] -= self.J * (self.psi[j] * np.sin(self.psi[j + self.N_wells] - self.psi[i + self.N_wells]))
					self.dpsi[i + self.N_wells] += self.J  * (self.psi[j] * np.cos(self.psi[j + self.N_wells] - self.psi[i + self.N_wells])) / self.psi[i]
		return self.dpsi.copy()

	def Relaxation_fast(self):

		self.dpsi *= 0

		self.xL *= 0
		self.yL *= 0

		for itup in self.wells_indices:
			i = self.wells_index_tuple_to_num[itup]
			# calculating the local field (xL, yL)
			for idx, jtup in enumerate(self.nearest_neighbours(itup)):
				j = self.wells_index_tuple_to_num[jtup]
				# Introduce anisotropy of J for the 3rd axis
				if idx > 3:
					self.xL[i] += self.anisotropy * self.J * self.psi[j] * np.cos(self.psi[j + self.N_wells])
					self.yL[i] += self.anisotropy * self.J * self.psi[j] * np.sin(self.psi[j + self.N_wells])
				else:
					self.xL[i] += self.J * self.psi[j] * np.cos(self.psi[j + self.N_wells])
					self.yL[i] += self.J * self.psi[j] * np.sin(self.psi[j + self.N_wells])
		if self.tempered_glass_cooling == True:
			self.dpsi[:self.N_wells] = 0
			self.dpsi[self.N_wells:] = - self.gamma_tempered * self.psi[:self.N_wells] * (self.xL * np.sin(self.psi[self.N_wells:]) - self.yL * np.cos(self.psi[self.N_wells:]))
		else:
			self.dpsi[:self.N_wells] = 0
			self.dpsi[self.N_wells:] = - self.gamma * self.psi[:self.N_wells] * (self.xL * np.sin(self.psi[self.N_wells:]) - self.yL * np.cos(self.psi[self.N_wells:]))
		return self.dpsi.copy()

	def effective_frequency(self, X0, Y0):
		return self.E_calibr

	def NN(self, i):
		j = []
		for idx in range(len(i)):
			if i[idx] < 0:
				j.append(self.N_tuple[idx] - 1)
			elif i[idx] == self.N_tuple[idx]:
				j.append(0)
			else:
				j.append(i[idx])
		return tuple(j)

	def nearest_neighbours(self, i):
		if self.dimensionality == 1:
			return [self.NN( (i[0] + 1, i[1], i[2]) ), self.NN( (i[0] - 1, i[1], i[2]) )]
		elif self.dimensionality == 2:
			return [self.NN( (i[0] + 1, i[1], i[2]) ), self.NN( (i[0] - 1, i[1], i[2]) ),
					self.NN( (i[0], i[1] + 1, i[2]) ), self.NN( (i[0], i[1] - 1, i[2]) )]
		elif self.dimensionality == 3:
			return [self.NN( (i[0] + 1, i[1], i[2]) ), self.NN( (i[0] - 1, i[1], i[2]) ),
					self.NN( (i[0], i[1] + 1, i[2]) ), self.NN( (i[0], i[1] - 1, i[2]) ),
					self.NN( (i[0], i[1], i[2]-1) ), self.NN( (i[0], i[1], i[2]+1) )]
		else:
			return 0

	def HamiltonianXY_fast(self, time=0.):

		self.dpsi *= 0

		if self.use_matrix_operations:
			self.dpsi[:self.N_wells] += self.e_disorder_flat * self.psi[self.N_wells:]
			self.dpsi[self.N_wells:] -= self.e_disorder_flat * self.psi[:self.N_wells]

			self.dpsi[:self.N_wells] += -self.J * (self.psi[self.N_wells:][self.nn_idx_1] +
					   							   self.psi[self.N_wells:][self.nn_idx_2] +
												   self.psi[self.N_wells:][self.nn_idy_1] +
												   self.psi[self.N_wells:][self.nn_idy_2] +
							    self.anisotropy * (self.psi[self.N_wells:][self.nn_idz_1] +
								 				   self.psi[self.N_wells:][self.nn_idz_2]
								   					  )
												   )

			self.dpsi[self.N_wells:] += self.J * (
											self.psi[:self.N_wells][self.nn_idx_1] +
					   						self.psi[:self.N_wells][self.nn_idx_2] +
											self.psi[:self.N_wells][self.nn_idy_1] +
					   						self.psi[:self.N_wells][self.nn_idy_2] +
						 self.anisotropy * (self.psi[:self.N_wells][self.nn_idz_1] +
											self.psi[:self.N_wells][self.nn_idz_2]
										   )
												)
		else:
			for itup in self.wells_indices:
				i = self.wells_index_tuple_to_num[itup]

				self.dpsi[i] += self.e_disorder[itup] * self.psi[i + self.N_wells]
				self.dpsi[i + self.N_wells] += - self.e_disorder[itup] * self.psi[i]
				for idx, jtup in enumerate(self.nearest_neighbours(itup)):
					# Introduce anisotropy of J for the 3rd axis
					j = self.wells_index_tuple_to_num[jtup]
					if idx > 3:
						self.dpsi[i] += - self.anisotropy * self.J * self.psi[j + self.N_wells]
						self.dpsi[i + self.N_wells] += self.anisotropy * self.J * self.psi[j]
					else:
						self.dpsi[i] += -self.J * self.psi[j + self.N_wells]
						self.dpsi[i + self.N_wells] += self.J * self.psi[j]

		# self.dpsi[i] += - self.h_ext_y
		# self.dpsi[i + self.N_wells] += self.h_ext_x
		self.dpsi[:self.N_wells] += self.psi[self.N_wells:] * (
					self.h_ext_x * self.psi[self.N_wells:] - self.h_ext_y * self.psi[:self.N_wells])
		self.dpsi[self.N_wells:] += -self.psi[:self.N_wells] * (
					self.h_ext_x * self.psi[self.N_wells:] - self.h_ext_y * self.psi[:self.N_wells])

		self.dpsi[:self.N_wells] += self.h_dis_y_flat
		self.dpsi[self.N_wells:] += -self.h_dis_x_flat

		self.dpsi[:self.N_wells] += self.beta_flat * (
					(self.psi[self.N_wells:] ** 2) + (self.psi[:self.N_wells] ** 2)) * self.psi[self.N_wells:]
		self.dpsi[self.N_wells:] += - self.beta_flat * (
					(self.psi[self.N_wells:] ** 2) + (self.psi[:self.N_wells] ** 2)) * self.psi[:self.N_wells]

		return self.dpsi.copy()

	def RelaxationXY_fast(self, time=0.):
		self.dpsi *= 0

		self.xL *= 0
		self.yL *= 0

		if self.use_matrix_operations:

			self.xL += self.J * (
								   self.psi[:self.N_wells][self.nn_idx_1] +
								   self.psi[:self.N_wells][self.nn_idx_2] +
								   self.psi[:self.N_wells][self.nn_idy_1] +
								   self.psi[:self.N_wells][self.nn_idy_2] +
				self.anisotropy * (self.psi[:self.N_wells][self.nn_idz_1] +
								   self.psi[:self.N_wells][self.nn_idz_2]
									  )
								   )
			self.yL += self.J * (
											self.psi[self.N_wells:][self.nn_idx_1] +
					   						self.psi[self.N_wells:][self.nn_idx_2] +
											self.psi[self.N_wells:][self.nn_idy_1] +
					   						self.psi[self.N_wells:][self.nn_idy_2] +
						 self.anisotropy * (self.psi[self.N_wells:][self.nn_idz_1] +
											self.psi[self.N_wells:][self.nn_idz_2]
										   )
												)

		else:
			for itup in self.wells_indices:
				i = self.wells_index_tuple_to_num[itup]
				# calculating the local field (xL, yL)
				for idx, jtup in enumerate(self.nearest_neighbours(itup)):
					j = self.wells_index_tuple_to_num[jtup]
					# Introduce anisotropy of J for the 3rd axis
					if idx > 3:
						self.xL[i] += self.anisotropy * self.J * self.psi[j]
						self.yL[i] += self.anisotropy * self.J * self.psi[j + self.N_wells]
					else:
						self.xL[i] += self.J * self.psi[j]
						self.yL[i] += self.J * self.psi[j + self.N_wells]

		if self.tempered_glass_cooling == True:
			self.dpsi[:self.N_wells] += self.gamma_tempered * self.psi[self.N_wells:] * (
						self.xL * self.psi[self.N_wells:] - self.yL * self.psi[:self.N_wells])
			self.dpsi[self.N_wells:] += -self.gamma_tempered * self.psi[:self.N_wells] * (
						self.xL * self.psi[self.N_wells:] - self.yL * self.psi[:self.N_wells])
		else:
			self.dpsi[:self.N_wells] += self.gamma * self.psi[self.N_wells:] * (
						self.xL * self.psi[self.N_wells:] - self.yL * self.psi[:self.N_wells])
			self.dpsi[self.N_wells:] += -self.gamma * self.psi[:self.N_wells] * (
						self.xL * self.psi[self.N_wells:] - self.yL * self.psi[:self.N_wells])

		if self.temperature_dependent_rate:
			if self.smooth_quench:
				self.dpsi = self.quenching_profile(time=time) * self.dpsi
			else:
				self.dpsi = self.get_gamma_reduction(self.psi, time=time) * self.dpsi

		return self.dpsi.copy()

	def HamiltonianXY_fast_old(self):

		self.dpsi *= 0

		for itup in self.wells_indices:
			i = self.wells_index_tuple_to_num[itup]

			self.dpsi[i] += self.e_disorder[itup] * self.psi[i+self.N_wells]
			self.dpsi[i + self.N_wells] += - self.e_disorder[itup] * self.psi[i]
			for idx, jtup in enumerate(self.nearest_neighbours(itup)):
				# Introduce anisotropy of J for the 3rd axis
				j = self.wells_index_tuple_to_num[jtup]
				if idx > 3:
					self.dpsi[i] += - self.anisotropy * self.J * self.psi[j+self.N_wells]
					self.dpsi[i + self.N_wells] += self.anisotropy * self.J * self.psi[j]
				else:
					self.dpsi[i] += -self.J * self.psi[j+self.N_wells]
					self.dpsi[i + self.N_wells] += self.J * self.psi[j]

		# self.dpsi[i] += - self.h_ext_y
		# self.dpsi[i + self.N_wells] += self.h_ext_x
		self.dpsi[:self.N_wells] += self.psi[self.N_wells:] * (self.h_ext_x * self.psi[self.N_wells:] - self.h_ext_y * self.psi[:self.N_wells])
		self.dpsi[self.N_wells:] += -self.psi[:self.N_wells] * (self.h_ext_x * self.psi[self.N_wells:] - self.h_ext_y * self.psi[:self.N_wells])

		self.dpsi[:self.N_wells] += self.beta_flat * ((self.psi[self.N_wells:] ** 2) + (self.psi[:self.N_wells] ** 2)) * self.psi[self.N_wells:]
		self.dpsi[self.N_wells:] += - self.beta_flat * ((self.psi[self.N_wells:] ** 2) + (self.psi[:self.N_wells] ** 2)) * self.psi[:self.N_wells]

		return self.dpsi.copy()


	def RelaxationXY_fast_old(self):
		self.dpsi *= 0

		self.xL *= 0
		self.yL *= 0

		for itup in self.wells_indices:
			i = self.wells_index_tuple_to_num[itup]
			# calculating the local field (xL, yL)
			for idx, jtup in enumerate(self.nearest_neighbours(itup)):
				j = self.wells_index_tuple_to_num[jtup]
				# Introduce anisotropy of J for the 3rd axis
				if idx > 3:
					self.xL[i] += self.anisotropy * self.J * self.psi[j]
					self.yL[i] += self.anisotropy * self.J * self.psi[j + self.N_wells]
				else:
					self.xL[i] += self.J * self.psi[j]
					self.yL[i] += self.J * self.psi[j + self.N_wells]
		if self.tempered_glass_cooling == True:
			self.dpsi[:self.N_wells] += self.gamma_tempered * self.psi[self.N_wells:] * (self.xL * self.psi[self.N_wells:] - self.yL * self.psi[:self.N_wells])
			self.dpsi[self.N_wells:] += -self.gamma_tempered * self.psi[:self.N_wells] * (self.xL * self.psi[self.N_wells:] - self.yL * self.psi[:self.N_wells])
		else:
			self.dpsi[:self.N_wells] += self.gamma * self.psi[self.N_wells:] * (self.xL * self.psi[self.N_wells:] - self.yL * self.psi[:self.N_wells])
			self.dpsi[self.N_wells:] += -self.gamma * self.psi[:self.N_wells] * (self.xL * self.psi[self.N_wells:] - self.yL * self.psi[:self.N_wells])
		self.dpsi = self.get_gamma_reduction(self.psi) * self.dpsi

		return self.dpsi.copy()

	def index_tuple_to_num(self, idx):
		return idx[0] + self.N_tuple[1] * (idx[1] + idx[2] * self.N_tuple[2])

	def FullJacobianWithRelaxXY_fast(self):
		self.dFdXY *= 0

		self.xL *= 0
		self.yL *= 0

		for itup in self.wells_indices:
			i = self.wells_index_tuple_to_num[itup]
			# calculating the local field (xL, yL)
			for idx, jtup in enumerate(self.nearest_neighbours(itup)):
				j = self.wells_index_tuple_to_num[jtup]
				# Introduce anisotropy of J for the 3rd axis
				if idx > 3:
					self.xL[i] += self.anisotropy * self.J * self.psiJac[j]
					self.yL[i] += self.anisotropy * self.J * self.psiJac[j + self.N_wells]
				else:
					self.xL[i] += self.J * self.psiJac[j]
					self.yL[i] += self.J * self.psiJac[j + self.N_wells]

		for itup in self.wells_indices:
			i = self.wells_index_tuple_to_num[itup]
			# dXi / dXj
			self.dFdXY[i, i] += 2. * self.beta_flat[i] * self.psiJac[i] * self.psiJac[i + self.N_wells] + self.gamma * self.psiJac[i + self.N_wells] * self.yL[i]
			# dXi / dYj
			self.dFdXY[i,i + self.N_wells] += self.beta_flat[i] * (self.psiJac[i] ** 2 + 3. * self.psiJac[i + self.N_wells] **2) - self.gamma * (2. * self.xL[i] * self.psiJac[i + self.N_wells] - self.yL[i] * self.psiJac[i])

			# dYi / dYj
			self.dFdXY[i+ self.N_wells,i + self.N_wells] += - 2. * self.beta_flat[i] * self.psiJac[i] * self.psiJac[i + self.N_wells] - self.gamma * self.psiJac[i] * self.xL[i]
			# dYi / dXj
			self.dFdXY[i+ self.N_wells,i] += - self.beta_flat[i] * (3. * self.psiJac[i] ** 2 + self.psiJac[i + self.N_wells] **2)  - self.gamma * (2. * self.yL[i] * self.psiJac[i] - self.xL[i] * self.psiJac[i + self.N_wells])

			for idx, jtup in enumerate(self.nearest_neighbours(itup)):
				# Introduce anisotropy of J for the 3rd axis
				if idx > 3:
					j = self.wells_index_tuple_to_num[jtup]
					self.dFdXY[i, j + self.N_wells] += - self.anisotropy * self.J + self.anisotropy * self.gamma * self.psiJac[i] * self.psiJac[i + self.N_wells]
					self.dFdXY[i+ self.N_wells, j] += self.anisotropy * self.J + self.anisotropy * self.gamma * self.psiJac[i] * self.psiJac[i + self.N_wells]

					self.dFdXY[i, j] += - self.anisotropy * self.gamma * (self.psiJac[i + self.N_wells] ** 2)
					self.dFdXY[i + self.N_wells, j + self.N_wells] += self.anisotropy * self.gamma * (self.psiJac[i] ** 2)
				else:
					j = self.wells_index_tuple_to_num[jtup]
					self.dFdXY[i, j + self.N_wells] += -self.J + self.gamma * self.psiJac[i] * self.psiJac[i + self.N_wells]
					self.dFdXY[i+ self.N_wells, j] += self.J + self.gamma * self.psiJac[i] * self.psiJac[i + self.N_wells]

					self.dFdXY[i, j] += - self.gamma * (self.psiJac[i + self.N_wells] ** 2)
					self.dFdXY[i + self.N_wells, j + self.N_wells] += self.gamma * (self.psiJac[i] ** 2)

	def FullJacobianWithRelaxXY(self, X, Y):
		dFdXY = np.zeros((2 * self.N_wells, 2 * self.N_wells))

		X0 = np.array(X)
		Y0 = np.array(Y)

		# X0 = psi[:self.N_wells].reshape(self.N_tuple)
		# Y0 = psi[self.N_wells:].reshape(self.N_tuple)

		xL = np.zeros(self.N_tuple, dtype=self.FloatPrecision)
		yL = np.zeros(self.N_tuple, dtype=self.FloatPrecision)

		for i in self.wells_indices:
			# calculating the local field (xL, yL)
			for idx, j in enumerate(self.nearest_neighbours(i)):
				# Introduce anisotropy of J for the 3rd axis
				if idx > 3:
					xL[i] += self.anisotropy * self.J * X0[j]
					yL[i] += self.anisotropy * self.J * Y0[j]
				else:
					xL[i] += self.J * X0[j]
					yL[i] += self.J * Y0[j]

		for itup in self.wells_indices:
			i = self.wells_index_tuple_to_num[itup]
			# dXi / dXj
			dFdXY[i, i] += 2. * self.beta_flat[i] * X0[itup] * Y0[itup] + self.gamma * Y0[itup] * yL[itup]
			# dXi / dYj
			dFdXY[i,i + self.N_wells] += self.beta_flat[i] * (X0[itup] ** 2 + 3. * Y0[itup] **2) - self.gamma * (2. * xL[itup] * Y0[itup] - yL[itup] * X0[itup])

			# dYi / dYj
			dFdXY[i+ self.N_wells,i + self.N_wells] += - 2. * self.beta_flat[i] * X0[itup] * Y0[itup] - self.gamma * X0[itup] * xL[itup]
			# dYi / dXj
			dFdXY[i+ self.N_wells,i] += - self.beta_flat[i] * (3. * X0[itup] ** 2 + Y0[itup] **2)  - self.gamma * (2. * yL[itup] * X0[itup] - xL[itup] * Y0[itup])

			for idx, jtup in enumerate(self.nearest_neighbours(itup)):
				# Introduce anisotropy of J for the 3rd axis
				if idx > 3:
					j = self.wells_index_tuple_to_num[jtup]
					dFdXY[i, j + self.N_wells] += - self.anisotropy * self.J + self.anisotropy * self.gamma * X0[itup] * Y0[itup]
					dFdXY[i+ self.N_wells, j] += self.anisotropy * self.J + self.anisotropy * self.gamma * X0[itup] * Y0[itup]

					dFdXY[i, j] += - self.anisotropy * self.gamma * (Y0[itup] ** 2)
					dFdXY[i + self.N_wells, j + self.N_wells] += self.anisotropy * self.gamma * (X0[itup] ** 2)
				else:
					j = self.wells_index_tuple_to_num[jtup]
					dFdXY[i, j + self.N_wells] += -self.J + self.gamma * X0[itup] * Y0[itup]
					dFdXY[i+ self.N_wells, j] += self.J + self.gamma * X0[itup] * Y0[itup]

					dFdXY[i, j] += - self.gamma * (Y0[itup] ** 2)
					dFdXY[i + self.N_wells, j + self.N_wells] += self.gamma * (X0[itup] ** 2)

		#eig_vals = np.linalg.eigvals(dFdXY)

		return dFdXY#, eig_vals


	def JacobianXY(self, X, Y):
		X0 = np.array(X)
		Y0 = np.array(Y)

		dFdXY = np.zeros((2 * self.N_wells, 2 * self.N_wells))

		for itup in self.wells_indices:
			i = self.wells_index_tuple_to_num[itup]
			# dXi / dXj
			dFdXY[i, i] += 2. * self.beta_flat[i] * X0[itup] * Y0[itup]
			# dXi / dYj
			dFdXY[i,i + self.N_wells] += self.beta_flat[i] * (X0[itup] ** 2 + 3. * Y0[itup] **2)

			# dYi / dYj
			dFdXY[i+ self.N_wells,i + self.N_wells] += - 2. * self.beta_flat[i] * X0[itup] * Y0[itup]
			# dYi / dXj
			dFdXY[i+ self.N_wells,i] += - self.beta_flat[i] * (3. * X0[itup] ** 2 + Y0[itup] **2)

			for idx, jtup in enumerate(self.nearest_neighbours(itup)):
				# Introduce anisotropy of J for the 3rd axis
				if idx > 3:
					j = self.wells_index_tuple_to_num[jtup]
					dFdXY[i, j + self.N_wells] += - self.anisotropy * self.J
					dFdXY[i+ self.N_wells, j] += self.anisotropy * self.J
				else:
					j = self.wells_index_tuple_to_num[jtup]
					dFdXY[i, j + self.N_wells] += -self.J
					dFdXY[i+ self.N_wells, j] += self.J

		#eig_vals = np.linalg.eigvals(dFdXY)

		return dFdXY#, eig_vals


	def calc_constants_of_motion(self, RHO, THETA, X, Y):
		number_of_particles = np.sum(RHO ** 2, axis=(0,1,2))
		energy = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		angular_momentum = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		for j in self.wells_indices:
			energy += (self.beta_volume[j]/2. * np.abs(RHO[j]**4) +
					   self.e_disorder[j] * np.abs(RHO[j]**2))
			for idx, k in enumerate(self.nearest_neighbours(j)):
				# Introduce anisotropy of J for the 3rd axis
				if idx > 3:
					energy += (- self.anisotropy * self.J * (RHO[k] * RHO[j] * np.cos(THETA[k] - THETA[j])))
				else:
					energy += (- self.J * (RHO[k] * RHO[j] * np.cos(THETA[k] - THETA[j])))

			# angular_momentum += - 2 * self.J * (X[:,j] * (0*Y[:,self.NN(j-1)] + Y[:,self.NN(j+1)]) - Y[:,j] * (0*X[:,self.NN(j-1)] + X[:,self.NN(j+1)]))
			# angular_momentum += - 2 * self.J * (X[:,j] * (0*Y[:,self.NN(j-1)] + Y[:,self.NN(j+1)]) - 0*Y[:,j] * (0*X[:,self.NN(j-1)] + X[:,self.NN(j+1)]))
			# angular_momentum += - 2 * self.J * (X[:,j] * (1./3*X[:,j] ** 2 + Y[:,j] ** 2))
			angular_momentum += - 2 * self.J * (X[j] * Y[k] - Y[j] * X[k])

		return energy, number_of_particles, angular_momentum

	def calc_constants_of_motion_local(self, RHO, THETA, X, Y):
		number_of_particles = np.sum(RHO ** 2, axis=(0,1,2))
		energy = 0.#np.zeros(self.n_steps, dtype=self.FloatPrecision)
		angular_momentum = 0.#np.zeros(self.n_steps, dtype=self.FloatPrecision)
		for j in self.wells_indices:
			energy += (self.beta_volume[j]/2. * np.abs(RHO[j]**4) +
					   self.e_disorder[j] * np.abs(RHO[j]**2))

			for idx, k in enumerate(self.nearest_neighbours(j)):
				# Introduce anisotropy of J for the 3rd axis
				if idx > 3:
					energy += (- self.anisotropy * self.J * (RHO[k] * RHO[j] * np.cos(THETA[k] - THETA[j])))
				else:
					energy += (- self.J * (RHO[k] * RHO[j] * np.cos(THETA[k] - THETA[j])))
			# angular_momentum += - 2 * self.J * (X[:,j] * (0*Y[:,self.NN(j-1)] + Y[:,self.NN(j+1)]) - Y[:,j] * (0*X[:,self.NN(j-1)] + X[:,self.NN(j+1)]))
			# angular_momentum += - 2 * self.J * (X[:,j] * (0*Y[:,self.NN(j-1)] + Y[:,self.NN(j+1)]) - 0*Y[:,j] * (0*X[:,self.NN(j-1)] + X[:,self.NN(j+1)]))
			# angular_momentum += - 2 * self.J * (X[:,j] * (1./3*X[:,j] ** 2 + Y[:,j] ** 2))
			angular_momentum += - 2 * self.J * (X[j] * Y[k] - Y[j] * X[k])

		return energy, number_of_particles, angular_momentum

	def set_constants_of_motion_local(self, i, inext):
		self.energy[i], self.number_of_particles[i], self.angular_momentum[i] = self.calc_constants_of_motion_local(self.RHO[:,:,:,inext], self.THETA[:,:,:,inext],
																													self.X[:,:,:,inext], self.Y[:,:,:,inext])
		# for i in self.wells_indices:
		# 	self.histograms[i] = np.histogram2d(np.float64(self.X[i]), np.float64(self.Y[i]), bins=100)
		# 	self.rho_histograms[i] = np.histogram(np.float64(self.RHO[i] ** 2), bins=100)

		self.participation_rate[i] = np.sum(self.RHO[:,:,:,inext] ** 4, axis=(0,1,2)) / (np.sum(self.RHO[:,:,:,inext] ** 2, axis=(0,1,2)) ** 2)
		self.effective_nonlinearity[i] = self.beta_amplitude * (self.participation_rate[i]) / self.N_wells

	def set_constants_of_motion(self):
		self.energy, self.number_of_particles, self.angular_momentum = self.calc_constants_of_motion(self.RHO, self.THETA, self.X, self.Y)
		for i in self.wells_indices:
			self.histograms[i] = np.histogram2d(np.float64(self.X[i]), np.float64(self.Y[i]), bins=100)
			self.rho_histograms[i] = np.histogram(np.float64(self.RHO[i] ** 2), bins=100)

		self.participation_rate = np.sum(self.RHO ** 4, axis=(0,1,2)) / (np.sum(self.RHO ** 2, axis=(0,1,2)) ** 2)
		self.effective_nonlinearity = self.beta_amplitude * (self.participation_rate) / self.N_wells

	def calc_traj_shift_XY(self, x0, y0, x1, y1):
		return np.sqrt(np.sum( ((x0 - x1) ** 2 + (y0 - y1) ** 2).flatten() ))

	def calc_energy_XY(self, x, y, E):
		E_new = -E
		if (self.use_matrix_operations) and (len(x.shape) == 1):
			E_new += np.sum(self.beta_flat/2. * ((x ** 2 + y ** 2) ** 2))
			E_new += np.sum(self.e_disorder_flat * (x ** 2 + y ** 2))
			E_new += np.sum(-self.J * (y * (
												   y[self.nn_idx_1] +
												   y[self.nn_idx_2] +
												   y[self.nn_idy_1] +
												   y[self.nn_idy_2] +
							   self.anisotropy * (y[self.nn_idz_1] +
												  y[self.nn_idz_2]
																      )
												   ) +
									   x * (
													x[self.nn_idx_1] +
													x[self.nn_idx_2] +
													x[self.nn_idy_1] +
													x[self.nn_idy_2] +
								self.anisotropy * (x[self.nn_idz_1] +
												   x[self.nn_idz_2]
									   )
									   )
									)
							)
			E_new += np.sum(self.h_dis_x_flat * x + self.h_dis_y_flat * y)
		else:
			for j in self.wells_indices:
				E_new += (self.beta_volume[j]/2. * ((x[j]**2 + y[j]**2)**2) +
						  self.e_disorder[j] * (x[j]**2 + y[j]**2) +
						  self.h_dis_x_volume[j] * x[j] +
						  self.h_dis_y_volume[j] * y[j]
						  )
				for idx, k in enumerate(self.nearest_neighbours(j)):
					# Introduce anisotropy of J for the 3rd axis
					if idx > 3:
						E_new += (-self.anisotropy * self.J * (x[j] * x[k] + y[j] * y[k]))
					else:
						E_new += (-self.J * (x[j] * x[k] + y[j] * y[k]))
		return E_new

	def calc_energy_XY_global(self, PSI):
		# PSI[time, 2*N_wells]
		if (self.use_matrix_operations_for_energy):
			return np.sum(self.beta_flat / 2. * ((PSI[:,:self.N_wells] ** 2 + PSI[:,self.N_wells:] ** 2) ** 2) +
			self.e_disorder_flat * (PSI[:,:self.N_wells] ** 2 + PSI[:,self.N_wells:] ** 2)
			-self.J * (PSI[:,self.N_wells:] * (
					PSI[:,self.N_wells:][:,self.nn_idx_1] +
					PSI[:,self.N_wells:][:,self.nn_idx_2] +
					PSI[:,self.N_wells:][:,self.nn_idy_1] +
					PSI[:,self.N_wells:][:,self.nn_idy_2] +
					self.anisotropy * (PSI[:,self.N_wells:][:,self.nn_idz_1] +
									   PSI[:,self.N_wells:][:,self.nn_idz_2]
									   )
			) +
									   PSI[:, :self.N_wells] * (
											   PSI[:,:self.N_wells][:,self.nn_idx_1] +
											   PSI[:,:self.N_wells][:,self.nn_idx_2] +
											   PSI[:,:self.N_wells][:,self.nn_idy_1] +
											   PSI[:,:self.N_wells][:,self.nn_idy_2] +
											   self.anisotropy * (PSI[:,:self.N_wells][:,self.nn_idz_1] +
																  PSI[:,:self.N_wells][:,self.nn_idz_2]
																  )
									   )
									   ) +
				self.h_dis_x_flat * PSI[:,:self.N_wells] + self.h_dis_y_flat * PSI[:,self.N_wells:], axis=1)
		else:
			return np.zeros(self.n_steps, dtype=self.FloatPrecision)

	def calc_nop_XY_global(self, PSI):
		# PSI[time, 2*N_wells]
		if (self.use_matrix_operations_for_energy):
			return np.sum((PSI[:,:self.N_wells] ** 2 + PSI[:,self.N_wells:] ** 2), axis=1)
		else:
			return np.zeros(self.n_steps, dtype=self.FloatPrecision)


	def calc_angular_momentum_XY(self, x, y):
		L = 0
		for j in self.wells_indices:
			for k in self.nearest_neighbours(j):
				L += - 2 * self.J * x[j] * y[k]
		return L

	def calc_full_energy_XY(self, x, y):
		E_kin = 0
		E_pot = 0
		E_noise = 0
		for j in self.wells_indices:
			for idx, k in enumerate(self.nearest_neighbours(j)):
				# Introduce anisotropy of J for the 3rd axis
				if idx > 3:
					E_kin += (-self.anisotropy * self.J * (x[j] * x[k] + y[j] * y[k]))
				else:
					E_kin += (-self.J * (x[j] * x[k] + y[j] * y[k]))

			E_pot += self.beta_volume[j]/2. * ((x[j]**2 + y[j]**2)**2)
			E_noise += self.e_disorder[j] * (x[j]**2 + y[j]**2)

		return E_kin, E_pot, E_noise

	def calc_number_of_particles_XY(self, x, y):
		return (np.sum((x ** 2) + (y ** 2)) - self.N_part)

	def make_exception(self, code):
		self.error_code += code
		self.consistency_checksum = 1

	def E_const_perturbation_XY(self, x0, y0, delta, degrees_of_freedom=30):

		bnds = np.hstack((x0.flatten(), y0.flatten()))

		dof_idx = np.arange(bnds.shape[0])
		# if bnds.shape[0] > degrees_of_freedom:
		# 	np.random.shuffle(dof_idx)
		# 	dof_idx = np.sort(dof_idx[:degrees_of_freedom])

		x_err = 1. * delta #0.01 * x0
		y_err = 1. * delta #0.01 * y0
		np.random.seed()
		x_next = x0 + x_err * np.random.randn(self.N_tuple[0], self.N_tuple[1], self.N_tuple[2])
		y_next = y0 + y_err * np.random.randn(self.N_tuple[0], self.N_tuple[1], self.N_tuple[2])
		zero_app = np.hstack((x_next.flatten(), y_next.flatten()))

		def get_x_full(xsh, zero_app, dof_idx):
			x = zero_app.copy()
			x[dof_idx] = xsh.copy()
			return x

		if self.use_matrix_operations:
			fun = lambda x: (((self.calc_energy_XY(get_x_full(x, zero_app, dof_idx)[:self.N_wells],
											   get_x_full(x, zero_app, dof_idx)[self.N_wells:],
											   self.E_calibr))/self.E_calibr) ** 2 +
						 (self.calc_number_of_particles_XY(get_x_full(x, zero_app, dof_idx)[:self.N_wells],
														   get_x_full(x, zero_app, dof_idx)[self.N_wells:]
														   )/self.N_part) ** 2)
		else:
			fun = lambda x: (((self.calc_energy_XY(get_x_full(x, zero_app, dof_idx)[:self.N_wells].reshape(self.N_tuple),
											   get_x_full(x, zero_app, dof_idx)[self.N_wells:].reshape(self.N_tuple),
											   self.E_calibr)) / self.E_calibr) ** 2 +
						 (self.calc_number_of_particles_XY(
							 get_x_full(x, zero_app, dof_idx)[:self.N_wells].reshape(self.N_tuple),
							 get_x_full(x, zero_app, dof_idx)[self.N_wells:].reshape(self.N_tuple)
							 ) / self.N_part) ** 2)

		zero_app_cut = zero_app.copy()[dof_idx]
		bnds_cut = bnds.copy()[dof_idx]
		opt = minimize(fun, zero_app_cut,
					   bounds=[(xi - 1.0 * delta, xi + 1.0 * delta) for xi in bnds_cut],
					   options={'ftol':self.FTOL})

		col = 0
		if self.use_matrix_operations:

			while (col < 10) and ((opt.success == False) or
									  (np.abs(self.calc_energy_XY(get_x_full(opt.x, zero_app, dof_idx)[:self.N_wells],
																  get_x_full(opt.x, zero_app, dof_idx)[self.N_wells:],
																  self.E_calibr))/ self.E_calibr > self.E_eps) or
									  (np.abs(self.calc_number_of_particles_XY(get_x_full(opt.x, zero_app, dof_idx)[:self.N_wells],
																			   get_x_full(opt.x, zero_app, dof_idx)[self.N_wells:])/self.N_part) > 0.01)):
				np.random.seed()
				zero_app = zero_app + delta * np.random.randn(zero_app.shape[0])
				zero_app_cut = zero_app[dof_idx]
				opt = minimize(fun, zero_app_cut,
							   bounds=[(xi - 10.0 * delta, xi + 10.0 * delta) for xi in bnds_cut],
							   options={'ftol':self.FTOL})
				col += 1
		else:
			while (col < 10) and ((opt.success == False) or
									  (np.abs(self.calc_energy_XY(get_x_full(opt.x, zero_app, dof_idx)[:self.N_wells].reshape(self.N_tuple),
																  get_x_full(opt.x, zero_app, dof_idx)[self.N_wells:].reshape(self.N_tuple),
																  self.E_calibr))/ self.E_calibr > self.E_eps) or
									  (np.abs(self.calc_number_of_particles_XY(get_x_full(opt.x, zero_app, dof_idx)[:self.N_wells].reshape(self.N_tuple),
																			   get_x_full(opt.x, zero_app, dof_idx)[self.N_wells:].reshape(self.N_tuple))/self.N_part) > 0.01)):
				np.random.seed()
				zero_app = zero_app + delta * np.random.randn(zero_app.shape[0])
				zero_app_cut = zero_app[dof_idx]
				opt = minimize(fun, zero_app_cut,
							   bounds=[(xi - 10.0 * delta, xi + 10.0 * delta) for xi in bnds_cut],
							   options={'ftol':self.FTOL})
				col += 1

		if self.use_matrix_operations:

			x1 = get_x_full(opt.x, zero_app, dof_idx)[:self.N_wells].reshape(self.N_tuple)
			y1 = get_x_full(opt.x, zero_app, dof_idx)[self.N_wells:].reshape(self.N_tuple)
			if np.abs(self.calc_energy_XY(x1.flatten(), y1.flatten(), self.E_calibr) / self.E_calibr) > self.E_eps:
				self.make_exception('Could not find a new initial on-shell state\n')
		else:
			x1 = get_x_full(opt.x, zero_app, dof_idx)[:self.N_wells].reshape(self.N_tuple)
			y1 = get_x_full(opt.x, zero_app, dof_idx)[self.N_wells:].reshape(self.N_tuple)
			if np.abs(self.calc_energy_XY(x1, y1, self.E_calibr) / self.E_calibr) > self.E_eps:
				self.make_exception('Could not find a new initial on-shell state\n')

		if np.abs((self.calc_number_of_particles_XY(x1,y1)) / self.N_part) > 0.01:
			self.make_exception('Could not find a new initial state with the same number of particles\n')
		# if np.abs(self.calc_traj_shift_XY(x1,y1, x0, y0) / delta) < 0.3:
		# 	self.make_exception('Could not find a trajectory on such a distance\n')
		# 	return x1, y1, 1
		if col == 10:
			self.make_exception('Exceeded number of attempts in E_const_perturbation\n')
			return x1, y1, 1
		else:
			return x1, y1, 0


	def E_const_perturbation_XY_not_optimal(self, x0, y0, delta):
		bnds = np.hstack((x0.flatten(), y0.flatten()))
		x_err = 1. * delta #0.01 * x0
		y_err = 1. * delta #0.01 * y0
		np.random.seed()
		x_next = x0 + x_err * np.random.randn(self.N_tuple[0], self.N_tuple[1], self.N_tuple[2])
		y_next = y0 + y_err * np.random.randn(self.N_tuple[0], self.N_tuple[1], self.N_tuple[2])
		zero_app = np.hstack((x_next.flatten(), y_next.flatten()))
		if self.use_matrix_operations:

			fun = lambda x: (((self.calc_energy_XY(x[:self.N_wells],
												   x[self.N_wells:],
												   self.E_calibr))/self.E_calibr) ** 2 +
							 (self.calc_number_of_particles_XY(x[:self.N_wells],
															   x[self.N_wells:]
															   )/self.N_part) ** 2)
		else:
			fun = lambda x: (((self.calc_energy_XY(x[:self.N_wells].reshape(self.N_tuple),
												   x[self.N_wells:].reshape(self.N_tuple),
												   self.E_calibr)) / self.E_calibr) ** 2 +
							 (self.calc_number_of_particles_XY(x[:self.N_wells].reshape(self.N_tuple),
															   x[self.N_wells:].reshape(self.N_tuple)
															   ) / self.N_part) ** 2)

		opt = minimize(fun, zero_app,
					   bounds=[(xi - 1.0 * delta, xi + 1.0 * delta) for xi in bnds],
					   options={'ftol':self.FTOL})

		col = 0
		if self.use_matrix_operations:

			while (col < 10) and ((opt.success == False) or
									  (np.abs(self.calc_energy_XY(opt.x[:self.N_wells],
																  opt.x[self.N_wells:],
																  self.E_calibr))/ self.E_calibr > self.E_eps) or
									  (np.abs(self.calc_number_of_particles_XY(opt.x[:self.N_wells],
																			   opt.x[self.N_wells:])/self.N_part) > 0.01)):
				np.random.seed()
				x0new = zero_app + delta * np.random.randn(zero_app.shape[0])
				opt = minimize(fun, x0new,
							   bounds=[(xi - 10.0 * delta, xi + 10.0 * delta) for xi in bnds],
							   options={'ftol':self.FTOL})
				col += 1
		else:
			while (col < 10) and ((opt.success == False) or
								  (np.abs(self.calc_energy_XY(opt.x[:self.N_wells].reshape(self.N_tuple),
															  opt.x[self.N_wells:].reshape(self.N_tuple),
															  self.E_calibr)) / self.E_calibr > self.E_eps) or
								  (np.abs(self.calc_number_of_particles_XY(opt.x[:self.N_wells].reshape(self.N_tuple),
																		   opt.x[self.N_wells:].reshape(
																			   self.N_tuple)) / self.N_part) > 0.01)):
				np.random.seed()
				x0new = zero_app + delta * np.random.randn(zero_app.shape[0])
				opt = minimize(fun, x0new,
							   bounds=[(xi - 10.0 * delta, xi + 10.0 * delta) for xi in bnds],
							   options={'ftol': self.FTOL})
				col += 1

		if self.use_matrix_operations:

			x1 = opt.x[:self.N_wells].reshape(self.N_tuple)
			y1 = opt.x[self.N_wells:].reshape(self.N_tuple)

			if np.abs(self.calc_energy_XY(x1.flatten(), y1.flatten(), self.E_calibr) / self.E_calibr) > self.E_eps:
				self.make_exception('Could not find a new initial on-shell state\n')
		else:
			x1 = opt.x[:self.N_wells].reshape(self.N_tuple)
			y1 = opt.x[self.N_wells:].reshape(self.N_tuple)

			if np.abs(self.calc_energy_XY(x1, y1, self.E_calibr) / self.E_calibr) > self.E_eps:
				self.make_exception('Could not find a new initial on-shell state\n')

		if np.abs((self.calc_number_of_particles_XY(x1,y1)) / self.N_part) > 0.01:
			self.make_exception('Could not find a new initial state with the same number of particles\n')
		# if np.abs(self.calc_traj_shift_XY(x1,y1, x0, y0) / delta) < 0.3:
		# 	self.make_exception('Could not find a trajectory on such a distance\n')
		# 	return x1, y1, 1
		if col == 10:
			self.make_exception('Exceeded number of attempts in E_const_perturbation\n')
			return x1, y1, 1
		else:
			return x1, y1, 0

	def calc_numerical_temperature_serial(self, x, y, N_samples=1000, pert_len=0.1):#pert_len=0.014):

		Es = np.zeros(N_samples)
		Ns = np.zeros(N_samples)
		Es_Amp = np.zeros(N_samples)
		Ns_Amp = np.zeros(N_samples)
		Es_Ph = np.zeros(N_samples)
		Ns_Ph = np.zeros(N_samples)
		if self.use_matrix_operations:
			E0 = self.calc_energy_XY(x.flatten(), y.flatten(), 0)
		else:
			E0 = self.calc_energy_XY(x, y, 0)
		total_part = np.sqrt(np.sum(x ** 2) + np.sum(y ** 2))
		pert_len = pert_len * total_part
		nx = x / total_part
		ny = y / total_part

		parti = np.sqrt(x ** 2 + y ** 2)
		nxi = x / parti
		nyi = y /parti

		for i in range(N_samples):
			dx = np.random.randn(x.shape[0], x.shape[1], x.shape[2])
			dy = np.random.randn(y.shape[0], y.shape[1], y.shape[2])

			diff = np.sum(dx * nx) + np.sum(dy * ny)
			dx -= diff * nx
			dy -= diff * ny

			dnorm = np.sqrt(np.sum(dx ** 2) + np.sum(dy ** 2))
			dx = dx / dnorm * pert_len
			dy = dy / dnorm * pert_len

			dx_Amp = nxi * (dx * nxi + dy * nyi)
			dy_Amp = nyi * (dx * nxi + dy * nyi)
			dx_Ph = dx - nxi * (dx * nxi + dy * nyi)
			dy_Ph = dy - nyi * (dx * nxi + dy * nyi)

			Ns[i] = self.calc_number_of_particles_XY(x + dx, y + dy)
			Ns_Amp[i] = self.calc_number_of_particles_XY(x + dx_Amp, y + dy_Amp)
			Ns_Ph[i] = self.calc_number_of_particles_XY(x + dx_Ph, y + dy_Ph)
			if self.use_matrix_operations:
				Es[i] = self.calc_energy_XY((x + dx).flatten(), (y + dy).flatten(), 0)
				Es_Amp[i] = self.calc_energy_XY((x + dx_Amp).flatten(), (y + dy_Amp).flatten(), 0)
				Es_Ph[i] = self.calc_energy_XY((x + dx_Ph).flatten(), (y + dy_Ph).flatten(), 0)
			else:
				Es[i] = self.calc_energy_XY(x + dx, y + dy, 0)
				Es_Amp[i] = self.calc_energy_XY(x + dx_Amp, y + dy_Amp, 0)
				Es_Ph[i] = self.calc_energy_XY(x + dx_Ph, y + dy_Ph, 0)

		return 0.5 * (np.std(Es - E0) ** 2) / np.mean(Es - E0), 0.5 * (np.std(Es_Amp - E0) ** 2) / np.mean(Es_Amp - E0), 0.5 * (np.std(Es_Ph - E0) ** 2) / np.mean(Es_Ph - E0)#Es, Ns

	def calc_temperature_old(self):
		T = np.zeros(self.X.shape[-1])
		T1 = np.zeros(self.X.shape[-1])
		T2 = np.zeros(self.X.shape[-1])
		lapl = np.zeros(self.X.shape[-1])
		len_grad = np.zeros(self.X.shape[-1])
		len_grad_len_grad_H_sqr = np.zeros(self.X.shape[-1])

		for it in range(T.shape[0]):
			x = self.X[:,:,:,it].copy()
			y = self.Y[:,:,:,it].copy()
			grad_H = np.zeros(self.N_wells * 2)
			len_grad_H_sqr = 0.
			grad_len_grad_H_sqr = np.zeros(self.N_wells * 2)
			# was wrong???
			# laplacian_H = 4. * self.beta * self.N_part
			laplacian_H = 8. * self.beta_amplitude * self.N_part
			for ind, i in enumerate(self.wells_indices):
				# x_derivative
				# was wrong???
				# grad_H[ind] = self.beta * (x[i] ** 2 + y[i] ** 2) * x[i]
				grad_H[ind] = 2. * self.beta_volume[i] * (x[i] ** 2 + y[i] ** 2) * x[i]
				grad_len_grad_H_sqr[ind] = 6. * (self.beta_volume[i] ** 2) * np.power(x[i] ** 2 + y[i] ** 2, 2) * x[i]

				# y_derivative
				# was wrong???
				# grad_H[ind + self.N_wells] = self.beta * (x[i] ** 2 + y[i] ** 2) * y[i]
				grad_H[ind + self.N_wells] = 2. * self.beta_volume[i] * (x[i] ** 2 + y[i] ** 2) * y[i]
				grad_len_grad_H_sqr[ind + self.N_wells] = 6. * (self.beta_volume[i] ** 2) * np.power(x[i] ** 2 + y[i] ** 2, 2) * y[i]

				xnn_sum = 0
				ynn_sum = 0
				xnn_brack = 0
				ynn_brack = 0
				xnnn_sum = 0
				ynnn_sum = 0

				for j in self.nearest_neighbours(i):
					xnn_sum += x[j]
					ynn_sum += y[j]
					#                 xnn_brack += (x[j] ** 2 + y[j] ** 2) * x[j]
					#                 ynn_brack += (x[j] ** 2 + y[j] ** 2) * y[j]

					for k in self.nearest_neighbours(j):
						xnnn_sum += x[k]
						ynnn_sum += y[k]
						xnn_brack += (x[k] ** 2 + y[k] ** 2) * x[k]
						ynn_brack += (x[k] ** 2 + y[k] ** 2) * y[k]

				grad_len_grad_H_sqr[ind] += 2. * (self.J ** 2) * xnnn_sum
				grad_len_grad_H_sqr[ind + self.N_wells] += 2. * (self.J ** 2) * ynnn_sum

				grad_len_grad_H_sqr[ind] += - 4. * (self.J * self.beta_volume[i]) * x[i] * y[i] * ynn_sum
				grad_len_grad_H_sqr[ind + self.N_wells] += - 4. * (self.J * self.beta_volume[i]) * x[i] * y[i] * xnn_sum

				grad_len_grad_H_sqr[ind] += - 2. * (self.J * self.beta_volume[i]) * (3 * (x[i] ** 2) + (y[i] ** 2)) * xnn_sum
				grad_len_grad_H_sqr[ind + self.N_wells] += - 2. * (self.J * self.beta_volume[i]) * ((x[i] ** 2) + 3 *(y[i] ** 2)) * ynn_sum

				grad_len_grad_H_sqr[ind] += - 2. * (self.J * self.beta_volume[i]) * xnn_brack
				grad_len_grad_H_sqr[ind + self.N_wells] += - 2. * (self.J * self.beta_volume[i]) * ynn_brack

				grad_H[ind] -= self.J * xnn_sum
				grad_H[ind + self.N_wells] -= self.J * ynn_sum

			len_grad_H_sqr = np.dot(grad_H, grad_H)
			#         len_grad_H_sqr += (self.J ** 2) * ((xnn_sum ** 2) + (ynn_sum ** 2))
			#         len_grad_H_sqr += (self.beta ** 2) * (np.power(x[i] ** 2 + y[i] ** 2, 3))
			#         len_grad_H_sqr -= 2. * self.J * self.beta * (x[i] ** 2 + y[i] ** 2) * (x[i] * xnn_sum + y[i] * ynn_sum)

			T1[it] = laplacian_H / len_grad_H_sqr
			T2[it] = -np.dot(grad_H, grad_len_grad_H_sqr) / (len_grad_H_sqr ** 2)
			lapl[it] = laplacian_H
			len_grad[it] = len_grad_H_sqr
			len_grad_len_grad_H_sqr = np.dot(grad_len_grad_H_sqr, grad_len_grad_H_sqr)
		return T1 + T2, T1, T2, lapl, len_grad, len_grad_len_grad_H_sqr

	def calc_temperature(self):
		T = np.zeros(self.X.shape[-1])
		T1 = np.zeros(self.X.shape[-1])
		T2 = np.zeros(self.X.shape[-1])
		lapl = np.zeros(self.X.shape[-1])
		len_grad = np.zeros(self.X.shape[-1])
		len_grad_len_grad_H_sqr = np.zeros(self.X.shape[-1])

		for it in range(T.shape[0]):
			x = self.X[:,:,:,it].copy()
			y = self.Y[:,:,:,it].copy()
			grad_H = np.zeros(self.N_wells * 2)
			len_grad_H_sqr = 0.
			grad_len_grad_H_sqr = np.zeros(self.N_wells * 2)
			# was wrong???
			# laplacian_H = 4. * self.beta * self.N_part
			laplacian_H = 8. * self.beta_amplitude * self.N_part
			for ind, i in enumerate(self.wells_indices):
				# x_derivative
				# was wrong???
				# grad_H[ind] = self.beta * (x[i] ** 2 + y[i] ** 2) * x[i]
				grad_H[ind] = 2. * self.beta_volume[i] * (x[i] ** 2 + y[i] ** 2) * x[i]
				grad_len_grad_H_sqr[ind] = 6. * (self.beta_volume[i] ** 2) * np.power(x[i] ** 2 + y[i] ** 2, 2) * x[i]

				# y_derivative
				# was wrong???
				# grad_H[ind + self.N_wells] = self.beta * (x[i] ** 2 + y[i] ** 2) * y[i]
				grad_H[ind + self.N_wells] = 2. * self.beta_volume[i] * (x[i] ** 2 + y[i] ** 2) * y[i]
				grad_len_grad_H_sqr[ind + self.N_wells] = 6. * (self.beta_volume[i] ** 2) * np.power(x[i] ** 2 + y[i] ** 2, 2) * y[i]

				xnn_sum = 0
				ynn_sum = 0
				xnn_brack = 0
				ynn_brack = 0
				xnnn_sum = 0
				ynnn_sum = 0

				for j in self.nearest_neighbours(i):
					xnn_sum += x[j]
					ynn_sum += y[j]
					#                 xnn_brack += (x[j] ** 2 + y[j] ** 2) * x[j]
					#                 ynn_brack += (x[j] ** 2 + y[j] ** 2) * y[j]

					for k in self.nearest_neighbours(j):
						xnnn_sum += x[k]
						ynnn_sum += y[k]
						xnn_brack += (x[k] ** 2 + y[k] ** 2) * x[k]
						ynn_brack += (x[k] ** 2 + y[k] ** 2) * y[k]

				grad_len_grad_H_sqr[ind] += 2. * (self.J ** 2) * xnnn_sum
				grad_len_grad_H_sqr[ind + self.N_wells] += 2. * (self.J ** 2) * ynnn_sum

				grad_len_grad_H_sqr[ind] += - 4. * (self.J * self.beta_volume[i]) * x[i] * y[i] * ynn_sum
				grad_len_grad_H_sqr[ind + self.N_wells] += - 4. * (self.J * self.beta_volume[i]) * x[i] * y[i] * xnn_sum

				grad_len_grad_H_sqr[ind] += - 2. * (self.J * self.beta_volume[i]) * (3 * (x[i] ** 2) + (y[i] ** 2)) * xnn_sum
				grad_len_grad_H_sqr[ind + self.N_wells] += - 2. * (self.J * self.beta_volume[i]) * ((x[i] ** 2) + 3 *(y[i] ** 2)) * ynn_sum

				grad_len_grad_H_sqr[ind] += - 2. * (self.J * self.beta_volume[i]) * xnn_brack
				grad_len_grad_H_sqr[ind + self.N_wells] += - 2. * (self.J * self.beta_volume[i]) * ynn_brack

				grad_H[ind] -= self.J * xnn_sum
				grad_H[ind + self.N_wells] -= self.J * ynn_sum

			len_grad_H_sqr = np.dot(grad_H, grad_H)
			#         len_grad_H_sqr += (self.J ** 2) * ((xnn_sum ** 2) + (ynn_sum ** 2))
			#         len_grad_H_sqr += (self.beta ** 2) * (np.power(x[i] ** 2 + y[i] ** 2, 3))
			#         len_grad_H_sqr -= 2. * self.J * self.beta * (x[i] ** 2 + y[i] ** 2) * (x[i] * xnn_sum + y[i] * ynn_sum)

			T1[it] = laplacian_H / len_grad_H_sqr
			T2[it] = -np.dot(grad_H, grad_len_grad_H_sqr) / (len_grad_H_sqr ** 2)
			lapl[it] = laplacian_H
			len_grad[it] = len_grad_H_sqr
			len_grad_len_grad_H_sqr = np.dot(grad_len_grad_H_sqr, grad_len_grad_H_sqr)
		return T1 + T2, T1, T2, lapl, len_grad, len_grad_len_grad_H_sqr

	def calc_numerical_temperature(self, x, y, N_samples=1000, n_proc=40, pert_len=0.1):#pert_len=0.014):

		iters = int(N_samples / n_proc)
		N_samples = int(iters * n_proc)

		Es = mp.Array('d', range(N_samples))
		Es_Amp = mp.Array('d', range(N_samples))
		Es_Ph = mp.Array('d', range(N_samples))
		# TASKS = mp.Array('i', range(n_proc))

		# print 'Start'
		if self.use_matrix_operations:
			E0 = self.calc_energy_XY(x.flatten(), y.flatten(), 0)
		else:
			E0 = self.calc_energy_XY(x, y, 0)

		total_part = np.sqrt(np.sum(x ** 2) + np.sum(y ** 2))
		pert_len = pert_len * total_part
		nx = x / total_part
		ny = y / total_part

		parti = np.sqrt(x ** 2 + y ** 2)
		nxi = x / parti
		nyi = y /parti

		# pool = mp.Pool(processes=n_proc)

		TASKS = np.arange(n_proc, dtype=np.int32)
		# q = mp.Queue()
		ps = []
		for t in TASKS:
			p = mp.Process(target=one_realization,
						   args=(t, self, x.flatten(), y.flatten(), nx.flatten(), ny.flatten(), nxi.flatten(), nyi.flatten(), pert_len, iters, Es, Es_Amp, Es_Ph))
			p.start()
			ps.append(p)

		for p in ps:
			p.join()
		# for t in TASKS:
		# 	q.put(t)
		# p.join()

		Es = np.array(Es[:])
		Es_Amp = np.array(Es_Amp[:])
		Es_Ph = np.array(Es_Ph[:])
		# print Es

		return 0.5 * (np.std(Es - E0) ** 2) / np.mean(Es - E0), 0.5 * (np.std(Es_Amp - E0) ** 2) / np.mean(Es_Amp - E0), 0.5 * (np.std(Es_Ph - E0) ** 2) / np.mean(Es_Ph - E0)#Es, Ns

	def calc_numerical_temperature_slow(self, x, y, N_samples=1000, n_proc=40, pert_len=0.1):#pert_len=0.014):

		iters = int(N_samples / n_proc)
		N_samples = int(iters * n_proc)

		Es = mp.Array('d', range(N_samples))
		Es_Amp = mp.Array('d', range(N_samples))
		Es_Ph = mp.Array('d', range(N_samples))
		# TASKS = mp.Array('i', range(n_proc))

		# print 'Start'
		if self.use_matrix_operations:
			E0 = self.calc_energy_XY(x.flatten(), y.flatten(), 0)
		else:
			E0 = self.calc_energy_XY(x, y, 0)

		total_part = np.sqrt(np.sum(x ** 2) + np.sum(y ** 2))
		pert_len = pert_len * total_part
		nx = x / total_part
		ny = y / total_part

		parti = np.sqrt(x ** 2 + y ** 2)
		nxi = x / parti
		nyi = y /parti

		# pool = mp.Pool(processes=n_proc)

		TASKS = np.arange(n_proc, dtype=np.int32)
		# q = mp.Queue()
		ps = []
		for t in TASKS:
			p = mp.Process(target=one_realization,
						   args=(t, self, x, y, nx, ny, nxi, nyi, pert_len, iters, Es, Es_Amp, Es_Ph))
			p.start()
			ps.append(p)

		for p in ps:
			p.join()
		# for t in TASKS:
		# 	q.put(t)
		# p.join()

		Es = np.array(Es[:])
		Es_Amp = np.array(Es_Amp[:])
		Es_Ph = np.array(Es_Ph[:])
		# print Es

		return 0.5 * (np.std(Es - E0) ** 2) / np.mean(Es - E0), 0.5 * (np.std(Es_Amp - E0) ** 2) / np.mean(Es_Amp - E0), 0.5 * (np.std(Es_Ph - E0) ** 2) / np.mean(Es_Ph - E0)#Es, Ns

#
def one_realization(t, self, x, y, nx, ny, nxi, nyi, pert_len, iters, Es, Es_Amp, Es_Ph):
	# t = q.get()
	#numpy.random.seed(int((time()+some_parameter*1000))
	np.random.seed(int(t * 1000 + time()))

	for i in range(iters):
		dx = np.random.randn(x.shape[0])
		dy = np.random.randn(y.shape[0])

		diff = np.sum(dx * nx) + np.sum(dy * ny)
		dx -= diff * nx
		dy -= diff * ny

		dnorm = np.sqrt(np.sum(dx ** 2) + np.sum(dy ** 2))
		dx = dx / dnorm * pert_len
		dy = dy / dnorm * pert_len

		dx_Amp = nxi * (dx * nxi + dy * nyi)
		dy_Amp = nyi * (dx * nxi + dy * nyi)
		dx_Ph = dx - nxi * (dx * nxi + dy * nyi)
		dy_Ph = dy - nyi * (dx * nxi + dy * nyi)
		# print t
		# print iters
		# print i
		ind = t * iters + i
		# print ind

		Es[ind] = self.calc_energy_XY(x + dx, y + dy, 0)
		Es_Amp[ind] = self.calc_energy_XY(x + dx_Amp, y + dy_Amp, 0)
		Es_Ph[ind] = self.calc_energy_XY(x + dx_Ph, y + dy_Ph, 0)

#
def one_realization_slow(t, self, x, y, nx, ny, nxi, nyi, pert_len, iters, Es, Es_Amp, Es_Ph):
	# t = q.get()
	#numpy.random.seed(int((time()+some_parameter*1000))
	np.random.seed(int(t * 1000 + time()))

	for i in range(iters):
		dx = np.random.randn(x.shape[0], x.shape[1], x.shape[2])
		dy = np.random.randn(y.shape[0], y.shape[1], y.shape[2])

		diff = np.sum(dx * nx) + np.sum(dy * ny)
		dx -= diff * nx
		dy -= diff * ny

		dnorm = np.sqrt(np.sum(dx ** 2) + np.sum(dy ** 2))
		dx = dx / dnorm * pert_len
		dy = dy / dnorm * pert_len

		dx_Amp = nxi * (dx * nxi + dy * nyi)
		dy_Amp = nyi * (dx * nxi + dy * nyi)
		dx_Ph = dx - nxi * (dx * nxi + dy * nyi)
		dy_Ph = dy - nyi * (dx * nxi + dy * nyi)
		# print t
		# print iters
		# print i
		ind = t * iters + i
		# print ind

		Es[ind] = self.calc_energy_XY(x + dx, y + dy, 0)
		Es_Amp[ind] = self.calc_energy_XY(x + dx_Amp, y + dy_Amp, 0)
		Es_Ph[ind] = self.calc_energy_XY(x + dx_Ph, y + dy_Ph, 0)
