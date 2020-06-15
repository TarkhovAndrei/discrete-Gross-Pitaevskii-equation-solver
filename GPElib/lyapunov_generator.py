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
import numpy as np
from sklearn.linear_model import LinearRegression
from .two_trajs_generator import TwoTrajsGenerator

class LyapunovGenerator(TwoTrajsGenerator):
	def __init__(self, **kwargs):
		TwoTrajsGenerator.__init__(self, **kwargs)
		self.Lyapunov_EPS = kwargs.get('Lyapunov_EPS', 1e-3)
		self.instability_stops = []
		self.reset_steps_duration = kwargs.get('reset_steps_duration', 10000)
		self.distance_check = []
		self.icurr = 0
		self.inext = 1

	def upload_backup_from_file(self, backup):
		lyap.lambdas = backup['lyap.lambdas']
		lyap.lambdas_no_regr = backup['lyap.lambdas_no_regr']


	def reset_perturbation_XY(self, x0, y0, x1, y1):
		dst = self.calc_traj_shift_XY(x0, y0, x1, y1)
		if dst < 1e-5:
			x1 = x0 + (x1 - x0) * 1e-3/dst
			y1 = y0 + (y1 - y0) * 1e-3/dst
		dst = self.calc_traj_shift_XY(x0, y0, x1, y1)
		x1 = x0 + (x1 - x0) * self.PERT_EPS /dst
		y1 = y0 + (y1 - y0) * self.PERT_EPS /dst
		return x1, y1

	def run_dynamics(self, no_pert=False):
		if no_pert == False:
			x1, y1 = self.constant_perturbation_XY(self.X[:,:,:,0], self.Y[:,:,:,0])
			self.set_init_XY(self.X[:,:,:,0], self.Y[:,:,:,0], x1, y1)
		self.instability_stops = [0]
		self.distance_check = [self.calc_traj_shift_XY(self.X[:,:,:,0], self.Y[:,:,:,0], self.X1[:,:,:,0], self.Y1[:,:,:,0])]
		self.distance[0] = self.calc_traj_shift_XY(self.X[:, :, :, 0], self.Y[:, :, :, 0], self.X1[:, :, :, 0],
												   self.Y1[:, :, :, 0])

		self.set_constants_of_motion_local(0, 0)

		icurr = 0
		inext = 1
		self.icurr = 0
		self.inext = 1

		for i in xrange(1, self.n_steps):

			if (np.any(self.RHO[:,:,:,icurr] ** 2 < self.threshold_XY_to_polar) or (np.any(self.RHO1[:,:,:,icurr] ** 2 < self.threshold_XY_to_polar))):
				psi = self.rk4_step_exp_XY(np.hstack((self.X[:,:,:,icurr].flatten(), self.Y[:,:,:,icurr].flatten())))
				psi1 = self.rk4_step_exp_XY(np.hstack((self.X1[:,:,:,icurr].flatten(), self.Y1[:,:,:,icurr].flatten())))
				self.X[:,:,:,inext] = psi[:self.N_wells].reshape(self.N_tuple)
				self.Y[:,:,:,inext] = psi[self.N_wells:].reshape(self.N_tuple)
				self.RHO[:,:,:,inext], self.THETA[:,:,:,inext] = self.from_XY_to_polar(self.X[:,:,:,inext], self.Y[:,:,:,inext])
				self.X1[:,:,:,inext] = psi1[:self.N_wells].reshape(self.N_tuple)
				self.Y1[:,:,:,inext] = psi1[self.N_wells:].reshape(self.N_tuple)
				self.RHO1[:,:,:,inext], self.THETA1[:,:,:,inext] = self.from_XY_to_polar(self.X1[:,:,:,inext], self.Y1[:,:,:,inext])
				# self.X1[:,:,:,i], self.Y1[:,:,:,i] = self.from_polar_to_XY(self.RHO1[:,:,:,i], self.THETA1[:,:,:,i])
			else:
				psi = self.rk4_step_exp(np.hstack((self.RHO[:,:,:,icurr].flatten(), self.THETA[:,:,:,icurr].flatten())))
				psi1 = self.rk4_step_exp(np.hstack((self.RHO1[:,:,:,icurr].flatten(), self.THETA1[:,:,:,icurr].flatten())))
				self.RHO[:,:,:,inext] = psi[:self.N_wells].reshape(self.N_tuple)
				self.THETA[:,:,:,inext] = psi[self.N_wells:].reshape(self.N_tuple)
				self.X[:,:,:,inext], self.Y[:,:,:,inext] = self.from_polar_to_XY(self.RHO[:,:,:,inext], self.THETA[:,:,:,inext])
				self.RHO1[:,:,:,inext] = psi1[:self.N_wells].reshape(self.N_tuple)
				self.THETA1[:,:,:,inext] = psi1[self.N_wells:].reshape(self.N_tuple)
				self.X1[:,:,:,inext], self.Y1[:,:,:,inext] = self.from_polar_to_XY(self.RHO1[:,:,:,inext], self.THETA1[:,:,:,inext])
			dist = self.calc_traj_shift_XY(self.X[:,:,:,inext], self.Y[:,:,:,inext], self.X1[:,:,:,inext], self.Y1[:,:,:,inext])
			self.distance[i] = dist
			self.distance_check.append(dist)
			# if (dist > self.Lyapunov_EPS) or (i - self.instability_stops[-1] > self.reset_steps_duration):
			if (i - self.instability_stops[-1] > self.reset_steps_duration):
				self.X1[:,:,:,inext], self.Y1[:,:,:,inext] = self.reset_perturbation_XY(self.X[:,:,:,inext], self.Y[:,:,:,inext], self.X1[:,:,:,inext], self.Y1[:,:,:,inext])
				self.RHO1[:,:,:,inext], self.THETA1[:,:,:,inext] = self.from_XY_to_polar(self.X1[:,:,:,inext], self.Y1[:,:,:,inext])
				self.instability_stops.append(i)
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

		if self.instability_stops[-1] != self.n_steps:
			self.instability_stops.append(self.n_steps)
		# self.distance = self.calc_traj_shift_matrix_cartesian_XY(self.X, self.Y, self.X1, self.Y1)
		if (np.abs(np.max(np.abs(self.energy - self.E_calibr)) / self.E_calibr) > 0.01) or (np.abs(np.max(np.abs(self.energy1 - self.E_calibr)) / self.E_calibr) > 0.01):
			self.make_exception('Energy is not conserved during the dynamics\n')
		if (np.abs(np.max(np.abs(self.number_of_particles - self.N_part)) / self.N_part) > 0.01) or (np.abs(np.max(np.abs(self.number_of_particles1 - self.N_part)) / self.N_part) > 0.01):
			self.make_exception('Number of particles is not conserved during the dynamics\n')
		if (np.max(self.distance_check) > 1.):
			self.make_exception('Discontinuity during the calculations (distance b/w trajectories > 1)\n')
		self.calculate_lambdas()

	def calculate_lambdas(self):
		self.lambdas = []
		self.lambdas_no_regr = []
		clf = LinearRegression()
		for ist in xrange(np.array(self.instability_stops).shape[0]-1):
			fr = self.instability_stops[ist] + 1
			to = self.instability_stops[ist+1] - 1
			try:
				clf.fit(self.T[fr:to].reshape(to-fr,1), np.log(self.distance[fr:to] + 1e-15).reshape(to-fr,1))
				self.lambdas.append(clf.coef_[0][0])
			except:
				print('Bad Lyapunov lambda')
				self.lambdas.append(0.)
			self.lambdas_no_regr.append((np.log(self.distance[to] + 1e-15) - np.log(self.distance[fr]+ 1e-15)) / (self.T[to] - self.T[fr]))
