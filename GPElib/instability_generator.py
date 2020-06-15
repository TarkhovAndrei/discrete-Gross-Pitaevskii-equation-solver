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

class InstabilityGenerator(TwoTrajsGenerator):
	def __init__(self, **kwargs):
		TwoTrajsGenerator.__init__(self, **kwargs)
		self.polarisation = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.polarisation1 = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.perturb_hamiltonian = kwargs.get('perturb_hamiltonian', False)
		self.error_beta = kwargs.get('error_beta', 0)
		self.error_J = kwargs.get('error_J', 0)
		self.error_disorder = kwargs.get('error_disorder', 0)

	def run_dynamics(self):
		TwoTrajsGenerator.run_dynamics(self)
		self.X1 = self.X.copy()
		self.Y1 = self.Y.copy()
		self.RHO1 = self.RHO.copy()
		self.THETA1 = self.THETA.copy()

		if self.perturb_hamiltonian:
			x1, y1 = self.X[:,:,:,self.n_steps-1], self.Y[:,:,:,self.n_steps-1]
		else:
			x1, y1 = self.constant_perturbation_XY(self.X[:,:,:,self.n_steps-1].copy(),self.Y[:,:,:,self.n_steps-1].copy())

		self.set_init_XY(x1, y1, self.X[:,:,:,0], self.Y[:,:,:,0])

		self.reverse_hamiltonian(self.error_J, self.error_beta, self.error_disorder)
		TwoTrajsGenerator.run_dynamics(self)
		self.reverse_hamiltonian(self.error_J, self.error_beta, self.error_disorder)

		idx = np.arange(self.n_steps)[::-1]
		# idx = np.arange(self.X1.shape[-1])[::-1]
		self.distance = self.calc_traj_shift_matrix_cartesian_XY(self.X[:,:,:,idx], self.Y[:,:,:,idx],
																 self.X1[:,:,:,idx[::-1]], self.Y1[:,:,:,idx[::-1]])

		# self.set_constants_of_motion()
		# self.calculate_polarisation()
		# if (np.abs(np.max(np.abs(self.energy - self.E_calibr)) / self.E_calibr) > 0.01) or (np.abs(np.max(np.abs(self.energy1 - self.E_calibr)) / self.E_calibr) > 0.01):
		# 	self.make_exception('Energy is not conserved during the dynamics\n')
		# if (np.abs(np.max(np.abs(self.number_of_particles - self.N_part)) / self.N_part) > 0.01) or (np.abs(np.max(np.abs(self.number_of_particles1 - self.N_part)) / self.N_part) > 0.01):
		# 	self.make_exception('Number of particles is not conserved during the dynamics\n')
		# self.calculate_lambdas()

	def calculate_polarisation(self):
		self.polarisation = np.sum(self.X, axis=(0,1,2))
		idx = np.arange(self.n_steps)[::-1]
		self.polarisation1 = np.sum(self.X1[:,:,:,idx], axis=(0,1,2))

	def calculate_lambdas(self):
		self.lambdas = []
		self.lambdas_no_regr = []
		clf = LinearRegression()
		fr = self.n_steps / 2
		to = self.n_steps - 1
		try:
			clf.fit(self.T[fr:to].reshape(to-fr,1), np.log(self.distance[fr:to] + 1e-15).reshape(to-fr,1))
			self.lambdas.append(clf.coef_[0][0])
		except:
			self.make_exception('Bad Lyapunov lambda\n')
			self.lambdas.append(0.)
		self.lambdas_no_regr.append((np.log(self.distance[to] + 1e-15) - np.log(self.distance[fr] + 1e-15)) / (self.T[to] - self.T[fr]))