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
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Lasso

class TrajectoryAnalysator(object):
	def __init__(self, **kwargs):
		self.J = kwargs.get('J', 1.0)
		# self.beta = kwargs.get('beta', 0.01)
		self.beta = kwargs.get('beta', 0.01)
		self.N_wells = kwargs.get('N_wells', 10)
		self.e_disorder = kwargs.get('disorder', np.zeros(self.N_wells))
		self.N_part = kwargs.get('N_part_per_well', 100000)
		self.N_part *= self.N_wells
		self.step = kwargs.get('step', 5.7e-05)
		self.tau_char = kwargs.get('tau_char', 1.0 / np.sqrt(3 * self.beta * self.J * self.N_part/self.N_wells))
		self.time = kwargs.get('time', 1.4 * 50)
		self.time *= self.tau_char
		self.n_steps = kwargs.get('n_steps', int(self.time / self.step))
		self.t0 = kwargs.get('t0', 0)
		self.traj_seed = kwargs.get('traj_seed', 78)
		self.pert_seed = kwargs.get('pert_seed', 123)
		self.FloatPrecision = kwargs.get('FloatPrecision', np.float128)
		self.E_calibr = kwargs.get('E_calibr', 0)
		self.threshold_XY_to_polar = kwargs.get('threshold_XY_to_polar', 1.)
		self.energy = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.participation_rate = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.effective_nonlinearity = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.number_of_particles = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.T = np.linspace(0, self.time, self.n_steps)
		self.RHO = np.zeros((self.n_steps, self.N_wells), dtype=self.FloatPrecision)
		self.THETA = np.zeros((self.n_steps, self.N_wells), dtype=self.FloatPrecision)
		self.X = np.zeros((self.n_steps, self.N_wells), dtype=self.FloatPrecision)
		self.Y = np.zeros((self.n_steps, self.N_wells), dtype=self.FloatPrecision)
		self.consistency_checksum = 0
		self.error_code = ""

		self.RHO1 = np.zeros((self.n_steps, self.N_wells), dtype=self.FloatPrecision)
		self.THETA1 = np.zeros((self.n_steps, self.N_wells), dtype=self.FloatPrecision)
		self.X1 = np.zeros((self.n_steps, self.N_wells), dtype=self.FloatPrecision)
		self.Y1 = np.zeros((self.n_steps, self.N_wells), dtype=self.FloatPrecision)
		self.energy1 = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.participation_rate1 = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.effective_nonlinearity1 = np.zeros(self.n_steps, dtype=self.FloatPrecision)

		self.number_of_particles1 = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.distance = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.lambdas = []
		self.lambdas_no_regr = []

		self.polarisation = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.polarisation1 = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		self.perturb_hamiltonian = kwargs.get('perturb_hamiltonian', False)
		self.error_beta = kwargs.get('error_beta', 0)
		self.error_J = kwargs.get('error_J', 0)
		self.error_disorder = kwargs.get('error_disorder', 0)

		self.Lyapunov_EPS = kwargs.get('Lyapunov_EPS', 1e-3)
		self.instability_stops = []
		self.reset_steps_duration = kwargs.get('reset_steps_duration', 10000)
		self.distance_check = []

		self.PERT_EPS = 1e-8
		self.FTOL = kwargs.get('FTOL', 1e-14)
		self.E_eps = kwargs.get('E_eps', 1e-2)
		self.singular_eps = 1e-8

	def NN(self, i):
		if i < 0:
			return self.N_wells - 1
		elif i == self.N_wells:
			return 0
		else:
			return i

	def calc_constants_of_motion(self, RHO, THETA, X, Y):
		number_of_particles = np.sum(RHO ** 2, axis=1)
		energy = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		angular_momentum = np.zeros(self.n_steps, dtype=self.FloatPrecision)
		for j in xrange(self.N_wells):
			energy += (- self.J * (RHO[:,self.NN(j+1)] * RHO[:,j] * np.cos(THETA[:,self.NN(j+1)] - THETA[:,j]) +
			                       RHO[:,self.NN(j-1)] * RHO[:,j] * np.cos(THETA[:,self.NN(j-1)] - THETA[:,j])) +
			           self.beta/2. * np.abs(RHO[:,j]**4) +
			           self.e_disorder[j] * np.abs(RHO[:,j]**2))
			# angular_momentum += - 2 * self.J * (X[:,j] * (0*Y[:,self.NN(j-1)] + Y[:,self.NN(j+1)]) - Y[:,j] * (0*X[:,self.NN(j-1)] + X[:,self.NN(j+1)]))
			# angular_momentum += - 2 * self.J * (X[:,j] * (0*Y[:,self.NN(j-1)] + Y[:,self.NN(j+1)]) - 0*Y[:,j] * (0*X[:,self.NN(j-1)] + X[:,self.NN(j+1)]))
			# angular_momentum += - 2 * self.J * (X[:,j] * (1./3*X[:,j] ** 2 + Y[:,j] ** 2))
			angular_momentum += - 2 * self.J * (X[:,j] * (0*Y[:,self.NN(j-1)] + Y[:,self.NN(j+1)]) - 0*Y[:,j] * (0*X[:,self.NN(j-1)] + X[:,self.NN(j+1)]))

		return energy, number_of_particles, angular_momentum

	def set_constants_of_motion(self):
		self.energy, self.number_of_particles, self.angular_momentum = self.calc_constants_of_motion(self.RHO, self.THETA, self.X, self.Y)
		self.participation_rate = np.sum(self.RHO ** 4, axis=1) / (np.sum(self.RHO ** 2, axis=1) ** 2)
		self.effective_nonlinearity = self.beta * (self.participation_rate) / self.N_wells

		self.energy1, self.number_of_particles1, self.angular_momentum1 = self.calc_constants_of_motion(self.RHO1, self.THETA1, self.X1, self.Y1)
		self.participation_rate1 = np.sum(self.RHO1 ** 4, axis=1) / (np.sum(self.RHO1 ** 2, axis=1) ** 2)
		self.effective_nonlinearity1 = self.beta * self.participation_rate1 / self.N_wells


	def calc_traj_shift_XY(self, x0, y0, x1, y1):
		return np.sqrt(np.sum((x0 - x1) ** 2 + (y0 - y1) ** 2))

	def calc_energy_XY(self, x, y, E):
		E_new = -E
		for j in xrange(self.N_wells):
			E_new += (-self.J * (x[j] * x[self.NN(j-1)] + y[j] * y[self.NN(j-1)] +
			                     x[j] * x[self.NN(j+1)] + y[j] * y[self.NN(j+1)]) +
			         self.beta/2. * ((x[j]**2 + y[j]**2)**2) +
			         self.e_disorder[j] * (x[j]**2 + y[j]**2))
		return E_new

	def calc_angular_momentum_XY(self, x, y):
		L = 0
		for j in xrange(self.N_wells):
			L += - 2 * self.J * x[j] * (y[self.NN(j-1)] + y[self.NN(j+1)])
		return L

	def calc_full_energy_XY(self, x, y):
		E_kin = 0
		E_pot = 0
		E_noise = 0
		for j in xrange(self.N_wells):
			E_kin += (-self.J * (x[j] * x[self.NN(j-1)] + y[j] * y[self.NN(j-1)] +
			                     x[j] * x[self.NN(j+1)] + y[j] * y[self.NN(j+1)]))
			E_pot += self.beta/2. * ((x[j]**2 + y[j]**2)**2)
			E_noise += self.e_disorder[j] * (x[j]**2 + y[j]**2)

		return E_kin, E_pot, E_noise

	def calc_number_of_particles_XY(self, x, y):
		return (np.sum((x ** 2) + (y ** 2)) - self.N_part)

	def make_exception(self, code):
		self.error_code += code
		self.consistency_checksum = 1

	def calc_traj_shift_matrix_cartesian_XY(self, X, Y, X1, Y1):
		return np.sqrt(np.sum((X - X1) ** 2 + (Y - Y1) ** 2, axis=1))

	def calculate_polarisation(self):
		self.polarisation = np.sum(self.X, axis=1)
		idx = np.arange(self.n_steps)[::-1]
		self.polarisation1 = np.sum(self.X1[idx,:], axis=1)

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
			print('Bad Lyapunov lambda')
			self.lambdas.append(0.)
		self.lambdas_no_regr.append((np.log(self.distance[to] + 1e-15) - np.log(self.distance[fr] + 1e-15)) / (self.T[to] - self.T[fr]))

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
