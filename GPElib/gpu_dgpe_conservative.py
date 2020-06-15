import torch
import numpy as np
# import torchdiffeq
# from torch.autograd import Variable


class DGPE_ODE(torch.nn.Module):

	def __init__(self, device, N_wells, J, anisotropy, gamma,
				 nn_idx_1, nn_idx_2, nn_idy_1, nn_idy_2, nn_idz_1, nn_idz_2,
				 h_dis_x_flat, h_dis_y_flat,
				 beta_disorder_array_flattened, beta_flat, e_disorder_flat
				 ):
		super(DGPE_ODE, self).__init__()

		self.J = torch.nn.Parameter(torch.tensor(np.zeros(N_wells) + J).to(device), requires_grad=False)
		self.anisotropy = torch.nn.Parameter(torch.tensor(np.zeros(N_wells) + anisotropy).to(device), requires_grad=False)

		self.gamma = torch.nn.Parameter(torch.tensor(np.zeros(N_wells) + gamma).to(device), requires_grad=False)

		# self.torch_N_wells = torch.tensor(self.N_wells, tf.int64)

		self.nn_idx_1 = torch.nn.Parameter(torch.tensor(nn_idx_1, dtype=torch.int64).to(device), requires_grad=False)
		self.nn_idx_2 = torch.nn.Parameter(torch.tensor(nn_idx_2, dtype=torch.int64).to(device), requires_grad=False)
		self.nn_idy_1 = torch.nn.Parameter(torch.tensor(nn_idy_1, dtype=torch.int64).to(device), requires_grad=False)
		self.nn_idy_2 = torch.nn.Parameter(torch.tensor(nn_idy_2, dtype=torch.int64).to(device), requires_grad=False)
		self.nn_idz_1 = torch.nn.Parameter(torch.tensor(nn_idz_1, dtype=torch.int64).to(device), requires_grad=False)
		self.nn_idz_2 = torch.nn.Parameter(torch.tensor(nn_idz_2, dtype=torch.int64).to(device), requires_grad=False)

		# self.torch_first_half = torch.nn.Parameter(torch.tensor(np.arange(N_wells), dtype=torch.int64).to(device))
		# self.torch_second_half = torch.nn.Parameter(torch.tensor(np.arange(N_wells, 2 * N_wells), dtype=torch.int64).to(device))

		self.N_wells = torch.nn.Parameter(torch.tensor(N_wells, dtype=torch.int64).to(device), requires_grad=False)

		# self.torch_psi = torch.tensor(self.psi, dtype=self.torch_FloatPrecision, device=self.torch_device)
		# self.torch_x = torch.tensor(self.psi[:self.N_wells], dtype=self.torch_FloatPrecision,
		# 							device=self.torch_device)
		# self.torch_y = torch.tensor(self.psi[self.N_wells:], dtype=self.torch_FloatPrecision,
		# 							device=self.torch_device)

		# self.dpsi = torch.nn.Parameter(torch.tensor(np.zeros(2 * N_wells)).to(device))

		# self.xL = torch.nn.Parameter(torch.tensor(np.zeros(N_wells)).to(device))
		# self.yL = torch.nn.Parameter(torch.tensor(np.zeros(N_wells)).to(device))

		# self.zero = torch.nn.Parameter(torch.tensor(np.zeros(N_wells)).to(device))

		self.h_dis_x_flat = torch.nn.Parameter(torch.tensor(h_dis_x_flat).to(device), requires_grad=False)
		self.h_dis_y_flat = torch.nn.Parameter(torch.tensor(h_dis_y_flat).to(device), requires_grad=False)

		self.beta_disorder_array_flattened = torch.nn.Parameter(torch.tensor(beta_disorder_array_flattened).to(device), requires_grad=False)
		self.beta = torch.nn.Parameter(torch.tensor(beta_flat).to(device), requires_grad=False)
		self.e_disorder = torch.nn.Parameter(torch.tensor(e_disorder_flat).to(device), requires_grad=False)

	def forward(self, t, y):

		return(torch.cat([
			self.e_disorder * y[self.N_wells:] - (self.J * (
					torch.gather(y[self.N_wells:], 0, self.nn_idx_1) +
					torch.gather(y[self.N_wells:], 0, self.nn_idx_2) +
					torch.gather(y[self.N_wells:], 0, self.nn_idy_1) +
					torch.gather(y[self.N_wells:], 0, self.nn_idy_2) +
					self.anisotropy * (torch.gather(y[self.N_wells:], 0, self.nn_idz_1) +
									   torch.gather(y[self.N_wells:], 0, self.nn_idz_2)
									   )
			)) + self.h_dis_y_flat + self.beta *
			(torch.pow(y[self.N_wells:], 2) + torch.pow(y[:self.N_wells], 2)) * y[self.N_wells:],

			-self.e_disorder * y[:self.N_wells] +
			(self.J * (
					torch.gather(y[:self.N_wells], 0, self.nn_idx_1) +
					torch.gather(y[:self.N_wells], 0, self.nn_idx_2) +
					torch.gather(y[:self.N_wells], 0, self.nn_idy_1) +
					torch.gather(y[:self.N_wells], 0, self.nn_idy_2) +
					self.anisotropy * (torch.gather(y[:self.N_wells], 0, self.nn_idz_1) +
									   torch.gather(y[:self.N_wells], 0, self.nn_idz_2)
									   )
			)) - self.h_dis_x_flat -self.beta *
			(torch.pow(y[self.N_wells:], 2) + torch.pow(y[:self.N_wells], 2)) * y[:self.N_wells]], dim=0)
		)