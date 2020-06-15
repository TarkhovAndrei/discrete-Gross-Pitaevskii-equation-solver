import torch
import numpy as np
# import torchdiffeq
# from torch.autograd import Variable


class DGPE_ODE_RELAXATION(torch.nn.Module):

	def __init__(self, device, N_wells, J, anisotropy, gamma,
				 nn_idx_1, nn_idx_2, nn_idy_1, nn_idy_2, nn_idz_1, nn_idz_2,
				 h_dis_x_flat, h_dis_y_flat,
				 beta_disorder_array_flattened, beta_flat, e_disorder_flat,
				 E_desired, gamma_reduction, lam1, lam2, smooth_quench, temperature_dependent_rate
				 ):
		super(DGPE_ODE_RELAXATION, self).__init__()

		self.J = torch.nn.Parameter(torch.tensor(np.zeros(N_wells) + J).to(device), requires_grad=False)
		self.anisotropy = torch.nn.Parameter(torch.tensor(np.zeros(N_wells) + anisotropy).to(device), requires_grad=False)

		self.gamma = torch.nn.Parameter(torch.tensor(np.zeros(N_wells) + gamma).to(device), requires_grad=False)

		self.nn_idx_1 = torch.nn.Parameter(torch.tensor(nn_idx_1, dtype=torch.int64).to(device), requires_grad=False)
		self.nn_idx_2 = torch.nn.Parameter(torch.tensor(nn_idx_2, dtype=torch.int64).to(device), requires_grad=False)
		self.nn_idy_1 = torch.nn.Parameter(torch.tensor(nn_idy_1, dtype=torch.int64).to(device), requires_grad=False)
		self.nn_idy_2 = torch.nn.Parameter(torch.tensor(nn_idy_2, dtype=torch.int64).to(device), requires_grad=False)
		self.nn_idz_1 = torch.nn.Parameter(torch.tensor(nn_idz_1, dtype=torch.int64).to(device), requires_grad=False)
		self.nn_idz_2 = torch.nn.Parameter(torch.tensor(nn_idz_2, dtype=torch.int64).to(device), requires_grad=False)

		self.N_wells = torch.nn.Parameter(torch.tensor(N_wells, dtype=torch.int64).to(device), requires_grad=False)

		self.h_dis_x_flat = torch.nn.Parameter(torch.tensor(h_dis_x_flat).to(device), requires_grad=False)
		self.h_dis_y_flat = torch.nn.Parameter(torch.tensor(h_dis_y_flat).to(device), requires_grad=False)

		self.beta_disorder_array_flattened = torch.nn.Parameter(torch.tensor(beta_disorder_array_flattened).to(device), requires_grad=False)
		self.beta = torch.nn.Parameter(torch.tensor(beta_flat).to(device), requires_grad=False)
		self.e_disorder = torch.nn.Parameter(torch.tensor(e_disorder_flat).to(device), requires_grad=False)

		self.E_new = torch.nn.Parameter(torch.tensor(np.zeros(N_wells)).to(device), requires_grad=True)
		self.E_desired = torch.nn.Parameter(torch.tensor(E_desired).to(device), requires_grad=False)
		self.gamma_reduction = torch.nn.Parameter(torch.tensor(gamma_reduction).to(device), requires_grad=False)
		self.lam1 = torch.nn.Parameter(torch.tensor(lam1).to(device), requires_grad=False)
		self.lam2 = torch.nn.Parameter(torch.tensor(lam2).to(device), requires_grad=False)
		self.smooth_quench = torch.nn.Parameter(torch.tensor(smooth_quench, dtype=torch.int64).to(device), requires_grad=False)
		self.temperature_dependent_rate = torch.nn.Parameter(torch.tensor(temperature_dependent_rate, dtype=torch.int64).to(device),
											requires_grad=False)

	def forward(self, t, y):
		xL = (self.J * (
							torch.gather(y[:self.N_wells], 0, self.nn_idx_1) +
							torch.gather(y[:self.N_wells], 0, self.nn_idx_2) +
							torch.gather(y[:self.N_wells], 0, self.nn_idy_1) +
							torch.gather(y[:self.N_wells], 0, self.nn_idy_2) +
							self.anisotropy * (torch.gather(y[:self.N_wells], 0, self.nn_idz_1) +
											   torch.gather(y[:self.N_wells], 0, self.nn_idz_2)
											   )
					))

		yL = (self.J * (
					torch.gather(y[self.N_wells:], 0, self.nn_idx_1) +
					torch.gather(y[self.N_wells:], 0, self.nn_idx_2) +
					torch.gather(y[self.N_wells:], 0, self.nn_idy_1) +
					torch.gather(y[self.N_wells:], 0, self.nn_idy_2) +
					self.anisotropy * (torch.gather(y[self.N_wells:], 0, self.nn_idz_1) +
									   torch.gather(y[self.N_wells:], 0, self.nn_idz_2)
									   )
			))
		if self.temperature_dependent_rate.item() == 0:
			return (torch.cat(
				[self.gamma * y[
																										   self.N_wells:] * (
						 xL * y[self.N_wells:] - yL * y[:self.N_wells]) +

				 self.e_disorder * y[self.N_wells:] - yL + self.h_dis_y_flat + self.beta *
				 (torch.pow(y[self.N_wells:], 2) + torch.pow(y[:self.N_wells], 2)) * y[
																					 self.N_wells:]
					,

				 -self.gamma * y[
																											:self.N_wells] * (
						 xL * y[self.N_wells:] - yL * y[:self.N_wells]) - self.e_disorder * y[:self.N_wells] +
				 xL - self.h_dis_x_flat - self.beta *
				 (torch.pow(y[self.N_wells:], 2) + torch.pow(y[:self.N_wells], 2)) * y[:self.N_wells]], dim=0)
			)
		else:
			if self.smooth_quench.item() > 0:
				return (torch.cat(
					[self.quenching_profile(t) * self.gamma * y[
																											   self.N_wells:] * (
							 xL * y[self.N_wells:] - yL * y[:self.N_wells]) +

					 self.e_disorder * y[self.N_wells:] - yL + self.h_dis_y_flat + self.beta *
					 (torch.pow(y[self.N_wells:], 2) + torch.pow(y[:self.N_wells], 2)) * y[
																						 self.N_wells:]
						,

					 -self.quenching_profile(t) * self.gamma * y[
																												:self.N_wells] * (
							 xL * y[self.N_wells:] - yL * y[:self.N_wells]) - self.e_disorder * y[:self.N_wells] +
					 xL - self.h_dis_x_flat - self.beta *
					 (torch.pow(y[self.N_wells:], 2) + torch.pow(y[:self.N_wells], 2)) * y[:self.N_wells]], dim=0)
				)
			else:
				return (torch.cat(
					[(self.gamma_reduction * (self.calc_energy_XY(y,xL,yL) - self.E_desired)) * self.gamma * y[self.N_wells:] * (
							xL * y[self.N_wells:] - yL * y[:self.N_wells]) +

					 self.e_disorder * y[self.N_wells:] - yL + self.h_dis_y_flat + self.beta *
					 (torch.pow(y[self.N_wells:], 2) + torch.pow(y[:self.N_wells], 2)) * y[
																						 self.N_wells:]
						,

					 -(self.gamma_reduction * (self.calc_energy_XY(y,xL,yL) - self.E_desired)) * self.gamma * y[:self.N_wells] * (
							 xL * y[self.N_wells:] - yL * y[:self.N_wells]) - self.e_disorder * y[:self.N_wells] +
					 xL - self.h_dis_x_flat - self.beta *
					 (torch.pow(y[self.N_wells:], 2) + torch.pow(y[:self.N_wells], 2)) * y[:self.N_wells]], dim=0)
						)

	def calc_energy_XY(self, y, xL, yL):
		return torch.sum(self.beta * 0.5 * (
			torch.pow(torch.pow(y[:self.N_wells], 2.) + torch.pow(y[self.N_wells:], 2.), 2.)) +

						 self.e_disorder * (torch.pow(y[:self.N_wells], 2) + torch.pow(y[self.N_wells:], 2))

						 - (y[:self.N_wells] * xL +
									 y[self.N_wells:] * yL) + self.h_dis_x_flat * y[:self.N_wells] + self.h_dis_y_flat * y[self.N_wells:]
						 )

	def quenching_profile(self, time):
		return -self.gamma * torch.pow(self.lam1-self.lam2, -1) * (self.lam1 * torch.exp(-self.lam1 * time) - self.lam2 * torch.exp(-self.lam2 * time))

# def forward(self, t, y):
	#
	#
	# 	return(torch.cat([(self.gamma_reduction * (self.calc_energy_XY(y) - self.E_desired)) * self.gamma * y[self.N_wells:] * (
	# 			(self.J * (
	# 					torch.gather(y[:self.N_wells], 0, self.nn_idx_1) +
	# 					torch.gather(y[:self.N_wells], 0, self.nn_idx_2) +
	# 					torch.gather(y[:self.N_wells], 0, self.nn_idy_1) +
	# 					torch.gather(y[:self.N_wells], 0, self.nn_idy_2) +
	# 					self.anisotropy * (torch.gather(y[:self.N_wells], 0, self.nn_idz_1) +
	# 									   torch.gather(y[:self.N_wells], 0, self.nn_idz_2)
	# 									   )
	# 			)) * y[self.N_wells:]- (self.J * (
	# 				torch.gather(y[self.N_wells:], 0, self.nn_idx_1) +
	# 				torch.gather(y[self.N_wells:], 0, self.nn_idx_2) +
	# 				torch.gather(y[self.N_wells:], 0, self.nn_idy_1) +
	# 				torch.gather(y[self.N_wells:], 0, self.nn_idy_2) +
	# 				self.anisotropy * (torch.gather(y[self.N_wells:], 0, self.nn_idz_1) +
	# 								   torch.gather(y[self.N_wells:], 0, self.nn_idz_2)
	# 								   )
	# 		)) * y[:self.N_wells]) +
	#
	# 		 self.e_disorder * y[self.N_wells:] - (self.J * (
	# 			torch.gather(y[self.N_wells:], 0, self.nn_idx_1) +
	# 			torch.gather(y[self.N_wells:], 0, self.nn_idx_2) +
	# 			torch.gather(y[self.N_wells:], 0, self.nn_idy_1) +
	# 			torch.gather(y[self.N_wells:], 0, self.nn_idy_2) +
	# 			self.anisotropy * (torch.gather(y[self.N_wells:], 0, self.nn_idz_1) +
	# 							   torch.gather(y[self.N_wells:], 0, self.nn_idz_2)
	# 							   )
	# 	)) + self.h_dis_y_flat + self.beta *
	# 								 (torch.pow(y[self.N_wells:], 2) + torch.pow(y[:self.N_wells], 2)) * y[
	# 																									 self.N_wells:]
	# 									,
	#
	# 		-(self.gamma_reduction * (self.calc_energy_XY(y) - self.E_desired)) * self.gamma * y[:self.N_wells] * (
	# 			 (self.J * (
	# 					 torch.gather(y[:self.N_wells], 0, self.nn_idx_1) +
	# 					 torch.gather(y[:self.N_wells], 0, self.nn_idx_2) +
	# 					 torch.gather(y[:self.N_wells], 0, self.nn_idy_1) +
	# 					 torch.gather(y[:self.N_wells], 0, self.nn_idy_2) +
	# 					 self.anisotropy * (
	# 								 torch.gather(y[:self.N_wells], 0, self.nn_idz_1) +
	# 								 torch.gather(y[:self.N_wells], 0, self.nn_idz_2)
	# 								 )
	# 			 )) * y[self.N_wells:] -  (self.J * (
	# 				torch.gather(y[self.N_wells:], 0, self.nn_idx_1) +
	# 				torch.gather(y[self.N_wells:], 0, self.nn_idx_2) +
	# 				torch.gather(y[self.N_wells:], 0, self.nn_idy_1) +
	# 				torch.gather(y[self.N_wells:], 0, self.nn_idy_2) +
	# 				self.anisotropy * (torch.gather(y[self.N_wells:], 0, self.nn_idz_1) +
	# 								   torch.gather(y[self.N_wells:], 0, self.nn_idz_2)
	# 								   )
	# 		)) * y[:self.N_wells]) -self.e_disorder * y[:self.N_wells] +
	# 		(self.J * (
	# 				torch.gather(y[:self.N_wells], 0, self.nn_idx_1) +
	# 				torch.gather(y[:self.N_wells], 0, self.nn_idx_2) +
	# 				torch.gather(y[:self.N_wells], 0, self.nn_idy_1) +
	# 				torch.gather(y[:self.N_wells], 0, self.nn_idy_2) +
	# 				self.anisotropy * (torch.gather(y[:self.N_wells], 0, self.nn_idz_1) +
	# 								   torch.gather(y[:self.N_wells], 0, self.nn_idz_2)
	# 								   )
	# 		)) - self.h_dis_x_flat -self.beta *
	# 		(torch.pow(y[self.N_wells:], 2) + torch.pow(y[:self.N_wells], 2)) * y[:self.N_wells]], dim=0)
   	# 	)
	#
	# def calc_energy_XY(self, y):
	# 	return torch.sum(self.beta * 0.5 * (
	# 		torch.pow(torch.pow(y[:self.N_wells], 2.) + torch.pow(y[self.N_wells:], 2.), 2.)) +
	#
	# 		self.e_disorder * (torch.pow(y[:self.N_wells], 2) + torch.pow(y[self.N_wells:], 2))
	#
	# 		-self.J * (y[:self.N_wells] * (
	# 			torch.gather(y[:self.N_wells], 0, self.nn_idx_1) +
	# 			torch.gather(y[:self.N_wells], 0, self.nn_idx_2) +
	# 			torch.gather(y[:self.N_wells], 0, self.nn_idy_1) +
	# 			torch.gather(y[:self.N_wells], 0, self.nn_idy_2) +
	# 			self.anisotropy * (torch.gather(y[:self.N_wells], 0, self.nn_idz_1) +
	# 									 torch.gather(y[:self.N_wells], 0, self.nn_idz_2)
	# 									 )) +
	# 		   y[self.N_wells:] * (
	# 				   torch.gather(y[self.N_wells:], 0, self.nn_idx_1) +
	# 				   torch.gather(y[self.N_wells:], 0, self.nn_idx_2) +
	# 				   torch.gather(y[self.N_wells:], 0, self.nn_idy_1) +
	# 				   torch.gather(y[self.N_wells:], 0, self.nn_idy_2) +
	# 				   self.anisotropy * (
	# 							   torch.gather(y[self.N_wells:], 0, self.nn_idz_1) +
	# 							   torch.gather(y[self.N_wells:], 0, self.nn_idz_2)
	# 							   )
	# 		   )) + self.h_dis_x_flat * y[:self.N_wells] + self.h_dis_y_flat * y[self.N_wells:]
	# 	)
