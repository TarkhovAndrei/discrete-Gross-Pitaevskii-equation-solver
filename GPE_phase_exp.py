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
from GPElib.dynamics_generator import DynamicsGenerator
from GPElib.visualisation import Visualisation
import matplotlib
print(matplotlib.matplotlib_fname())
import matplotlib.pyplot as plt
import sys

sys.stderr = sys.stdout

def init_instability(inst, traj_seed=42, E=1., from_backup=False, init_conds=None):
    if from_backup == False:
        inst.generate_init(traj_seed, E, kind='random_population_and_phase')
        inst.set_init_XY(inst.X[:,:,:,0], inst.Y[:,:,:,0])
        err = 0
    else:
        inst.set_init_XY(init_conds[0], init_conds[1])
        err = 0
    return err

print(len(sys.argv), sys.argv)

if len(sys.argv) > 4:
	seed_from = int(sys.argv[1])
	seed_to = int(sys.argv[2])
	my_id = int(sys.argv[3])
	unique_id = sys.argv[4]
else:
	seed_from = 0
	seed_to = 1
	needed_trajs = np.arange(seed_from, seed_to)
	my_id = 0
	unique_id = 'ID_not_stated139'

needed_trajs = np.arange(seed_from, seed_to)
perturb_seeds = np.arange(123,124)#(2381,2382)#(100, 110)#(106,108)#(97,98)#(97, 100)#(15, 18) #[53, 12, 20, 87]

# time = 30 * 40 * 80.
time = 51.# * 40 * 80.
time_backup = 51.
N_backups = int(np.ceil(time / time_backup)) + 1
time = N_backups * time_backup

step_large = 0.5
step = 0.05
step_small = 0.01
step_LE = 1.

N_wells = 10.
W = 0.
gamma_tmp = 1.

lyap = DynamicsGenerator(N_part_per_well=1.,
                         W=W, disorder_seed=53,
                         N_wells=(30,30,30), dimensionality=3, anisotropy=1.0,
                         threshold_XY_to_polar=0.25,
                         beta=10.,  #beta_disorder_amplitude=2.,
						local_disorder_amplitude=0.00,
						 FloatPrecision=np.float64,
                        integration_method='RK45',
                         rtol=1e-8, atol=1e-8,
			smooth_quench=True,
						reset_steps_duration=5,
                         calculation_type='lyap_save_all',
			 	integrator='scipy',
				 time=time_backup, step=step, gamma=gamma_tmp)

lyap_dynamic = DynamicsGenerator(N_part_per_well=1.,
                         W=W, disorder_seed=53,
                         N_wells=(30,30,30), dimensionality=3, anisotropy=1.0,
                         threshold_XY_to_polar=0.25,
                         beta=10.,
						 local_disorder_amplitude=0.00,
						 #beta_disorder_amplitude=2.,
						 FloatPrecision=np.float64,
                        integration_method='RK45',
                         rtol=1e-8, atol=1e-8,
						reset_steps_duration=5,
                         calculation_type='lyap_save_all',
			 			integrator='scipy',
				 		time=time_backup, step=step, gamma=1.)

grname = 'GPE_phase_' + unique_id
vis = Visualisation(is_local=0,  HOMEDIR='/data/tarkhov/data/', GROUP_NAMES=grname)
vis_backup = Visualisation(is_local=0,  HOMEDIR='/data/tarkhov/data/backups/', GROUP_NAMES=grname)

# vis = Visualisation(is_local=0,  HOMEDIR='/data1/andrey/data/', GROUP_NAMES=grname)
# vis_backup = Visualisation(is_local=0,  HOMEDIR='/data1/andrey/data/backups/', GROUP_NAMES=grname)

# vis = Visualisation(is_local=1,  HOMEDIR='/Users/tarkhov/tmp/', GROUP_NAMES=grname)
# vis_backup = Visualisation(is_local=1,  HOMEDIR='/Users/tarkhov/tmp/backups/', GROUP_NAMES=grname)

def try_find_backup():
	try:
		backup_intro = np.load(vis_backup.filename('_BACKUP_PRESENT_' + str(my_id)))
		backup_id = backup_intro['backup_id']
		print(backup_id)
		backup = np.load(vis_backup.filename('_BACKUP_PRESENT_' + str(my_id) + '_intermediate_' + str(backup_id)))
		print('Start from backup file')
		backup_present = True
		return backup, backup_id, backup_present
	except:
		print('No backup file, start from 0')
		backup_id = -1
		backup_present = False
		backup = None
		return backup, backup_id, backup_present

backup, backup_id, backup_present = try_find_backup()

print("Noise ", W)
print("Characteristic, full, step times, n_steps")
print(lyap.tau_char, lyap.time, lyap.step, lyap.n_steps)

if backup_present == False:
	num_good = 0
	lmbdas = []
	lmbdas_no_regr = []
	chosen_trajs = []
	effective_nonlinearity = []
	energies = []
	temperatures = []
	energies_true = []
	order_parameters = []
	distances = []
	numb_of_part = []
	next_traj = 0
	next_seed = 0
	backup_id = -1
	curr_traj = -1
	curr_seed = -1
else:
	num_good = backup['num_good']
	lmbdas = backup['lambdas']
	lmbdas_no_regr = backup['lambdas_no_regr']
	chosen_trajs = backup['chosen']
	effective_nonlinearity = backup['eff_nonl']
	energies = backup['energies']
	temperatures = backup['temperatures']
	energies_true = backup['energies_true']
	order_parameters = backup['order_parameters']
	distances = backup['distance']
	numb_of_part = backup['numb_of_part']
	curr_traj = backup['curr_traj']
	perturb_seeds = backup['pert_seeds']
	curr_seed = backup['curr_seed']
	time_finished = backup['time_finished']
	backup_id = backup['backup_id']

def save_backup(backup_id):
	if backup_id == 0:
		init_conds =  [lyap.X[:,:,:,0], lyap.Y[:,:,:,0]]
	else:
		init_conds =  [lyap.X[:,:,:,lyap.icurr], lyap.Y[:,:,:,lyap.icurr]]


	np.savez_compressed(vis_backup.filename('_BACKUP_PRESENT_' + str(my_id)),
	                  backup_id=backup_id)

	np.savez_compressed(vis_backup.filename('_BACKUP_PRESENT_' + str(my_id) + '_intermediate_' + str(backup_id)),
	                 lambdas=lmbdas, lambdas_no_regr=lmbdas_no_regr,
	                 eff_nonl=effective_nonlinearity,
	                 init_conds=init_conds,
			         numb_of_part=numb_of_part, energies=energies,
					temperatures = temperatures,
					energies_true = energies_true,
					order_parameters = order_parameters,
			          pert_seeds=perturb_seeds,
			         chosen=chosen_trajs, step=lyap.step, time=lyap.time, n_steps=lyap.n_steps,
			         my_info=[seed_from, seed_to, my_id], needed_trajs=needed_trajs,
			         checksum=lyap.consistency_checksum, error_code=lyap.error_code,
			         distance=distances,
	                 curr_traj=i_traj, curr_seed=j_traj,
	                    time_finished=k_traj * time_backup,
	                    backup_id=backup_id, num_good=num_good
	                    )

for i_traj, traj_seed in enumerate(needed_trajs):
	if i_traj < curr_traj:
		continue
	if num_good > needed_trajs.shape[0] - 1:
		print('We really have enough trajs, exit!')
		break
	for j_traj, pert_seed in enumerate(perturb_seeds):
		if j_traj < curr_seed:
			continue
		if num_good > needed_trajs.shape[0] - 1:
			print('We really have enough trajs, exit!')
			break
		for k_traj in range(N_backups):
			if k_traj <= backup_id:
				continue
			backup, backup_id, backup_present = try_find_backup()

			lyap.X = lyap.X * 0.
			lyap.Y = lyap.Y * 0.
			lyap.RHO = lyap.RHO * 0.
			lyap.THETA = lyap.THETA * 0.
			lyap.icurr = 0
			lyap.inext = 1

			if backup_present:
				print('Backup found: ', backup_id, k_traj)
				# err = init_instability(lyap, traj_seed, from_backup=True, init_conds=backup['init_conds'])
				# lyap.traj_seed = traj_seed
				# lyap.pert_seed = pert_seed
				#
				np.random.seed()
				traj_seed = np.random.randint(100000)
				pert_seed = np.random.randint(100000)
				lyap.traj_seed = traj_seed
				lyap.pert_seed = pert_seed
				err = 1
				while err == 1:
					traj_seed = np.random.randint(100000)
					pert_seed = np.random.randint(100000)
					print("SEED: ", traj_seed)
					err = init_instability(lyap, traj_seed)
					if err == 1:
						print('Bad trajectory! ', i_traj)
				print('Good trajectory found!')

			else:
				print('No backup found, going to find a track')
				np.random.seed()
				traj_seed = np.random.randint(100000)
				pert_seed = np.random.randint(100000)
				lyap.traj_seed = traj_seed
				lyap.pert_seed = pert_seed
				err = 1
				while err == 1:
					traj_seed = np.random.randint(100000)
					print("SEED: ", traj_seed)
					err = init_instability(lyap, traj_seed)
					if err == 1:
						print('Bad trajectory! ', i_traj)
				print('Good trajectory found!')
				lyap.n_steps = int(time_backup / step)
				lyap.time = time_backup
				save_backup(k_traj)
				continue

			x0 = lyap.X[:,:,:,0].copy()
			y0 = lyap.Y[:,:,:,0].copy()
			lyap.X *= 0
			lyap.Y *= 0
			lyap.RHO *= 0
			lyap.THETA *= 0
			lyap.icurr = 0
			lyap.inext = 1

			lyap.set_init_XY(x0, y0)
			lyap.step = step_small
			lyap.run_relaxation(E_desired=(-4. + lyap.beta_amplitude/2)*lyap.N_wells, temperature_dependent_rate=False, N_max=200)

			x0 = lyap.X[:,:,:,lyap.icurr-1].copy()
			y0 = lyap.Y[:,:,:,lyap.icurr-1].copy()

			E0 = lyap.calc_energy_XY(x0, y0, 0)
			
			lyap.X *= 0
			lyap.Y *= 0
			lyap.RHO *= 0
			lyap.THETA *= 0
			lyap.icurr = 0
			lyap.inext = 1

			lyap.set_init_XY(x0, y0)
			lyap.step = step_large
			lyap.n_steps = 200

			lyap.run_dynamics(no_pert=False)

			x0 = lyap.X[:,:,:,lyap.icurr-1].copy()
			y0 = lyap.Y[:,:,:,lyap.icurr-1].copy()

			E_min_all = (-2.7 + lyap.beta_amplitude/2.) * lyap.N_wells
			E_max_all = (3 + lyap.beta_amplitude/2.) * lyap.N_wells

			Energies = np.linspace(E_min_all, E_max_all, num=40)
			gammas = np.linspace(0.1, 2., num=40)
			#Efinal = (np.linspace(-4,1, num=40) + lyap.beta_amplitude/2) * lyap.N_wells
			Efinal = (np.zeros(40) - 4. + lyap.beta_amplitude/2) * lyap.N_wells
			#Efinal = (np.linspace(-4,1, num=40) + 10./2) * 1000
			T_conservative = 40.
			dt_conservative = 0.05
			n_conservative = int(T_conservative/dt_conservative)

			T_quench = 0.
			dt_quench = 0.05
			n_quench = int(T_quench/dt_quench)

			T_conservative_post_quench = 0.
			dt_conservative_post_quench = 0.05
			n_conservative_post_quench = int(T_conservative_post_quench/dt_conservative_post_quench)

			total_time_steps = n_conservative + n_quench + n_conservative_post_quench

			energy_i = np.full((1, Energies.shape[0],) + (total_time_steps,), np.nan)
			order_parameter_i = np.full((1, Energies.shape[0],) + lyap.X.shape[:-1] + (total_time_steps,), np.nan, dtype=np.complex64)
			order_parameter_i_1 = np.full((1, Energies.shape[0],) + lyap.X.shape[:-1] + (total_time_steps,), np.nan, dtype=np.complex64)
			temperature_i = np.full((1, Energies.shape[0],) + (total_time_steps,), np.nan)
			temperature_Amp_i = np.full((1, Energies.shape[0],) + (total_time_steps,), np.nan)
			temperature_Ph_i = np.full((1, Energies.shape[0],) + (total_time_steps,), np.nan)

			x1 = x0.copy()
			y1 = y0.copy()

			idx_iterations = np.hstack(
										(np.hstack((np.nonzero(Energies > E0)[0], (-1))),
										 np.nonzero(np.logical_not(Energies > E0))[0][::-1])
									  )
			for j in idx_iterations:
				if j == -1:
					continue
				x1 = x0.copy()
				y1 = y0.copy()

				print('Energy %f' % Energies[j])

				lyap.X *= 0
				lyap.Y *= 0
				lyap.RHO *= 0
				lyap.THETA *= 0
				lyap.icurr = 0
				lyap.inext = 1

				lyap.set_init_XY(x1, y1)
				lyap.step = dt_conservative
				lyap.n_steps = n_conservative
				lyap.gamma = gammas[j]
				#lyap.run_dynamics(no_pert=False)
				lyap.run_relaxation(no_pert=False, E_desired=-1e+9, temperature_dependent_rate=True,
									N_max=lyap.n_steps)
				for istep in np.arange(lyap.n_steps):
					energy_i[0,j,istep] = lyap.calc_energy_XY(lyap.X[:,:,:,istep],lyap.Y[:,:,:,istep], 0)
					order_parameter_i[0,j,:,:,:,istep] = lyap.X[:,:,:,istep] + 1j * lyap.Y[:,:,:,istep]
					order_parameter_i_1[0,j,:,:,:,istep] = lyap.X[:,:,:,istep] + 1j * lyap.Y[:,:,:,istep]


			if backup_present:
				print('Backup present, go adding results')
				num_good = backup['num_good']
				lmbdas = backup['lambdas']
				lmbdas_no_regr = backup['lambdas_no_regr']
				chosen_trajs = backup['chosen']
				effective_nonlinearity = backup['eff_nonl']
				energies = backup['energies']
				temperatures = backup['temperatures']
				energies_true = backup['energies_true']
				order_parameters = backup['order_parameters']
				distances = backup['distance']
				numb_of_part = backup['numb_of_part']
				curr_traj = backup['curr_traj']
				curr_seed = backup['curr_seed']
				time_finished = backup['time_finished']
				backup_id = backup['backup_id']


				if len(energies_true) == 0:
					energies_true = energy_i
					temperatures = temperature_i
					temperatures_Ph = temperature_Ph_i

					temperatures_Amp = temperature_Amp_i
					order_parameters = order_parameter_i
					order_parameters_1 = order_parameter_i_1
				else:
					# print(energies_true.shape, energy_i.shape)
					energies_true = np.concatenate((energies_true, energy_i), axis=0)
					temperatures = np.concatenate((temperatures, temperature_i), axis=0)
					temperatures_Amp = np.concatenate((temperatures_Amp, temperature_Amp_i), axis=0)
					temperatures_Ph = np.concatenate((temperatures_Ph, temperature_Ph_i), axis=0)
					# print('stack axis 0: ', order_parameters.shape, order_parameter_i.shape)
					order_parameters = np.concatenate((order_parameters, order_parameter_i), axis=0)
					order_parameters_1 = np.concatenate((order_parameters_1, order_parameter_i_1), axis=0)

				energies = np.hstack((energies, lyap.energy))

				numb_of_part = np.hstack((numb_of_part, lyap.number_of_particles))

				save_backup(k_traj)
		num_good += 1

print("Error code: ", lyap.error_code)
print("\n\nChecksum: ", lyap.consistency_checksum)

np.savez_compressed(vis.filename(my_id),
         lambdas=lmbdas, lambdas_no_regr=lmbdas_no_regr,
         eff_nonl=effective_nonlinearity,
         numb_of_part=numb_of_part, energies=energies,
        temperatures = temperatures,
		energies_true = energies_true,
		order_parameters = order_parameters,
         chosen=chosen_trajs, step=lyap.step, time=lyap.time, n_steps=lyap.n_steps,
         my_info=[seed_from, seed_to, my_id], needed_trajs=needed_trajs,
         checksum=lyap.consistency_checksum, error_code=lyap.error_code,
         distance=distances)
