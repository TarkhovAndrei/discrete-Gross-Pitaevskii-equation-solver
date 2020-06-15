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
print matplotlib.matplotlib_fname()
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

print len(sys.argv), sys.argv

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
	unique_id = 'ID_not_stated13'

needed_trajs = np.arange(seed_from, seed_to)
perturb_seeds = np.arange(123,124)#(2381,2382)#(100, 110)#(106,108)#(97,98)#(97, 100)#(15, 18) #[53, 12, 20, 87]

# time = 30 * 40 * 80.
time = 100.# * 40 * 80.
time_backup = 100.
N_backups = int(np.ceil(time / time_backup)) + 1
time = N_backups * time_backup

step = 0.1
N_wells = 10.
W = 0.
gamma_tmp = 1.

lyap = DynamicsGenerator(N_part_per_well=1.,
                         W=W, disorder_seed=53,
                         N_wells=(20,20,20), dimensionality=3, anisotropy=1.0,
                         threshold_XY_to_polar=0.25,
                         beta=3., FloatPrecision=np.float64,
                        integration_method='RK45',
                         rtol=1e-8, atol=1e-8, 
			reset_steps_duration=5,
                         calculation_type='lyap_save_all',
			 integrator='scipy',
                         time=time_backup, step=step, gamma=gamma_tmp)

grname = 'GPE_phase_' + unique_id
vis = Visualisation(is_local=0,  HOMEDIR='/data1/andrey/data/', GROUP_NAMES=grname)
vis_backup = Visualisation(is_local=0,  HOMEDIR='/data1/andrey/data/backups/', GROUP_NAMES=grname)

# vis = Visualisation(is_local=1,  HOMEDIR='/Users/tarkhov/tmp/', GROUP_NAMES=grname)
# vis_backup = Visualisation(is_local=1,  HOMEDIR='/Users/tarkhov/tmp/backups/', GROUP_NAMES=grname)

def try_find_backup():
	try:
		backup_intro = np.load(vis_backup.filename('_BACKUP_PRESENT_' + str(my_id)))
		backup_id = backup_intro['backup_id']
		print backup_id
		backup = np.load(vis_backup.filename('_BACKUP_PRESENT_' + str(my_id) + '_intermediate_' + str(backup_id)))
		print 'Start from backup file'
		backup_present = True
		return backup, backup_id, backup_present
	except:
		print 'No backup file, start from 0'
		backup_id = -1
		backup_present = False
		backup = None
		return backup, backup_id, backup_present

backup, backup_id, backup_present = try_find_backup()

print "Noise ", W
print "Characteristic, full, step times, n_steps"
print lyap.tau_char, lyap.time, lyap.step, lyap.n_steps

if backup_present == False:
	num_good = 0
	lmbdas = []
	lmbdas_no_regr = []
	chosen_trajs = []
	effective_nonlinearity = []
	energies = []
	temperatures = []
	temperatures_Amp = []
	temperatures_Ph = []
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
	temperatures_Amp = backup['temperatures_Amp']
	temperatures_Ph = backup['temperatures_Ph']
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
					temperatures_Amp = temperatures_Amp,
					temperatures_Ph = temperatures_Ph,
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
		print 'We really have enough trajs, exit!'
		break
	for j_traj, pert_seed in enumerate(perturb_seeds):
		if j_traj < curr_seed:
			continue
		if num_good > needed_trajs.shape[0] - 1:
			print 'We really have enough trajs, exit!'
			break
		for k_traj in xrange(N_backups):
			if k_traj <= backup_id:
				continue
			backup, backup_id, backup_present = try_find_backup()
			if backup_present:
				print 'Backup found: ', backup_id, k_traj
				# err = init_instability(lyap, traj_seed, from_backup=True, init_conds=backup['init_conds'])
				# lyap.traj_seed = traj_seed
				# lyap.pert_seed = pert_seed
				#
				np.random.seed()
				traj_seed = np.random.randint(100000)
				lyap.traj_seed = traj_seed
				lyap.pert_seed = pert_seed
				err = 1
				while err == 1:
					traj_seed = np.random.randint(100000)
					print "SEED: ", traj_seed
					err = init_instability(lyap, traj_seed)
					if err == 1:
						print 'Bad trajectory! ', i_traj
				print 'Good trajectory found!'

			else:
				print 'No backup found, going to find a track'
				np.random.seed()
				traj_seed = np.random.randint(100000)
				lyap.traj_seed = traj_seed
				lyap.pert_seed = pert_seed
				err = 1
				while err == 1:
					traj_seed = np.random.randint(100000)
					print "SEED: ", traj_seed
					err = init_instability(lyap, traj_seed)
					if err == 1:
						print 'Bad trajectory! ', i_traj
				print 'Good trajectory found!'
				lyap.n_steps = int(time_backup / step)
				lyap.time = time_backup
				save_backup(k_traj)
				continue

			lyap.n_steps = int(0.5/step)
			lyap.run_dynamics(no_pert=False)
			
			x0 = lyap.X[:,:,:,0]
			y0 = lyap.Y[:,:,:,0]
			
			
			init_conds = []
			init_energies = []
			lyap.run_relaxation(E_desired=1e+7, N_max=50)
			for i in xrange(lyap.n_steps):
				init_energies.append(lyap.energy[i])
				init_conds.append((lyap.X[:,:,:,i].copy(),lyap.Y[:,:,:,i].copy()))

			#E_max = lyap.calc_energy_XY(lyap.X[:,:,:,0],lyap.Y[:,:,:,0], 0)
			lyap.set_init_XY(x0, y0)
			lyap.run_relaxation(E_desired=-1e+7, N_max=50)
			for i in xrange(lyap.n_steps):
				init_energies.append(lyap.energy[i])
				init_conds.append((lyap.X[:,:,:,i].copy(),lyap.Y[:,:,:,i].copy()))
			init_energies = np.array(init_energies)
			E_max = np.max(init_energies)
			E_min = np.min(init_energies)
			isort_en = np.argsort(init_energies)
			init_energies = init_energies[isort_en]
			#init_conds = init_conds[isort_en]
			
			#E_min = lyap.calc_energy_XY(lyap.X[:,:,:,0],lyap.Y[:,:,:,0], 0)

			E_min_all = -5 * 8000
			E_max_all = 8 * 8000

			print E_min, E_max

			Energies = np.linspace(E_min_all, E_max_all, num=70)
			energy_i = np.zeros(Energies.shape[0])
			order_parameter_i = np.zeros(Energies.shape[0])
			temperature_i = np.zeros(Energies.shape[0])
			temperature_Amp_i = np.zeros(Energies.shape[0])
			temperature_Ph_i = np.zeros(Energies.shape[0])

			for j in np.nonzero(np.logical_and(Energies > E_min, Energies < E_max))[0]:#xrange(Energies.shape[0]):
				print 'Energy %f' % Energies[j]
				i_en = np.nonzero(init_energies >= Energies[j])[0][0]
				lyap.set_init_XY(init_conds[isort_en[i_en]][0], init_conds[isort_en[i_en]][1])
				#lyap.set_init_XY(x0, y0)
				#lyap.run_relaxation(E_desired=Energies[j], N_max=100)
				#lyap.run_quench(E_desired=Energies[j])
				lyap.n_steps = int(0.5/step)
				
				lyap.run_dynamics(no_pert=False)
				
				energy_i[j] = lyap.calc_energy_XY(lyap.X[:,:,:,lyap.n_steps - 1],lyap.Y[:,:,:,lyap.n_steps - 1], 0)
				T, T_Amp, T_Ph = lyap.calc_numerical_temperature(lyap.X[:,:,:,lyap.n_steps - 1],lyap.Y[:,:,:,lyap.n_steps - 1], N_samples=5000)
				order_parameter_i[j] = np.sqrt(np.sum(lyap.X[:,:,:,lyap.n_steps - 1]) ** 2 + np.sum(lyap.Y[:,:,:,lyap.n_steps - 1]) ** 2)
				temperature_i[j] = T
				temperature_Amp_i[j] = T_Amp
				temperature_Ph_i[j] = T_Ph	
				print 'Temperature %f' % T
				print 'Temperature Amp %f' % T_Amp
				print 'Temperature Ph %f' % T_Ph

			if backup_present:
				print 'Backup present, go adding results'
				num_good = backup['num_good']
				lmbdas = backup['lambdas']
				lmbdas_no_regr = backup['lambdas_no_regr']
				chosen_trajs = backup['chosen']
				effective_nonlinearity = backup['eff_nonl']
				energies = backup['energies']
				temperatures = backup['temperatures']
				temperatures_Amp = backup['temperatures_Amp']
				temperatures_Ph = backup['temperatures_Ph']
				energies_true = backup['energies_true']
				order_parameters = backup['order_parameters']
				distances = backup['distance']
				numb_of_part = backup['numb_of_part']
				curr_traj = backup['curr_traj']
				curr_seed = backup['curr_seed']
				time_finished = backup['time_finished']
				backup_id = backup['backup_id']

				if len(energies) == 0:
					energies_true = energy_i
					temperatures = temperature_i
					temperatures_Amp = temperature_Amp_i
					temperatures_Ph = temperature_Ph_i
					order_parameters = order_parameter_i
				else:
					energies_true = np.vstack((energies_true, energy_i))
					temperatures = np.vstack((temperatures, temperature_i))
					temperatures_Amp = np.vstack((temperatures_Amp, temperature_Amp_i))
					temperatures_Ph = np.vstack((temperatures_Ph, temperature_Ph_i))
					order_parameters = np.vstack((order_parameters, order_parameter_i))
				#
				# lmbdas = np.hstack((lmbdas, lyap.lambdas[:-1]))
				# lmbdas_no_regr =  np.hstack((lmbdas_no_regr, lyap.lambdas_no_regr[:-1]))
				# chosen_trajs.append((traj_seed, pert_seed))
				# effective_nonlinearity = np.hstack((effective_nonlinearity, lyap.effective_nonlinearity))
				energies = np.hstack((energies, lyap.energy))
				# print energies
				# distances = np.hstack((distances, lyap.distance))
				numb_of_part = np.hstack((numb_of_part, lyap.number_of_particles))
				# print lyap.lambdas
				# print lyap.lambdas_no_regr

				save_backup(k_traj)
				# np.savez_compressed(vis_backup.filename(my_id),
				#          lambdas=lmbdas, lambdas_no_regr=lmbdas_no_regr,
				#          eff_nonl=effective_nonlinearity,
				#          numb_of_part=numb_of_part, energies=energies,
				#          chosen=chosen_trajs, step=lyap.step, time=lyap.time, n_steps=lyap.n_steps,
				#          my_info=[seed_from, seed_to, my_id], needed_trajs=needed_trajs,
				#          checksum=lyap.consistency_checksum, error_code=lyap.error_code,
				#          distance=lyap.distance)
		num_good += 1

	# plt.semilogy(lyap.T, lyap.distance)
	# np.savez_compressed(vis.filename(my_id) + '_traj_' + str(i_traj),
	#          step=lyap.step, time=lyap.time,
	#          traj_seed=lyap.traj_seed,
	#          pert_seed=lyap.pert_seed,
	#          disorder_seed=lyap.disorder_seed,
	#          disorder=lyap.e_disorder,
	#          n_steps=lyap.n_steps,
	#          wells_indices=lyap.wells_indices,
	#          beta=lyap.beta, W=lyap.W,
	#          J=lyap.J, N_tuple=lyap.N_tuple,
	#          energy=lyap.energy, number_of_particles=lyap.number_of_particles,
	#          eff_nonl=lyap.effective_nonlinearity,
	#          error_code=lyap.error_code, checksum=lyap.consistency_checksum,
	#          distance=lyap.distance,
	#          x=lyap.X, y=lyap.Y, x1=lyap.X1, y1=lyap.Y1,
	#          lambdas=lyap.lambdas, lambdas_no_regr=lyap.lambdas_no_regr,
	#          hist2d=lyap.histograms, hist1d=lyap.rho_histograms,
	#          hist2d1=lyap.histograms1, hist1d1=lyap.rho_histograms1)

# plt.savefig(vis.HOMEDIR + 'pics/Lyap_' + unique_id + '_' + str(my_id)+'.png', format='png', dpi=100)

print "Error code: ", lyap.error_code
print "\n\nChecksum: ", lyap.consistency_checksum

np.savez_compressed(vis.filename(my_id),
         lambdas=lmbdas, lambdas_no_regr=lmbdas_no_regr,
         eff_nonl=effective_nonlinearity,
         numb_of_part=numb_of_part, energies=energies,
        temperatures = temperatures,
	temperatures_Amp = temperatures_Amp,
	temperatures_Ph = temperatures_Ph,
		energies_true = energies_true,
		order_parameters = order_parameters,
         chosen=chosen_trajs, step=lyap.step, time=lyap.time, n_steps=lyap.n_steps,
         my_info=[seed_from, seed_to, my_id], needed_trajs=needed_trajs,
         checksum=lyap.consistency_checksum, error_code=lyap.error_code,
         distance=distances)
