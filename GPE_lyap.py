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
from GPElib.lyapunov_generator import LyapunovGenerator
from GPElib.visualisation import Visualisation
import matplotlib
print matplotlib.matplotlib_fname()
import matplotlib.pyplot as plt
import sys

sys.stderr = sys.stdout

def init_instability(inst, traj_seed, from_backup=False, init_conds=None):
	if from_backup == False:
		inst.generate_init('random', traj_seed, 100.)
		# delta = (2. * np.sqrt(1.0 * inst.N_part/inst.N_wells)) * np.random.rand()
		delta = (2. * np.sqrt(1.0 * inst.N_part)) * np.random.rand()
		x0, y0, err = inst.E_const_perturbation_XY(inst.X[:,:,:,0], inst.Y[:,:,:,0], delta, degrees_of_freedom=10)
		x1, y1 = inst.constant_perturbation_XY(x0,y0)
		inst.set_init_XY(x0,y0,x1,y1)
	else:
		inst.set_init_XY(init_conds[0], init_conds[1], init_conds[2], init_conds[3])
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
time = 10.# * 40 * 80.
time_backup = 1.
N_backups = int(np.ceil(time / time_backup)) + 1
time = N_backups * time_backup

# time = 1.
# time = 100. * 15
# step = 0.00015625
step = 0.001
N_wells = 10.
W = 0.

lyap = LyapunovGenerator(N_part_per_well=100,
                         W=W, disorder_seed=53,
                         # N_wells=(10,1,1), dimensionality=1, threshold_XY_to_polar=0.25,
                         # N_wells=(10,10,1), dimensionality=2, threshold_XY_to_polar=0.25,
                         N_wells=(10,1,1), dimensionality=1, threshold_XY_to_polar=0.25,
                         reset_steps_duration=5,
                         # reset_steps_duration=150,
                         time=time_backup, step=step)

grname = 'GPE_lyap_' + unique_id
# vis = Visualisation(is_local=0,  HOMEDIR='/data1/andrey/data/', GROUP_NAMES=grname)
# vis_backup = Visualisation(is_local=0,  HOMEDIR='/data1/andrey/data/backups/', GROUP_NAMES=grname)

vis = Visualisation(is_local=1,  HOMEDIR='/Users/tarkhov/tmp/', GROUP_NAMES=grname)
vis_backup = Visualisation(is_local=1,  HOMEDIR='/Users/tarkhov/tmp/backups/', GROUP_NAMES=grname)

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
	distances = backup['distance']
	numb_of_part = backup['numb_of_part']
	curr_traj = backup['curr_traj']
	perturb_seeds = backup['pert_seeds']
	curr_seed = backup['curr_seed']
	time_finished = backup['time_finished']
	backup_id = backup['backup_id']

def save_backup(backup_id):
	if backup_id == 0:
		init_conds =  [lyap.X[:,:,:,0], lyap.Y[:,:,:,0], lyap.X1[:,:,:,0], lyap.Y1[:,:,:,0]]
	else:
		init_conds =  [lyap.X[:,:,:,lyap.icurr], lyap.Y[:,:,:,lyap.icurr], lyap.X1[:,:,:,lyap.icurr], lyap.Y1[:,:,:,lyap.icurr]]


	np.savez_compressed(vis_backup.filename('_BACKUP_PRESENT_' + str(my_id)),
	                  backup_id=backup_id)

	np.savez_compressed(vis_backup.filename('_BACKUP_PRESENT_' + str(my_id) + '_intermediate_' + str(backup_id)),
	                 lambdas=lmbdas, lambdas_no_regr=lmbdas_no_regr,
	                 eff_nonl=effective_nonlinearity,
	                 init_conds=init_conds,
			         numb_of_part=numb_of_part, energies=energies,
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
				err = init_instability(lyap, traj_seed, from_backup=True, init_conds=backup['init_conds'])
				lyap.traj_seed = traj_seed
				lyap.pert_seed = pert_seed
			else:
				print 'No backup found, going to find a track'
				np.random.seed()
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

			lyap.n_steps = int(time_backup / step)
			lyap.time = time_backup
			lyap.run_dynamics(no_pert=True)

			if backup_present:
				print 'Backup present, go adding results'
				num_good = backup['num_good']
				lmbdas = backup['lambdas']
				lmbdas_no_regr = backup['lambdas_no_regr']
				chosen_trajs = backup['chosen']
				effective_nonlinearity = backup['eff_nonl']
				energies = backup['energies']
				distances = backup['distance']
				numb_of_part = backup['numb_of_part']
				curr_traj = backup['curr_traj']
				curr_seed = backup['curr_seed']
				time_finished = backup['time_finished']
				backup_id = backup['backup_id']

				lmbdas = np.hstack((lmbdas, lyap.lambdas[:-1]))
				lmbdas_no_regr =  np.hstack((lmbdas_no_regr, lyap.lambdas_no_regr[:-1]))
				# chosen_trajs.append((traj_seed, pert_seed))
				effective_nonlinearity = np.hstack((effective_nonlinearity, lyap.effective_nonlinearity))
				energies = np.hstack((energies, lyap.energy))
				print energies
				distances = np.hstack((distances, lyap.distance))
				numb_of_part = np.hstack((numb_of_part, lyap.number_of_particles))
				print lyap.lambdas
				print lyap.lambdas_no_regr

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
         chosen=chosen_trajs, step=lyap.step, time=lyap.time, n_steps=lyap.n_steps,
         my_info=[seed_from, seed_to, my_id], needed_trajs=needed_trajs,
         checksum=lyap.consistency_checksum, error_code=lyap.error_code,
         distance=distances)
