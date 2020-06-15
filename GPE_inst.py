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
from GPElib.instability_generator import InstabilityGenerator
from GPElib.visualisation import Visualisation
import matplotlib
print matplotlib.matplotlib_fname()
import matplotlib.pyplot as plt
import sys

sys.stderr = sys.stdout

def init_instability(inst, traj_seed):
	inst.generate_init('random', traj_seed, 100.)
	# delta = (2. * np.sqrt(1.0 * inst.N_part/inst.N_wells)) * np.random.rand()
	delta = (2. * np.sqrt(1.0 * inst.N_part)) * np.random.rand()
	x0, y0, err = inst.E_const_perturbation_XY(inst.X[:,:,:,0], inst.Y[:,:,:,0], delta)
	x1, y1 = inst.constant_perturbation_XY(x0,y0)
	inst.set_init_XY(x0,y0,x1,y1)
	return err

print len(sys.argv), sys.argv

if len(sys.argv) > 4:
	seed_from = int(sys.argv[1])
	seed_to = int(sys.argv[2])
	my_id = int(sys.argv[3])
	unique_id = sys.argv[4]
else:
	seed_from = 0
	seed_to = 3
	my_id = 0
	unique_id = 'ID_not_stated'

needed_trajs = np.arange(seed_from, seed_to)

# time = 100.
time = 60.
# step = 0.00015625
step = 0.001
N_wells = 100
W = 0.

inst = InstabilityGenerator(N_part_per_well=100,
                            N_wells=(100,1,1), dimensionality=1,
                            # N_wells=(10,10,1), dimensionality=2,
                            # N_wells=(4,4,4), dimensionality=3,
                            disorder_seed=53, time=time, step=step,
                            perturb_hamiltonian=False, calculation_type='inst',
                            error_J=0, error_beta=0, error_disorder=0)

grname = 'Instability_' + unique_id
vis = Visualisation(is_local=0, GROUP_NAMES=grname)

print "Characteristic, full, step times, n_steps"
print inst.tau_char, inst.time, inst.step, inst.n_steps

answers = []
lambdas = []
lambdas_no_regr = []
polarisation = []
polarisation1 = []
dist = []
energy = []

full_tmp = []
for ii in needed_trajs:
	inst.traj_seed = ii
	np.random.seed()

	err = 1
	while err == 1:
		traj_seed = np.random.randint(100000)
		print "SEED: ", traj_seed
		err = init_instability(inst, traj_seed)
		if err == 1:
			print 'Bad trajectory! ', ii

	inst.set_pert_seed(ii)
	inst.run_dynamics()
	answers.append(inst.distance)
	lambdas.append(inst.lambdas[0])
	lambdas_no_regr.append(inst.lambdas_no_regr[0])
	polarisation.append(inst.polarisation)
	polarisation1.append(inst.polarisation1)
	dist.append(inst.distance)
	energy.append(inst.energy)
	rhosq = (inst.X ** 2 + inst.Y ** 2)
	rho1sq = (inst.X1 ** 2 + inst.Y1 ** 2)
	rev_idx = np.arange(rho1sq.shape[3])[::-1]
	full_tmp.append(np.sqrt(np.sum((rhosq - rho1sq[:,:,:,rev_idx]) ** 2, axis=(0,1,2))))


np.savez_compressed(vis.filename(my_id),
         full=full_tmp,
         data=answers, lambdas=lambdas, lambdas_no_regr=lambdas_no_regr,
         polar=polarisation, polar1=polarisation1,
         distance=dist, energy=energy,
         step=inst.step, time=inst.time, n_steps=inst.n_steps,
         well_indices=inst.wells_indices,
         my_info=[seed_from, seed_to, my_id],
         needed_trajs=needed_trajs,
         description='Loschmidt echo for 10000 replicates, trying to calculate F(t)')
