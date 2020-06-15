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
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

class Visualisation:
	def __init__(self, **kwargs):
		self.fontlabel = 20
		self.fontticks = 20
		self.configure(kwargs)

	def configure(self, kwargs):
		self.is_local = kwargs.get('is_local', 1)
		if self.is_local == 1:
			self.HOMEDIR = kwargs.get('HOMEDIR', '/Users/tarkhov/Skoltech/DGPE_data/data/')
		else:
			self.HOMEDIR = kwargs.get('HOMEDIR', '/data/andrey/data/')
		self.GROUP_NAMES = kwargs.get('GROUP_NAMES', 'Lyapunov_calcs_')
		self.FILE_TYPE = kwargs.get('FILE_TYPE', '.npz')

	def filename(self, i):
		return self.HOMEDIR + self.GROUP_NAMES + str(i) + self.FILE_TYPE

	def plot_3D_dynamics(self, ax, x_dat, y_dat, z_dat, i, N_wells, color):
		ax.set_xlabel(r'$Re(\psi)$',fontsize=self.fontlabel)
		ax.set_ylabel(r'$Im(\psi)$',fontsize=self.fontlabel)
		ax.set_zlabel(r'$Time$',fontsize=self.fontlabel)
		ax.set_title('Spin ' + str(i), fontsize=self.fontlabel)
		plt.setp(ax.get_xticklabels(), fontsize=self.fontticks)
		plt.setp(ax.get_yticklabels(), fontsize=self.fontticks)
		plt.setp(ax.get_zticklabels(), fontsize=self.fontticks)
		plt.xlim([-300,300])
		plt.setp(ax.get_zticklabels(), fontsize=self.fontticks)
		ax.plot(x_dat, y_dat, z_dat,color=color)

	def plot_2D_dynamics(self, ax, x_dat, y_dat, i, N_wells, color):
		ax.set_xlabel(r'$\theta$',fontsize=self.fontlabel)
		ax.set_ylabel(r'$|\psi|$',fontsize=self.fontlabel)
		ax.set_title('Spin ' + str(i), fontsize=self.fontlabel)
		plt.setp(ax.get_xticklabels(), fontsize=self.fontticks)
		plt.setp(ax.get_yticklabels(), fontsize=self.fontticks)
		ax.plot(x_dat, y_dat,color=color)

	def animate_dynamics(self, X, Y, video_prefix):
		# First set up the figure, the axis, and the plot element we want to animate
		nx = 1
		ny = X.shape[1]
		# fig = plt.figure(figsize=(5,20))
		fig = plt.figure(figsize=(30,5))
		# ax = fig.add_axes([.05, .90, .9, .08], polar=False, xticks=[], yticks=[])
		ax = fig.add_axes([.05, .05, .9, .9], polar=False, xticks=[], yticks=[])
		# ax1 = fig.add_axes([.05, .90, .9, .08], polar=False, xticks=[], yticks=[])
		shift = 2 * np.max(np.abs(X)) + 1
		shift_vec = 1.0 * np.arange(ny) * shift
		#     lines = [axarr[j].plot([0, x],[0,y])[0] for j in xrange(data.shape[0])]

		for j in xrange(ny):
			#         ax.grid(1)
			# title = ax1.text (0.02, 0.5, '', fontsize=14, transform=ax1.transAxes)
			ax.plot([shift_vec + np.min(X)-1,shift_vec + np.max(X)+1],[0,0], 'k', linewidth=3)
			ax.plot([shift_vec,shift_vec],[np.min(Y)-1,np.max(Y)+1], 'k', linewidth=3)
		ax.set_xlim([np.min(X)-1,np.max(X) + np.max(shift_vec)])
		ax.set_ylim([np.min(Y)-1, np.max(Y)+1])
		#         lines.append([axarr[j].plot(X[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data])

		# animation function.  This is called sequentially
		def animate(i):
			ax.clear()
			ax.set_xlim([np.min(X)-1,np.max(X) + np.max(shift_vec)])
			ax.set_ylim([np.min(Y)-1, np.max(Y)+1])
			for j in xrange(X.shape[1]):
				ax.plot([shift_vec[j],X[i,j] + shift_vec[j]],[0, Y[i,j]], 'k', lw=2)
				ax.plot([X[i,j] + shift_vec[j]],[Y[i,j]], 'ro')

		anim = animation.FuncAnimation(fig, animate,
		                               #                                 frames=200,
		                               frames=X.shape[0],
		                               interval=20, blit=False)
		anim.save(self.HOMEDIR + self.GROUP_NAMES + video_prefix + '_dynamics.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
		plt.show()

	def animate_dynamics_old(self, X, Y, video_prefix):
		# First set up the figure, the axis, and the plot element we want to animate
		nx = 10
		ny = 1
		# fig, axarr = plt.subplots(nx, ny)
		fig, axarr = plt.figure()
		fig.set_figheight(7)
		fig.set_figwidth(20)
		# fig = plt.figure(figsize=(10,10))
		ax1 = fig.add_axes([.05, .90, .9, .08], polar=False, xticks=[], yticks=[])
		# line = [] * X.shape[1]
		for j, ax in enumerate(axarr):
			ax = fig.add_axes([.05, .05, .9, .8], polar=False)
			ax.set_ylim(np.min(Y)-1, np.max(Y)+1)
			ax.set_xlim(np.min(X)-1, np.max(X)+1)
			ax.grid(1)
			xdata, ydata = [], []
			# title = ax1.text (0.02, 0.5, '', fontsize=14, transform=ax1.transAxes)
			# ax.plot([np.min(X)-1,np.max(X)+1],[0,0], 'k', linewidth=3)
			# ax.plot([0,0],[np.min(Y)-1,np.max(Y)+1], 'k', linewidth=3)
			# line[j], = ax.plot([], [], 'd', marker='o')#lw=2)

		# initialization function: plot the background of each frame
		def init():
			# for j in xrange(10):
			# 	line[j].set_data([], [])
			# return line,
			pass

		# animation function.  This is called sequentially
		def animate(i):
			#     x = np.linspace(0, 2, 1000)
			#     y = np.sin(2 * np.pi * (x - 0.01 * i))
			for j in xrange(X.shape[1]):
				axarr[j+1,1].plot([0,0],[X[i,j], Y[i,j]], 'k', lw=3)
				# line[j].set_data(X[i,j], Y[i,j])
			# return line,

		anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=X.shape[0], interval=20, blit=True)
		anim.save(self.HOMEDIR + self.GROUP_NAMES + video_prefix + '_dynamics.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
		plt.show()