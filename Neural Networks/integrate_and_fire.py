#!/usr/bin/env python
#simulate_Hopfield.py
""" Replicates Hopfield and Hertz simultions using 400 leaky Integrate and 
	fire neurons arranged on 20*20 planar grid. Each neuron is locally 
	connected to its four closest neighbors (n,s,e,w) with a fixed positive 
	connection weight of 0.24. The membrane threhold potential after which a  
	neuron fires is fixed to 1.
	Attributes:
		WEIGHT: Fixed positive connection weight, fixed to 0.24.
		RAN_SEED: A random seed number used to make results reproducible.
		WIDTH: Grid dimensions, fixed to 20.
		I_EXT: External input assumed to come outside of the grid.
"""

import numpy as np
import matplotlib.pyplot as ply

__author__ = "Venkata Sarika Kondra"
__version__ = "1.0.1"
__email__ = "c00219805@louisiana.edu"


RAN_SEED = 314159 #256214 #
N_OF_DTs =   2500
DT       =   .001
DTover6  = DT/6.0
WIDTH    =     20
I_EXT    =    1.1
WEIGHT   =   0.24

class NeuronFire():

	def __init__(self):
		self.grid_u   = np.zeros((WIDTH,WIDTH)) # preallocate arrays for efficiency
		self.fired    = np.zeros((WIDTH,WIDTH))
		self.u        = np.zeros((WIDTH,WIDTH))
		self.k1       = np.zeros((WIDTH,WIDTH))
		self.k2       = np.zeros((WIDTH,WIDTH))
		self.k3       = np.zeros((WIDTH,WIDTH))
		self.k4       = np.zeros((WIDTH,WIDTH))
		self.spike_data = np.zeros((WIDTH, WIDTH, N_OF_DTs))

	def init_grid(self):
		np.random.seed(RAN_SEED);     # set rand generator to make results reproducible
		self.grid_u = np.random.rand(WIDTH,WIDTH)
		return self.grid_u
	   

	def clear_fired_grid(self):
		self.fired = np.zeros((WIDTH,WIDTH))
	   
	def step1_grid(self):
		self.u  = self.grid_u
		self.k1 = -self.u + I_EXT
		self.k2 = -(self.u + 0.5*self.k1*DT) + I_EXT
		self.k3 = -(self.u + 0.5*self.k2*DT) + I_EXT
		self.k4 = -(self.u +     self.k3*DT) + I_EXT
		self.grid_u = self.grid_u + DTover6*(self.k1 + 2*self.k2 + 2*self.k3 +self.k4)

	def check_fired_grid(self,step):
	#      self.fired = grid_u >= 1.0
	#      self.grid_u = grid_u - fired
		for i in range(0,WIDTH):
			for j in range(0,WIDTH):
				self.check_fired_neuron(j,i)
		self.spike_data[:,:,step] = self.fired #Verify this step Sarika
	   	   
	def check_fired_neuron(self,j,i):
		if self.grid_u[j,i] > 1:
			self.grid_u[j,i] = self.grid_u[j,i] - 1
			self.fired[j,i] = 1
			self.propagate_to_neighbors(j,i)
			self.check_fired_neighbors(j,i)

	def check_fired_neighbors(self,j,i):
		print j, i
		if j > 1:
	           self.check_fired_neuron(j-1,i)
		else:
	           self.check_fired_neuron(WIDTH-1,i)
	       
		if j < WIDTH-1:
	           self.check_fired_neuron(j+1,i)
		else:
	           self.check_fired_neuron(1,i)
	       
		if i > 1:
	           self.check_fired_neuron(j,i-1)
		else:
	           self.check_fired_neuron(j,WIDTH-1)
	       
		if i < WIDTH-1:
	           self.check_fired_neuron(j,i+1)
		else:
	           self.check_fired_neuron(j,1)

	def propagate_to_neighbors(self,j,i):
	       if j > 1:
	           self.grid_u[j-1,i]   = self.grid_u[j-1,i] + WEIGHT
	       else:
	           self.grid_u[WIDTH-1,i] = self.grid_u[WIDTH-1,i] + WEIGHT
	       
	       if j < WIDTH-1:
	           self.grid_u[j+1,i] = self.grid_u[j+1,i] + WEIGHT
	       else:
	           self.grid_u[1,i]   = self.grid_u[1,i]  + WEIGHT
	       
	       if i > 1:
	           self.grid_u[j,i-1] = self.grid_u[j,i-1] + WEIGHT
	       else:
	           self.grid_u[j,WIDTH-1] = self.grid_u[j,WIDTH-1] + WEIGHT
	       
	       if i < WIDTH-1:
	           self.grid_u[j,i+1] = self.grid_u[j,i+1] + WEIGHT
	       else:
	           self.grid_u[j,1]   = self.grid_u[j,1] + WEIGHT

if __name__ == '__main__':
   #set(0,'RecursionLimit',1000) 
   neuron = NeuronFire()
   grid = neuron.init_grid()
   #figure(gcf)
   x =[]
   y =[]
   dt_noOfNeurons= {}
   [X, Y] = np.where(neuron.fired == 0)
   #ply.ion() # enable interactivity, can be default
   ply.plot(X,Y,'bs',markersize=17)
   ply.axis([0, WIDTH, 0, WIDTH])
   for step in range(0,N_OF_DTs):
       print step
       neuron.clear_fired_grid();
       [X, Y] = np.where(neuron.fired == 0)
       ply.plot(X,Y,'bs',markersize=17)
       neuron.step1_grid();
       neuron.check_fired_grid(step);
       [X, Y] = np.where(neuron.spike_data[:,:,step] !=0);
       #for i in X: x.append(i) 
       #for j in Y: y.append(j)
       dt_noOfNeurons[step] = len(X)
       print X,Y
       ply.plot(X,Y,'rs',markersize=17)
       ply.title("At Step "+ str(step))
       ply.axis([0, WIDTH, 0, WIDTH])
       if(len(X) > 6):
       		ply.savefig('D:\\Neural Networks\\step_'+str(step)+'.png')
       #ply.pause(1)
       #ply.show()
   ply.figure()
   ply.stem(np.squeeze(sum(sum(neuron.spike_data))))
   ply.xlabel('time step number')
   ply.ylabel('Number of neurons fired')
   ply.title("With Delta_t = " +str(DT)+", n = " +str(N_OF_DTs)+", I_EXT = "+str(I_EXT))
   ply.savefig('D:\\Neural Networks\\synchornization.png')
   print np.squeeze(sum(sum(neuron.spike_data)))
   print dt_noOfNeurons
   ply.pause(10)
   ply.show()
   