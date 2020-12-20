"""
Created on Fri Oct 23 20:36:38 2020

Author: Ollie Breach and James Walkling

Description: Implementation of Ising model with both Wolff and Metropolis algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


grid_size=16
T = 2
J=1   #J>0 for ferromagnetism
h=0
mu=1
k=1

#simulation is build around the Board object, representing a square 2d grid of spins with periodic boundary conditions

class Board:

    def __init__(self, grid_size):
        self.size=grid_size
        self.grid=self.init_board(grid_size)
        
    def init_board(self, grid_size):
        zeros_and_ones = np.random.randint(0,2,(grid_size,grid_size)) #create grid of zeros and ones
        initial_grid = np.where(zeros_and_ones==0, -1,zeros_and_ones) #replace zeros with ones - these are our initial spin states
        return initial_grid
    
    def display_board(self):
        plt.axis("off")
        plt.imshow(self.grid, cmap="Greys")
        
    def reset_board(self):
        self.grid=self.init_board(self.size)
        
    def custom_board(self, custom_grid):
        self.grid=custom_grid
        self.size=np.shape(custom_grid)[0]
        
    #energy calculations    
    
    def bond_energy(self, position, J):
        i,j=position
        u=-J*self.get_spin([i,j])*(self.get_spin([(i+1)%self.size,j])+self.get_spin([(i-1)%self.size,j])+self.get_spin([i,(j+1)%self.size])+self.get_spin([i,(j-1)%self.size]))
        return u
    
    def field_energy(self, position, h, mu):
        return -self.get_spin(position)*mu*h

    def energy(self, position, J, h, mu):
        return self.bond_energy(position, J)+self.field_energy(position, h, mu)

    def energy_difference(self, position, J, h, mu):  #energy difference if we were to flip a given spin
        return -2*self.energy(position, J, h, mu)
    
    def boltzmann(self, de, T):
        return np.exp(-de/(k*T))
    
    
    #metropolis update
    
    def update_metropolis(self, J, h, mu, T):
        position = np.random.randint(0,self.size, size=2)
        de=self.energy_difference(position, J, h, mu)
        if de<=0:
            self.flip_spin(position)
        elif np.random.random()<self.boltzmann(de, T):
            self.flip_spin(position)
     

    #wolff update
        
    
    def update_wolff(self, J, h, mu, T):

        init_pos=(np.random.randint(0,self.size), np.random.randint(0,self.size))
        c=[init_pos]  #cluster
        f_old=[init_pos]  #frontier

        while len(f_old) != 0:  #updates the frontier - i.e. expands it outwards with the checking procedure

            f_new=[]

            for pos in f_old:  #for each spin in frontier, adds neighbouring spins with a probability if they are the same spin

                this_spin=self.get_spin(pos)
                for neighb in self.neighbours(pos):

                    if self.get_spin(neighb)==this_spin:
                        if neighb not in c:
                            if np.random.random() < 1-self.boltzmann(2*J, T):
                                f_new.append(neighb)  #adding neighbour to new frontier
                                c.append(neighb)   #adding neighbour to the cluster of spins that will be flipped
            f_old = f_new   #updating the frontier
        
        #checking whether to flip cluster based on field strength:
        field_energy_difference=self.de_cluster_field(c, J, h, mu, T)
        if field_energy_difference<=0:
            self.flip_cluster(c)
        elif np.random.random()<self.boltzmann(field_energy_difference, T):
            self.flip_cluster(c)

            
    def neighbours(self,pos):   #returns all the adjacent neighbours of a given spin
        i,j=pos
        return [((i-1)%self.size,j),(i,(j+1)%self.size),((i+1)%self.size,j), (i, (j-1)%self.size)]    

    
    def check_cluster(self, cluster, neighb): #check to see whether a position is already in the cluster
        for pair in cluster:
            if np.array_equal(pair,neighb):
                return True
        else:
            return False
        

    def de_cluster_field(self, cluster, J, h, mu, T): #change in energy due to magnetic field only if cluster is flipped
        field_energy = 0
        for pos in cluster:
            field_energy+= - h*mu*self.get_spin(pos)
        return -2*field_energy
    
    
    #tools
    
    def get_spin(self, pos):
        i,j=pos
        return self.grid[i][j]
    
    def flip_spin(self, pos):
        i,j=pos
        self.grid[i][j]=-self.grid[i][j]

    def flip_cluster(self, cluster):
        for pos in cluster:
            self.flip_spin(pos)
            
            
    #board properties        
            
    def board_energy(self, J, h, mu):
        energy=0
        for i in range(self.size):
            for j in range(self.size):
                energy+=(self.bond_energy([i,j], J)/2 + self.field_energy([i,j], h, mu)) #bond energy divided by two to avoid double counting pairs
        return energy
    
    def board_magnetisation(self):
        return np.sum(self.grid)
    
    


"""
Creating animations
"""

def metropolis_animation(board, update_skip, num_frames, J, h, mu, T): #update_skip is number of board updates between frames
  
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ax2.set_xlim([0,5])
    #ax2.autoscale_view()
    ax2.set_ylim([-board.size**2,board.size**2])
    ax2.set_xticks([])
    ax2.set_ylabel('magnetisation')
    
    #list of magnetisations for each update
    mag=[]
    
    #list of energies for each update
    energies=[]
    
    im = ax1.imshow(board.grid, cmap="Greys")
    im2, = ax2.plot([], [], color=(0,0,1))
    
    ax1.set_title('Metropolis')
    ax2.set_title('Magnetisation')
    
    def func(n,mag):    #generates frames of animation
        
        for j in range(update_skip):
            board.update_metropolis(J,h,mu,T)
        im.set_array(board.grid)
       
        mag.append(board.board_magnetisation())
       
        im2.set_xdata(np.arange(n))
        im2.set_ydata(np.array(mag[len(mag)-n:])) 
        lim = ax2.set_xlim(0, n+1)
        
        if n==num_frames-1:  #for when animation repeats
            im2.cla()
            board.reset_board()
        
        return im, im2
    
    animation.FuncAnimation(fig, func,fargs=(mag,), frames=num_frames, interval=30, blit=True)


def wolff_animation(board, update_skip, num_frames, J, h, mu, T):
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)


    ax2.set_xlim([0,5])
    #ax2.autoscale_view()
    ax2.set_ylim([-board.size**2,board.size**2])
    ax2.set_xticks([])
    ax2.set_ylabel('magnetisation')
    
    #set up list of images for animation
    mag=[board.board_magnetisation()]
    
    im = ax1.imshow(board.grid, cmap="Greys")
    im2, = ax2.plot([], [], color=(0,0,1))
    
    ax1.set_title('Wolff')
    ax2.set_title('Magnetisation')
    
    def func(n,mag):   #generates frames of animation
        for j in range(update_skip):
            board.update_wolff(J,h,mu,T)
        im.set_array(board.grid)
       
        mag.append(board.board_magnetisation())
       
        im2.set_xdata(np.arange(n))
        im2.set_ydata(np.array(mag[len(mag)-n:])) 
        lim = ax2.set_xlim(0, n+1)
        if n==num_frames-1:
            im2.cla()
            board.reset_board()
            mag=[]
        
        return im, im2
    
    animation.FuncAnimation(fig, func, fargs=(mag,),frames=num_frames, interval=30, blit=True)



#Examples


#Make 20*20 board:
board = Board(50)

#display board
board.display_board()

#single updates
board.update_metropolis(J, h, mu, T)
board.update_wolff(J, h, mu, T) 

#reset board
board.reset_board()

#Make animation
update_skip = 10                                                        
num_frames = 10000
metropolis_animation(board, update_skip, num_frames, J, h, mu, T)
#board.reset_board()
#wolff_animation(board, 1, num_frames, J, h, mu, T)






            
