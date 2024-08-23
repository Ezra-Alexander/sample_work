import numpy as np
import sys
import math
from geom_helper import *
import scipy
import time
#start=time.time()

def main():
	#this script aims to find a way to produce some sort of metric of how much a given MO orbital overlaps with certain atoms
	#takes .cube files as input
	#measures overlap with all atoms bound to a given center (excluding that center)
	#this implementation doesn't read all the cube info, just the relevant parts

	center_element=sys.argv[1] #element of center atom
	center_index=int(sys.argv[2]) #element-specific index of center atom (1-indexed)
	cube=sys.argv[3] #the cube file of interest

	score,score2=quantify_orbital_overlap(center_element,center_index,cube)

	#print("Total orbital overlap with other atoms is:",round(score/score2,3))
	print("Total orbital overlap with other atoms is:",round(score/score2,3))
	print("Entire Cube is normalized to:",score2)
	#end=time.time()
	#print(end-start)

def quantify_orbital_overlap(center_element,center_index,cube):
	'''
	Takes a cube file and calculates how much the cube overlaps with all atoms bound to a specified atom
	Inputs:
		center_element: str. the element of the center atom
		center_index: int. the atom-specific index of the center atom (1-indexed)
		cube: str. the cube file of interest
	Outputs:
		score: the described calculated overlap
		score2: the total value of the cube file summed over all voxels. In case you want to normalize (which you probably should)
	'''

	covalent_radii={"P":1.07,"In":1.42,"Ga":1.22,"Li":1.28,"Cl":1.02,"F":0.57} #manual dictionaries
	atomic_numbers={'15':"P",'49':"In",'31':"Ga",'3':"Li",'17':"Cl","9":"F"}
	grace=1.5 #only considers voxels within grace of the minimum and maximum bonded element coordinated

	#read everything from the cube file
	n_grid=[]
	grid_spacing=[]
	atoms=[]
	with open(cube,"r") as cb:
		for i,line in enumerate(cb):
			if i==2:
				n_atoms=int(line.strip().split()[0])
				origin=[float(line.strip().split()[1]),float(line.strip().split()[2]),float(line.strip().split()[3])]
			elif i>2 and i<6:
				line_list=line.strip().split()
				n_grid.append(int(line_list[0]))
				grid_spacing.append([float(line_list[1]),float(line_list[2]),float(line_list[3])])
				if i==6:
					break

	coords=np.loadtxt(cube,max_rows=n_atoms,skiprows=6,usecols=(2,3,4))
	atoms_numbers=np.loadtxt(cube,max_rows=n_atoms,skiprows=6,usecols=(0,))
	for number in atoms_numbers:
		atoms.append(atomic_numbers[str(int(number))])

	n_grid=tuple(n_grid)
	x=0
	y=0
	z=0
	score=0
	score2=0

	#COORDS ARE IN UNITS Bohr
	coords=np.array(coords)*.529177249
	grid_spacing=np.array(grid_spacing)*.529177249
	origin = np.array(origin)*.529177249

	element_count=0
	for i,atom in enumerate(atoms):
		if atom==center_element:
			element_count=element_count+1
			if element_count==center_index:
				total_index=i
				break

	dists=dist_all_points(coords)
	connectivity=smart_connectivity_finder(dists,atoms)
	bonded_atoms=[]
	bonded_coords=[]
	for i,atom in enumerate(atoms):
		if i in connectivity[total_index]:
			bonded_atoms.append(atom)
			bonded_coords.append(coords[i])

	max_coords=np.max(bonded_coords,axis=0)
	min_coords=np.min(bonded_coords,axis=0)


	mo_data=np.loadtxt(cube,skiprows=(6+n_atoms))
	for i,row in enumerate(mo_data):
		xy_coords=(x*grid_spacing[0] + y*grid_spacing[1])+origin
		if xy_coords[0]>(min_coords[0]-grace) and xy_coords[0]<(max_coords[0]+grace) and xy_coords[1]>(min_coords[1]-grace) and xy_coords[1]<(max_coords[1]+grace): #skipping Z because each row shares X and Y but differs in Z
			for j,value in enumerate(row):
				#score2=score2+(value**2)
				# print(z)
				# input()
				coordinates=(x*grid_spacing[0] + y*grid_spacing[1] + z*grid_spacing[2])+origin 

				if np.all(coordinates > (min_coords-grace)) and np.all(coordinates<(max_coords+grace)) :
					for k,coord in enumerate(bonded_coords):
						if np.linalg.norm(coordinates-coord)<covalent_radii[bonded_atoms[k]]:
							score=score+(float(value)**2)
							break
				#orbital[x][y][z]=float(value)
				if z==(n_grid[2]-1):
					z=0
				else:
					z=z+1
		else:
			if z==(n_grid[2]-6+(n_grid[2]%6)):
				z=0
			else:
				z=z+6
		if z==0:
			if y==(n_grid[1]-1):
					y=0
			else:
					y=y+1

			if y==0:
				if x==(n_grid[0]-1):
						x=0
				else:
						x=x+1
					




	# # coords_to_check=[-1,   -1,   0.00000000]

	# # shift=coords_to_check-origin

	# # n_x=math.floor(shift[0]/grid_spacing[0][0])
	# # n_y=math.floor(shift[1]/grid_spacing[1][1])
	# # n_z=math.floor(shift[2]/grid_spacing[2][2])

	# #print(coords)

	score2=np.sum(np.square(mo_data))
	# # for x,plane in enumerate(orbital):
	# # 	for y,line in enumerate(plane):
	# #  		for z,voxel in enumerate(line):
	# #  			score2=score2+voxel

	return score, score2

if __name__ == "__main__":
	main()