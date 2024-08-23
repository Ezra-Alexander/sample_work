import numpy as np
import sys
import math
from geom_helper import *

#The goal of this script is to take a .xyz file that represents a QD with a roughly zincblende crystal structure
#and add a specified atomic species to any number of vacancies in that QD

def main():
	#The orginal .xyz file that I am editing
	xyz = sys.argv[1]

	#The name of the .xyz file this will make
	name = sys.argv[2]

	#what element I want to add
	add = sys.argv[3]

	#the following inputs are any number of sets of the form: N El1 n11 n21 .. El2 ... where N is the total number of neighbors (ex 3 P 7 14 Cl 8)
	#this defines all the atoms in the .xyz that make up the neighbors of the vacancy
	input_targets=sys.argv[4:]

	coords,atoms=read_input_xyz(xyz)

	#convert input targets to a set of total indices
	first=True
	sequences=[]
	count=0
	for i,entry in enumerate(input_targets):
		if count==0:
			count=int(entry)
			if first:
				first=False
			else:
				sequences.append(sub_seq)
			sub_seq=[]
			
		elif not entry.isdigit():
			element=entry
		else:
			index=to_total_index(atoms,element,int(entry))
			sub_seq.append(index)
			count=count-1

			if count==0 and (i+1)==len(input_targets):
				sequences.append(sub_seq)

	for sequence in sequences:
		atoms, coords = vacancy_fill(coords,atoms,add,sequence)

	write_xyz(name, atoms, coords)

def vacancy_fill(coords,atoms,add,sequence,error_margin=1,flex=1.25):
	'''
	this problem is trivial with a sequence lenghth of 4 but harder with fewer
	The strategy is similar to the coordination sphere approach but we also test angles
	the first atom in the sequence is arbitrarily chosen as the start of the coordination sphere and determines the bond length
	error margin lowered recursively to ensure convergence
	'''

	covalent_radii={"In":1.42,"Ga":1.22,"P":1.07,"Cl":1.02,"F":0.57,"Zn":1.22,"S":1.05,"Se":1.20,"O":0.66,"Al":1.21,"H":0.31,"Br":1.2,"Si":1.11,"C":0.76,"N":0.71,"Li":1.28}
	bond_length=covalent_radii[add]+covalent_radii[atoms[sequence[0]]]
	sphere=get_coordination_sphere(coords[sequence[0]],bond_length,10000)
	dists=dist_all_points(coords)
	connect=smart_connectivity_finder(dists,atoms,flexibility=flex)

	good_points=[]
	errors=[]
	for point in sphere:
		point_failed=False
		error=0
		#print(np.linalg.norm(point-coords[sequence[0]]))
		for neighbor in sequence:
			for neighbor2 in sequence:
				if neighbor!=neighbor2:
					angle=get_angle_from_points(point,coords[neighbor],coords[neighbor2])
					if abs(angle-109.5)>(1*error_margin):
						point_failed=True
						break
					else:
						error=error+abs(angle-109.5)
			if point_failed:
				break
			for next_nearest in connect[neighbor]:
				angle=get_angle_from_points(coords[neighbor],coords[next_nearest],point)
				if abs(angle-109.5)>(10*error_margin):
						point_failed=True
						break
				else:
						error=error+abs(angle-109.5)
			if point_failed:
				break
		if not point_failed:

			inverse_sum=0 #add inverse distance info
			for j,atom in enumerate(atoms):
				inverse_dist=1/((np.linalg.norm(point-coords[j]))**2)
				inverse_sum=inverse_sum+inverse_dist
			error=error+inverse_sum

			good_points.append(point)
			errors.append(error)



	# for point in good_points:
	# 	print("H",point[0],point[1],point[2])

	if len(good_points)>0:
		errors=np.array(errors)
		new_atoms=np.copy(atoms)
		new_coords=np.copy(coords)
		new_atoms=np.append(new_atoms,add)
		chosen_point=good_points[np.argmin(errors)]
		new_coords=np.append(new_coords,[chosen_point],axis=0)
	else:
		new_error=(error_margin*1.5)
		new_atoms, new_coords = vacancy_fill(coords,atoms,add,sequence,error_margin=new_error)
		

	return new_atoms, new_coords


if __name__ == '__main__':
	main()