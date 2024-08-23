import numpy as np
import sys
import math
from geom_helper import *
from qchem_helper import read_xyz, write_xyz

#a replacement of the geom_adder script
#this adds one of a specified atomic type to any number of atoms in your QD
#it tries to do this in the best way possible, including addition to 4c atoms


def mass_add(coords,atoms,add,add_base,input_targets,add_type='sphere'):
	'''

	#The orginal .xyz file that I am editing
	xyz = sys.argv[1]

	#The name of the .xyz file this will make
	name = sys.argv[2]

	#tetrahedral or spherical algorithm
	add_strat=sys.argv[3]

	#what atom I want to add
	add = sys.argv[4]

	#the species you are adding to
	add_base=sys.argv[5]

	#the following inputs are any number of atom-specific
	input_targets=[int(x) for x in sys.argv[6:]]

	#if add_type is set to sphere, if adds to the point in the coordination sphere with minimal inverse distance to all other atoms. 
	#if add_type is set to tetrahedral, if adding to 3c it tries to get as 3c as possible
	'''

	if type(input_targets)!=list:
		input_targets=[int(input_targets)]

	bond_max=2.86 #an arbitrary parameter for determining the upper distance between two atoms for them to be considered bonded
	bond_lengths={"In":{"F":2.02,"Cl":2.37,"P":2.6,"N":2.13},"Ga":{"F":1.79,"Cl":2.19,"P":2.48,"N":1.93},"P":{"O":1.53,"Ga":2.5,"In":2.51},"N":{"H":1.02,"C":1.47},"C":{"H":1.07},"Cl":{"In":2.37}} #hard coded (terminal) bond lengths. not everything will be supported. Don't have to be perfect, these are just pre-opt estimates

	temp_coords=np.copy(coords)
	temp_atoms=np.copy(atoms)

	#convert atom_specific to overall
	count=0
	targets=[]
	for i,atom in enumerate(atoms):
		if atom==add_base:
			count=count+1
			if count in input_targets:
				targets.append(i)


	#do the thing
	for i,target in enumerate(targets):
		bond_length=bond_lengths[temp_atoms[target]][add]
		dists=dist_all_points(temp_coords)
		connectivity=smart_connectivity_finder(dists,temp_atoms)
		#print(connectivity[target])
		#compute center of dot (no mass)
		center=np.array([0,0,0])
		for i,atom in enumerate(temp_atoms):
			center=center+temp_coords[i]
		center=center/len(temp_atoms)

		if len(connectivity[target])==1:
			# anchor=connectivity[target][0]
			# anchor_bonds=connectivity[anchor]
			# vector=temp_coords[anchor]-temp_coords[anchor_bonds[0]]
			# unit=vector/np.linalg.norm(vector)

			# test_vector=temp_coords[target]-center
			# dot=np.dot(unit,test_vector)
			# i=0
			# while dot<0:
			# 	i=i+1
			# 	vector=temp_coords[anchor]-temp_coords[anchor_bonds[i]]
			# 	unit=vector/np.linalg.norm(vector)
			# 	dot=np.dot(unit,test_vector)

			# added_coords=temp_coords[target]+(unit*bond_length)
			# temp_coords=np.append(temp_coords,[added_coords],axis=0)
			# temp_atoms=np.append(temp_atoms,add)

			sphere_points=get_coordination_sphere(temp_coords[target],bond_length,1000)

			inverse_dists=[]
			for i,point in enumerate(sphere_points):
				inverse_sum=0
				for j,atom in enumerate(atoms):
					inverse_dist=1/((np.linalg.norm(point-coords[j]))**2)
					inverse_sum=inverse_sum+inverse_dist
				inverse_dists.append(inverse_sum)
			inverse_dists=np.array(inverse_dists)

			min_point=np.argmin(inverse_dists)

			added_coords=sphere_points[min_point]
			
			temp_coords=np.append(temp_coords,[added_coords],axis=0)
			temp_atoms=np.append(temp_atoms,add)

		elif len(connectivity[target])==2:
			# v1=temp_coords[target]-temp_coords[connectivity[target][0]]
			# v2=temp_coords[target]-temp_coords[connectivity[target][1]]

			# parallel=(v1+v2)/np.linalg.norm(v1+v2)
			# perp=np.cross(v1,v2)
			# perp_unit=perp/np.linalg.norm(perp)

			# adjacent=bond_length*math.cos(math.radians(109.5/2))
			# opposite=bond_length*math.sin(math.radians(109.5/2))

			# added_coords=temp_coords[target]+(parallel*adjacent)+(perp_unit*opposite)

			# if np.linalg.norm(added_coords-center)<np.linalg.norm(temp_coords[target]-center):
			# 	added_coords=temp_coords[target]+(parallel*adjacent)-(perp_unit*opposite)

			# temp_coords=np.append(temp_coords,[added_coords],axis=0)
			# temp_atoms=np.append(temp_atoms,add)

			sphere_points=get_coordination_sphere(temp_coords[target],bond_length,1000)

			inverse_dists=[]
			for i,point in enumerate(sphere_points):
				inverse_sum=0
				for j,atom in enumerate(atoms):
					inverse_dist=1/((np.linalg.norm(point-coords[j]))**2)
					inverse_sum=inverse_sum+inverse_dist
				inverse_dists.append(inverse_sum)
			inverse_dists=np.array(inverse_dists)

			min_point=np.argmin(inverse_dists)

			added_coords=sphere_points[min_point]
			
			temp_coords=np.append(temp_coords,[added_coords],axis=0)
			temp_atoms=np.append(temp_atoms,add)

		elif len(connectivity[target])==3:
				if add_type.lower()=="tetrahedral":
					temp_atoms,temp_coords=geom_adder(temp_atoms,temp_coords,target,add)
				elif add_type.lower()=="sphere":

					sphere_points=get_coordination_sphere(temp_coords[target],bond_length,1000)

					inverse_dists=[]
					for i,point in enumerate(sphere_points):
						inverse_sum=0
						for j,atom in enumerate(atoms):
							inverse_dist=1/((np.linalg.norm(point-coords[j]))**2)
							inverse_sum=inverse_sum+inverse_dist
						inverse_dists.append(inverse_sum)
					inverse_dists=np.array(inverse_dists)

					min_point=np.argmin(inverse_dists)

					added_coords=sphere_points[min_point]
					
					temp_coords=np.append(temp_coords,[added_coords],axis=0)
					temp_atoms=np.append(temp_atoms,add)
				else:
					raise Exception("Add type should be either sphere or tetrahedral")



		elif len(connectivity[target])==4:

				# This implementation splits the biggest angle (or pseudo 4-atom angle) weighted by bond lengths, and then attempts to prevent internal addition by checking the dot product of the target vector against the COM to base vector
				# It works well for some systems but still struggles with intenal additions

				# max_angle=0 #arbitrary large number
				# angles=[]
				# for j,bond1 in enumerate(connectivity[target]):
				# 	for k,bond2 in enumerate(connectivity[target]):
				# 		if j>k:
							
				# 			v1=temp_coords[bond1]-temp_coords[target]
				# 			v2=temp_coords[bond2]-temp_coords[target]

				# 			num=np.dot(v1,v2)
				# 			denom=dists[bond1][target]*dists[bond2][target]

				# 			angle = math.acos(num/denom)
				# 			weight=(dists[bond1][target]+dists[bond2][target])/2
				# 			angles.append([angle*weight,bond1,bond2])
				# 			# if angle>max_angle:
				# 			# 	max_angle=angle
				# 			# 	max_index_1=bond1
				# 			# 	max_index_2=bond2

				# #let's also consider dummy 4-atom angles
				# for j,bond1 in enumerate(connectivity[target]):
				# 	for k,bond2 in enumerate(connectivity[target]):
				# 		for l,bond3 in enumerate(connectivity[target]):
				# 			if j>k and k>l:

				# 				dummy_coords=(temp_coords[bond1]+temp_coords[bond2])/2
				# 				v1=temp_coords[bond3]-temp_coords[target]
				# 				v2=dummy_coords-temp_coords[target]

				# 				num=np.dot(v1,v2)
				# 				denom=np.linalg.norm(v1)*np.linalg.norm(v2)

				# 				angle = math.acos(num/denom)
				# 				weight=(np.linalg.norm(v1)+dists[bond1][target]+dists[bond2][target])/3
				# 				angles.append([angle*weight,bond3,[bond1,bond2]])


				# angles.sort(key = lambda x: x[0], reverse=True)

				# success=False
				# i=0
				# while not success:

				# 	v1=temp_coords[angles[i][1]]-temp_coords[target]
				# 	if isinstance(angles[i][2],int):
				# 		v2=temp_coords[angles[i][2]]-temp_coords[target]
				# 	else:
				# 		dummy_coords=(temp_coords[angles[i][2][0]]+temp_coords[angles[i][2][1]])/2
				# 		v2=dummy_coords-temp_coords[target]

				# 	#I need to implement some sort of test that makes sure the additions to 4c point "outward"
				# 	vector=v1+v2
				# 	unit=vector/np.linalg.norm(vector)

				# 	test_vector=temp_coords[target]-center
				# 	dot=np.dot(unit,test_vector)

				# 	if dot>0:
				# 		success=True

				# 	else:
				# 		i=i+1

				#added_coords=temp_coords[target]+(unit*bond_length)

				# This new implementation aims to instead minimize the inverse distance between each point on the coordination sphere and every other atom

				sphere_points=get_coordination_sphere(temp_coords[target],bond_length,1000)

				inverse_dists=[]
				for i,point in enumerate(sphere_points):
					inverse_sum=0
					for j,atom in enumerate(atoms):
						inverse_dist=1/(np.linalg.norm(point-coords[j]))
						inverse_sum=inverse_sum+inverse_dist
					inverse_dists.append(inverse_sum)
				inverse_dists=np.array(inverse_dists)

				min_point=np.argmin(inverse_dists)


				added_coords=sphere_points[min_point]
				
				temp_coords=np.append(temp_coords,[added_coords],axis=0)
				temp_atoms=np.append(temp_atoms,add)



		else:
				print("you have a species here that is either 0 coordinate or more than 4 coordinate, neither of which is implemented yet")



	return  temp_atoms, temp_coords

if __name__ == '__main__':
	#The orginal .xyz file that I am editing
	xyz = sys.argv[1]

	#The name of the .xyz file this will make
	name = sys.argv[2]

	#tetrahedral or spherical algorithm
	add_strat=sys.argv[3]

	#what atom I want to add
	add = sys.argv[4]

	#the species you are adding to
	add_base=sys.argv[5]

	#the following inputs are any number of atom-specific
	input_targets=[int(x) for x in sys.argv[6:]]

	coords,atoms=read_xyz(xyz)

	atoms, coords = mass_add(coords,atoms,add,add_base,input_targets,add_type=add_strat)

	write_xyz(name, atoms, coords)