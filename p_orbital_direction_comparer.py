import numpy as np
import sys
import math
from geom_helper import *
import scipy
from qchem_helper import read_cube

def main():
	#Goal: provide a quantitative measure of how well the orientation of two p orbitals in two cube files match each other

	#Approach: 	computes a vector from - to + phase for each p orbital using the weighted average of each
	#			adjusts those vectors to be measured relative to the same internal coordinate system using relative bond lengths
	#			computes the dot product between the two vectors

	#intended for the case where the two center atoms have the same bond angles

	center_element1=sys.argv[1] #element of center atom in the first cube file
	center_index1=int(sys.argv[2]) #element-specific index of center atom (1-indexed) in the first cube file
	cube1=sys.argv[3] #the first cube file of interest
	center_element2=sys.argv[4] #element of center atom in the second cube file
	center_index2=sys.argv[5] #element-specific index of center atom (1-indexed) in the second cube file. Can enter "search" and it will find the angle match to the interpolation
	cube2=sys.argv[6] #the second cube file of interest

	dot=compare_orbital_direction(center_element1,center_index1,cube1,center_element2,center_index2,cube2)

	print()
	print(abs(dot))
	print()


def compare_orbital_direction(center_element1,center_index1,cube1,center_element2,center_index2,cube2):	
	'''
	Computes the similarity between two p orbitals in two cube files as the dot product after rotation to a common reference frame
	'''

	covalent_radii={"P":1.07,"In":1.42,"Ga":1.22,"Li":1.28,"Cl":1.02,"F":0.57} #manual dictionaries
	atomic_numbers={'15':"P",'49':"In",'31':"Ga",'3':"Li",'17':"Cl","9":"F"}
	grace=0 #only considers voxels within grace of the minimum and maximum bonded element coordinated
	#grace2=10 #using different graces for each doesn't seem to change much and may actually be less accurate

	#read in cube 1
	atoms1, coords1, grid_spacing1, origin1, mo_data1 = read_cube(cube1)
	#print(atoms1)
	total_index1=to_total_index(atoms1,center_element1,center_index1)
	dists1=dist_all_points(coords1)
	connectivity1=smart_connectivity_finder(dists1,atoms1)
	angles1=get_close_angles(coords1,dists1,connectivity1)

	# bonded_atoms1=[]
	# bonded_coords1=[]
	max_dist1=0
	for i,atom in enumerate(atoms1):
		if i in connectivity1[total_index1]:
			if dists1[total_index1][i]>max_dist1:
				max_dist1=dists1[total_index1][i]
	# 		bonded_atoms1.append(atom)
	# 		bonded_coords1.append(coords1[i])
	# max_coords1=np.max(bonded_coords1,axis=0)+grace1
	# min_coords1=np.min(bonded_coords1,axis=0)-grace1
	max_dist1=max_dist1+grace
	#print(max_dist1)
	#print(coords1[total_index1]-origin1)
	#print(total_index1)
	# #extract all relevant voxels from cube 1
	voxels_near_center_1=[]
	xyzs_near_center_1=[]
	for i,plane in enumerate(mo_data1):
		x_coord=((i+0.5)*grid_spacing1[0])+origin1
		#if x_coord[0]>=min_coords1[0] and x_coord[0]<=max_coords1[0]:
		if abs(x_coord[0]- coords1[total_index1][0])<=max_dist1:
			for j,line in enumerate(plane):
				y_coord=((j+0.5)*grid_spacing1[1])+origin1
				#if y_coord[1]>=min_coords1[1] and y_coord[1]<=max_coords1[1]:
				if abs(y_coord[1]- coords1[total_index1][1])<=max_dist1:				
					for k,point in enumerate(line):
						z_coord=((k+0.5)*grid_spacing1[2])+origin1
						#if z_coord[2]>=min_coords1[2] and z_coord[2]<=max_coords1[2]:
						distance_from_center=np.linalg.norm([x_coord[0],y_coord[1],z_coord[2]]-coords1[total_index1])
						#print(distance_from_center)
						if distance_from_center<=max_dist1:
							#print([x_coord[0],y_coord[1],z_coord[2]]-origin1)
							voxels_near_center_1.append(float(point))
							xyzs_near_center_1.append([x_coord[0],y_coord[1],z_coord[2]])

	#determine max and min voxels

	# max_phase=0
	# for i,element in enumerate(voxels_near_center_1):
	# 	if element[3]>max_phase:
	# 		max_phase=element[3]
	# 		xyzs=element[:3]
	# print(xyzs,max_phase)
	voxels_near_center_1=np.array(voxels_near_center_1)
	xyzs_near_center_1=np.array(xyzs_near_center_1)
	#print(voxels_near_center_1.shape)
	xyz_max1=np.average(xyzs_near_center_1[voxels_near_center_1>0],axis=0,weights=voxels_near_center_1[voxels_near_center_1>0])
	xyz_min1=np.average(xyzs_near_center_1[voxels_near_center_1<0],axis=0,weights=voxels_near_center_1[voxels_near_center_1<0])
	# voxels_near_center_1=[]
	# for i,plane in enumerate(mo_data1):
	# 	x_coord=(i*grid_spacing1[0])+origin1[0]
	# 	if x_coord[0]>=min_coords1[0] and x_coord[0]<=max_coords1[0]:
	# 		for j,line in enumerate(plane):
	# 			y_coord=(j*grid_spacing1[1])+origin1[1]
	# 			if y_coord[1]>=min_coords1[1] and y_coord[1]<=max_coords1[1]:
	# 				for k,point in enumerate(line):
	# 					z_coord=(k*grid_spacing1[2])+origin1[2]
	# 					if z_coord[2]>=min_coords1[2] and z_coord[2]<=max_coords1[2]:
	# 						voxels_near_center_1.append([i,j,k,float(point)])

	# #determine max and min voxels

	# # max_phase=0
	# # for i,element in enumerate(voxels_near_center_1):
	# # 	if element[3]>max_phase:
	# # 		max_phase=element[3]
	# # 		xyzs=element[:3]
	# # print(xyzs,max_phase)
	# voxels_near_center_1=np.array(voxels_near_center_1)
	# max_voxel1=voxels_near_center_1[np.argmax(voxels_near_center_1,axis=0)[3]]
	# xyz_max1=(max_voxel1[0]*grid_spacing1[0] + max_voxel1[1]*grid_spacing1[1] + max_voxel1[2]*grid_spacing1[2])
	# min_voxel1=voxels_near_center_1[np.argmin(voxels_near_center_1,axis=0)[3]]
	# # xyz_min1=(min_voxel1[0]*grid_spacing1[0] + min_voxel1[1]*grid_spacing1[1] + min_voxel1[2]*grid_spacing1[2])
	# print(xyz_max1/.529177249)
	# print(xyz_min1/.529177249)
	vector1=xyz_max1 - xyz_min1 #positive-negative
	unit_vector1=vector1/np.linalg.norm(vector1)
	#I need to define a new, locally consistent basis set
	inds_increasing_dists1=np.argsort(dists1[total_index1])
	basis_vector1_1=coords1[inds_increasing_dists1[1]]-coords1[total_index1] #shortest bond length
	basis_unit1_1=basis_vector1_1/np.linalg.norm(basis_vector1_1)
	next_bond_vector1=coords1[inds_increasing_dists1[2]]-coords1[total_index1]
	basis_vector1_2=basis_unit1_1-(next_bond_vector1/np.dot(next_bond_vector1,basis_unit1_1))#coplanar to basis vector 1 and the next shortest bond length, orthogonal to basis vector 1, positive orientation relative to basis vector 1
	basis_unit1_2=basis_vector1_2/np.linalg.norm(basis_vector1_2)
	basis_vector1_3=np.cross(basis_unit1_1,basis_vector1_2)
	basis_unit1_3=basis_vector1_3/np.linalg.norm(basis_vector1_3)
	transformed_orientation1=np.matmul(np.linalg.inv(np.column_stack((basis_unit1_1,basis_unit1_2,basis_unit1_3))),unit_vector1)


	#repeat everything for cube 2
	atoms2, coords2, grid_spacing2, origin2, mo_data2 = read_cube(cube2)
	dists2=dist_all_points(coords2)
	connectivity2=smart_connectivity_finder(dists2,atoms2)
	angles2=get_close_angles(coords2,dists2,connectivity2)
	if center_index2=="search":
		center_1_angles=[]
		for point2 in angles1[total_index1]:
			for angle in point2:
				if not np.isnan(angle):
					if round(angle,2) not in center_1_angles:
						center_1_angles.append(round(angle,2))
		for i,atom in enumerate(atoms2):
			if atom==center_element2:
				test=True
				for j in connectivity2[i]:
					for k in connectivity2[i]:
						if j>k:
							if round(angles2[i][j][k],2) not in center_1_angles:
								test=False
								break
					if test==False:
						break
				if test==True:
					total_index2=i
					break
	else:
		total_index2=to_total_index(atoms2,center_element2,int(center_index2))
	print(total_index2)
	max_dist2=0
	for i,atom in enumerate(atoms2):
		if i in connectivity2[total_index2]:
			if dists2[total_index2][i]>max_dist2:
				max_dist2=dists2[total_index2][i]
	# 		bonded_atoms1.append(atom)
	# 		bonded_coords1.append(coords1[i])
	# max_coords1=np.max(bonded_coords1,axis=0)+grace1
	# min_coords1=np.min(bonded_coords1,axis=0)-grace1
	max_dist2=max_dist2+grace
	#print(max_dist1)
	#print(coords1[total_index1]-origin1)
	#print(total_index1)
	# #extract all relevant voxels from cube 1
	voxels_near_center_2=[]
	xyzs_near_center_2=[]
	for i,plane in enumerate(mo_data2):
		x_coord=((i+0.5)*grid_spacing2[0])+origin2
		#if x_coord[0]>=min_coords1[0] and x_coord[0]<=max_coords1[0]:
		if abs(x_coord[0]- coords2[total_index2][0])<=max_dist2:
			for j,line in enumerate(plane):
				y_coord=((j+0.5)*grid_spacing2[1])+origin2
				#if y_coord[1]>=min_coords1[1] and y_coord[1]<=max_coords1[1]:
				if abs(y_coord[1]- coords2[total_index2][1])<=max_dist2:				
					for k,point in enumerate(line):
						z_coord=((k+0.5)*grid_spacing2[2])+origin2
						#if z_coord[2]>=min_coords1[2] and z_coord[2]<=max_coords1[2]:
						distance_from_center=np.linalg.norm([x_coord[0],y_coord[1],z_coord[2]]-coords2[total_index2])
						#print(distance_from_center)
						if distance_from_center<=max_dist2:
							#print([x_coord[0],y_coord[1],z_coord[2]]-origin1)
							voxels_near_center_2.append(float(point))
							xyzs_near_center_2.append([x_coord[0],y_coord[1],z_coord[2]])


	# bonded_atoms2=[]
	# bonded_coords2=[]
	# for i,atom in enumerate(atoms2):
	# 	if i in connectivity2[total_index2]:
	# 		bonded_atoms2.append(atom)
	# 		bonded_coords2.append(coords2[i])
	# max_coords2=np.max(bonded_coords2,axis=0)+grace2
	# min_coords2=np.min(bonded_coords2,axis=0)-grace2

	# #extract all relevant voxels from cube 2
	# voxels_near_center_2=[]
	# xyzs_near_center_2=[]
	# for i,plane in enumerate(mo_data2):
	# 	x_coord=((i+0.5)*grid_spacing2[0])+origin2[0]
	# 	if x_coord[0]>=min_coords2[0] and x_coord[0]<=max_coords2[0]:
	# 		for j,line in enumerate(plane):
	# 			y_coord=((j+0.5)*grid_spacing2[1])+origin2[1]
	# 			if y_coord[1]>=min_coords2[1] and y_coord[1]<=max_coords2[1]:
	# 				for k,point in enumerate(line):
	# 					z_coord=((k+0.5)*grid_spacing2[2])+origin2[2]
	# 					if z_coord[2]>=min_coords2[2] and z_coord[2]<=max_coords2[2]:
	# 						#distance_from_center=np.linalg.norm([x_coord,y_coord,z_coord]-coords2[total_index2])
	# 						voxels_near_center_2.append(float(point))
	# 						xyzs_near_center_2.append([x_coord[0],y_coord[1],z_coord[2]])
	#determine max and min voxels

	# max_phase=0
	# for i,element in enumerate(voxels_near_center_1):
	# 	if element[3]>max_phase:
	# 		max_phase=element[3]
	# 		xyzs=element[:3]
	# print(xyzs,max_phase)
	voxels_near_center_2=np.array(voxels_near_center_2)
	xyzs_near_center_2=np.array(xyzs_near_center_2)
	xyz_max2=np.average(xyzs_near_center_2[voxels_near_center_2>0],axis=0,weights=voxels_near_center_2[voxels_near_center_2>0])
	xyz_min2=np.average(xyzs_near_center_2[voxels_near_center_2<0],axis=0,weights=voxels_near_center_2[voxels_near_center_2<0])
	# max_voxel2=voxels_near_center_2[np.argmax(voxels_near_center_2,axis=0)[3]]
	# xyz_max2=(max_voxel2[0]*grid_spacing2[0] + max_voxel2[1]*grid_spacing2[1] + max_voxel2[2]*grid_spacing2[2])+origin2
	# min_voxel2=voxels_near_center_2[np.argmin(voxels_near_center_2,axis=0)[3]]
	# xyz_min2=(min_voxel2[0]*grid_spacing2[0] + min_voxel2[1]*grid_spacing2[1] + min_voxel2[2]*grid_spacing2[2])+origin2
	# print(xyz_max2/.529177249)
	# print(xyz_min2/.529177249)
	vector2=xyz_max2 - xyz_min2 #positive-negative
	unit_vector2=vector2/np.linalg.norm(vector2)
	#I need to define a new, locally consistent basis set
	inds_increasing_dists2=np.argsort(dists2[total_index2])
	basis_vector2_1=coords2[inds_increasing_dists2[1]]-coords2[total_index2] #shortest bond length
	basis_unit2_1=basis_vector2_1/np.linalg.norm(basis_vector2_1)
	next_bond_vector2=coords2[inds_increasing_dists2[2]]-coords2[total_index2]
	basis_vector2_2=basis_unit2_1-(next_bond_vector2/np.dot(next_bond_vector2,basis_unit2_1))#coplanar to basis vector 1 and the next shortest bond length, orthogonal to basis vector 1, positive orientation relative to basis vector 1
	basis_unit2_2=basis_vector2_2/np.linalg.norm(basis_vector2_2)
	basis_vector2_3=np.cross(basis_unit2_1,basis_vector2_2)
	basis_unit2_3=basis_vector2_3/np.linalg.norm(basis_vector2_3)
	transformed_orientation2=np.matmul(np.linalg.inv(np.column_stack((basis_unit2_1,basis_unit2_2,basis_unit2_3))),unit_vector2)


	#Compute the desired dot product
	dot=np.dot(transformed_orientation1,transformed_orientation2)

	if center_index2=="search":
		return abs(dot), to_atom_specific_index(atoms2,total_index2)
	else:
		return abs(dot)

if __name__ == "__main__":
	main()