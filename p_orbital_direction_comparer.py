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

	if center_index2=="search":
		dot,_=compare_orbital_direction(center_element1,center_index1,cube1,center_element2,center_index2,cube2)
	else:
		dot=compare_orbital_direction(center_element1,center_index1,cube1,center_element2,center_index2,cube2)

	print()
	print(abs(dot))
	print()


def test_p(total_index,atoms, coords, grid_spacing, origin, mo_data,dist_thresh=6,value_thresh=0.00001):
	'''
	Tests to see if the provided cube is in fact centered on the center atom	
	'''
	#weighted sum of each voxel by inverse squared distance and makes sure the center wins

	scores=np.zeros(len(atoms))

	x_base = np.linspace(0.5 * grid_spacing[0], (len(mo_data)-0.5) * grid_spacing[0], len(mo_data)) + origin
	y_base = np.linspace(0.5 * grid_spacing[1], (len(mo_data[0])-0.5) * grid_spacing[1], len(mo_data[0])) + origin
	z_base = np.linspace(0.5 * grid_spacing[2], (len(mo_data[0][0])-0.5) * grid_spacing[2], len(mo_data[0][0])) + origin

	for i,plane in enumerate(mo_data):
		if np.max(plane)>value_thresh:
			x_coord=np.copy(x_base[i])
			if np.min(np.abs(coords[:, 0] - x_coord[0]))<dist_thresh: #distance cutoff for calculation efficiency
				for j,line in enumerate(plane):
					if np.max(line)>value_thresh:
						y_coord=np.copy(y_base[j])
						if np.min(np.abs(coords[:, 1] - y_coord[1]))<dist_thresh: #distance cutoff for calculation efficiency
							z_coords=np.copy(z_base.T)
	
							z_coords[0]=np.full(len(z_coords[0]),x_coord[0])
							z_coords[1]=np.full(len(z_coords[1]),y_coord[1])

							voxel_fast_dists=np.linalg.norm(coords[:,None,:] - z_coords.T, axis=2)

							points=np.array(line).astype(float)

							scores+=np.sum(np.abs(points)/(voxel_fast_dists**2),axis=1)

	if np.argmax(scores)!=total_index:
		print("Not centered on center!")
		return True

	return False

def get_center_2_index(total_index1,angles1,atoms2,center_element2,connectivity2,angles2):
	'''
	Given a small xyz and a chosen atom in that xyz and a bigger xyz, finds the total index of the corresponding atom in that big xyz
	'''
	center_1_angles=[]
	for point2 in angles1[total_index1]:
		for angle in point2:
			if not np.isnan(angle):
				if round(angle,2) not in center_1_angles:
					center_1_angles.append(round(angle,1))
	for i,atom in enumerate(atoms2):
		if atom==center_element2:
			test=True
			for j in connectivity2[i]:
				for k in connectivity2[i]:
					if j>k:
						if round(angles2[i][j][k],1) not in center_1_angles:
							test=False
							break
				if test==False:
					break
			if test==True:
				return i				

def compare_orbital_direction(center_element1,center_index1,cube1,center_element2,center_index2,cube2,grace=0):	
	'''
	Computes the similarity between two p orbitals in two cube files as the dot product after rotation to a common reference frame

	grace: only considers voxels within grace of the minimum and maximum bonded element coordinated
	'''

	covalent_radii={"P":1.07,"In":1.42,"Ga":1.22,"Li":1.28,"Cl":1.02,"F":0.57} #manual dictionaries
	atomic_numbers={'15':"P",'49':"In",'31':"Ga",'3':"Li",'17':"Cl","9":"F"}

	#grace2=10 #using different graces for each doesn't seem to change much and may actually be less accurate

	#read in cubes
	atoms1, coords1, grid_spacing1, origin1, mo_data1 = read_cube(cube1)
	#print(atoms1)
	total_index1=to_total_index(atoms1,center_element1,center_index1)
	dists1=dist_all_points(coords1)
	connectivity1=smart_connectivity_finder(dists1,atoms1)
	angles1=get_close_angles(coords1,dists1,connectivity1)
	atoms2, coords2, grid_spacing2, origin2, mo_data2 = read_cube(cube2)
	dists2=dist_all_points(coords2)
	connectivity2=smart_connectivity_finder(dists2,atoms2)
	angles2=get_close_angles(coords2,dists2,connectivity2)

	if center_index2=="search":
		total_index2=get_center_2_index(total_index1,angles1,atoms2,center_element2,connectivity2,angles2)
	else:
		total_index2=to_total_index(atoms2,center_element2,int(center_index2))

	badtest=test_p(total_index1,atoms1, coords1, grid_spacing1, origin1, mo_data1) #we only test if the first is bad here for time saving, but you could easily test both

	if badtest:	
		if center_index2=="search":	
			return 0, to_atom_specific_index(atoms2,total_index2)
		else:
			return 0

	transformed_orientation1=find_p_orbital_orientation(atoms1,connectivity1,total_index1,dists1,grace,mo_data1,grid_spacing1,origin1,coords1)
	transformed_orientation2=find_p_orbital_orientation(atoms2,connectivity2,total_index2,dists2,grace,mo_data2,grid_spacing2,origin2,coords2)

	#Compute the desired dot product
	dot=np.dot(transformed_orientation1,transformed_orientation2)

	if center_index2=="search":
		return abs(dot), to_atom_specific_index(atoms2,total_index2)
	else:
		return abs(dot)

def find_p_orbital_orientation(atoms,connectivity,total_index,dists,grace,mo_data,grid_spacing,origin,coords):
	max_dist=0
	for i,atom in enumerate(atoms):
		if i in connectivity[total_index]:
			if dists[total_index][i]>max_dist:
				max_dist=dists[total_index][i]
	max_dist=max_dist+grace

	#extract all relevant voxels from cube 1
	voxels_near_center=[]
	xyzs_near_center=[]
	for i,plane in enumerate(mo_data):
		x_coord=((i+0.5)*grid_spacing[0])+origin
		if abs(x_coord[0]- coords[total_index][0])<=max_dist:
			for j,line in enumerate(plane):
				y_coord=((j+0.5)*grid_spacing[1])+origin
				if abs(y_coord[1]- coords[total_index][1])<=max_dist:				
					for k,point in enumerate(line):
						z_coord=((k+0.5)*grid_spacing[2])+origin
						distance_from_center=np.linalg.norm([x_coord[0],y_coord[1],z_coord[2]]-coords[total_index])
						if distance_from_center<=max_dist:

							voxels_near_center.append(float(point))
							xyzs_near_center.append([x_coord[0],y_coord[1],z_coord[2]])

	#determine max and min voxels
	voxels_near_center=np.array(voxels_near_center)
	xyzs_near_center=np.array(xyzs_near_center)
	xyz_max=np.average(xyzs_near_center[voxels_near_center>0],axis=0,weights=voxels_near_center[voxels_near_center>0])
	xyz_min=np.average(xyzs_near_center[voxels_near_center<0],axis=0,weights=voxels_near_center[voxels_near_center<0])



	vector=xyz_max - xyz_min #positive-negative
	unit_vector=vector/np.linalg.norm(vector)
	#I need to define a new, locally consistent basis set
	inds_increasing_dists=np.argsort(dists[total_index])
	basis_vector1=coords[inds_increasing_dists[1]]-coords[total_index] #shortest bond length
	basis_unit1=basis_vector1/np.linalg.norm(basis_vector1)
	next_bond_vector=coords[inds_increasing_dists[2]]-coords[total_index]
	basis_vector2=basis_unit1-(next_bond_vector/np.dot(next_bond_vector,basis_unit1))#coplanar to basis vector 1 and the next shortest bond length, orthogonal to basis vector 1, positive orientation relative to basis vector 1
	basis_unit2=basis_vector2/np.linalg.norm(basis_vector2)
	basis_vector3=np.cross(basis_unit1,basis_vector2)
	basis_unit3=basis_vector3/np.linalg.norm(basis_vector3)
	transformed_orientation=np.matmul(np.linalg.inv(np.column_stack((basis_unit1,basis_unit2,basis_unit3))),unit_vector)

	return transformed_orientation

	

if __name__ == "__main__":
	main()