import sys
import os
import numpy
from qchem_helper import *
from geom_helper import dist_all_points, center_coords, get_close_angles, smart_connectivity_finder, angles_on_center, match_cutout

#there's a wrapper script for running this in many directories and actually running the jobs it generates
#from a job directory, run five new jobs which apply an electric field in a direction that agrees with the QD but varies in field strength
#field strengths
#	maximum: the coordinate vector projected onto the full dipole moment of the QD with direction preserved
#	minimum: the max dipole moment of the QD divided by the total # of anions
#	middle:	25%, 50%, and 75% of the way between these extremes
#while it may be a better interpolation if you scale up the electric field throughout the interpolation
#but it turns out this leads to convergence issues
#so, since we are only interested in the final frame for these anyway we use a constant dipole
#Q-Chem prints dipole moments in Debye but takes electric fields in atomic units. I am using a conversion of 1 Debye = 0.393430 au

#this script makes a dipoles subdirectory and five subdirectories therein, one for each interpolation

def main():

	interp=sys.argv[1] #the interpolation.in
	out=sys.argv[2] #the plot.out

	dir_name="with_proj_dipole" #the name of the directory that each new interpolation directory will be created in

	#read QD .out
	dipole=get_dipole(out)
	atoms,coords=get_geom_io(out)
	centered_coords=center_coords(coords)
	ind_an = np.logical_or(atoms=="P",np.logical_or(atoms=="S",atoms=="Se"))
	n_an=np.count_nonzero(ind_an)

	#read geom from serial.in
	interp_atoms, interp_coords=get_final_serial_geom(interp)
	centered_interp=center_coords(interp_coords)
	
	centsort_cutout, matched_atoms, sorted_icoords, sorted_iatoms, coordinate_center  = match_cutout(atoms,centered_coords,interp_atoms,centered_interp)
	
	#project the coordinate vector onto the dipole, preserving direction
	num=abs(np.dot(coordinate_center,dipole))
	den=np.dot(dipole,dipole)
	proj_dipole=(num/den)*dipole

	rot_dipole=rotate_dipole(centsort_cutout,sorted_icoords,proj_dipole)

	os.makedirs(dir_name,exist_ok=True)
	os.makedirs(dir_name+"/max",exist_ok=True)
	os.makedirs(dir_name+"/min",exist_ok=True)
	os.makedirs(dir_name+"/25",exist_ok=True)
	os.makedirs(dir_name+"/50",exist_ok=True)
	os.makedirs(dir_name+"/75",exist_ok=True)

	with open(interp,"r") as file:
		lines=file.readlines()

	name_split=interp.split(".")
	dipolify_serial_qchem(lines,rot_dipole,dir_name+"/max",name_split[0]+"_max_dipole.in")
	dipolify_serial_qchem(lines,rot_dipole/n_an,dir_name+"/min",name_split[0]+"_min_dipole.in")
	dipolify_serial_qchem(lines,(rot_dipole/n_an)+((rot_dipole-(rot_dipole/n_an))/4),dir_name+"/25",name_split[0]+"_25_dipole.in")
	dipolify_serial_qchem(lines,(rot_dipole/n_an)+((rot_dipole-(rot_dipole/n_an))/2),dir_name+"/50",name_split[0]+"_50_dipole.in")
	dipolify_serial_qchem(lines,(rot_dipole/n_an)+((3*(rot_dipole-(rot_dipole/n_an)))/4),dir_name+"/75",name_split[0]+"_75_dipole.in")

def dipolify_serial_qchem(lines,dipole,path,name):
	'''
	Takes in the lines of a serial qchem input fike
	And writes a copy of the file to a specified path
	Except the new file has a specified electric field applied that matches the dipole
	Inputs:
		lines: a list of the lines in the file, [...,"line_i backslash n",...]
		dipole: a len 3 np array of floats representing the dipole in debye
		path: the path to the directory to write the new file into
		name: the name of the file to write
	'''
	dipole=0.393430*dipole #convert to a.u.
	count=sum(s.count("$molecule") for s in lines)-1
	i=0
	with open(path+"/"+name,"w") as file:
		for line in lines:
			if line.find("$rem")!=-1:
				file.write("$multipole_field \n")
				# file.write("	X "+str((i*dipole[0])/count)+" \n")
				# file.write("	Y "+str((i*dipole[1])/count)+" \n")
				# file.write("	Z "+str((i*dipole[2])/count)+" \n")
				file.write("	X "+str(dipole[0])+" \n") #turns out that using a constant dipole improves convergence
				file.write("	Y "+str(dipole[1])+" \n")
				file.write("	Z "+str(dipole[2])+" \n")
				file.write("$end \n")
				file.write("\n")
				file.write(line)
				file.write("scf_algorithm rca_diis \n")
				file.write("max_scf_cycles 500 \n")
				file.write("THRESH_RCA_SWITCH 5 \n")
				file.write("MAX_RCA_CYCLES 250 \n")
				i=i+1
			else:
				file.write(line)

def rotate_dipole(centsort_cutout,sorted_icoords,dipole):
	'''
	Takes two sets of coordinates and a vector
	And uses the two sets of coordinates to find the rotation matrix that transforms the dipole from the first set of coordinates to the second
	both sets of coordinates need to be centered at the origin
	The order of the atoms in each coordinate must match
	Inputs:
		centsort_cutout: np array of len 3 np arrays of floats. the coordinates that align with the dipole
		sorted_icoords: np array of len 3 np arrays of floats. the new coordinates
		dipole: a len 3 np array of floats
	Outputs:
		rot_dipole: the same vector but rotated to match the new coordinates
	'''

	cov=np.dot(centsort_cutout.T,sorted_icoords)
	u,s,vt=np.linalg.svd(cov)
	r=np.dot(vt.T,u.T)

	rot_dipole=np.dot(r,dipole)

	return rot_dipole

if __name__ == "__main__":
	main()