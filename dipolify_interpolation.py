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

	#manual parameters
	dir_name="with_set_dipole" #the name of the directory that each new interpolation directory will be created in
	run_names=["n0.02","n0.04","n0.06","n0.08","n0.10","0.01","0.02","0.03","0.04","0.05","0.06","0.07","0.08","0.09","0.10","0.12","0.14","0.16","0.18","0.20"] #names of the directories to write
	run_type="set" #set or mult. Determines how the run strengths are applied. if set, the run strength is the field strength. If mult, the field strength is the total dipole times that value
	interpolate_field=True #if true, field is gradually increased across interpolation. If false, it is kept constant at the final value
	run_strengths=[-0.02,-0.04,-0.06,-0.08,-0.10,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.12,0.14,0.16,0.18,0.20] #the strength of the field for each run, corresponding to run names

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

	with open(interp,"r") as file:
			lines=file.readlines()
	name_split=interp.split(".")

	os.makedirs(dir_name,exist_ok=True)

	for i,subd in enumerate(run_names):

		os.makedirs(dir_name+"/"+subd,exist_ok=True)
	
		if run_type=="set":
			run_dipole= (rot_dipole*run_strengths[i])/np.linalg.norm(rot_dipole)
		elif run_type=="mult":
			run_dipole= (rot_dipole*run_strengths[i])
		else:
			raise Exception("Invalid value for run_type: must be 'set' or 'mult'")

		dipolify_serial_qchem(lines,run_dipole,dir_name+"/"+subd,name_split[0]+"_"+subd+".in",interpolate_field)
		
def dipolify_serial_qchem(lines,dipole,path,name,interpolate_field=True):
	'''
	Takes in the lines of a serial qchem input fike
	And writes a copy of the file to a specified path
	Except the new file has a specified electric field applied that matches the dipole
	Inputs:
		lines: a list of the lines in the file, [...,"line_i backslash n",...]
		dipole: a len 3 np array of floats representing the dipole in debye
		path: the path to the directory to write the new file into
		name: the name of the file to write
		interpolate_field: whether the field should be interpolated or kept constant
	'''
	dipole=0.393430*dipole #convert to a.u.
	count=sum(s.count("$molecule") for s in lines)-1
	i=0
	with open(path+"/"+name,"w") as file:
		for line in lines:
			if line.find("$rem")!=-1:
				file.write("$multipole_field \n")
				if interpolate_field:
					file.write("	X "+str((i*dipole[0])/count)+" \n")
					file.write("	Y "+str((i*dipole[1])/count)+" \n")
					file.write("	Z "+str((i*dipole[2])/count)+" \n")
				else:
					file.write("	X "+str(dipole[0])+" \n") #turns out that using a constant dipole improves convergence
					file.write("	Y "+str(-dipole[1])+" \n")
					file.write("	Z "+str(dipole[2])+" \n")
				file.write("$end \n")
				file.write("\n")
				file.write(line)
				file.write("max_scf_cycles 500 \n")
				# file.write("scf_algorithm rca_diis \n")
				# file.write("THRESH_RCA_SWITCH 5 \n")
				# file.write("MAX_RCA_CYCLES 250 \n")
				i=i+1
			elif line.find("grid_range")!=-1:
				file.write("grid_range (-5,5) (-5,5) (-5,5) \n")
			elif line.find("grid_points")!=-1:
				file.write("grid_points 90 90 90 \n")
			elif line.find("alpha_molecular_orbital")!=-1:
				file.write("alpha_molecular_orbital 13 \n")
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