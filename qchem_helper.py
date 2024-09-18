import numpy as np
import sys
from pdos_helper import dos_grid_general,get_alpha,get_ao_ind, get_ind_ao_underc

'''
Helper functions for quantum chemistry

python3 compatible
'''

def get_final_serial_geom(file):
    '''
    Extracts the final input geometry from the input or output of a series of serial qchem single points
    Inputs:
        file -- .out or .in file
    Outputs: 
        xyz_coords -- np array with the coordinates of all the atoms (float, NAtoms x 3). Indexed the same as atom_names
        atom_names -- np array with the atom names (str, NAtoms)
    '''

    with open(file,'r') as out:

        all_lines=out.readlines()
        lines=[]
        for i in reversed(range(len(all_lines))):
            if all_lines[i].find("$molecule")!=-1:
                pos1=i
                break
        for i in range(pos1+2,len(all_lines)):
            if all_lines[i].find("$end")!=-1:
                break
            else:
                lines.append(all_lines[i].strip().split())
        lines=np.array(lines)
        atom_names=lines[:,0]
        xyz_coords=lines[:,1:].astype(float)
        return atom_names, xyz_coords

def get_all_alpha(atoms,nbas,mo_mat,orb_per_atom):
    '''
    computes alpha, the sum of all squared MO coefficients from a set of basis functions in a given MO, for each atom
    Inputs:
        atoms: np array of strs. elements in xyz. NAtoms
        nbas: int. the number of basis functions
        mo_mat: 2D np array of floats. the matrix of MO coefficients. nbas x nbas
        orb_per_atom: a dictionary of strs->ints. the number of basis functions for each element in a given basis set
    Outputs:
        alpha_total: 2D array of floats. NAtoms x nbas (atoms by orbs)
    '''

    every_atom=[]
    for i,atom in enumerate(atoms):
        ind_i=np.full(len(atoms),False)
        ind_i[i]=True
        every_atom.append(ind_i)
    every_atom=np.array(every_atom)

    every_atom_ao=get_ao_ind(every_atom,atoms,nbas,orb_per_atom)
    alpha_total=get_alpha(mo_mat,every_atom_ao)

    return np.array(alpha_total)

def get_total_lowdin(low_per_orb,atoms,nocc):
    '''
    computes the total lowdin charge for each atom from a lowdin charge per orb
    Inputs:
        low_per_orb: np array of floats. total lowdin charge for each atom in each orbital. NAtoms x NBas
        atoms: np array of strs. elements in xyz. NAtoms
        nocc: int. the number of occupied orbitals
    Outputs:
        np array of floats. total lowdin charge per atom. len NAtoms
    '''
    atomic_numbers={'In':21,"Ga":31,"P":15,"Cl":17,"Br":35,"F":9,"H":1,"O":8,"C":6,"S":16,"Li":3,"Al":13,"Zn":30,"Se":34,"Si":14} #wih ecps included

    total_lowdins=[]
    for i,atom in enumerate(atoms):
        total_charge=np.sum(low_per_orb[i][:nocc])
        total_lowdins.append(atomic_numbers[atoms[i]]-total_charge)
    return np.array(total_lowdins)

def read_cube(cube):
    '''
    This function takes a Q-Chem generated cube file and extracts the data in np arrays
    Grid spacing and origin are needed to map voxels to the correct xyz coordinates
    Inputs:
        cube: str. The name (and path if necessary) of the cube file
    Outputs:
        atoms: np array. Elements of each atom, len NAtoms
        coords: np array. xyz coordinates for each atom in Angstroms. len NAtomsx3        
        grid_spacing: np array. the spacing between grid points in each dimension in Angstroms. len 3x3
        origin: np array. xyz coordinates (A) of the origin, from which all grid points move positively. len 3
        mo_data: np array. the value of each voxel, transformed to 3D. NXpoints x NYpoints x NZpoints
    '''

    covalent_radii={"P":1.07,"In":1.42,"Ga":1.22,"Li":1.28,"Cl":1.02,"F":0.57} #manual dictionaries
    atomic_numbers={'15':"P",'49':"In",'31':"Ga",'3':"Li",'17':"Cl","9":"F"}

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

    # n_init=np.copy(n_grid)
    # n_init[2]=1
    n_init=tuple(n_grid)

    #COORDS ARE IN UNITS Bohr
    coords=np.array(coords)*.529177249
    grid_spacing=np.array(grid_spacing)*.529177249
    origin = np.array(origin)*.529177249

    #transform the mo_data into 3D
    mo_raw=np.loadtxt(cube,skiprows=(6+n_atoms))
    x=0
    y=0
    z=0
    mo_data=np.zeros(n_init)
    for i,line in enumerate(mo_raw):
        # # print(line)
        # if mo_data[x][y]==[0]:
        #     mo_data[x][y]=line
        # else:
        #     mo_data[x][y]=np.append(mo_data[x][y],line)
        mo_data[x][y][z:z+len(line)]=line

        if (z+6)>=n_grid[2]:
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

    return atoms, coords, grid_spacing, origin, mo_data



def get_tot_e(outfile):
    '''
    This function takes in a qchem output file and extracts the total energy
    Makes sure to get the first energy if it is a multi-job file
    Inputs:
        outfile: str. name of qchem.out Ideally a single point
    Outputs:
        tot_e: float. The total energy in the final basis set, in eV
    '''
    with open(outfile,"r") as out:
        for i, line in enumerate(out):
            if line.find("Total energy in the final basis set")!=-1:
                tot_e=float(line.strip().split()[-1])*27.2114
                return tot_e



def get_dipole(outfile):
    '''
    This function will eventually get the dipole vector from a qchem output file
    It zero-centers it too
    Inputs:
        outfile: str. name of the qchem.out. Ideally a single point. if a multi-job, gets the 1st dipole
    '''

    atoms,coords=get_geom_io(outfile)

    with open(outfile,'r',encoding="ISO-8859-1") as out:
        get_dipole=False
        get_charge=False
        for i,line in enumerate(out):
            if line.find("Charge (ESU x 10^10)")!=-1:
                get_charge=True
            elif get_charge:
                charge=float(line.strip().split()[0])
                get_charge=False
            elif line.find("Dipole Moment (Debye)") != -1:
                get_dipole=True
            elif get_dipole:
                line_array=line.strip().split()
                dipole=[float(line_array[1]),float(line_array[3]),float(line_array[5])]
                dipole=np.array(dipole)
                break

    average_coord_vector=np.array([0,0,0])
    for i,atom in enumerate(atoms):
        average_coord_vector=average_coord_vector+(coords[i])

    average_coord_vector=average_coord_vector/len(atoms)

    dipole=dipole-(average_coord_vector*charge)

    return np.array(dipole)

def read_beta_lowdin(outfile,atom,index,orb_num,orb_type):
    '''
    Function to extract the MO index corresponding to the first core orbital of a given type on a given atom
    For use in running ddft calculations, mainly
    Pretty much exactly the same as the original but reads Beta instead 

    Inputs: outfile -- name of qchem output file with Lowdin populations (str)
            atom -- atomic species targeted (str)
            index -- atom-specific index of targeted species (str). If 0, just looks for the first atom of the chosen element of any index 
            orb_num -- orbital shell to target. 2 for 2p, for example (int)
            orb_type -- orbital type to target. p for 2p, for example (str). Only works for s,p,d

    Outputs: mo_index -- mo index of target, not 0-indexed (int)
    '''

    #because the 2p orbital is the 1st p orbital
    if orb_type=="p":
        orb_num=orb_num-1
    elif orb_type=="d":
        orb_num=orb_num-2

    with open(outfile,'r') as out:
        flag=False
        lowdin=[]
        for i,line in enumerate(out):
            if flag==False:
                if line.find("Partial Lowdin        Populations for Occupied Beta Orbitals")!=-1:
                    flag=True
                    i_start=i
            else:
                if line.find("-----------------------------------------------------------------")!=-1:
                    flag=False
                elif i>i_start+1:
                    line=line[:10]+" "+line[10:]  #gotta add a space to separate "InX"
                    lowdin.append(line.strip().split())

    lowdin.pop(-1)

    orb_count=0
    converged=False
    six_count=0
    orbs=[]

    for i,line in enumerate(lowdin):
        if len(line)==6:
            six_count=six_count+1

        if six_count==1:
            orbs.append(int(line[0]))
        
        if line[1]==atom and (line[2]==index or index=="0") and line[3]==orb_type:            
            for j,pop in enumerate(line[4:]):
                if float(pop)>0.9: #this way it works for the unrestricted girlies too
                    orb_count=orb_count+1
                    if orb_count==orb_num:
                        final_i=i
                        final_j=j
                        converged=True
                        break
        if converged==True:
            break
    n_orbs=max(orbs)

    for i,line in enumerate(lowdin):
        if len(line)==6:
            if final_i-i<n_orbs:
                return line[final_j]

def my_write_plot(file_name,atoms,coords,spch_string,basis,static='2000',bound_string=None,cluster=None):
    '''
    A function to write plot.in files in a manner comparable to write_ipr. capable of making a plot-free single-point
    Inputs: file_name -- the name of the .in file to write (str)
            atoms -- np array of elements, in order (NAtoms x 1)
            coords -- np array of atomic positions, in Angstroms, in the same order (NAtoms x 3)
            spch_string -- a string containing the charge of the system, a space, and then the spin state of the system. For a neutral singlet, "0 1"
            basis -- the basis set to use (str). automatically uses def2 ecps if a def2 basis set is specified. must be in qchem format
            static -- (str) the value to set the qchem $rem variable MEM_STATIC to (default 2000 on T, 1000 on U)
            bound_string -- a string containing the lower and upper bound of alpha molecular orbitals to plot, separated with a dash. If uninitialized, does not make plots
            cluster -- the priority to run the job at, determining memory thresholds. If uninitialized, assumes Ulysses (4 GB total, 1 GB static)
    '''
    with open(file_name,"w") as plt:
        plt.write("$molecule \n")
        plt.write(spch_string+" \n")
        for i,atom in enumerate(atoms):
            plt.write( atom + " " + str(coords[i][0]) + " " + str(coords[i][1])+ " " + str(coords[i][2]) +" \n")
        plt.write("$end \n")

        plt.write("\n")

        plt.write("$rem \n")
        plt.write("jobtype sp \n")
        plt.write("method pbe0 \n")
        plt.write("basis "+basis+" \n")
        if basis[:4]=="def2":
            plt.write("ecp def2-ecp \n")

        if cluster is None:
            plt.write("mem_total 4000 \n")
            plt.write("mem_static 1000 \n")
        elif cluster=='veryhigh':
            plt.write("mem_total 16000 \n")
            plt.write("mem_static "+static+" \n")
        else:
            raise Exception("Priority not yet implemented!")

        plt.write("ithrsh_dft 15 \n")
        plt.write("scf_algorithm rca_diis \n")

        if bound_string is None:
            plt.write("$end \n")
        else:
            plt.write("plots true \n")
            plt.write("make_cube_files true \n")
            plt.write("$end \n")

            plt.write(" \n")

            plt.write("$plots \n")
            plt.write("grid_range (-20,20) (-20,20) (-20,20) \n")
            plt.write("grid_points 150 150 150 \n")
            plt.write("alpha_molecular_orbital "+bound_string+" \n")
            plt.write("$end \n")

def write_submit(file_name,cores,mode,cluster=None):
    '''
    A function that writes an sbatch submit submit_plot.sh script for the plot job (namely that saves the scratch directory)
    Inputs: file_name -- the name of the .in file to run (str). JOB_DESCRIPTOR.in
            cores -- (str) the number of cores to run the job on
            mode -- (str) the jobtype. If its a pure ipr, I don't need to monkey around with the scratch
            cluster -- the priority to run the job at. If uninitialized, assumes Ulysses (no priority specification)            
     '''

    descriptor=file_name[:-3]
    with open("submit_plot.sh",'w') as sh:
        sh.write("#!/bin/bash  \n")
        sh.write(" \n")
        sh.write("#SBATCH -J "+descriptor+" \n")
        sh.write("#SBATCH -o "+descriptor+".log \n")
        sh.write("#SBATCH -e "+descriptor+".log \n")
              

        if cluster is None:
            sh.write("#SBATCH --mem-per-cpu"+str(4000)+" \n")
            sh.write("#SBATCH -c "+cores+" \n")
            sh.write("#SBATCH --time unlimited \n")
        elif cluster=="veryhigh":
            sh.write("#SBATCH --mem-per-cpu 16000 \n")
            sh.write("#SBATCH -p "+cluster+" \n")
            sh.write("#SBATCH -c "+cores+" \n")
            sh.write("#SBATCH --time unlimited \n")
        elif cluster=="expanse":
            sh.write('#SBATCH --partition=shared \n')
            sh.write('#SBATCH --mem=150000 \n')
            sh.write('#SBATCH -t 48:00:00 \n')
            sh.write('#SBATCH --nodes=1 \n')
            sh.write('#SBATCH --ntasks-per-node='+cores+' \n')
            sh.write('#SBATCH --account=mit178 \n')
            sh.write(" \n")
            sh.write('module load cpu/0.15.4 \n')
            sh.write('module load gcc/10.2.0 \n')
            sh.write('module load mvapich2/2.3.6 \n')
            sh.write('module load qchem/6.0.2 \n')
            sh.write('\n')
            sh.write('export QCSCRATCH=/scratch/$USER/job_$SLURM_JOB_ID \n:')

        else:
            raise Exception("Supported priority values are 'ulysses', 'veryhigh', and 'expanse'")


        if mode=="ipr":
            sh.write(" \n")
            if cluster=="expanse":
                sh.write("qchem -nt "+cores+" "+file_name+" "+descriptor+".out  \n")
            else:
                sh.write("qchem.latest -nt "+cores+" "+file_name+" "+descriptor+".out  \n")
        else:
            sh.write(" \n")
            sh.write("scratch='scratch_"+descriptor+"' \n")
            sh.write("curr_d=$PWD \n")
            sh.write(" \n")
            sh.write("rm -r $QCSCRATCH/$scratch \n")
            sh.write("cp -r $scratch $QCSCRATCH \n")
            sh.write(" \n")
            if cluster=="expanse":
                sh.write("qchem -save -nt "+cores+" "+file_name+" "+descriptor+".out $scratch  \n")
            else:
                sh.write("qchem.latest -save -nt "+cores+" "+file_name+" "+descriptor+".out $scratch  \n")
            sh.write(" \n")
            sh.write("cp -r $QCSCRATCH/$scratch $curr_d \n")
            sh.write('rm -r $QCSCRATCH/$scratch \n')
            sh.write(" \n")
            sh.write("scp $scratch/53.0 . \n")
            sh.write("scp $scratch/320.0 . \n")
            sh.write("rm -r $scratch \n")



def write_ipr(file_name,atoms,coords,spch_string,new_charge,basis,static='2000',bound_string=None,cluster=None):
    '''
    A function to write ipr.in qchem files, with some flexibility
    Inputs: file_name -- the name of the .in file to write (str)
            atoms -- np array of elements, in order (NAtoms x 1)
            coords -- np array of atomic positions, in Angstroms, in the same order (NAtoms x 3)
            spch_string -- a string containing the charge of the system, a space, and then the spin state of the system. For a neutral singlet, "0 1"
            new_charge -- a string containing the "charge" of the fully-occupied system (charge - n_virtual orbitals)
            basis -- the basis set to use (str). automatically uses def2 ecps if a def2 basis set is specified. must be in qchem format
            static -- (str) the value to set the qchem $rem variable MEM_STATIC to (default 2000 on T, 1000 on U)
            bound_string -- a string containing the lower and upper bound of alpha molecular orbitals to plot, separated with a dash. If uninitialized, does not make plots
            cluster -- the priority to run the job at, determining memory thresholds. If uninitialized, assumes Ulysses (4 GB total, 1 GB static)
    '''    

    with open(file_name,"w") as plt:
        plt.write("$molecule \n")
        plt.write(spch_string+" \n")
        for i,atom in enumerate(atoms):
            plt.write( atom + " " + str(coords[i][0]) + " " + str(coords[i][1])+ " " + str(coords[i][2]) +" \n")
        plt.write("$end \n")

        plt.write("\n")

        plt.write("$rem \n")
        plt.write("jobtype sp \n")
        plt.write("method pbe0 \n")

        plt.write("basis "+basis+" \n")
        if basis[:4]=="def2":
            plt.write("ecp def2-ecp \n")

        if cluster is None:
            plt.write("mem_total 4000 \n")
            plt.write("mem_static 1000 \n")
        elif cluster=='veryhigh':
            plt.write("mem_total 16000 \n")
            plt.write("mem_static "+static+" \n")
        else:
            raise Exception("Priority not yet implemented!")

        plt.write("ithrsh_dft 15 \n")
        plt.write("scf_algorithm rca_diis \n")

        if bound_string is None:
            plt.write("$end \n")
        else:
            plt.write("plots true \n")
            plt.write("make_cube_files true \n")
            plt.write("$end \n")

            plt.write(" \n")

            plt.write("$plots \n")
            plt.write("grid_range (-20,20) (-20,20) (-20,20) \n")
            plt.write("grid_points 150 150 150 \n")
            plt.write("alpha_molecular_orbital "+bound_string+" \n")
            plt.write("$end \n")

        plt.write(" \n")
        plt.write("@@@ \n")
        plt.write(" \n")

        plt.write("$molecule \n")
        plt.write(new_charge+" 1 \n")
        for i,atom in enumerate(atoms):
            plt.write( atom + " " + str(coords[i][0]) + " " + str(coords[i][1])+ " " + str(coords[i][2]) +" \n")
        plt.write("$end \n")

        plt.write(" \n")

        plt.write("$rem \n")
        plt.write("jobtype sp \n")
        plt.write("method pbe0 \n")
        plt.write("basis "+basis+" \n")
        if basis[:4]=="def2":
            plt.write("ecp def2-ecp \n")
        plt.write("ithrsh_dft 15 \n")
        plt.write("scf_algorithm rca_diis \n")

        plt.write("skip_scfman true \n")
        plt.write("scf_guess read \n")
        plt.write("lowdin_population 2 \n")

        if cluster is None:
            plt.write("mem_total 4000 \n")
            plt.write("mem_static 1000 \n")
        elif cluster=='veryhigh':
            plt.write("mem_total 16000 \n")
            plt.write("mem_static "+static+" \n")
        else:
            raise Exception("Priority not yet implemented!")

        plt.write("$end \n")



def read_lowdin(outfile,atom,index,orb_num,orb_type):
    '''
    Function to extract the MO index corresponding to the first core orbital of a given type on a given atom
    For use in running ddft calculations, mainly
    Only looks at Alpha by default. 

    Inputs: outfile -- name of qchem output file with Lowdin populations (str)
            atom -- atomic species targeted (str)
            index -- atom-specific index of targeted species (str). If 0, just looks for the first atom of the chosen element of any index 
            orb_num -- orbital shell to target. 2 for 2p, for example (int)
            orb_type -- orbital type to target. p for 2p, for example (str). Only works for s,p,d

    Outputs: mo_index -- mo index of target, not 0-indexed (int)
    '''

    #because the 2p orbital is the 1st p orbital
    if orb_type=="p":
        orb_num=orb_num-1
    elif orb_type=="d":
        orb_num=orb_num-2

    with open(outfile,'r') as out:
        flag=False
        lowdin=[]
        for i,line in enumerate(out):
            if flag==False:
                if line.find("Partial Lowdin")!=-1:
                    flag=True
                    i_start=i
            else:
                if line.find("-----------------------------------------------------------------")!=-1:
                    flag=False
                elif line.find("Populations for Occupied Beta Orbitals")!=-1:
                    flag=False
                    lowdin.pop(-1)
                elif i>i_start+1:
                    line=line[:10]+" "+line[10:]  #gotta add a space to separate "InX"
                    lowdin.append(line.strip().split())

    lowdin.pop(-1)

    orb_count=0
    converged=False
    six_count=0
    orbs=[]

    for i,line in enumerate(lowdin):
        if len(line)==6:
            six_count=six_count+1

        if six_count==1:
            orbs.append(int(line[0]))
        
        if line[1]==atom and (line[2]==index or index=="0") and line[3]==orb_type:            
            for j,pop in enumerate(line[4:]):
                if float(pop)>0.9: #this way it works for the unrestricted girlies too
                    orb_count=orb_count+1
                    if orb_count==orb_num:
                        final_i=i
                        final_j=j
                        converged=True
                        break
        if converged==True:
            break
    n_orbs=max(orbs)

    for i,line in enumerate(lowdin):
        if len(line)==6:
            if final_i-i<n_orbs:
                return line[final_j]



def get_geom_cube(inputfile):
    '''
    Function to extract geometry from qchem cube file

    Inputs:  inputfile  -- name of qchem cube file with geometry
    Outputs: atom_names -- np array with the atom names (str)
             coords     -- np array with the xyz coords, in Bohr (float)
             cube_origin -- np array with the cube origin, which is in Bohr (float)
    '''
    with open(inputfile,'r') as inp:
        line_count = 0
        geom = []
        for i,line in enumerate(inp):
            if line_count==2:
                cube_origin=line.strip().split()
            if line_count>5 and len(line.strip().split()) > 5:
                break
            if line_count > 5:
                geom.append(line.strip().split())
            line_count=line_count+1

    geom = np.array(geom)
    atom_nums = geom[:,0]
    #cube files use numbers for atoms. Currently brute forcing this
    atom_names=[]
    for i,num in enumerate(atom_nums):
        if num=="13":
            atom_names.append("Al")
        elif num=="9":
            atom_names.append("F")
        elif num=="49":
            atom_names.append("In")
        elif num=='15':
            atom_names.append("P")
        elif num=='31':
            atom_names.append("Ga")
        elif num=='30':
            atom_names.append("Zn")
        elif num=='17':
            atom_names.append("Cl")
        elif num=='16':
            atom_names.append("S")
        elif num=='34':
            atom_names.append("Se")
        else:
            print("Element not supported for cube reading (yet)")
    atom_names=np.array(atom_names)
    
    coords = geom[:,2:].astype(float)

    cube_origin=np.array(cube_origin)
    cube_origin=cube_origin[1:].astype(float)

    return atom_names, coords, cube_origin

def get_converged_geom(opt_out):
    '''
    Another Ezra original, right off the cuff 
    Extracts the converged final geometry from a qchem optimization output

    Inputs: opt_out -- .out file with optimization
    Outputs: xyz_coords -- np array with the coordinates of all the atoms (float, NAtoms x 3). Indexed the same as atom_names
             atom_names -- np array with the atom names (str, NAtoms)
    '''
    with open(opt_out,'r') as out:
        flag=0
        lines=[]
        for i,line in enumerate(out):
            if line.find("**  OPTIMIZATION CONVERGED  **")!=-1:
                flag=1
            elif flag==1 and line.find("ATOM                X               Y               Z")!=-1:
                flag=2
            elif flag==2 and line.find("Z-matrix Print:")!=-1:
                break
            elif flag==2:
                lines.append(line.strip().split())
        lines.pop(-1)
        lines=np.array(lines)
        atom_names=lines[:,1]
        xyz_coords=lines[:,2:].astype(float)
        return xyz_coords, atom_names


def read_xyz(input_xyz):
    '''
    Function that reads xyz file into arrays

    Inputs: input_xyz -- .xyz file with the QD coordinates
    Outputs: xyz_coords -- np array with the coordinates of all the atoms (float)
             atom_names -- np array with the atom names (str)
    '''
    xyz_coords = np.loadtxt(input_xyz,skiprows=2,usecols=(1,2,3))
    atom_names = np.loadtxt(input_xyz,skiprows=2,usecols=(0,),dtype=str)
    return xyz_coords, atom_names

def write_geom(xyz_file,atom_names,atom_xyz):
    '''
    Function to write the geometry part of an xyz/input file

    Inputs: xyz_file   -- file object, open for writing, to write xyz to
            atom_names -- np array or list of atom names (str)
            atom_xyz   -- np array or list of atom xyz coordinates (float)

    Outputs: Writes to out_file
    '''

    for i, atom in enumerate(atom_names):
        xyz_file.write('{:2}     {:15.10f}     {:15.10f}    {:15.10f}\n'.format(atom, atom_xyz[i][0], atom_xyz[i][1], atom_xyz[i][2]))
    return

def write_xyz(out_file, atom_names, atom_xyz, comment=''):
    '''
    Function that writes xyz coordinate arrays to .xyz file

    Inputs: out_file   -- name of the file to write coordinates to
            atom_names -- np array or list of atom names (str)
            atom_xyz   -- np array or list of atom xyz coordinates (float)
            comment    -- comment line to write in the xyz file

    Outputs: Writes to xyz_file
    '''
    with open(out_file,'w') as xyz_file:
        xyz_file.write(str(len(atom_names))+'\n')
        xyz_file.write(comment+'\n')
        write_geom(xyz_file,atom_names,atom_xyz)
    return

def write_input(inpfile,atom_names,atom_xyz,rem,spcharge,comment=''):
    '''
    Function that writes qchem input file given geom and $rem

    Inputs: inpfile   -- name of the file to write info to
            atom_names -- np array or list of atom names (str)
            atom_xyz   -- np array or list of atom xyz coordinates (float)
            comment    -- comment line to write in the input file
            rem        -- string with the $rem section (one string with \n for line breaks)
            spcharge   -- string with the spin and charge

    Outputs: Writes to inpfile
    '''

    with open(inpfile,'w') as inputfile:
        inputfile.write('$comment\n')
        inputfile.write(comment+'\n')
        inputfile.write('$end\n')
        inputfile.write('\n')
        inputfile.write('$molecule\n')
        inputfile.write(spcharge)
        write_geom(inputfile,atom_names,atom_xyz)
        inputfile.write('$end\n')
        inputfile.write('\n')
        inputfile.write(rem)
    return

def get_geom_io(inputfile):
    '''
    Function to extract geometry from qchem input/output

    Inputs:  inputfile  -- name of qchem input/output file with geometry
    Outputs: atom_names -- np array with the atom names (str)
             coords     -- np array with the xyz coords (float)
    '''
    with open(inputfile,'r') as inp:
        flag = 0
        geom = []
        for i,line in enumerate(inp):
            if line.find('$end') != -1 and flag >= 1:
                break
            if flag > 1:
                geom.append(line.strip().split())
            if flag > 0:
                flag += 1
            if line.find('$molecule') != -1:
                flag = 1

    geom = np.array(geom)
    atom_names = geom[:,0]
    coords = geom[:,1:].astype(float)
    return atom_names, coords

def get_geom_cycle(inputfile,Nopt):
    '''
    Function to extract geometry from qchem optimization output

    Inputs:  inputfile  -- name of qchem input/output file with geometry
             Nopt       -- cycle of the optimization that you want
    Outputs: atom_names -- np array with the atom names (str)
             coords     -- np array with the xyz coords (float)
    '''
    with open(inputfile,'r') as inp:
        flag = 0
        geom = []
        for i,line in enumerate(inp):
            if flag == 3 and line.find('--------------------------') != -1:
                break
            if flag == 3:
                geom.append(line.strip().split()[1:])
            if flag > 0 and line.find('--------------------------') != -1:
                flag += 1
            if line.find('Optimization Cycle: {:>3}'.format(Nopt)) != -1:
                flag = 1

    geom = np.array(geom)
    # print(geom)
    atom_names = geom[:,0]
    coords = geom[:,1:].astype(float)
    # print(coords)
    return atom_names, coords

def get_geom_e_opt_cts(lines):
    '''
    Function to extract lowest E geometry from qchem geom opt

    Inputs:  lines  -- readlines(opt file)
    Outputs: lowest_e -- energy of lowest E structure
             lowest_g -- np array with the geom (str with atom name and coords)
             low_i    -- index of lowest energy cycle
    '''
    flag = 0
    lowest_e = 0
    low_i = 0
    lowest_g = []
    energies=[]
    geoms=[]
    for i,line in enumerate(lines):
        if line.find("Energy is") != -1:
            e = float(line.split()[-1])
            if e < lowest_e:
                lowest_e = e
                lowest_g = geom

            energies.append(e)
        if line.find('Point Group') != -1:
            flag=0
            geoms.append(geom)
        if flag > 0:
            geom.append(line.strip().split()[1:])
        if line.find('ATOM') != -1:
            geom = []
            flag = 1

    return lowest_e, np.array(lowest_g), np.argmin(energies)

def get_geom_e_opt_last(lines):
    '''
    Function to extract last geometry from qchem geom opt

    Inputs:  lines  -- readlines(opt file)
    Outputs: last_g -- np array with the geom (str with atom name and coords)
    '''
    flag = 0
    geom=[]
    for i,line in enumerate(reversed(lines)): # goes through backwards, so first instance of "Angstroms" should be the last printed geom
        # ADD GEOM/END GEOM
        if flag == 2 or flag == 3:
            if line.find("------------------") != -1:
                break
            elif line.find("ATOM") != -1:
                break
            else:
                geom.insert(0,line.strip().split()[1:])
        if flag == 1:
            flag += 1 # need to skip 1 line if in "SNO" format

        # START GEOM
        if line.find("Molecular Point Group") != -1: # "standard nuclear orientation" format
            flag=1
        elif line.find("Number of degrees of freedom") != -1:
            flag=3

    return np.array(geom)

def get_rem_sp(lines):
    '''
    Function to extract $rem section and spin/charge from qchem input/output

    Inputs:  lines  -- readlines(qchem input/output file)
    Outputs: rem -- string with the rem section separated by \n
             spcharge     -- string with charge and spin
    '''
    flag = 0
    rem = []
    spcharge=[]
    for i,line in enumerate(lines):
        if line.find('$molecule') != -1:
            spcharge = lines[i+1]
        if line.find('$rem') != -1:
            flag += 1
        if flag > 1:
            rem.append(line)
        if flag>1 and line.find('$end') != -1:
            break

    #print(rem)

    return ''.join(rem), spcharge

def my_get_rem_sp(inputfile):
    '''
    Function to extract $rem section and spin/charge from qchem input/output

    Inputs:  inputfile  -- (qchem input/output file)
    Outputs: rem -- string with the rem section separated by \n
             spcharge     -- string with charge and spin
    '''
    with open(inputfile, "r",encoding="ISO-8859-1") as inp:
        rem_flag = False
        rem = []
        spcharge = 0
        sp_flag = False
        input_start=False
        for i,line in enumerate(inp):
            if sp_flag:
                spcharge = line
                sp_flag = False
            if line.find('$molecule') != -1:
                sp_flag = True
                input_start
            if line.find('$rem') != -1 and input_start:
                rem_flag = True
            if rem_flag:
                rem.append(line)
            if rem_flag and line.find('$end') != -1:
                break
        return rem, spcharge

# def get_geom_e_opt_argmin(lines):
#     flag = 0
#     lowest_e = 0
#     lowest_g = []
#     energies=[]
#     geoms=[]
#     for i,line in enumerate(lines):
#         if line.find("Energy is") != -1:
#             e = float(line.split()[-1])
#             energies.append(e)
#         if line.find('Point Group') != -1:
#             flag=0
#             geoms.append(geom)
#         if flag > 0:
#             geom.append(line.strip().split()[1:])
#         if line.find('ATOM') != -1:
#             geom = []
#             flag = 1
#
#     low_i = np.argmin(energies)
#     lowest_e = energies[low_i]
#     lowest_g = np.array(geoms[low_i])
#     return lowest_e, lowest_g
