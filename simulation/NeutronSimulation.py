import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from numbers import Number

def get_all_files(filename,ending='.root'):
    dir_path = './'

    index=[i+1 for i in range(len(filename)) if  filename[i]=='/']
    if len(index)>0:
        dir_path=filename[:index[-1]]
        filename=filename[index[-1]:]

    res = []
    filelist=os.listdir(dir_path)
    filelist.sort()
    for file in filelist:
        if file.startswith(filename) and file.endswith(ending):
            res.append(f'{dir_path}{file}')
    return res


class SimulationFile():
    def __init__(self,file="neutron-sim-design0.csv", skiprows=1):
        # load the panda data frame of the simulation output from csv file
        self.file = file
        if not os.path.exists(file):
            print("Warning ", file, "does not exist.")
            self.df = pd.DataFrame()
        else:
            self.df = pd.read_csv(file,skiprows=1)
    
    def print(self):
        # print the data frame
        return display(self.df)
    
    def keys(self):
        '''
        This function returns the available keys in the data frame corresponding to the simulation parameter outputs (e.j. x[m], time[ms]...)
        '''
        return self.df.keys()

    def is_captured(self, option=""):
        '''
        This function returns a vector flagging the neutrons if captured on Ge77
        '''
        nCZ= self.get("nC_Z")
        nCA= self.get("nC_A")
        cross_cryo= self.get("cross_cryo")
        if option=="in_LAr_only":
            is_in_LAr = self.is_in_LAr()
            vec = [ 1 if (nCZ[i]==32 and nCA[i]==77 and cross_cryo[i]!=-1) else 0  for i in range(len(nCZ)) if is_in_LAr[i]==1]
        else:
            vec = [1 if (nCZ[i]==32 and nCA[i]==77 and cross_cryo[i]!=-1) else 0  for i in range(len(nCZ))]
        return vec
    
    def NGe77(self):
        '''
        This function returns the number of neutron captures on Ge77 in file
        '''
        return len([1 for i in self.is_captured() if i == 1])
    
    def get_lnE0vsET(self):
        ekin= self.get("ekin[eV]")
        lnE0vsET=[]
        for e in ekin:
            if e[-1]==0.:
                e[-1]=0.0000001
            if e[0]==0.:
                e[0]=0.0000001
            lnE0vsET.append(np.log(e[0]/e[-1]))
        return lnE0vsET
    
    def get_r(self):
        x=self.get("x[m]")
        y=self.get("y[m]")
        r = [np.sqrt(x[i]**2+y[i]**2) for i,_ in enumerate(x)]
        return r

    def get_L(self):
        x=self.get("x[m]")
        y=self.get("y[m]")
        z=self.get("z[m]")
        return [np.sqrt(x[i]**2+y[i]**2+z[i]**2) for i,_ in enumerate(x)]
    
    def get_nsteps(self):
        x=self.get("x[m]")
        nsteps = [len(i) for i in x]
        return nsteps
    
    def get_total_nC_Ge77(self):
        nC_Ge77=self.is_captured()
        nsec_with_nC=self.get("nsecondaries_with_nC")
        total_nC_Ge77 = [nC_Ge77[i]+nsec_with_nC[i] for i in range(len(nC_Ge77))]
        return total_nC_Ge77
    
    
    def is_in_LAr(self):
        '''
        This function returns a vector flagging the neutrons entering the LAr or being created in LAr
        '''
        cross_cryo= self.get("cross_cryo")
        cross_uar= self.get("cross_uar")

        is_in_LAr=[]
        for i in range(len(cross_cryo)):
            if cross_cryo[i] != -1 or (cross_cryo[i]==-1 and cross_uar[i]!=-1):
                is_in_LAr.append(1)
            else: 
                is_in_LAr.append(0)
        return is_in_LAr 
       
    
    def get(self, key, option=""):
        '''
        This function returns a numpy array of numpy arrays (e_ij) for the given key (e.j. x[m], time[ms]...), i runs over the neutrons, j runs over the key entries (e.j. x1, x2, x3,...)
        '''
        if key in self.df:
            vec=self.df[key].to_numpy()
            if isinstance(vec[0],Number)==False:
                vec=[np.fromstring(x[1:-1], sep=' ') for x in vec ]
        elif key == "r[m]":
            vec=self.get_r()
        elif key == "L[m]":
            vec=self.get_L()
        elif key == "ln(E0vsET)":
            vec=self.get_lnE0vsET()
        elif key == "nsteps":
            vec=self.get_nsteps()
        elif key == "total_nC_Ge77[cts]":
            vec=self.get_total_nC_Ge77()
        else:
            if not self.df.empty:
                print ("Warning",key, "does not exists in the DataFrame. Returning empty vector",self.keys())
            return [0]
        
        if option=="last" and isinstance(vec[0],Number)==False:
                vec=[x[len(x)-1] for x in vec]
        elif option=="first"  and isinstance(vec[0],Number)==False:
                vec=[x[0] for x in vec]

        if isinstance(vec[0],Number):
                vec=[x for x in vec] 
        else:
            if option=="last":
                element=1
                if "edep" in key or "ekin" in key or "xmom[m]" in key or "ymom[m]" in key or "zmom[m]" in key:
                    element=2
                vec=[x[len(x)-element] for x in vec]
            elif option=="first":
                vec=[x[0] for x in vec]
            else:
                vec=[x for x in vec]
        return vec

    
    def get_is_in_LAr(self, key, option=""):
        '''
        This function returns a numpy array of numpy arrays (e_ij) for the given key (e.j. x[m], time[ms]...), i runs over only these neutrons which cross the LAr, j runs over the key entries (e.j. x1, x2, x3,...)
        '''

        if key in self.df:
            vec=self.df[key].to_numpy()
            if isinstance(vec[0],Number)==False:
                vec=[np.fromstring(x[1:-1], sep=' ') for x in vec ]
        elif key == "r[m]":
            vec=self.get_r()
        elif key == "L[m]":
            vec=self.get_L()
        elif key == "ln(E0vsET)":
            vec=self.get_lnE0vsET()
        elif key == "nsteps":
            vec=self.get_nsteps()
        else:
            if not self.df.empty:
                print ("Warning",key, "does not exists in the DataFrame. Returning empty vector",self.keys())
            return [0]
        
        is_in_LAr=self.is_in_LAr()
        if isinstance(vec[0],Number):
                vec=[x for i,x in enumerate(vec) if is_in_LAr[i] == 1 ] 
        else:
            if option=="last":
                element=1
                if "edep" in key or "ekin" in key or "xmom[m]" in key or "ymom[m]" in key or "zmom[m]" in key:
                    element=2
                vec=[x[len(x)-element] for i,x in enumerate(vec) if is_in_LAr[i] == 1 ]
            elif option=="first":
                vec=[x[0] for i,x in enumerate(vec) if is_in_LAr[i] == 1 ]
            else:
                vec=[x for i,x in enumerate(vec) if is_in_LAr[i] == 1 ]
        return vec

    def neutrons_in_LAr(self):
        '''
        This function returns the number of neutron entering the LAr or being created in LAr
        '''
        return len([1 for i in self.is_in_LAr() if i == 1])
    
    def get_design(self):
        '''
        This function returns the design parameters of the moderator: [fidelity, radius, thickness, npanels, phi, theta, length, height, z_offset, volume ]
        '''
        
        with open(self.file) as f:
            first_line = f.readline()
            first_line = first_line.split('#', 1)[1]
            first_line = first_line.split('#', 1)[0]
            first_line = first_line.split(']', 1)[0]
            if 'HF' in first_line:
                first_line = first_line.split('HF,', 1)[1]
                fidelity = 1
            if 'LF' in first_line:
                first_line = first_line.split('LF,', 1)[1]
                fidelity = 0
        design=np.fromstring(first_line, sep=',')
        design[0]=fidelity
        return design
    
    

class NeutronSimulation(SimulationFile):
    def __init__(self, filename="neutron-sim-design0"):
        self.files = get_all_files(filename,".csv")
        print("files: ", self.files)

    def keys(self):
        sim_tmp=SimulationFile(self.files[0])
        print(sim_tmp.keys())

    def NGe77(self):
        nge77=0
        for file in tqdm(self.files):
            sim_tmp=SimulationFile(file)
            nge77 += sim_tmp.NGe77()
        return nge77
    
    def is_captured(self, option=""):
        is_captured=[]
        for file in tqdm(self.files):
            sim_tmp=SimulationFile(file)
            is_captured.extend(sim_tmp.is_captured(option))
        return is_captured
    
    def is_in_LAr(self):
        is_in_LAr=[]
        for file in tqdm(self.files):
            sim_tmp=SimulationFile(file)
            is_in_LAr.extend(sim_tmp.is_in_LAr())
        return is_in_LAr
    
    def neutrons_in_LAr(self):
        n_in_LAr=0
        for file in tqdm(self.files):
            sim_tmp=SimulationFile(file)
            n_in_LAr += sim_tmp.neutrons_in_LAr()
        return n_in_LAr

    def get(self, keys, option=""):
        nkeys=len(keys)
        vec=[]
        for l in range(nkeys):
            vec.append([])
            if option=="summary":
                vec.append([])
        if option=="summary":
            vec.append([])
        for file in tqdm(self.files):
            sim_tmp=SimulationFile(file)
            vec[0].extend(sim_tmp.is_captured())
            for l in range(nkeys):
                if option=="summary":
                    vec[l+1].extend(sim_tmp.get(keys[l],"first"))
                    vec[nkeys+l+1].extend(sim_tmp.get(keys[l],"last"))
                else:
                    vec[l].extend(sim_tmp.get(keys[l],option))
        return vec

    def get_is_in_LAr(self, keys, option=""):
        nkeys=len(keys)
        vec=[]
        
        for l in range(nkeys):
            vec.append([])
            if option=="summary":
                vec.append([])
        if option=="summary":
            vec.append([])

        for file in tqdm(self.files):
            sim_tmp=SimulationFile(file)
            vec[0].extend(sim_tmp.is_captured("in_LAr_only"))
            for l in range(nkeys):
                if option=="summary":
                    vec[l+1].extend(sim_tmp.get_is_in_LAr(keys[l],"first"))
                    vec[nkeys+l+1].extend(sim_tmp.get_is_in_LAr(keys[l],"last"))
                else:
                    vec[l].extend(sim_tmp.get_is_in_LAr(keys[l],option))
        return vec
    
    def get_design(self):
        sim_tmp=SimulationFile(self.files[0])
        design = sim_tmp.get_design()
        return design