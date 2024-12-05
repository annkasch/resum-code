import uproot
import awkward as ak
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
from tqdm import tqdm
import argparse

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

def GetTreeKeySimulation(filekeys):
    if IsTreeKey(filekeys,'IndividualNeutronStepInfo')==False:
        return 'Score'
    else:
        return 'IndividualNeutronStepInfo'

def IsTreeKey(filekeys,keyname):
    for k in filekeys:
        if keyname in k:
            return True
    return False

def get_neutron_info(filename, option='detailed'):
    file=uproot.open(filename)
    print(filename)
    if len(file.keys())==0:
        file.close()
        print(f"WARNING: Empty file {filename}", flush=True)
        sys.exit(1)
    # read in root TTree named Score
    key=GetTreeKeySimulation(file.keys())
    tree = file[key]
    keyprefix=''
    if key=='Score':
        keyprefix='IndividualNeutronStep_'
    tree.keys()
    branches = tree.arrays(library='np')
    # read in TBranches containing neutron step info vector<double> => list of arrays
    nidx=branches[f'{keyprefix}NeutronIndex']
    tid=branches[f'{keyprefix}ID']
    
    xpos=branches[f'{keyprefix}Position_x_in_m']

    if option == "detailed":
        time=branches[f'{keyprefix}Timing_in_ms']
        ypos=branches[f'{keyprefix}Position_y_in_m']
        zpos=branches[f'{keyprefix}Position_z_in_m']
        xmom=branches[f'{keyprefix}Momentum_x_in_m']
        ymom=branches[f'{keyprefix}Momentum_y_in_m']
        zmom=branches[f'{keyprefix}Momentum_z_in_m']
        ekin=branches[f'{keyprefix}Ekin_in_eV']
        edep=branches[f'{keyprefix}Edep_in_eV']
        

    process=branches[f'{keyprefix}Process']
    volume=branches[f'{keyprefix}Volume']
    nC_A=branches[f'{keyprefix}Secondary_A']
    nC_Z=branches[f'{keyprefix}Secondary_Z']
    tree_score = file["Score"]
    branches_score = tree_score.arrays(library='np')
    weight=branches_score['EventWeights']


    x_new=[]
    y_new=[]
    z_new=[]
    tid_new=[]
    weight_new=[]
    xmom_new=[]
    ymom_new=[]
    zmom_new=[]
    time_new=[]
    vol_new=[]
    ekin_new=[]
    edep_new=[]
    process_new=[]
    nC_Z_new=[]
    nC_A_new=[]
    nidx_new=[]
    cross_water=[]
    cross_cryo=[]
    cross_uar=[]
    tracksize=0
    startTime = datetime.now()
    last_index=0
    nsec=[]
    nsec_with_nC=[]
    for i in tqdm(range(len(xpos))):
    #for i in range(len(xpos)):
        #if i % 2000 == 0:
        #    print(f"event {i}")
        # if no secondary neutron in event i -> xpos[i] empty
        if (len(xpos[i])==0):
            continue
        # x[i] stores all individual x step values for all neutrons appended in an array of dimension 1
        # x[i]= [x_j^k] (1...j...number of steps of neutron k, number of steps is variable for each neutron k
        # e.g. i contrains 3 (k from 0 to 2) neutron with 1, 3 and 2 (j from 0 to 0,2 and 1) steps per track: x[i]=[x_j^k]=[x_0^0, x_0^1,x_1^1,x_2^1,x_0^2,x_1^2]
        # same applies for stepwise y, z, momentum, time, energy, ... parameters
        # nidx[i] contains the mapping e.g. nidx[i]=[0,1,1,1,2,2]
        last_index=0
        for j in range(len(xpos[i])): 
                # find all indices where nidx[i][j-1]!=nidx[i][j] => e.g. index = 1, slit array at index
                if ((nidx[i][j-1]!=nidx[i][j] and j>0) or j==len(xpos[i])-1):
                    index=j
                    if index == len(xpos[i])-1:
                        index = len(xpos[i])
                    if len(xpos[i][last_index:index]) > tracksize:
                        tracksize=len(xpos[i][last_index:index])
                    
                    if (nidx[i][last_index]==0):
                        nsec.append(nidx[i][-1]+1)
                        weight_new.append(weight[i])
                        
                        if option == "detailed":
                            # individual number of steps for each neutron => variable lenght of each x_new[i], len(x_new)!=const
                            x_new.append(xpos[i][last_index:index])
                            y_new.append(ypos[i][last_index:index])
                            z_new.append(zpos[i][last_index:index])
                            time_new.append(time[i][last_index:index])
                            xmom_new.append(xmom[i][last_index:index])
                            ymom_new.append(ymom[i][last_index:index])
                            zmom_new.append(zmom[i][last_index:index])
                            ekin_new.append(ekin[i][last_index:index])
                            edep_new.append(edep[i][last_index:index])

                    ## containes the detector volume ID for each step
                    ## ID gets bigger going inwards (water 3, gap4, cyro wall 5, lar 7, uar 11,...), going outwards ID gets smaller
                    ## Lookup Volume:
                    ## lookup["Cavern_log"]   = 0;
                    ## lookup["Hall_log"]     = 1;
                    ## lookup["Tank_log"]     = 2;
                    ## lookup["Water_log"]    = 3;
                    ## lookup["Cout_log"]     = 4;
                    ## lookup["Cvac_log"]     = 5;
                    ## lookup["Cinn_log"]     = 6;
                    ## lookup["Lar_log"]      = 7;
                    ## lookup["Lid_log"]      = 8;
                    ## lookup["Bot_log"]      = 9;
                    ## lookup["Copper_log"]   = 10;
                    ## lookup["ULar_log"]     = 11;
                    ## lookup["Ge_log"]       = 12;
                    ## lookup["Pu_log"]       = 13;
                    ## lookup["Membrane_log"] = 14;

                        vol_new.append(volume[i][last_index:index])
                        # contains interaction physics process for each step, process_new[i][0] contains neutron creation process
                        process_new.append(process[i][last_index:index])

                        # the final step of each neutron track containes mass and charge of particle in case it got captured, else nC_A/Z[i,j]=0
                        # split array for each neutron
                        tmp=tid[i][last_index:index]
                        tid_new.append(tmp[0])
                        nsec_with_nC.append(0)

                        # get atomic mass of last element/ step of neutron track
                        tmp=nC_A[i][last_index:index]
                        nC_A_new.append(tmp[-1])
                        tmp=nC_Z[i][last_index:index]
                        nC_Z_new.append(tmp[-1])
  
                        nidx_new.append([i,nidx[i][last_index]])

                        tmp=vol_new[len(vol_new)-1]
                        # find first index in vol_new where neutron crosses between two volumes by changing volume ID 
                        # crossing from inside into the water volume (only water) which has ID=3, crossing when vol[i-1]>vol[i] and vol[i]=3
                        itmp=[i for i in range(len(tmp)) if tmp[i]==3 and (i>0 and tmp[i-1]>3)]
                        if len(itmp)>0:
                            cross_water.append(itmp[0])
                        else:
                            cross_water.append(-1)
                        # crossing from outside into all volumes inside cryostat (> ID 7), crossing when vol[j-1]<vol[i] and vol[i]>7
                        itmp=[i for i in range(len(tmp)) if tmp[i]>=7 and (i==0 or (i>0 and tmp[i-1]<tmp[i]))]
                        if len(itmp)>0:
                            cross_cryo.append(itmp[0])
                        else:
                            cross_cryo.append(-1)
                        # crossing from outside into all volumes indsie the uar volume (> ID 11), crossing when vol[j-1]<vol[i] and vol[i]>11
                        itmp=[i for i in range(len(tmp)) if tmp[i]>=11 and (i==0 or (i>0 and tmp[i-1]<tmp[i]))]
                        if len(itmp)>0:
                            cross_uar.append(itmp[0])
                        else:
                            cross_uar.append(-1)
                    
                    else:
                        #nsec.append(-1)
                        if (nC_A[i][index-1]) == 77 and nC_Z[i][index-1] == 32:
                            nsec_with_nC[-1]+=1

                    last_index=j

    file.close()
    # avoid truncation, works only when displaying data frame, not when printing to file
    np.set_printoptions(threshold=sys.maxsize)
    

    if option == "detailed":
        df = pd.DataFrame({'nidx':nidx_new,'tid':tid_new,'weight': ["{:.5e}".format(val[0]) for val in weight_new],'time[mus]':time_new,'x[m]':x_new,'y[m]': y_new,'z[m]': z_new,'xmom[m]': xmom_new,'ymom[m]': ymom_new,'zmom[m]': zmom_new,'ekin[eV]': ekin_new,'edep[eV]': edep_new,'vol': vol_new,'process': process_new,'nC_A': nC_A_new,'nC_Z': nC_Z_new,'cross_water': cross_water, 'cross_cryo': cross_cryo, 'cross_uar': cross_uar,'nsecondaries': nsec, 'nsecondaries_with_nC': nsec_with_nC})
    # print only parameters used for Ge77 rate calclulation
    else:
        df = pd.DataFrame({'nidx':nidx_new, 'tid':tid_new, 'weight': weight_new,'nC_A': nC_A_new, 'nC_Z': nC_Z_new, 'cross_cryo': cross_cryo})

    #print(f"{tracksize} maximal steps per track, {len(x_new)}")
    #print("runtime: ",datetime.now()-startTime,flush=True)
    return df

def get_volume_from_root(filename):
    file=uproot.open(filename)
    if len(file.keys())==0:
        file.close()
        print(f"WARNING: Empty file {filename}", flush=True)
        sys.exit(1)
    keyname='TurbineAndTube'
    tree_found=False
    for k in file.keys():
        if keyname in k:
            tree_found=True
            break
    if tree_found==False:
        print(f"WARNING: No design parameter info in file {filename}", flush=True)
        return 0
    # read in root TTree named TurbineAndTube
    tree = file[keyname]
    tree.keys()
    branches = tree.arrays(library='np')
    vol=np.round(branches['Volume_cm3'][0][0],2)

    return vol

def get_design_parameters_from_root(filename):
    file=uproot.open(filename)
    if len(file.keys())==0:
        file.close()
        print(f"WARNING: Empty file {filename}", flush=True)
        sys.exit(1)
    keyname='TurbineAndTube'
    params=[]
    tree_found=False
    for k in file.keys():
        if keyname in k:
            tree_found=True
            break
    if tree_found==False:
        print(f"WARNING: No design parameter info in file {filename}", flush=True)
        return params
    # read in root TTree named TurbineAndTube
    tree = file[keyname]
    tree.keys()
    branches = tree.arrays(library='np')
    params.append(branches['Design'][0][0])
    params.append(branches['Radius_cm'][0][0])
    params.append(branches['Thickness_cm'][0][0])
    params.append(np.round(branches['NPanels'][0][0],0))
    params.append(np.round(branches['Phi_deg'][0][0],2))
    params.append(branches['Theta_deg'][0][0])
    params.append(branches['Length_cm'][0][0])
    params.append(branches['Height_cm'][0][0])
    params.append(branches['ZPosition_cm'][0][0])
    params.append(np.round(branches['Volume_cm3'][0][0],2))

    return params

def get_design_parameters_from_csv(filename,idx):
    params=[]
    dfsamples=pd.read_csv(filename)
    params.append(dfsamples.at[idx,'design'])
    params.append(dfsamples.at[idx,'radius'])
    params.append(dfsamples.at[idx,'thickness'])
    params.append(dfsamples.at[idx,'npanels'])
    params.append(dfsamples.at[idx,'phi'])
    params.append(dfsamples.at[idx,'theta'])
    params.append(dfsamples.at[idx,'length'])
    params.append(dfsamples.at[idx,'height'])
    params.append(dfsamples.at[idx,'zpos'])
    params.append(0.0)

    return params

def main(filename, start, end, mode, params, events): # python file
#def main(filename, start=0, end=0, mode='LF', params=""): # jupyter notebook
    filenames=[]
    if end>0:
        for k in range(start,end):
            name=f'{filename}_{k:04d}.root'
            filenames.extend(get_all_files(name))
    else:
       filenames.extend(get_all_files(filename))
       print(filenames)

    for file in filenames:
        #print(f"Processing {file}", flush=True)
        filename_out = file[:file.find('.root')]
        filename_out = filename_out.replace("tier0", "tier1")
        
        if not params:
            tmp = filename_out.replace("/root/", "/csv/tier0/")
            #with open(f"{tmp}.csv") as f:
            #    params = f.readline().strip('\n')
            params=f"{get_design_parameters_from_root(file)}".replace(' ', '')

        df_out = get_neutron_info(file)
        filename_out = filename_out.replace("/root/", "/csv/tier1/")
        try:
            if len(df_out) > 0:
                pass
        except TypeError:
            continue
        
        
        #print(file[:file.find('.root')],filename_out)
        try:
            os.remove(f"{filename_out}.csv")
        except FileNotFoundError:
            pass
        writing_mode='w'
        #print(f"Writing {filename_out}.csv", flush=True)
        if params:
            f = open(f"{filename_out}.csv", "w")#
            #f.write(params+"\n")
            f.write(f"# [{mode}, {', '.join(params[1:-1].split(','))}] # mode design r d npanels phi theta L H z V"+"\n")
            f.close()
            writing_mode='a'
        startTime = datetime.now()
        if len(df_out) < events:
            sys.exit(1)
        df_out.to_csv(f"{filename_out}.csv", mode=writing_mode)
        #print("writing file runtime: ",datetime.now() - startTime,flush=True)
        del df_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)
    parser.add_argument('--mode', default='LF')
    parser.add_argument('--events',type=int,default=0)
    parser.add_argument('--params', type=str, default="")
    args = parser.parse_args()

    main(args.filename, args.start, args.end, args.mode, args.params, args.events)