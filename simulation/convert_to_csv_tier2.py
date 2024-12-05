import numpy as np
import pandas as pd
from NeutronSimulation import NeutronSimulation as sim
import argparse
import os
import sys

def main(filenames, path_to_files, fidelity, nprimaries, fileout): # python file
    r_water = 11./2.
    frac_enter_exp = 0.27
    rate_muon = 3.6 * 1.e-4 * (np.pi*r_water**2) * (60.*60.*24.*365.)
    frac_neutrons_LAr = 1./0.13
    exposure = 1000.


    if fidelity == "HF":
            fidelity = 1
            frac_neutrons_LAr = 1.

    if fidelity =="LF":
            fidelity = 0
    
    for i in range(len(filenames)):
        print(filenames[i])
        filenames[i] = filenames[i].replace(".root", ".csv")
        filenames[i] = filenames[i].replace("root", "csv/tier1")
        filenames[i] = filenames[i].replace("tier0", "tier1")
        if path_to_files != "":
            filenames[i]='{}/{}'.format(path_to_files,filenames[i])

    if len(filenames)==1:
        #filenames[0] = filenames[0].replace(".csv", "")
        sim_tmp=sim(filenames[0])
    else:
        sim_tmp=sim()
        sim_tmp.set_files(filenames)

    if len(sim_tmp.files)==0:
        print("Error: No files found")
        return
        sys.exit(1)

    nprimaries = len(sim_tmp.files) * nprimaries
    [is_captured,time_0,x_0,y_0,z_0,r_0,L_0,px_0,py_0,pz_0,ekin_0,edep_0,lnE0vsET,vol_0,nCZ,nCA,nsec,nsec_with_nC,weight,total_nC_Ge77,nsteps,time_t,x_t,y_t,z_t,r_t,L_t,px_t,py_t,pz_t,ekin_t,edep_t,_,vol_t,_,_,_,_,_,_,_]=sim_tmp.get(["time[mus]","x[m]","y[m]","z[m]","r[m]","L[m]","xmom[m]","ymom[m]","zmom[m]","ekin[eV]","edep[eV]","ln(E0vsET)","vol","nC_Z","nC_A","nsecondaries","nsecondaries_with_nC","weight","total_nC_Ge77[cts]","nsteps"],"summary")
    
    sum_nsec_weighted=np.sum([i*j for i,j in zip(nsec,weight)])
    factor = frac_neutrons_LAr * frac_enter_exp / rate_muon * exposure
    scaling_factor = [i/((1+sum_nsec_weighted)*factor) for i in weight]
    
    prod_rate_Ge77 = [i*j for i,j in zip(total_nC_Ge77,scaling_factor)]
    
    if fidelity == 0:
        [fidelity, radius, thickness, npanels, _, theta, length, height, z_offset, volume] = sim_tmp.get_design()
    else:
        infile = open(sim_tmp.files[0], 'r')
        design = infile.readline().split("]")[0].split("[")[1]
        [_, _,radius, thickness, npanels, _, theta, length, height, z_offset, volume] = design.split(",")
    #primaries = np.zeros(len(is_captured))
    #primaries[0] = nprimaries
    
    df = pd.DataFrame({'fidelity': fidelity, 'radius': radius, 'thickness': thickness,'npanels': npanels, 'theta': theta, 'length': length, 'height': height, 'z_offset': z_offset, 'volume': volume,'nC_Ge77': is_captured,'prod_rate_Ge77[nuc/(kg*yr)]': prod_rate_Ge77,'total_nC_Ge77[cts]': total_nC_Ge77,'nC_Ge77_scaling': scaling_factor,'time_0[ms]':time_0,'x_0[m]':x_0,'y_0[m]': y_0,'z_0[m]': z_0,'r_0[m]': r_0,'L_0[m]': L_0,'px_0[m]': px_0,'py_0[m]': py_0,'pz_0[m]': pz_0,'ekin_0[eV]': ekin_0,'edep_0[eV]': edep_0,'vol_0': vol_0,'time_t[ms]':time_t,'x_t[m]':x_t,'y_t[m]': y_t,'z_t[m]': z_t,'r_t[m]': r_t,'L_t[m]': L_t,'px_t[m]': px_t,'py_t[m]': py_t,'pz_t[m]': pz_t,'ekin_t[eV]': ekin_t,'edep_t[eV]': edep_t,'ln(E0vsET)': lnE0vsET,'vol_t': vol_t,'nsteps': nsteps,'nC_A': nCA,'nC_Z': nCZ, 'nsec': nsec, 'nsec_with_nC': nsec_with_nC, 'nprimaries': nprimaries/len(nsec), 'weights': weight})
    if fileout=="":
        fileout = sim_tmp.files[0].replace("tier1", "tier2")
    #print("path to",path_to_files,"path out",path_out)
    if fidelity == "LF" and nprimaries != len(total_nC_Ge77):
        print("Error: number of events doesn't equal number of primaries")
        return

    df.to_csv(fileout)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenames', nargs='+', type=str, default=["neutron-sim-LF"],help="List of filenames to process")
    parser.add_argument('--path_to_files', type=str, default=".")
    parser.add_argument('--filename_out', type=str, default="")
    parser.add_argument('--nprimaries', type=int, default=50000)
    parser.add_argument('--fidelity', type=str, default="LF")
    args = parser.parse_args()

    main(args.filenames, args.path_to_files, args.fidelity, args.nprimaries, args.filename_out)