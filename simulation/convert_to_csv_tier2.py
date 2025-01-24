import numpy as np
import pandas as pd
from NeutronSimulation import NeutronSimulation as sim
import argparse
import os

def main(filename_base, path_to_files, tier, fidelity, start, end, nprimaries): # python file
    r_water = 11./2.
    frac_enter_exp = 0.27
    rate_muon = 3.6 * 1.e-4 * (np.pi*r_water**2) * (60.*60.*24.*365)
    frac_neutrons_LAr = 1./0.13
    exposure = 1000.
    
    for m in range(start, end):
        if fidelity == "HF":
            fidelity = 1
            filename=f'{path_to_files}/{filename_base}-{tier}_{m:04d}'
            frac_neutrons_LAr = 1.
        else:
            filename=f'{path_to_files}/{filename_base}-{m:04d}-{tier}_'
        sim_tmp=sim(filename)
        if fidelity =="LF":
            fidelity = 0
            nfiles_expected = int(nprimaries/25000)
            if len(sim_tmp.files) != nfiles_expected:
                continue

        if len(sim_tmp.files)==0:
            continue

        [is_captured,time_0,x_0,y_0,z_0,r_0,L_0,px_0,py_0,pz_0,ekin_0,edep_0,lnE0vsET,vol_0,nCZ,nCA,nsec,nsec_with_nC,total_nC_Ge77,nsteps,time_t,x_t,y_t,z_t,r_t,L_t,px_t,py_t,pz_t,ekin_t,edep_t,_,vol_t,_,_,_,_,_,_]=sim_tmp.get(["time[mus]","x[m]","y[m]","z[m]","r[m]","L[m]","xmom[m]","ymom[m]","zmom[m]","ekin[eV]","edep[eV]","ln(E0vsET)","vol","nC_Z","nC_A","nsecondaries","nsecondaries_with_nC","total_nC_Ge77[cts]","nsteps"],"summary")
        prod_rate_Ge77 = total_nC_Ge77*rate_muon*frac_enter_exp*r_water*exposure*frac_neutrons_LAr
        if fidelity == 0:
            [fidelity, radius, thickness, npanels, _, theta, length, height, z_offset, volume] = sim_tmp.get_design()
        else:
            infile = open(sim_tmp.files[0], 'r')
            design = infile.readline().split("]")[0].split("[")[1]
            [_, _,radius, thickness, npanels, _, theta, length, height, z_offset, volume] = design.split(",")
        primaries = np.zeros(len(is_captured))
        primaries[0] = nprimaries
        df = pd.DataFrame({'fidelity': fidelity, 'radius': radius, 'thickness': thickness,'npanels': npanels, 'theta': theta, 'length': length, 'height': height, 'z_offset': z_offset, 'volume': volume,'nC_Ge77': is_captured,'prod_rate_Ge77[nuc/(kg*yr)]': prod_rate_Ge77,'total_nC_Ge77[cts]': total_nC_Ge77,'time_0[ms]':time_0,'x_0[m]':x_0,'y_0[m]': y_0,'z_0[m]': z_0,'r_0[m]': r_0,'L_0[m]': L_0,'px_0[m]': px_0,'py_0[m]': py_0,'pz_0[m]': pz_0,'ekin_0[eV]': ekin_0,'edep_0[eV]': edep_0,'vol_0': vol_0,'time_t[ms]':time_t,'x_t[m]':x_t,'y_t[m]': y_t,'z_t[m]': z_t,'r_t[m]': r_t,'L_t[m]': L_t,'px_t[m]': px_t,'py_t[m]': py_t,'pz_t[m]': pz_t,'ekin_t[eV]': ekin_t,'edep_t[eV]': edep_t,'ln(E0vsET)': lnE0vsET,'vol_t': vol_t,'nsteps': nsteps,'nC_A': nCA,'nC_Z': nCZ, 'nsec': nsec, 'nsec_with_nC': nsec_with_nC, 'nprimaries': primaries})
    
        path_out = path_to_files.replace(f"{tier}", "tier2")
        print("path to",path_to_files,"path out",path_out)
        if fidelity == "LF" and nprimaries != len(total_nC_Ge77):
            print("Error: number of events doesn't equal number of primaries")
            continue

        df.to_csv(f'{path_out}/{filename_base}-{m:04d}-tier2.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="neutron-sim-LF")
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=300)
    parser.add_argument('--path_to_files', type=str, default=".")
    parser.add_argument('--tier', type=str, default="tier1")
    parser.add_argument('--nprimaries', type=int, default=50000)
    parser.add_argument('--fidelity', type=str, default="LF")
    args = parser.parse_args()

    main(args.filename, args.path_to_files, args.tier, args.fidelity, args.start, args.end, args.nprimaries)