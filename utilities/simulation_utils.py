import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import sys
sys.path.append('../utilities')
import utilities as utils

def overwrite_first_line(files_base_name,first_line_new,ending='.csv'):
    files=utils.get_all_files(files_base_name, ending)

    for file in tqdm(files):
        print(file)
        #first_line_new = "# [HF, 3, 95, 10.0, 360, 0.0, 0.0, 1.7, 300.0, 42.0, 5620309.26] # mode design r d npanels phi theta L H z V"
        #first_line_new = "# [HF, 3, 200, 10.0, 360, 0.0, 0.0, 3.5, 300.0, 42.0, 13998936.86] # mode design r d npanels phi theta L H z V"

        df_in = pd.read_csv(file,skiprows=1)
        df_in = df_in.loc[:, ~df_in.columns.str.contains('^Unnamed')]
        df_in
        f = open(file, "w")#
        f.write(f"{first_line_new}"+"\n")
        f.close()
        writing_mode='a'
        df_in.to_csv(file, mode=writing_mode)

def print_geant4_macro(x, idx, mode='LF', version='v2'):

    path_to_macros=f"out/{mode}/{version}/macros/"
    if (os.path.exists(path_to_macros)==False):
        os.makedirs(path_to_macros)
    f = open(f"{path_to_macros}/neutron-sim-{mode}-{version}-{idx}.mac", "w")
    f.write("# minimal command set test"+ "\n")
    f.write("# verbose"+ "\n")
    f.write("#/random/setSeeds 9530 7367"+"\n"+"\n")

    f.write("/run/verbose 0"+"\n")
    f.write("/tracking/verbose 0"+"\n")
    f.write("/run/setCut 3.0 cm"+"\n")
    f.write("/tracking/verbose 0"+"\n")
    f.write("#/random/setSeeds 9530 7367"+"\n")
    f.write("/run/verbose 0"+"\n")
    f.write("/tracking/verbose 0"+"\n")
    f.write("/run/setCut 3.0 cm"+"\n")
    f.write("/tracking/verbose 0"+"\n"+"\n")

    f.write("/WLGD/detector/setGeometry baseline_large_reentrance_tube"+"\n")
    f.write("/WLGD/detector/Cryostat_Radius_Outer 325"+"\n")
    f.write("/WLGD/detector/Cryostat_Height 325"+"\n")
    f.write("#/WLGD/detector/Cryostat_Vacgap 50"+"\n")
    f.write("/WLGD/detector/With_Gd_Water 1"+"\n"+"\n")

    f.write("/WLGD/detector/With_NeutronModerators 4 # Design 4 (with lids)"+"\n")
    f.write(f"/WLGD/detector/TurbineAndTube_Radius {round(x[0],1)} cm"+"\n")
    f.write(f"/WLGD/detector/TurbineAndTube_Width {round(x[1],1)} cm"+"\n")
    f.write(f"/WLGD/detector/TurbineAndTube_NPanels {round(x[2],0)}"+"\n")
    f.write(f"/WLGD/detector/TurbineAndTube_Angle {round(x[3],1)}"+"\n")
    f.write(f"/WLGD/detector/TurbineAndTube_Length {round(x[4],1)} cm"+"\n")
    f.write("/WLGD/detector/TurbineAndTube_Height 300 cm"+"\n")
    f.write("/WLGD/detector/TurbineAndTube_zPosition 42 cm"+"\n")
    f.write("/WLGD/detector/Which_Material PMMA"+"\n"+"\n")

    f.write("/WLGD/event/saveAllEvents 1"+"\n")
    f.write("#/WLGD/event/saveAllProductions 1"+"\n"+"\n")

    f.write("#Init"+"\n")
    f.write("/run/initialize"+"\n"+"\n")

    f.write("#Idle state"+"\n")
    f.write("#/WLGD/runaction/WriteOutNeutronProductionInfo 1"+"\n")
    f.write("#/WLGD/runaction/WriteOutGeneralNeutronInfo 1"+"\n")
    f.write("/WLGD/runaction/WriteOutAllNeutronInfoRoot 0"+"\n")
    f.write("#/WLGD/runaction/getIndividualGeDepositionInfo 1"+"\n")
    f.write("#/WLGD/runaction/getIndividualGdDepositionInfo 1"+"\n")
    f.write("/WLGD/runaction/getIndividualNeutronStepInfo 1"+"\n"+"\n")

    f.write("/WLGD/step/getDepositionInfo 1"+"\n")
    f.write("/WLGD/step/getIndividualDepositionInfo 1"+"\n")
    if mode=='LF':
        f.write("/WLGD/generator/getReadInSeed 0"+"\n")
        f.write("/WLGD/generator/setGenerator NeutronsFromFile                    # set the primary generator to the (Alpha,n) generator in the moderators (options: \"MeiAndHume\", \"Musun\", \"Ge77m\", \"Ge77andGe77m\", \"ModeratorNeutrons\" = generate neutrons inside the neutron moderators, \"ExternalNeutrons\" (generate neutrons from outside the water tank)))"+"\n")
        f.write("/WLGD/generator/setMUSUNFile ${MUSUN_DIR}/neutron-inputs-design0_25k_${RUN_NUMBER}_${VERSION_IN}.dat"+"\n"+"\n")
    elif mode=='HF':
        f.write("/WLGD/generator/setGenerator Musun     # set the primary generator"+"\n")
        f.write("/WLGD/generator/setMUSUNFile ${MUSUN_DIR}/musun_gs_50k_${RUN_NUMBER}.dat"+"\n"+"\n")
    f.write("# start"+"\n")
    f.write("/run/beamOn ${EVENTS}"+"\n")
    f.close()