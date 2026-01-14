from icecube import icetray, dataio,dataclasses,simclasses,recclasses,gulliver,millipede
from icecube import dataio, dataclasses,icetray
from icecube.icetray import OMKey
from icecube import tableio, hdfwriter
from I3Tray import *
from icecube.hdfwriter import I3HDFWriter
import pandas as pd
import numpy as np
from .add_glashow import add_glashow
from .add_mode_ratio import add_tau_decay_mode

def penetrating_depth(frame, gcd, depth_name_suffix=''):
    ##  add penetrating depth dependence to the self veto probability calculation

    from icecube import MuonGun
    p = frame['cscdSBU_MCPrimary']

    # previously used MuonGun.Cylinder(1000, 500) surface, which could result in a few events having a "nan" surface.intersection
    surface = MuonGun.Cylinder(1000, 500)
    d = surface.intersection(p.pos, p.dir)
    getDepth=p.pos + d.first*p.dir
    impactDepth = MuonGun.depth((getDepth).z)*1.e3
    frame["penetrating_depth"+depth_name_suffix+"_old"] = dataclasses.I3Double(impactDepth)
    surface_hex = MuonGun.ExtrudedPolygon.from_file(gcd)
    d = surface_hex.intersection(p.pos, p.dir)
    getDepth=p.pos + d.first*p.dir
    impactDepth = MuonGun.depth((getDepth).z)*1.e3
    frame["penetrating_depth"+depth_name_suffix] = dataclasses.I3Double(impactDepth)
    print('Finishing adding penetrating depth')

#Muongun functions
@icetray.traysegment
def get_muongun_weight(tray,name):
        def put_muon(frame):
                if not 'Muon' in frame:
                    tree = frame['I3MCTree']
                    muons = [p for p in tree if (p.type == dataclasses.I3Particle.MuMinus or p.type == dataclasses.I3Particle.MuPlus)]
                    maxmuon = muons[np.argmax([p.energy for p in muons])] # get most energetic one
                    frame['cscdSBU_MCMuon'] = maxmuon

        tray.AddModule(put_muon)
        from icecube import MuonGun

        def harvest_generator(infile):
                generator = None
                fname = infile
                f = dataio.I3File(fname)
                fr = f.pop_frame(icetray.I3Frame.Stream('S'))
                for k in fr.keys():
                    if not k == 'InIceErrataKeys':
                        v = fr[k]
                        if isinstance(v, MuonGun.GenerationProbability):
                            generator = v

                f.close()
                return generator

        infiles_21315 = "/data/sim/IceCube/2016/filtered/level2/MuonGun/21315/0000000-0000999/Level2_IC86.2016_MuonGun.021315.000000.i3.zst"
        infiles_21316 = "/data/sim/IceCube/2016/filtered/level2/MuonGun/21316/0000000-0000999/Level2_IC86.2016_MuonGun.021316.000000.i3.zst"
        infiles_21317 = "/data/sim/IceCube/2016/filtered/level2/MuonGun/21317/0000000-0000999/Level2_IC86.2016_MuonGun.021317.000000.i3.zst"
        infiles_21318 = "/data/sim/IceCube/2016/filtered/level2/MuonGun/21318/0000000-0000999/Level2_IC86.2016_MuonGun.021318.000000.i3.zst"
        infiles_21319 = "/data/sim/IceCube/2016/filtered/level2/MuonGun/21319/0000000-0000999/Level2_IC86.2016_MuonGun.021319.000000.i3.zst"
        infiles_22358 = "/data/sim/IceCube/2016/filtered/level2/MuonGun/22358/0000000-0000999/Level2_IC86.2016_MuonGun.022358.000000.i3.zst"
        infiles_22359 = "/data/sim/IceCube/2016/filtered/level2/MuonGun/22359/0000000-0000999/Level2_IC86.2016_MuonGun.022359.000000.i3.zst"

        generator_21315 = harvest_generator(infiles_21315)*15000
        generator_21316 = harvest_generator(infiles_21316)*39995
        generator_21317 = harvest_generator(infiles_21317)*19994
        generator_21318 = harvest_generator(infiles_21318)*99975
        generator_21319 = harvest_generator(infiles_21319)*99636
        generator_22358 = harvest_generator(infiles_22358)*10000
        generator_22359 = harvest_generator(infiles_22359)*9999

        generator=generator_21315+generator_21316+generator_21317+generator_21318+generator_21319+generator_22358+generator_22359

        tray.AddModule('I3MuonGun::WeightCalculatorModule', 'cscdSBU_MuonWeight_GaisserH4a',
                Model=MuonGun.load_model('GaisserH4a_atmod12_SIBYLL'),
                Generator=generator)
        tray.AddModule('I3MuonGun::WeightCalculatorModule', 'cscdSBU_MuonWeight_DPMC',
                Model=MuonGun.load_model('GaisserH4a_atmod12_DPMJET-C'),
                Generator=generator)
    
def get_weight_sum(frame):
        frame["cscdSBU_MuonWeight_sum"] = dataclasses.I3Double(frame["cscdSBU_MuonWeight_DPMC"].value+frame["cscdSBU_MuonWeight_GaisserH4a"].value)
        return True




def converti3toh5(infile,outfile,ismgun,isdata):
    print('processing i3 to h5 convert')
    key1 = ['cscdSBU_MonopodFit4_noDC','cscdSBU_MonopodFit4_noDCFitParams','cscdSBU_MonopodFit4_noDC_Delay_ice','cscdSBU_MaxQtotRatio_SplitInIcePulses','cscdSBU_MCTruth','I3EventHeader','I3MCWeightDict','cscdSBU_AtmWeight_Conv','cscdSBU_AtmWeight_Prompt','cscdSBU_AtmWeight_Prompt_PassRate','cscdSBU_AtmWeight_Conv_PassRate', 'cscdSBU_Qtot_HLC', 'CscdL3_CascadeLlhVertexFit','cscdSBU_MaxQtotRatio_HLC','cscdSBU_VetoDepthFirstHit']
    tau_keys = ['Cascade1_vis_truth_tau','Cascade2_vis_truth_tau','MCTruth_Cascade_Distance','MCTruth_Tau_Asymmetry','TrueTau','cc']
    key1.extend(tau_keys)
    taupede_keys = ['Taupede1_spice3Particles','Taupede2_spice3Particles','Taupede_Distance','Taupede_Asymmetry','TauMonoDiff_rlogl','NuTaudecaytype']
    key1.extend(taupede_keys)
    add_keys = ['CVMultiplicity','CVStatistics','cscdSBU_VertexRecoDist_CscdLLh']
    key1.extend(add_keys)
    #not necessarily snowstorm unique, but added after switching to snowstorm
    depth_name_suffix = '_v1_gcd'
    snowstorm_keys = ['SnowstormParameterDict','PolyplopiaPrimary','cscdSBU_AtmWeight_Prompt_berss','cscdSBU_L4StartingTrackHLC_cscdSBU_MonopodFit4_OfflinePulsesHLC_noDCVetoCharge','cscdSBU_LE_bdt_cascade','cscdSBU_LE_bdt_hybrid','cscdSBU_LE_bdt_track','cscdSBU_LE_bdt_input','cscdSBU_MCPrimary','cscdSBU_VetoMaxDomChargeOM',"penetrating_depth"+depth_name_suffix]
    key1.extend(snowstorm_keys)
    additional_treatments_keys = ['TotalWeight_glashowcorrection','Energy_tau_lepton','Energy_tau_decayproduct','Number_tau_decaypions','Number_tau_decaypi0','Energy_tau_decaynutau','Charmtype','TrueCharm']
    key1.extend(additional_treatments_keys)
    Taupede_name1 = 'Taupede_newmonoseed'
    Taupede_name2 = 'New_Taupede'
    Monopod_name = 'MonopodFit_iMIGRAD_PPB0'
    newreco_keys = [Monopod_name,
                    Monopod_name+'AmpSeed',
                    Monopod_name+'AmpSeedFitParams',
                    Monopod_name+'FitParams',
                    Monopod_name+'_Seed',
                    Monopod_name+'_VertexRecoDist_CscdLLh',  
                    Taupede_name1,
                    Taupede_name1+'FitParams',
                    Taupede_name1+'Particles',
                    Taupede_name1+'_1Particles',
                    Taupede_name1+'_2Particles',
                    Taupede_name1+'_Distance',
                    Taupede_name1+'_Asymmetry',
                    Taupede_name1+'MonoDiff_rlogl',
                    Taupede_name2,
                    Taupede_name2+'FitParams',
                    Taupede_name2+'Particles',
                    Taupede_name2+'_1Particles',
                    Taupede_name2+'_2Particles',
                    Taupede_name2+'_Distance',
                    Taupede_name2+'_Asymmetry',
                    Taupede_name2+'MonoDiff_rlogl',
                    'NewMono_VertexRecoDist_CscdLLh']
    key1.extend(newreco_keys)
    Taupede_name_add_list = ['TaupedeFit_iMIGRAD_PPB0',
                             'TaupedeFit_iMIGRAD_PPB0_monoseed',
                             'TaupedeFit_iMIGRAD_testing',
                             'Taupede_lengthbound_200',
                             'Taupede_lengthbound_200_tianlu_step',
                             'Taupede_lengthbound_500',
                             'Taupede_tianlu_step',
                             'Taupede_ftp',
                             'Taupede_spice3',
                             ]
    for Taupede_name3 in Taupede_name_add_list:
        testing_keys = [Taupede_name3,
                        Taupede_name3+'FitParams',
                        Taupede_name3+'Particles',
                        Taupede_name3+'_1Particles',
                        Taupede_name3+'_2Particles',
                        Taupede_name3+'_Distance',
                        Taupede_name3+'_Asymmetry',
                        Taupede_name3+'MonoDiff_rlogl']
        key1.extend(testing_keys)
    #pmap_name = 'InIcePulses'
    print('Running i3toh5convert')
    try:
        tray = I3Tray()
        #print('Reading i3 file ',infile)
        tray.AddModule('I3Reader',
                        filename = infile)
        print('Running I3 Reader for ',infile)
        #gcd_file = '/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz'
        #import sys
        #sys.path.append("/home/zchen/taupede/new_reconstruct/code/")
        #from add_variables import Add_calculated_variables
        #tray.AddSegment(Add_calculated_variables,'TaupedeFit_iMIGRAD_PPB0')
        gcd_file = '/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz'
        if ismgun:
            muon_keys = ['cscdSBU_MCMuon','cscdSBU_MuonWeight_GaisserH4a','cscdSBU_MuonWeight_DPMC','cscdSBU_MuonWeight_sum']
            key1.extend(muon_keys)
            tray.Add(get_muongun_weight,'get_muongun_weight')
            tray.Add(get_weight_sum,'get_weight_sum')
        elif isdata:
            print('Running convert for data')
        else:
            tray.AddModule(penetrating_depth, gcd=gcd_file,depth_name_suffix='_v1_gcd')
            #tray.AddSegment(add_glashow,'glashow')
            #tray.AddSegment(add_tau_decay_mode,"check_tau_decay")
            print('Running conversion for nugen')
        
        print('outfile name: ',outfile)

        tray.AddSegment(I3HDFWriter,
                        output = outfile,
                        keys = key1,
                        SubEventStreams = ['InIceSplit'],
                        )
        tray.Execute()
        tray.Finish()
    except:
        pass


    
