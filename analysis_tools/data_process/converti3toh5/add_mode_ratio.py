from icecube import icetray, dataio, dataclasses
import math as m
import numpy as np
from I3Tray import *
from icecube.icetray import traysegment, load, I3Units, module_altconfig

def add_tau_decay_mode(tray,name):
    tray.AddModule(add_tau_mode,name=name)

def add_tau_mode(frame,name):
    interactiontype = frame['I3MCWeightDict']['InteractionType']
    if interactiontype == 1:
        neutrinos,vertices = [],[]
        if frame.Has('I3MCTree'):
            tree = frame['I3MCTree']
            #select all neutrinos that have daughters (otherwise lenght == nan)
            for p in tree:
                if p.is_neutrino == True and p.location_type == dataclasses.I3Particle.LocationType.InIce and not m.isnan(p.length) and p.length>0:
                    neutrinos.append(p)
                    print ("appending...")
            if len(neutrinos)==0:
                print ("no neutrino")
                return True
            else:
                for p in neutrinos:
                    print("houston we have a neutrino")
                    sec = tree.get_daughters(p)[0] #all daughters produced at the same position
                    pos = sec.pos
                    dist = m.sqrt(m.pow(pos.x,2)+m.pow(pos.y,2)+m.pow(pos.z,2)) # distance of interaction to center of detector
                    vertices.append(dist)
                mom_nu = neutrinos[int(np.argmin(vertices))] # want to have the last nu that interacted -> the one that created shower
                print ("momma nu", mom_nu)
                nu_daught = tree.get_daughters(mom_nu)
                print ("die mutter hat die toechter: ", nu_daught)
                mctype = mom_nu.type
                print ("die mutter ist: ", mctype)
                if mctype == dataclasses.I3Particle.NuTauBar or mctype == dataclasses.I3Particle.NuTau:
                    for p in nu_daught:
                        if p.type == dataclasses.I3Particle.TauPlus or p.type == dataclasses.I3Particle.TauMinus:
                            if frame.Has("TrueTau"):
                                pass
                            else:
                                frame.Put("TrueTau",dataclasses.I3Particle(p))
                            tert = tree.get_daughters(p)
                            max_tau_time = 0
                            for tau_daughter in tert:
                                if tau_daughter.type == dataclasses.I3Particle.TauPlus or tau_daughter.type == dataclasses.I3Particle.TauMinus:
                                    print('Tau daughter type is ',tau_daughter.type)
                                    if tau_daughter.time > max_tau_time:
                                        print('Find a tau, energy is ', tau_daughter.energy)
                                        max_tau_time = tau_daughter.time
                                        E_tau = tau_daughter.energy
                                else:
                                    pass
                                if tau_daughter.type == dataclasses.I3Particle.NuTau or tau_daughter.type == dataclasses.I3Particle.NuTauBar:
                                    E_secondarynutau = tau_daughter.energy
                                    frame.Put("Energy_tau_decaynutau",dataclasses.I3Double(E_secondarynutau))
                                else:
                                    pass
                            if E_tau >0:
                                pass
                            else:
                                E_tau = p.energy
                            print('Tau lepton energy is ', E_tau)
                            E_tau_decay_product = 0
                            if frame.Has('NuTaudecaytype'):
                                if frame['NuTaudecaytype'].value==1:
                                    modes =  [dataclasses.I3Particle.Hadrons,dataclasses.I3Particle.PiPlus,dataclasses.I3Particle.PiMinus,dataclasses.I3Particle.Pi0]
                                    N_pions = 0
                                    for t in tert:
                                        if t.type in modes:
                                            N_pions = N_pions +1
                                            E_tau_decay_product = E_tau_decay_product + t.energy
                                    frame.Put("Number_tau_decaypions",dataclasses.I3Double(N_pions))
                                    N_pi0 = 0
                                    for t in tert:
                                        if t.type in [dataclasses.I3Particle.Pi0]:
                                            N_pi0 = N_pi0 +1
                                    frame.Put("Number_tau_decaypi0",dataclasses.I3Double(N_pi0))
                                elif frame['NuTaudecaytype'].value==2:
                                    modes = [dataclasses.I3Particle.EPlus,dataclasses.I3Particle.EMinus]
                                    for t in tert:
                                        if t.type in modes:
                                            E_tau_decay_product = E_tau_decay_product + t.energy
                                elif frame['NuTaudecaytype'].value==3:
                                    modes = [dataclasses.I3Particle.MuPlus,dataclasses.I3Particle.MuMinus]
                                    for t in tert:
                                        if t.type in modes:
                                            E_tau_decay_product = E_tau_decay_product + t.energy
                            else:
                                pass
                            print('Tau decay product energy is ', E_tau_decay_product)
                            if frame.Has('Energy_tau_lepton'):
                                pass
                            else:
                                frame.Put("Energy_tau_lepton",dataclasses.I3Double(E_tau))
                                frame.Put("Energy_tau_decayproduct",dataclasses.I3Double(E_tau_decay_product))
    if frame.Has('Energy_tau_lepton'):
        pass
    else:
        frame.Put("Energy_tau_lepton",dataclasses.I3Double(0))
        frame.Put("Energy_tau_decayproduct",dataclasses.I3Double(0))
    if frame.Has('Number_tau_decaypions'):
        pass
    else:
        frame.Put("Number_tau_decaypions",dataclasses.I3Double(0))
        frame.Put("Number_tau_decaypi0",dataclasses.I3Double(0))
    if frame.Has('Energy_tau_decaynutau'):
        pass
    else:
        frame.Put("Energy_tau_decaynutau",dataclasses.I3Double(0))