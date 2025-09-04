from I3Tray import *
from icecube import icetray, dataio, dataclasses
import sys, math
import copy
import glob
from scipy.optimize import curve_fit

import os
from icecube.dataclasses import I3Double, I3Particle, I3Direction, I3Position, I3VectorI3Particle, I3Constants, I3VectorOMKey
from icecube.simclasses import I3MMCTrack
from icecube import MuonGun, simclasses,millipede
import numpy as np
import pandas as pd

from icecube.hdfwriter import I3HDFTableService, I3HDFWriter

def add_glashow(tray, name):
    def glashow_correction(frame):
         if not frame.Has('MuonWeight'):
            #print('start adding glashow correction')
            from scipy.interpolate import interp1d
            nutype=frame["I3MCWeightDict"]["PrimaryNeutrinoType"]
            inter_type=frame['I3MCWeightDict']['InteractionType']
            en = frame['I3MCWeightDict']['PrimaryNeutrinoEnergy']
            if (abs(nutype)==12 and inter_type==3.0 and en>4e6):
                old_spline=pd.read_csv('/home/abalagopalv/diffuse/TauStudies/Glashow_old.csv',header=None)
                new_spline=pd.read_csv('/home/abalagopalv/diffuse/TauStudies/Glashow_new.csv',header=None)
        
                x = old_spline[0]
                y = old_spline[1]
        
                xn = new_spline[0]
                yn = new_spline[1]
                f1 = interp1d(x, y, kind='cubic')
                f2 = interp1d(xn, yn, kind='cubic')
                if en<9.9e6:
                    
                    num = f2(en/1e6)
                    denom = f1(en/1e6)
                    ratio = num/denom
                    frame['TotalWeight_glashowcorrection'] = dataclasses.I3Double(frame['I3MCWeightDict']['TotalWeight']*ratio)
                elif en>=9.9e6:
                    num = f2(9.89)
                    denom = f1(9.89)
                    ratio = num/denom
                    frame['TotalWeight_glashowcorrection'] = dataclasses.I3Double(frame['I3MCWeightDict']['TotalWeight']*ratio)
                #print(ratio)
            else:
                frame['TotalWeight_glashowcorrection'] = dataclasses.I3Double(frame['I3MCWeightDict']['TotalWeight'])
    print('Successfully added glashow correction')
    tray.Add(glashow_correction)