from icecube import dataclasses
import numpy as np

# ---- Individual Adders ----
def add_reco_energy_variables(frame, reco_key='MonopodFit'):  
    if reco_key in frame:
        particle = frame[reco_key]
        frame['RecoEnergy'] = dataclasses.I3Double(particle.energy)

def add_interaction_type_variable(frame, weightdict_key='I3MCWeightDict'):
    if weightdict_key in frame:
        interaction = frame[weightdict_key].get('InteractionType', None)
        if interaction is not None:
            frame['InteractionType'] = dataclasses.I3Double(interaction)

def add_primary_energy_variable(frame, weightdict_key='I3MCWeightDict'):
    if weightdict_key in frame:
        energy = frame[weightdict_key].get('PrimaryNeutrinoEnergy', None)
        if energy is not None:
            frame['PrimaryNeutrinoEnergy'] = dataclasses.I3Double(energy)



# ---- Master Adder ----
def add_all_variables(
    frame,
    add_reco_energy=True,
    add_interaction_type=True,
    add_primary_energy=True,
    reco_key='MonopodFit',
    weightdict_key='I3MCWeightDict'
):
    """
    Add all standard analysis variables to the I3Frame.
    Each category can be toggled.
    """
    if add_reco_energy:
        add_reco_energy_variables(frame, reco_key)

    if add_interaction_type:
        add_interaction_type_variable(frame, weightdict_key)

    if add_primary_energy:
        add_primary_energy_variable(frame, weightdict_key)

    # Add future flags here as needed
    return True