import sys
import os
BASE_DIR = '/data/user/zchen/analysis_tools/'
sys.path.append(str(BASE_DIR))


from I3Tray import I3Tray
from icecube import dataio, dataclasses
from analysis_tools.icetray_segments.decay_mode_selector import select_decay_mode
from analysis_tools.icetray_segments.decay_mode_selector import TAU_DECAY_MODES, TAU_DECAYTYPE_MAP 
from analysis_tools.icetray_segments.tau_decay_segment import TauDecayClassifier

'''
def test_module(frame):
    if not frame.Has("I3MCTree"):
        return

    tree = frame["I3MCTree"]
    result = select_decay_mode(
        frame=frame,
        tree=tree,
        mother_type=dataclasses.I3Particle.NuTau,
        decay_modes=TAU_DECAY_MODES,
        decaytype_map=TAU_DECAYTYPE_MAP,
        key_prefix="NuTau"
    )

    if result:
        print(f"✅ Event {frame['I3EventHeader'].event_id} → Matched mode:",
              frame["NuTauDecayLabel"].value)
    else:
        print(f"❌ Event {frame['I3EventHeader'].event_id} → No match.")

'''

indir = "/data/user/zchen/analysis_tools/analysis_tools/tests/file/input/"
tray = I3Tray()
tray.AddModule("I3Reader", "reader", Filename=f"{indir}/Finallevel_NuTau_NuGenCCNC.022692.000999.i3.zst")  # << Replace with a real test file
tray.Add(TauDecayClassifier,'tau')
tray.Execute()