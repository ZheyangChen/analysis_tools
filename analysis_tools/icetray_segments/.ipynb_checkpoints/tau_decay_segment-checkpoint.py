from icecube import icetray, dataclasses, dataio
from analysis_tools.icetray_segments.decay_mode_selector import select_decay_mode
#from icecube.analysis_tools.mc_tree_utils import get_primary_nu_tau  

@icetray.traysegment
def TauDecayClassifier(tray, name):
    def classify(frame):
        if not frame.Has("I3MCTree") or not frame.Has("I3MCWeightDict"):
            return

        tree = frame["I3MCTree"]
        interaction_type = frame["I3MCWeightDict"]["InteractionType"]

        from analysis_tools.icetray_segments.decay_mode_selector import TAU_DECAYTYPE_MAP

        if interaction_type == 2:
            # NC interaction → label 0
            frame.Put("NuTauDecayTypeID", dataclasses.I3Double(0))
            return

        # Try CC: find ντ and classify its decay
        success = select_decay_mode(
            frame=frame,
            tree=tree,
            mother_type=dataclasses.I3Particle.NuTau,
            decay_modes={
                "tau_hadronic": [dataclasses.I3Particle.Hadrons, dataclasses.I3Particle.PiPlus, dataclasses.I3Particle.PiMinus],
                "tau_electronic": [dataclasses.I3Particle.EPlus, dataclasses.I3Particle.EMinus],
                "tau_muonic": [dataclasses.I3Particle.MuPlus, dataclasses.I3Particle.MuMinus],
            },
            decaytype_map={
                "tau_hadronic": 1,
                "tau_electronic": 2,
                "tau_muonic": 3,
            },
            key_prefix="NuTau"
        )

        if not success:
            # Found ντ but couldn't match its decay mode
            frame.Put("NuTauDecayTypeID", dataclasses.I3Double(4))

    tray.AddModule(classify, name + "_classify")