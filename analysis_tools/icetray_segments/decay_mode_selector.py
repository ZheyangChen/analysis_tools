from icecube import dataclasses

TAU_DECAY_MODES = {
    "tau_to_hadrons": [
        dataclasses.I3Particle.Hadrons,
        dataclasses.I3Particle.PiPlus,
        dataclasses.I3Particle.PiMinus,
    ],
    "tau_to_e": [
        dataclasses.I3Particle.EPlus,
        dataclasses.I3Particle.EMinus,
    ],
    "tau_to_mu": [
        dataclasses.I3Particle.MuPlus,
        dataclasses.I3Particle.MuMinus,
    ],
}


TAU_DECAYTYPE_MAP = {
    "nc": 0,
    "hadronic": 1,
    "electronic": 2,
    "muonic": 3,
    "other": 4,
}

def find_decay_mode(tree, mother_types, matcher_fn, require_inice=True):
    """
    Find a decay mode by matching a mother particle and its second-generation daughters.

    Parameters
    ----------
    tree : I3MCTree
        The MC truth tree.
    mother_types : list of I3Particle.ParticleType
        Types of the mother particle to search for (e.g. NuTau, DsMinus).
    matcher_fn : function
        A function that takes (mother_particle, daughters) and returns a label if matched.
    require_inice : bool
        Whether the mother particle must be in-ice.

    Returns
    -------
    tuple or None
        (mother, label) if matched, otherwise None
    """
    if not isinstance(mother_types, (list, tuple)):
        mother_types = [mother_types]

    for particle in tree:
        if particle.type in mother_types:
            if require_inice and particle.location_type != dataclasses.I3Particle.LocationType.InIce:
                continue

            daughters = list(tree.get_daughters(particle))
            label = matcher_fn(particle, daughters)

            if label is not None:
                return particle, label

    return None

def select_decay_mode(
    frame,
    tree,
    mother_types,
    decay_matchers,
    decaytype_map=None,
    key_prefix="Tau"
):
    """
    Selects the decay mode of a given mother particle (e.g. NuTau, charm) and
    stores relevant info in the frame.

    Parameters
    ----------
    frame : I3Frame
        Current physics frame.
    tree : I3MCTree
        MC truth tree from the frame.
    mother_types : list of I3Particle.ParticleType
        Type(s) of mother particle to search for.
    decay_matchers : dict[str, function]
        Dictionary of mode label → function(mother, daughters) → label or None.
    decaytype_map : dict[str, int], optional
        Mapping from mode label to decay type ID.
    key_prefix : str
        Prefix for frame keys.

    Returns
    -------
    bool
        True if matched, else False
    """
    for mode_label, matcher_fn in decay_matchers.items():
        result = find_decay_mode(tree, mother_types, matcher_fn)
        if result is not None:
            mother, matched_label = result

            # Save truth particle
            frame.Put(f"{key_prefix}TruthParticle", dataclasses.I3Particle(mother))
            frame.Put(f"{key_prefix}DecayLabel", dataclasses.I3String(matched_label))

            decay_id = decaytype_map.get(matched_label, -1) if decaytype_map else -1
            frame.Put(f"{key_prefix}DecayTypeID", dataclasses.I3Double(decay_id))

            return True

    return False


'''
Example usage
from icetray_segments.decay_mode_selector import select_decay_mode, TAU_DECAY_MODES, TAU_DECAYTYPE_MAP
from icecube import dataclasses

def process_event(frame):
    if not frame.Has("I3MCTree"):
        return

    tree = frame["I3MCTree"]
    success = select_decay_mode(
        frame=frame,
        tree=tree,
        mother_type=dataclasses.I3Particle.NuTau,
        decay_modes=TAU_DECAY_MODES,
        decaytype_map=TAU_DECAYTYPE_MAP,
        key_prefix="NuTau"
    )

    if success:
        print("Decay mode matched:", frame["NuTauDecayLabel"].value)
    else:
        print("No matching ντ decay mode found.")
'''