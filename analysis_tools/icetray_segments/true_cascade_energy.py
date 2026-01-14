# File: analysis_tools/icetray_segments/true_cascade_energy.py
from icecube import dataclasses
from icecube.icetray import I3Units
import numpy as np

__all__ = ["add_true_cascade_energies"]

def shift_to_maximum(shower, ref_energy):
    """
    Shift a cascade to its shower maximum, approximated using a longitudinal profile model.
    """
    a = 2.03 + 0.604 * np.log(ref_energy/I3Units.GeV)
    b = 0.633
    lrad = (35.8*I3Units.cm/0.9216)
    lengthToMaximum = ((a-1.)/b)*lrad

    p = dataclasses.I3Particle(shower)
    p.energy = ref_energy
    p.fit_status = p.OK
    p.pos.x = shower.pos.x + p.dir.x * lengthToMaximum
    p.pos.y = shower.pos.y + p.dir.y * lengthToMaximum
    p.pos.z = shower.pos.z + p.dir.z * lengthToMaximum
    return p

def collect_cascade_particles(particle, tree, output):
    """
    Recursively collect visible cascade daughters from a decay chain.
    """
    daughters = tree.get_daughters(particle)
    for d in daughters:
        if d.shape != d.Dark and d.is_cascade and d.location_type == d.InIce:
            output.append(d)
        collect_cascade_particles(d, tree, output)

def compute_visible_energy(particles):
    """
    Compute visible energy from a list of particles.
    """
    losses = 0.0
    center = dataclasses.I3Position(0,0,0)
    for p in particles:
        if p.type in [p.Hadrons, p.PiPlus, p.PiMinus, p.NuclInt]:
            if p.energy < 1*I3Units.GeV:
                e = 0.8*p.energy
            else:
                energyScalingFactor = 1.0 + ((p.energy/I3Units.GeV/0.399)**-0.130)*(0.467 - 1)
                e = energyScalingFactor * p.energy
        else:
            e = p.energy
        losses += e
        center += e * p.pos

    return losses, center / losses if losses > 0 else dataclasses.I3Position(0,0,0)

def add_true_cascade_energies(frame):
    if "I3MCTree" not in frame or "I3MCWeightDict" not in frame:
        return

    tree = frame["I3MCTree"]
    int_type = frame["I3MCWeightDict"].get("InteractionType", None)
    if int_type != 1:
        return  # Only for CC events

    # Find the NuTau or NuTauBar that interacted in ice
    neutrino_candidates = [
        p for p in tree if p.type in [p.NuTau, p.NuTauBar] and 
        p.location_type == p.InIce and p.is_neutrino and not np.isnan(p.length) and p.length > 0
    ]

    if not neutrino_candidates:
        return

    # Find closest vertex to origin
    def dist_to_origin(p):
        daughters = list(tree.get_daughters(p))
        if not daughters:
            return 1e10
        pos = daughters[0].pos
        return np.sqrt(pos.x**2 + pos.y**2 + pos.z**2)

    primary = min(neutrino_candidates, key=dist_to_origin)
    daughters = list(tree.get_daughters(primary))

    # Identify first cascade and tau
    first_cascade = None
    tau = None
    for d in daughters:
        if d.type in [d.Hadrons]:
            first_cascade = d
        elif d.type in [d.TauMinus, d.TauPlus]:
            tau = d

    # Collect tau daughter cascades
    tau_cascades = []
    if tau:
        tau_daughters = list(tree.get_daughters(tau))
        for td in tau_daughters:
            if td.type not in [td.MuMinus, td.MuPlus]:  # skip muon decays
                collect_cascade_particles(td, tree, tau_cascades)

    # Collect primary hadronic cascade cascades
    had_cascades = []
    if first_cascade:
        collect_cascade_particles(first_cascade, tree, had_cascades)

    # Compute energies
    e1, _ = compute_visible_energy(had_cascades)
    e2, _ = compute_visible_energy(tau_cascades)

    if first_cascade:
        p1 = shift_to_maximum(first_cascade, e1)
        frame["Cascade1_vis_truth_tau"] = p1
    if tau:
        p2 = shift_to_maximum(tau, e2)
        frame["Cascade2_vis_truth_tau"] = p2
