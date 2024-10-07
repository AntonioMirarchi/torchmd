import os
import torch
from torchmd.systems import System
from moleculekit.molecule import Molecule
from torchmd.forcefields.forcefield import ForceField
from torchmd.parameters import Parameters
from torchmd.forces import Forces
from torchmd.integrator import Integrator
from torchmd.wrapper import Wrapper
import numpy as np
from tqdm import tqdm
import argparse
import importlib
from torchmd.integrator import maxwell_boltzmann
from torchmd.minimizers import minimize_bfgs
from torchmd.npzmol import npzMolecule
from torchmd.utils import save_argparse, xyz_writer, LogWriter, LoadFromFile, HeavyMasses, get_init_xtc #, fix_resume_files

FS2NS = 1e-6

def viewFrame(mol, pos, forces):
    from ffevaluation.ffevaluate import viewForces

    mol.coords[:, :, 0] = pos[0].cpu().detach().numpy()
    mol.view(guessBonds=False)
    viewForces(mol, forces[0].cpu().detach().numpy()[:, :, None] * 0.01)


def get_args(arguments=None):
    parser = argparse.ArgumentParser(description="TorchMD", prefix_chars="--")
    parser.add_argument(
        "--conf",
        type=open,
        action=LoadFromFile,
        help="Use a configuration file, e.g. python run.py --conf input.conf",
    )
    parser.add_argument("--timestep", default=1, type=float, help="Timestep in fs")
    parser.add_argument(
        "--temperature",
        default=300,
        type=float,
        help="Assign velocity from initial temperature in K",
    )
    parser.add_argument(
        "--langevin-temperature",
        default=0,
        type=float,
        help="Temperature in K of the thermostat",
    )
    parser.add_argument(
        "--langevin-gamma", default=0.1, type=float, help="Langevin relaxation ps^-1"
    )
    parser.add_argument("--device", default="cpu", help='Type of device, e.g. "cuda:1"')
    parser.add_argument("--structure", default=None, help="Deprecated: Input PDB")
    parser.add_argument("--topology", default=None, type=str, help="Input topology")
    parser.add_argument(
        "--coordinates", default=None, type=str, help="Input coordinates"
    )
    parser.add_argument(
        "--forcefield",
        default="tests/argon/argon_forcefield.yaml",
        help="Forcefield .yaml file",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--output-period",
        type=int,
        default=10,
        help="Store trajectory and print monitor.csv every period",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=0,
        help="Dump trajectory to npy file. By default 10 times output-period.",
    )
    parser.add_argument(
        "--steps", type=int, default=10000, help="Total number of simulation steps"
    )
    parser.add_argument("--log-dir", default="./", help="Log directory")
    parser.add_argument(
        "--output", default="output", help="Output filename for trajectory"
    )
    parser.add_argument(
        "--forceterms",
        nargs="+",
        default="LJ",
        help="Forceterms to include, e.g. --forceterms Bonds LJ",
    )
    parser.add_argument(
        "--cutoff", default=None, type=float, help="LJ/Elec/Bond cutoff"
    )
    parser.add_argument(
        "--switch_dist", default=None, type=float, help="Switching distance for LJ"
    )
    parser.add_argument(
        "--precision", default="single", type=str, help="LJ/Elec/Bond cutoff"
    )
    parser.add_argument(
        "--external", default=None, type=dict, help="External calculator config"
    )
    parser.add_argument(
        "--rfa",
        default=False,
        action="store_true",
        help="Enable reaction field approximation",
    )
    parser.add_argument(
        "--replicas", type=int, default=1, help="Number of different replicas to run"
    )
    parser.add_argument(
        "--extended_system", default=None, type=float, help="xsc file for box size"
    )
    parser.add_argument(
        "--minimize",
        default=None,
        type=int,
        help="Minimize the system for `minimize` steps",
    )
    parser.add_argument(
        "--exclusions",
        default=("bonds", "angles", "1-4"),
        type=tuple,
        help="exclusions for the LJ or repulsionCG term",
    )
    parser.add_argument(
        "--npz_file", default=None, type=str, help="Input file.npz with coord and z"
    )
    parser.add_argument(
        "--useIncreasedMasses",
        default=False,
        help="Use increased masses for heavy atoms, suggested for noHydrogen systems",
    )
    parser.add_argument('--resume-dir', default=None, type=str, help='Path to the directory to resume the simulation')

    args = parser.parse_args(args=arguments)
    os.makedirs(args.log_dir, exist_ok=True)
    save_argparse(args, os.path.join(args.log_dir, "input.yaml"), exclude="conf")

    if isinstance(args.forceterms, str):
        args.forceterms = [args.forceterms]
    if args.steps % args.output_period != 0:
        raise ValueError("Steps must be multiple of output-period.")
    if args.save_period == 0:
        args.save_period = 10 * args.output_period
    if args.save_period % args.output_period != 0:
        raise ValueError("save-period must be multiple of output-period.")

    return args


precisionmap = {"single": torch.float, "double": torch.double}


def setup(args, batch_comp=False):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # We want to set TF32 to false by default to avoid precision problems
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    steps_done = None # for resuming the simulation
    
    if args.topology is not None:
        mol = Molecule(args.topology)
    elif args.structure is not None:
        mol = Molecule(args.structure)
        mol.box = (
            np.array([mol.crystalinfo["a"], mol.crystalinfo["b"], mol.crystalinfo["c"]])
            .reshape(3, 1)
            .astype(np.float32)
        )
    elif args.npz_file is not None:
        mol = npzMolecule(args.npz_file)
        batch_comp = True

    if args.coordinates is not None:
        mol.read(args.coordinates)

    if args.extended_system is not None:
        mol.read(args.extended_system)

    if args.resume_dir is not None:
        print(f"Resuming simulation from {args.resume_dir}")
        #fix_resume_files(args.resume_dir, n_replicas=args.replicas, output_period=args.output_period)
        mol, steps_done = get_init_xtc(mol, args.resume_dir)
        print(f"Resuming from step {steps_done}")
        print(f"Initial pos shape: {mol.coords.shape}")
        
    if args.useIncreasedMasses:
        print("Using Increased Masses for Heavy Atoms")
        print(f"Initial Molecule Mass: {np.sum(mol.masses)}")
        mol.masses = np.array([HeavyMasses[el] for el in mol.atomtype]).astype(np.float32)
        print(f"New Molecule Mass: {np.sum(mol.masses)}")
        
    precision = precisionmap[args.precision]

    print("Force terms: ", args.forceterms)
    ff = ForceField.create(mol, args.forcefield)
    parameters = Parameters(
        ff, mol, args.forceterms, precision=precision, device=device
    )

    external = None
    if args.external is not None:
        externalmodule = importlib.import_module(args.external["module"])
        if batch_comp:
            embeddings = torch.tensor(mol.embedding).repeat(args.replicas, 1)
        else:
            if isinstance(args.external["embeddings"], str):
                embeddings = torch.tensor(
                    np.load(args.external["embeddings"]).astype(int)
                ).repeat(args.replicas, 1)
            else:
                embeddings = torch.tensor(args.external["embeddings"]).repeat(
                    args.replicas, 1
                )

        file = args.external["file"]
        # remove from args.external the items that have been already passed to the external module
        args.external = {
            key: value
            for key, value in args.external.items()
            if key not in ["module", "file", "embeddings"]
        }
        external = externalmodule.External(
            file, embeddings, device=device, **args.external
        )

    system = System(mol.numAtoms, args.replicas, precision, device)
    system.set_positions(mol.coords)
    system.set_box(mol.box)
    system.set_velocities(
        maxwell_boltzmann(parameters.masses, args.temperature, args.replicas)
    )

    forces = Forces(
        parameters,
        terms=args.forceterms,
        external=external,
        cutoff=args.cutoff,
        rfa=args.rfa,
        switch_dist=args.switch_dist,
        exclusions=args.exclusions,
    )
    return mol, system, forces, steps_done


def dynamics(args, mol, system, forces, steps_done=None):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    integrator = Integrator(
        system,
        forces,
        args.timestep,
        device,
        gamma=args.langevin_gamma,
        T=args.langevin_temperature,
    )
    wrapper = Wrapper(mol.numAtoms, mol.bonds if len(mol.bonds) else None, device)

    outputname, outputext = os.path.splitext(args.output)
    logs = []
    trajs = []
    #forces = []
    for k in range(args.replicas):
        logs.append(
            LogWriter(
                args.log_dir,
                keys=("iter", "ns", "epot", "ekin", "etot", "T"),
                name=f"monitor_{k}.csv",
                append=False if steps_done is None else True,
            )
        )
        trajs.append([])
        #forces.append([])
        if steps_done is not None:
            resume_traj = np.load(os.path.join(args.log_dir, f"{outputname}_{k}{outputext}.npy"))
            #resume_force = np.load(os.path.join(args.log_dir, f"forces_{outputname}_{k}{outputext}.npy"))
            for frame_i in range(resume_traj.shape[2]):
                trajs[k].append(list(resume_traj[:, :, frame_i]))
                #forces.append(list(resume_force[:, :, frame_i]))
                        
    if args.minimize != None:
        minimize_bfgs(system, forces, steps=args.minimize)

    iterator = tqdm(range(1, int(args.steps / args.output_period) + 1))
    Epot = forces.compute(system.pos, system.box, system.forces)

    for i in iterator:
        if steps_done is not None and i < (steps_done+1):
            # best way to resume instead of modifying the iterator, the +1 is because we start from 1
            # we just skip the steps that have been already done, and 'iter' and 'ns' in monitor.csv will be correct
            continue 
        
        # viewFrame(mol, system.pos, system.forces)
        Ekin, Epot, T = integrator.step(niter=args.output_period)
        wrapper.wrap(system.pos, system.box)
        currpos = system.pos.detach().cpu().numpy().copy()
        #currforces = system.forces.detach().cpu().numpy().copy()
        
        for k in range(args.replicas):
            trajs[k].append(currpos[k])
            #forces[k].append(currforces[k])
            if (i * args.output_period) % args.save_period == 0:
                np.save(
                    os.path.join(args.log_dir, f"{outputname}_{k}{outputext}"),
                    np.stack(trajs[k], axis=2),
                )  # ideally we want to append
                #np.save(
                #    os.path.join(args.log_dir, f"forces_{outputname}_{k}{outputext}"),
                #    np.stack(forces[k], axis=2),
                #)  # ideally we want to append
            
            logs[k].write_row(
                {
                    "iter": i * args.output_period,
                    "ns": FS2NS * i * args.output_period * args.timestep,
                    "epot": Epot[k],
                    "ekin": Ekin[k],
                    "etot": Epot[k] + Ekin[k],
                    "T": T[k],
                }
            )

    # new for on replicas because we start from .npy file saved in the previous step
    for k in range(args.replicas):
        npy_name = os.path.join(args.log_dir, args.output + f"_{k}.npy")
        xyz_name = os.path.join(args.log_dir, args.output + f"_{k}.xyz")
        xyz_writer(npy_name, xyz_name, mol.element)

if __name__ == "__main__":
    args = get_args()
    mol, system, forces, steps = setup(args)
    dynamics(args, mol, system, forces, steps)
