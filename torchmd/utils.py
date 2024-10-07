import csv
import json
import os
import numpy as np
import time
import argparse
import yaml
import pandas as pd

class LogWriter(object):
    # kind of inspired form openai.baselines.bench.monitor
    # We can add here an optional Tensorboard logger as well
    def __init__(self, path, keys, header="", name="monitor.csv", append=False):
        self.keys = tuple(keys) + ("t",)
        assert path is not None

        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, name)
        if os.path.exists(filename) and not append:
            os.remove(filename)

        print("Writing logs to ", filename)

        fmode = "a+" if append else "wt"
        self.f = open(filename, fmode)
        self.logger = csv.DictWriter(self.f, fieldnames=self.keys)

        if not append:
            if isinstance(header, dict):
                header = "# {} \n".format(json.dumps(header))
            self.f.write(header)
            self.logger.writeheader()
            self.tstart = time.time()
        else:
            monitor = pd.read_csv(filename)
            self.tstart = monitor["t"].iloc[-1]
        
        self.f.flush()
            
    def write_row(self, epinfo):
        if self.logger:
            t = time.time() - self.tstart
            epinfo["t"] = t
            self.logger.writerow(epinfo)
            self.f.flush()

class LoadFromFile(argparse.Action):
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__(self, parser, namespace, values, option_string=None):
        if values.name.endswith("yaml") or values.name.endswith("yml"):
            with values as f:
                namespace.__dict__.update(yaml.load(f, Loader=yaml.FullLoader))
                return

        with values as f:
            input = f.read()
            input = input.rstrip()
            for lines in input.split("\n"):
                k, v = lines.split("=")
                typ = type(namespace.__dict__[k])
                v = typ(v) if typ is not None else v
                namespace.__dict__[k] = v


def save_argparse(args, filename, exclude=None):
    if filename.endswith("yaml") or filename.endswith("yml"):
        if isinstance(exclude, str):
            exclude = [
                exclude,
            ]
        args = args.__dict__.copy()
        for exl in exclude:
            del args[exl]
        with open(filename, "w") as fout:
            yaml.dump(args, fout)
    else:
        with open(filename, "w") as f:
            for k, v in args.__dict__.items():
                if k is exclude:
                    continue
                f.write(f"{k}={v}\n")


def xyz_writer(input_file, output_file, mol_elements):
    """Convert npy trajectory to xyz format
    Args:
    input_file (str): path to npy trajectory
    output_file (str): path to output xyz file
    mol_elements (list): list of elements in the molecule
    """
    npy_traj = np.load(input_file)
    Nats, _, Nsteps = npy_traj.shape
    with open(output_file, "a") as f:
        for i in range(Nsteps):
            f.write(f"{Nats}\n\n")
            for j in range(Nats):
                f.write(f"{mol_elements[j]} ")
                f.write(" ".join(map(str, npy_traj[j, :, i])))
                f.write("\n")

HeavyMasses = {"CH3": 15.03452, "CH2": 14.02658, "CH1": 13.01864, "CH0": 12.0107, 
                "NH3": 17.03052, "NH2": 16.02258, "NH1": 15.01464, "NH0": 14.0067,
                "OH1": 17.00734, "OH0": 15.994, "SH1": 33.0729, "SH0": 32.065,
                }

def fix_resume_files(resume_dir, n_replicas, output_period):
    last_frame = None
    for i in range(n_replicas):
        file_name = os.path.join(resume_dir, f'output_{i}.npy')
        monitor_name = os.path.join(resume_dir, f'monitor_{i}.csv')
        if i == 0:
            c = np.load(file_name)
            last_frame = c.shape[2]
            #print(f'last frame: {last_frame}')
            c = c[:, :, :last_frame+1]
        else:
            c = np.load(file_name)[:, :, :last_frame+1]
        #print(f'file {i} with shape {c.shape}')
        np.save(file_name, c)
        monitor = pd.read_csv(monitor_name)
        monitor = monitor.iloc[:last_frame]
        iter = int(monitor['iter'].iloc[-1] / output_period)
        assert iter == c.shape[2], f'{iter} != {c.shape[2]}'
        #print(f'monitor {i} : {monitor.iloc[-1]}')
        monitor.to_csv(monitor_name, index=False)
    
def reset_monitor_csv(monitor_files, last_step):
    """Since the npy files are updated every output_period, when we resume simulation, 
    the monitor.csv needs to be updated as well."""
    for m in monitor_files:
        monitor = pd.read_csv(m)
        monitor = monitor.iloc[:last_step]
        monitor.to_csv(m, index=False)
    return        
    
def get_init_xtc(mol, replaydir):
    """Get initial coordinates from replay directory, each npy contains a trajectory which is read
    and the last frame is taken as the initial coordinates."""
    import glob
    npys = glob.glob(replaydir + "/output_*.npy")
    npys = sorted(npys, key=lambda x: int(x.split("_")[-1].split(".")[0])) # sort by replica number
    monitors = [f.replace("output", "monitor").replace(".npy", ".csv") for f in npys]
    
    all_init_coords = []
    steps_done = []
    for npy in npys:
        c = np.load(npy)
        steps_done.append(c.shape[2])
        all_init_coords.append(c[:, :, -1])
        
    assert len(set(steps_done)) == 1, f"All replicas should have same number of steps: {set(steps_done)}"
    init_coords = np.moveaxis(np.array(all_init_coords), 0, -1) # (Natoms, 3, Nreplicas)
    
    reset_monitor_csv(monitors, steps_done[0])
    mol.coords = init_coords
    return mol, steps_done[0]