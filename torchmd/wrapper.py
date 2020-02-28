import torch

class Wrapper:
    def __init__(self,natoms,bonds,device):
        self.groups, self.nongrouped = calculateMoleculeGroups(natoms, bonds, device)

    def wrap(self,pos, box, wrapidx=None):
        nmol = len(self.groups)

        if wrapidx is not None:
            # Get COM of wrapping center group
            com = torch.sum(pos[wrapidx], dim=0) / len(wrapidx)
            # Subtract COM from all atoms so that the center mol is at [box/2, box/2, box/2]
            pos = (pos - com) + (box / 2)

        if nmol != 0:
            # Work out the COMs and offsets of every group and move group to [0, box] range
            for i, group in enumerate(self.groups):
                tmp_com = torch.sum(pos[group], dim=0) / len(group)
                offset = torch.floor(tmp_com / box) * box
                pos[group] -= offset

        # Move non-grouped atoms
        if len(self.nongrouped):
            offset = torch.floor(pos[self.nongrouped] / box) * box
            pos[self.nongrouped] -= offset


def calculateMoleculeGroups(natoms, bonds, device):
    import networkx as nx

    # Calculate molecule groups and non-bonded / non-grouped atoms
    if bonds is not None:
        bondGraph = nx.Graph()
        bondGraph.add_nodes_from(range(natoms))
        bondGraph.add_edges_from(bonds)
        molgroups = list(nx.connected_components(bondGraph))
        nongrouped = torch.tensor(
            [list(group)[0] for group in molgroups if len(group) == 1]
        ).to(device)
        molgroups = [
            torch.tensor(list(group)).to(device)
            for group in molgroups
            if len(group) > 1
        ]
    else:
        molgroups = []
        nongrouped = torch.arange(0, natoms).to(device)
    return molgroups, nongrouped
