import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import subprocess
from matplotlib import cm
from scipy.stats import entropy


class BCI:
    def __init__(self,
                 data,
                 verbose=False):
        # Path to the input data file which may be manipulated
        # by the .transform() function
        self.data = data
        # Retain a reference to the untransformed data
        self._data = data
        # Get the name of this sample
        self.samp = data.split("/")[-1].split(".")[0]
        # Is the input fastq or fasta?
        self._ftype = data.split(".")[-1]

        self.tmpdir = f"./.tmpdir-{self.samp}"
        if not os.path.exists(self.tmpdir):
            os.mkdir(self.tmpdir)

        # The default label is the sample name for untransformed data
        self._label = self.samp
        # A dictionary for storing label/bci results in the case
        # of running multiple transformations. The dictionary
        # maps labels to lists of BCI results, which themselves are lists
        self._results = {}
        self._verbose = verbose


    def _build_cmds(self):
        self.tols = np.linspace(100, 80, 20)/100
        self.cmds = []
        self.seed_files = []
        for tol in self.tols:
            outfile = f"{self.tmpdir}/{self.samp}-{tol:.3f}"
            cmd = ["vsearch",
                   "-cluster_smallmem", f"{self.data}",
                   "-strand", "plus",
                   "-id", f"{tol}",
                   "-userout", f"{outfile}.utmp",
                   "-userfields", "query+target+id+gaps+qstrand+qcov",
                   "-maxaccepts", "1",
                   "-maxrejects", "0",
                   "-notmatched", f"{outfile}.htmp",
                   "-fasta_width", "0",
                   "-fulldp",
                   "-usersort"]
            self.cmds.append(' '.join(cmd))
            self.seed_files.append(f"{outfile}.htmp")


    ## FIXME: Make n_jobs dynamic
    def run(self, verbose=False):
        # Build the list of vsearch commands
        self._build_cmds()
        # Run all vsearch commands in parallel
        results = joblib.Parallel(n_jobs=20)(joblib.delayed(\
                                    self._systemcall)(f) for f in self.cmds)
        # Count the number of lines in each seed file, this is the number
        # of clusters at a given clustering threshold 
        # Divide by 2 here because the seeds file has 2 lines per seed
        self.bci = sorted([int(len(open(x).readlines())/2) for x in self.seed_files], reverse=True)
        if self._verbose or verbose: print(self.bci)
        # Store the results
        self._results.setdefault(self._label, [])
        self._results[self._label].append(self.bci)


    def plot(self, ax=None, log=True, normalize=True, **kwargs):
        if not ax:
            fig, ax = plt.subplots(figsize=(8, 8))
        fig = ax.get_figure()
        # log transform
        dat = np.log(self.bci) if log else self.bci
        # normalize
        norm = dat[0] if normalize else 1
        ax.plot(np.array(dat)/norm, label=self._label, **kwargs)
        ax.legend(loc='upper right', bbox_to_anchor=(0.97, 0.97))
        return fig, ax


    def plot_all(self, ax=None, log=True, normalize=True, cmap="Spectral", **kwargs):
        if not ax:
            fig, ax = plt.subplots(figsize=(8, 8))
        fig = ax.get_figure()

        cmap = cm.get_cmap(cmap)
        # Get a dictionary mapping labels to evenly spaced color values in the cmap
        cdict = {k:cmap(i/len(self._results)) for i, k in enumerate(self._results.keys())}

        for label, bcis in self._results.items():
            for bci in bcis:
                # log transform
                dat = np.log(bci) if log else bci
                # normalize
                norm = dat[0] if normalize else 1
                ax.plot(np.array(dat)/norm, label=label, color=cdict[label], **kwargs)
        ax.legend(loc='upper right', bbox_to_anchor=(0.97, 0.97))
        return fig, ax


    def transform(self, transformation=None, fraction=0.5):
        """
        Perform a transformation on the raw data.
        raw - Reset data to clean raw format
        disturbance - Downsample the # of sequences to simulate a disturbance
        invasion - Randomly sample one individual and replace a fraction of the community with it
        """
        # Rewind any transformations applied and revert to the raw data
        self.data = self._data
        if transformation == "raw":
            # If raw then just reset the raw data ane return
            self._label = self.samp
        elif transformation in ["disturbance", "invasion"]:
            trans = transformation[:4]
            self._label = f"{self.samp}-{trans}-{fraction}"
            with open(self.data, 'r') as infile:
                ofile = os.path.join(self.tmpdir, f"{self._label}.{self._ftype}")
                if self._verbose: print(ofile)
                with open(ofile, 'w') as outfile:
                    lines = 4 if 'q' in self._ftype else 2
                    seqdealer = zip(*[iter(infile)] * lines)

                    if transformation == "disturbance":
                        for x in seqdealer:
                            if random.random() < fraction:
                                outfile.write(''.join(x))
                    else:
                        invader = next(seqdealer)
                        outfile.write(''.join(invader))
                        for x in seqdealer:
                            if random.random() < fraction:
                                outfile.write(''.join(invader))
                            else:
                                outfile.write(''.join(x))
            self.data = ofile      
        else:
            raise ValueError("transform() argument must be one of: raw, disturbance.")


    def _systemcall(self, to_exec):
        proc = subprocess.Popen(to_exec,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        retval = proc.returncode
        return(stdout, stderr, retval)


###################
# Utility functions
###################

def plot_multi(bci_list, ax=None, log=True, normalize=False, cmap="Spectral", **kwargs):
    """
    Plot multiple BCIs from different datasets
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(8, 8))
        fig = ax.get_figure()

        cmap = cm.get_cmap(cmap)
        # Get a dictionary mapping bcis to evenly spaced color values in the cmap
        bci_labels = sorted(set([x.samp for x in bci_list]))
        cdict = {x:cmap(i/len(bci_labels)) for i, x in enumerate(bci_labels)}

        for bci in bci_list:
            # Raw results for untransformed data will be saved in the results
            # dict keyed by the sample name
            data = bci._results[bci.samp][0]
            # log transform
            data = np.log(data) if log else data
            # normalize
            norm = data[0] if normalize else 1
            ax.plot(np.array(data)/norm, label=bci.samp, color=cdict[bci.samp], **kwargs)
        ax.legend(loc='upper right', bbox_to_anchor=(0.97, 0.97), labels=bci_labels)
        return fig, ax


def phylip_to_fasta(phylip, verbose=False):
    fasta_out = phylip.rsplit(".", 1)[0] + ".fasta"
    with open(phylip, 'r') as infile:
        # skip the phylip header
        with open(fasta_out, 'w') as outfile:
            for x in infile.readlines()[1:]:
                name, seq = x.split()
                outfile.write(f">{name}\n{seq}\n")
    if verbose: print(fasta_out)
    return fasta_out


def get_args():
    psr  = argparse.ArgumentParser(
        prog="BCI.py",
        usage="BCI.py [options]",
        description="Generate a BCI from an input metabarcoding dataset."
    )

    psr.add_argument('-i',
                     '--input',
                     help="[Mandatory] Specify a input fasta file.",
                     required=True)
    args = psr.parse_args()
    
    return args


if __name__ == '__main__':
    args = get_args()
    bci = BCI(args.input)
