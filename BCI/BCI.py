import argparse
import glob
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import shutil
import subprocess
from collections import Counter
from itertools import combinations
from matplotlib import cm
from scipy.stats import entropy


class BCI:
    def __init__(self,
                 data,
                 project_dir=".",
                 verbose=False):
        # Path to the input data file which may be manipulated
        # by the .transform() function
        self.data = data
        if not os.path.exists(self.data):
            raise Exception(f"  Input data file not found: {self.data}")

        # Retain a reference to the untransformed data
        self._data = data
        # Get the name of this sample
        self.samp = data.split("/")[-1].split(".")[0]
        # Is the input fastq or fasta?
        self._ftype = data.split(".")[-1]

        self.tmpdir = f"{project_dir}/.tmpdir-{self.samp}"
        if not os.path.exists(self.tmpdir):
            os.mkdir(self.tmpdir)

        # The default label is the sample name for untransformed data
        self._label = self.samp
        # A dictionary for storing label/bci results in the case
        # of running multiple transformations. The dictionary
        # maps labels to lists of BCI results, which themselves are lists
        self._results = {}
        self._verbose = verbose

        self._min_clust_threshold = 80
        self._vsearch_threads = 4
        self._pseudo_variable_sites = 0
        self.cores = 20


    # FIXME: Is swarm better here? It works quite differently, and wouldn't work
    #        well with ASV-table-like data (because it needs sequence counts).
    def _build_cmds(self):
        self.tols = np.arange(100, self._min_clust_threshold, -1)/100
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
                   "-threads", f"{self._vsearch_threads}",
                   "-usersort"]
            self.cmds.append(' '.join(cmd))
            self.seed_files.append(f"{outfile}.htmp")


    ## FIXME: Add a check that vsearch is installed
    ## FIXME: Make n_jobs dynamic
    ## FIXME: Auto-calculate nucleotide diversity at 97% clustering
    def run(self, simulated=False, verbose=False):
        # Build the list of vsearch commands
        self._build_cmds()
        # Run all vsearch commands in parallel
        results = joblib.Parallel(n_jobs=self.cores)(joblib.delayed(\
                                    self._systemcall)(f) for f in self.cmds)
        # Count the number of lines in each seed file, this is the number
        # of clusters at a given clustering threshold 
        # Divide by 2 here because the seeds file has 2 lines per seed
        self.bci = sorted([int(len(open(x).readlines())/2) for x in self.seed_files], reverse=True)
        if self._verbose or verbose: print(self.bci)
        # Store the results
        self._results.setdefault(self._label, [])
        self._results[self._label].append(self.bci)

        try:
            self.nucleotide_diversity(simulated=simulated, verbose=verbose)
        except pd.errors.EmptyDataError:
            print(f"  No 3% diversity in {self._label}, skipping nucleotide diversity.")
        except Exception as inst:
            raise


    def clean(self):
        shutil.rmtree(self.tmpdir)


    def plot(self, ax=None, log=True, normalize=False, plot_pis=False, **kwargs):
        if not ax:
            fig, ax = plt.subplots(figsize=(8, 8))
        fig = ax.get_figure()

        if plot_pis:
            try:
                dat = np.array(sorted(self.pis.values(), reverse=True))
            except:
                print("  BCI does not have pi data, giving up the plot.")
                return None, None
        else:
            # log transform (pis have 0 values so don't like log'ing
            dat = np.log(self.bci) if log else np.array(self.bci)

        # normalize
        norm = dat.sum() if normalize else 1
        ax.plot(np.array(dat)/norm, label=self._label, **kwargs)
        ax.legend(loc='upper right', bbox_to_anchor=(0.97, 0.97))
        return fig, ax


    def plot_all(self, ax=None, log=True, normalize=False, cmap="Spectral", **kwargs):
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
                norm = dat.sum() if normalize else 1
                ax.plot(np.array(dat)/norm, label=label, color=cdict[label], **kwargs)
        ax.legend(loc='upper right', bbox_to_anchor=(0.97, 0.97))
        return fig, ax


    def transform(self, transformation=None, fraction=0.5, count=None):
        """
        Perform a transformation on the raw data.
        raw - Reset data to clean raw format
        disturbance - Downsample the # of sequences to simulate a disturbance
        invasion - Randomly sample one individual and replace a fraction of the community with it

        fraction is the proportion of the community to transform by the disturbance
        count should be positive int and is only used by the 'resample' method
        """
        # Rewind any transformations applied and revert to the raw data
        self.data = self._data
        if transformation == "reset":
            # If reset then just reset the raw data and return
            self._label = self.samp
        elif transformation in ["disturbance", "invasion", "resample"]:
            trans = transformation[:4]
            self._label = f"{self.samp}-{trans}-{fraction}"
            if not count == None and transformation == "resample":
                self._label = f"{self.samp}-{trans}-{count}"
            with open(self.data, 'r') as infile:
                ofile = os.path.join(self.tmpdir, f"{self._label}.{self._ftype}")
                with open(ofile, 'w') as outfile:
                    lines = 4 if 'q' in self._ftype else 2
                    seqdealer = zip(*[iter(infile)] * lines)

                    if transformation == "disturbance":
                        for x in seqdealer:
                            if random.random() > fraction:
                                outfile.write(''.join(x))
                    elif transformation == "invasion":
                        invader = next(seqdealer)
                        outfile.write(''.join(invader))
                        for x in seqdealer:
                            if random.random() < fraction:
                                outfile.write(''.join(invader))
                            else:
                                outfile.write(''.join(x))
                    elif transformation == "resample":
                        if count <= 0:
                            raise Exception(f"In BCI.transform() `count` must be >= 0. You put: {count}")
                        nseqs = len(open(self.data, 'r').readlines())/lines
                        # Get random indices to retain equal to length `count`
                        replace = False
                        if count > nseqs:
                            replace = True
                            print(  f"  warning: count < nseqs: resampling with replacement")
                        sidx = np.random.choice(range(int(nseqs)), count, replace=replace)
                        for idx, x in enumerate(seqdealer):
                            if idx in sidx:
                                outfile.write(''.join(x))
                if self._verbose: print(f"  Wrote transformed data to: {ofile}")

            self.data = ofile      
        else:
            raise ValueError("transform() argument must be one of: reset, disturbance, invasion, resample.")


    def _systemcall(self, to_exec):
        proc = subprocess.Popen(to_exec,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        retval = proc.returncode
        return(stdout, stderr, retval)


    def nucleotide_diversity(self, OTU_threshold=0.97, pseudo_variable_sites=0, simulated=False, verbose=False):
        """
        Calculate nucleotide diversity per species/OTU for the entire dataset.

        simulated - Whether the input fasta is simulated or empirical. If it's
                    simulated then the sequences are organized into known species
                    and we can calculate pi for both the known simulated species
                    identies and also the 97% OTUs (for comparison).
        """
        def pi_from_fasta(data):
            """
            An inner function for calculating pi from an input fasta file.
            The nested function is so we can call it multiple times for simulated
            data on both the raw file and clustered OTU file.
            """
            with open(data) as infile:
                #Drop the trailing newline
                dat = infile.read().strip().split()
            # Simulated data species ids are of the form >r0_x
            self.pis = {x:0 for x in set([x.rsplit("_", 1)[0][1:] for x in dat[::2]])}
            # Set the index to the species id
            # Creates a df of the form:
            #   species_ID_1    ACGTC...
            #   species_ID_1    ACGGC...
            #   species_ID_2    CCGGC
            seq_df = pd.DataFrame(dat[1::2], index=[x.rsplit("_", 1)[0][1:] for x in dat[::2]], columns=["seqs"])
            for spid in self.pis.keys():
                # handle the case where there is only one seq in the seq_df, in which case
                # it returns a raw string rather than an array
                seqs = seq_df.loc[spid]["seqs"]
                if isinstance(seqs, str):
                    seqs = pd.Series(seqs)
                self.pis[spid] = self._nucleotide_diversity(seqs)

        if simulated:
            ## Only want to do this for simulated data because empirical data doesn't have
            ## known species membership
            pi_from_fasta(self.data)
            self.sim_pis = self.pis.copy()

        aligned = self._align_OTUs(OTU_threshold=OTU_threshold,
                                    pseudo_variable_sites=pseudo_variable_sites,
                                    verbose=verbose)
        pi_from_fasta(aligned)


    ## FIXME: This should be in a util or stats package
    def _nucleotide_diversity(self, seqs, verbose=False):
        """
        Calculate nucleotide diversity from a list of sequences.
        `seqs` input should be a list of aligned sequences
        """
        pi = 0

        ## If no sequences or no variation
        if len(seqs) <= 1: return 0

        ## Transpose, so now we have a list of lists of all bases at each
        ## position.
        dat = np.transpose(np.array([list(x) for x in seqs]))

        ## for each position
        for d in dat:
            ## If the position is _not_ monomorphic
            if len(Counter(d)) > 1:
                if verbose: print(Counter(d))
                ## Enumerate the possible comparisons and for each
                ## comparison calculate the number of pairwise differences,
                ## summing over all sites in the sequence.
                base_count = Counter(d)
                ## ignore indels
                del base_count["-"]
                del base_count["N"]
                for c in combinations(base_count.values(), 2):
                    #print(c)
                    n = c[0] + c[1]
                    n_comparisons = float(n) * (n - 1) / 2
                    pi += float(c[0]) * (n-c[0]) / n_comparisons
        return pi/len(seqs.iloc[0])


    def _align_OTUs(self, OTU_threshold=0.97, pseudo_variable_sites=0, verbose=False):
        # Read the utmp file to get hits matching to seeds
        # Retain only columns 0 (hits) and 1 (seeds). Set the index to the seed names
        utmp = glob.glob(self.tmpdir+f"/*{OTU_threshold}*.utmp")[0]
        if not utmp:
            # Something happened, no utmp file
            raise Exception("No utmp file found with OTU_threshold: {OTU_threshold}")

        # Make a new tmp directory to contain the aligned fasta files
        # Force clean up the old directory if it exists
        tmp_fastadir = self.tmpdir+f"/OTU-{OTU_threshold}_fastas"
        shutil.rmtree(tmp_fastadir, ignore_errors=True)
        os.mkdir(tmp_fastadir)

        # Using the 'seeds' column as the index and retaining the 'hits' column as data
        clusts = pd.read_csv(utmp, sep="\t", header=None, usecols=[0,1], index_col=1)
        # otus == the seed sequence IDs
        otus = set(clusts.index)

        # Get a data frame formatted with the zotu name as the index, like this:
        #   zotu1   AACATGCT...
        #   zotu2   AAGATCCT...
        seq_df = self._fasta_to_df()

        for otu in otus:
            # Force zotu ids to be str to avoid conflict if zotu ids are auto-detected as int
            zids = np.append(clusts.loc[otu].values.astype(str), otu)
            seqs = seq_df.loc[zids].values
            with open(f"{tmp_fastadir}/{otu}.fasta", 'w') as outfile:
                for idx, seq in enumerate(seqs):
                    outfile.write(f">{otu}_{idx}\n{seq}\n")

        # Identify singleton sequences (unique sequences w/o any hits)
        # Singletons are any sequences in the seq_df that are NOT a hit or seed in the utmp file
        hits = np.append(clusts[0].values, list(otus))
        singletons = seq_df[~seq_df.index.isin(hits)]

        for zid, seq in singletons.items():
            seqs = [seq.lower()]
            if pseudo_variable_sites:
                seqs = seqs.append(seq.replace('a', 't', pseudo_variable_sites))
            with open(f"{tmp_fastadir}/{zid}.fasta", 'w') as outfile:
                for idx, seq in enumerate(seqs):
                    outfile.write(f">{zid}_{idx}\n{seq}\n")

        if verbose: print("Aligning..")
        self._muscle_cmds = []
        for fa in glob.glob(f"{tmp_fastadir}/*.fasta"):
            outfile = f"{fa}.aln"
            cmd = ["muscle",
                    "-align", f"{fa}",
                    "-output", f"{outfile}",
                    "-quiet",
                    "-threads", "2"]
            self._muscle_cmds.append(' '.join(cmd))

        # Run all muscle commands in parallel
        results = joblib.Parallel(n_jobs=20)(joblib.delayed(\
                                    self._systemcall)(f) for f in self._muscle_cmds)

        # Gather all aligned fastas into one file
        # The muscle output breaks sequence data at 80 characters, so we have
        # to recompose it into one long line
        aligned = f"{tmp_fastadir}/combined-aligned-{OTU_threshold}.fasta"
        with open(aligned, 'w') as outfile:
            for fa in glob.glob(f"{tmp_fastadir}/*.aln"):
                dat = open(fa).read().split()
                for line in dat:
                    if ">" in line:
                        outfile.write(f"\n{line}\n")
                    else:
                        outfile.write(line)

        # Remove the leading newline. This is dumb, but it works
        dat = open(aligned).readlines()
        with open(aligned, 'w') as outfile:
            outfile.write("".join(dat[1:]).strip())

        return aligned


    def _fasta_to_df(self):
        seq_data = open(self.data).read().split()
        ## Doing some formatting to make metadata and fasta zotu names agree:
        # Drop the leading >
        # Force all lowercase
        # Remove trailing underscore
        zotus = [x[1:] for x in seq_data[::2]]
        # Strip any trailing gaps (not sure why they are there)
        fastas = [x.rstrip("-") for x in seq_data[1::2]]

        return pd.Series(fastas, index=zotus)


    def write_bci(self, outdir=None):
        self.write_results(outdir=outdir, data="bci")

    def write_pis(self, outdir=None):
        self.write_results(outdir=outdir, data="pis")

    def write_results(self, outdir=None, data="bci"):
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if data == "bci":
            data = self.bci
        elif data == "pis":
            data = [str(x) for x in self.pis.values()]
        else:
            raise Exception("  Bad 'data' in 'write_results'. Should be 'bci' or 'pis'.")

        with open(os.path.join(outdir, self.samp)+".pis", 'w') as outfile:
            outfile.write(",".join(sorted(data, reverse=True))+"\n")


###################
# Utility functions
###################

def plot_multi(bci_list, ax=None, log=True, normalize=False, plot_pis=False, cmap="Spectral", keyed_cmaps=None, **kwargs):
    """
    Plot multiple BCIs from different datasets

    keyed_cmaps (dict) - Keys are substrings common to groups of samples
                        (site name, habititat, etc), values are cmaps to use
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()

    if keyed_cmaps:
        # Use one coloramp for each group of samples keyed by identifiers in the sample names
        cdict = {}
        for k, v in keyed_cmaps.items():
            bci_labels = [x.samp for x in bci_list if k in x.samp]
            cmap = cm.get_cmap(v)(np.linspace(0.3, 0.9, len(bci_labels)))
            cdict.update({x:cmap[i] for i, x in enumerate(bci_labels)})
    else:
        # use one colormap for all samples. linspace trims off the 'edges' of the cmap
        # to make it only use colors that are readily visible against a white background
        cmap = cm.get_cmap(cmap)(np.linspace(0.3, 0.9, len(bci_list)))
        # Get a dictionary mapping bcis to evenly spaced color values in the cmap
        bci_labels = [x.samp for x in bci_list]
        cdict = {x:cmap[i] for i, x in enumerate(bci_labels)}

        # Alternate method for sampling cmaps that samples the full range
        #cmap = cm.get_cmap(cmap)
        #bci_labels = [x.samp for x in bci_list]
        #cdict = {x:cmap(i/len(bci_labels)) for i, x in enumerate(bci_labels)}

    for bci in bci_list:
        if plot_pis:
            try:
                data = np.array(sorted(bci.pis.values(), reverse=True))
            except:
                print("  BCI does not have pi data, giving up the plot.")
                return None, None
        else:
            # Raw results for untransformed data will be saved in the results
            # dict keyed by the sample name
            # Why did I do this? It's not obvious why you wouldn't want to
            # just plot the most recent run of the data. Here is what I was doing before:
            # data = bci._results[bci.samp][0]
            data = bci.bci
            # log transform
            data = np.log(data) if log else np.array(data)
        # normalize
        norm = data.sum() if normalize else 1
        ax.plot(data/norm, label=bci.samp, color=cdict[bci.samp], **kwargs)
    #ax.legend(loc='upper right', bbox_to_anchor=(0.97, 0.97), labels=bci_labels)
    plt.legend()
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
