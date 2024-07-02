import argparse
import glob
import gzip
import numpy as np
import os
import pandas as pd
import random
import shutil
import BCI


class Project:
    def __init__(self,
                 asv_table,
                 fasta_file,
                 sitemap=None,
                 drop_duplicates=True,
                 verbose=False):
        """
        """

        self.asv_table, self.zotus_per_sample = self._read_asv_table(asv_table)

        self._fasta_file = fasta_file
        self.seq_df = self._read_fasta(fasta_file)

        #TODO: Check asv_table/fasta data for consistency

        self.project_dir = "./"
        self._sample_fastadir = os.path.join(self.project_dir, "sample_fastas")

        # Make a new fasta file for all the sequences w/in a sample
        self.sample_fastas = self._make_sample_fastas(verbose)

        self.site_fastas = {}
        self.sitemap = {}
        if not sitemap == None:
            self.sitemap, self.samples_per_site = self._read_sitemap(sitemap)
            self._site_fastadir = os.path.join(self.project_dir, "site_fastas")

            self.site_fastas = self._make_site_fastas(drop_duplicates=drop_duplicates, verbose=verbose)

        self.sample_bcis = {}
        self.site_bcis = {}


    @property
    def samples(self):
        return list(self.zotus_per_sample.keys())


    @property
    def sites(self):
        return list(self.samples_per_site.keys())


    def _read_asv_table(self, asv_table, verbose=False):

        asv_table = pd.read_csv(asv_table, index_col=0, sep=None, engine='python')

        zotus_per_sample = {}

        for c in asv_table.columns:
            tmp = asv_table[c][asv_table[c] != 0]
            zotus_per_sample[c] = tmp.keys().astype(str)

        if verbose:
            for k, v in zotus_per_sample.items():
                print(k, "\t", len(v))

        return asv_table, zotus_per_sample


    def _read_sitemap(self, sitemap):
        # Allow to auto-detect csv delimiter
        sitemap = pd.read_csv(sitemap, comment="#", names=["sample", "site"], sep=None, engine='python', dtype=str)

        group = sitemap.groupby(["site"])
        samples_per_site = {site[0]: list(set(group["sample"])) for site, group in group}
        return sitemap, samples_per_site


    def _read_fasta(self, fasta_file):
        if fasta_file.endswith((".gz", "gzip")):
            _open = gzip.open
            _flags = 'rt'
        else:
            _open = open
            _flags = 'r'

        dat = _open(fasta_file, _flags).readlines()

        # Collapsing multiline fasta into single line per sequence
        dat = ["\n"+x if ">" in x else x.strip() for x in dat]
        # This seems silly and is probably not efficient for very large data
        # Remove the leading newline before the first seq ID
        dat = "".join(dat)[1:].split()

        names = dat[::2]
        seqs = dat[1::2]

        # Remove leading '>' and trailing size info and newlines
        names = [x[1:].split(";")[0].strip() for x in names]
        seqs = [x.strip() for x in seqs]
        seq_df = pd.Series({x:y for x, y in zip(names, seqs)})
        return seq_df


    def _make_sample_fastas(self, verbose=False):
        if not os.path.exists(self._sample_fastadir):
            os.mkdir(self._sample_fastadir)

        sample_fastas = {}
        for sample, zotus in self.zotus_per_sample.items():
            fasta_data = self.seq_df.loc[zotus]
            # Check fasta files should be this length
            if verbose: print(site, len(fasta_data)*2)
            sample_fasta = f"{self._sample_fastadir}/{sample}.fasta"
            with open(sample_fasta, 'w') as outfile:
                for k, v in fasta_data.items():
                    outfile.write(f">{k}\n{v}\n")
            sample_fastas[sample] = sample_fasta
        return sample_fastas


    def _make_site_fastas(self, subset_samples=None, drop_duplicates=False, verbose=False):
        if os.path.exists(self._site_fastadir):
            shutil.rmtree(self._site_fastadir)
        if not os.path.exists(self._site_fastadir):
            os.mkdir(self._site_fastadir)

        site_fastas = {}
        for site, samples in self.samples_per_site.items():
            site = site.replace(" ", "_")
            site_fasta = f"{self._site_fastadir}/{site}.fasta"
            # Standardize sampling to n random samples per site 
            if subset_samples:
                samples = np.random.choice(samples, subset_samples, replace=False)
            # Accumulate ASV fasta data across all samples
            fasta_data = []
            for sample in samples:
                zotus = self.zotus_per_sample[sample]
                fasta_data.append(self.seq_df.loc[zotus])
            # Join all ASVs across samples into one Series
            fasta_data = pd.concat(fasta_data)
            # If there are duplicate ASVs among sites remove these before clustering
            if drop_duplicates:
                fasta_data = fasta_data.drop_duplicates()
            with open(site_fasta, 'a') as outfile:
                for k, v in fasta_data.items():
                    outfile.write(f">{k}\n{v}\n")
            site_fastas[site] = site_fasta
        return site_fastas


    def run(self, samples=True, sites=True, resample=None, verbose=False):
        if samples:
            self.sample_bcis = {}
            if samples == True: samples = self.samples
            print(f"  Processing {len(samples)} samples.")
            for sample in samples:
                if verbose: print(sample)
                self.sample_bcis[sample] = BCI.BCI(data=self.sample_fastas[sample], verbose=verbose)
                self.sample_bcis[sample]._min_clust_threshold = 70
                if resample:
                    self.sample_bcis[sample].transform(transformation="resample", count=resample)
                self.sample_bcis[sample].run()

        # Only process sites if self.sitemap has been loaded
        if sites and len(self.sitemap):
            self.site_bcis = {}
            if sites == True: sites = self.sites
            print(f"  Processing {len(sites)} sites.")
            for site in sites:
                if verbose: print(site)
                self.site_bcis[site] = BCI.BCI(data=self.site_fastas[site], verbose=verbose)
                self.site_bcis[site]._min_clust_threshold = 70
                if resample:
                    self.site_bcis[site].transform(transformation="resample", count=resample)
                self.site_bcis[site].run()


    def plot_samples(self, include=None, exclude=None):
        pass


    def plot_sites(self, ax=None, include=None, exclude=None, log=True, plot_pis=False, cmaps=None):
        if cmaps == None:
            cmaps = {"Spectral":self.site_bcis.keys()}

        for cmap, sites in cmaps.items():
            fig, ax = BCI.plot_multi([self.site_bcis[x] for x in sites],
                   ax=ax, cmap=cmap, log=log, normalize=False, plot_pis=plot_pis)

        return fig, ax



