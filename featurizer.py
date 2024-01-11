from multiprocessing import set_start_method
# set_start_method("spawn")

import os
import pickle
import numpy as np
from numpy import nan as Nan
import pandas as pd

from ase import atoms
from ase.io import read, write
from dscribe.descriptors import SOAP
import matminer.featurizers.composition as mm_composition
import matminer.featurizers.structure as mm_structure
import matminer.featurizers.structure.bonding as mm_bonding
import pymatgen as mg
from pymatgen.io import ase
from pymatgen.io.cif import CifParser
from pymatgen.io.cif import CifWriter

from tqdm import notebook as tqdm
from tqdm.auto import tqdm as tqdm_pandas
tqdm_pandas.pandas()

AAA = ase.AseAtomsAdaptor

# # CAVD imports, comment out when using 3.7
# from numpy import nan as NaN
# from monty.io import zopen
# from cavd.channel import Channel
# from cavd.netstorage import AtomNetwork, connection_values_list
# from cavd.local_environment import CifParser_new, LocalEnvirCom
# import re

class Feature_Creator:
    """
    A class to handle calculation of the various features/descriptors. 

    ...

    Attributes
    ----------
    structures_df : pd.DataFrame
        The dataframe that all the structures and simplified representations are stored in
    
    mode_list : list
        A list containing the 9 modes that the class can use. If a different mode is passed then an error is thrown. 

    mode : str
        The mode string signifies what mode the class is operating in
        
    unique_atoms : list
        The SOAP featurizer requires knowledge of all unique atoms in the structure. This list stores the unique atoms. 
        
    n_jobs : int
        The number of CPU cores that will be used for featurizers that support parallel processing

    Methods
    -------
    set_mode(mode):
        Set the mode that the class operates in. Each move corresponds to one of the nine structure representations. 
        
    calculate_unique_atoms(mode):
        Calculate the unqiue atoms in the structure. Used for the SOAP representation. 
        
    run_atomic_packing_efficiency_featurizer(mode):
        Calculate the atomic packing efficiency using matminer.featurizers.composition.AtomicPackingEfficiency()  
        
    run_band_center_featurizer(mode):
        Calculate the band centers using matminer.featurizers.composition.BandCenter()    
                
    run_bond_fraction_featurizer(mode):
        Calculate the bond fractions using matminer.featurizers.structure.BondFractions()  
                
    run_chemical_ordering_featurizer(mode):
        Calculate the chemical ordering using matminer.featurizers.structure.ChemicalOrdering()    
                
    run_density_featurizer(mode):
        Calculate features related to density using matminer.featurizers.structure.DensityFeatures(("density", "vpa", "packing fraction"))    
                
    end_featurizer_helper(structure, end_featurizer):
        A helper for the run_electron_negativity_difference_featurizer     
                
    run_electron_negativity_difference_featurizer(mode):
        Calculates the electron negativity difference for atoms in the composition using matminer.featurizers.composition.ElectronegativityDiff()   
                
    run_ewald_energy_featurizer(mode):
        Calculates the Ewald energy using matminer.featurizers.structure.EwaldEnergy()    
                
    run_global_instability_index_featurizer(mode, rcut_list):
        Calculates the global instability index using matminer.featurizers.structure.GlobalInstabilityIndex(r_cut=rcut)    
                
    run_jarvis_cfid_featurizer(mode):
        A jarvis CFID calculation using matminer.featurizers.structure.JarvisCFID()    
                
    run_maximum_packing_efficiency_featurizer(mode):
        Calculates the packing efficiency using matminer.featurizers.structure.MaximumPackingEfficiency()  
                
    run_meredig_featurizer(mode):
        Calculates Meredig features using matminer.featurizers.composition.Meredig()    
                
    run_orbital_field_matrix_featurizer(mode):
        Calculates the orbital field matrix from matminer.featurizers.structure.OxidationStates()    
                
    run_oxidation_states_featurizer(mode):
        Grabs the oxidation states from the composition using matminer.featurizers.composition.OxidationStates()   
                
    run_rdf_featurizer(mode, cutoff_list, bin_size_list):
        Calculates a radial distribution function matminer.featurizers.structure.RadialDistributionFunction(cutoff=cutoff, bin_size=bin_size)    

    run_sine_coulomb_featurizer(mode):
        Calculates the Sine Coulomb matrix using matminer.featurizers.structure.SineCoulombMatrix()    
        
    run_SOAP(mode, rcut_list, nmax_list, lmax_list, average):
        Calculates a Smooth Overlap of Atomic Positions representaion using dscribe.descriptors.SOAP()    
        
    run_structural_complexity_featurizer(mode):
        Calculates the structural complexity using matminer.featurizers.structure.StructuralComplexity()    
        
    run_structural_heterogeneity_featurizer(mode):
        Calculates the structural heterogeneity using matminer.featurizers.structure.StructuralHeterogeneity()    
        
    run_valence_orbital_featurizer(mode):
        Calculates valence obrbital information using matminer.featurizers.composition.ValenceOrbital()    
            
    run_XRD_featurizer(mode, pattern_length_list):
        Calculates a powder X-ray diffraction pattern using matminer.featurizers.structure.XRDPowderPattern(pattern_length=pattern_length)
        
    run_yang_solid_solution_featurizer(mode):
        Calculates the yang solid solution information from matminer.featurizers.composition.YangSolidSolution() 
    """
    
    def __init__(self, structures_df, path):     
        self.structures_df = structures_df
        self.mode_list = ['structure', 'structure_A', 'structure_AM', 'structure_CAN', 'structure_CAMN', 'structure_A40', 'structure_AM40', 'structure_CAN40', 'structure_CAMN40']
        self.mode = 'structure'
        self.unique_atoms = []
        self.n_jobs = 63
        self.path = path
    
    def set_mode(self, mode):
        """
        Function to set the operating mode for the class. This mode tells the class which represenation of the structure to use
        (i.e., 'structure', 'structure_A', 'structure_AM', 'structure_CAN', 'structure_CAMN', 'structure_A40', 
        'structure_AM40', 'structure_CAN40', 'structure_CAMN40')

        Parameters
        ----------
        mode : str
            The class can operate in 9 modes. This str is used to set the mode attribute. 
            
        Raises
        ------
        Exception
            If the mode is not supported. 
        """
        if mode in self.mode_list:
            self.mode = mode
        else:
            raise Exception('The mode \'{}\' is not supported.'.format(mode))
    
    def calculate_unique_atoms(self, mode):
        """
        Function to identify the unique atoms that exist for a given mode. Then sets the 
        unique_atoms attribute so that it can be used by the SOAP featurizer. 

        Parameters
        ----------
        mode : str
            The class can operate in 9 modes. This str is used to set the mode attribute. 
        """
        self.set_mode(mode)
        self.unique_atoms = []
        for structure in tqdm.tqdm(self.structures_df[mode]):
            for num in structure.symbol_set:
                if num not in self.unique_atoms:
                    self.unique_atoms.append(num)
        self.unique_atoms = np.sort(self.unique_atoms)

    """ 
    Featurizer functions (and any helpers) are listed below this line in alphabetical order.
    Because calculated feature representations can be quite large they are directly saved into the 'features' repository. 
    """
    def run_atomic_packing_efficiency_featurizer(self, mode):
        """
        Function to run the atomic packing efficiency featurizer.
        Saves the files with the prefix "ape" and a suffix indicating the mode.

        Parameters
        ----------
        mode : str
            The class can operate in 9 modes. This str is used to set the mode attribute. 
        """
        self.set_mode(mode)
        ape_featurizer = mm_composition.AtomicPackingEfficiency()
        ape_featurizer_result = np.array(self.structures_df[mode].progress_apply(lambda x: ape_featurizer.featurize(x.composition)).values.tolist())
        np.save('{}/ape_features_mode-{}'.format(self.path, self.mode), ape_featurizer_result)
        
    def run_band_center_featurizer(self, mode):
        """
        Function to run the band center featurizer.
        Saves the files with the prefix "bc" and a suffix indicating the mode.

        Parameters
        ----------
        mode : str
            The class can operate in 9 modes. This str is used to set the mode attribute. 
        """
        self.set_mode(mode)
        bc_featurizer = mm_composition.BandCenter()
        bc_featurizer_result = np.array(self.structures_df[mode].progress_apply(lambda x: bc_featurizer.featurize(x.composition)).values.tolist())
        np.save('{}/bc_features_mode-{}'.format(self.path, self.mode), bc_featurizer_result)

    def run_bond_fraction_featurizer(self, mode):
        """
        Function to run the bond fraction featurizer.
        Saves the files with the prefix "bf" and a suffix indicating the mode.

        Parameters
        ----------
        mode : str
            The class can operate in 9 modes. This str is used to set the mode attribute. 
        """
        self.set_mode(mode)
        bf_featurizer = mm_structure.BondFractions()
        bf_featurizer.fit(self.structures_df[self.mode])
        bf_featurizer.set_n_jobs = self.n_jobs
        bf_featurizer_result = bf_featurizer.featurize_many(self.structures_df[self.mode], ignore_errors=True)
        np.save('{}/bf_features_mode-{}'.format(self.path, self.mode), bf_featurizer_result)
      
    def run_chemical_ordering_featurizer(self, mode):
        """
        Function to run the chemical ordering featurizer.
        Saves the files with the prefix "co" and a suffix indicating the mode.

        Parameters
        ----------
        mode : str
            The class can operate in 9 modes. This str is used to set the mode attribute. 
        """
        self.set_mode(mode)
        co_featurizer = mm_structure.ChemicalOrdering()
        co_featurizer.fit(self.structures_df[self.mode])
        co_featurizer.set_n_jobs = self.n_jobs
        co_featurizer_result = co_featurizer.featurize_many(self.structures_df[self.mode], ignore_errors=True)
        np.save('{}/co_features_mode-{}'.format(self.path, self.mode), co_featurizer_result)

    def run_density_featurizer(self, mode):
        """
        Function to run the density featurizer.
        Saves the files with the prefix "density" and a suffix indicating the mode.

        Parameters
        ----------
        mode : str
            The class can operate in 9 modes. This str is used to set the mode attribute. 
        """
        self.set_mode(mode)
        density_featurizer = mm_structure.DensityFeatures(("density", "vpa", "packing fraction"))
        density_featurizer.fit(self.structures_df[self.mode])
        density_featurizer.set_n_jobs = self.n_jobs
        density_featurizer_result = density_featurizer.featurize_many(self.structures_df[self.mode], ignore_errors=True)
        np.save('{}/density_features_mode-{}'.format(self.path, self.mode), density_featurizer_result)

    def end_featurizer_helper(self, structure, end_featurizer):
        """
        A helper function for the run_electron_negativity_difference_featurizer() function.
        The helper catches any value errors and returns a usable represntation. 
        This function is intended to be run using the pandas.DataFrame.apply() method

        Parameters
        ----------
        structure : pymatgen.core.structure
            A pymatgen structure
            
        end_featurizer : matminer.featurizers.composition.ElectronegativityDiff()
            The featurizer from matminer. 
        """
        try: 
            return end_featurizer.featurize(structure.composition)
        except ValueError:
            return [0, 0, 0, 0, 0]
        except:
            return [Nan, Nan, Nan, Nan, Nan]
        
    def run_electron_negativity_difference_featurizer(self, mode):
        """
        Function to run the electron negativity difference featurizer.
        Saves the files with the prefix "end" and a suffix indicating the mode.

        Parameters
        ----------
        mode : str
            The class can operate in 9 modes. This str is used to set the mode attribute. 
        """
        self.set_mode(mode)
        end_featurizer = mm_composition.ElectronegativityDiff()
        end_featurizer_result = np.array(self.structures_df[self.mode].progress_apply(self.end_featurizer_helper, end_featurizer=end_featurizer).values.tolist())
        np.save('{}/end_features_mode-{}'.format(self.path, self.mode), end_featurizer_result)

    def run_ewald_energy_featurizer(self, mode):
        """
        Function to run the Ewald energy featurizer.
        Saves the files with the prefix "ee" and a suffix indicating the mode.

        Parameters
        ----------
        mode : str
            The class can operate in 9 modes. This str is used to set the mode attribute. 
        """
        self.set_mode(mode)
        ee_featurizer = mm_structure.EwaldEnergy()
        ee_featurizer.fit(self.structures_df[self.mode])
        ee_featurizer.set_n_jobs = self.n_jobs
        ee_featurizer_result = ee_featurizer.featurize_many(self.structures_df[self.mode], ignore_errors=True)
        np.save('{}/ee_features_mode-{}'.format(self.path, self.mode), ee_featurizer_result)

    def run_global_instability_index_featurizer(self, mode, rcut_list):
        """
        Function to run the global instability index featurizer.
        The function will generate a feature for each radial cutoff that is passed in the list: rcut_list
        Saves the files with the prefix "gii" and a suffix indicating the mode. 

        Parameters
        ----------
        mode : str
            The class can operate in 9 modes. This str is used to set the mode attribute. 
        
        rcut_list : list
            A list containing one or more different radial cutoff (in angstroms)
        """
        self.set_mode(mode)
        for rcut in rcut_list:
            gii_featurizer = mm_structure.GlobalInstabilityIndex(r_cut=rcut)
            gii_featurizer.fit(self.structures_df[self.mode])
            gii_featurizer.set_n_jobs = self.n_jobs
            gii_featurizer_result = gii_featurizer.featurize_many(self.structures_df[self.mode], ignore_errors=True)
            np.save('{}/gii_features_rcut-{}_mode-{}'.format(self.path, rcut, self.mode), gii_featurizer_result)

    def run_jarvis_cfid_featurizer(self, mode):
        """
        Function to run the Jarvis CFID featurizer.
        Saves the files with the prefix "jc" and a suffix indicating the mode.

        Parameters
        ----------
        mode : str
            The class can operate in 9 modes. This str is used to set the mode attribute. 
        """
        self.set_mode(mode)
        jcfid_featurizer = mm_structure.JarvisCFID()
        jcfid_featurizer.fit(self.structures_df[self.mode])
        jcfid_featurizer.set_n_jobs = self.n_jobs
        jcfid_featurizer_result = jcfid_featurizer.featurize_many(self.structures_df[self.mode], ignore_errors=True)
        np.save('{}/jcfid_features_mode-{}'.format(self.path, self.mode), jcfid_featurizer_result)

    def run_maximum_packing_efficiency_featurizer(self, mode):
        """
        Function to run the maximum packing efficiency featurizer.
        Saves the files with the prefix "mpe" and a suffix indicating the mode.

        Parameters
        ----------
        mode : str
            The class can operate in 9 modes. This str is used to set the mode attribute. 
        """
        self.set_mode(mode)
        mpe_featurizer = mm_structure.MaximumPackingEfficiency()
        mpe_featurizer.fit(self.structures_df[self.mode])
        mpe_featurizer.set_n_jobs = self.n_jobs
        mpe_featurizer_result = mpe_featurizer.featurize_many(self.structures_df[self.mode], ignore_errors=True)
        np.save('{}/mpe_features_mode-{}'.format(self.path, self.mode), mpe_featurizer_result)

    def run_meredig_featurizer(self, mode):
        """
        Function to run the MereDig featurizer.
        Saves the files with the prefix "md" and a suffix indicating the mode.

        Parameters
        ----------
        mode : str
            The class can operate in 9 modes. This str is used to set the mode attribute. 
        """
        self.set_mode(mode)
        md_featurizer = mm_composition.Meredig()
        md_featurizer_result = np.array(self.structures_df[mode].progress_apply(lambda x: md_featurizer.featurize(x.composition)).values.tolist())
        np.save('{}/md_features_mode-{}'.format(self.path, self.mode), md_featurizer_result)
        
    def run_orbital_field_matrix_featurizer(self, mode):
        """
        Function to run the orbital field matrix featurizer.
        Saves the files with the prefix "ofm" and a suffix indicating the mode.

        Parameters
        ----------
        mode : str
            The class can operate in 9 modes. This str is used to set the mode attribute. 
        """
        self.set_mode(mode)
        ofm_featurizer = mm_structure.OrbitalFieldMatrix(period_tag=True)
        ofm_featurizer.fit(self.structures_df[self.mode])
        ofm_featurizer.set_n_jobs = self.n_jobs
        ofm_featurizer_result = ofm_featurizer.featurize_many(self.structures_df[self.mode], ignore_errors=True)
        np.save('{}/ofm_features_mode-{}'.format(self.path, self.mode), ofm_featurizer_result)
        
    def run_oxidation_states_featurizer(self, mode):
        """
        Function to run the oxidation states featurizer.
        Saves the files with the prefix "os" and a suffix indicating the mode.

        Parameters
        ----------
        mode : str
            The class can operate in 9 modes. This str is used to set the mode attribute. 
        """
        self.set_mode(mode)
        os_featurizer = mm_composition.OxidationStates()
        os_featurizer_result = np.array(self.structures_df[mode].progress_apply(lambda x: os_featurizer.featurize(x.composition)).values.tolist())
        np.save('{}/os_features_mode-{}'.format(self.path, self.mode), os_featurizer_result)
        
    def run_rdf_featurizer(self, mode, cutoff_list, bin_size_list):
        """
        Function to run the radial distribution function featurizer.
        This function will generate an rdf feature for all the combinations of radial cutoffs and bin sizes that are passed
        into cutoff_list and bin_size_list.
        Saves the files with the prefix "rdf" and a suffix indicating the mode.

        Parameters
        ----------
        mode : str
            The class can operate in 9 modes. This str is used to set the mode attribute. 
            
        cutoff_list : list
            A list containing all the desired radial cutoffs. 
        
        bin_size_list : list
            A list containing all the desired bin sizes. 
        
        """
        self.set_mode(mode)
        
        # iteratre over the cutoff_list and the bin_size_list
        for cutoff in cutoff_list:
            for bin_size in bin_size_list:
                rdf_featurizer = mm_structure.RadialDistributionFunction(cutoff=cutoff, bin_size=bin_size)
                rdf_featurizer.fit(self.structures_df[self.mode])
                rdf_featurizer.set_n_jobs = self.n_jobs
                rdf_featurizer_result = rdf_featurizer.featurize_many(self.structures_df[self.mode], ignore_errors=True)
                
                # capture errors and fill with zeroes
                error = 0
                radial_recreate = []
                for row in rdf_featurizer_result:
                    try:
                        radial_recreate.append(row[0]['distribution'].flatten())
                    except:
                        error+=1
                        radial_recreate.append([0]*int(cutoff/bin_size))
                print("There were {} errors when using mode: {} with cutoff={} and bin_size-{}. Filling those rows with zeroes.".format(error, self.mode, cutoff, bin_size))
                np.save('{}/rdf_features_cutoff-{}_binsize-{}_mode-{}'.format(self.path, cutoff, bin_size, self.mode), radial_recreate)

    def run_sine_coulomb_featurizer(self, mode):
        """
        Function to run the sine coulomb featurizer.
        Saves the files with the prefix "scm" and a suffix indicating the mode.

        Parameters
        ----------
        mode : str
            The class can operate in 9 modes. This str is used to set the mode attribute. 
        """
        self.set_mode(mode)
        scm_featurizer = mm_structure.SineCoulombMatrix()
        scm_featurizer.fit(self.structures_df[self.mode])
        scm_featurizer.set_n_jobs = self.n_jobs
        scm_featurizer_result = scm_featurizer.featurize_many(self.structures_df[self.mode], ignore_errors=True)
        np.save('{}/scm_features_mode-{}'.format(self.path, self.mode), scm_featurizer_result)
                
    def run_SOAP(self, mode, rcut_list, nmax_list, lmax_list, average):
        """
        Function to run the smooth overlap of atomic position featurizer.
        The function will automatically generate and save representations for all possible combinations 
        of the integers in rcut_list, nmax_list, and lmax_list. 
        Saves the files with the prefix "SOAP" and a suffix indicating the mode.

        Parameters
        ----------
        mode : str
            The class can operate in 9 modes. This str is used to set the mode attribute. 
            
        rcut_list : list
            A list containing all the desired radial cutoffs in angstroms. 
        
        nmax_list : list
            A list containing all the desired radial basis functions.
        
        lmax_list : list
            A list containing all the desired values for maximum degree of spherical harmonics. 
        
        average : str
            The averaging strategy used. Either 'inner' or 'outer'.
        """

        self.calculate_unique_atoms(mode)
        
        # iterate over all of the rcut, nmax and lmax values
        for rcut in rcut_list:
            for nmax in nmax_list:
                for lmax in lmax_list:
                    average_soap = SOAP(
                        species=self.unique_atoms,
                        r_cut=rcut,
                        n_max=nmax,
                        l_max=lmax,
                        periodic=True,
                        average=average,
                        sparse=True
                    )
                    ase_structures = self.structures_df[mode].progress_apply(AAA.get_atoms).to_numpy()
                    average_soap_data = average_soap.create(ase_structures, n_jobs=31, verbose=False)
                    pairings = np.concatenate([np.r_[average_soap.get_location(("S", x))] for x in self.unique_atoms])
                    print(type(average_soap_data[:,pairings]))
                    print(type(average_soap_data[:,pairings].todense()))
                    np.save('{}/SOAP_features_partialS_{}_rcut-{}_nmax-{}_lmax-{}_mode-{}'.format(self.path, average, rcut, nmax, lmax, mode), average_soap_data[:,pairings].todense())

    def run_structural_complexity_featurizer(self, mode):
        """
        Function to run the structural complexity featurizer.
        Saves the files with the prefix "sc" and a suffix indicating the mode.

        Parameters
        ----------
        mode : str
            The class can operate in 9 modes. This str is used to set the mode attribute. 
        """
        self.set_mode(mode)
        sc_featurizer = mm_structure.StructuralComplexity()
        sc_featurizer.fit(self.structures_df[self.mode])
        sc_featurizer.set_n_jobs = self.n_jobs
        sc_featurizer_result = sc_featurizer.featurize_many(self.structures_df[self.mode], ignore_errors=True)
        np.save('{}/sc_features_mode-{}'.format(self.path, self.mode), sc_featurizer_result)
                    
    def run_structural_heterogeneity_featurizer(self, mode):
        """
        Function to run the structural heterogeneity featurizer.
        Saves the files with the prefix "sh" and a suffix indicating the mode.

        Parameters
        ----------
        mode : str
            The class can operate in 9 modes. This str is used to set the mode attribute. 
        """
        self.set_mode(mode)
        sh_featurizer = mm_structure.StructuralHeterogeneity()
        sh_featurizer.fit(self.structures_df[self.mode])
        sh_featurizer.set_n_jobs = self.n_jobs
        sh_featurizer_result = sh_featurizer.featurize_many(self.structures_df[self.mode], ignore_errors=True)
        np.save('{}/sh_features_mode-{}'.format(self.path, self.mode), sh_featurizer_result)

    def run_valence_orbital_featurizer(self, mode):
        """
        Function to run the valence orbital featurizer.
        Saves the files with the prefix "vo" and a suffix indicating the mode.

        Parameters
        ----------
        mode : str
            The class can operate in 9 modes. This str is used to set the mode attribute. 
        """
        self.set_mode(mode)
        vo_featurizer = mm_composition.ValenceOrbital()
        vo_featurizer_result = np.array(self.structures_df[mode].progress_apply(lambda x: vo_featurizer.featurize(x.composition)).values.tolist())
        np.save('{}/vo_features_mode-{}'.format(self.path, self.mode), vo_featurizer_result)        

    def run_XRD_featurizer(self, mode, pattern_length_list):
        """
        Function to run the powder XRD featurizer.
        The function will automatically generate and save representations for every pattern length
        value that is contained in the list: pattern_length_list.
        Saves the files with the prefix "xrd" and a suffix indicating the mode.

        Parameters
        ----------
        mode : str
            The class can operate in 9 modes. This str is used to set the mode attribute. 
            
        pattern_length_list : list
            A list containing all the desired values for the pattern length. 
        """
        self.set_mode(mode)
        for pattern_length in pattern_length_list:
            xrd_featurizer = mm_structure.XRDPowderPattern(pattern_length=pattern_length)
            xrd_featurizer.fit(self.structures_df[self.mode])

            xrd_featurizer.set_n_jobs = self.n_jobs
            xrd_featurizer_result = xrd_featurizer.featurize_many(self.structures_df[self.mode], ignore_errors=True)
            np.save('{}/xrd_features_pattern_length-{}_mode-{}'.format(self.path, pattern_length, self.mode), xrd_featurizer_result)        

    def run_yang_solid_solution_featurizer(self, mode):
        """
        Function to run the yang solid solution featurizer.
        Saves the files with the prefix "yss" and a suffix indicating the mode.

        Parameters
        ----------
        mode : str
            The class can operate in 9 modes. This str is used to set the mode attribute. 
        """
        self.set_mode(mode)
        yss_featurizer = mm_composition.YangSolidSolution()
        yss_featurizer_result = np.array(self.structures_df[mode].progress_apply(lambda x: yss_featurizer.featurize(x.composition)).values.tolist())
        np.save('{}/yss_features_mode-{}'.format(self.path, self.mode), yss_featurizer_result)

    # def run_structural_heterogeneity_featurizer(self, mode):
    #     """
    #     Function to run StructuralHeterogeneity featurizer.
    #     Saves the files with the prefix "sh" and a suffix indicating the mode.

    #     Parameters
    #     ----------
    #     mode : str
    #         The class can operate in 9 modes. This str is used to set the mode attribute. 
    #     """
    #     self.set_mode(mode)

    #     heterogeneity_featurizer = mm_composition.StructuralHeterogeneity()
    #     heterogeneity_featurizer_result = np.array(self.structures_df[mode].progress_apply(lambda x: heterogeneity_featurizer.featurize(x.composition)).values.tolist())

    #     np.save('{}/ht_features_mode-{}'.format(self.path, self.mode), heterogeneity_featurizer_result)