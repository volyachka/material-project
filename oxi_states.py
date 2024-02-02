def structure_simplifications(structure_input, simplification_dict):
    """
    Function intended to be used with pandas.DataFrame.apply()

    The function takes a structure and then simplifies it based on the contents of simplification_dict.
    The simplified structure is returned. 


    Parameters
    ----------
    structure_input : pymatgen.core.structure
        a pymatgen structure file
        
    simplification_dict : dictionary
        A dictionary containing boolean values for the keys: 'C', 'A', 'M', 'N', '40'

    Returns
    ------
    structure: pymatgen.core.structure
        the simplified structure
    """
    
    # copy the structure in case modification fails
    structure = structure_input.copy()
    
    # create lists to keep track of the indices for the different atom types: cation, anion, mobile, neutral
    cation_list = []
    anion_list = []
    mobile_list = []
    neutral_list = []
    
    # create list to keep track of which atoms will be removed
    removal_list = []
    
    # integer to keep track of how to scale the lattice (for the representations that end in '40')
    scaling_counter = 0
    for idx, site in enumerate(structure):
        # grab the element name at the site
        element = Element(site.species.elements[0])
        # grab the charge at the site
        charge = Element(site.species.elements[0]).common_oxidation_states[0]
        # if the site is the mobile atom
        if element == 'Li':
            mobile_list.append(idx)
        else:
            # if the site holds a neutral atom
            if charge == 0:
                neutral_list.append(idx)
                scaling_counter+=1
                structure.replace(idx, Species("Mg", oxidation_state=charge))
            # if the site holds a cation
            elif charge>0:
                cation_list.append(idx)
                structure.replace(idx, Species("Al", oxidation_state=charge))
            # if the site holds an anion
            else:
                anion_list.append(idx)
                scaling_counter+=1
                structure.replace(idx, Species("S", oxidation_state=charge))
    
    # comparison to simplification_dict to decide which sites are removed
    if not simplification_dict['C']:
        removal_list += cation_list     
    if not simplification_dict['A']:
        removal_list += anion_list                
    if not simplification_dict['M']:
        removal_list += mobile_list
    if not simplification_dict['N']:
        removal_list += neutral_list
    # Special cases for the structures_A and structures_CAN representations
    # Some structures have only Li. For these we are going to handle them as anions (because every representations includes anions)
    if len(structure) == len(mobile_list):
        if not simplification_dict['M']:
            for idx in mobile_list:
                structure.replace(idx, Species("S", oxidation_state=charge))
    
    # Some structures have only neutrals or cations. For these we are going to handle them as anions (because every representation includes anions)
    elif len(structure) == len(removal_list):
        if len(neutral_list) > 0:
            for idx in neutral_list:
                structure.replace(idx, Species("S", oxidation_state=charge))
            structure.remove_sites(cation_list+mobile_list)
        elif len(mobile_list) > 0:
            for idx in mobile_list:
                structure.replace(idx, mg.Specie("S", oxidation_state=charge))
            structure.remove_sites(cation_list+neutral_list)
        elif len(cation_list) > 0:
            for idx in cation_list:
                structure.replace(idx, Species("S", oxidation_state=charge))
            structure.remove_sites(neutral_list+mobile_list)
    
    # otherwise just remove whatever is in the removal list
    else:
        structure.remove_sites(removal_list)
    # if simplification_dict indicates that the lattice should be scaled
    if simplification_dict['40']:              
        if scaling_counter > 0:
            structure.scale_lattice(40*scaling_counter)
    
    return structure



def apply_charge_decoration(structure):
    """
    Function intended to be used with pandas.DataFrame.apply()

    For each structure passed into the function, it sequentially attempts three charge decoration strategies:
    (1) a manual charge decoration using the oxidation_dictionary
    (2) charge decoration using OxidationStateDecorationTransformation
    (3) charge decoration using the built-in add_oxidation_state_by_guess() method
    
    If any of these strategies result in a structure with nearly zero charge (<= 0.5), then the decoration is accepted.

    Parameters
    ----------
    structure : pymatgen.core.structure
        a pymatgen structure file

    Returns
    ------
    structure: pymatgen.core.structure
        either charge decorated or not (if the decorations failed)
    """
    
    # try the manual decoration strategy
    temp_structure = structure.copy()
    try:
        manually_transformed_structure = oxidation_decorator.apply_transformation(temp_structure)
        if abs(manually_transformed_structure.charge) < 0.5:
            return manually_transformed_structure
    except:
        pass
    
    # try Pymatgen's auto decorator
    temp_structure = structure.copy()
    try:
        auto_transformed_structure = oxidation_auto_decorator.apply_transformation(temp_structure)
        if abs(auto_transformed_structure.charge) < 0.5:
            return auto_transformed_structure
    except:
        pass
    
    # allow Pymatgen to guess the oxidation states
    temp_structure = structure.copy()
    try:
        structure.add_oxidation_state_by_guess()
        return structure
    except:
        pass 

    return structure