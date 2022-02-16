# placeholder script for aggregating experiment results.

def aggregate_search_metadata(input_dir):
    """
    read all metadata data files in a directory, and concat them into a dataframe. read algorithm names from the
    metadata file basename. the metadata files can be produced by any SearchAbstractClass search method.

    we interpret all files that end with "_metadata.zip" as metadata files.

    return: the resulting dataframe
    """
    # TODO
    pass

def aggregate_dataseta_featuers(input_dir):
    """read data from a split directory and produce derived features. return a dataframe"""
    # TODO
    pass

def aggregate_all_data(df_metadata, df_datasets):
    """merge the metadata df and the dataset feature df. return the merged dataframe"""
    # TODO
    pass

if __name__=="__main__":
    pass