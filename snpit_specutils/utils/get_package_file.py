from importlib import resources


def get_package_file(filename):
    """
    Helper utiltity that looks in the package data for a reference file

    v1.0 returns the first file in the list, smarter logic might include
    version numbers of the files

    R. Ryan
    Jun 2026    
    """

    path = resources.files('snpit_specutils') / 'cal_files'
    files = path.rglob(filename)
    try:
        fullfile = next(files)
    except StopIteration:
        raise FileNotFoundError(f'{filename} not found in {path}.')
    
    return fullfile

    
