from __future__ import print_function

import numpy as np

def list_runs(folder='./database/'):
    from os import listdir
    from os.path import isfile, join
    allfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    runfiles = [f for f in allfiles if f[0:4] == 'run_']
    datatypes = [filename.split('_')[-1].split('.')[ 0] for filename in runfiles]
    filetypes = [filename.split('_')[-1].split('.')[-1] for filename in runfiles]
    # Dates
    run_names = list(set([filename.split('_')[1] for filename in runfiles]))
    run_names.sort()
    # Sort filenames into runs
    runs = {run_name: {datatype: filename
                       for datatype, filename in zip(datatypes, runfiles) if run_name in filename.split('_')}
            for run_name in run_names}

    return run_names, runs

def import_run(run_name, folder='./database/'):
    # list runs
    run_names, runs = list_runs(folder)
    # import run
    from import_export import load_segments, load_features, load_matches, load_classes
    segments, sids         = load_segments(folder=folder, filename=runs[run_name]['segments'])
    features, fnames, fids = load_features(folder=folder, filename=runs[run_name]['features'])
    matches                = load_matches( folder=folder, filename=runs[run_name]['matches'])
    classes, cids          = load_classes( folder=folder, filename=runs[run_name]['classes']) if 'classes' in runs[run_name] else ([], [])
    # The loaded ids should match.
    assert len(sids) == len(fids)
    non_matching_ids = np.where(np.array(sids) != np.array(fids))[0]
    assert non_matching_ids.shape[0] == 0
    ids = sids
    assert len(ids) == len(segments)
    print("  Found " + str(len(ids)) + " segment ids")
    if len(cids) != 0:
      assert len(cids) == len(sids)
      non_matching_ids = np.where(np.array(sids) != np.array(cids))[0] if cids else np.array([])
      assert non_matching_ids.shape[0] == 0
      assert len(classes) == len(segments)
    print("  Found classes for " + str(len(classes)) + " segments")

    return segments, features, fnames, matches, classes, ids
