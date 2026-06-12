#!/usr/bin/env python
import argparse
import json
import time
import os
import cProfile
from coffea import processor, util
from coffea.nanoevents import NanoAODSchema

import sys
import ttbar_utilities.filing.filesets as uFiles
from utils.parse_rwgt_card import acquire_SM_rwgt_index

LST_OF_KNOWN_EXECUTORS = ["futures", "work_queue", "debug"]
LST_DATASET_FORMATS = ["sbi", "dctr"]

def main():
    parser = argparse.ArgumentParser(description='You can customize your run')
    parser.add_argument('project',  help='Specify project for formatting')
    parser.add_argument('jsonFile', nargs='?', help = 'Json file containing files and metadata')
    parser.add_argument('--executor', '-x', default='futures', help = 'Which executor to use')
    parser.add_argument('--prefix',   '-r', nargs='?', default='fnalEOS', help = 'Prefix or redirector to look for the files')
    parser.add_argument('--nworkers', '-n', default=4  , help = 'Number of workers')
    parser.add_argument('--chunksize','-s', default=50000  , help = 'Number of events per chunk')
    parser.add_argument('--nchunks',  '-c', default=None  , help = 'You can choose to run only a number of chunks')
    parser.add_argument('--outpath',  '-o', default=None, help = 'Path to put the output files')
    parser.add_argument('--postname', '-m', default='')
    parser.add_argument('--treename'      , default='Events', help = 'Name of the tree inside the files')
    parser.add_argument('--port', default='9123-9130', help = 'Specify the Work Queue port. An integer PORT or an integer range PORT_MIN-PORT_MAX.')
    # ARGUMENTS FOR DCTR RUNNING
    parser.add_argument('--rwgtcard', '-w', default=None, help='Name of the reweight card used to generate the EFT sample.')
    parser.add_argument('--processor','-p', default='tops', help='Specify processor')
    parser.add_argument('--validation', '-v', action="store_true", default=False, help='Should files be saved for validation?')
    # ARGUMENTS FOR SBI RUNNING
    parser.add_argument('--runFit', '-f', action="store_true", help='Run own fitting for structure coefficients')
    
    args        = parser.parse_args()
    project     = args.project
    jsonFile    = args.jsonFile
    executor    = args.executor
    prefix      = args.prefix
    nworkers    = int(args.nworkers)
    chunksize   = int(args.chunksize)
    nchunks     = int(args.nchunks) if not args.nchunks is None else args.nchunks
    outpath     = args.outpath
    postname    = args.postname
    treename    = args.treename
    proc        = args.processor
    runFit      = args.runFit

    if outpath is None:
        if project=="sbi":
            outpath = "/uscms_data/d3/ywkao/Outputs_sbi/pretraining"
        elif project=="dctr":
            outpath = "/uscms_data/d3/ywkao/Outputs_nlo/pretraining"
    if executor=="debug":
        outpath = os.path.join(outpath, "testing")
        
    #proc_file = proc+'.py'
    #print("\n running with processor: ", proc_file, '\n')
    if executor not in LST_OF_KNOWN_EXECUTORS:
        raise Exception(f"The \"{executor}\" executor is not known. Please specify an executor from the known executors ({LST_OF_KNOWN_EXECUTORS}). Exiting.")

    # PREPARE THE FILESET
    fileset = uFiles.construct_fileset([jsonFile], prefix=prefix)
    if executor=="debug": fileset = {list(fileset.keys())[0]: fileset[list(fileset.keys())[0]][:5]}
    # TODO: CHECK THAT ALL .ROOT HAVE THE SAME WC LIST
    if args.rwgtcard is not None:
        #TODO: if idx is list, check that the weights for each idx are the same (aka are all truly SM)
        SMweight_idx = acquire_SM_rwgt_index(args.rwgtcard)
    else:
        SMweight_idx = 0
    if project=='dctr' and args.validation:
        n = 10
        dataset_name = list(fileset.keys())[0]
        flist = fileset[dataset_name]
        validation_files = flist[0::n]
        fileset = {dataset_name: [f for i, f in enumerate(flist) if i%n!= 0]}

    # PREP THE EXECUTOR
    if executor == "debug":
        exec_instance = processor.IterativeExecutor()
        runner = processor.Runner(exec_instance, schema=NanoAODSchema, chunksize=chunksize, maxchunks=nchunks, skipbadfiles=True)
    elif executor == "futures":
        exec_instance = processor.FuturesExecutor(workers=nworkers)
        runner = processor.Runner(exec_instance, schema=NanoAODSchema, chunksize=chunksize, maxchunks=nchunks, skipbadfiles=True)
    elif executor=="lpcjq":
        # TODO: NEED TO COMPLETE THIS OPTION
        executor = processor.DaskExecutor()

    # PREP THE PROCESSOR
    tstart = time.time()
    if project=="sbi":
        import pretraining.ml_data as ml_data
        processor_instance = ml_data.SemiLepProcessor(runFit=runFit)
    elif proc == 'tops':
        import pretraining.df_maker as ml_data
        processor_instance = ml_data.SemiLepProcessor(SMweight_idx=SMweight_idx)
        print("Will run on top information")
    elif proc == 'vars':
        import pretraining.df_maker_variables as ml_data
        processor_instance = ml_data.SemiLepProcessor(SMweight_idx=SMweight_idx)
        print("Will run on top and save variable information")
    else:
        print('Processor to run not understood.')
        return 1
    output = runner(fileset, processor_instance, treename=treename)

    # PRINT processing stats
    dt = time.time() - tstart
    nevts_total = output['metadata']['InputEventCount']
    print(f'Processed {nevts_total} events in {dt:.2f} seconds ({nevts_total/dt:.2f} evts/sec).')
    if executor == "futures":
        print(f'Processing time: {dt:.2f} seconds with {nworkers} ({dt*nworkers:.2f} cpu overall)')

    # OUTPUT
    if not os.path.isdir(outpath): os.system("mkdir -p %s"%outpath)
    if project == "sbi":
        from torch import save, seed, float64
        from torch.utils.data import random_split, TensorDataset

        # Split samples
        train, test, val = random_split(TensorDataset(output['features'].get(), output['fit_coefs'].get()), [0.8, 0.1, 0.1])
        # Save the outputs
        print(f"\nSaving output in {outpath} ...")
        save(TensorDataset(train[:][0], train[:][1]), os.path.join(outpath,"train.p"))
        save(TensorDataset(test[:][0],  test[:][1]),  os.path.join(outpath,"test.p"))
        save(TensorDataset(val[:][0],   val[:][1]),   os.path.join(outpath,"validation.p"))
        
        output['metadata']['seed'] = seed()
        output['metadata']['nTrain'] = train[:][0].shape[0]
        output['metadata']['nTest']  = test[:][0].shape[0]
        output['metadata']['nVal']   = val[:][0].shape[0]
        util.save(output['metadata'], os.path.join(outpath,"metadata.coffea"))
    elif project == "dctr":
        import cloudpickle
        import gzip
        outname = jsonFile.split("/")[-1].replace(".json","")
        if args.executor == 'debug': outname += "_test"

        # Write the metadata to a json
        out_json_file = os.path.join(outpath, f'metadata_{outname}_{proc}_{postname}.json')
        print(f"Saving output in {out_json_file}...")
        #TODO: add wc list to jsons
        #TODO: add rwgt pts
        metadata = dict(output['metadata'])
        if args.validation: metadata['validation'] = validation_files
        with open(out_json_file,"w") as fout:
            json.dump(metadata, fout, indent=4)
        # Write the dataframe to a pickle file
        out_pkl_file = os.path.join(outpath, f'dataframe_{outname}_{proc}_{postname}.pkl.gz')
        print(f"\nSaving output in {out_pkl_file}...")
        with gzip.open(out_pkl_file, "wb") as fout:
            cloudpickle.dump(output['dataframe'].get(), fout)
            
    print("Done!")

if __name__ == '__main__':
    profile = False
    if profile:
        cProfile.run('main()', filename='profile.txt')
    else:
        main()
