# Utility program to generate the best-performing set of calibration parameter
# files for one or more basins of a calibration job. This can be ran at any
# point during a calibration (it does not require all iterations to be
# complete). The best iteration is chosen from the completed iterations logged
# in the database, using the same rule as the R calibration code (lowest
# objective function, with ties going to the later iteration). The parameter
# values for that iteration are then applied to the baseline parameter files
# in RUN.CALIB/BASELINE_PARAMETERS, following the same rules as
# adjust_parameters.py and generate_parameters.py.

import argparse
import configparser
import math
import os
import shutil
import sqlite3
import sys

import numpy as np
import pandas as pd
from netCDF4 import Dataset

# Parameters that scale the baseline field in soil_properties.nc.
SOIL_MULT_PARAMS = ['bexp', 'smcmax', 'dksat', 'cwpvt', 'vcmx25', 'mp', 'hvt',
                    'mfsno', 'AXAJ', 'BXAJ', 'XXAJ']

# Parameters that replace the baseline field in soil_properties.nc.
SOIL_ABS_PARAMS = ['slope', 'refkdt', 'rsurfexp', 'z0sno', 'ssi', 'snowretfac',
                   'swemx', 'tau0', 'graingrowth', 'extragrowth', 'dirtsoot',
                   'bats_cosz', 'bats_visnew', 'bats_nirnew', 'bats_visage',
                   'bats_nirage', 'bats_visdir', 'bats_nirdir', 'rsurfsnow',
                   'refsnowdens', 'frac_direct', 'frac_visible', 'scamax',
                   'unload_temp', 'unload_wind', 'maxsno_sp']

# Parameters that replace the baseline field in Fulldom.nc.
FULLDOM_ABS_PARAMS = {'lksatfac': 'LKSATFAC',
                      'retdeprtfac': 'RETDEPRTFAC',
                      'ovroughrtfac': 'OVROUGHRTFAC'}

# Parameters that scale the baseline field in HYDRO_TBL_2D.nc.
HYDRO_MULT_PARAMS = {'dksat': 'LKSAT',
                     'smcmax': 'SMCMAX1'}

# Parameters that replace the baseline values in GWBUCKPARM.nc.
GW_ABS_PARAMS = {'zmax': 'Zmax',
                 'expon': 'Expon',
                 'Loss': 'Loss',
                 'Coeff': 'Coeff'}

# Parameters that scale columns of CHANPARM.TBL (gridded routing only).
CHANPARM_COLS = {'Bw': 1, 'HLINK': 2, 'ChSSlp': 3, 'MannN': 4}

MISSING = -9999.0


def main(argv):
    parser = argparse.ArgumentParser(description='Utility for generating the best set of ' + \
                                     'calibration parameter files for a job, whether or not ' + \
                                     'the calibration has completed all iterations.')
    parser.add_argument('jobID', metavar='jobID', type=int, nargs='+',
                        help='Required job ID to pull from')
    parser.add_argument('inDB', metavar='inDB', type=str, nargs='+',
                        help='Required path to sqllite3 DB file.')
    parser.add_argument('--gages', type=str, nargs='+',
                        help='Optional list of gages to process. Defaults to all gages in the job.')
    parser.add_argument('--outDir', type=str, nargs='?',
                        help='Optional top-level output directory. Files are placed in ' + \
                             '<outDir>/<gage>. Defaults to <jobDir>/<gage>/RUN.CALIB/BEST_PARAMETERS.')
    parser.add_argument('--iteration', type=int, nargs='?',
                        help='Optional iteration to use instead of the best completed iteration.')
    parser.add_argument('--setupConfig', type=str, nargs='?',
                        help='Optional path to the job setup file. Defaults to <jobDir>/setup.config.')

    args = parser.parse_args()
    jobID = args.jobID[0]
    dbPath = args.inDB[0]

    if not os.path.isfile(dbPath):
        print("ERROR: Unable to locate DB file: " + dbPath)
        sys.exit(1)

    try:
        conn = sqlite3.connect(dbPath)
        dbCursor = conn.cursor()
    except:
        print("ERROR: Unable to connect to: " + dbPath)
        sys.exit(1)

    dbCursor.execute("SELECT \"Job_Directory\", num_iter from \"Job_Meta\" where \"jobID\"=?;", (jobID,))
    results = dbCursor.fetchone()
    if not results:
        print("ERROR: User-provided Job ID: " + str(jobID) + " returned no results.")
        sys.exit(1)
    jobDir = str(results[0])
    numIter = int(results[1])

    # The GW, channel routing, and mask options are not stored in the DB. Pull
    # them from the copy of the setup file the workflow places in the job directory.
    configPath = args.setupConfig if args.setupConfig else jobDir + "/setup.config"
    if not os.path.isfile(configPath):
        print("ERROR: Unable to locate job setup file: " + configPath + \
              ". Please provide one with --setupConfig.")
        sys.exit(1)
    config = configparser.ConfigParser()
    config.read(configPath)
    gwFlag = int(config.get('hydroPhysics', 'gwBaseSw'))
    chRtFlag = int(config.get('hydroPhysics', 'chanRtOpt'))
    enableMask = int(config.get('logistics', 'enableMask'))

    # Pull the gages (and their domain IDs) that are part of this job.
    dbCursor.execute("SELECT DISTINCT \"Calib_Stats\".\"domainID\", \"Domain_Meta\".gage_id from \"Calib_Stats\" " + \
                     "join \"Domain_Meta\" on \"Calib_Stats\".\"domainID\"=\"Domain_Meta\".\"domainID\" " + \
                     "where \"Calib_Stats\".\"jobID\"=?;", (jobID,))
    jobGages = dbCursor.fetchall()
    if len(jobGages) == 0:
        print("ERROR: No calibration basins found for Job ID: " + str(jobID))
        sys.exit(1)

    if args.gages:
        jobGageNames = [str(g[1]) for g in jobGages]
        for gage in args.gages:
            if gage not in jobGageNames:
                print("ERROR: Gage: " + gage + " is not part of Job ID: " + str(jobID))
                sys.exit(1)
        jobGages = [g for g in jobGages if str(g[1]) in args.gages]

    numFailed = 0
    for domainID, gage in jobGages:
        gage = str(gage)
        if args.outDir:
            outDir = args.outDir + "/" + gage
        else:
            outDir = jobDir + "/" + gage + "/RUN.CALIB/BEST_PARAMETERS"
        try:
            processGage(dbCursor, jobID, int(domainID), gage, jobDir, numIter, outDir,
                        args.iteration, gwFlag, chRtFlag, enableMask)
        except Exception as e:
            print("ERROR: Gage: " + gage + " - " + str(e))
            numFailed = numFailed + 1

    conn.close()

    if numFailed > 0:
        sys.exit(1)


def processGage(dbCursor, jobID, domainID, gage, jobDir, numIter, outDir,
                iterOverride, gwFlag, chRtFlag, enableMask):
    """
    Find the best completed iteration for a basin, extract its parameter values,
    and generate parameter files from the baseline parameter files.
    """
    dbCursor.execute("SELECT iteration, \"objfnVal\", best, complete from \"Calib_Stats\" " + \
                     "where \"jobID\"=? and \"domainID\"=? order by iteration;", (jobID, domainID))
    statRows = dbCursor.fetchall()

    # An iteration is usable if R logged it as complete with a real objective
    # function value. Basins with missing obs/model data are marked complete
    # with -9999 values, so those get filtered out here.
    completed = []
    dbBestIter = None
    for iteration, objF, best, complete in statRows:
        objF = toFloat(objF)
        if toFloat(complete) == 1.0 and objF is not None and objF != MISSING:
            completed.append((int(iteration), objF))
        if toFloat(best) == 1.0:
            dbBestIter = int(iteration)

    if len(completed) == 0:
        raise Exception("No completed calibration iterations found.")

    if iterOverride is not None:
        matches = [c for c in completed if c[0] == iterOverride]
        if len(matches) == 0:
            raise Exception("Iteration " + str(iterOverride) + " has not been completed.")
        bestIter, bestObj = matches[0]
    else:
        # Mirror the R workflow, which updates the best when F_new <= F_best.
        bestIter, bestObj = min(completed, key=lambda c: (c[1], -c[0]))
        if dbBestIter is not None and dbBestIter != bestIter:
            print("WARNING: Gage: " + gage + " - DB best flag is on iteration " + str(dbBestIter) + \
                  ", but the lowest objective function is on iteration " + str(bestIter) + ".")

    dbCursor.execute("SELECT \"paramName\", \"paramValue\" from \"Calib_Params\" " + \
                     "where \"jobID\"=? and \"domainID\"=? and iteration=?;", (jobID, domainID, bestIter))
    params = {}
    for paramName, paramValue in dbCursor.fetchall():
        paramValue = toFloat(paramValue)
        if paramValue is not None and paramValue != MISSING:
            params[str(paramName)] = paramValue

    if len(params) == 0:
        raise Exception("No parameter values found for iteration " + str(bestIter) + ".")

    print("Gage: " + gage + " - " + str(len(completed)) + " of " + str(numIter) + \
          " iterations complete. Using iteration " + str(bestIter) + \
          " (objective function = " + str(bestObj) + ").")

    calibDir = jobDir + "/" + gage + "/RUN.CALIB"
    baseDir = calibDir + "/BASELINE_PARAMETERS"
    if not os.path.isdir(baseDir):
        raise Exception("Baseline parameter directory: " + baseDir + " not found.")

    applyParams(params, baseDir, calibDir, outDir, gwFlag, chRtFlag, enableMask)

    # Write out a record of which iteration/values were used.
    with open(outDir + "/parms_best.tbl", 'w') as fileObj:
        fileObj.write('paramName,paramValue\n')
        for paramName, paramValue in params.items():
            fileObj.write(paramName + ", " + str(paramValue) + "\n")
    with open(outDir + "/best_iteration.txt", 'w') as fileObj:
        fileObj.write("jobID: " + str(jobID) + "\n")
        fileObj.write("gage: " + gage + "\n")
        fileObj.write("iteration: " + str(bestIter) + "\n")
        fileObj.write("objfnVal: " + str(bestObj) + "\n")
        fileObj.write("completedIterations: " + str(len(completed)) + "\n")
        fileObj.write("totalIterations: " + str(numIter) + "\n")


def applyParams(params, baseDir, calibDir, outDir, gwFlag, chRtFlag, enableMask):
    """
    Copy the baseline parameter files into the output directory and adjust them
    using the same rules as adjust_parameters.py and generate_parameters.py.
    """
    useGw = gwFlag == 1 or gwFlag == 4
    useChanParm = chRtFlag == 3

    if not os.path.isdir(outDir):
        os.makedirs(outDir)

    fileNames = ['Fulldom.nc', 'HYDRO_TBL_2D.nc', 'soil_properties.nc']
    if useGw:
        fileNames.append('GWBUCKPARM.nc')
    if useChanParm:
        fileNames.append('CHANPARM.TBL')
    for fileName in fileNames:
        shutil.copy(baseDir + "/" + fileName, outDir + "/" + fileName)

    for paramName in params:
        known = paramName in SOIL_MULT_PARAMS or paramName in SOIL_ABS_PARAMS or \
                paramName in FULLDOM_ABS_PARAMS or paramName in HYDRO_MULT_PARAMS or \
                paramName == 'nexp' or paramName in GW_ABS_PARAMS or paramName in CHANPARM_COLS
        if not known:
            print("WARNING: Parameter: " + paramName + " is not recognized and will not be applied.")

    if useChanParm:
        chanParmOrig = baseDir + "/CHANPARM.TBL"
        with open(chanParmOrig, 'r') as fileIn, open(outDir + "/CHANPARM.TBL", 'w') as fileOut:
            for lineNum, line in enumerate(fileIn):
                # First three lines are header.
                if lineNum < 3:
                    fileOut.write(line)
                    continue
                lineSplit = line.split(',')
                values = [float(v) for v in lineSplit[1:5]]
                for paramName, col in CHANPARM_COLS.items():
                    if paramName in params:
                        values[col - 1] = values[col - 1] * params[paramName]
                fileOut.write(lineSplit[0] + ", " + ", ".join(str(v) for v in values) + "\n")

    # Masks: 1 is where the baseline values are kept as is.
    keepCoarse = None
    keepFine = None
    keepGwIds = None
    if enableMask == 1:
        import xarray as xr
        with xr.open_dataset(calibDir + "/mask.coarse.tif", engine='rasterio') as mask:
            keepCoarse = np.flipud(np.asarray(mask['band_data'][0])) == 1
        with xr.open_dataset(calibDir + "/mask.fine.tif", engine='rasterio') as maskFine:
            keepFine = np.asarray(maskFine['band_data'][0]) == 1
        keepGwIds = pd.read_csv(calibDir + "/mask.GWBUCKET.csv", dtype={0: int})

    idSoil = Dataset(outDir + "/soil_properties.nc", 'a')
    idFullDom = Dataset(outDir + "/Fulldom.nc", 'a')
    idHydroTbl = Dataset(outDir + "/HYDRO_TBL_2D.nc", 'a')

    for paramName, value in params.items():
        if paramName in SOIL_MULT_PARAMS:
            adjustVar(idSoil.variables[paramName], value, True, keepCoarse)
        if paramName in SOIL_ABS_PARAMS:
            adjustVar(idSoil.variables[paramName], value, False, keepCoarse)
        if paramName in FULLDOM_ABS_PARAMS:
            adjustVar(idFullDom.variables[FULLDOM_ABS_PARAMS[paramName]], value, False, keepFine)
        if paramName in HYDRO_MULT_PARAMS:
            adjustVar(idHydroTbl.variables[HYDRO_MULT_PARAMS[paramName]], value, True, keepCoarse)
        if paramName == 'nexp':
            # The workflow replaces NEXP without a mask, but scales it with a mask.
            adjustVar(idHydroTbl.variables['NEXP'], value, enableMask == 1, keepCoarse)

    idSoil.close()
    idFullDom.close()
    idHydroTbl.close()

    if useGw:
        idGw = Dataset(outDir + "/GWBUCKPARM.nc", 'a')
        for paramName, varName in GW_ABS_PARAMS.items():
            if paramName not in params:
                continue
            if keepGwIds is None:
                idGw.variables[varName][:] = params[paramName]
            else:
                adjustInd = np.isin(idGw.variables['ComID'][:], keepGwIds, invert=True)
                idGw.variables[varName][adjustInd] = params[paramName]
        idGw.close()


def adjustVar(var, value, multiply, keep):
    """
    Scale (multiply=True) or replace (multiply=False) a NetCDF variable with a
    parameter value. Where keep is True, the baseline values are left as is.
    """
    base = var[:]
    if multiply:
        new = base * value
    else:
        new = base * 0 + value
    if keep is None:
        var[:] = new
    else:
        var[:] = np.where(keep, base, new)


def toFloat(value):
    """
    Convert a DB value to a float. Returns None for values that can't be
    converted or aren't finite (R can produce NA/NaN statistics).
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main(sys.argv[1:])
