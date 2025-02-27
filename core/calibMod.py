# Module file for containing functions for iterations of the workflow. 

# Logan Karsten
# National Center for Atmospheric Research
# Research Applications Laboratory

import os,sys
from core import calibIoMod
from core import namelistMod
from core import statusMod
from core import errMod
import subprocess
import time
import psutil
import pandas as pd
from yaml import SafeDumper
import yaml
import warnings
warnings.filterwarnings("ignore")

def runTroute(statusData,staticData,db,gageID,gage,gageMeta,basinNum):

    if statusData.trouteFlag == 0:
        return
    groupNum = statusData.gageGroup[basinNum]
    runDir = statusData.jobDir + "/" + gage + "/RUN.CALIB/OUTPUT/"
    workDir = statusData.jobDir + "/" + gage + "/RUN.CALIB/"
    if not os.path.isdir(workDir):
        statusData.errMsg = "ERROR: " + workDir + " not found."
        raise Exception()
    if not os.path.isdir(runDir):
        statusData.errMsg = "ERROR: " + runDir + " not found."
        raise Exception()

    lockPath = workDir + "/TROUTE.LOCK"
    pidPath  = runDir + "/tpid.txt"
    trouteCompleteFlag = runDir + '/trouteFlag.COMPLETE'
    if os.path.exists(trouteCompleteFlag):
        print('Troute processing already complete.\n')
        
    else:
        yamlPath = runDir + '/troute_config.yaml'
        try:
            generateTrouteScript(statusData,runDir,yamlPath,basinNum)
        except:
            raise
    
        yamlFile = open(statusData.trouteConfig)
        yamlDict = yaml.load(yamlFile, Loader=yaml.FullLoader)

        begDate = staticData.bCalibDate
        endDate = staticData.eCalibDate

        runStatus = statusMod.walkModTroute(begDate,endDate,runDir,yamlDict)
        begDate = runStatus[0]
        endDate = runStatus[1]
        runFlag = runStatus[2]
    
        if runFlag == False:
        
            if not os.path.exists(trouteCompleteFlag):
                try:
                    open(trouteCompleteFlag, 'a').close()
                except:
                    statusData.errMsg = "Unable to create complete flag: " + trouteCompleteFlag
                    #errMod.sendMsg(statusData)
                    #errMod.errOut(statusData)
                    raise
        
        else:
            if os.path.isfile(lockPath):
                print("There is a lock path " + lockPath)
            
            else:
            #Ready to run TROUTE. Getting ready to do a warm start with the last spinup restart file.
                warm_restart_file = '%s/channel_restart_%s' %(runDir,staticData.bCalibDate.strftime('%Y%m%d%H%M'))
                if not os.path.exists(warm_restart_file):
                    cp_cmd = 'cp %s/channel_restart_%s %s/channel_restart_%s' %(runDir.replace('CALIB','SPINUP'),staticData.eSpinDate.strftime('%Y%m%d%H%M'),runDir,staticData.bCalibDate.strftime('%Y%m%d%H%M'))
                    try:
                        ret = os.system(cp_cmd)
                    except Exception as e:
                        statusData.genMsg = "ERROR: Unable to copy SPINUP lite restart file for gage: " + str(gage) + str(e)
                        
                        raise
                if str(gageMeta.lkFile) != '-9999':
                    warm_waterbody_restart_file = '%s/waterbody_restart_%s' %(runDir,staticData.bCalibDate.strftime('%Y%m%d%H%M'))
                    if not os.path.exists(warm_waterbody_restart_file):
                        cp_cmd = 'cp %s/waterbody_restart_%s %s/waterbody_restart_%s' %(runDir.replace('CALIB','SPINUP'),staticData.eSpinDate.strftime('%Y%m%d%H%M'),runDir,staticData.bCalibDate.strftime('%Y%m%d%H%M'))
                        try:
                            ret = os.system(cp_cmd)
                        except Exception as e:
                            statusData.errMsg = "ERROR: Unable to copy SPINUP lite waterbody restart file for gage: " + str(gage) + str(e)
                            raise
                    yamlDict['compute_parameters']['restart_parameters']['lite_waterbody_restart_file'] = runDir + '/waterbody_restart_' + begDate.strftime('%Y%m%d%H%M')

                yamlDict['compute_parameters']['restart_parameters']['start_datetime'] = begDate.strftime('%Y-%m-%d_%H:%M')
                yamlDict['compute_parameters']['restart_parameters']['lite_channel_restart_file'] = runDir + '/channel_restart_' + begDate.strftime('%Y%m%d%H%M')
                yamlDict['compute_parameters']['forcing_parameters']['qlat_input_folder'] = runDir
                yamlDict['compute_parameters']['forcing_parameters']['nts'] = (endDate - begDate).days * 24 * 12
                yamlDict['output_parameters']['lite_restart']['lite_restart_output_directory'] = runDir
                yamlDict['output_parameters']['chanobs_output']['chanobs_output_directory'] = runDir
                yamlDict['network_topology_parameters']['supernetwork_parameters']['geo_file_path'] = str(gageMeta.rtLnk)
                if str(gageMeta.lkFile) != '-9999':
                    yamlDict['network_topology_parameters']['waterbody_parameters']['level_pool']['level_pool_waterbody_parameter_file_path'] = str(gageMeta.lkFile)
                    yamlDict['compute_parameters']['restart_parameters']['wrf_hydro_waterbody_ID_crosswalk_file'] = str(gageMeta.lkFile)
                yamlDict['compute_parameters']['restart_parameters']['wrf_hydro_channel_ID_crosswalk_file'] = str(gageMeta.rtLnk)
                yamlDict['compute_parameters']['restart_parameters']['wrf_hydro_waterbody_crosswalk_filter_file'] = str(gageMeta.rtLnk)
                SafeDumper.add_representer(type(None), lambda dumper, value: dumper.represent_scalar(u'tag:yaml.org,2002:null', ''))

                with open(yamlPath, 'w') as output:
                    yaml.safe_dump(yamlDict, output, default_flow_style=False)

                cmd = runDir + "/run_troute.sh 1>" + runDir + "/troute_" + \
                      str(statusData.jobID) + "_" + str(gageID) + ".out" + \
                      ' 2>' + runDir + "/troute_" + str(statusData.jobID) + "_" + str(gageID) + ".err"
                try:
                    #p = os.system(cmd)
                    p = subprocess.Popen([cmd], shell=True)
                except:
                    statusData.errMsg = "ERROR: Unable to launch WRF-Hydro job for gage: " + str(gage)
                    raise
            
                with open(pidPath, "w") as fh:
                    fh.write(str(p.pid))

def runModel(statusData,staticData,db,gageID,gage,keySlot,basinNum,iteration,pbsJobId):
    """
    Generic function for running the model. Some basic information about
    the run directory, beginning date, ending dates, account keys,
    number of cores to use, etc will be used to compose BSUB/QSUB
    submision scripts, or run mpiexec/mpirun. 
    This function will walk the run directory 
    to determine where the model left off. If no restart files exist,
    then the function will assume the model has not ran at all. Both
    the LSM and hydro restart files must be present in order for the
    model to restart. This function will also check to see if parameter estimation
    /generation code needs to executed on Yellowstone compute nodes. 
    """
    # First check to make sure previous iteration's status is 1.0 (unless iteration 0).
    # This is to prevent the program from doing unecessary work. 
    if iteration > 0:
        if keySlot[basinNum,iteration-1] < 1.0:
            return
            
    # Compose directory paths for calibration/model simulations.
    runDir = statusData.jobDir + "/" + gage + "/RUN.CALIB/OUTPUT"
    workDir = statusData.jobDir + "/" + gage + "/RUN.CALIB"
    if not os.path.isdir(workDir):
        statusData.errMsg = "ERROR: " + workDir + " not found."
        raise Exception()
    if not os.path.isdir(runDir):
        statusData.errMsg = "ERROR: " + runDir + " not found."
        raise Exception()
        
    # Pull gage metadata for this particular basin.
    gageMeta = calibIoMod.gageMeta()
    try:
        gageMeta.pullGageMeta(statusData,db,gage,gageID)
    except:
        raise

    # Specify our S3 directory to hold calibration output files
    gageMeta.s3OutDir = staticData.s3TopOutDir + "/" + gage

    # Create the shell scripts that will use the MPI command specified by the user to run
    # or restart the model.
    runFile = runDir + "/run_WH.sh"
    rstFile = runDir + "/run_WH_Restart.sh"
    try:
        generateMpiCalibScript(statusData, int(gageID), int(basinNum), runDir, workDir, staticData)
    except:
        raise

    if os.path.isfile(runFile):
        os.remove(runFile)
    if os.path.isfile(rstFile):
        os.remove(rstFile)
    try:
        generateMpiScript(statusData, int(gageID), int(basinNum), runDir)
    except:
        raise
    try:
        generateMpiRstScript(statusData, int(gageID), int(basinNum), runDir)
    except:
        raise

    # Calculate datetime objects
    begDate = statusData.bCalibDate
    endDate = statusData.eCalibDate
        
    # Initialize status
    keyStatus = keySlot[basinNum,iteration]
    
    # Check to see if a model simulation is occurring.
    try:
        basinStatus = statusMod.checkBasJob(statusData,basinNum,pbsJobId)
    except:
        raise
        
    # Check to see if an R script calibration job is occurring.
    try:
        calibStatus = statusMod.checkCalibJob(statusData,basinNum,pbsJobId)
    except:
        raise
     
    # Establish paths to LOCK files,etc
    lockPath = workDir + "/RUN.LOCK"
    calibLockPath = workDir + "/CALIB.LOCK"
    calibCompleteFlag = workDir + "/CALIB_ITER.COMPLETE"
    missingFlag = workDir + "/CALC_STATS_MISSING"
    calibTbl = workDir + "/params_new.txt"
    statsTbl = workDir + "/params_stats.txt"
    rDataFile = workDir + "/proj_data.Rdata"
    trouteCompleteFlag = runDir + '/trouteFlag.COMPLETE'
    pidPath  = runDir + "/tpid.txt"

    # Initialize flags to False. These flags will help guide the workflow
    # in decision making. 
    runFlag = False
    runCalib = False
    if keyStatus == 1.0:
        # Calibration and simulation for this iteration has completed
        runFlag = False
        runCalib = False
        return

    if keyStatus == 0.65:
        if os.path.isfile(trouteCompleteFlag):
            keySlot[basinNum,iteration] = 0.75
            keyStatus = 0.75
            runCalib = True
        else:
            yamlFile = open(statusData.trouteConfig)
            yamlDict = yaml.load(yamlFile, Loader=yaml.FullLoader)
            runStatus = statusMod.walkModTroute(staticData.bCalibDate,staticData.eCalibDate,runDir,yamlDict)
            begDate = runStatus[0]
            endDate = runStatus[1]
            tRunFlag = runStatus[2]
            if tRunFlag == False:
                if not os.path.exists(trouteCompleteFlag):
                    print("1111")
                    try:
                        open(trouteCompleteFlag, 'a').close()
                    except Exception as e:
                        statusData.errMsg = "Unable to create complete flag: " + trouteCompleteFlag + str(e)
                        #errMod.errOut(statusData)
                        raise
            else:
                print("2222")
                if os.path.exists(pidPath):
                    print("23333")
                    with open(pidPath,"r") as ph:
                        tpid = int(ph.readlines()[0])
                        if not psutil.pid_exists(tpid):
                            #Troute is not actively running, so run troute again
                            runTroute(statusData,staticData,db,gageID,gage,gageMeta,int(basinNum))
                            keyStatus = -0.65
                            keySlot[basinNum,iteration] = -0.65

    if keyStatus == -0.65:
        if os.path.isfile(trouteCompleteFlag):
            keySlot[basinNum,iteration] = 0.75
            keyStatus = 0.75
            runCalib = True
        else:
            yamlFile = open(statusData.trouteConfig)
            yamlDict = yaml.load(yamlFile, Loader=yaml.FullLoader)
            runStatus = statusMod.walkModTroute(staticData.bCalibDate,staticData.eCalibDate,runDir,yamlDict)
            begDate = runStatus[0]
            endDate = runStatus[1]
            tRunFlag = runStatus[2]
            if tRunFlag == False:
               if not os.path.exists(trouteCompleteFlag):
                    try:
                        open(trouteCompleteFlag, 'a').close()
                    except Exception as e:
                        statusData.errMsg = "Unable to create complete flag: " + trouteCompleteFlag + str(e)
                        #errMod.errOut(statusData)
                        raise
            else:
                print("55555")
                with open(pidPath,"r") as ph:
                    tpid = int(ph.readlines()[0])
                    if not psutil.pid_exists(tpid):
                    #Troute failed a second time, so lock it
                        print("6666")
                        tLockPath = workDir + "/TROUTE.LOCK"
                        if os.path.isfile(tLockPath):
                            print("Lock file present")
                            keySlot[basinNum,iteration] = -0.66
                            keyStatus = -0.66
                        else:
                            open(tLockPath,'a').close()
                            print("created lock file")
                            keySlot[basinNum,iteration] = -0.66
                            keyStatus = -0.66
                            statusData.genMsg = "Unable to create complete flag because Troute didn't run successfully. Remove TROUTE.LOCK file: " + tLockPath
                            errMod.sendMsg(statusData)
                            #errMod.errOut(statusData) 

    if keyStatus == -0.66:
        #TROUTE LOCK FILE CREATED. 
        tLockPath = workDir + "/TROUTE.LOCK"
        print("Checking troute lock path")
        if not os.path.exists(tLockPath):
            #TROUTE LOCK FILE IS REMOVED, SO RESTART TROUTE
            print("no troute lock file found. Running troute again")
            runTroute(statusData,staticData,db,gageID,gage,gageMeta,int(basinNum))
            keySlot[basinNum,iteration] = 0.65
            keyStatus = 0.65 
               
    # For uncompleted simulations that are still listed as running.
    if keyStatus == 0.5:
        # If a model is running for this basin, continue and set keyStatus to 0.5
        if basinStatus:
            print("MODEL RUN FOUND")
            keySlot[basinNum,iteration] = 0.5
            keyStatus = 0.5
            runFlag = False
            runCalib = False
        else:
            # Either simulation has completed, or potentially crashed. Walk the run
            # run directory and see where the model is at based on RESTART files. 
            runStatus = statusMod.walkMod(begDate,endDate,runDir)
            begDate = runStatus[0]
            endDate = runStatus[1]
            runFlag = runStatus[2]
            if runFlag:
                # Model crashed as simulation is not complete but no processes are running.
                # Decided to comment this out for now. Not necessary, but may be nice for other
                # experiments. 
                print("MODEL HAS CRASHED ONCE")
                keySlot[basinNum,iteration] = -0.25
                keyStatus = -0.25
                runFlag = True
                runCalib = False
            else:
                # Model has completed. Set to 0.75, which indicates calibration code
                # needs to be ran.
                # Clean up any previous iteration calib files if they are around.
                try:
                    errMod.cleanCalib(statusData,workDir,runDir)
                    errMod.scrubParams(statusData,workDir,staticData)
                except:
                    raise
                print("MODEL HAS COMPLETED AND IS READY FOR PARAMETER GENERATION")
                if statusData.trouteFlag == 1:
                    runTroute(statusData,staticData,db,gageID,gage,gageMeta,int(basinNum))
                    keySlot[basinNum,iteration] = 0.65
                    keyStatus = 0.65
                    runFlag = False
                    runCalib = True
                else:
                    keySlot[basinNum,iteration] = 0.75
                    keyStatus = 0.75
                    runFlag = False
                    runCalib = True

    # For when the R/Python DDS code failed once.
    if keyStatus == -0.7 or keyStatus == -0.05:
        if calibStatus:
            print("Running calibration code")
            # Calib code running, upgrade status.
            if keyStatus == -0.70:
                keySlot[basinNum, iteration] = 0.90
                keyStatus = 0.90
                runFlag = False
                runCalib = False
            if keyStatus == -0.05:
                keySlot[basinNum, iteration] = 0.25
                keyStatus = 0.25
                runFlag = False
                runCalib = False
        else:
            if keyStatus == -0.05:
                # If calibration COMPLETE flag listed, upgrade status to 0.0 with runFlag on, signalling
                # to proceed with model simulation.
                if os.path.isfile(calibCompleteFlag):
                    # Copy parameter files to the DEFAULT directory
                    try:
                        calibIoMod.copyDefaultParms(statusData, runDir, gage, staticData)
                    except:
                        raise
                    # Enter in parameters for iteration update.
                    try:
                        db.logCalibParams(statusData, int(statusData.jobID), int(gageID), calibTbl, int(iteration))
                    except:
                        raise
                    print("FIRST CALIB/PARAM CODE DONE, READY TO RUN THE MODEL")
                    keySlot[basinNum, iteration] = 0.0
                    keyStatus = 0.0
                    runFlag = True
                    runCalib = False
                elif os.path.isfile(missingFlag):
                    # This is a unique situation where either an improper COMID (linkID) was passed to
                    # the R program, pulling NA from the model. Or, the observations file contains
                    # all missing values. For this, convey this to the user through a message, set the
                    # status for all iterations to 1.
                    statusData.genMsg = "WARNING: Either a bad COMID exists for this gage, or there are no " + \
                                        "observations for the evaluation period."
                    errMod.sendMsg(statusData)
                    # Copy parameter files to the DEFAULT directory
                    try:
                        calibIoMod.copyDefaultParms(statusData, runDir, gage, staticData)
                    except:
                        raise
                    # set the status for all iterations to 1.
                    try:
                        db.fillMisingBasin(statusData, int(statusData.jobID), int(gageID))
                    except:
                        raise
                    # Clean everything up.
                    try:
                        errMod.cleanCalib(statusData, workDir, runDir)
                        errMod.scrubParams(statusData, runDir, staticData)
                    except:
                        raise
                    keySlot[basinNum, :] = 1.0
                    keyStatus = 1.0
                    runFlag = False
                    runCalib = False
                else:
                    # This means the calibration code failed TWICE. Lock up and send an error message.
                    statusData.genMsg = "ERROR: 1st Calibration Scripts failed a second time for gage: " + \
                                        statusData.gages[basinNum] + \
                                        " Iteration: " + str(iteration) + " Failed. Please remove LOCK file: " + \
                                        calibLockPath
                    # Scrub calib-related files that were created as everything will need to be re-ran.
                    try:
                        errMod.scrubParams(statusData, runDir, staticData)
                    except:
                        raise
                    print("CALIB CODE HAS CRASHED TWICE.")
                    open(calibLockPath, 'a').close()
                    errMod.sendMsg(statusData)
                    keySlot[basinNum, iteration] = -0.1
                    keyStatus = -0.1
                    runFlag = False
                    runCalib = False
            if keyStatus == -0.70:
                # If calibration COMPLETE flag listed, upgrade status to 1.0, and make entry into
                # database as this iteration being completed.
                # Also scrub calib-related files (minus new parameters).
                if os.path.isfile(calibCompleteFlag):
                    try:
                        # If we are on the last iteration, no new parameters are created.
                        if int(iteration + 1) < int(statusData.nIter):
                            # The if statment is to handle the last iteration where no
                            # new parameters are generated at the end.
                            try:
                                db.logCalibParams(statusData, int(statusData.jobID), int(gageID), calibTbl,
                                                  int(iteration) + 1)
                            except:
                                raise
                        db.logCalibStats(statusData, int(statusData.jobID), int(gageID), str(gage), int(iteration),
                                         statsTbl, staticData)
                        errMod.cleanCalib(statusData, workDir, runDir)
                    except:
                        raise
                    print("CALIB/PARAM CODE COMPLETE")
                    keySlot[basinNum, iteration] = 1.0
                    keyStatus = 1.0
                    runFlag = False
                    runCalib = False
                elif os.path.isfile(missingFlag):
                    # This is a unique situation where either an improper COMID (linkID) was passed to
                    # the R program, pulling NA from the model. Or, the observations file contains
                    # all missing values. For this, convey this to the user through a message, set the
                    # status for all iterations to 1.
                    statusData.genMsg = "WARNING: Either a bad COMID exists for this gage, or there are no " + \
                                        "observations for the evaluation period."
                    errMod.sendMsg(statusData)
                    # set the status for all iterations to 1. This will force the workflow to skip
                    # over this basin in the future.
                    try:
                        db.fillMisingBasin(statusData, int(statusData.jobID), int(gageID))
                    except:
                        raise
                    # Clean everything up.
                    try:
                        errMod.cleanCalib(statusData, workDir, runDir)
                        errMod.scrubParams(statusData, runDir, staticData)
                    except:
                        raise
                    keySlot[basinNum, :] = 1.0
                    keyStatus = 1.0
                    runFlag = False
                    runCalib = False
                else:
                    # This means the calibration code failed TWICE. Lock up and send error message
                    statusData.genMsg = "ERROR: Calibration Scripts failed a second time for gage: " + \
                                        statusData.gages[basinNum] + \
                                        " Iteration: " + str(iteration) + " Failed. Please remove LOCKFILE: " + \
                                        calibLockPath
                    # Scrub calib-related files that were created as everything will need to be re-ran.
                    try:
                        errMod.cleanCalib(statusData, workDir, runDir)
                        errMod.scrubParams(statusData, runDir, staticData)
                    except:
                        raise
                    open(calibLockPath, 'a').close()
                    errMod.sendMsg(statusData)
                    keySlot[basinNum, iteration] = -0.75
                    keyStatus = -0.75
                    runFlag = False
                    runCalib = False
                
    # For when the model simulation has completed, but the calibration is still 
    # listed as running.
    if keyStatus == 0.90:
        print("CALIB/PARAM CODE IS RUNNING")
        # If the calibration is still running, keep status as 0.90.
        if calibStatus:
            keySlot[basinNum,iteration] = 0.90
            keyStatus = 0.90
            runFlag = False
            runCalib = False
        else:
            # If calibration COMPLETE flag listed, upgrade status to 1.0, and make entry into
            # database as this iteration being completed.
            # Also scrub calib-related files (minus new parameters).
            if os.path.isfile(calibCompleteFlag):
                try:
                    # If we are on the last iteration, no new parameters are created.
                    if int(iteration+1) < int(statusData.nIter):
                        # The if statment is to handle the last iteration where no 
                        # new parameters are generated at the end. 
                        try:
                            db.logCalibParams(statusData,int(statusData.jobID),int(gageID),calibTbl,int(iteration)+1)
                        except:
                            raise
                    db.logCalibStats(statusData,int(statusData.jobID),int(gageID),str(gage),int(iteration),statsTbl,staticData)
                    errMod.cleanCalib(statusData,workDir,runDir)
                except:
                    raise
                # If our S3 object has been initialized - Upload our calibration files for this basin to the bucket
                if staticData.s3Client is not None:
                    calibIoMod.calibOutputS3Upload(gage,staticData,gageMeta,workDir)

                print("CALIB/PARAM CODE COMPLETE")
                keySlot[basinNum,iteration] = 1.0
                keyStatus = 1.0
                runFlag = False
                runCalib = False
            elif os.path.isfile(missingFlag):
                # This is a unique situation where either an improper COMID (linkID) was passed to 
                # the R program, pulling NA from the model. Or, the observations file contains 
                # all missing values. For this, convey this to the user through a message, set the
                # status for all iterations to 1.
                statusData.genMsg = "WARNING: Either a bad COMID exists for this gage, or there are no " + \
                                    "observations for the evaluation period."
                errMod.sendMsg(statusData)
                # set the status for all iterations to 1. This will force the workflow to skip
                # over this basin in the future. 
                try:
                    db.fillMisingBasin(statusData,int(statusData.jobID),int(gageID))
                except:
                    raise
                # Clean everything up.
                try:
                    errMod.cleanCalib(statusData,workDir,runDir)
                    errMod.scrubParams(statusData,runDir,staticData)
                except:
                    raise
                keySlot[basinNum,:] = 1.0
                keyStatus = 1.0
                runFlag = False
                runCalib = False
            else:
                # This means the calibration code failed ONCE. We will allow one re-try, like we do with
                # the model code.
                #statusData.genMsg = "ERROR: Calibration Scripts failed for gage: " + statusData.gages[basinNum] + \
                #                    " Iteration: " + str(iteration) + " Failed. Please remove LOCKFILE: " + calibLockPath
                # Scrub calib-related files that were created as everything will need to be re-ran.
                #try:
                #    errMod.cleanCalib(statusData,workDir,runDir)
                #    errMod.scrubParams(statusData,runDir,staticData)
                #except:
                #    raise
                #open(calibLockPath,'a').close()
                #errMod.sendMsg(statusData)
                #statusData.genMsg = "Calibration Scripts failed ONCE for gage: " + statusData.gages[basinNum]
                #errMod.sendMsg(statusData)
                #runTroute(statusData,staticData,db,gageID,gage,gageMeta)
                print("CALIB CODE HAS CRASHED ONCE.")
                keySlot[basinNum, iteration] = -0.705
                keyStatus = -0.705
                runFlag = False
                runCalib = True
                #keySlot[basinNum,iteration] = -0.75
                #keyStatus = -0.75
                #runFlag = False
                #runCalib = False
                
    # For when the first calibration is running for the first iteration.
    if keyStatus == 0.25:
        print("FIRST SET OF CALIB/PARAM CODE RUNNING")
        # If calibration is still running, keep status as 0.10.
        if calibStatus:
            keySlot[basinNum,iteration] = 0.25
            keyStatus = 0.25
            runFlag = False
            runCalib = False
        else:
            # If calibration COMPLETE flag listed, upgrade status to 0.0 with runFlag on, signalling 
            # to proceed with model simulation. 
            if os.path.isfile(calibCompleteFlag):
                # Copy parameter files to the DEFAULT directory
                try:
                    calibIoMod.copyDefaultParms(statusData,runDir,gage,staticData)
                except:
                    raise
                # Enter in parameters for iteration update.
                try:
                    db.logCalibParams(statusData,int(statusData.jobID),int(gageID),calibTbl,int(iteration))
                except:
                    raise
                print("FIRST CALIB/PARAM CODE DONE, READY TO RUN THE MODEL")
                keySlot[basinNum,iteration] = 0.0
                keyStatus = 0.0
                runFlag = True
                runCalib = False
            elif os.path.isfile(missingFlag):
                # This is a unique situation where either an improper COMID (linkID) was passed to 
                # the R program, pulling NA from the model. Or, the observations file contains 
                # all missing values. For this, convey this to the user through a message, set the
                # status for all iterations to 1.
                statusData.genMsg = "WARNING: Either a bad COMID exists for this gage, or there are no " + \
                                    "observations for the evaluation period."
                errMod.sendMsg(statusData)
                # Copy parameter files to the DEFAULT directory
                try:
                    calibIoMod.copyDefaultParms(statusData,runDir,gage,staticData)
                except:
                    raise
                # set the status for all iterations to 1.
                try:
                    db.fillMisingBasin(statusData,int(statusData.jobID),int(gageID))
                except:
                    raise
                # Clean everything up.
                try:
                    errMod.cleanCalib(statusData,workDir,runDir)
                    errMod.scrubParams(statusData,runDir,staticData)
                except:
                    raise
                keySlot[basinNum,:] = 1.0
                keyStatus = 1.0
                runFlag = False
                runCalib = False
            else:
                # This means the calibration code failed ONCE. We will allow one re-try, like we do with
                #statusData.genMsg = "ERROR: 1st Calibration Scripts failed for gage: " + statusData.gages[basinNum] + \
                #                    " Iteration: " + str(iteration) + " Failed. Please remove LOCK file: " + calibLockPath
                # Scrub calib-related files that were created as everything will need to be re-ran.
                #try:
                #    errMod.scrubParams(statusData,runDir,staticData)
                #except:
                #    raise
                #statusData.genMsg = "1st Calibration Scripts failed ONCE for gage: " + statusData.gages[basinNum]
                #errMod.sendMsg(statusData)
                
                print("CALIB CODE HAS CRASHED ONCE.")
                #runTroute(statusData,staticData,db,gageID,gage,gageMeta)
                keySlot[basinNum, iteration] = -0.0505
                keyStatus = -0.0505
                runFlag = False
                runCalib = True
                #open(calibLockPath,'a').close()
                #errMod.sendMsg(statusData)
                #keySlot[basinNum,iteration] = -0.1
                #keyStatus = -0.1
                #runFlag = False
                #runCalib = False
           
    # For iterations that are ready for run. The first step is to clean out old model output
    # and fire off simulation with newly updated model parameter values. 
    if keyStatus == 0.0:
        if iteration > 0:
            # Iteration 0 needs to have an inital calibration procedure to generate
            # an initial parameter dataset.
            if os.path.isfile(lockPath):
                print("MODEL IS LOCKED")
                # Simulation was locked up. Set status to -1.0.
                keySlot[basinNum,iteration] = -1.0
                keyStatus = -1.0
                runFlag = False
                runCalib = False
            elif os.path.isfile(calibLockPath):
                print("CALIB IS LOCKED")
                # Calibration failed and locked directory up.
                keySlot[basinNum,iteration] = -0.75
                keyStatus = -0.75
                runFlag = False
                runCalib = False
            else:
                if basinStatus:
                    print("MODEL IS STILL RUNNING")
                    # Model is still running from previous instance of workflow. Allow it to continue.
                    keySlot[basinNum,iteration] = 0.5
                    keyStatus = 0.5
                    runFlag = False
                    runCalib = False
                else:
                    # Model is not running. Walk the model run directory and determine where
                    # it's at based on RESTART files. 
                    runStatus = statusMod.walkMod(begDate,endDate,runDir)
                    begDate = runStatus[0]
                    endDate = runStatus[1]
                    runFlag = runStatus[2]
                    if not runFlag:
                        # Model simulation completed before workflow was restarted. Ready to 
                        # move onto calibration R code. However.... to be safe, we are going
                        # to re-run the model. There are a couple cracks that occur when we assume
                        # this is the current model. If a previous model completed, but wasn't 
                        # completely wiped clean, this will screw up calibration statistics. 
                        # Cleanup any previous calib-related files that may be sitting around.
                        try:
                            errMod.cleanCalib(statusData,workDir,runDir)
                        except:
                            raise
                        print("FIRST CALIB CODE COMPLETE, READY TO RUN MODEL")
                        keySlot[basinNum,iteration] = 0.0
                        keyStatus = 0.0
                        runFlag = True
                        runCalib = False
                        if calibStatus:
                            # Model has completed, and calibration routines are currently being ran.
                            keySlot[basinNum,iteration] = 0.90
                            keyStatus = 0.90
                            runFlag = False
                            runCalib = False
                    else:
                        # Model is not running, but it's not complete. This means a simulation needs
                        # to be started or re-started.
                        # To be safe, cleanup any calib-related files (minus parameters) that
                        # may be lying around.
                        try:
                            errMod.cleanCalib(statusData,workDir,runDir)
                        except:
                            raise
                        print("READY TO RUN MODEL")
                        keySlot[basinNum,iteration] = 0.0
                        keyStatus = 0.0
                        runFlag = True
                        runCalib = False
        else:
            # Run calibration procedure to generate initial table of values to adjust
            # parameters with.
            if os.path.isfile(lockPath):
                print("MODEL IS LOCKED")
                # Simulation was locked up. Set status to -1.0.
                keySlot[basinNum,iteration] = -1.0
                keyStatus = -1.0
                runFlag = False
                runCalib = False
            elif os.path.isfile(calibLockPath):
                # Calibration failed and locked directory up.
                # Double check to make sure this is not the first calib that 
                # failed.
                if not basinStatus:
                    runStatus = statusMod.walkMod(begDate,endDate,runDir)
                    begDate = runStatus[0]
                    endDate = runStatus[1]
                    runFlag = runStatus[2]
                    if begDate == statusData.bCalibDate and runFlag:
                        # The first calib for iteration 1 failed.
                        print("FIRST CALIB CODE FAILED")
                        keySlot[basinNum,iteration] = -0.10
                        keyStatus = -0.10
                        runFlag = False
                        runCalib = False
                    else:
                        print("CALIB CODE FAILED")
                        # The main calib failed.
                        keySlot[basinNum,iteration] = -0.75
                        keyStatus = -0.75
                        runFlag = False
                        runCalib = False
            else:
                if basinStatus:
                    # This means that we are rebooting the program. Parameters were 
                    # adjusted already and the simulation for iteration 0 is underway
                    # still from a previous crash. This is a rare situation as statuses
                    # are updated in the DB dynamically. 
                    print("MODEL IS RUNNING")
                    keySlot[basinNum,iteration] = 0.5
                    keyStatus = 0.5
                    runFlag = False
                    runCalib = False
                else:
                    runStatus = statusMod.walkMod(begDate,endDate,runDir)
                    begDate = runStatus[0]
                    endDate = runStatus[1]
                    runFlag = runStatus[2]
                    if begDate == statusData.bCalibDate and runFlag:
                        # Model has not ran at all yet.
                        if calibStatus:
                            # First calibration taking place to prepare parameters for
                            # first iteration.
                            keySlot[basinNum,iteration] = 0.25
                            keyStatus = 0.25
                            runFlag = False
                            runCalib = False
                        else:
                            if os.path.isfile(calibCompleteFlag):
                                # First calibration completed. Ready to run the model.
                                # Cleanup any previou calib-related parameters (minus new parameter files)
                                print("FIRST CALIB CODE COMPLETE")
                                keySlot[basinNum,iteration] = 0.0
                                keyStatus = 0.0
                                runFlag = True
                                runCalib = False
                            else:
                                # Check for LOCK file, meaning the first calibration failed.
                                if os.path.isfile(calibLockPath):
                                    if os.path.isfile(rDataFile):
                                        os.remove(rDataFile)
                                    print("FIRST CALIB CODE LOCKED")
                                    keySlot[basinNum,iteration] = -0.10
                                    keyStatus = -0.10
                                    runFlag = False
                                    runCalib = False
                                else:
                                    # Run first calibration and parameter adjustment.
                                    # Clean all previous calib files that may be laying around.
                                    try:
                                        errMod.cleanCalib(statusData,workDir,runDir)
                                        errMod.scrubParams(statusMod,runDir,staticData)
                                    except:
                                        raise
                                    print("READY TO RUN FIRST CALIB CODE")
                                    #runTroute(statusData,staticData,db,gageID,gage,gageMeta)
                                    keySlot[basinNum,iteration] = 0.0
                                    keyStatus = 0.0
                                    runFlag = False
                                    runCalib = True
                    if begDate != statusData.bCalibDate and runFlag:
                        # This implies a general crash that did not result in the creation of
                        # a model LOCK file. In this case, set the status to 0.0 with the runFlag
                        # on to tell the workflow to skip the first calibration. This implies it has
                        # already occurred.
                        keySlot[basinNum,iteration] = 0.0
                        keyStatus = 0.0
                        runFlag = True
                        runCalib = False
                        if not runFlag:
                            # Model has completed.
                            if calibStatus:
                                # First calibration completed, model simulation completed, and second calibration
                                # underway.
                                print("CALIB CODE RUNNING")
                                keySlot[basinNum,iteration] = 0.90
                                keyStatus = 0.90
                                runFlag = False
                                runCalib = False
                            else:
                                # First calibration completed, model simulation completed, second calibration 
                                # needs to be ran.
                                # Clean any previou calib files that may be laying around.
                                try:
                                    errMod.cleanCalib(statusData,workDir,runDir)
                                    errMod.scrubParams(statusMod,runDir,staticData)
                                except:
                                    raise
                                print("MODEL COMPLETE, READY TO RUN CALIB CODE")
                                if statusData.trouteFlag == 1:
                                    runTroute(statusData,staticData,db,gageID,gage,gageMeta,int(basinNum))
                                    keySlot[basinNum,iteration] = 0.65
                                    keyStatus = 0.65
                                    runFlag = False
                                    runCalib = True
                                
                                else:
                                    keySlot[basinNum,iteration] = 0.75
                                    keyStatus = 0.75
                                    runFlag = False
                                    runCalib = True
                    if begDate == statusData.eCalibDate and not runFlag:
                        # Both the first calibration and model simulation completed. Ready for
                        # second (main) calibration.
                        try:
                            errMod.cleanCalib(statusData,workDir,runDir)
                            errMod.scrubParams(statusMod,runDir,staticData)
                        except:
                            raise
                        print("MODEL COMPLETE, READY TO RUN CALIB CODE")
                        if statusData.trouteFlag == 1:
                            runTroute(statusData,staticData,db,gageID,gage,gageMeta,int(basinNum))
                            keySlot[basinNum,iteration] = 0.65
                            keyStatus = 0.65
                            runFlag = False
                            runCalib = True
                        else:
                            keySlot[basinNum,iteration] = 0.75
                            keyStatus = 0.75
                            runFlag = False
                            runCalib = True
    # For when the model failed TWICE and is locked.
    if keyStatus == -1.0:
        # If LOCK file exists, no simulation will take place. File must be removed
        # manually by user.
        if os.path.isfile(lockPath):
            print("MODEL IS LOCKED")
            runFlag = False
            runCalib = False
            keySlot[basinNum,iteration] = -1.0
            keyStatus = -1.0
        else:
            # LOCK file was removed, upgrade status to 0.0 temporarily
            runStatus = statusMod.walkMod(begDate,endDate,runDir)
            begDate = runStatus[0]
            endDate = runStatus[1]
            runFlag = runStatus[2]
            if runFlag:
                print("READ TO RUN MODEL FOR FIRST ITERATION")
                keySlot[basinNum,iteration] = 0.0
                keyStatus = 0.0
                runFlag = True
                runCalib = False
            else:
                # Model sucessfully completed. Ready to move onto calibration R code
                # Remove any previous calibration files.
                try:
                    errMod.cleanCalib(statusData,workDir,runDir)
                    errMod.scrubParams(statusMod,runDir,staticData)
                except:
                    raise
                print("MODEL COMPLETE, READY TO RUN CALIB CODE")

                if statusData.trouteFlag == 1: 
                    runTroute(statusData,staticData,db,gageID,gage,gageMeta,int(basinNum))
                    keySlot[basinNum,iteration] = 0.65
                    keyStatus = 0.65
                    runFlag = False
                    runCalib = True
                else:
                    keySlot[basinNum,iteration] = 0.75
                    keyStatus = 0.75
                    runFlag = False
                    runCalib = True
                
    # For when calibration R code and parameter adjustment failed.
    if keyStatus == -0.75:
        # If LOCK file exists, no calibrations can take place. File must
        # be removed manually by user.
        if os.path.isfile(calibLockPath):
            # Cleanup any calibration related files.
            try:
                errMod.cleanCalib(statusData,workDir,runDir)
                errMod.scrubParams(statusMod,runDir,staticData)
            except:
                raise
            print("CALIB LOCKED")
            runFlag = False
            runCalib = False
            keySlot[basinNum,iteration] = -0.75
            keyStatus = -0.75
        else:
            # Cleanup any previous calib files that may be laying around.
            try:
                errMod.cleanCalib(statusData,workDir,runDir)
                errMod.scrubParams(statusMod,runDir,staticData)
            except:
                raise
            print("READY TO RUN CALIB CODE")
            # LOCK file was removed, upgrade status.
            if statusData.trouteFlag == 1:
                runTroute(statusData,staticData,db,gageID,gage,gageMeta,int(basinNum))
                keySlot[basinNum,iteration] = 0.65
                keyStatus = 0.65
                runFlag = False
                runCalib = True
            else:
                keySlot[basinNum,iteration] = 0.75
                keyStatus = 0.75
                runFlag = False
                runCalib = True        
    # For when the first calibration during the first iteration failed.
    if keyStatus == -0.1:
        # If LOCK file exists, no calibrations can take place. File must
        # be removed manually by user.
        if os.path.isfile(calibLockPath):
            if os.path.isfile(rDataFile):
                os.remove(rDataFile)
            print("FIRST CALIB CODE LOCKED")
            keySlot[basinNum,iteration] = -0.10
            keyStatus = -0.10
            runFlag = False
            runCalib = False
        else:
            # Cleanup any previous calibration files
            try:
                errMod.cleanCalib(statusData,workDir,runDir)
                errMod.scrubParams(statusMod,runDir,staticData)
            except:
                raise
            print("READY TO RUN FIRST CALIB CODE")
            # LOCK file was removed, upgrade status.
            #runTroute(statusData,staticData,db,gageID,gage,gageMeta)
            keySlot[basinNum,iteration] = 0.0
            runFlag = False
            runCalib = True
                
    # For when the model crashed ONCE
    if keyStatus == -0.5:
        if basinStatus:
            print("RUNNING MODEL")
            # Model is running again, upgrade status
            keySlot[basinNum,iteration] = 0.5
            keyStatus = 0.5
            runFlag = False
            runCalib = False
        else:
            runStatus = statusMod.walkMod(begDate,endDate,runDir)
            begDate = runStatus[0]
            endDate = runStatus[1]
            runFlag = runStatus[2]
            if runFlag:
                # Model has crashed again, time to lock it up and send a message out.
                statusData.genMsg = "ERROR: SIMULATION FOR GAGE: " + statusData.gages[basinNum] + \
                                    " HAS FAILED A SECOND TIME. PLEASE FIX ISSUE AND " + \
                                    "MANUALLY REMOVE LOCK FILE: " + lockPath
                errMod.sendMsg(statusData)
                open(lockPath,'a').close()
                keySlot[basinNum,iteration] = -1.0
                keyStatus = -1.0
                runFlag = False
                runCalib = False
            else:
                # Model sucessfully completed from first failed attempt. Ready for
                # calibration R code.
                # Cleanup any previous calib files.
                try:
                    errMod.cleanCalib(statusData,workDir,runDir)
                    errMod.scrubParams(statusMod,runDir,staticData)
                except:
                    raise
                if statusData.trouteFlag == 1: 
                    runTroute(statusData,staticData,db,gageID,gage,gageMeta,int(basinNum))
                    print("MODEL COMPLETE, READY TO RUN CALIB CODE")
                    keySlot[basinNum,iteration] = 0.65
                    keyStatus = 0.65
                    runFlag = False
                    runCalib = True
                else:
                    keySlot[basinNum,iteration] = 0.75
                    keyStatus = 0.75
                    runFlag = False
                    runCalib = True
                
    if keyStatus == -0.25:
        # Restarting model from one crash
        # First delete namelist files if they exist.
        check = runDir + "/namelist.hrldas"
        check2 = runDir + "/hydro.namelist"
        if os.path.isfile(check):
            os.remove(check)
        if os.path.isfile(check2):
            os.remove(check2)
            
        if staticData.coldStart == 0:
            # Make symbolic links as necssary.
            try:
                linkToRst(statusData,gage,runDir,gageMeta,staticData)
            except:
                raise
            
        # Since these are calibration simulations, we are always going to be 
        # starting the model rom an existing RESTART file. startType = 1 is for
        # when we have cold starts. Note 2 indicates we are restarting the model. 
        startType = 2
            
        if startType == 2:
            # Clean run directory of any old diagnostics files
            try:
                errMod.cleanRunDir(statusData,runDir)
            except:
                raise
                
        if begDate == staticData.bCalibDate:
            if staticData.coldStart == 1:
                # We are cold-starting this simulation for the beginning of the iteration.
                # This is per user request. 
                startType = 1
            else:
                # This is a unique situtation where the model failed right away, so treat it
                # as a new simulation. 
                startType = 3
            
        # Create new namelist files. 
        try:
            namelistMod.createHrldasNL(statusData,gageMeta,staticData,runDir,startType,begDate,endDate,1)
            namelistMod.createHydroNL(statusData,gageMeta,staticData,runDir,startType,begDate,endDate,1)
        except:
            raise
            
        # Double check to make sure all old calibration files have been cleaned up, except for newly
        # created parameter files.
        try:
            errMod.cleanCalib(statusData,workDir,runDir)
        except:
            raise
        print("RESTARTING MODEL")
        # Fire off model.
        cmd = runDir + "/run_WH_Restart.sh 1>" + runDir + "/WH_" + \
              str(statusData.jobID) + "_" + str(gageID) + ".out" + \
              ' 2>' + runDir + "/WH_" + str(statusData.jobID) + "_" + str(gageID) + ".err"
        try:
            p = subprocess.Popen([cmd], shell=True)
        except:
            statusData.errMsg = "ERROR: Unable to launch WRF-Hydro job for gage: " + str(gageMeta.gage[basinNum])
            raise

        # Revert statuses to -0.5 for next loop to convey the model crashed once. 
        keyStatus = -0.5
        keySlot[basinNum,iteration] = -0.5
        runFlag = False
        runCalib = False
        
    if keyStatus == 0.0 and runFlag:
        # Model needs to be either ran from the beginning of the calibration period.
        # First delete namelist files if they exist.
        check = runDir + "/namelist.hrldas"
        check2 = runDir + "/hydro.namelist"
        if os.path.isfile(check):
            os.remove(check)
        if os.path.isfile(check2):
            os.remove(check2)
            
        if staticData.coldStart == 0:
            # Make symbolic links as necssary.
            try:
                linkToRst(statusData,gage,runDir,gageMeta,staticData)
            except:
                raise
        
        # Since these are calibration simulations, we are always going to be 
        # starting the model rom an existing RESTART file. startType = 1 is for
        # when we have cold starts. 
        startType = 3
        
        if begDate == staticData.bCalibDate:
            if staticData.coldStart == 1:
                # We are cold-starting this simulation for the beginning of the iteration.
                # This is per user request. 
                startType = 1
        
        # Create namelist files. 
        try:
            namelistMod.createHrldasNL(statusData,gageMeta,staticData,runDir,startType,begDate,endDate,1)
            namelistMod.createHydroNL(statusData,gageMeta,staticData,runDir,startType,begDate,endDate,1)
        except:
            raise
            
        if startType == 2:
            # Clean run directory of any old diagnostics files
            try:
                errMod.cleanRunDir(statusData,runDir)
            except:
                raise
                
        # clean up old calibration related files, except for new parameter files.
        try:
            errMod.cleanCalib(statusData,workDir,runDir)
        except:
            raise
            
        print("FIRING OFF MODEL SIMULATION")
        # Fire off model.
        cmd = runDir + "/run_WH.sh 1>" + runDir + "/WH_" + \
              str(statusData.jobID) + "_" + str(gageID) + ".out" + \
              ' 2>' + runDir + "/WH_" + str(statusData.jobID) + "_" + str(gageID) + ".err"
        try:
            p = subprocess.Popen([cmd], shell=True)
        except:
            statusData.errMsg = "ERROR: Unable to launch WRF-Hydro job for gage: " + str(gageMeta.gage[basinNum])
            raise

        keyStatus = 0.5
        keySlot[basinNum,iteration] = 0.5
        
    if keyStatus == 0.0 and runCalib:
        # Unique situation where we are on iteration 1, and we need to run
        # a calibration script and adjust parameters once before beginning
        # the model. This generates default parameters. 
        
        # First cleanup any old model output or calibration output that 
        # is from previous iterations.
        try:
            errMod.removeOutput(statusData,runDir)
            errMod.cleanCalib(statusData,workDir,runDir)
            errMod.scrubParams(statusData,runDir,staticData)
        except:
            raise
            
        # If any proj_data.Rdata exists, remove it as it might have been from a failed first attempt.
        if os.path.isfile(workDir + '/proj_data.Rdata'):
            try:
                os.remove(workDir + '/proj_data.Rdata')
            except:
                statusData.errMsg = "ERROR: Failure to remove: " + workDir + "/proj_data.Rdata"
                raise
            
        try:
            generateRScript(staticData,gageMeta,gage,int(iteration))
        except:
            statusData.errMsg = "ERROR: Failure to write calibration R script."
            raise
            
        print("FIRING OFF FIRST CALIBRATION CODE")
        # Fire off calibration programs.
        cmd = workDir + "/run_WH_CALIB.sh 1>" + runDir + "/WH_CALIB_" + \
              str(statusData.jobID) + "_" + str(gageID) + ".out" + \
              ' 2>' + runDir + "/WH_CALIB_" + str(statusData.jobID) + "_" + str(gageID) + ".err"
        try:
            p2 = subprocess.Popen([str(cmd)], shell=True)
            time.sleep(5)
        except:
            statusData.errMsg = "ERROR: Unable to launch WRF-Hydro Calib job for gage: " + str(gageMeta.gage[basinNum])
            raise

        keyStatus = 0.25
        keySlot[basinNum,iteration] = 0.25

    if keyStatus == -0.0505 and runCalib:
        # Situation where the first calibration code failed. We need to restart.
        # First cleanup any old model output or calibration output that
        # is from previous iterations.
        try:
            errMod.removeOutput(statusData, runDir)
            errMod.cleanCalib(statusData, workDir, runDir)
            errMod.scrubParams(statusData, runDir, staticData)
        except:
            raise

        # If any proj_data.Rdata exists, remove it as it might have been from a failed first attempt.
        if os.path.isfile(workDir + '/proj_data.Rdata'):
            try:
                os.remove(workDir + '/proj_data.Rdata')
            except:
                statusData.errMsg = "ERROR: Failure to remove: " + workDir + "/proj_data.Rdata"
                raise

        try:
            generateRScript(staticData, gageMeta, gage, int(iteration))
        except:
            statusData.errMsg = "ERROR: Failure to write calibration R script."
            raise

        print("RESTARTING FIRST CALIBRATION CODE")
        # Fire off calibration programs.
        cmd = workDir + "/run_WH_CALIB.sh 1>" + runDir + "/WH_CALIB_" + \
              str(statusData.jobID) + "_" + str(gageID) + ".out" + \
              ' 2>' + runDir + "/WH_CALIB_" + str(statusData.jobID) + "_" + str(gageID) + ".err"
        try:
            p2 = subprocess.Popen([str(cmd)], shell=True)
            time.sleep(5)
        except:
            statusData.errMsg = "ERROR: Unable to launch WRF-Hydro Calib job for gage: " + str(gageMeta.gage[basinNum])
            raise
        # Set values to check on next pass-around.
        keyStatus = -0.05
        keySlot[basinNum, iteration] = -0.05

    if keyStatus == 0.75 and runCalib:
        # Fire off calibration for simulation.
        
        # First cleanup any old calibration related files. This should have 
        # already been done per workflow, but this is a fail safe.
        try:
            errMod.cleanCalib(statusData,workDir,runDir)
            errMod.scrubParams(statusMod,runDir,staticData)
        except:
            raise
            
        try:
            generateRScript(staticData,gageMeta,gage,int(iteration)+1)
        except:
            statusData.errMsg = "ERROR: Failure to write calibration R script."
            raise
            
        print("FIRING OFF CALIB CODE")
        # Fire off calibration program.
        cmd = workDir + "/run_WH_CALIB.sh 1>" + runDir + "/WH_CALIB_" + \
              str(statusData.jobID) + "_" + str(gageID) + ".out" + \
              ' 2>' + runDir + "/WH_CALIB_" + str(statusData.jobID) + "_" + str(gageID) + ".err"
        try:
            p3 = subprocess.Popen([cmd], shell=True)
            time.sleep(5)
        except:
            statusData.errMsg = "ERROR: Unable to launch WRF-Hydro Calib job for gage: " + str(gageMeta.gage[basinNum])
            raise

        keyStatus = 0.90
        keySlot[basinNum,iteration] = 0.90

    if keyStatus == -0.705 and runCalib:
        # Situation where we are restarting the calibration code.
        # First cleanup any old calibration related files. This should have
        # already been done per workflow, but this is a fail safe.
        try:
            errMod.cleanCalib(statusData, workDir, runDir)
            errMod.scrubParams(statusMod, runDir, staticData)
        except:
            raise

        try:
            generateRScript(staticData, gageMeta, gage, int(iteration) + 1)
        except:
            statusData.errMsg = "ERROR: Failure to write calibration R script."
            raise

        print("FIRING OFF CALIB CODE")
        # Fire off calibration program.
        cmd = workDir + "/run_WH_CALIB.sh 1>" + runDir + "/WH_CALIB_" + \
              str(statusData.jobID) + "_" + str(gageID) + ".out" + \
              ' 2>' + runDir + "/WH_CALIB_" + str(statusData.jobID) + "_" + str(gageID) + ".err"
        try:
            p3 = subprocess.Popen([cmd], shell=True)
            time.sleep(5)
        except:
            statusData.errMsg = "ERROR: Unable to launch WRF-Hydro Calib job for gage: " + str(gageMeta.gage[basinNum])
            raise
        # Set values to check on next pass-around.
        keyStatus = -0.7
        keySlot[basinNum, iteration] = -0.7
        
    # Update job status in the Database table.
    try:
        db.updateIterationStatus(statusData,int(gageMeta.gageID),iteration,str(gageMeta.gage),keyStatus)
    except:
        raise
    
                
def generateRestartBsubScript(jobData,gageID,runDir):
    """
    Generic function to create a run script that will be called by bsub
    to execute the model. This run script is used specifically to restart
    the model instead of removing all output prior to running the model.
    """
    
    outFile = runDir + "/run_WH_Restart.sh"
    
    if os.path.isfile(outFile):
        jobData.errMsg = "ERROR: Run script: " + outFile + " already exists."
        raise Exception()
        
    try:
        fileObj = open(outFile,'w')
        fileObj.write('#!/bin/bash\n')
        fileObj.write('#\n')
        fileObj.write('# LSF Batch Script to Run WRF-Hydro Calibration Simulations\n')
        fileObj.write('#\n')
        if len(jobData.acctKey.strip()) > 0:
            inStr = "#BSUB -P " + str(jobData.acctKey) + '\n'
            fileObj.write(inStr)
        fileObj.write('#BSUB -x\n')
        inStr = "#BSUB -n " + str(jobData.nCoresMod) + '\n'
        fileObj.write(inStr)
        inStr = "#BSUB -J WH_" + str(jobData.jobID) + "_" + str(gageID) + '\n'
        fileObj.write(inStr)
        inStr = '#BSUB -o ' + runDir + '/%J.out\n'
        fileObj.write(inStr)
        inStr = '#BSUB -e ' + runDir + '/%J.err\n'
        fileObj.write(inStr)
        fileObj.write('#BSUB -W 6:00\n')
        if len(jobData.queName.strip()) > 0:
            inStr = '#BSUB -q ' + str(jobData.queName) + '\n'
            fileObj.write(inStr)
        fileObj.write('\n')
        if len(jobData.ldPath.strip()) > 0:
                inStr = "export LD_LIBRARY_PATH=" + jobData.ldPath + ":$LD_LIBRARY_PATH" + '\n'
                fileObj.write(inStr)
        if len(jobData.binPath.strip()) > 0:
                inStr = "export PATH=" + jobData.binPath + ":$PATH" + '\n'
                fileObj.write(inStr)
        inStr = 'cd ' + runDir + '\n'
        fileObj.write(inStr)
        fileObj.write('mpirun.lsf ./wrf_hydro.exe\n')
        fileObj.close
    except:
        jobData.errMsg = "ERROR: Failure to create: " + outFile
        raise
        
def generateRestartPbsScript(jobData,gageID,runDir):
    """
    Generic function to create a run script that will be called by qsub
    to execute the model. This run script is used specifically to restart the model
    instead of removing all output prior to running the model.
    """
    
    outFile = runDir + "/run_WH_Restart.sh"
    
    if os.path.isfile(outFile):
        jobData.errMsg = "ERROR: Run script: " + outFile + " already exists."
        raise Exception()
        
    try:
        fileObj = open(outFile,'w')
        fileObj.write('#!/bin/bash\n')
        fileObj.write('#\n')
        fileObj.write('# PBS Batch Script to Run WRF-Hydro Calibration Simulations\n')
        fileObj.write('#\n')
        inStr = "#PBS -N WH_" + str(jobData.jobID) + "_" + str(gageID) + '\n'
        fileObj.write(inStr)
        if len(jobData.acctKey.strip()) > 0:
            inStr = "#PBS -A " + str(jobData.acctKey) + '\n'
            fileObj.write(inStr)
        inStr = "#PBS -l walltime=08:00:00\n"
        fileObj.write(inStr)
        if len(jobData.queName.strip()) > 0:
            inStr = "#PBS -q " + str(jobData.queName) + "\n"
            fileObj.write(inStr)
        inStr = "#PBS -o " + runDir + "/WH_" + str(jobData.jobID) + "_" + str(gageID) + ".out\n"
        fileObj.write(inStr)
        inStr = "#PBS -e " + runDir + "/WH_" + str(jobData.jobID) + "_" + str(gageID) + ".err\n"
        fileObj.write(inStr)
        nCoresPerNode = int(jobData.nCoresMod/jobData.nNodesMod)
        inStr = "#PBS -l select=" + str(jobData.nNodesMod) + ":ncpus=" + str(nCoresPerNode) + \
                ":mpiprocs=" + str(nCoresPerNode) + "\n"
        fileObj.write(inStr)
        fileObj.write("\n")
        if len(jobData.ldPath.strip()) > 0:
                inStr = "export LD_LIBRARY_PATH=" + jobData.ldPath + ":$LD_LIBRARY_PATH" + '\n'
                fileObj.write(inStr)
        if len(jobData.binPath.strip()) > 0:
                inStr = "export PATH=" + jobData.binPath + ":$PATH" + '\n'
                fileObj.write(inStr)
        inStr = 'cd ' + runDir + '\n'
        fileObj.write(inStr)
        fileObj.write('mpiexec ./wrf_hydro.exe\n')
        fileObj.close
    except:
        jobData.errMsg = "ERROR: Failure to create: " + outFile
        raise
        
def generateRestartSlurmScript(jobData,gageID,runDir):
    """
    Generic function to create a run script that will be called by Slurm
    to execute the model. This run script is used specifically to restart the model
    instead of removing all output prior to running the model.
    """
    
    outFile = runDir + "/run_WH_Restart.sh"
    
    if os.path.isfile(outFile):
        jobData.errMsg = "ERROR: Run script: " + outFile + " already exists."
        raise Exception()
    
    try:
        fileObj = open(outFile,'w')
        fileObj.write('#!/bin/bash\n')
        fileObj.write('#\n')
        fileObj.write('# Slurm Batch Script to Run WRF-Hydro Calibration Simulations\n')
        fileObj.write('#\n')
        inStr = "#SBATCH -J WH_" + str(jobData.jobID) + "_" + str(gageID) + '\n'
        fileObj.write(inStr)
        if len(jobData.acctKey.strip()) > 0:
            inStr = "#SBATCH -A " + str(jobData.acctKey) + '\n'
            fileObj.write(inStr)
        inStr = "#SBATCH -t 08:00:00\n"
        fileObj.write(inStr)
        if len(jobData.queName.strip()) > 0:
            inStr = "#SBATCH -p " + str(jobData.queName) + "\n"
            fileObj.write(inStr)
        inStr = "#SBATCH -o " + runDir + "/WH_" + str(jobData.jobID) + "_" + str(gageID) + ".out\n"
        fileObj.write(inStr)
        inStr = "#SBATCH -e " + runDir + "/WH_" + str(jobData.jobID) + "_" + str(gageID) + ".err\n"
        fileObj.write(inStr)
        inStr = "#SBATCH -N " + str(jobData.nNodesMod) + '\n'
        fileObj.write(inStr)
        inStr = "#SBATCH -n " + str(jobData.nCoresMod) + "\n"
        fileObj.write(inStr)
        fileObj.write("\n")
        if len(jobData.ldPath.strip()) > 0:
                inStr = "export LD_LIBRARY_PATH=" + jobData.ldPath + ":$LD_LIBRARY_PATH" + '\n'
                fileObj.write(inStr)
        if len(jobData.binPath.strip()) > 0:
                inStr = "export PATH=" + jobData.binPath + ":$PATH" + '\n'
                fileObj.write(inStr)
        inStr = 'cd ' + runDir + '\n'
        fileObj.write(inStr)
        if jobData.jobRunType == 3:
            inStr = 'srun ./wrf_hydro.exe\n'
        if jobData.jobRunType == 6:
            inStr = 'mpirun -n ' + str(jobData.nCoresMod) + ' ./wrf_hydro.exe\n'
        fileObj.write(inStr)
        fileObj.close
    except:
        jobData.errMsg = "ERROR: Failure to create: " + outFile
        raise
        
def generateMpiRstScript(jobData,gageID,basinNum, runDir):
    """
    Generic function to create a run script that will be called by mpiexec/mpirun
    to execute the model. This script is used specifically to restart the
    model instead of removing all output prior to running the model.
    """
    
    outFile = runDir + "/run_WH_Restart.sh"
    
    if os.path.isfile(outFile):
        jobData.errMsg = "ERROR: Run script: " + outFile + " already exists."
        raise Exception()
        
    try:
        fileObj = open(outFile,'w')
        fileObj.write('#!/bin/bash\n')
        if len(jobData.ldPath.strip()) > 0:
                inStr = "export LD_LIBRARY_PATH=" + jobData.ldPath + ":$LD_LIBRARY_PATH" + '\n'
                fileObj.write(inStr)
        if len(jobData.binPath.strip()) > 0:
                inStr = "export PATH=" + jobData.binPath + ":$PATH" + '\n'
                fileObj.write(inStr)
        inStr = 'cd ' + runDir + '\n'
        fileObj.write(inStr)
        if len(jobData.cpuPinCmd) > 0:
            inStr = jobData.mpiCmd + " " + str(jobData.nCoresMod) + " " + jobData.cpuPinCmd + " " + \
                str(jobData.gageBegModelCpu[basinNum]) + "-" + \
                str(jobData.gageEndModelCpu[basinNum]) + " ./W" + \
                str(jobData.jobID) + str(gageID) + '\n'
        else:
            inStr = jobData.mpiCmd + " " + str(jobData.nCoresMod) + " ./W" + \
                    str(jobData.jobID) + str(gageID) + '\n'
        fileObj.write(inStr)
        fileObj.close
    except:
        jobData.errMsg = "ERROR: Failure to create: " + outFile
        raise
        
    # Make the file an executable
    cmd = "chmod +x " + outFile
    try:
        subprocess.call(cmd,shell=True)
    except:
        jobData.errMsg = "ERROR: Failure to convert: " + outFile + " to an executable."
        raise
        
def generateBsubScript(jobData,gageID,runDir):
    """
    Generic function to create a run script that will be called by bsub
    to execute the model. For this particular BSUB script, we clean out
    all prior model output in preparation for the next iteration. 
    """
    
    outFile = runDir + "/run_WH.sh"
    
    if os.path.isfile(outFile):
        jobData.errMsg = "ERROR: Run script: " + outFile + " already exists."
        raise Exception()
        
    try:
        fileObj = open(outFile,'w')
        fileObj.write('#!/bin/bash\n')
        fileObj.write('#\n')
        fileObj.write('# LSF Batch Script to Run WRF-Hydro Calibration Simulations\n')
        fileObj.write('#\n')
        if len(jobData.acctKey.strip()) > 0:
            inStr = "#BSUB -P " + str(jobData.acctKey) + '\n'
            fileObj.write(inStr)
        fileObj.write('#BSUB -x\n')
        inStr = "#BSUB -n " + str(jobData.nCoresMod) + '\n'
        fileObj.write(inStr)
        inStr = "#BSUB -J WH_" + str(jobData.jobID) + "_" + str(gageID) + '\n'
        fileObj.write(inStr)
        inStr = '#BSUB -o ' + runDir + '/%J.out\n'
        fileObj.write(inStr)
        inStr = '#BSUB -e ' + runDir + '/%J.err\n'
        fileObj.write(inStr)
        fileObj.write('#BSUB -W 6:00\n')
        if len(jobData.queName.strip()) > 0:
            inStr = '#BSUB -q ' + str(jobData.queName) + '\n'
            fileObj.write(inStr)
        fileObj.write('\n')
        if len(jobData.ldPath.strip()) > 0:
                inStr = "export LD_LIBRARY_PATH=" + jobData.ldPath + ":$LD_LIBRARY_PATH" + '\n'
                fileObj.write(inStr)
        if len(jobData.binPath.strip()) > 0:
                inStr = "export PATH=" + jobData.binPath + ":$PATH" + '\n'
                fileObj.write(inStr)
        inStr = 'cd ' + runDir + '\n'
        fileObj.write(inStr)
        inStr = 'for FILE in HYDRO_RST.*; do if [ ! -L $FILE ] ; then rm -rf $FILE; fi; done\n'
        fileObj.write(inStr)
        inStr = 'for FILE in RESTART.*; do if [ ! -L $FILE ] ; then rm -rf $FILE; fi; done\n'
        fileObj.write(inStr)
        fileObj.write('mpirun.lsf ./wrf_hydro.exe\n')
        fileObj.close
    except:
        jobData.errMsg = "ERROR: Failure to create: " + outFile
        raise
        
def generatePbsScript(jobData,gageID,runDir):
    """
    Generic function to create a script that will be called by qsub
    to execute the model.
    """
    
    outFile = runDir + "/run_WH.sh"
    
    if os.path.isfile(outFile):
        jobData.errMsg = "ERROR: Run script: " + outFile + " already exists."
        raise Exception()
        
    try:
        fileObj = open(outFile,'w')
        fileObj.write('#!/bin/bash\n')
        fileObj.write('#\n')
        fileObj.write('# PBS Batch Script to Run WRF-Hydro Calibration Simulations\n')
        fileObj.write('#\n')
        inStr = "#PBS -N WH_" + str(jobData.jobID) + "_" + str(gageID) + '\n'
        fileObj.write(inStr)
        if len(jobData.acctKey.strip()) > 0:
            inStr = "#PBS -A " + str(jobData.acctKey) + '\n'
            fileObj.write(inStr)
        inStr = "#PBS -l walltime=08:00:00\n"
        fileObj.write(inStr)
        if len(jobData.queName.strip()) > 0:
            inStr = "#PBS -q " + str(jobData.queName) + "\n"
            fileObj.write(inStr)
        inStr = "#PBS -o " + runDir + "/WH_" + str(jobData.jobID) + "_" + str(gageID) + ".out\n"
        fileObj.write(inStr)
        inStr = "#PBS -e " + runDir + "/WH_" + str(jobData.jobID) + "_" + str(gageID) + ".err\n"
        fileObj.write(inStr)
        nCoresPerNode = int(jobData.nCoresMod/jobData.nNodesMod)
        inStr = "#PBS -l select=" + str(jobData.nNodesMod) + ":ncpus=" + str(nCoresPerNode) + \
                ":mpiprocs=" + str(nCoresPerNode) + "\n"
        fileObj.write(inStr)
        fileObj.write("\n")
        if len(jobData.ldPath.strip()) > 0:
                inStr = "export LD_LIBRARY_PATH=" + jobData.ldPath + ":$LD_LIBRARY_PATH" + '\n'
                fileObj.write(inStr)
        if len(jobData.binPath.strip()) > 0:
                inStr = "export PATH=" + jobData.binPath + ":$PATH" + '\n'
                fileObj.write(inStr)
        inStr = 'cd ' + runDir + '\n'
        fileObj.write(inStr)
        inStr = 'for FILE in HYDRO_RST.*; do if [ ! -L $FILE ] ; then rm -rf $FILE; fi; done\n'
        fileObj.write(inStr)
        inStr = 'for FILE in RESTART.*; do if [ ! -L $FILE ] ; then rm -rf $FILE; fi; done\n'
        fileObj.write(inStr)
        fileObj.write('mpiexec ./wrf_hydro.exe\n')
        fileObj.close
    except:
        jobData.errMsg = "ERROR: Failure to create: " + outFile
        raise
        
def generateSlurmScript(jobData,gageID,runDir):
    """
    Generic function to create a script that will be called by Slurm
    to execute the model.
    """
    
    outFile = runDir + "/run_WH.sh"
    
    if os.path.isfile(outFile):
        jobData.errMsg = "ERROR: Run script: " + outFile + " already exists."
        raise Exception()
        
    try:
        fileObj = open(outFile,'w')
        fileObj.write('#!/bin/bash\n')
        fileObj.write('#\n')
        fileObj.write('# Slurm Batch Script to Run WRF-Hydro Calibration Simulations\n')
        fileObj.write('#\n')
        inStr = "#SBATCH -J WH_" + str(jobData.jobID) + "_" + str(gageID) + '\n'
        fileObj.write(inStr)
        if len(jobData.acctKey.strip()) > 0:
            inStr = "#SBATCH -A " + str(jobData.acctKey) + '\n'
            fileObj.write(inStr)
        inStr = "#SBATCH -t 08:00:00\n"
        fileObj.write(inStr)
        if len(jobData.queName.strip()) > 0:
            inStr = "#SBATCH -p " + str(jobData.queName) + "\n"
            fileObj.write(inStr)
        inStr = "#SBATCH -o " + runDir + "/WH_" + str(jobData.jobID) + "_" + str(gageID) + ".out\n"
        fileObj.write(inStr)
        inStr = "#SBATCH -e " + runDir + "/WH_" + str(jobData.jobID) + "_" + str(gageID) + ".err\n"
        fileObj.write(inStr)
        inStr = "#SBATCH -N " + str(jobData.nNodesMod) + "\n"
        fileObj.write(inStr)
        fileObj.write("\n")
        if len(jobData.ldPath.strip()) > 0:
                inStr = "export LD_LIBRARY_PATH=" + jobData.ldPath + ":$LD_LIBRARY_PATH" + '\n'
                fileObj.write(inStr)
        if len(jobData.binPath.strip()) > 0:
                inStr = "export PATH=" + jobData.binPath + ":$PATH" + '\n'
                fileObj.write(inStr)
        inStr = 'cd ' + runDir + '\n'
        fileObj.write(inStr)
        inStr = 'for FILE in HYDRO_RST.*; do if [ ! -L $FILE ] ; then rm -rf $FILE; fi; done\n'
        fileObj.write(inStr)
        inStr = 'for FILE in RESTART.*; do if [ ! -L $FILE ] ; then rm -rf $FILE; fi; done\n'
        fileObj.write(inStr)
        if jobData.jobRunType == 3:
            inStr = "srun -n " + str(jobData.nCoresMod) + " ./wrf_hydro.exe\n"
        if jobData.jobRunType == 6:
            inStr = "mpirun -n " + str(jobData.nCoresMod) + " ./wrf_hydro.exe\n"
        fileObj.close
    except:
        jobData.errMsg = "ERROR: Failure to create: " + outFile
        raise
        
def generateMpiScript(jobData,gageID,basinNum,runDir):
    """
    Generic function to create a run script that will be called by mpiexec/mpirun
    to execute the model. For this particular script, we clean out all prior
    moel output in preparation for the next iteration.
    """
    
    outFile = runDir + "/run_WH.sh"
    
    if os.path.isfile(outFile):
        jobData.errMsg = "ERROR: Run script: " + outFile + " already exists."
        raise Exception()
        
    try:
        fileObj = open(outFile,'w')
        fileObj.write('#!/bin/bash\n')
        if len(jobData.ldPath.strip()) > 0:
                inStr = "export LD_LIBRARY_PATH=" + jobData.ldPath + ":$LD_LIBRARY_PATH" + '\n'
                fileObj.write(inStr)
        if len(jobData.binPath.strip()) > 0:
                inStr = "export PATH=" + jobData.binPath + ":$PATH" + '\n'
                fileObj.write(inStr)
        inStr = 'cd ' + runDir + '\n'
        fileObj.write(inStr)
        inStr = 'for FILE in HYDRO_RST.*; do if [ ! -L $FILE ] ; then rm -rf $FILE; fi; done\n'
        fileObj.write(inStr)
        inStr = 'for FILE in RESTART.*; do if [ ! -L $FILE ] ; then rm -rf $FILE; fi; done\n'
        fileObj.write(inStr)
        if len(jobData.cpuPinCmd) > 0:
            inStr = jobData.mpiCmd + " " + str(jobData.nCoresMod) + " " + jobData.cpuPinCmd + " " + \
                    str(jobData.gageBegModelCpu[basinNum]) + "-" + \
                    str(jobData.gageEndModelCpu[basinNum]) + " ./W" + \
                    str(jobData.jobID) + str(gageID) + '\n'
        else:
            inStr = jobData.mpiCmd + " " + str(jobData.nCoresMod) + " ./W" + \
                    str(jobData.jobID) + str(gageID) + '\n'
        fileObj.write(inStr)
        fileObj.close
    except:
        jobData.errMsg = 'Failure to create: ' + outFile
        raise
        
    # Make the file an executable.
    cmd = "chmod +x " + outFile
    try:
        subprocess.call(cmd,shell=True)
    except:
        jobData.errMsg = "ERROR: Failure to convert: " + outFile + " to an executable."
        raise
        
def generateRScript(jobData,gageMeta,gageNum,iteration):
    """
    Generic function to create R script that will be sourced by R during
    calibration.
    """
    outPath = jobData.outDir + "/" + jobData.jobName + "/" + str(gageMeta.gage) + \
              "/RUN.CALIB/calibScript.R"
              
    if os.path.isfile(outPath):
        # Over-write to update mCurrent to reflect current iteration
        os.remove(outPath)
    
    if os.path.isfile(outPath):
        jobData.errMsg = "ERROR: Calibration R script: " + outPath + " aleady exists."
        raise Exception()

    try:
        fileObj = open(outPath,'w')
        fileObj.write('#### Model Parameters ####\n')
        inStr = "enableStreamflowCalib <- " + str(jobData.enableStreamflowCalib) + "\n"
        fileObj.write(inStr)
        inStr = "enableSnowCalib <- " + str(jobData.enableSnowCalib) + "\n"
        fileObj.write(inStr)
        inStr = "enableSoilMoistureCalib <- " + str(jobData.enableSoilMoistureCalib) + "\n"
        fileObj.write(inStr)
        inStr = "streamflowObjFunc <- \"" + str(jobData.streamflowObjFunc) + "\"\n"
        fileObj.write(inStr)
        inStr = "snowObjFunc <- \"" + str(jobData.snowObjFunc) + "\"\n"
        fileObj.write(inStr)
        inStr = "soilMoistureObjFunc <- \"" + str(jobData.soilMoistureObjFunc) + "\"\n"
        fileObj.write(inStr)
        inStr = "streamflowWeight <- " + str(jobData.streamflowWeight) + "\n"
        fileObj.write(inStr)
        inStr = "snowWeight <- " + str(jobData.snowWeight) + "\n"
        fileObj.write(inStr)
        inStr = "soilMoistureWeight <- " + str(jobData.soilMoistureWeight) + "\n"
        fileObj.write(inStr)
        fileObj.write('# Specify number of calibration iterations.\n')
        inStr = "m <- " + str(jobData.nIter) + '\n'
        fileObj.write(inStr)
        inStr = "mCurrent <- " + str(iteration) + "\n"
        fileObj.write(inStr)
        fileObj.write('# Specify parameter for event metrics.\n') # Xia 20210610
        inStr = "basinType <- " + str(jobData.basinType) + "\n"
        fileObj.write(inStr)
        inStr = "weight1 <- " + str(jobData.weight1Event) + "\n"
        fileObj.write(inStr)
        inStr = "weight2 <- " + str(jobData.weight2Event) + "\n"
        fileObj.write(inStr)
        fileObj.write('# Specify DDS parameter (if used).\n')
        inStr = "r <- " + str(jobData.ddsR) + "\n"
        fileObj.write(inStr)
        fileObj.write("# Specify run directory containing calibration simulations.\n")
        inStr = "runDir <- '" + jobData.outDir + "/" + jobData.jobName + "/" + \
                str(gageMeta.gage) + "/RUN.CALIB'\n"
        fileObj.write(inStr)
        fileObj.write('# Basin-Specific Metadata\n')
        inStr = "siteId <- '" + str(gageMeta.gage) + "'\n"
        fileObj.write(inStr)
        inStr = "siteName <- '" + str(gageMeta.siteName) + "'\n"
        fileObj.write(inStr)
        inStr = "rtlinkFile <- '" + str(gageMeta.rtLnk) + "'\n"
        fileObj.write(inStr)
        inStr = "linkId <- " + str(gageMeta.comID) + "\n"
        fileObj.write(inStr)
        fileObj.write('# Start date for evaluation period (e.g., after spinup period)\n')
        inStr = "startDate <- as.POSIXct(\"" + jobData.bCalibEvalDate.strftime('%Y-%m-%d') + "\", " + \
                 "format=\"%Y-%m-%d\", tz=\"UTC\")\n"
        fileObj.write(inStr)
        fileObj.write('# Specify number of cores to use\n')
        inStr = "ncores <- 1\n"
        fileObj.write(inStr)
        fileObj.write('# Specify whether to run daily or hourly analysis\n')
        if jobData.dailyAnalysis == 1:
            fileObj.write("calcDailyStats <- TRUE\n")
        else:
            fileObj.write("calcDailyStats <- FALSE\n")
        fileObj.write('# Hydro Option on the SPLIT_OUTPUT_COUNT\n')
        inStr = "hydro_SPLIT_OUTPUT_COUNT <- " + str(int(jobData.SplitOutputCount)) + "\n"
        fileObj.write(inStr)
        fileObj.write('# LSM Option on the SPLIT_OUTPUT_COUNT\n')
        inStr = "lsm_SPLIT_OUTPUT_COUNT <- " + str(int(jobData.lsmSplitOutputCount)) + "\n"
        fileObj.write(inStr)
        fileObj.write('# Option to use multiple sites for calibration\n')
        inStr = "enableMultiSites <- " + str(int(jobData.enableMultiSites)) + "\n"
        fileObj.write(inStr)

        fileObj.close
    except:
        jobData.errMsg = "ERROR: Failure to create: " + outPath
        raise        
        
def generateBsubCalibScript(jobData,gageID,runDir,workDir,staticData):
    """
    Generic Function function to create BSUB script for running R
    calibration routines. These jobs will be shorter than 
    the model runs, but still need to be ran through Yellowstone
    compute nodes. This function also creates the shell script that
    will execute R and Python to modify parameters.
    """
    
    outFile1 = workDir + "/run_WH_CALIB.sh"
    
    if os.path.isfile(outFile1):
        # We are just going to manually over-write the file everytime to be safe.
        os.remove(outFile1)
    
    if not os.path.isfile(outFile1):
        try:
            fileObj = open(outFile1,'w')
            fileObj.write('#!/bin/bash\n')
            fileObj.write('#\n')
            fileObj.write('# LSF Batch Script to Run WRF-Hydro Calibration R Code\n')
            fileObj.write('#\n')
            if len(jobData.acctKey.strip()) > 0:
                inStr = "#BSUB -P " + str(jobData.acctKey) + '\n'
                fileObj.write(inStr)
            inStr = "#BSUB -n 1\n"
            fileObj.write(inStr)
            inStr = "#BSUB -J WH_CALIB_" + str(jobData.jobID) + "_" + str(gageID) + '\n'
            fileObj.write(inStr)
            inStr = '#BSUB -o ' + workDir + '/%J.out\n'
            fileObj.write(inStr)
            inStr = '#BSUB -e ' + workDir + '/%J.err\n'
            fileObj.write(inStr)
            # We are using 2 hours to be safe here. 
            fileObj.write('#BSUB -W 2:00\n')
            if len(jobData.queName.strip()) > 0:
                inStr = '#BSUB -q ' + str(jobData.queName) + '\n'
                fileObj.write(inStr)
            fileObj.write('\n')
            if len(jobData.ldPath.strip()) > 0:
                inStr = "export LD_LIBRARY_PATH=" + jobData.ldPath + ":$LD_LIBRARY_PATH" + '\n'
                fileObj.write(inStr)
            if len(jobData.binPath.strip()) > 0:
                inStr = "export PATH=" + jobData.binPath + ":$PATH" + '\n'
                fileObj.write(inStr)
            inStr = 'cd ' + workDir + '\n'
            fileObj.write(inStr)
            fileObj.write('./calibCmd.sh\n')
            fileObj.close
        except:
            jobData.errMsg = "ERROR: Failure to create: " + outFile1
            raise    
            
    outFile2 = workDir + "/calibCmd.sh"
    
    runRProgram = workDir + "/calib_workflow.R"
    srcScript = workDir + "/calibScript.R"
        
    if not os.path.isfile(outFile2):
        # This is the file that will run the R code first to generate params_new.txt and
        # params_stats.txt. Python is called next, which will read in 
        try:
            fileObj = open(outFile2,'w')
            fileObj.write('#!/bin/bash\n')
            fileObj.write('Rscript ' + runRProgram + " " + srcScript + '\n')
            fileObj.write('python ' + workDir + '/adjust_parameters.py ' + workDir + ' ' + runDir + ' ' + \
                          str(staticData.gwBaseFlag) + ' ' + str(staticData.chnRtOpt) + ' ' + str(staticData.enableMask) +' \n')
            fileObj.write('exit\n')
        except:
            jobData.errMsg = "ERROR: Failure to create: " + outFile2
            raise
            
    # Make shell script an executable.
    cmd = 'chmod +x ' + outFile2
    try:
        subprocess.call(cmd,shell=True)
    except:
        jobData.errMsg = "ERROR: Failure to convert: " + outFile2 + " to an executable."
        raise
        
def generatePbsCalibScript(jobData,gageID,runDir,workDir,staticData):
    """
    Generic Function function to create PBS script for running R
    calibration routines. These jobs will be shorter than 
    the model runs, but still need to be ran through HPC
    compute nodes. This function also creates the shell script that
    will execute R and Python to modify parameters.
    """
    
    outFile1 = workDir + "/run_WH_CALIB.sh"
    
    if os.path.isfile(outFile1):
        # We are just going to manually over-write the file everytime to be safe.
        os.remove(outFile1)
    
    if not os.path.isfile(outFile1):
        try:
            fileObj = open(outFile1,'w')
            fileObj.write('#!/bin/bash\n')
            fileObj.write('#\n')
            fileObj.write('# PBS Batch Script to Run WRF-Hydro Calibration R Code\n')
            fileObj.write('#\n')
            if len(jobData.acctKey.strip()) > 0:
                inStr = "#PBS -A " + str(jobData.acctKey) + '\n'
                fileObj.write(inStr)
            inStr = "#PBS -N WH_CALIB_" + str(jobData.jobID) + "_" + str(gageID) + '\n'
            fileObj.write(inStr)
            inStr = '#PBS -o ' + workDir + '/WH_CALIB_' + str(jobData.jobID) + '_' + str(gageID) + '.out\n'
            fileObj.write(inStr)
            inStr = '#PBS -e ' + workDir + '/WH_CALIB_' + str(jobData.jobID) + '_' + str(gageID) + '.err\n'
            fileObj.write(inStr)
            inStr = "#PBS -l select=1:ncpus=1:mpiprocs=1\n"
            fileObj.write(inStr)
            # We are using 2 hours to be safe here. 
            fileObj.write('#PBS -l walltime=02:00:00\n')
            if len(jobData.queName.strip()) > 0:
                inStr = '#PBS -q ' + str(jobData.queName) + '\n'
                fileObj.write(inStr)
            fileObj.write('\n')
            if len(jobData.ldPath.strip()) > 0:
                inStr = "export LD_LIBRARY_PATH=" + jobData.ldPath + ":$LD_LIBRARY_PATH" + '\n'
                fileObj.write(inStr)
            if len(jobData.binPath.strip()) > 0:
                inStr = "export PATH=" + jobData.binPath + ":$PATH" + '\n'
                fileObj.write(inStr)
            inStr = 'cd ' + workDir + '\n'
            fileObj.write(inStr)
            fileObj.write('./calibCmd.sh\n')
            fileObj.close
        except:
            jobData.errMsg = "ERROR: Failure to create: " + outFile1
            raise    
            
    outFile2 = workDir + "/calibCmd.sh"
    
    runRProgram = workDir + "/calib_workflow.R"
    srcScript = workDir + "/calibScript.R"
        
    if not os.path.isfile(outFile2):
        # This is the file that will run the R code first to generate params_new.txt and
        # params_stats.txt. Python is called next, which will read in 
        try:
            fileObj = open(outFile2,'w')
            fileObj.write('#!/bin/bash\n')
            fileObj.write('Rscript ' + runRProgram + " " + srcScript + '\n')
            fileObj.write('python ' + workDir + '/adjust_parameters.py ' + workDir + ' ' + \
                          runDir + ' ' + str(staticData.gwBaseFlag) + ' ' + \
                          str(staticData.chnRtOpt) + ' ' + str(staticData.enableMask) + ' \n')
            fileObj.write('exit\n')
        except:
            jobData.errMsg = "ERROR: Failure to create: " + outFile2
            raise
            
    # Make shell script an executable.
    cmd = 'chmod +x ' + outFile2
    try:
        subprocess.call(cmd,shell=True)
    except:
        jobData.errMsg = "ERROR: Failure to convert: " + outFile2 + " to an executable."
        raise
        
def generateSlurmCalibScript(jobData,gageID,runDir,workDir,staticData):
    """
    Generic Function function to create Slurm script for running R
    calibration routines. These jobs will be shorter than 
    the model runs, but still need to be ran through HPC
    compute nodes. This function also creates the shell script that
    will execute R and Python to modify parameters.
    """
    
    outFile1 = workDir + "/run_WH_CALIB.sh"
    
    if os.path.isfile(outFile1):
        # We are just going to manually over-write the file everytime to be safe.
        os.remove(outFile1)
    
    if not os.path.isfile(outFile1):
        try:
            fileObj = open(outFile1,'w')
            fileObj.write('#!/bin/bash\n')
            fileObj.write('#\n')
            fileObj.write('# Slurm Batch Script to Run WRF-Hydro Calibration R Code\n')
            fileObj.write('#\n')
            if len(jobData.acctKey.strip()) > 0:
                inStr = "#SBATCH -A " + str(jobData.acctKey) + '\n'
                fileObj.write(inStr)
            inStr = "#SBATCH -J WH_CALIB_" + str(jobData.jobID) + "_" + str(gageID) + '\n'
            fileObj.write(inStr)
            inStr = '#SBATCH -o ' + workDir + '/WH_CALIB_' + str(jobData.jobID) + '_' + str(gageID) + '.out\n'
            fileObj.write(inStr)
            inStr = '#SBATCH -e ' + workDir + '/WH_CALIB_' + str(jobData.jobID) + '_' + str(gageID) + '.err\n'
            fileObj.write(inStr)
            inStr = "#SBATCH -N 1\n"
            fileObj.write(inStr)
            # We are using 2 hours to be safe here. 
            fileObj.write('#SBATCH -t 02:00:00\n')
            if len(jobData.queName.strip()) > 0:
                inStr = '#SBATCH -p ' + str(jobData.queName) + '\n'
                fileObj.write(inStr)
            fileObj.write('\n')
            if len(jobData.ldPath.strip()) > 0:
                inStr = "export LD_LIBRARY_PATH=" + jobData.ldPath + ":$LD_LIBRARY_PATH" + '\n'
                fileObj.write(inStr)
            if len(jobData.binPath.strip()) > 0:
                inStr = "export PATH=" + jobData.binPath + ":$PATH" + '\n'
                fileObj.write(inStr)
            inStr = 'cd ' + workDir + '\n'
            fileObj.write(inStr)
            fileObj.write('./calibCmd.sh\n')
            fileObj.close
        except:
            jobData.errMsg = "ERROR: Failure to create: " + outFile1
            raise    
            
    outFile2 = workDir + "/calibCmd.sh"
    
    runRProgram = workDir + "/calib_workflow.R"
    srcScript = workDir + "/calibScript.R"
        
    if not os.path.isfile(outFile2):
        # This is the file that will run the R code first to generate params_new.txt and
        # params_stats.txt. Python is called next, which will read in 
        try:
            fileObj = open(outFile2,'w')
            fileObj.write('#!/bin/bash\n')
            fileObj.write('Rscript ' + runRProgram + " " + srcScript + '\n')
            fileObj.write('python ' + workDir + '/adjust_parameters.py ' + workDir + \
                          ' ' + runDir + ' ' + str(staticData.gwBaseFlag) + ' ' + \
                          str(staticData.chnRtOpt) + ' ' + str(staticData.enableMask) +' \n')
            fileObj.write('exit\n')
        except:
            jobData.errMsg = "ERROR: Failure to create: " + outFile2
            raise
            
    # Make shell script an executable.
    cmd = 'chmod +x ' + outFile2
    try:
        subprocess.call(cmd,shell=True)
    except:
        jobData.errMsg = "ERROR: Failure to convert: " + outFile2 + " to an executable."
        raise
        
def generateMpiCalibScript(jobData,gageID,basinNum,runDir,workDir,staticData):
    """
    Generic function to create mpiexec/mpirun script for running R calibration
    routines. This function also creates the shell script that will execute
    R and Python to modify parameters. 
    """
    
    outFile1 = workDir + "/run_WH_CALIB.sh"
    
    if os.path.isfile(outFile1):
        # We are just gonig to manually over-write the file everytime to be safe.
        os.remove(outFile1)
        
    if not os.path.isfile(outFile1):
        try:
            fileObj = open(outFile1,'w')
            fileObj.write('#!/bin/bash\n')
            if len(jobData.ldPath.strip()) > 0:
                inStr = "export LD_LIBRARY_PATH=" + jobData.ldPath + ":$LD_LIBRARY_PATH" + '\n'
                fileObj.write(inStr)
            if len(jobData.binPath.strip()) > 0:
                inStr = "export PATH=" + jobData.binPath + ":$PATH" + '\n'
                fileObj.write(inStr)
            inStr = 'cd ' + workDir + '\n'
            fileObj.write(inStr)
            if len(jobData.cpuPinCmd) > 0:
                inStr = jobData.mpiCmd + " 1 " + jobData.cpuPinCmd + " " + \
                        str(jobData.gageBegModelCpu[basinNum]) + " ./C" + \
                        str(jobData.jobID) + str(gageID) + '\n'
            else:
                inStr = jobData.mpiCmd + " 1 ./C" + \
                        str(jobData.jobID) + str(gageID) + '\n'
            fileObj.write(inStr)
            fileObj.close
        except:
            jobData.errMsg = "ERROR: Failure to create " + outFile1
            raise
            
    # Make the file an executable.
    cmd = "chmod +x " + outFile1
    try:
        subprocess.call(cmd,shell=True)
    except:
        jobData.errMsg = "ERROR: Failure to convert: " + outFile1 + " to an executable."
        raise
            
    outFile2 = workDir + '/calibCmd.sh'
    outLink2 = workDir + '/C' + str(jobData.jobID) + str(gageID) 
    
    runRProgram = workDir + '/calib_workflow.R'
    srcScript = workDir + '/calibScript.R'
    
    if not os.path.isfile(outFile2):
        # This is the file that will run R code. First to generate params_new.txt and
        # params_stats.txt. Python is called next, which will generate new parameters.
        try:
            fileObj = open(outFile2,'w')
            fileObj.write('#!/bin/bash\n')
            fileObj.write('Rscript ' + runRProgram + " " + srcScript + '\n')
            fileObj.write('python ' + workDir + '/adjust_parameters.py ' + workDir + \
                          ' ' + runDir + ' ' + str(staticData.gwBaseFlag) + ' ' + \
                          str(staticData.chnRtOpt) + ' ' + str(staticData.enableMask) + ' \n')
            fileObj.write('exit\n')
        except:
            jobData.errMsg = "ERROR: Failure to create: " + outFile2
            raise
            
    # Make shell script an executable.
    cmd = 'chmod +x ' + outFile2
    try:
        subprocess.call(cmd,shell=True)
    except:
        jobData.errMsg = "ERROR: Failure to convert: " + outFile2 + " to an executable."
        raise
        
    # Make symbolic link to newly created executable, which will be called by
    # mpiexec/mpirun.
    if not os.path.islink(outLink2):
        try:
            os.symlink(outFile2,outLink2)
        except:
            jobData.errMsg = "ERROR: Failure to create symbolic link: " + outLink2
            raise
        
def linkToRst(statusData,gage,runDir,gageMeta,staticData):
    """
    Generic function to link to necessary restart files from the spinup.
    This was broken out as a function as sometimes the output directory
    is scrubbed, and links need to be re-made in preparation for a new 
    iteration simulation. If the user has specified a custom restart file, or 
    to optionally cold start the model through calibration, accomodate here. 
    """
    link1 = runDir + "/RESTART." + statusData.bCalibDate.strftime('%Y%m%d') + "00_DOMAIN1"
    link2 = runDir + "/HYDRO_RST." + statusData.bCalibDate.strftime('%Y-%m-%d') + "_00:00_DOMAIN1"

    if staticData.optSpinFlag == 0: 
        # Check to make sure symbolic link to spinup state exists.
        check1 = statusData.jobDir + "/" + gage + "/RUN.SPINUP/OUTPUT/RESTART." + statusData.eSpinDate.strftime('%Y%m%d') + "00_DOMAIN1"
        check2 = statusData.jobDir + "/" + gage + "/RUN.SPINUP/OUTPUT/HYDRO_RST." + statusData.eSpinDate.strftime('%Y-%m-%d') + "_00:00_DOMAIN1"
        
        if not os.path.isfile(check1):
            statusData.errMsg = "ERROR: Spinup state: " + check1 + " not found."
            raise Exception()
        if not os.path.isfile(check2):
            statusData.errMsg = "ERROR: Spinup state: " + check2 + " not found."
            raise Exception()
        
        # Create links if they don't exist
        if not os.path.islink(link1):
            os.symlink(check1,link1)
        if not os.path.islink(link2):
            os.symlink(check2,link2)
    elif staticData.optSpinFlag != 1:
        # Check to see if file exists, then create symbolic link to it. 
        if gageMeta.optLandRstFile == "-9999":
            statusData.errMsg = "ERROR: User has specified to use an optional land " + \
                                "restart file when none exists."
            raise Exception()
        if gageMeta.optHydroRstFile == "-9999":
            statusData.errMsg = "ERROR: User has specified to use an optional hydro " + \
                                "restart file when none exists."
            raise Exception()
        if not os.path.isfile(gageMeta.optLandRstFile):
            statusData.errMsg = "ERROR: Spinup state: " + gageMeta.optLandRstFile + " not found."
            raise Exception()
        if not os.path.isfile(gageMeta.optHydroRstFile):
            statusData.errMsg = "ERROR: Spinup state: " + gageMeta.optHydroRstFile + " not found."
            raise Exception()
        # Create links if they don't exist
        if not os.path.islink(link1):
            os.symlink(gageMeta.optLandRstFile,link1)
        if not os.path.islink(link2):
            os.symlink(gageMeta.optHydroRstFile,link2)

def generateTrouteScript(statusData,runDir,yamlPath, basinNum):
    """
    Generic function to create a run script that will be used to execute the troute model.
    """
    outFile = runDir + "/run_troute.sh"
    """
    if os.path.isfile(outFile):
        statusData.errMsg = "ERROR: Run script: " + outFile + " already exists."
        raise Exception()
    """
    try:
        fileObj = open(outFile,'w')
        fileObj.write('#!/bin/bash\n')
        for m in statusData.moduleLoadTrouteStr:
            fileObj.write(m)
            fileObj.write("\n")
        inStr = 'cd ' + runDir + '\n'
        fileObj.write(inStr)
        #inStr = 'for FILE in channel_restart*; do if [ ! -L $FILE ] ; then rm -rf $FILE; fi; done\n'

        #fileObj.write(inStr)
        if len(statusData.cpuPinCmd) > 0:
            inStr = "dplace -c " + \
                    str(statusData.gageBegModelCpu[basinNum]) + "-" + \
                    str(statusData.gageEndModelCpu[basinNum]) +  \
                    " python3 -u -m nwm_routing -V3 -f %s" %yamlPath
            fileObj.write(inStr)

        else:
            inStr = "python3 -u -m nwm_routing -V3 -f %s" %yamlPath
            fileObj.write(inStr)
    except:
        statusData.errMsg = "ERROR: Failure to create: " + outFile
        raise

    # Make the file an executable.
    cmd = "chmod +x " + outFile
    try:
        subprocess.call(cmd,shell=True)
    except:
        statusData.errMsg = "ERROR: Failure to convert: " + outFile + " to an executable."
        raise
