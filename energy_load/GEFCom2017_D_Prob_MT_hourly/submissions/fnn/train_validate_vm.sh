#!/bin/bash
path=energy_load/GEFCom2017_D_Prob_MT_hourly
for i in `seq 1 4`;
do
    echo "Parameter Set $i"
    start=`date +%s`
    #echo 'Creating features...'
    #python $path/submissions/fnn/feature_engineering.py --submission fnn

    echo 'Training and validation...'
    Rscript $path/submissions/fnn/train_validate.R $i

    end=`date +%s`
    echo 'Running time '$((end-start))' seconds'
done
