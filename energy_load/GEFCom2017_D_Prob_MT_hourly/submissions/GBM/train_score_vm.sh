#!/bin/bash
path=energy_load/GEFCom2017_D_Prob_MT_hourly
for i in `seq 1 5`;
do
    echo "Run $i"
    start=`date +%s`
    echo 'Creating features...'
    python $path/submissions/GBM/feature_engineering.py --submission GBM

    echo 'Training and predicting...'
    Rscript $path/submissions/GBM/train_predict.R $i

    end=`date +%s`
    echo 'Running time '$((end-start))' seconds'
done
