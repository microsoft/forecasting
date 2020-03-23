#!/bin/bash
path=benchmarks/GEFCom2017_D_Prob_MT_hourly
for i in `seq 1 5`;
do
    echo "Run $i"
    start=`date +%s`
    echo 'Creating features...'
    python $path/baseline/compute_features.py --submission baseline

    echo 'Training and predicting...'
    Rscript $path/baseline/train_predict.R $i

    end=`date +%s`
    echo 'Running time '$((end-start))' seconds'
done