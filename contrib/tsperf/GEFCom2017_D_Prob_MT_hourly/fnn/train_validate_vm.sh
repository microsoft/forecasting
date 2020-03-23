#!/bin/bash
path=benchmarks/GEFCom2017_D_Prob_MT_hourly
for i in `seq 1 40`;
do
    echo "Parameter Set $i"
    start=`date +%s`
    echo 'Creating features...'
    python $path/fnn/compute_features.py --submission fnn

    echo 'Training and validation...'
    Rscript $path/fnn/train_validate.R $i

    end=`date +%s`
    echo 'Running time '$((end-start))' seconds'
done
