path=benchmarks/GEFCom2017_D_Prob_MT_hourly
for i in `seq 1 5`;
do
    echo "Run $i"
    start=`date +%s`
    echo 'Creating features...'
    python $path/qrf/compute_features.py --submission qrf

    echo 'Training and predicting...'
    python $path/qrf/train_score.py --data-folder $path/qrf/data --output-folder $path/qrf --seed $i

    end=`date +%s`
    echo 'Running time '$((end-start))' seconds'
done
echo 'Training and scoring are completed'
