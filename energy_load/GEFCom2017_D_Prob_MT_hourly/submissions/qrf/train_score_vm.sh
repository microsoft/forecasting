path=energy_load/GEFCom2017_D_Prob_MT_hourly
for i in `seq 1 5`;
do
    echo "Run $i"
    start=`date +%s`
    echo 'Creating features...'
    python $path/submissions/qrf/feature_engineering.py --submission qrf

    echo 'Training and predicting...'
    python $path/submissions/qrf/train_score.py --data-folder $path/submissions/qrf/data --output-folder $path/submissions/qrf --seed $i

    end=`date +%s`
    echo 'Running time '$((end-start))' seconds'
done
echo 'Training and scoring are completed'
