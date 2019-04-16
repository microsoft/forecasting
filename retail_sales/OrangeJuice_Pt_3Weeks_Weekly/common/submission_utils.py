import os
import retail_sales.OrangeJuice_Pt_3Weeks_Weekly.common.benchmark_paths as bp
import retail_sales.OrangeJuice_Pt_3Weeks_Weekly.common.benchmark_settings as bs

def create_submission(raw_predictions, seed, model_name=None, save_to_csv=True):
    """
    Utility function for creating the submission in the required format.
    
    Args:
        raw_predictions (pd.DataFrame): Raw prediction data frame including the following columns:
                                        store, brand, week, move, round
        seed (int): Random seed from 1 to 5 used in each experiment
        save_to_csv (bool)
    """
    submission = raw_predictions
    submission["weeks_ahead"] = submission.apply(lambda x: x["week"] - bs.TRAIN_END_WEEK_LIST[x["round"]], axis=1)
    submission.rename(columns={"move": "prediction"}, inplace=True)
    submission = submission[["round", "store", "brand", "week", "weeks_ahead", "prediction"]]
    if save_to_csv:
        filename = "submission_seed_" + str(seed) + ".csv"
        submission_path = os.path.join(bp.SUBMISSIONS_DIR, model_name)
        submission.to_csv(os.path.join(submission_path, filename), index=False)
    return submission