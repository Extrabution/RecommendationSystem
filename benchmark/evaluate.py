import pandas as pd
import numpy as np
from tqdm import tqdm
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from surprise import KNNBasic
from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

ratings_df = pd.read_csv("../merged_dataset.csv")
ratings_df.drop(columns=["Unnamed: 0"], inplace=True)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['uid', 'iid', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=.25)

LOOCV = LeaveOneOut(n_splits=1, random_state=1)

train_loocv, test_loocv = list(LOOCV.split(data))[0]

RMSE = []
HITRATE = []
def GetTopN(predictions, n=10, minimumRating=4.0):
    topN = defaultdict(list)

    for userID, movieID, actualRating, estimatedRating, _ in predictions:
        if estimatedRating >= minimumRating:
            topN[userID].append((movieID, estimatedRating))

    for userID, ratings in topN.items():
        ratings.sort(key=lambda x: x[1], reverse=True)
        topN[userID] = ratings[:n]

    return topN


def HitRate(topNPredicted, leftOutPredictions):
    hits = 0
    total = 0

    # For each left-out rating
    for leftOut in leftOutPredictions:
        userID = leftOut[0]
        leftOutMovieID = leftOut[1]
        # Is it in the predicted top 10 for this user?
        hit = False
        for movieID, predictedRating in topNPredicted[userID]:
            if leftOutMovieID == movieID:
                hit = True
                break
        if (hit):
            hits += 1

        total += 1

    # Compute overall precision
    global HITRATE
    HITRATE.append(hits / total)
    return hits / total


def get_hitrate_results(algo, train_loocv, test_loocv):
    algo.fit(train_loocv)
    left_out_predictions = algo.test(test_loocv)
    loocv_anti_testset = train_loocv.build_anti_testset()
    all_predictions = algo.test(loocv_anti_testset)
    top_n_predicted = GetTopN(all_predictions)
    hitrate = HitRate(top_n_predicted, left_out_predictions)
    print(f'HitRate: {hitrate}')
    return all_predictions


def get_algo_results(algo, trainset, testset):
    algo.fit(trainset)
    predictions = algo.test(testset)
    global RMSE
    RMSE.append(accuracy.rmse(predictions))


def get_most_similar_movies(movies_df, movie_embeddings, trainset, target_movie_id, top_k=10):
    inner_movie_id = trainset.to_inner_iid(target_movie_id)
    sims = cosine_similarity(movie_embeddings, movie_embeddings)
    target_movie_sims_sorted = [trainset.to_raw_iid(x) for x in np.argsort(sims[inner_movie_id])[::-1]]
    most_similar_movies = movies_df.loc[target_movie_sims_sorted].iloc[:top_k]
    return most_similar_movies


def filter_predictions_for_user(predictions, user_id, movies_df, top_k=10):
    top_preds = sorted([pred for pred in predictions if pred.uid == user_id], key=lambda pred: pred.est, reverse=True)[
                :top_k]
    movie_ids = [pred.iid for pred in top_preds]
    relevant_movies = movies_df.loc[movie_ids]
    relevant_movies['rating'] = [pred.est for pred in top_preds]
    return relevant_movies


def get_algorithm_report(algo_class, trainset, testset, train_loocv, test_loocv, movies_df, target_movie_id=1,
                         target_user_id=1, top_k=10, algo_args=[], algo_kwargs={}, calc_most_similar=True):
    algo_inst = algo_class(*algo_args, **algo_kwargs)
    get_algo_results(algo_inst, trainset, testset)
    algo_inst_for_hitrate = algo_class(*algo_args, **algo_kwargs)
    all_predictions = get_hitrate_results(algo_inst_for_hitrate, train_loocv, test_loocv)
    if calc_most_similar:
        if hasattr(algo_inst_for_hitrate, 'qi'):
            sims = algo_inst_for_hitrate.qi
        else:
            sims = algo_inst_for_hitrate.sim

    predictions_for_user = filter_predictions_for_user(all_predictions, target_user_id, movies_df)

    return predictions_for_user.head(top_k)


class SVDWithTqdm(SVD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test(self, testset, verbose=False):
        # The ratings are translated back to their original scale.
        predictions = [self.predict(uid,
                                    iid,
                                    r_ui_trans,
                                    verbose=verbose)
                       for (uid, iid, r_ui_trans) in tqdm(testset, desc='making predictions')]
        return predictions


class KNNBasicWithTqdm(KNNBasic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test(self, testset, verbose=False):
        # The ratings are translated back to their original scale.
        predictions = [self.predict(uid,
                                    iid,
                                    r_ui_trans,
                                    verbose=verbose)
                       for (uid, iid, r_ui_trans) in tqdm(testset, desc='making predictions')]
        return predictions


algo_kwargs = dict(k=50, sim_options={'name': 'pearson', 'user_based': True, 'verbose': True})
print("EVALUATING SVD...")
svg_results = get_algorithm_report(SVDWithTqdm, trainset, testset, train_loocv, test_loocv,
                                   ratings_df, target_movie_id=242, target_user_id=196, top_k=10,
                                   calc_most_similar=False)
print("EVALUATING KNN...")
knn_results = get_algorithm_report(KNNBasicWithTqdm, trainset, testset, train_loocv, test_loocv,
                                   ratings_df, target_movie_id=242, target_user_id=196, top_k=10,
                                   calc_most_similar=False)

print("\n"*10)
print("-"*50)
print("RMSE SVD:", RMSE[0])
print("HitRate SVD:", HITRATE[0])
print("RMSE KNN:", RMSE[1])
print("HitRate KNN:", HITRATE[1])
