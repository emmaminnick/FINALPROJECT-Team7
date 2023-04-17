from typing import List, Dict, Union, Optional, Tuple
from collections import defaultdict
from com.twitter.simclusters_v2.common import ClusterId, SimClustersEmbedding, TweetId
from com.twitter.simclusters_v2.thriftscala import InternalId, SimClustersEmbeddingId
from com.twitter.simclustersann.thriftscala import ScoringAlgorithm, SimClustersANNConfig
from com.twitter.snowflake.id import SnowflakeId
from com.twitter.util import Duration, Time

ScoredTweet = Tuple[int, float]


# Defining the class ApproximateCosineSimilarity
class ApproximateCosineSimilarity:
    
    InitialCandidateMapSize = 16384

    #This is the max number of results that can be returned by the ANN algorithm
    MaxNumResultsUpperBound = 1000

    #This is the max age of a tweet that can be considered as a candidate for a given tweet
    MaxTweetCandidateAgeUpperBound = 175200
    
    def __call__(
        self,
        sourceEmbedding: SimClustersEmbedding,
        sourceEmbeddingId: SimClustersEmbeddingId,
        config: SimClustersANNConfig,
        candidateScoresStat: int,
        clusterTweetsMap: Dict[ClusterId, Optional[List[Tuple[TweetId, float]]]],
        clusterTweetsMapArray: Optional[Dict[ClusterId, Optional[List[Tuple[TweetId, float]]]]] = None
    ) -> List[ScoredTweet]:
        

        #Here we are getting the current time and the earliest and latest tweet id from the user
        now = Time.now()
        earliest_tweet_id = 0 if config.maxTweetCandidateAgeHours >= ApproximateCosineSimilarity.MaxTweetCandidateAgeUpperBound else SnowflakeId.firstIdFor(now - Duration.from_hours(config.maxTweetCandidateAgeHours))
        latest_tweet_id = SnowflakeId.firstIdFor(now - Duration.from_hours(config.minTweetCandidateAgeHours))
        candidate_scores_map = defaultdict(float)
        candidate_normalization_map = defaultdict(float)
        
        #In the following for loop, we are iterating over the clusterTweetsMap and getting the cluster_id and tweet_scores
        for cluster_id, tweet_scores in clusterTweetsMap.items():
            if tweet_scores and sourceEmbedding.contains(cluster_id):
                source_cluster_score = sourceEmbedding[cluster_id]
                
                for i in range(min(len(tweet_scores), config.maxTopTweetsPerCluster)):
                    tweet_id, score = tweet_scores[i]
                    
                    #Here we are checking if the tweet_id is not equal to the sourceEmbeddingId and if the tweet_id is between the earliest and latest tweet id
                    if InternalId(tweet_id) != sourceEmbeddingId.internalId and earliest_tweet_id <= tweet_id <= latest_tweet_id:
                        candidate_scores_map[tweet_id] += score * source_cluster_score
                        candidate_normalization_map[tweet_id] += score * score
        
        candidate_scores_stat(len(candidate_scores_map))
        

        #Here we are normalizing the scores
        processed_candidate_scores = []
        for candidate_id, score in candidate_scores_map.items():
            if config.annAlgorithm == ScoringAlgorithm.LogCosineSimilarity:
                processed_score = score / sourceEmbedding.logNorm / math.log(1 + candidate_normalization_map[candidate_id])
            elif config.annAlgorithm == ScoringAlgorithm.CosineSimilarity:
                processed_score = score / sourceEmbedding.l2norm / math.sqrt(candidate_normalization_map[candidate_id])
            elif config.annAlgorithm == ScoringAlgorithm.CosineSimilarityNoSourceEmbeddingNormalization:
                processed_score = score / math.sqrt(candidate_normalization_map[candidate_id])
            elif config.annAlgorithm == ScoringAlgorithm.DotProduct:
                processed_score = score
            else:
                raise ValueError(f"Invalid scoring algorithm {config.annAlgorithm}")
            
            #Here we are appending the processed scores to the processed_candidate_scores list
            processed_candidate_scores.append((candidate_id, processed_score))
        
        # take top M tweet candidates by score
        processed_candidate_scores.sort(key=lambda x: x[1], reverse=True)
        top_m = min(len(processed_candidate_scores), ApproximateCosineSimilarity.MaxNumResultsUpperBound)
        return processed_candidate_scores[:top_m]



