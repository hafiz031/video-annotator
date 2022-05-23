import logging
import os
import sys

import fasttext
from concurrencycontrol.handle_race_condition import RaceConditionHandler
from utils.io_utils import create_dir_if_dne
from utils import config

from .preprocessors.algo_specific_preprocessing import AlgoSpecificPreprocessor

import time
import sys

from concurrencycontrol import prefetch_models as pm
from .hpsearch import HyperParameterSearchOnEstimator
from mlmodules.model_hyperparameters import ModelHyperparameters

from concurrencycontrol.redis_pubsub import RedisPubSub

pubsub = RedisPubSub()

class FastText:
    def __init__(self, account_id, bot_id, name_space = None):
        self.account_id = account_id
        self.bot_id = bot_id
        self.name_space = name_space


    # def fit(self, learning_rate = 0.5, epoch = 100, wordNgrams = 2, bucket_size = 100000, dim = 150, min_char_ngram = 3, 
    #         max_char_ngram = 5, loss_one_vs_all = True, hyperparam_search = True):
    # def fit(self, learning_rate = 0.38992253139718636, epoch = 74, wordNgrams = 5, bucket_size = 300000, dim = 138, min_char_ngram = 2, 
    #         max_char_ngram = 5, loss_one_vs_all = True, hyperparam_search = True):
    def fit(self, learning_rate = 0.43835694813968507, epoch = 130, wordNgrams = 5, bucket_size = 204416, dim = 100, min_char_ngram = 3, 
            max_char_ngram = 5, loss_one_vs_all = True, hyperparam_search = True):
        algo_spec_prep = AlgoSpecificPreprocessor(account_id = self.account_id, 
                                                  bot_id = self.bot_id,
                                                  name_space = self.name_space,
                                                  targetted_algo = "fasttext")
        preprocessed_data = algo_spec_prep.prepare_data_for_algo()
        # p int(preprocessed_data)
        create_dir_if_dne('data/mlmodules/temp/fasttext_temp') # Every relative path is understood with respect to the root dir of the project folder.
        data_path = f'data/mlmodules/temp/fasttext_temp/fasttext_{self.account_id}_{self.bot_id}.data'

        # create_dir_if_dne('temp/dumps') # Every relative path is understood with respect to the root dir of the project folder.
        # data_path = f'{self.account_id}_{self.bot_id}.data'

        with open(data_path, mode='w', encoding='utf-8') as f:
            f.write('\n'.join(preprocessed_data))

        # RaceConditionHandler() is to avoid race conditions happen during the reading of the model file while it being written on.
        if self.name_space.lower() != 'under_evaluation':
            cc = RaceConditionHandler(self.account_id, self.bot_id)
            latest_timestamp = cc.generate_timestamp()
        else:
            latest_timestamp = 'UNDER_EVALUATION'

        create_dir_if_dne('data/mlmodules/trained_binaries/fasttext_trained_model')
        # Remember! Following line shouldn't contain any extension, as FastText itself will creates two files (.bin and .vec) using this base name.
        model_path = f'data/mlmodules/trained_binaries/fasttext_trained_model/ft_{latest_timestamp}_{self.account_id}_{self.bot_id}'
        
        if hyperparam_search == False:
            if loss_one_vs_all == True:
                # Training using One Vs All
                cmd = f'./fastText-0.9.2/fasttext supervised -input {data_path} -output {model_path} ' \
                    f'-lr {learning_rate} -epoch {epoch} -wordNgrams {wordNgrams} -bucket {bucket_size} -dim {dim} \
                        -minn {min_char_ngram} -maxn {max_char_ngram} -loss one-vs-all -thread 4 -minCount 1'
            else:
                # Training using Hierarchical Softmax
                cmd = f'./fastText-0.9.2/fasttext supervised -input {data_path} -output {model_path} ' \
                    f'-lr {learning_rate} -epoch {epoch} -wordNgrams {wordNgrams} -bucket {bucket_size} -dim {dim} \
                        -minn {min_char_ngram} -maxn {max_char_ngram} -loss hs -thread 4 -minCount 1'
            
            os.system(cmd)
            print("INSIDE TRAIN")
            print(model_path)
        else:
            hyp = ModelHyperparameters(model_name = "fasttext").get_hyperparameters()
            # It is the responsibility of the HyperParameterSearchOnEstimator to save the model itself,
            # or get it done by other.
            hyperparam_search_estimator = HyperParameterSearchOnEstimator(estimator_name = "fasttext",
                                                        model_path = model_path,
                                                        account_id = self.account_id,
                                                        bot_id = self.bot_id,
                                                        name_space = self.name_space,
                                                        data_path = data_path,
                                                        preprocessed_data = preprocessed_data,
                                                        grid_parameters = hyp)

            best_params = hyperparam_search_estimator.do_hyperparameter_search()
            # Getting the best parameters, now fit() with them (this time hyperparam_search = False)
            try:
                self.fit(learning_rate = best_params["learning_rate"], epoch = best_params["epoch"], wordNgrams = best_params["wordNgrams"], 
                        bucket_size = best_params["bucket_size"], dim = best_params["dim"], min_char_ngram = best_params["min_char_ngram"], 
                        max_char_ngram = best_params["max_char_ngram"], loss_one_vs_all = best_params["loss_one_vs_all"], hyperparam_search = False)
            except Exception as exception:
                logging.error(exception)
                logging.error("Couldn't train the final model with the parameters found from gridsearch.")
            
            # We must return it here, as it is a recursion (we called fit() from fit()), 
            # hence we have to prevent the program from going forward from here
            # otherwise it will set an older timestamp which is associated with parent level call.
            # Again, the latest model and associated timestamp was already saved from this recursive call to child,
            # hence, we are preventing it from setting the timestamp again with older value, when the recursion 
            # backtracks to this parent level.
            # Hence, returning it immediately from here.
            return


        # When the model building is completed, save the timestamp in the database associated with it.
        ###############
        if self.name_space.lower() != 'under_evaluation':
            print("#"*50)
            print("fsfssd")
            print(epoch)
            print(latest_timestamp)
            print("#"*50)
            cc.set_latest_model_timestamp(latest_timestamp)
        ##############
        logging.info("[INFO] SAVED_MODEL_PATH:" + model_path)

        ## Deleting old models
        try:
            print("INSIDE DELETE OLDER MODELS")
            print(model_path)
            print("INSIDE DELETE OLDER MODELS")
            cc.delete_older_model_files(model_path) # model path shouldn't contain file extension (not it is)
        except:
            pass

        ## After each training, we must make sure the model is prefetched with the new version under this same account_id, bot_id and name_space
        tolerance = 3
        while tolerance:
            if os.path.isfile(model_path + '.bin'):
                print("INDIDE PREFETECH")
                print(model_path)
                print("INDIDE PREFETECH")
                pm.prefetch_model_to_memory(algo_name = 'fasttext',
                                        account_id = self.account_id,
                                        bot_id = self.bot_id,
                                        name_space = self.name_space,
                                        model_file = fasttext.load_model(model_path + '.bin'))
                break
            else:
                time.sleep(3) # For safety, larger model might need more time to save (and therefore, be available for loading from disk)
            tolerance -= 1

        # Now publish message to other workers to update their models
        pubsub.publish_to_channel(channel = "sync-ml-models",
                                  account_id = self.account_id,
                                  bot_id = self.bot_id,
                                  name_space = self.name_space,
                                  algo_name = "fasttext",
                                  model_path = model_path)


    def filter_predictions(self, responses, probabilities, confidence_threshold):
        """If the intent having the maximum confidence has a lower confidenc than the allowed value.
        Therefore, return it alone. It will go to fallback."""
        if probabilities[0] < config.MINIMUM_REQUIRED_PREDICTION_CONFIDENCE:
            return [int(responses[0][9:])], [min(1.0, probabilities[0])]
        elif probabilities[0] >= confidence_threshold:
            return [int(responses[0][9:])], [min(1.0, probabilities[0])]
        else:
            _responses = []
            _probabilities = []
            for response, probability in zip(responses, probabilities):
                if probability > config.MINIMUM_REQUIRED_PREDICTION_CONFIDENCE:
                    _responses.append(int(response[9:]))
                    _probabilities.append(min(1.0, probability))
            return _responses[:3], _probabilities[:3] # take top 3


    def predict(self, text, confidence_threshold = 0):
        if confidence_threshold:
            confidence_threshold = float(confidence_threshold)
        else:
            confidence_threshold = 0

        if self.name_space.lower() != 'under_evaluation':
            cc = RaceConditionHandler(self.account_id, self.bot_id)
            existing_latest_model_timestamp = cc.get_latest_model_timestamp()
        else:
            existing_latest_model_timestamp = 'UNDER_EVALUATION'
        print("INSIDE PREDICT")
        print(existing_latest_model_timestamp)
        model_path = f'data/mlmodules/trained_binaries/fasttext_trained_model/ft_{existing_latest_model_timestamp}_{self.account_id}_{self.bot_id}.bin'

        if os.path.isfile(model_path):

            start_time = time.time()
            print("[TIMER] Started loading model", file = sys.stderr)

            model = pm.get_prefetched_model_from_memory(algo_name = 'fasttext',
                                            account_id = self.account_id,
                                            bot_id = self.bot_id,
                                            name_space = self.name_space)
            
            if model == None:
                """This block just handles a corner case, if it was not there, we needed to train the model at least once for each time we start
                ml service, as, without it, the model is only prefetched into memory just after training. So, to remove this dependency, this 
                block loads the model into memory (if not previously loaded ) from disk without the need of training again.
                """
                if os.path.isfile(model_path): # if the model file exists load it
                    pm.prefetch_model_to_memory(algo_name = 'fasttext',
                                        account_id = self.account_id,
                                        bot_id = self.bot_id,
                                        name_space = self.name_space,
                                        model_file = fasttext.load_model(model_path))
                    model = pm.get_prefetched_model_from_memory(algo_name = 'fasttext', # now update the model variable
                                                account_id = self.account_id,
                                                bot_id = self.bot_id,
                                                name_space = self.name_space)


            now = time.time()
            print("[TIMER] Time taken for loading model " + str(now - start_time), file = sys.stderr)

            start_time = time.time()
            print("[TIMER] Started fetching prediction", file = sys.stderr)
            responses, probabilities = model.predict(text, k = 3)
            # print(responses)
            # print(probabilities)
            responses, probabilities = self.filter_predictions(responses, probabilities, confidence_threshold)
            print("RESPONSES_&_PROBABILITIES", file = sys.stderr)
            print(responses, file = sys.stderr)
            print(probabilities, file = sys.stderr)
            print("RESPONSES_&_PROBABILITIES", file = sys.stderr)
            now = time.time()
            print("[TIMER] Time taken for fetching prediction " + str(now - start_time), file = sys.stderr)

            return responses, probabilities
        else: # if the model file is not found
            return [None], [None]

