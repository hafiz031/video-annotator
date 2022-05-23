from mlmodules.services.languagedetect.lang_detect import LanguageDetect
from .fasttext_api import FastText
# from .mlp_api import MultilayerPerceptron
from mlmodules.services.mlutils import find_bad_words_count
from mlmodules.services.smalltalk.small_talk_service import get_smalltalk_response
from datapipeline.get_corpus_n_knowledgebases import GetCorpusKB
from mlmodules.preprocessors.basic_preprocessing import BasicPreprocessor
import logging
from mlmodules.services.outdomain.out_of_domain_detector import OutOfDomainDetector
from mlmodules.elastic_search import ElasticSearch
import json
from flask import jsonify
from datapipeline.get_corpus_n_knowledgebases import GetCorpusKB

from utils import config

import sys
import time
language_detection_model = LanguageDetect() # This was taking additional 1 sec, when it was inside __init__

# from mlmodules.preprocessors.semantic_spell_correction_ import SemanticSpellCorrector

from concurrencycontrol.redis_pubsub import RedisPubSub
pubsub = RedisPubSub()
es = ElasticSearch()

class Model:
    def __init__(self, account_id, bot_id, name_space, model_name = None):
        self.account_id = account_id
        self.bot_id = bot_id
        self.name_space = name_space
        self.model_name = model_name

        if self.model_name == "fasttext":
            self.model = FastText(account_id = self.account_id, bot_id = self.bot_id, name_space = self.name_space)
        # elif self.model_name == "mlp":
        #     self.model = MultilayerPerceptron(account_id = self.account_id, bot_id = self.bot_id, name_space = self.name_space)
        # if self.model_name == "tensorflow.charLSTM":
        #     self.model = MyLSTM(account_id = self.account_id, bot_id = self.bot_id, name_space = self.name_space)

    def train(self):

        corp_kb = GetCorpusKB(bot_id = self.bot_id, account_id = self.account_id, name_space = self.name_space)
        corp_kb.sync_all_local_backup_of_corpus_kb()
        corp_kb.get_corpus(shuffle = True, read_from_cache = False, build_cache = True)

        self.model.fit(hyperparam_search = config.HYPERPARAMETER_SEARCH)

        oodd = OutOfDomainDetector(account_id = self.account_id,
                        bot_id = self.bot_id,
                        name_space = self.name_space)
        logging.info("Outdomain detector training started")
        oodd.train_and_save_model_outdomain_detector()
        logging.info("Outdomain detector training finish")
        corp_kb.clean_in_memory_cache(key = f"a_{self.account_id}__b_{self.bot_id}__ns_{self.name_space}.json")

        # Updating the CDF for current instance is already done by CDF method (it is implemented there as spell corr module has a dependency
        # on it and spell check module comes after it)
        # Also need to tell the other instances to take the updated top-tokens list
        pubsub.publish_to_channel(channel = "sync-ml-models",
            account_id = self.account_id,
            bot_id = self.bot_id,
            name_space = self.name_space,
            algo_name = "top_tokens",
            model_path = '')


    # def predict(self, text): # Here all the db queries are being called (for syno, bad words, intent_name)
    #     _start_time = time.time()
    #     print("[TIMER] ##################", file = sys.stderr)
    #     text_original = text
    #     bp = BasicPreprocessor(bot_id = self.bot_id, account_id = self.account_id, name_space = self.name_space, is_predict = True)
    #     text = bp.replace_synonyms_with_originals(text)
    #     badwords_count = find_bad_words_count(self.bot_id, bp.do_basic_preprocessing(text, remove_punc = False, remove_num = False, lemmatize = False, spell_check_n_correct = False, remove_predefined_stopwords = False))

    #     if badwords_count > 0:
    #         prediction_response = {
    #                     'status': False,
    #                     'sentiment': {
    #                             'is_badword_exists': True,
    #                             'badwords_count': badwords_count,
    #                     }}

    #         logging.info("prediction_response")
    #         logging.info(prediction_response)
    #         return prediction_response

    #     text_for_ood_chk = text

    #     # smalltalk checking is kept before OOD checking (OOD of this ml-service), as it might be a case when we don't have
    #     # any training data for bot yet. In that case if smalltalk checking was not kept before, then, any input
    #     # will be considered as OOD, even if they were smalltalks. As OOD will not allow the program to reach
    #     # to smalltalk call.
    #     # Keeping smalltalk before OOD was not possible before, as the previous version of smalltalk service didn't have any OOD checking for
    #     # itself inside its own implementation. Hence, it had to relay on the OOD checking of ml-service. But this is
    #     # no longer required. As the newer implementation of smalltalk service itself has its own OOD check, so if some query
    #     # doesn't even a smalltalk, it will be handled in smalltalk-service and the service will return False. Therefore,
    #     # it will be further checked if the query is an OOD wrt ml-service or not. If not, than the intent will be identified.
    #     # Otherwise, the query will be considered as OOD finally.
    #     text = bp.do_basic_preprocessing(text, remove_predefined_stopwords = False)
    #     smalltalk = get_smalltalk_response(text)

    #     if smalltalk['status']:
    #         return {
    #             'smalltalk': smalltalk
    #         }


    #     oodd = OutOfDomainDetector(account_id = self.account_id,
    #                         bot_id = self.bot_id,
    #                         name_space = self.name_space)
    #     try:
    #         ret = oodd.test_if_outdomain(text_for_ood_chk)

    #         if ret == True:
    #             with open('mlmodules/services/outdomain/sample_ood_response.json', 'r') as jf:
    #                 prediction_response = json.load(jf)

    #             logging.info("prediction_response OOD")
    #             logging.info(prediction_response)

    #             return prediction_response

    #     except Exception as e:
    #         logging.error("ERROR IN OOD PRED")
    #         logging.error(e)


    #     # remove_num = False, lemmatize = False, spell_check_n_correct = False as they are already done in previous call, so no need to do them again
    #     text = bp.do_basic_preprocessing(text, remove_num = False, lemmatize = False, spell_check_n_correct = False, remove_predefined_stopwords = True)
    #     intent_id, confidence = self.model.predict(text)

    #     if intent_id == None and confidence == None:
    #         return {
    #             'status': False,
    #             'message': 'Model does not exist'
    #             }

    #     intent_name = GetCorpusKB(account_id = self.account_id, bot_id = self.bot_id, intent_id = intent_id, name_space = self.name_space).get_intent_name()

    #     start_time = time.time()
    #     print("[TIMER] Started Language Detection", file = sys.stderr)
    #     lang = language_detection_model.detect(text = bp.do_basic_preprocessing(text_original, lemmatize = False, spell_check_n_correct = False, remove_predefined_stopwords = False))
    #     now = time.time()
    #     print("[TIMER] Time taken for Language Detection " + str(now - start_time), file = sys.stderr)

    #     _now = time.time()
    #     print("[TIMER] ################## " + str(_now - _start_time), file = sys.stderr)

    #     return {
    #         'smalltalk': smalltalk,
    #         'status': True,
    #         'payload': {
    #             'intent': {
    #                 'id': intent_id,
    #                 'name': intent_name[0],
    #                 'account_id': self.account_id,
    #                 'bot_id': int(self.bot_id)
    #             },
    #             'confidence': confidence,
    #             'lang': lang
    #         },
    #         'sentiment': {
    #             'is_badword_exists': badwords_count > 0,
    #             'badwords_count': badwords_count,
    #         }

    #     }


    def predict(self, text, confidence_threshold = None, use_elastic_search = True): # Here all the db queries are being called (for syno, bad words, intent_name)
        time_stamp = round(int(time.time() * 1000))

        if config.SYNC_QUERY_LOG_IN_EVERY_SECONDS >= 0:
            corp_kb = GetCorpusKB(bot_id = self.bot_id, account_id = self.account_id, name_space = self.name_space)

        _start_time = time.time()
        print("[TIMER] ##################", file = sys.stderr)

        bp = BasicPreprocessor(bot_id = self.bot_id, account_id = self.account_id, name_space = self.name_space, is_predict = True)
        text_original = text

        """Lowercasing, newline removal, whitespace removal
        """
        prep0 = bp.do_basic_preprocessing(text,
                                            remove_punc = False,
                                            remove_num= False,
                                            lemmatize = False,
                                            spell_check_n_correct = False,
                                            remove_predefined_stopwords = False,
                                            ner_replacement = False)

        """Badword detection
        From previous step, additionally we will be substituting synonyms
        """
        prep1 = bp.replace_synonyms_with_originals(prep0) #[UPDATE: No, it doesn't] this also returns making the text lowercase, so calling ner_replacement after it, after the next step, no lowecasing operations will be performed
        prep1 = bp.do_basic_preprocessing(prep1,
                                            remove_punc = False,
                                            remove_num= False,
                                            lemmatize = False,
                                            lowercasing = False,
                                            spell_check_n_correct = False,
                                            remove_predefined_stopwords = False,
                                            ner_replacement = True) # only when ner_replacement is true, ner replacement to be done with minimal processed text, as this model is trained on minimal processed text

        badwords_count = find_bad_words_count(self.bot_id, prep1)

        if badwords_count > 0:
            if config.SYNC_QUERY_LOG_IN_EVERY_SECONDS >= 0:
                corp_kb.log_it(query_text = str(text_original),
                        query_type = "bad_word",
                        bot_id = int(self.bot_id),
                        time_stamp = time_stamp)

            prediction_response = {
                        'status': False,
                        'sentiment': {
                                'is_badword_exists': True,
                                'badwords_count': badwords_count,
                        }}

            logging.info("prediction_response")
            logging.info(prediction_response)
            return prediction_response


        """Remove punctuations and number, skip all the preprocessing steps which are already done,
        by setting them to False.
        """
        prep2 = bp.do_basic_preprocessing(prep1,
                                            remove_punc = True,
                                            remove_num = True,
                                            lemmatize = False,
                                            lowercasing = False,
                                            spell_check_n_correct = False,
                                            remove_predefined_stopwords = False)

        """Now pass this to detect smalltalk"""
        # smalltalk checking is kept before OOD checking (OOD of this ml-service), as it might be a case when we don't have
        # any training data for bot yet. In that case if smalltalk checking was not kept before, then, any input
        # will be considered as OOD, even if they were smalltalks. As OOD will not allow the program to reach
        # to smalltalk call.
        # Keeping smalltalk before OOD was not possible before, as the previous version of smalltalk service didn't have any OOD checking for
        # itself inside its own implementation. Hence, it had to relay on the OOD checking of ml-service. But this is
        # no longer required. As the newer implementation of smalltalk service itself has its own OOD check, so if some query
        # doesn't even a smalltalk, it will be handled in smalltalk-service and the service will return False. Therefore,
        # it will be further checked if the query is an OOD wrt ml-service or not. If not, than the intent will be identified.
        # Otherwise, the query will be considered as OOD finally.

        smalltalk = get_smalltalk_response(prep2)

        if smalltalk['status']:
            smalltalk["payload"]["lang"] = language_detection_model.detect(text = bp.do_basic_preprocessing(text_original, lemmatize = False, spell_check_n_correct = False, remove_predefined_stopwords = False))

            if config.SYNC_QUERY_LOG_IN_EVERY_SECONDS >= 0:
                corp_kb.log_it(query_text = str(text_original),
                        query_type = "smalltalk",
                        bot_id = int(self.bot_id),
                        time_stamp = time_stamp)
            return {
                'smalltalk': smalltalk
            }


        """Now pass it to OOD detector"""
        oodd = OutOfDomainDetector(account_id = self.account_id,
                            bot_id = self.bot_id,
                            name_space = self.name_space)
        try:
            # ret = oodd.test_if_outdomain(prep3)
            ret = oodd.test_if_outdomain(prep2)
            if ret == True:
                if config.SYNC_QUERY_LOG_IN_EVERY_SECONDS >= 0:
                    corp_kb.log_it(query_text = str(text_original),
                        query_type = "out_domain_input",
                        bot_id = int(self.bot_id),
                        time_stamp = time_stamp)

                with open('mlmodules/services/outdomain/sample_ood_response.json', 'r') as jf:
                    prediction_response = json.load(jf)

                logging.info("prediction_response OOD")
                logging.info(prediction_response)

                return prediction_response

        except Exception as e:
            logging.error("ERROR IN OOD PRED")
            logging.error(e)


        """For Classification Model we will be applying all preprocessing,
        but as we have applied many of them, so we will be skipping them by setting them to False.
        """

        spell_chk = True
        if config.ML_ALGORITHM == "tensorflow.charLSTM":
            spell_chk = False

        prep3 = bp.do_basic_preprocessing(prep2,
                                            remove_punc = False,
                                            remove_num = False,
                                            lemmatize = True,
                                            lowercasing = False,
                                            spell_check_n_correct = spell_chk, ################## for LSTM it is turned to False
                                            # spell_check_n_correct=False,
                                            remove_predefined_stopwords = True,
                                            ner_replacement = False)
        print("############# TEXT ###################")
        print(prep3)
        print("############# TEXT ###################")

        print("Pass the same thing to Intent Classifier", file=sys.stderr)
        print(prep3, file=sys.stderr)
        print("Pass the same thing to Intent Classifier", file=sys.stderr)
        """Pass the same thing to Intent Classifier"""
        intent_ids, confidences = self.model.predict(prep3, confidence_threshold)

        if intent_ids[0] == None and confidences[0] == None:
            return {
                'status': False,
                'message': 'Model does not exist'
                }

        # intent_names = []
        # for intent_id in intent_ids:
        #     intent_name = GetCorpusKB(account_id = self.account_id, bot_id = self.bot_id, intent_id = intent_id, name_space = self.name_space).get_intent_name()
        #     intent_names.append(intent_name)

        start_time = time.time()
        print("[TIMER] Started Language Detection", file = sys.stderr)

        """For language detection we cannot lemmatize as it might remove prefixes and suffixes which can lead to misinterpretation of language
        specially for Banglish. So, we are preprocessing it separately.
        """
        lang = language_detection_model.detect(text = bp.do_basic_preprocessing(text_original, lemmatize = False, spell_check_n_correct = False, remove_predefined_stopwords = False))
        now = time.time()
        print("[TIMER] Time taken for Language Detection " + str(now - start_time), file = sys.stderr)

        _now = time.time()
        print("[TIMER] ################## " + str(_now - _start_time), file = sys.stderr)

        if config.SYNC_QUERY_LOG_IN_EVERY_SECONDS >= 0:
            corp_kb.log_it(query_text = str(text_original),
                        query_type = "intent_sample",
                        score = confidences[0], # Saving the maximum one
                        bot_id = int(self.bot_id),
                        confidence_threshold = confidence_threshold,
                        time_stamp = time_stamp)

        # _intent_ids = intent_ids
        # _confidences = confidences
        # if confidences[0] < config.MINIMUM_REQUIRED_PREDICTION_CONFIDENCE:
        # # if True: # DBG
        #     intent_ids, confidences = es.predict(text = prep3, bot_id = int(self.bot_id))
        #     if len(intent_ids) <= 0 or len(confidences) <= 0:
        #         intent_ids = _intent_ids
        #         confidences = _confidences
                         
        ## ElasticSearch Start
        if use_elastic_search:
            _intent_ids = intent_ids # Temporarily holding main model's outputs
            _confidences = confidences

            if config.ELASTIC_SEARCH_STRATEGY == "base_model_only":
                if confidence_threshold == None:
                    confidence_threshold = 0.5
                else:
                    confidence_threshold = float(confidence_threshold)

                if _confidences[0] >= confidence_threshold:
                    intent_ids = [_intent_ids[0]]
                    confidences = [_confidences[0]]
                    print(f"BASE_MODEL_PRED: Confidence high, giving a single response.", file = sys.stderr)
                    print(f"BASE_PRED_ID: {intent_ids}", file = sys.stderr)
                    print(f"BASE_PRED_CONFIDENCE: {confidences}", file = sys.stderr)
                else:
                    intent_ids = []
                    confidences = []

                    for id, cf in zip(_intent_ids, _confidences):
                        if cf >= config.MINIMUM_REQUIRED_PREDICTION_CONFIDENCE and cf < confidence_threshold:
                            intent_ids.append(id)
                            confidences.append(cf)
                    intent_ids = intent_ids[:config.ELASTIC_TOP_K]
                    confidences = confidences[:config.ELASTIC_TOP_K]

                    if len(intent_ids) > 0:
                        print(f"BASE_MODEL_PRED: Confidence low, giving suggestive responses.", file = sys.stderr)
                        print(f"BASE_SUGGEST_IDs: {intent_ids}", file = sys.stderr)
                        print(f"BASE_SUGGEST_CONFIDENCEs: {confidences}", file = sys.stderr)
                    else:
                        print(f"BASE_MODEL_PRED: None of the predicted intents satisfy the minimum required confidence", file = sys.stderr)
                        print(f"...so, returning the first prediction with hardcoded 0 confidence to force a mandatory fallback.", file = sys.stderr)
                        intent_ids = [_intent_ids[0]]
                        confidences = [0]

            if config.ELASTIC_SEARCH_STRATEGY == "base_model_with_elastic_validation":
                if confidence_threshold == None:
                    confidence_threshold = 0.5
                else:
                    confidence_threshold = float(confidence_threshold)

                if _confidences[0] >= confidence_threshold:
                    intent_ids, confidences = es.predict(text = prep3, bot_id = int(self.bot_id), top_k = config.ELASTIC_TOP_K)
                    print(f"ElasticReturnIDs: {intent_ids}", file = sys.stderr)
                    print(f"ElasticReturnConfs: {confidences}", file = sys.stderr)

                    if len(intent_ids) <= 0 or len(confidences) <= 0:
                        print("ELAST_LOG: ElasticSearch returned blank response. Forwording main model's output with a mandatory fallback.", file = sys.stderr)
                        """In this case, as ElasticSearch output is empty, we have no options but go with main model's output.
                        Empty output is returned sometimes from ElasticSearch if no good matches are found from that end.
                        But to reduce the False positive cases we need to pass main model's confidence = 0 as ElasticSearch doesn't 
                        find any appropriate response for this query."""
                        intent_ids = [_intent_ids[0]] # only taking the first one as our target to reduce the False Positives.
                        confidences = [0] # Also not taking the main model's output when the elasticsearch doesn't return anything
                    else:
                        assert type(_intent_ids[0]) == type(intent_ids[0]) # found: both are integers
                        if _intent_ids[0] in intent_ids[:config.ELASTIC_SHORTLIST_K]:
                            intent_ids = [_intent_ids[0]]
                            confidences = [_confidences[0]]
                            print(f"BASE_MODEL_PRED: Confidence high, also found in elastic shortlisted IDs. So, giving a single response.", file = sys.stderr)
                            print(f"BASE_PRED_ID: {intent_ids}", file = sys.stderr)
                            print(f"BASE_PRED_CONFIDENCE: {confidences}", file = sys.stderr)
                        else:
                            intent_ids = [_intent_ids[0]]
                            confidences = [0]
                            print(f"BASE_MODEL_PRED: Confidence high, but NOT found in elastic shortlisted IDs. So, giving a mandatory fallback.", file = sys.stderr)
                            print(f"BASE_PRED_ID: {intent_ids}", file = sys.stderr)
                            print(f"BASE_PRED_CONFIDENCE: {confidences}", file = sys.stderr)

                else:
                    intent_ids = []
                    confidences = []

                    for id, cf in zip(_intent_ids, _confidences):
                        if cf >= config.MINIMUM_REQUIRED_PREDICTION_CONFIDENCE and cf < confidence_threshold:
                            intent_ids.append(id)
                            confidences.append(cf)
                    intent_ids = intent_ids[:config.ELASTIC_TOP_K]
                    confidences = confidences[:config.ELASTIC_TOP_K]

                    if len(intent_ids) > 0:
                        print(f"BASE_MODEL_PRED: Confidence low, giving suggestive responses.", file = sys.stderr)
                        print(f"BASE_SUGGEST_IDs: {intent_ids}", file = sys.stderr)
                        print(f"BASE_SUGGEST_CONFIDENCEs: {confidences}", file = sys.stderr)
                    else:
                        print(f"BASE_MODEL_PRED: None of the predicted intents satisfy the minimum required confidence", file = sys.stderr)
                        print(f"...so, returning the first prediction with hardcoded 0 confidence to force a mandatory fallback.", file = sys.stderr)
                        intent_ids = [_intent_ids[0]]
                        confidences = [0]

            elif config.ELASTIC_SEARCH_STRATEGY == "standalone":
                if _confidences[0] < config.MINIMUM_REQUIRED_PREDICTION_CONFIDENCE:
                # if True: # DBG
                    intent_ids, confidences = es.predict(text = prep3, bot_id = int(self.bot_id), top_k = config.ELASTIC_TOP_K)
                    print(f"ElasticReturnIDs: {intent_ids}", file = sys.stderr)
                    print(f"ElasticReturnConfs: {confidences}", file = sys.stderr)

                    if len(intent_ids) <= 0 or len(confidences) <= 0:
                        print("ELAST_LOG: ElasticSearch returned blank response. Forwording main model's output.", file = sys.stderr)
                        intent_ids = _intent_ids
                        confidences = _confidences

            elif config.ELASTIC_SEARCH_STRATEGY == "associated":

                if confidence_threshold == None:
                    confidence_threshold = 0.5
                else:
                    confidence_threshold = float(confidence_threshold)

                if _confidences[0] >= confidence_threshold:
                # if True: # DBG
                    intent_ids, confidences = es.predict(text = prep3, bot_id = int(self.bot_id), top_k = config.ELASTIC_TOP_K)
                    print(f"ElasticReturnIDs: {intent_ids}", file = sys.stderr)
                    print(f"ElasticReturnConfs: {confidences}", file = sys.stderr)

                    if len(intent_ids) <= 0 or len(confidences) <= 0:
                        print("ELAST_LOG: ElasticSearch returned blank response. Forwording main model's output with a mandatory fallback.", file = sys.stderr)
                        """In this case, as ElasticSearch output is empty, we have no options but go with main model's output.
                        Empty output is returned sometimes from ElasticSearch if no good matches are found from that end.
                        But to reduce the False positive cases we need to pass main model's confidence = 0 as ElasticSearch doesn't 
                        find any appropriate response for this query."""
                        intent_ids = [_intent_ids[0]] # only taking the first one as our target to reduce the False Positives.
                        confidences = [0] # Also not taking the main model's output when the elasticsearch doesn't return anything
                    else:
                        """It will be accessed if ElasticSearch gives a non-empty output. Matching with only
                        the best output from the main model."""
                        assert type(_intent_ids[0]) == type(intent_ids[0]) # found: both are integers
                        if _intent_ids[0] in intent_ids:
                            print("ELAST_LOG: Base model's best response found in elastic suggestions.", file = sys.stderr)
                            # The main model's best prediction is within the ELASTIC_TOP_K
                            intent_ids = [_intent_ids[0]] # only taking the first one as our target to reduce the False Positives.
                            confidences = [_confidences[0]]
                        else:
                            print("ELAST_LOG: Base model's best response NOT found in elastic suggestions.", file = sys.stderr)
                            intent_ids = [_intent_ids[0]] # only taking the first one as our target to reduce the False Positives.
                            confidences = [0] # Purposefully making it 0 to force to fallback
                else:
                    print("ELAST_LOG: Fallback in main model.", file = sys.stderr)
                    intent_ids = [_intent_ids[0]] # only taking the first one as our target to reduce the False Positives.
                    confidences = [_confidences[0]] # This will eventually go to fallback from chat server

            elif config.ELASTIC_SEARCH_STRATEGY == "associated_with_suggestions":

                if confidence_threshold == None:
                    confidence_threshold = 0.5
                else:
                    confidence_threshold = float(confidence_threshold)

                intent_ids, confidences = es.predict(text = prep3, bot_id = int(self.bot_id), top_k = config.ELASTIC_TOP_K)
                if _confidences[0] >= confidence_threshold:
                # if True: # DBG
                    print(f"ElasticReturnIDs: {intent_ids}", file = sys.stderr)
                    print(f"ElasticReturnConfs: {confidences}", file = sys.stderr)

                    if len(intent_ids) <= 0 or len(confidences) <= 0:
                        print("ELAST_LOG: ElasticSearch returned blank response. Forwording main model's output.", file = sys.stderr)
                        """In this case, as ElasticSearch output is empty, we have no options but go with main model's output.
                        Empty output is returned sometimes from ElasticSearch if no good matches are found from that end.
                        But to reduce the False positive cases we need to pass main model's confidence = 0 as ElasticSearch doesn't 
                        find any appropriate response for this query."""
                        intent_ids = [_intent_ids[0]] # only taking the first one as our target to reduce the False Positives.
                        confidences = [0] # Also not taking the main model's output when the elasticsearch doesn't return anything
                    else:
                        """It will be accessed if ElasticSearch gives a non-empty output. Matching with only
                        the best output from the main model."""
                        assert type(_intent_ids[0]) == type(intent_ids[0]) # found: both are integers
                        if _intent_ids[0] in intent_ids[:config.ELASTIC_SHORTLIST_K]:
                            print(f"ELAST_LOG: Base model's best response found in elastic suggestions within top: {config.ELASTIC_SHORTLIST_K}", file = sys.stderr)
                            # The main model's best prediction is within the ELASTIC_TOP_K
                            intent_ids = [_intent_ids[0]] # only taking the first one as our target to reduce the False Positives.
                            confidences = [_confidences[0]]
                        else:
                            print("ELAST_LOG: Base model's best response NOT found in elastic suggestions.", file = sys.stderr)
                else:
                    print("ELAST_LOG: Fallback in main model.", file = sys.stderr)
        ## ElasticSearch END

        if intent_ids[0] == None and confidences[0] == None:
            return {
                'status': False,
                'message': 'Model does not exist'
                }

        payload_segments = []
        for intent_id, confidence in zip(intent_ids, confidences):
            intent_name = GetCorpusKB(account_id = self.account_id, bot_id = self.bot_id, intent_id = intent_id, name_space = self.name_space).get_intent_name()
            segment = {
                'intent': {
                    'id': intent_id,
                    'name': intent_name[0],
                    'account_id': self.account_id,
                    'bot_id': int(self.bot_id)
                },
                'confidence': confidence,
                'lang': lang
            }
            payload_segments.append(segment)


        return {
            'smalltalk': smalltalk,
            'status': True,
            'payload': payload_segments,
            'processed_text': prep3,
            'sentiment': {
                'is_badword_exists': badwords_count > 0,
                'badwords_count': badwords_count,
            }
        }

