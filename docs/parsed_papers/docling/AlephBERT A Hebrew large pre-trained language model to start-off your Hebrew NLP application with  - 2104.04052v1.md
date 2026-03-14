## AlephBERT: A Hebrew Large Pre-Trained Language Model to Start-off your Hebrew NLP Application With

Amit Seker, Elron Bandel, Dan Bareket, Idan Brusilovsky, Refael Shaked Greenfeld, Reut Tsarfaty

Bar-Ilan University, Computer Science Department, Ramat-Gan, Israel { aseker00,elronbandel,dbareket,shakedgreenfeld,brusli1,reut.tsarfaty } @gmail.com

## Abstract

Large Pre-trained Language Models (PLMs) have become ubiquitous in the development of language understanding technology and lie at the heart of many artificial intelligence advances. While advances reported for English using PLMs are unprecedented, reported advances using PLMs in Hebrew are few and far between. The problem is twofold. First, Hebrew resources available for training NLP models are not at the same order of magnitude as their English counterparts. Second, there are no accepted tasks and benchmarks to evaluate the progress of Hebrew PLMs on. In this work we aim to remedy both aspects. First, we present AlephBERT , a large pre-trained language model for Modern Hebrew, which is trained on larger vocabulary and a larger dataset than any Hebrew PLM before. Second, using AlephBERT we present new stateof-the-art results on multiple Hebrew tasks and benchmarks, including: Segmentation, Partof-Speech Tagging, full Morphological Tagging, Named-Entity Recognition and Sentiment Analysis. We make our AlephBERT model publicly available, providing a single point of entry for the development of Hebrew NLP applications.

## 1 Introduction

Contextualized word representations, provided by models such as BERT (Devlin et al., 2019) and RoBERTa (Liu et al., 2019), were shown in recent years to be critical for obtaining state-of-the-art performance on a wide range of Natural Language Processing (NLP) tasks - such as syntactic and semantic parsing, question answering, natural language inference, text summarization, natural language generation, and more. These contextualized word representations are obtained by pre-training a large language model on massive quantities of unlabeled data, aiming to maximize a simple yet effective objective of masked word prediction.

While advances reported for English using such models are unprecedented, in Hebrew previously reported results using BERT-based models are far from impressive. Specifically, the BERT-based Hebrew section of multilingual-BERT (Devlin et al., 2019) (henceforth, mBERT), did not provide a similar boost in performance to what is observed for the English section of mBERT. In fact, for several reported tasks, the mBERT model results are on a par with pre-neural models, or neural models based on non-contextialized embeddings (Tsarfaty et al., 2020; Klein and Tsarfaty, 2020). An additional Hebrew BERT-based model, HeBERT (Chriqui and Yahav, 2021), has been released, yet there is no reported evidence on performance improvements on key component of the Hebrew NLP pipeline which includes, at the very least: morphological segmentation, full morphological tagging, and full (token/morpheme-based) named entity recognition.

In this work we present AlephBERT , a Hebrew pre-trained language model, larger and more effective than any Hebrew PLM before. Using AlephBERT we show substantial improvements on all essential tasks in the Hebrew NLP pipeline, tasks tailored to fit a morphologically-rich language , including: Segmentation, Part-of-Speech Tagging, full morphological tagging, Named Entity Recognition and Sentiment Analysis . Since previous Hebrew NLP studies used varied corpora and annotation schemes, we confirm our results on all existing Hebrew benchmarks and variants. For morphology and POS tagging, we test on both the Hebrew section of the SPMRL shared task (Seddah et al., 2013), and the Hebrew UD corpus (Sadde et al., 2018). For Named Entity recognition, we test on both the corpus of Ben Mordecai and Elhadad (2005) and that of Bareket and Tsarfaty (2020). For sentiment analysis we test on the facebook corpus of Amram et al. (2018), as well as a newer (fixed) variant of this benchmark.

We make our pre-trained model publicly available 1 and additionally we deliver an online demo 2 allowing to qualitatively compare the maskprediction capacity of different PLMs available for Hebrew. In the near future we will release the complete AlephBERT -geared pipeline we developed, containing the aforementioned tasks, as means for evaluating and comparing future Hebrew PLMs, and as a starting point for developing further downstream applications and tasks. We also plan to showcase AlephBERT 's capacities on downstream language understanding tasks such as: Information Extraction, Text Summarization, Reading Comprehension, and more. As future research, we are pursuing a plan to investigate the effect of different word decomposition algorithms and input representation variants on the different tasks in the Pipeline.

## 2 The Challenge

This paper presents a case study for PLM development for a morphologically-rich and resource-poor language. Specifically, we address Modern Hebrew, a Semitic, morphologically-rich language, that is long known to be notoriously hard to parse.

The challenges posed to automatically processing Hebrew texts and obtaining good accuracies on downstream tasks stem from (at least) two main factors. The first is the internal-complexity of wordtokens, resulting from the rich morphology, complex orthography, and lack of diacritization in Hebrew written texts. Space-delimited tokens have non-transparent decomposition and are highly ambiguous, making even the simplest of the tasks in the pipeline very challenging (Tsarfaty et al., 2019). The second factor is the fact that Modern Hebrew, with only a few dozens of millions of native speakers, is often studied in resource-scarce settings.

The resource-scarce setting is problematic for PLM development in at least two ways. First, there are insufficient amounts of free unlabeled text for pre-training. To wit, the Hebrew Wikipedia that was the source for training multilingual BERT is of orders of magnitude smaller than the English Wikipedia (See Table 1 ). 3 Secondly, there are no large-scale open-access commonly accepted benchmarks for fine-tuning and/or evaluating the performance of Hebrew PLMs on NLP/NLU downstream tasks.

1 huggingface.co/onlplab/alephbert-base 2 nlp.biu.ac.il/˜elronbandel/alephbert/

3 Of course, ample Hebrew data does exist online, but most of it is closed due to copy-right issues and paywalls.

Table 1: Corpora Size Comparison: High-resource (and Medium-resourced) languages vs. Hebrew.

| Language   | Oscar Size   | Wikipedia Articles   |
|------------|--------------|----------------------|
| English    | 2.3T         | 6,282,774            |
| Russian    | 1.2T         | 1,713,164            |
| Chinese    | 508G         | 1,188,715            |
| French     | 282G         | 2,316,002            |
| Arabic     | 82G          | 1,109,879            |
| Hebrew     | 20G          | 292,201              |

Table 2: Data Statistics for AlephBERT's training sets.

| Corpus          | File Size   | Sentences   | Words   |
|-----------------|-------------|-------------|---------|
| Oscar (deduped) | 9.8GB       | 20.9M       | 1,043M  |
| Twitter         | 6.9GB       | 71.5M       | 774M    |
| Wikipedia       | 1.1GB       | 6.3M        | 127M    |
| Total           | 17.9GB      | 98.7M       | 1.9B    |

Previous studies on various tasks on Hebrew data do exist, each relying on disparate data sources, with varied evaluation metrics and annotation schemes even for the same task. To investigate Hebrew PLMs and probe their ability to capture linguistic structure, we introduce and evaluate Hebrew PLMs on the full set of tasks, sentence-based , token-based and morpheme-based tasks, including specific task variants and evaluation metrics.

## 3 The Model

Data. The PLM nicknamed AlephBERT is trained on a larger dataset and a larger vocabulary than any Hebrew BERT instantiation before. Data statistics are provided in Table 2. Specifically, we employ the following datasets for pre-training:

- Oscar: A deduplicated Hebrew portion of the OSCAR corpus, which is 'extracted from Common Crawl via language classification, filtering and cleaning' (Ortiz Su´ arez et al., 2020).
- Twitter: Texts of Hebrew tweets collected between 2014-09-28 and 2018-03-07. We slightly cleaned up the texts by removing retweet signals ' RT: ', user mentions (e.g. ' @username '), and URLs.
- Wikipedia: The texts in all of Hebrew Wikipedia, 4 extracted using Attardi (2015) This corpus is available on our github. 5

4 Dump: hewiki-20200201-pages-articles.xml.bz2

5 https://github.com/OnlpLab/AlephBERT/ blob/main/data/wikipedia/

One of the most important factors driving the success of PLMs in other languages is the availability of enormous amounts of text to learn from. The Hebrew portions of Oscar and Wikipedia provides us with a training set size which is an order of magnitude smaller compared with resource-savvy languages, as shown in Table 1. In order to build a strong PLM we need a considerable boost in the amount of text that the PLM can learn from, which in our case comes form massive amounts of tweets added to the training set. The textual utterances provided by the Twitter sample API tend be short and diverge from valid syntax and canonical language use for the most part. And while the free form language expressed in tweets might differ significantly from the text found in Oscar and Wikipedia, the sheer volume of tweets helps us close the resource gap substantially. Combining all resources together we have tweets comprising the lion's share of sentences in our dataset (72%).

Training We used the Transformers training framework of Huggingface (Wolf et al., 2020) and trained two different models - a small model with 6 hidden layers learned from the Oscar portion of our dataset, and a base model with 12 hidden layers which was trained on the entire dataset. The processing units used in both the small and base AlephBERT models are wordpieces generated by training BERT tokenizers over the respective datasets with a vocabulary size of 52K in both cases.

Traditionally, BERT models are optimized with an objective function optimized using both masked token prediction as well as next sentence prediction losses. Following the work on RoBERTa (Liu et al., 2019) we employ masked-token prediction loss only in our training objective. Incidentally our choice of dataset also forces us to ignore next sentence prediction because a large portion of our data comprises of tweets which are unrelated and independent of each other (we did not attempt to reconstruct the discourse threads of retweets and replies). For more training details see the Appendix.

## 4 Experiments

Goal We set out to pre-train Hebrew PLMs and evaluate them empirically on a range of Hebrew NLP tasks. We evaluated the two AlephBERT variants (small and base) on the different tasks, in order to empirically gauge the effect of model size and data size on the quality of the language model. In addition, we compared the performance of our models to existing Hebrew BERT-based instantiations (mBERT (Devlin et al., 2019) and HeBERT (Chriqui and Yahav, 2021)). We evaluated the PLMs on all key tasks of the Hebrew NLP pipeline.

Benchmarks We evaluate our BERT-based models on various Hebrew NLP tasks using the following benchmarks:

## · Word Segmentation, Part-of-Speech Tagging, Full Morphological Tagging:

- -The Hebrew Section of the SPMRL Task (Seddah et al., 2013)
- -The Hebrew Section of the UD 6 treebanks collection (Sadde et al., 2018)

## · Named Entity Recognition:

- -Token-based NER evaluation based on the corpus of Ben-Mordecai and Elhadad (Ben Mordecai and Elhadad, 2005)
- -Token-based and Morpheme-based NER evaluation based on the Named Entities and MOrphology (henceforth NEMO) corpus (Bareket and Tsarfaty, 2020)

## · Sentiment Analysis:

- -Sentiment Analysis evaluation based on the corpus of Amram et al. (2018).
- -Since the aforementioned corpus is reported to be leaking (shared material between test and train), we provide a cleaned up version and evaluate on the updated split.

## 5 Tasks and Modeling Strategies

A key question when assessing BERT-based PLM performance for Hebrew concerns how to develop models for the different levels of granularity. Here we briefly sketch our modeling strategies, starting with the easiest (classification) tasks and continuing to the more challenging setups, involving the use of PLMs to predict the tokens' internal structures.

## 5.1 Sentence-Based Modeling

Sentiment Analysis The first task we report on is a simple sentence classification task, classifying the sentiment of a given sentence to one of three values: negative, positive, neutral. We trained and evaluated BERT-based sentence classification on

6 https://universaldependencies.org

two variants of the Hebrew Sentiment dataset of Amram et al. (2018).

The first variant is the original sentiment dataset of Amram et al. (2018) with an additional split to create a dev set (the original paper had only train and test split, and the test set remains the same). The dev set contains 10% of the train data which leaves us with a split of 70-10-20.

Unfortunately, the original dataset of Amram et al. had a significant data leakage between the splits, with duplicates in the data samples. After removing the duplicates out of the original 12,804 sentences, we are left with a dataset of size 8,465. 7

We fine-tuned all the models for 15 epochs with the default Huggingface (Wolf et al., 2020) parameters on 5 different seeds. We report per-comment accuracy, and take the mean of these 5 runs.

## 5.2 Token-Based Modeling

Named Entity Recognition For the NER task, we initially assume a token-based sequence labeling model. The input comprises of the sequence of tokens in the sentence, and the output contains BIOES tags indicating entity spans. The tokenbased model is a simple fine-tuned model using the Transformer's token-classification script of Wolf et al. (2020).

We evaluate this model on two corpora. The first is the corpus by Ben Mordecai and Elhadad (2005), henceforth, the BMC corpus. The BMC corpus annotates entities at Token-level. This means that a Hebrew token containing both a preposition and an entity mention will not deliver the entity-mention boundaries. The BMC contains 3294 sentences and 4600 entities, and has seven different entity categories (DATE, LOC, MONEY, ORG, PER, PERCENT, TIME). To remain compatible with the original work we train and test the models on the 3 different splits as in Bareket and Tsarfaty (2020). 8 For the BMC corpus we report token-based F1 scores on the detected entity mentions.

The second corpus is an extension of the SPMRL dataset with Named Entities annotation, also marked by BIOSE tags, respecting the precise (token-internal) morphological boundaries of NEs (henceforth, NEMO, standing for Named Entities and MOrphology) (Bareket and Tsarfaty, 2020). This corpus provides both a token-based and a

7 https://github.com/OnlpLab/ Hebrew-Sentiment-Data

8 https://github.com/OnlpLab/ HebrewResources/tree/master/BMCNER

morpheme-based annotation of the entities, where the latter contains the accurate (token-internal) entity boundaries. The NEMO corpus has nine categories (ANG, DUC, EVE, FAC, GPE, LOC, ORG, PER, WOA). It contains 6220 sentences and 7713 entities, and we used the standard SPMRL TrainDev-Test, as in Bareket and Tsarfaty (2020)

The models were trained over 15 epochs and no hyper parameter tuning. For the BMC we used 3 different seeds for each split set, leading to overall nine different training rounds, and for the NEMO set we used the average mean of five different seeds. For both benchmarks we report token-based F1 scores on the detected entity mentions.

## 5.3 Morpheme-Based Modeling

Modern Hebrew is a Semitic language with rich morphology and complex orthography. As a result, the basic processing units in the language are typically smaller than a given token's span. To probe AlephBERT's capacity to accurately predict such token-internal linguistic structure, we test our models on four tasks that require knowledge of the internal morphology of the raw tokens:

## · Segmentation

Input: A Hebrew sentence containing raw space-delimited tokens

Output: A sequence of morphological segments representing basic processing units. 9

## · Part-of-Speech Tagging

Input: A Hebrew sentence containing raw space-delimited tokens

Output: Segmentation of the tokens to basic processing units as above, where each segment is tagged with its single disambiguated part-of-speech tag.

## · Morphological Tagging

Input: A Hebrew sentence containing raw space-delimited tokens

Output: Segmentation of the tokens to basic processing units as above, where each segment is tagged with a single POS tag and a set of morphological features. 10

9 These units comply with the 2-level representation of tokens defined by UD, where each basic unit corresponds to a single POS tag. https://universaldependencies. org/u/overview/tokenization.html

10

Equivalent to the AllTags evaluation metric defined in the CoNLL18 shared task. https: //universaldependencies.org/conll18/ results-alltags.html

## · Morpheme-Based NER

Input: A Hebrew sentence containing raw space-delimited tokens

Output: Segmentation of the tokens to basic processing as above where segment is tagged with a BIOSE tags indicating entity spans, along with the entity-type label.

An illustration of these tasks is given in Table 3.

As opposed to fine-tuning the PLM model parameters, as done in sentence-based and tokenbased classification tasks, segmented morphemes are not readily available in the BERT representation. In order to provide proper segmentation and labeling for the four aforementioned tasks we developed a model designated to produce the morphological segments of each token in context.

The morphological segmentation model which we designed is composed of a PLM responsible for transforming input tokens into contextualized embedded vectors, which we then feed into a charbased seq2seq module that extracts the output segments. The seq2seq module is composed of an encoder implemented as a simple char-based BiLSTM, and a decoder implemented as a char-based LSTM generating the output character symbols, or a space symbol signalling the end of a morphological segment. We train the model for 15 epochs, optimizing next-character prediction loss function.

For the other tasks, involving both segmentation and labeling we deploy an MTL (multi-task learning) setup. That is, when generating an endof-segment symbol, the model then predicts task labels which can be one or more of the following: POS-tag, NER-tag, morphological features. In order to guide the training to learn we optimize the combined segmentation and label prediction loss values. Currently we simply add together the loss values, but we note that as a future improvement it is likely that assigning different weights to the different loss values could prove to be beneficial. All of these morphological labeling models are trained for 15 epochs and evaluated on both the UD (Sadde et al., 2018) and SPMRL data (Seddah et al., 2013).

In addition, we design another setup for running the various morphological labeling tasks in which we first segment the text (using the abovementioned segmentation model) and then perform fine-tuning with a token classification attention head directly applied to the PLM (similar to the way we fine-tune the PLM for the token-based NER task described in the previous section). In this pipeline setup we utilize the PLM twice; as part of the segmentation model to generate segments, which we then feed directly into the PLM (augmented with a token classification head) which is fine-tuned for the specific labeling task. We acknowledge the fact that we are fine-tuning the PLM using morphological segments even though it was originally pre-trained without any knowledge of sub-token units. But, as we shall see shortly, this seemingly unintuitive strategy performs surprisingly well.

## 6 Results

Sentence-Based Tasks The Sentiment analysis experimental results are provided in Table 5. As can be seen, all BERT-based models substantially outperform the original CNN Baseline reported by Amram et al. (2018). Interestingly, both AlephBERT-small and AlephBERT-base outperform all BERT-based variants, with BERT-base setting new SOTA results on the new (fixed) dataset.

Token-Based Tasks For our two NER benchmarks, we report the NER F1 scores on the tokenbased fine-tuned model in Table 4.

Here, although we see noticeable improvements for the mBERT and HeBert variants over the current SOTA, the most significant increase is in the AlephBERT-base model. We also see a substantial difference between the AlephBERT-small and AlephBERT-base models, with the latter providing a new SOTA results on these both data sets. Crucially, this holds for the token-based evaluation metrics (as defined in Bareket and Tsarfaty (2020)).

Morpheme-Based Tasks As a particular novelty of this work, we report BERT-based results on subtoken (segment-level) information. Specifically, we evaluate segmentation F1, POS F1, Morphological Features F1 and morphem-base NER F1, compared against the disambiguated labeled segments. In all cases we use raw space-delimited tokens as input, letting the BERT-based models perform both the segmentation and labeling.

Table 6 presents the segmentation, POS tags, and morphological tags F1 for the SPMRL dataset, all evaluated at the granularity of morphological segments. We report the aligned multiset F1 Scores as in previous work on Hebrew (More et al., 2019).

We see that segmentation results for all BERTbased models are similar, and they are already at

/endofword

| Raw input              | /endofword/nunfinal הלב לבית   | /endofword/nunfinal הלב לבית   | /endofword/nunfinal הלב לבית   | /endofword/nunfinal הלב לבית   | /endofword/nunfinal הלב לבית   |
|------------------------|--------------------------------|--------------------------------|--------------------------------|--------------------------------|--------------------------------|
| Space-delimited tokens | /endofword/nunfinal הלב        | /endofword/nunfinal הלב        | לבית                           | לבית                           | לבית                           |
| Segmentation           | /endofword/nunfinal לב         | ה                              | בית                            | ה                              | ל                              |
| POS                    | ADJ                            | DET                            | NOUN                           | DET                            | ADP                            |
| Morphology             | Gender=Masc &#124; Number=Sing | PronType=Art                   | Gender=Masc &#124; Number=Sing | PronType=Art                   | -                              |
| Token-level NER        | E-ORG                          | E-ORG                          | B-ORG                          | B-ORG                          | B-ORG                          |
| Morpheme-level NER     | E-ORG                          | I-ORG                          | I-ORG                          | B-ORG                          | O                              |

/endofword

/endofword

/endofword

Table 3: Illustration of Evaluated Token and Morpheme-Based Downstream Tasks. The input is the two-word input phrase ' /endofword/nunfinal לבית הלב ' ( to the White House ). Sequence and Hebrew text goes from right to left.

|                 |   NEMO |   BMC |
|-----------------|--------|-------|
| Previous SOTA   |  77.75 | 85.22 |
| mBERT           |  79.07 | 87.77 |
| HeBERT          |  81.48 | 89.41 |
| AlephBERT-small |  78.69 | 89.07 |
| AlephBERT-base  |  84.91 | 91.12 |

Table 4: Token-Based NER Results on the NEMO and the Ben-Mordecai Corpora. Previous SOTA on both corpora has been reported by the NEMO models of Bareket and Tsarfaty (2020).

the higher range of 97-98 F1 scores, which are hard to improve further. 11 For POS tagging and morphological features, all BERT-based models significantly outperform the previous SOTA provided by (Seker and Tsarfaty, 2020) (referred to as PtrNet) for POS tags and (More et al., 2019) (referred to as YAP) for morphological features. With respect to all BERT-based variants, we see an improvement for AlephBERT on all other alternatives, but on a small scale. That said, we do notice a repeating trend that places AlephBERT-base as the best model for all of our morphological tasks, indicating that the improvement provided by the depth of the model and a larger dataset does also improve the ability to capture token-internal structure.

These trends are replicated on the UD Hebrew corpus, for two different evaluation metrics - the Aligned MultiSet F1 Scores as in previous work on Hebrew (More et al., 2019), (Seker and Tsarfaty, 2020), and the Aligned F1 scores metrics in the UD shared task (Zeman et al., 2018) - as reported in Tables 7 and 8 respectively. AlephBERT obtains the best results for all tasks, even if not by a large margin.

Morpheme-Based NER Earlier in this section we considered NER as a token-based task that sim-

11 Some of these errors are due to annotation errors, or truly ambiguous cases.

ply requires fine-tuning on the token labels. However, this setup is not accurate enough and less useful for downstream tasks, since the exact entity boundaries are often token internal (Bareket and Tsarfaty, 2020). We hence also report here morpheme-based NER evaluation, respecting the exact boundaries of the Entity mentions. To obtain morpheme-based labeled-span of Named Entities as discussed above we could either employ a pipeline, first predicting segmentation and then applying a fine tuned labeling model directly on the segments , or we can use the MTL model and predict NER labels while performing the segmentation.

Table 9 presents segmentation and NER results for three different scenarios: (i) a pipeline assuming gold segmentation (ii) a pipeline assuming the best predicted segmentation (as predicted above) (iii) obtaining the segmentation and NER labels jointly in the MTL setup.

As our results indicate, AlephBERT-base consistently scores highest in both pipeline (oracle and predicted) and multi-task setups. Looking at the Pipeline-Predicted scores, there is a clear correlation between a higher segmentation quality of a PLM and its ability to produce better NER results. Moreover, the differences in NER scores between the models are considerable (unlike the subtle differences in segmentation, POS and morphological features scores) and draw our attention to the relationship between the size of the PLM, the size of the pre-training data and the quality of the final NER models. Specifically, HeBERT and AlephBERT-small were pre-trained with similar datasets - HeBERT with Oscar and Wikipedia, AlephBERT-small with Oscar only (the Wikipedia portion is order of magnitude smaller compared with Oscar) and comparable vocabulary sizes (heBERT with 30K and AlephBERT-small with 52K). However we notice that HeBERT, with its 12 hidden layers, performs significantly better

/endofword

|                 |   Old(leak) token |   Old(leak) morph | New(fixed) token   | New(fixed) morph   |
|-----------------|-------------------|-------------------|--------------------|--------------------|
| Previous SOTA   |             89.2  |             87.5  | NA                 | NA                 |
| mBERT           |             92.12 |             92.18 | 84.21              | 85.58              |
| HeBERT          |             92.48 |             92.27 | 87.13              | 86.88              |
| AlephBERT-small |             93.15 |             92.7  | 88.3               | 87.38              |
| AlephBERT-base  |             91.63 |             92.01 | 89.02              | 88.71              |

Table 5: Sentiment Analysis Scores on the Facebook Corpus. Previous SOTA is reported by Amram et al. (2018).

|                       | Segmentation F1   |   POS F1 |   Morphological Features F1 |
|-----------------------|-------------------|----------|-----------------------------|
| Previous SOTA         | NA                |    90.49 |                       85.98 |
| mBERT-morph           | 97.36             |    93.37 |                       89.36 |
| HeBERT-morph          | 97.97             |    94.61 |                       90.93 |
| AlephBERT-small-morph | 97.71             |    94.11 |                       90.56 |
| AlephBERT-base-morph  | 98.10             |    94.9  |                       91.41 |

Table 6: Morpheme-Based Aligned MultiSet (mset) Results on the SPMRL Corpus. Previous SOTA is as reported by (Seker and Tsarfaty, 2020) (POS) and (More et al., 2019) (morphological features)

Table 7: Morpheme-Based Aligned MultiSet (mset) Results on the UD Corpus. Previous SOTA is as reprted by (Seker and Tsarfaty, 2020) (POS)

|                       | Segmentation F1   |   POS F1 | Morphological Features F1   |
|-----------------------|-------------------|----------|-----------------------------|
| Previous SOTA         | NA                |    94.02 | NA                          |
| mBERT-morph           | 97.70             |    94.76 | 90.98                       |
| HeBERT-morph          | 98.05             |    96.07 | 92.53                       |
| AlephBERT-small-morph | 97.86             |    95.58 | 92.06                       |
| AlephBERT-base-morph  | 98.20             |    96.2  | 93.05                       |

Table 8: Morpheme-Based Aligned (CoNLL shared task) Results on the UD Corpus. Previous SOTA is as reported by Minh Van Nguyen and Nguyen (2021)

|                       |   Segmentation F1 |   POS F1 |   Morphological Features F1 |
|-----------------------|-------------------|----------|-----------------------------|
| Previous SOTA         |             96.03 |    93.75 |                       91.24 |
| mBERT-morph           |             97.17 |    94.27 |                       90.51 |
| HeBERT-morph          |             97.54 |    95.6  |                       92.15 |
| AlephBERT-small-morph |             97.31 |    95.13 |                       91.65 |
| AlephBERT-base-morph  |             97.7  |    95.84 |                       92.71 |

Table 9: Morpheme-Based NER Evaluation on the NEMO Corpus. Previous SOTA is as reported by Bareket and Tsarfaty (2020) for the Pipeline (Oracle), Pipeline (Predicted) and a Hybrid (almost-joint) Scenarios, respectively.

| Architecture Segmentation   | Pipeline (Oracle)   | Pipeline (Oracle)   | Pipeline (Predicted)   | Pipeline (Predicted)   | MultiTask   | MultiTask   |
|-----------------------------|---------------------|---------------------|------------------------|------------------------|-------------|-------------|
| Scores (aligned mset F1)    | Seg                 | NER                 | Seg                    | NER                    | Seg         | NER         |
| Previous SOTA (NEMO)        | 100.00              | 79.10               | 95.15                  | 69.52                  | 97.05       | 77.11       |
| mBERT                       | 100.00              | 77.92               | 97.68                  | 72.72                  | 97.24       | 72.97       |
| HeBERT                      | 100.00              | 82                  | 98.15                  | 76.74                  | 97.92       | 74.86       |
| AlephBERT-small             | 100.00              | 79.44               | 97.78                  | 73.08                  | 97.74       | 72.46       |
| AlephBERT-base              | 100.00              | 83.94               | 98.29                  | 80.15                  | 98.19       | 79.15       |

compared to AlephBERT-small which is composed of only 6 hidden layers. It thus appears that semantic information is learned in those deeper layers which helps in both learning to discriminate entities and improve the overall morphological segmentation capacity.

In addition, comparing HeBERT to AlephBERTbase we point to the fact that they are both modeled with the same 12 hidden layer architecture, the only differences between them are in the size of their vocabularies (30K vs 52K respectively) and the size of the training data (Oscar-Wikipedia vs Oscar-Wikipedia-Tweets). The improvements exhibited by AlephBERT-base, compared to HeBERT, suggests that it is a result of the large amounts of training data and larger vocabulary available in our setup. By exposing AlephBERT-base to an amount of text which order of magnitude larger we increased its NER capacity.

Finally, our NER experiments suggest that a pipeline composed of our near-to-perfect morphological segmentation model followed by AlephBERT-base augmented with a token classification head is the best strategy for generating morphologically-aware NER labels.

## 7 Qualitative Assessment

To allow for qualitative assessment of the PLMs, we deliver an online demo where one can compare the masked-word prediction capacities of the different models, and get the impression of the strengths and weaknesses. Our demo, available at https:// nlp.biu.ac.il/˜elronbandel/alephbert/ , offers friendly graphical interface that allows one to mask an item in a running Hebrew text and obtain the top-N list of alternatives predicted by each of the models. The demo allows to explore the predictions of our models both at token level and at sub-token level, masking individual word-pieces.

Note that the AlephBERT family of models is still under development, and we will add new model variants as we proceed. Stay tuned!

## 8 Conclusion

Modern Hebrew, a morphologically rich and resource-scarce language, has for long suffered from a gap in the resources available for NLP applications, and lower level of empirical results than observed in other, resource-rich languages. This work provides the first step in remedying the situation, by making available a large Hebrew PLM, nicknamed AlephBERT, with larger vocabulary and larger training set than any Hebrew PLM before, and with clear evidence as to its empirical advantages. Our AlephBERT-base model obtains stateof-the- art results on the tasks of segmentation, Part of Speech Tagging, Named Entity Recognition, and Sentiment Analysis. We outperform both general multilingual PLMs (mBERT) as well as language specific instantiations (HeBERT). More importantly, using the new AlephBERT models we are now gaining similar benefits as achieved in high resource languages from PLMs.

## 9 Acknowledgements

We are enormously grateful to Roee Aharoni from Google and Yoav Goldberg from Bar Ilan University for technical advise during the project. No less importantly, we are indebted to Roee Aharoni for coining the brand-name AlephBERT . The research reported in this paper is funded by an individual grant by the Israel Science Foundation (ISF grant # 1739/26) and a starting grant by the European Research Council (ERC-StG Grant # 677352), for which we are grateful.

Table 10: Huggingface BERT Configurations Comparison. †Only 2450 vocabulary entries contain Hebrew letters

|                         | AlephBERT-base   | AlephBERT-small   | HeBERT   | mBERT-cased   |
|-------------------------|------------------|-------------------|----------|---------------|
| max position embeddings | 512              | 512               | 512      | 512           |
| num attention heads     | 12               | 12                | 12       | 12            |
| num hidden layers       | 12               | 6                 | 12       | 12            |
| vocab size              | 52K              | 52K               | 30K      | 120K†         |

## References

Adam Amram, Anat Ben-David, and Reut Tsarfaty. 2018. Representations and architectures in neural sentiment analysis for morphologically rich languages: A case study from modern hebrew. In Proceedings of the 27th International Conference on Computational Linguistics, COLING 2018, Santa Fe, New Mexico, USA, August 20-26, 2018 , pages 2242-2252.

Giusepppe Attardi. 2015. Wikiextractor. https:// github.com/attardi/wikiextractor .

Dan Bareket and Reut Tsarfaty. 2020. Neural modeling for named entities and morphology (nemoˆ2). CoRR , abs/2007.15620.

Naama Ben Mordecai and Michael Elhadad. 2005. Hebrew named entity recognition.

Avihay Chriqui and Inbal Yahav. 2021. Hebert -&amp; hebemo: a hebrew bert model and a tool for polarity analysis and emotion recognition.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) , pages 4171-4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Stav Klein and Reut Tsarfaty. 2020. Getting the ##life out of living: How adequate are word-pieces for modelling complex morphology? In Proceedings of the 17th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology, SIGMORPHON 2020, Online, July 10, 2020 , pages 204-209.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. RoBERTa: A Robustly Optimized BERT Pretraining Approach.

Amir Pouran Ben Veyseh Minh Van Nguyen, Viet Lai and Thien Huu Nguyen. 2021. Trankit: A lightweight transformer-based toolkit for multilingual natural language processing. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations .

Amir More, Amit Seker, Victoria Basmova, and Reut Tsarfaty. 2019. Joint transition-based models for morpho-syntactic parsing: Parsing strategies for mrls and a case study from modern hebrew. Trans. Assoc. Comput. Linguistics , 7:33-48.

Pedro Javier Ortiz Su´ arez, Laurent Romary, and Benoˆ ıt Sagot. 2020. A monolingual approach to contextualized word embeddings for mid-resource languages. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics , pages 1703-1714, Online. Association for Computational Linguistics.

Shoval Sadde, Amit Seker, and Reut Tsarfaty. 2018. The hebrew universal dependency treebank: Past present and future. In Proceedings of the Second Workshop on Universal Dependencies, UDW@EMNLP2018, Brussels, Belgium, November 1, 2018 , pages 133-143.

Djam´ e Seddah, Reut Tsarfaty, Sandra K¨ ubler, Marie Candito, Jinho D. Choi, Rich´ ard Farkas, Jennifer Foster, Iakes Goenaga, Koldo Gojenola Galletebeitia, Yoav Goldberg, Spence Green, Nizar Habash, Marco Kuhlmann, Wolfgang Maier, Joakim Nivre, Adam Przepi´ orkowski, Ryan Roth, Wolfgang Seeker, Yannick Versley, Veronika Vincze, Marcin Wolinski, Alina Wr´ oblewska, and ´ Eric Villemonte de la Clergerie. 2013. Overview of the SPMRL 2013 shared task: A cross-framework evaluation of parsing morphologically rich languages. In Proceedings of the Fourth Workshop on Statistical Parsing of Morphologically-Rich Languages, SPMRL@EMNLP 2013, Seattle, Washington, USA, October 18, 2013 , pages 146-182.

Amit Seker and Reut Tsarfaty. 2020. A pointer network architecture for joint morphological segmentation and tagging. In Findings of the Association for Computational Linguistics: EMNLP 2020 , pages 4368-4378, Online. Association for Computational Linguistics.

Reut Tsarfaty, Dan Bareket, Stav Klein, and Amit Seker. 2020. From SPMRL to NMRL: what did we learn (and unlearn) in a decade of parsing morphologically-rich languages (mrls)? In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020 , pages 7396-7408.

- Reut Tsarfaty, Shoval Sadde, Stav Klein, and Amit Seker. 2019. What's wrong with hebrew nlp? and how to make it right. In Proceedings of the

2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019 - System Demonstrations , pages 259264.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R´ emi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations , pages 38-45, Online. Association for Computational Linguistics.

Daniel Zeman, Jan Hajiˇ c, Martin Popel, Martin Potthast, Milan Straka, Filip Ginter, Joakim Nivre, and Slav Petrov. 2018. CoNLL 2018 shared task: Multilingual parsing from raw text to Universal Dependencies. In Proceedings of the CoNLL 2018 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies , pages 1-21, Brussels, Belgium. Association for Computational Linguistics.

## A AlephBERT Training Details

For reference and to make our work reproducible we specify here the main steps taken and parameters used during training of AlephBERT. We utilized the Huggingface Transformers framework with most of the default training parameter values. Table-10 lists all of the training parameters that we have manually specified in our code. We also list the values used by the other models.

Training our AlephBERT-base model using the entire dataset proved to be technically challenging due to the model size and data size. With the naive approach training the entire dataset without splitting it into chunks did not utilize the full processing capacity of the GPUs and would have taken several weeks to complete. To overcome this issue we followed the advice to split the dataset into chunks based on the number of tokens in a sentence. The first chunk consisted of 70M senetences with 32 or less tokens. By limiting the maximum number tokens we consequently limit the size of the training matrices used by this chunk which consequently allowed for significantly increasing the batch size which resulted in dramatically shorter training time - these 70M sentences took only 2.5 days to complete 5 epochs. The second chunk consisted of sentences having between 32 and 64 tokens, the third chunk between 64 and 128 and the final last chunk all sentences with more than 128 tokens. We trained each chunk for 5 epochs with setting the learning rate to 1e-4. Once we went over the entire dataset we trained for another 5 epochs with a learning rate set to 5e-5 for a total of 10 epochs. We trained our base model over the entire dataset for 10 epochs on a NVidia DGX server with 8 V100 GPUs which took 8 days. The small model was trained over 10 epochs using 4 GTX 2080ti GPUs for 5 days in total.