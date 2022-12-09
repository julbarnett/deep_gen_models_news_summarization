# Summarizing the News

A repo to hold the work for a news summarization generative deep models tool. 

* [Mowafak Allaham](https://mallaham.github.io/) - MowafakAllaham2021 [@] u [dot] northwestern [dot] edu
* [Julia Barnett](https://www.juliabbarnett.com/) - JuliaBarnett [@] u [dot] northwestern [dot] edu

## Table of Contents
1. [Research Objectives](#research-objectives)
2. [Intro to Computational Journalism](#intro-to-computational-journalism)
3. [Text Summarization Tools and Methods](#text-summarization-tools-and-methods)
4. [Text-To-Text Transfer Transformer (T5 Model)](#text-to-text-transfer-transformer-model)
5. [Fine Tuning the Model and Data](#fine-tuning-the-model-and-data)
6. [Summarizing Low Credible Sources](#summarizing-low-credible-sources)
7. [Qualitative Analysis](#qualitative-analysis)
8. [Future Work](#future-work)
9. [Contact Us](#contact-us)
### Research Objectives

Our research ojectives are two-fold:

1. Develop a proof-of-concept tool using a deep generative model that can help journalists summarize news articles from a range of news sources and across news categories
2. Highlight potential risks that may occur from these summarizations—especially in regards to conspiracy claims in low credible news sources

###

### Intro to Computational Journalism

Newsrooms around the world have been using more and more computational methods in many areas such as:
* Information gathering
* Production
* Sensemaking
* Distribution
* Audience consumption behavior
* Computational news discovery

The average amount of time a user spends on a news sites has been decreasing yearly–culminating at a staggering estimation of the average user spending less than 2 minutes on news sites in 2020 [4]. Half of U.S adults get their news media from social media [8]. Among these adults, around 40% of Facebook users and 50% of Twitter users regularly get the news on these social media platforms. Accordingly, it is essential to offer the public readable and credible news in order to share an unbiased lens to domestic policies, and international affairs.

One challenge to news curators is how to present news informa- tion to the readers in a way that keeps them engaged with relevant news. One approach is to offer the readers a daily briefing, similar to the morning newsletters by the New York Times, that provides bullet-point summary of daily news events and articles published online. One challenge to achieve this objective is the laborious effort that journalists put forth to sift through relevant news articles and summarize ones that are relevant to the readers.

We aim to solve this gap by summarizing news articles in a concise and comprehensive manner to aid both journalists and the average reader alike.
###

### Text Summarization Tools and Methods

We looked into using a variety of deep generative text summarization models, ultimately using T-5:

* **GPT-3** (OpenAI) 
  - *Drawbacks:* enormous and expensive to use
* Mini versions of GPT-3
  - **GPT-Neo**
  - **GPT-J**
  - *Drawbacks:* couldn’t get these models to work for text summarization. 
  - However, they do work for a general text generation task based on prompt.
* **PEGASUS**
  - *Drawbacks:* Did not yield interpretable summariers
* **T5** (***T**ext-**T**o-**T**ext **T**ransfer **T**ransformer*)
  - Our choice!
  - Most affordable and versatile model for text summarization
  - Papers we relied on heavily are [Raffel et. al, 2020](https://arxiv.org/pdf/1910.10683.pdf) and [Roberts, Raffel, and Shazeer, 2020](https://arxiv.org/pdf/2002.08910.pdf)
###

### Text-to-Text Transfer Transformer Model

![T5 Downstream Tasks](https://1.bp.blogspot.com/-o4oiOExxq1s/Xk26XPC3haI/AAAAAAAAFU8/NBlvOWB84L0PTYy9TzZBaLf6fwPGJTR0QCLcBGAsYHQ/s640/image3.gif)

Image source: Source: [AI Google Blog - Exploring Transfer Learning with T5](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)


**Motivation behind T-5 Model:**
* Provide a unified **text-to-text framework** (input and output are text)
* Combine successful **transfer learning techniques** in NLP models into a single model
* Train a language model on **“clean” data**: Colossal Clean Crawled Corpus (C4)

**T5 Model Architecture and Training**

* **Transformer-based** model which includes an encoder-decoder transformer
* The encoder-decoder transformer is inspired by transformers with **self attention architecture** (Vaswani et al., 2014)
* Removed Layer Norm bias
* Placed layer normalization outside residual path
* Used different positional embedding 
  - Variation of relative positional embedding
* Offers the following model variants:
  - Base (220 million parameters)
  - Large (770 million parameters)
  - 3B (3 Billion parameters)
  - 11B (11 Billion parameters)

![T5 Model Architecture](https://miro.medium.com/max/1400/1*iJcUH1F0TmCQE5p2wQt9og.png)

Image source: [Jay Alammar’s blog](http://jalammar.github.io/illustrated-transformer/)

**Training Overview**

* Models are trained on lines that ended in a terminal punctuation mark (. ? ! .”)
* Discarded:
  - Pages with < 5 sentences
  - Sentences with < 3 words
  - Pages containing any word on the [“List of Dirty, Naughty, Obscene or Otherwise Bad Words”](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/blob/master/en) (Shutterstock 2020)
  - Deduplicated the data set by discarding all but one of any three-sentence span occurring more than once in the data set

*An example downstream task: Trivia*

![Pre-training and fine-tuning](https://1.bp.blogspot.com/-89OY3FjN0N0/XlQl4PEYGsI/AAAAAAAAFW4/knj8HFuo48cUFlwCHuU5feQ7yxfsewcAwCLcBGAsYHQ/s640/image2.png)
![Trivia T5 Example](https://1.bp.blogspot.com/-SllNg6Q4DEE/Xk7ZRCtzXaI/AAAAAAAAFVY/PaaM-FEgyFIdSn7VeT_XhvG9PXQdSC3_wCLcBGAsYHQ/s640/t5-trivia-lrg.gif)

Images courtesy of [AI Google Blog - Exploring Transfer Learning with T5](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)

###

### Fine Tuning the Model and Data

We trained this model on the CNN and Daily Mail dataset ([See, Liu, and Manning 2017](https://arxiv.org/pdf/1704.04368.pdf)) in order to understand news article and summarization pairs. This dataset is comprised of 229,690 article and summary pairs in the training set and 57,423 pairs in the test set.

![Data Distribution](https://github.com/julbarnett/deep_gen_models_news_summarization/blob/main/data_distribution.png)

For fine-tuning the model:
* Prefixed all articles with “summarize:” prompt
* Max article length = 1024 (truncating enabled)
* Max summary length = 100 (truncation enabled)

Summarization generation:
* Min summary length = 50
* Max summary length = 100

![Selected Data to Train On >](https://github.com/julbarnett/deep_gen_models_news_summarization/blob/main/data_distribution_annotated.png)

Trained the model on 5 steps with 
* least cross entropy loss achieved: 0.07478
###

### Summarizing Low Credible Sources

In order to highlight some potential risks of news summarization, our test case was to summarize news articles from low credible sources that are known to have conspiracy claims on climate change ([full list of conspiracy claims in our repo](https://github.com/julbarnett/deep_gen_models_news_summarization/blob/main/claims_description.txt)), and qualitatively evaluated the summaries our model generated for these articles. 

We had access to a database of 185,080 articles and blog posts (published between 2017 and 2022) substantially discussing climate change, and prior from the Center for Communication & Public Policy directed by Dr. [Erik C. Nisbet](https://communication.northwestern.edu/faculty/erik-nisbet.html) work identified the presence of false or misleading climate change claims at the paragraph level of analysis (using [CARDS (Computer-assisted Recognition of CC Denial and Skepticism)](https://cardsclimate.com/). We generated summaries for 150 low credible articles.

Data details:
* Summary length = 50-100 words
* Max length to encode is 1024 characters (with truncation enabled)
* Max sequence length is 1024 
* The model caps on 512 and a batch size of 128 ~ 65K tokens in a single step
* Each token is assigned a unique ID to the input sequence

After generating the summaries, both authors went through 20 summaries to evaluate whether the claim was still present in the generated summary. To establish inner-rater reliability, we acheived a Cohen's Kappa of 0.78 and then one author then moved forward to evaluate 150 examples. 

###

### Qualitative Analysis

*Note: this data has yet to be published, so we are only including select examples to highlight our findings.*

**Main types of summaries:**
* Accurate summarization:
  - Perpetuated the claim (22% of summaries)
  - Removed the claim entirely (78% of summaries)
* Poor quality/not intelligible

**1. Perpetuating the claim**:

*1A. Original article*: argues both COVID and climate change were political hoaxes.

*1A. Generated summary*: global elites are pivoting on climate change. there will be other hysteries, but no real political push on the climate front ever again – just as war on terror did in 2004!


**2. Removing the claim**:

*2A. Original article*: endorsed Pruit’s false claims of climate change.

*2A. Generated summary*: environmentalists want epa inspector general to investigate whether pruitt violated internal policies. sierra club attorney: pruits comments contradict basic scientific findings of research on climate change, not contrary ones from studies - 'obviously false' former vice president al gore said comments were "independent and unfounded"

*2B. Original article*: claimed that hurricanes were slightly worse than normal, but not a result of climate change and would revert back to neutral soon.

*2B. Generated summary*: last year's atlantic hurricane season produced 10 storm surge events, including six major hurricane types. the 2018 forecast has three or four storm potential impacts on the u-pacific coast if they do hit the u us ; but that'll be less than last summer' average for this season compared to last august eve a few storm seasons ago dubbed "the worst of our history


**3. Poor quality/not intelligible**:

*3A. Generated summary*: a new study shows that plants are to blame for warming, not man's. the theory is that man’d produce more co2 to feed people and poison them - but it doesnt work this way if we increase carbon dioxide levels in our cars or on highway roads... it'll be madness.

You can tell the claim is there, but the summary doesn’t quite make sense.

###

### Future Work
* Analyze other types of content perpetuated or removed by summaries
  - Political bias
  - Other dangerous conspiracy theories
* Examine how the credibility of the news organization affects the summaries
* Train a model to classify whether the claims are present or not
* Extend to different data sources
* Add more user specificity to the summarization tool (“I want a summary about this part of the article”)

###

### Contact Us
We hope you've enjoyed this work! We are both pursuing PhDs in Technology and Social Behavior at Northwestern University. If you have any questions feel free to reach out to either of us.

* [Mowafak Allaham](https://mallaham.github.io/) - MowafakAllaham2021 [@] u [dot] northwestern [dot] edu
* [Julia Barnett](https://www.juliabbarnett.com/) - JuliaBarnett [@] u [dot] northwestern [dot] edu

###
