---
title: "Practical Machine Learning course project"
author: "Ray Bem"
date: "10/18/2020"
output:
  md_document:
    variant: gfm
    preserve_yaml: TRUE
---

## Synopsis

Exercise is great for people, though injury and enjoyment vary with
technique. The experience may be enhanced using technology to classify
technique, and a model-based classification of these techniques are the
focus of this analysis. In particular, the simple dumbbell lift provides
a rich array of data and is ripe for trying out – in this case three –
different models. [**Addtional github details (Rmd/md
versions)**](https://github.com/mountainray/pml_final).

### Exploratory Data Analysis

These are exercise-related data kindly offered by the **Human Activity
Recognition** website. From a sample of 8 indiviuals, numerous
continuous measurements were taken. The data came in the form of a comma
separated file with 160 features and 19622 observations. More details
are provided
[here](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har).

#### Character and Time variable reduction

Sample prints and diagnostics identified a small subset of variables
that we set aside as they describe elements not included in this effort.
For example, the `user_name` variable adds a user-specific dimension if
included in the models, we are trying to build a dumb classifier,
independent of user.

The usefulness of the time variables was difficult to assess, the
[original
paper](http://web.archive.org/web/20161224072740/http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf)
goes into more detail on how these `new_window = yes` records are
calculated (e.g., sliding 2.5s intervals). The table below summarizes.

| varname                 | varindex | description            | reason\_removed                                             |
| :---------------------- | -------: | :--------------------- | :---------------------------------------------------------- |
| X1                      |        1 | a sequential record id | obviously this would distort the model if left in           |
| user\_name              |        2 | study participant name | we want to predict classe for any person, therefore removed |
| raw\_timestamp\_part\_1 |        3 | raw timestamp part 1   | time disregarded in this analysis                           |
| raw\_timestamp\_part\_2 |        4 | raw timestamp part 2   |                                                             |
| cvtd\_timestamp         |        5 | converted timestamp    |                                                             |
| new\_window             |        6 | new\_window            | used to separate ‘yes’ data, n=406                          |
| num\_window             |        7 | num\_window            | an incremental window counter                               |

**Character and Time variables removed from consideration (n=7)**

With relatively high dimensionality, summary functions were used to
highlight characteristics. The first aspect examined was the
`new_window` variable. This was determined to be a summarization of
detail records tagged with “no”, and set aside early in the code. The
remainder of this analysis focuses on the “no” data, 19216 records.

#### Highly correlated variable reduction

The `cor` function was used to identify correlations above 80%, and
several (n=100) were found to be highly correlated.

As we know the nature of the measurements should include things that
will correlate (there are only three dimensions being measured on a
single body with a limited, intended motion), we leave some correlated
covariates in the model (i.e., we don’t exclude all correlated
variables). For example, an analysis of a dance movement would have more
complicated relationships and a more strict reduction might be fine,
here, the motion is so limited. Below is a summary.

| varname            | varindex | high\_corr\_no |
| :----------------- | -------: | :------------- |
| roll\_belt         |        8 | corr above 0.8 |
| pitch\_belt        |        9 | corr above 0.8 |
| yaw\_belt          |       10 | corr above 0.8 |
| total\_accel\_belt |       11 | corr above 0.8 |
| accel\_belt\_x     |       40 | corr above 0.8 |

**High Correlation variables removed from consideration (n=22, sample of
5 below)**

#### Low or near-zero variance variable reduction

In a similar fashion, the R function `nearzeroVariance` was used to
identify variables that would add little additional information to our
models. This process identified several variables with zero variance,
these were removed to avoid inflation of the model variance.

| varname                | varindex | freqRatio | percentUnique | zeroVar |
| :--------------------- | -------: | --------: | ------------: | :------ |
| kurtosis\_roll\_belt   |       12 |         0 |             0 | TRUE    |
| kurtosis\_picth\_belt  |       13 |         0 |             0 | TRUE    |
| kurtosis\_yaw\_belt    |       14 |         0 |             0 | TRUE    |
| skewness\_roll\_belt   |       15 |         0 |             0 | TRUE    |
| skewness\_roll\_belt.1 |       16 |         0 |             0 | TRUE    |

**Near-zero Variance variables removed from consideration (n=100, sample
of 5 below)**

#### Missing data

Finally, missing data were explored. At this point we have 1. only
detail “no” data 2. highly correlated and near-zero removed

We observe no missing data, a convenience as our model choices do not
require any imputation of data.

#### Transformations

While the main focus has been on dimension reduction, density plots (not
presented here) of remaining variables indicated heteroskedasticity. As
some were of course negative, the Yeo-Johnson transformation is applied
to `preProcess` the model builds, resulting in a better fit.

#### Final data

This yields a final modeling dataset having the following
characteristics…at this point the dimensionality has been thoughtfully
reduced, addressing covariance, near-zero variance, and missing data
realities.

Before we traded computational speed for accuracy, in an effort to see
the behavior of the model tuning features (for GBM). Some observations:

1.  Given significant dimension reduction, some hopeful results
2.  Adjusting GBM models allows flexibility to react to overfitting
3.  Similar variable importance (covered later)

The model building comes next, where data and cross validation are
expanded.

| classe | gyros\_belt\_x | gyros\_belt\_y | gyros\_belt\_z | magnet\_belt\_y | magnet\_belt\_z |
| :----- | -------------: | -------------: | -------------: | --------------: | --------------: |
| A      |           0.00 |           0.00 |         \-0.02 |             599 |           \-313 |
| A      |           0.02 |           0.00 |         \-0.02 |             608 |           \-311 |
| A      |           0.00 |           0.00 |         \-0.02 |             600 |           \-305 |
| A      |           0.02 |           0.00 |         \-0.03 |             604 |           \-310 |
| A      |           0.02 |           0.02 |         \-0.02 |             600 |           \-302 |

Sample Print of final dataset (raw)

| observations | variables |
| -----------: | --------: |
|        13451 |        31 |

Dimensions

| classe |    n |
| :----- | ---: |
| A      | 3821 |
| B      | 2601 |
| C      | 2365 |
| D      | 2182 |
| E      | 2482 |

Distribution of classe variable

## Building the classification model

A *modeling process* was built to more easily explore the data. The R
package `caret` was used to create data partitions separating training
data into a set to build models on, and a set to test. These test data
are used afterwards to assess our estimated out of sample error rate.

We *explored* a smaller (10/90) random sample of the training data for
the GBM model grid, and gave the faster treebag and ldabag models a more
realistic 70/30 training/validation split.

The final model comparisons are done with all settings identical (70/30
training/test, 5 times repeated resampling on 10 folds).

#### Gradient Boosting Model (GBM)

The Gradient Boosting Model was of interest – particularly in the spirit
of model tuning. A `caret` grid of variations of model tuning features
was built to generate estimates using a variety of the `gbm` tuning
parameters. This allowed for the exploration of 270 models in a
convenient way (less code, obviously).

Below are the results, where one can see the increase in model
performance as we adjust the required minimum in each branch of the tree
(these form the columns), as well as how much information is retained
for future branch development (shrinkage, forming the plot rows). The
plots themselves are of increasing accuracy, as we subject the model to
more boosting iterations. The model results are gathered and summarized,
and we have a set of 270 model objects in the end.

<img src="index_files/figure-gfm/plots-gbm-1.png" style="display: block; margin: auto;" />

One observation is the matrix above suggests requiring stricter pathways
has noticeably less accuracy – that is, a minimum of 100 in each node
had to be satisfied (column 3), leaving fewer choices for the model to
decide `classe`. Also the effects of shrinkage, where one observes
initial Accuracy gains, controlling the learning rate. When .1, the
model “forgets” it’s current mapping and has a wider set of choices to
solve (higher performance).

#### Bagged, Boosted Trees (treebag)

A second model was built using `treebag`. There are no tuning options
for `treebag`. A comparison will follow.

#### Bagged Linear Discriminate Analysis (LDA)

A third `lda` model was built as well, again there are no tuning
parameters. Next we will examine the resamples.

## Selecting the Classification Model

#### Cross Validation

Cross validation was performed using 10-folds of the training data,
repeated five times. Output below shows not only differences in Accuracy
(placement on plot), but the pattern of solutions, including variance
(indicated by the width).

<img src="index_files/figure-gfm/resamples-analysis-1.png" style="display: block; margin: auto;" />

Obviously the `ldabag` try is out of the question it’s performance is
little better than a guess…but we can use this cross-validation
information to set expectations around the out of sample error rate. One
thing that stands out is how tight the spread is in the plot, indicating
a high degree of fitting.

#### Variable Importance

Below are output from two of our models, the GBM model does not have an
analog that works in `knitr`, but results are similar. Noteworthy is the
agreement across the two lists, though in a different format, each
indicates the same important variables affecting the classifications.
For example, the magnet dimensions play a huge role in these
classifiers.

From the `caret` documentation…For multi-class outcomes, the problem is
decomposed into all pair-wise problems and the area under the curve is
calculated for each class pair (i.e. class 1 vs. class 2, class 2
vs. class 3 etc.). For a specific class, the maximum area under the
curve across the relevant pair-wise AUC’s is used as the variable
importance measure.

    treebag variable importance
    
      only 20 most important variables shown (out of 30)
    
                         Overall
    magnet_dumbbell_z     100.00
    gyros_belt_z           84.04
    magnet_dumbbell_y      81.53
    magnet_belt_y          68.83
    roll_forearm           68.73
    magnet_belt_z          60.93
    pitch_forearm          59.84
    accel_dumbbell_y       57.23
    roll_dumbbell          51.83
    magnet_dumbbell_x      46.60
    magnet_forearm_z       41.28
    roll_arm               35.54
    yaw_arm                33.31
    total_accel_dumbbell   31.13
    gyros_dumbbell_y       29.80
    accel_arm_y            26.96
    pitch_arm              25.91
    magnet_forearm_y       25.18
    accel_forearm_x        25.06
    yaw_forearm            24.50

    ROC curve variable importance
    
      variables are sorted by maximum importance across the classes
      only 20 most important variables shown (out of 30)
    
                              A       B      C      D       E
    pitch_forearm       61.7707 100.000 66.938 61.771 100.000
    accel_forearm_x     42.1399  80.594 42.140 42.140  80.594
    magnet_forearm_x    38.3199  68.920 33.952 33.952  68.920
    magnet_dumbbell_y   47.7577  47.758 47.758 67.091  45.240
    magnet_belt_y       14.5288   7.963 66.731  9.831  14.529
    magnet_dumbbell_x   63.0164  63.016 63.016 63.016  48.771
    roll_dumbbell       41.1058  50.200 31.414 62.682  50.200
    magnet_dumbbell_z   54.7421  36.355 53.774 22.845  54.742
    magnet_belt_z        0.7332   3.302 49.830  2.684   3.302
    pitch_arm           24.7029  40.705 48.890 24.703  40.705
    accel_dumbbell_y    34.7208  18.250 18.250 42.398  34.721
    total_accel_arm     29.7083  39.423 31.418 14.722  39.423
    magnet_forearm_y    20.3198  38.212 29.578 24.136  38.212
    roll_forearm        35.7099   4.501 11.850 24.957  35.710
    roll_arm            34.1398  34.140 34.140 34.140  26.047
    total_accel_forearm 22.6557  24.859 31.706 22.656  24.859
    accel_arm_z         27.9410  27.005 13.400 21.384  27.941
    accel_forearm_y     27.2440   2.803  2.803 23.730  27.244
    accel_arm_y         14.2548  17.965 21.917 15.485  17.965
    yaw_forearm         10.5281  16.451 12.734 21.347  16.451

## Testing the models – predicting new data

With a final model chosen (in the case of GBM), we have three models to
run against the validation subset of our data. Recall these models were
built on 70% of the training data, leaving 30% as a completely fresh
sample.

<div class="figure" style="text-align: center">

<img src="index_files/figure-gfm/plot-predictions-1.png" alt="**Prediction versus Truth**"  />

<p class="caption">

**Prediction versus Truth**

</p>

</div>

Therefore we can see excellent performance from the GBM, a bit less for
treebag, and a mere guess for the LDA. We will select the GBM, knowing
the treebag is not a bad option, and with a little more variation
observed in the resampling, might do better with fresh data.

The **out of sample error rate** is estimated to be between **.007 and
.01** (95% confidence).

## Choices made

The analysis choices made can be summarized:

1.  variable reduction was key – we greatly reduced the concern space
      - high correlation
      - near zero variance
2.  LDA option was not good in this case
3.  low variance in the resampling indicates probable overfitting

## Summary

In conclusion, the discussion here leads us to further questions, as
usual and a good thing. First, we might consult the engineering experts
and design simplified measures revolving around the tighter, core set we
classify with. Second, these data are for less than ten people, so
generalizing (and exploring generalizing models) might be a challenge.
Finally, a new appreciation for budding, open-source technologies and
model build complexities was valuable.

### Citations

The data used in this analysis were graciously provided by the **Human
Activity Recognition** website, which can be accesed
[here](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har).
Thank you HAR\!

## System Information

This work was developed on the following system, using
`R.version.string`:

``` 
  Model Name: iMac
  Processor Name: Quad-Core Intel Core i7
  Memory: 32 GB
```

The following R libraries were utilized:

`library(tidyverse)` `library(rattle)` `library(Hmisc)`
`library(corrgram)` `library(caret)` `library(gridExtra)`
`library(adabag)` `library(fastAdaboost)` `library(rlist)`
`library(stringi)`

# Code Appendix

# quiet load libraries

suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(rattle))
suppressPackageStartupMessages(library(Hmisc))
suppressPackageStartupMessages(library(corrgram))
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(gridExtra))
suppressPackageStartupMessages(library(adabag))
suppressPackageStartupMessages(library(fastAdaboost))
suppressPackageStartupMessages(library(rlist))
suppressPackageStartupMessages(library(stringi))

high\_correlation\_cutoff\_no\<-.8 high\_correlation\_cutoff\_yes\<-.8

original\_training\<-data.frame(read\_csv(“pml-training.csv”, na=c(“NA”,
“\#DIV/0\!”)), stringsAsFactors = F)
TESTING\<-data.frame(read\_csv(“pml-testing.csv”, na=c(“NA”,
“\#DIV/0\!”)), stringsAsFactors = F) TESTING
\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
TESTING\_net\_new\<-data.frame(read\_csv(“WearableComputing\_weight\_lifting\_exercises\_biceps\_curl\_variations.csv”,
na=c(“NA”, “\#DIV/0\!”)), stringsAsFactors = F)

names(TESTING\_net\_new)\[which(\!names(TESTING\_net\_new) %in%
names(original\_training))\]
names(original\_training)\[which(\!names(original\_training) %in%
names(TESTING\_net\_new))\]
TESTING\_net\_new\<-setdiff(TESTING\_net\_new\[,-15\],original\_training\[,-c(1,16)\])

head(TESTING\_net\_new\[1:10,1:10\])
\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#

\#TESTING\<-data.frame(read\_csv(“pml-testing.csv”, na=c(“NA”,
“\#DIV/0\!”)), stringsAsFactors = F) dim(original\_training)
names(original\_training) str(original\_training\[,1:10\])
varmap\_original\<-data.frame( varname=colnames(original\_training),
varindex=c(1:length(colnames(original\_training))), stringsAsFactors =
F) vartypes\<-data.frame(typex=sapply(original\_training, typeof),
varname=colnames(original\_training), stringsAsFactors = F)
varmap\_original\<-left\_join(varmap\_original, vartypes)
dim(original\_training)

# split off junk variables, but have them avaialable as needed…

junk\_frame\<-select(original\_training, user\_name, cvtd\_timestamp,
new\_window, X1, raw\_timestamp\_part\_1, raw\_timestamp\_part\_2,
num\_window) head(junk\_frame)

# remove the summary records here, maybe deal with later (new\_window==“yes”)

training\_new\_window\_no\<-filter(original\_training,
new\_window==“no”)%\>%select(-c(user\_name, cvtd\_timestamp, X1,
raw\_timestamp\_part\_1, raw\_timestamp\_part\_2, num\_window,
new\_window)) training\_new\_window\_yes\<-filter(original\_training,
new\_window==“yes”)%\>%select(-c(user\_name, cvtd\_timestamp, X1,
raw\_timestamp\_part\_1, raw\_timestamp\_part\_2, num\_window,
new\_window)) head(training\_new\_window\_no\[1:10,1:10\])
head(training\_new\_window\_yes\[1:10,1:10\])

# reorder columns, character variables up front for easier subsetting

df\_charvars\_check\<-data.frame(sapply(names(training\_new\_window\_no),
function(x){is.character(training\_new\_window\_no\[,x\])}))
newvarlist\<-c(which(df\_charvars\_check\>0),which(df\_charvars\_check\<1))
training\_new\_window\_no\<-training\_new\_window\_no\[,newvarlist\]
str(training\_new\_window\_no\[,1:10\]) dim(training\_new\_window\_no)

df\_charvars\_check\<-data.frame(sapply(names(training\_new\_window\_yes),
function(x){is.character(training\_new\_window\_yes\[,x\])}))
newvarlist\<-c(which(df\_charvars\_check\>0),which(df\_charvars\_check\<1))
training\_new\_window\_yes\<-training\_new\_window\_yes\[,newvarlist\]
dim(training\_new\_window\_yes)
head(training\_new\_window\_yes\[1:10,1:10\])

# identify very low variance variables

nzv\_no\<-data.frame(nearZeroVar(training\_new\_window\_no, saveMetrics
= T), varname=names(training\_new\_window\_no), nzv\_no=rep(“no”,
length(names(training\_new\_window\_no))), stringsAsFactors =
F)%\>%filter(zeroVar==TRUE & near(percentUnique,0)) dim(nzv\_no)
nzv\_yes\<-data.frame(nearZeroVar(training\_new\_window\_yes,
saveMetrics = T), varname=names(training\_new\_window\_yes),
nzv\_yes=rep(“yes”, length(names(training\_new\_window\_yes))),
stringsAsFactors = F)%\>%filter(zeroVar==TRUE & near(percentUnique,0))
dim(nzv\_yes)

(lst\_nzv\_reductions\_no\<-which(names(training\_new\_window\_no) %in%
as.list(nzv\_no\(name))) (lst_nzv_reductions_yes<-which(names(training_new_window_yes) %in% as.list(nzv_yes\)name)))

# identify high correlation variables, will remove later, adjust cutoff

corr\_matrix\<-abs(cor(training\_new\_window\_no\[,-1\]))
diag(corr\_matrix)\<-0
high\_corr\_no\<-unique(names(which(corr\_matrix\>high\_correlation\_cutoff\_no,
arr.ind = T)\[,1\]))
(lst\_high\_corr\_no\<-which(names(training\_new\_window\_no) %in%
high\_corr\_no)) corr\_matrix\_no\<-corr\_matrix

corr\_matrix\<-abs(cor(training\_new\_window\_yes\[,-1\]))
diag(corr\_matrix)\<-0
high\_corr\_yes\<-unique(names(which(corr\_matrix\>high\_correlation\_cutoff\_yes,
arr.ind = T)\[,1\]))
(lst\_high\_corr\_yes\<-which(names(training\_new\_window\_yes) %in%
high\_corr\_yes))

# NEED TO DISTINGUISH YES NO VERSIONS OF DROPS IN VARMAP…\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!

nrow(nzv\_yes)-nrow(inner\_join(nzv\_no, nzv\_yes, by = c(“freqRatio”,
“percentUnique”, “zeroVar”, “nzv”, “varname”)))
varmap\<-left\_join(varmap\_original,
data.frame(varname=names(junk\_frame), source1=“junk both”,
junk\_status=“junk”))%\>% left\_join(data.frame(varname=high\_corr\_no,
high\_corr\_no=str\_c(“corr above”, high\_correlation\_cutoff\_no)))%\>%
left\_join(data.frame(varname=high\_corr\_yes,
high\_corr\_yes=str\_c(“corr above”,
high\_correlation\_cutoff\_yes)))%\>%
left\_join(nzv\_no)%\>%left\_join(nzv\_yes) head(varmap,30)

clean\_no\<-which(names(training\_new\_window\_no) %in%
as.list(filter(varmap, is.na(junk\_status)==T & is.na(nzv\_no)==T &
is.na(high\_corr\_no)==T))\(varname) clean_yes<-which(names(training_new_window_yes) %in% as.list(filter(varmap, is.na(junk_status)==T & is.na(nzv_yes)==T & is.na(high_corr_yes)==T))\)varname)

no\_final\<-training\_new\_window\_no\[,clean\_no\] head(no\_final)
yes\_final\<-training\_new\_window\_yes\[,clean\_yes\] head(yes\_final)

# identify missing values

length(which(sapply(names(no\_final), function(x)
sum(is.na(no\_final\[,x\]))\>0))) length(which(sapply(names(yes\_final),
function(x) sum(is.na(yes\_final\[,x\]))\>0)))

# update varmap

tmp\<-data.frame(count\_missing\_no=sapply(names(no\_final), function(x)
sum(is.na(no\_final\[,x\])))) varmap\<-left\_join(varmap,
data.frame(varname=rownames(tmp), tmp), by = “varname”)
tmp\<-data.frame(count\_missing\_yes=sapply(names(yes\_final),
function(x) sum(is.na(yes\_final\[,x\])))) varmap\<-left\_join(varmap,
data.frame(varname=rownames(tmp), tmp), by = “varname”)

TESTING\<-data.frame(read\_csv(“pml-testing.csv”, na=c(“NA”,
“\#DIV/0\!”)), stringsAsFactors = F) TESTING str(TESTING)
\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
TESTING\_net\_new\<-data.frame(read\_csv(“WearableComputing\_weight\_lifting\_exercises\_biceps\_curl\_variations.csv”,
na=c(“NA”, “\#DIV/0\!”)), stringsAsFactors = F) str(TESTING\_net\_new)
\# names(TESTING\_net\_new) \# names(TESTING)
names(TESTING\_net\_new)\[15\]\<-“skewness\_roll\_belt.1”
names(TESTING\_net\_new)\[which(\!names(TESTING\_net\_new) %in%
names(TESTING))\]
names(original\_training)\[which(\!names(original\_training) %in%
names(TESTING\_net\_new))\]
TESTING\_net\_new\<-setdiff(TESTING\_net\_new\[,-15\],original\_training\[,-c(1,16)\])
nrow(inner\_join(original\_training, TESTING\_net\_new))
\#head(TESTING\_net\_new\[1:10,1:10\])

TESTING\<-filter(TESTING, new\_window==“no”)%\>%select(-c(user\_name,
cvtd\_timestamp, X1, raw\_timestamp\_part\_1, raw\_timestamp\_part\_2,
num\_window, new\_window)) TESTING\_net\_new\<-filter(TESTING\_net\_new,
new\_window==“no”)%\>%select(-c(user\_name, cvtd\_timestamp,
raw\_timestamp\_part\_1, raw\_timestamp\_part\_2, num\_window,
new\_window))

\#clean\_no\<-which(names(training\_new\_window\_no) %in%
as.list(filter(varmap, is.na(junk\_status)==T & is.na(nzv\_no)==T &
is.na(high\_corr\_no)==T))\(varname) clean_no_TESTING<-which(names(TESTING) %in% as.list(filter(varmap, is.na(nzv_no)==T & is.na(high_corr_no)==T))\)varname)
clean\_no\_TESTING\_net\_new\<-which(names(TESTING\_net\_new) %in%
as.list(filter(varmap, is.na(nzv\_no)==T &
is.na(high\_corr\_no)==T))$varname)

TESTING\_net\_new\<-TESTING\_net\_new\[,clean\_no\_TESTING\_net\_new\]
TESTING\<-TESTING\[,clean\_no\_TESTING\]

dim(TESTING\_net\_new) dim(TESTING)

head(TESTING\[1:10,1:10\]) head(TESTING\_net\_new\[1:10,1:10\])
\#nrow(filter(varmap, is.na(nzv\_no)==F))

junk\_drops\<-data.frame( varmap\[1:7, 1:2\], description=c( “a
sequential record id”, “study participant name”, “raw timestamp part 1”,
“raw timestamp part 2”, “converted timestamp”, “new\_window”,
“num\_window”), reason\_removed=c( “obviously this would distort the
model if left in”, “we want to predict classe for any person, therefore
removed”, “time disregarded in this analysis”, "“,”“,”used to separate
‘yes’ data, n=406“,”an incremental window counter"))

# 

testrun\_portion1\<-.1 testrun\_foldsx1\<-4 testrun\_repeatsx1\<-2
testrun\_portion2\<-.1 testrun\_foldsx2\<-2 testrun\_repeatsx2\<-2
testrun\_portion3\<-.1 testrun\_foldsx3\<-2 testrun\_repeatsx3\<-2

final\_portion\<-.7 final\_foldsx\<-10 final\_repeatsx\<-5
\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#

cluster \<- makeCluster(detectCores() - 1) \# convention to leave 1 core
for OS registerDoParallel(cluster)

set.seed(123)

main\<-no\_final

impute\_methodx\<-c(“center”,“scale”,“YeoJohnson”)

(basex\<-nrow(main)) samplex\<-sample(basex,
floor(testrun\_portion1\*basex)) dfx\<-main\[samplex,\]

impute\<-preProcess(dfx\[,-1\], method = impute\_methodx,
allowParallel=T) preprocessed\<-predict(impute, dfx\[,-1\])

gbmGrid\<-expand.grid(interaction.depth = c(1, 5, 10), n.trees =
(1:10)\*50, shrinkage = c(0.1, 0.2, .3), n.minobsinnode = c(20, 50,100))
nrow(gbmGrid) head(gbmGrid)

fitControl\<-trainControl(method = “repeatedcv”, number =
testrun\_foldsx1, repeats = testrun\_repeatsx1, allowParallel = T)

set.seed(825)

gbmFit\_dotx \<- train(y=dfx$classe, x = preprocessed, method = “gbm”,
trControl = fitControl, verbose = FALSE, tuneGrid = gbmGrid)

pred\_dfx\<-data.frame(ground\_truth=dfx\(classe, prediction=predict(gbmFit_dotx,preprocessed)) confusionMatrix(pred_dfx\)prediction,
pred\_dfx$ground\_truth)

freshy\<-main\[-samplex,\] preprocessed\_freshy\<-predict(impute,
freshy\[,-1\])
pred\_freshy\<-data.frame(ground\_truth=freshy\(classe, prediction=predict(gbmFit_dotx,preprocessed_freshy)) confusionMatrix(pred_freshy\)prediction,
pred\_freshy$ground\_truth)

preprocessed\_TESTING\_net\_new\<-predict(impute,
TESTING\_net\_new\[,-31\])
pred\_TESTING\_net\_new\<-data.frame(ground\_truth=TESTING\_net\_new\(classe, prediction=predict(gbmFit_dotx,preprocessed_TESTING_net_new)) confusionMatrix(pred_TESTING_net_new\)prediction,
pred\_TESTING\_net\_new$ground\_truth)

stopCluster(cluster) registerDoSEQ()

best\_within\_1pct\<-cbind(tolerance=“within 1 pct”,
gbmFit\_dotx\(results[tolerance(gbmFit_dotx\)results, metric =
“Accuracy”, tol = 1, maximize = TRUE),1:6\])
best\_within\_2pct\<-cbind(tolerance=“within 2 pct”,
gbmFit\_dotx\(results[tolerance(gbmFit_dotx\)results, metric =
“Accuracy”, tol = 2, maximize = TRUE),1:6\])
best\_within\_3pct\<-cbind(tolerance=“within 3 pct”,
gbmFit\_dotx\(results[tolerance(gbmFit_dotx\)results, metric =
“Accuracy”, tol = 3, maximize = TRUE),1:6\])
best\_within\_4pct\<-cbind(tolerance=“within 4 pct”,
gbmFit\_dotx\(results[tolerance(gbmFit_dotx\)results, metric =
“Accuracy”, tol = 4, maximize = TRUE),1:6\])
best\_within\_5pct\<-cbind(tolerance=“within 5 pct”,
gbmFit\_dotx\(results[tolerance(gbmFit_dotx\)results, metric =
“Accuracy”, tol = 5, maximize = TRUE),1:6\])

(best\_alternative\_models\<-rbind(best\_within\_1pct,
best\_within\_2pct, best\_within\_3pct, best\_within\_4pct,
best\_within\_5pct))

# 

# check the real stuff…

temptesting\<-data.frame(read\_csv(“pml-testing.csv”, na=c(“NA”,
“\#DIV/0\!”)), stringsAsFactors = F)
tempreal\<-data.frame(read\_csv(“WearableComputing\_weight\_lifting\_exercises\_biceps\_curl\_variations.csv”,
na=c(“NA”, “\#DIV/0\!”)), stringsAsFactors = F)
t\<-temptesting%\>%select(user\_name, raw\_timestamp\_part\_1,
raw\_timestamp\_part\_2, gyros\_belt\_x, gyros\_belt\_y)
j\<-predict(impute, TESTING) p\<-predict(gbmFit\_dotx, j)
combo\<-data.frame(cbind(t,p)) (r\<-tempreal%\>%select(user\_name,
raw\_timestamp\_part\_1, raw\_timestamp\_part\_2, gyros\_belt\_x,
gyros\_belt\_y, classe)%\>%inner\_join(combo)%\>%mutate(good=classe==p))
\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
nrow(dfx)
round(confusionMatrix(pred\_dfx\(prediction, pred_dfx\)ground\_truth)\(overall,3) nrow(freshy) round(confusionMatrix(pred_freshy\)prediction,
pred\_freshy\(ground_truth)\)overall,3) nrow(TESTING\_net\_new)
round(confusionMatrix(pred\_TESTING\_net\_new\(prediction, pred_TESTING_net_new\)ground\_truth)\(overall,3) nrow(r) sum(r\)good/20)

ggplot(gbmFit\_dotx, main=“gbmFit\_dotx”)+ggtitle(“Repeated Cross
Validation, GBM tuning matrix”)

suppressPackageStartupMessages(library(doParallel)) cluster \<-
makeCluster(detectCores() - 1) \# convention to leave 1 core for OS
registerDoParallel(cluster)

set.seed(123)

main\<-no\_final

impute\_methodx\<-c(“center”,“scale”,“YeoJohnson”)

(basex\<-nrow(main)) samplex\<-sample(basex,
floor(testrun\_portion2\*basex)) dfx\<-main\[samplex,\]

system.time(impute\<-preProcess(dfx\[,-1\], method = impute\_methodx,
allowParallel=T)) preprocessed\<-predict(impute, dfx\[,-1\])

fitControl\<-trainControl(method = “repeatedcv”, number =
testrun\_foldsx2, repeats = testrun\_repeatsx2, allowParallel = T)

set.seed(825)

treebagFit\_dotx \<- train(y=dfx$classe, x = preprocessed, method =
“treebag”, trControl = fitControl, verbose = FALSE)

pred\_dfx\<-data.frame(ground\_truth=dfx\(classe, prediction=predict(treebagFit_dotx,preprocessed)) confusionMatrix(pred_dfx\)prediction,
pred\_dfx$ground\_truth)

freshy\<-main\[-samplex,\] preprocessed\_freshy\<-predict(impute,
freshy\[,-1\])
pred\_freshy\<-data.frame(ground\_truth=freshy\(classe, prediction=predict(treebagFit_dotx,preprocessed_freshy)) confusionMatrix(pred_freshy\)prediction,
pred\_freshy$ground\_truth)

TESTING\_net\_newx\<-TESTING\_net\_new%\>%mutate(roll\_dumbbell=ifelse(is.na(roll\_dumbbell)==T,
mean(TESTING\_net\_new\(roll_dumbbell, na.rm = T), roll_dumbbell)) preprocessed_TESTING_net_newx<-predict(impute, TESTING_net_newx[,-31]) pred_TESTING_net_newx<-data.frame(ground_truth=TESTING_net_newx\)classe,
prediction=predict(treebagFit\_dotx,preprocessed\_TESTING\_net\_newx))
confusionMatrix(pred\_TESTING\_net\_newx\(prediction, pred_TESTING_net_newx\)ground\_truth)

stopCluster(cluster) registerDoSEQ()

# 

# check the real stuff…

temptesting\<-data.frame(read\_csv(“pml-testing.csv”, na=c(“NA”,
“\#DIV/0\!”)), stringsAsFactors = F)
tempreal\<-data.frame(read\_csv(“WearableComputing\_weight\_lifting\_exercises\_biceps\_curl\_variations.csv”,
na=c(“NA”, “\#DIV/0\!”)), stringsAsFactors = F)
t\<-temptesting%\>%select(user\_name, raw\_timestamp\_part\_1,
raw\_timestamp\_part\_2, gyros\_belt\_x, gyros\_belt\_y)
j\<-predict(impute, TESTING) p\<-predict(treebagFit\_dotx, j)
combo\<-data.frame(cbind(t,p)) (r\<-tempreal%\>%select(user\_name,
raw\_timestamp\_part\_1, raw\_timestamp\_part\_2, gyros\_belt\_x,
gyros\_belt\_y, classe)%\>%inner\_join(combo)%\>%mutate(good=classe==p))
\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
nrow(dfx)
round(confusionMatrix(pred\_dfx\(prediction, pred_dfx\)ground\_truth)\(overall,3) nrow(freshy) round(confusionMatrix(pred_freshy\)prediction,
pred\_freshy\(ground_truth)\)overall,3) nrow(TESTING\_net\_newx)
nrow(TESTING\_net\_new)
round(confusionMatrix(pred\_TESTING\_net\_newx\(prediction, pred_TESTING_net_newx\)ground\_truth)\(overall,3) nrow(r) sum(r\)good/20)

suppressPackageStartupMessages(library(doParallel)) cluster \<-
makeCluster(detectCores() - 1) \# convention to leave 1 core for OS
registerDoParallel(cluster)

set.seed(123)

main\<-no\_final

impute\_methodx\<-c(“center”,“scale”,“YeoJohnson”)

(basex\<-nrow(main)) samplex\<-sample(basex,
floor(testrun\_portion3\*basex)) dfx\<-main\[samplex,\]

system.time(impute\<-preProcess(dfx\[,-1\], method = impute\_methodx,
allowParallel=T)) preprocessed\<-predict(impute, dfx\[,-1\])

fitControl\<-trainControl(method = “repeatedcv”, number =
testrun\_foldsx3, repeats = testrun\_repeatsx3, allowParallel = T)

set.seed(825)

ldaFit\_dotx \<- train(y=dfx$classe, x = preprocessed, method = “lda”,
trControl = fitControl, verbose = FALSE)

pred\_dfx\<-data.frame(ground\_truth=dfx\(classe, prediction=predict(ldaFit_dotx,preprocessed)) confusionMatrix(pred_dfx\)prediction,
pred\_dfx$ground\_truth)

freshy\<-main\[-samplex,\] preprocessed\_freshy\<-predict(impute,
freshy\[,-1\])
pred\_freshy\<-data.frame(ground\_truth=freshy\(classe, prediction=predict(ldaFit_dotx,preprocessed_freshy)) confusionMatrix(pred_freshy\)prediction,
pred\_freshy$ground\_truth)

TESTING\_net\_newx\<-TESTING\_net\_new%\>%mutate(roll\_dumbbell=ifelse(is.na(roll\_dumbbell)==T,
mean(TESTING\_net\_new\(roll_dumbbell, na.rm = T), roll_dumbbell)) preprocessed_TESTING_net_newx<-predict(impute, TESTING_net_newx[,-31]) pred_TESTING_net_newx<-data.frame(ground_truth=TESTING_net_newx\)classe,
prediction=predict(ldaFit\_dotx,preprocessed\_TESTING\_net\_newx))
confusionMatrix(pred\_TESTING\_net\_newx\(prediction, pred_TESTING_net_newx\)ground\_truth)

stopCluster(cluster) registerDoSEQ()

# 

# check the real stuff…

temptesting\<-data.frame(read\_csv(“pml-testing.csv”, na=c(“NA”,
“\#DIV/0\!”)), stringsAsFactors = F)
tempreal\<-data.frame(read\_csv(“WearableComputing\_weight\_lifting\_exercises\_biceps\_curl\_variations.csv”,
na=c(“NA”, “\#DIV/0\!”)), stringsAsFactors = F)
t\<-temptesting%\>%select(user\_name, raw\_timestamp\_part\_1,
raw\_timestamp\_part\_2, gyros\_belt\_x, gyros\_belt\_y)
j\<-predict(impute, TESTING) p\<-predict(ldaFit\_dotx, j)
combo\<-data.frame(cbind(t,p)) (r\<-tempreal%\>%select(user\_name,
raw\_timestamp\_part\_1, raw\_timestamp\_part\_2, gyros\_belt\_x,
gyros\_belt\_y, classe)%\>%inner\_join(combo)%\>%mutate(good=classe==p))
\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
nrow(dfx)
round(confusionMatrix(pred\_dfx\(prediction, pred_dfx\)ground\_truth)\(overall,3) nrow(freshy) round(confusionMatrix(pred_freshy\)prediction,
pred\_freshy\(ground_truth)\)overall,3) nrow(TESTING\_net\_newx)
nrow(TESTING\_net\_new)
round(confusionMatrix(pred\_TESTING\_net\_newx\(prediction, pred_TESTING_net_newx\)ground\_truth)\(overall,3) nrow(r) sum(r\)good/20)

suppressPackageStartupMessages(library(doParallel)) cluster \<-
makeCluster(detectCores() - 1) \# convention to leave 1 core for OS
registerDoParallel(cluster)

set.seed(123)

choice\<-2 use\_best\<-“yes”

best\_within\_1pct\<-cbind(tolerance=“within 1 pct”,
gbmFit\_dotx\(results[tolerance(gbmFit_dotx\)results, metric =
“Accuracy”, tol = 1, maximize = TRUE),1:6\])
best\_within\_2pct\<-cbind(tolerance=“within 2 pct”,
gbmFit\_dotx\(results[tolerance(gbmFit_dotx\)results, metric =
“Accuracy”, tol = 2, maximize = TRUE),1:6\])
best\_within\_3pct\<-cbind(tolerance=“within 3 pct”,
gbmFit\_dotx\(results[tolerance(gbmFit_dotx\)results, metric =
“Accuracy”, tol = 3, maximize = TRUE),1:6\])
best\_within\_4pct\<-cbind(tolerance=“within 4 pct”,
gbmFit\_dotx\(results[tolerance(gbmFit_dotx\)results, metric =
“Accuracy”, tol = 4, maximize = TRUE),1:6\])
best\_within\_5pct\<-cbind(tolerance=“within 5 pct”,
gbmFit\_dotx\(results[tolerance(gbmFit_dotx\)results, metric =
“Accuracy”, tol = 5, maximize = TRUE),1:6\])

(best\_alternative\_models\<-rbind(best\_within\_1pct,
best\_within\_2pct, best\_within\_3pct, best\_within\_4pct,
best\_within\_5pct))

(shrinkagex\<-best\_alternative\_models\[choice,2\])
(interactiondepthx\<-best\_alternative\_models\[choice,3\])
(nminobsinnodex\<-best\_alternative\_models\[choice,4\])
(ntreesx\<-best\_alternative\_models\[choice,5\])

if(use\_best==“yes”)
(shrinkagex\<-gbmFit\_dotx\(results[best(gbmFit_dotx\)results,
“Accuracy”, maximize = T),1\]) if(use\_best==“yes”)
(interactiondepthx\<-gbmFit\_dotx\(results[best(gbmFit_dotx\)results,
“Accuracy”, maximize = T),2\]) if(use\_best==“yes”)
(nminobsinnodex\<-gbmFit\_dotx\(results[best(gbmFit_dotx\)results,
“Accuracy”, maximize = T),3\]); if(use\_best==“yes”)
(ntreesx\<-gbmFit\_dotx\(results[best(gbmFit_dotx\)results, “Accuracy”,
maximize = T),4\])

main\<-no\_final

impute\_methodx\<-c(“center”,“scale”,“YeoJohnson”)

(basex\<-nrow(main)) samplex\<-sample(basex,
floor(final\_portion\*basex)) dfx\<-main\[samplex,\]

system.time(impute\<-preProcess(dfx\[,-1\], method = impute\_methodx,
allowParallel=T)) preprocessed\<-predict(impute, dfx\[,-1\])

gbmGrid\<-expand.grid(interaction.depth = interactiondepthx, n.trees =
ntreesx, shrinkage = shrinkagex, n.minobsinnode = nminobsinnodex)
nrow(gbmGrid) head(gbmGrid)

fitControl\<-trainControl(method = “repeatedcv”, number = final\_foldsx,
repeats = final\_repeatsx, allowParallel = T)

set.seed(825)

gbmFit\_dotxfinal \<- train(y=dfx$classe, x = preprocessed, method =
“gbm”, trControl = fitControl, verbose = FALSE, tuneGrid = gbmGrid)

pred\_dfx\<-data.frame(ground\_truth=dfx\(classe, prediction=predict(gbmFit_dotxfinal,preprocessed)) confusionMatrix(pred_dfx\)prediction,
pred\_dfx$ground\_truth)

freshy\<-main\[-samplex,\] preprocessed\_freshy\<-predict(impute,
freshy\[,-1\])
pred\_freshy\<-data.frame(ground\_truth=freshy\(classe, prediction=predict(gbmFit_dotxfinal,preprocessed_freshy)) confusionMatrix(pred_freshy\)prediction,
pred\_freshy$ground\_truth)

preprocessed\_TESTING\_net\_new\<-predict(impute,
TESTING\_net\_new\[,-31\])
pred\_TESTING\_net\_new\<-data.frame(ground\_truth=TESTING\_net\_new\(classe, prediction=predict(gbmFit_dotxfinal,preprocessed_TESTING_net_new)) confusionMatrix(pred_TESTING_net_new\)prediction,
pred\_TESTING\_net\_new$ground\_truth)

# special, store this models freshy

freshy\_gbm\<-freshy impute\_gbm\<-impute

stopCluster(cluster) registerDoSEQ()

# 

# check the real stuff…

temptesting\<-data.frame(read\_csv(“pml-testing.csv”, na=c(“NA”,
“\#DIV/0\!”)), stringsAsFactors = F)
tempreal\<-data.frame(read\_csv(“WearableComputing\_weight\_lifting\_exercises\_biceps\_curl\_variations.csv”,
na=c(“NA”, “\#DIV/0\!”)), stringsAsFactors = F)
t\<-temptesting%\>%select(user\_name, raw\_timestamp\_part\_1,
raw\_timestamp\_part\_2, gyros\_belt\_x, gyros\_belt\_y)
j\<-predict(impute, TESTING) p\<-predict(gbmFit\_dotxfinal, j)
combo\<-data.frame(cbind(t,p)) (r\<-tempreal%\>%select(user\_name,
raw\_timestamp\_part\_1, raw\_timestamp\_part\_2, gyros\_belt\_x,
gyros\_belt\_y, classe)%\>%inner\_join(combo)%\>%mutate(good=classe==p))
\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
nrow(dfx)
round(confusionMatrix(pred\_dfx\(prediction, pred_dfx\)ground\_truth)\(overall,3) nrow(freshy) round(confusionMatrix(pred_freshy\)prediction,
pred\_freshy\(ground_truth)\)overall,3) nrow(TESTING\_net\_new)
round(confusionMatrix(pred\_TESTING\_net\_new\(prediction, pred_TESTING_net_new\)ground\_truth)\(overall,3) nrow(r) sum(r\)good/20)

suppressPackageStartupMessages(library(doParallel)) cluster \<-
makeCluster(detectCores() - 1) \# convention to leave 1 core for OS
registerDoParallel(cluster)

set.seed(123)

main\<-no\_final

impute\_methodx\<-c(“center”,“scale”,“YeoJohnson”)

(basex\<-nrow(main)) samplex\<-sample(basex,
floor(final\_portion\*basex)) dfx\<-main\[samplex,\]

system.time(impute\<-preProcess(dfx\[,-1\], method = impute\_methodx,
allowParallel=T)) preprocessed\<-predict(impute, dfx\[,-1\])

fitControl\<-trainControl(method = “repeatedcv”, number = final\_foldsx,
repeats = final\_repeatsx, allowParallel = T)

set.seed(825)

treebagFit\_dotxfinal \<- train(y=dfx$classe, x = preprocessed, method =
“treebag”, trControl = fitControl, verbose = FALSE)

pred\_dfx\<-data.frame(ground\_truth=dfx\(classe, prediction=predict(treebagFit_dotxfinal,preprocessed)) confusionMatrix(pred_dfx\)prediction,
pred\_dfx$ground\_truth)

freshy\<-main\[-samplex,\] preprocessed\_freshy\<-predict(impute,
freshy\[,-1\])
pred\_freshy\<-data.frame(ground\_truth=freshy\(classe, prediction=predict(treebagFit_dotxfinal,preprocessed_freshy)) confusionMatrix(pred_freshy\)prediction,
pred\_freshy$ground\_truth)

TESTING\_net\_newx\<-TESTING\_net\_new%\>%mutate(roll\_dumbbell=ifelse(is.na(roll\_dumbbell)==T,
mean(TESTING\_net\_new\(roll_dumbbell, na.rm = T), roll_dumbbell)) preprocessed_TESTING_net_newx<-predict(impute, TESTING_net_newx[,-31]) pred_TESTING_net_newx<-data.frame(ground_truth=TESTING_net_newx\)classe,
prediction=predict(treebagFit\_dotxfinal,preprocessed\_TESTING\_net\_newx))
confusionMatrix(pred\_TESTING\_net\_newx\(prediction, pred_TESTING_net_newx\)ground\_truth)

# special, store this models freshy

freshy\_treebag\<-freshy impute\_treebag\<-impute

stopCluster(cluster) registerDoSEQ()

# 

# check the real stuff…

temptesting\<-data.frame(read\_csv(“pml-testing.csv”, na=c(“NA”,
“\#DIV/0\!”)), stringsAsFactors = F)
tempreal\<-data.frame(read\_csv(“WearableComputing\_weight\_lifting\_exercises\_biceps\_curl\_variations.csv”,
na=c(“NA”, “\#DIV/0\!”)), stringsAsFactors = F)
t\<-temptesting%\>%select(user\_name, raw\_timestamp\_part\_1,
raw\_timestamp\_part\_2, gyros\_belt\_x, gyros\_belt\_y)
j\<-predict(impute, TESTING) p\<-predict(treebagFit\_dotxfinal, j)
combo\<-data.frame(cbind(t,p)) (r\<-tempreal%\>%select(user\_name,
raw\_timestamp\_part\_1, raw\_timestamp\_part\_2, gyros\_belt\_x,
gyros\_belt\_y, classe)%\>%inner\_join(combo)%\>%mutate(good=classe==p))
\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
nrow(dfx)
round(confusionMatrix(pred\_dfx\(prediction, pred_dfx\)ground\_truth)\(overall,3) nrow(freshy) round(confusionMatrix(pred_freshy\)prediction,
pred\_freshy\(ground_truth)\)overall,3) nrow(TESTING\_net\_newx)
nrow(TESTING\_net\_new)
round(confusionMatrix(pred\_TESTING\_net\_newx\(prediction, pred_TESTING_net_newx\)ground\_truth)\(overall,3) nrow(r) sum(r\)good/20)

suppressPackageStartupMessages(library(doParallel)) cluster \<-
makeCluster(detectCores() - 1) \# convention to leave 1 core for OS
registerDoParallel(cluster)

set.seed(123)

main\<-no\_final

impute\_methodx\<-c(“center”,“scale”,“YeoJohnson”)

(basex\<-nrow(main)) samplex\<-sample(basex,
floor(final\_portion\*basex)) dfx\<-main\[samplex,\]

system.time(impute\<-preProcess(dfx\[,-1\], method = impute\_methodx,
allowParallel=T)) preprocessed\<-predict(impute, dfx\[,-1\])

fitControl\<-trainControl(method = “repeatedcv”, number = final\_foldsx,
repeats = final\_repeatsx, allowParallel = T)

set.seed(825)

ldaFit\_dotxfinal \<- train(y=dfx$classe, x = preprocessed, method =
“lda”, trControl = fitControl, verbose = FALSE)

pred\_dfx\<-data.frame(ground\_truth=dfx\(classe, prediction=predict(ldaFit_dotxfinal,preprocessed)) confusionMatrix(pred_dfx\)prediction,
pred\_dfx$ground\_truth)

freshy\<-main\[-samplex,\] preprocessed\_freshy\<-predict(impute,
freshy\[,-1\])
pred\_freshy\<-data.frame(ground\_truth=freshy\(classe, prediction=predict(ldaFit_dotxfinal,preprocessed_freshy)) confusionMatrix(pred_freshy\)prediction,
pred\_freshy$ground\_truth)

TESTING\_net\_newx\<-TESTING\_net\_new%\>%mutate(roll\_dumbbell=ifelse(is.na(roll\_dumbbell)==T,
mean(TESTING\_net\_new\(roll_dumbbell, na.rm = T), roll_dumbbell)) preprocessed_TESTING_net_newx<-predict(impute, TESTING_net_newx[,-31]) pred_TESTING_net_newx<-data.frame(ground_truth=TESTING_net_newx\)classe,
prediction=predict(ldaFit\_dotxfinal,preprocessed\_TESTING\_net\_newx))
confusionMatrix(pred\_TESTING\_net\_newx\(prediction, pred_TESTING_net_newx\)ground\_truth)

# special, store this models freshy

freshy\_lda\<-freshy impute\_lda\<-impute

stopCluster(cluster) registerDoSEQ()

# 

# check the real stuff…

temptesting\<-data.frame(read\_csv(“pml-testing.csv”, na=c(“NA”,
“\#DIV/0\!”)), stringsAsFactors = F)
tempreal\<-data.frame(read\_csv(“WearableComputing\_weight\_lifting\_exercises\_biceps\_curl\_variations.csv”,
na=c(“NA”, “\#DIV/0\!”)), stringsAsFactors = F)
t\<-temptesting%\>%select(user\_name, raw\_timestamp\_part\_1,
raw\_timestamp\_part\_2, gyros\_belt\_x, gyros\_belt\_y)
j\<-predict(impute, TESTING) p\<-predict(ldaFit\_dotx, j)
combo\<-data.frame(cbind(t,p)) (r\<-tempreal%\>%select(user\_name,
raw\_timestamp\_part\_1, raw\_timestamp\_part\_2, gyros\_belt\_x,
gyros\_belt\_y, classe)%\>%inner\_join(combo)%\>%mutate(good=classe==p))
\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
nrow(dfx)
round(confusionMatrix(pred\_dfx\(prediction, pred_dfx\)ground\_truth)\(overall,3) nrow(freshy) round(confusionMatrix(pred_freshy\)prediction,
pred\_freshy\(ground_truth)\)overall,3) nrow(TESTING\_net\_newx)
nrow(TESTING\_net\_new)
round(confusionMatrix(pred\_TESTING\_net\_newx\(prediction, pred_TESTING_net_newx\)ground\_truth)\(overall,3) nrow(r) sum(r\)good/20)

resamps\<-resamples(list( gbm = gbmFit\_dotxfinal, ldabag =
ldaFit\_dotxfinal, treebag = treebagFit\_dotxfinal))
\#summary(resamps)\(statistics[1] #summary(resamps)\)statistics\[2\]
ggplot(resamps)+aes(color=resamps$models)+ggtitle(“Model Accuracy over
repeated Cross Validated samples”)+ylab(“Accuracy”)

treebagImportance\<-varImp(treebagFit\_dotxfinal) treebagImportance
ldaImportance\<-varImp(ldaFit\_dotxfinal) ldaImportance

freshyx\<-predict(impute\_gbm, freshy\_gbm)
gbm\<-ggplot(mapping=aes(x=freshy\_gbm\(classe, y=predict(gbmFit_dotxfinal, freshyx), color=freshy_gbm\)classe))+
geom\_jitter(show.legend = F)+xlab("“)+ylab(”Predicted
classe“)+ggtitle(”GBM“) freshyx\<-predict(impute\_treebag,
freshy\_treebag)
treebag\<-ggplot(mapping=aes(x=freshy\_treebag\(classe, y=predict(treebagFit_dotxfinal, freshyx), color=freshy_treebag\)classe))+
geom\_jitter(show.legend = F)+xlab(”Truth“)+ylab(”“)+ggtitle(”treebag“)
freshyx\<-predict(impute\_lda, freshy\_lda)
lda\<-ggplot(mapping=aes(x=freshy\_lda\(classe, y=predict(ldaFit_dotxfinal, freshyx), color=freshy_lda\)classe))+
geom\_jitter(show.legend = F)+xlab(”“)+ylab(”“)+ggtitle(”ldabag")
grid.arrange(gbm, treebag, lda, nrow = 1)
