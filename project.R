# LIBRARIES ###################################################################
# Helper packages
library(dplyr)        # For data manipulation
library(ggplot2)      # For awesome graphics
library(RColorBrewer) # For coloring the points on the plots
library(gridExtra)    # For putting plots side by side
library(doParallel)   # For parallel backend to foreach
library(foreach)      # For parallel processing with for loops
library(tidyr)        # For data manipulation
library(stringr)      # For string functionality

# Modeling process packages
library(rsample)     # For resampling procedures
library(rpart)       # Direct engine for decision tree application
library(caret)       # For resampling and model training
library(ipred)       # For fitting bagged decision trees
library(ranger)      # A c++ implementation of random forest 
library(gbm)         # For original implementation of regular and stochastic GBMs
library(cluster)     # For general clustering algorithms
library(factoextra)  # For visualizing cluster results

# Model interpretability packages
library(vip)      # For feature importance
library(corrplot) # For correlation charts


# DATA ########################################################################
leaves <- read.csv(file = '/home/walter/DSSC/I/ML/exam/data/leaf.csv')

## EXPLORING THE DATA #########################################################
dim(leaves) # 339 rows (leaves samples) x 16 columns (the first indicates the 
            # leaf species of the sample, the second the sample number and the
            # others are features related to the shape and form of the sample)
head(leaves$X1) # Response variable: each number is associated to a leaf 
                # species

# Notice that in the data given are only present 30 out of 40 of the species
# indicated in the article: the ones from 1 to 15 and from 22 to 36 that
# should exhibit simple leaves
leaves.names <- c('Quercus_suber', 'Salix_atrocinera', 'Populus_nigra', 
                  'Alnus_sp.', 'Quercus_robur', 'Crataegus_monogyna', 
                  'Ilex_aquifolium', 'Nerium_oleander', 'Betula_pubescens',
                  'Tilia_tomentosa', 'Acer_palmatum', 'Celtis_sp.',
                  'Corylus_avellana', 'Castanea_sativa', 'Populus_alba',
                  'Primula_vulgaris', 'Erodium_sp.', 'Bougainvillea_sp.',
                  'Arisarum_vulgare', 'Euonymus_japonicus', 
                  'Ilex_perado_ssp._azorica', 'Magnolia_soulangeana', 
                  'Buxus_sempervirens', 'Urtica_dioica', 'Podocarpus_sp.', 
                  'Acca_sellowiana', 'Hydrangea_sp.', 'Pseudosasa_japonica',
                  'Magnolia_grandiflora', 'Geranium_sp.')
leaves$X1 <- factor(leaves$X1) 
levels(leaves$X1) <- leaves.names

# We have 14 features
features <- c('Species', 'Specimen_Number', 'Eccentricity', 'Aspect_Ratio', 
              'Elongation', 'Solidity', 'Stochastic_Convexity', 
              'Isoperimetric_Factor', 'Maximal_Indentation_Depth', 'Lobedness', 
              'Average_Intensity', 'Average_Contrast', 'Smoothness', 
              'Third_moment', 'Uniformity', 'Entropy')
names(leaves) <- features

# Dropping the specimen number 
leaves <- leaves %>% select(-Specimen_Number)

# Getting the number of specimens for each species
# n.specimens <- leaves %>% group_by(Species) %>% summarise(n_rows = length(Species))
# names(n.specimens) <- c('Species', 'Original')

## DATA SPLITTING #############################################################
# To provide an accurate understanding of the generalizability of our final 
# optimal model, we can split our data into training and test data sets:
#  - training set: these data are used to develop feature sets, train our
#    algorithms, tune hyperparameters, compare models, and all of the other 
#    activities required to choose a final model
#  - test set: having chosen a final model, these data are used to estimate an 
#    unbiased assessment of the model’s performance, which we refer to as the
#    generalization error.

### SIMPLE RANDOM SPLITTING ###################################################
# The simplest way to split the data into training and test sets is to take a 
# simple random sample: we simply split the data into training and test sets
# (roughly 80% and 20% of the total data respectively since we have few samples
# for each leaf)

# Using rsample package
# set.seed(1)  # For reproducibility
# help(createDataPartition) # Helpful documentation: the function can create also
                          # bootstrap samples or a cross-validation split
# index <- createDataPartition(leaves$Species, p = 0.8, list = FALSE)
# train <- leaves[index, ]
# test <- leaves[-index, ]
# n.train <- train %>% group_by(Species) %>% summarise(n_rows = length(Species))
# names(n.train) <- c('Species', 'Simple_sampling')
# n.test <- test %>% group_by(Species) %>% summarise(n_rows = length(Species))

### RESAMPLING METHODS ########################################################
# Resampling methods provide an alternative approach by allowing us to 
# repeatedly fit a model of interest to parts of the training data and test its
# performance on other parts
# The two most commonly used resampling methods include k-fold cross validation
# and bootstrapping

#### k-FOLD CROSS VALIDATION ##################################################
# k-fold cross-validation (aka k-fold CV) randomly divides the training data
# into k groups (folds) of approximately equal size
# The model is fit on k−1 folds and then the remaining one is used to compute 
# model performance
# This procedure is repeated k times, each time, a different fold is treated as 
# the validation set 
# This process results in k estimates of the generalization error
# The k-fold CV estimate is computed by averaging the k test errors,providing
# an approximation of the error we might expect on unseen data

# Using rsample package
# cv.folds <- vfold_cv(leaves, v = 4)
# train.first.fold <- leaves[cv.folds$splits[[1]]$in_id, ]
# test.first.fold <- leaves[-cv.folds$splits[[1]]$in_id, ]
# n.train.first.fold <- train.first.fold %>% group_by(Species) %>% 
#                       summarise(n_rows = length(Species))
# names(n.train.first.fold) <- c('Species', 'Fist_fold')
# n.test.first.fold <- test.first.fold %>% group_by(Species) %>% 
#                     summarise(n_rows = length(Species))

# first.join <- merge(n.specimens, n.train, by = 'Species')
# count <- merge(first.join, n.train.first.fold, by = 'Species')
# leaves <- leaves[, -2]

#### BOOTSTRAPPING ############################################################
# A bootstrap sample is a random sample of the data taken with replacement 
# This means that, after a data point is selected for inclusion in the subset, 
# it’s still available for further selection
# A bootstrap sample is the same size as the original data set from which it 
# was constructed
# Furthermore, bootstrap sampling will contain approximately the same 
# distribution of values as the original data set

# Using rsample package
# bootstrap_samples <- bootstraps(leaves, times = 5)

### PUTTING THINGS TOGETHER ###################################################
# First we break our data into training and test data while ensuring we have 
# consistent distributions between the training and test sets

# Sampling with the rsample package
set.seed(1234) # For reproducibility
leaves.split <- initial_split(leaves, prop = 0.8)
leaves.train <- training(leaves.split)
leaves.test <- testing(leaves.split)

# Next, we apply a k-nearest neighbor regressor to our data
# To do so, we’ll use caret, which is a meta-engine to simplify the 
# resampling, grid search, and model application processes
# The following defines:
#  - resampling method: we use 4-fold CV repeated 10 times.
#  - grid search: we specify the hyperparameter values to assess (k = 1, 3,... 
#    , 10)
#  - Model training & Validation: we train a k-nearest neighbor (method = "knn")
#    model using our pre-specified resampling procedure (trControl = cv), 
#    grid search (tuneGrid = hyper_grid), and preferred loss function
#    (metric = "Accuracy").

# Specify resampling strategy
# cv <- trainControl(
#  method = "repeatedcv", 
#  number = 4,
#  repeats = 10
# )

# Create grid of hyperparameter values
# hyper.grid <- expand.grid(k = seq(1, 10, by = 2))

# Tune a knn model using grid search
# knn_fit <- train(
#   Species ~ ., 
#   data = leaves.train, 
#   method = "knn", 
#   trControl = cv, 
#   tuneGrid = hyper.grid,
#   metric = "Accuracy"
# )

# ggplot(knn_fit)

## NUMERIC FEATURESS ENGINEERING ##############################################
# Numeric features can create a host of problems for certain models when their 
# distributions are skewed, contain outliers, or have a wide range in 
# magnitudes

### SKEWNESS ##################################################################
# Parametric models that have distributional assumptions can benefit from 
# minimizing the skewness of numeric features
# When normalizing many variables, it’s best to use the Box-Cox (when feature 
# values are strictly positive) or Yeo-Johnson (when feature values are not 
# strictly positive) procedures as these methods will identify if a
# transformation is required and what the optimal transformation will be
# Non-parametric models are rarely affected by skewed features, however, 
# normalizing features will not have a negative effect on these models’ 
# performance
# For example, normalizing features will only shift the optimal split points 
# in tree-based algorithms
# Consequently, when in doubt, normalize

# Normalize all numeric columns
# recipe(Species ~ ., data = leaves) %>%
#  step_BoxCox(all_numeric())  

### STANDARDIZATION ###########################################################
# We must also consider the scale on which the individual features are measured
# What are the largest and smallest values across all features and do they span 
# several orders of magnitude? Models that incorporate smooth functions of 
# input features are sensitive to the scale of the inputs
# For these models and modeling components, it is often a good idea to 
# standardize the features
# Standardizing features includes centering and scaling so that numeric
# variables have zero mean and unit variance, which provides a common 
# comparable unit of measure across all the variables
# You should standardize your variables within the recipe blueprint so that
# both training and test data standardization are based on the same mean and 
# variance
# This helps to minimize data leakage

# std.leaves <- leaves
# for(i in 3:ncol(leaves)) {
#   std.leaves[, i] <- scale(leaves[, i])
# }

# Classic palette BuPu, with 4 colors
# colors <- brewer.pal(10, "PuOr") 

# Add more colors to this palette
# colors <- colorRampPalette(colors)(30)

# names(colors) <- levels(leaves$Species)
# colors.scale <- scale_colour_manual(name = "Species", values = colors)

# std.features.plot <- ggplot(std.leaves, aes(Maximal_Indentation_Depth, Solidity, 
#                       colour = Species)) + geom_point() + colors.scale
# features.plot <- ggplot(leaves, aes(Maximal_Indentation_Depth, Solidity, 
#                        colour = Species)) + geom_point() + colors.scale

# grid.arrange(features.plot, std.features.plot, ncol = 2)

## DIMENSION REDUCTION ########################################################
# In many data analyses and modeling projects we end up with hundreds or even
# thousands of collected features
# From a practical perspective, a model with more features often becomes 
# harder to interpret and is costly to compute
# Some models are more resistant to non-informative predictors
# Dimension reduction is an alternative approach to filter out non-informative 
# features without manually removing them
# For example, we may wish to reduce the dimension of our features with 
# principal components analysis and retain the number of components required to
# explain, say, 95% of the variance and use these components as features in 
# downstream modeling

# recipe(Species ~ ., data = leaves.train) %>%
#   step_center(all_numeric()) %>%
#   step_scale(all_numeric()) %>%
#   step_pca(all_numeric(), threshold = .95)

# k-NEAREST NEIGHBORS #########################################################
# k-nearest neighbor (kNN) is a very simple algorithm in which each observation
# is predicted based on its “similarity” to other observations
# kNN is a memory-based algorithm and cannot be summarized by a closed-form 
# model
# This means the training samples are required at run-time and predictions are 
# made directly from the sample relationships

## PRE-PROCESSING #############################################################
# Due to the squaring, the Euclidean distance is more sensitive to outliers
# Furthermore, most distance measures are sensitive to the scale of features
# Data with features that have different scales will bias the distance measures
# as those predictors with the largest values will contribute most to the
# distance between two samples

# std.leaves.split <- initial_split(std.leaves, prop = 0.8)
# std.leaves.train <- training(std.leaves.split)
# std.leaves.test <- testing(std.leaves.split)

## CHOOSING k #################################################################
# The performance of KNNs is very sensitive to the choice of k
# When k = 1, we base our prediction on a single observation that has the closest
# distance measure
# When k = n, we are using the average (regression) or most common class
# (classification) across all training samples as our predicted value

# Create a resampling method
cv <- trainControl(
  method = "repeatedcv", 
  number = 5, 
  repeats = 10
)

# Create a hyperparameter grid search
# When using KNN for classification, it is best to assess odd numbers for k 
# to avoid ties in the event there is equal proportion of response levels
hyper.grid <- expand.grid(
  k = seq(1, 25, by = 2)
)

start.time <- Sys.time()

knn.leaves.train <- train(
  Species ~ ., 
  data = leaves.train,
  method = "knn", 
  trControl = cv, 
  tuneGrid = hyper.grid,
  preProc = c("center", "scale"), # Preprocessing
  metric = "Accuracy"
)

end.time <- Sys.time()
knn.time.taken <- end.time - start.time
knn.time.taken

ggplot(knn.leaves.train)

knn.predict <- predict(knn.leaves.train, newdata = leaves.test)
mean(knn.predict == leaves.test$Species) # Almost as predicted

knn.leaves <- train(
  Species ~ ., 
  data = leaves,
  method = "knn", 
  trControl = cv, 
  tuneGrid = hyper.grid,
  preProc = c("center", "scale"), # Preprocessing
  metric = "Accuracy"
)

ggplot(knn.leaves)

# Feature importance for kNN is computed by finding the features with the
# smallest distance measure
# Since the response variable is multiclass, the variable importance scores
# below sort the features by maximum importance across the classes
var.importance <- varImp(knn.leaves)
var.importance

# DECISION TREES ##############################################################
# There are many methodologies for constructing decision trees but the most
# well-known is the classification and regression tree (CART) 
# The tree is a set of rules that allows us to make predictions by asking 
# simple yes-or-no questions about each feature
# For classification, predicted probabilities can be obtained using the 
# proportion of each class within the subgroup

# Using only the training data
tree.leaves.train <- train(
  Species ~ .,
  data = leaves.train,
  method = "rpart",
  trControl = cv,
  tuneLength = 20 #  20 different values alpha parameter 
)

tree.predict <- predict(tree.leaves.train, newdata = leaves.test)
ggplot(tree.leaves.train)
mean(tree.predict == leaves.test$Species)

# Using all the data with cross validation results
start.time <- Sys.time()

tree.leaves <- train(
  Species ~ .,
  data = leaves,
  method = "rpart",
  trControl = cv,
  tuneLength = 20 #  20 different values alpha parameter 
)

end.time <- Sys.time()

tree.time.taken <- end.time - start.time
tree.time.taken

ggplot(tree.leaves)
best.tree.error <- 1 - max(tree.leaves$results$Accuracy)

## FEATURE INTERPRETATION ##################################################### 
# To measure feature importance, the reduction in the loss function attributed 
# to each variable at each split is tabulated
# In some instances, a single variable could be used multiple times in a tree,
# consequently, the total reduction in the loss function across all splits by a
# variable are summed up and used as the total feature importance
# When using caret, these values are standardized so that the most important
# feature has a value of 100 and the remaining features are scored based on
# their relative reduction in the loss function
# Also, since there may be candidate variables that are important but are not 
# used in a split, the top competing variables are also tabulated at each split
# Decision trees perform automated feature selection where uninformative 
# features are not used in the model
# We can see this the bottom feature in the plot have zero importance

vip(tree.leaves,  num_features = 14, geom = "point")

# BAGGING #####################################################################
# Bootstrap aggregating (bagging) prediction models is a general method for 
# fitting multiple versions of a prediction model and then combining (or
# ensembling) them into an aggregated prediction
# A benefit to creating ensembles via bagging, which is based on resampling 
# with replacement, is that it can provide its own internal estimate of 
# predictive performance with the out-of-bag (OOB) sample 
# The OOB sample can be used to test predictive performance and the results
# usually compare well compared to k-fold CV assuming your data set is 
# sufficiently large (say n≥1,000)

bag.error <- c()

for (i in 10:200) {
  bag.leaves.i <- bagging(
    formula = Species ~ .,
    data = leaves,
    nbagg = i,  
    coob = TRUE,
    control = rpart.control(minsplit = 2, cp = 0)
  )
  
  bag.error[i - 9] <- bag.leaves.i$err
}

plot(10:200, bag.error, type = 'l', cex = 0.8, ylim = c(0.2, 0.6), 
     xlim = c(10, 200))
abline(h = best.tree.error, lty = 'dashed')
abline(h = mean(bag.error[seq(from = 100, to = 190, by = 1)]), lty = 'dashed', 
       col = 'red')

bag.leaves.train <- bagging(
  formula = Species ~ .,
  data = leaves.train,
  nbagg = 200,  
  coob = TRUE,
  control = rpart.control(minsplit = 2, cp = 0)
)

1 - bag.leaves.train$err
bag.predict <- predict(bag.leaves.train, newdata = leaves.test)
mean(bag.predict == leaves.test$Species)

# Using all the data
start.time <- Sys.time()

bag.leaves <- bagging(
  formula = Species ~ .,
  data = leaves,
  nbagg = 200,  
  coob = TRUE,
  control = rpart.control(minsplit = 2, cp = 0)
)

end.time <- Sys.time()
bag.time.taken <- end.time - start.time
bag.time.taken
1 - bag.leaves$err

## CROSS-VALIDATION ###########################################################
# We can also apply bagging within caret and use 5-fold CV to see how well our 
# ensemble will generalize
# We see that the cross-validated accuracy for 200 trees is similar to the OOB
# estimate 
# However, using the OOB error took 12 seconds to compute whereas performing 
# the following 10-fold CV took roughly 48 seconds on my machine

bag.leaves.cv.train <- train(
  Species ~ .,
  data = leaves.train,
  method = "treebag",
  trControl = trainControl(method = "cv", number = 4),
  nbagg = 200,  
  control = rpart.control(minsplit = 2, cp = 0)
)

bag.leaves.cv.train
bag.predict.cv <- predict(bag.leaves.cv.train, newdata = leaves.test)
mean(bag.predict == leaves.test$Species)

# Using all the data
start.time <- Sys.time()

bag.leaves.cv <- train(
  Species ~ .,
  data = leaves,
  method = "treebag",
  trControl = trainControl(method = "cv", number = 4),
  nbagg = 200,  
  control = rpart.control(minsplit = 2, cp = 0)
)

end.time <- Sys.time()
bag.cv.time.taken <- end.time - start.time
bag.cv.time.taken
bag.leaves.cv

## EASILY PARALLELIZE #########################################################
# Bagging can become computationally intense as the number of iterations 
# increases
# Fortunately, the process of bagging involves fitting models to each of the 
# bootstrap samples which are completely independent of one another
# This means that each model can be trained in parallel and the results 
# aggregated in the end for the final model
# Consequently, if you have access to a large cluster or number of cores, you
# can more quickly create bagged ensembles on larger data sets
# The following illustrates parallelizing the bagging algorithm (with b = 160
# decision trees) on the data using four cores and returning the predictions
# for the test data for each of the trees

# Binds ans sums data.frames
bind.sum <- function(x, y) {
  tmp <- bind_rows(
    x %>% select_if(is.numeric) %>% tibble::rownames_to_column(), 
    y %>% select_if(is.numeric) %>% tibble::rownames_to_column()
  )
  
  tmp$rowname <- as.numeric(tmp$rowname)
  tmp <- tmp %>%
    group_by(rowname) %>% 
    summarise_all(sum, na.rm = T)  %>% 
    bind_cols(leaves %>% select_if(is.factor))
  
  return(tmp)
}

# Create a parallel socket cluster
cl <- makeCluster(8) # Use 4 workers
registerDoParallel(cl) # Register the parallel backend

# Fit trees in parallel and compute predictions on the test set

predictions <- foreach(
  icount(200), # B = 160 decision trees
  .packages = c("rpart", "dplyr", "tidyr"),
  .combine = "cbind"
) %dopar% {
  # Bootstrap copy of training data
  flag <- 1
  while (length(flag)) {
    index <- sample(nrow(leaves.train), replace = TRUE)
    leaves.train.boot <- leaves.train[index, ]  
    flag <- which(!(leaves.test$Species %in% leaves.train.boot$Species))
  }

  # Fit tree to bootstrap copy
  bagged.tree <- rpart(
    Species ~ ., 
    control = rpart.control(minsplit = 2, cp = 0),
    data = leaves.train.boot
  ) 
  
  tmp <- predict(bagged.tree, newdata = leaves.test) 
  index <- which(tmp == 1, arr.ind = TRUE)
  tmp[index] <- colnames(tmp)[index[,"col"]]
  tmp <- data.frame(tmp)
  tmp[tmp == 0] <- NA  
  tmp <- tmp %>% unite(data = tmp, col = ., remove = TRUE, na.rm = TRUE)
}

# Shutdown parallel cluster
stopCluster(cl)

predictions <- apply(predictions, 1, function(x) names(which.max(table(x))))
mean(predictions == leaves.test$Species)

# Using all the data
cl <- makeCluster(8) # Use 4 workers
registerDoParallel(cl) # Register the parallel backend

# Fit trees in parallel and compute predictions on the test set
start.time <- Sys.time()

OOB <- foreach(
  icount(200), # B = 160 decision trees
  .packages = c("rpart", "dplyr", "tidyr"),
  .combine = "+"
) %dopar% {
  # Bootstrap copy of training data
  flag <- 1
  while (length(flag)) {
    index <- sample(rep(1:nrow(leaves), 2), nrow(leaves))
    leaves.train.boot <- leaves[index, ]  
    leaves.test.boot <- leaves[-unique(index), ] 
    flag <- which(!(leaves.test.boot$Species %in% leaves.train.boot$Species))
  }
  
  # Fit tree to bootstrap copy
  bagged.tree <- rpart(
    Species ~ ., 
    control = rpart.control(minsplit = 2, cp = 0),
    data = leaves.train.boot
  ) 
  
  tmp <- predict(bagged.tree, newdata = leaves.test.boot) 
  elem <- which(tmp == 1, arr.ind = TRUE)
  tmp[elem] <- colnames(tmp)[elem[,"col"]]
  tmp <- data.frame(tmp)
  tmp[tmp == 0] <- NA  
  tmp <- tmp %>% unite(data = tmp, col = ., remove = TRUE, na.rm = TRUE)
  mean(tmp[,1] == leaves.test.boot$Species)
}

# Shutdown parallel cluster
stopCluster(cl)

end.time <- Sys.time()
paralle.bag.time.taken <- end.time - start.time
paralle.bag.time.taken
OOB / 200

# Alternative sampling
rm(index)
index <- sample(rep(1:nrow(leaves), 2), nrow(leaves))
leaves.train.boot <- leaves[index, ]  
leaves.test.boot <- leaves[-unique(index), ] 
nrow(leaves.test.boot)
nrow(leaves.train.boot)

## FEATURE INTERPRETATION #####################################################
# For bagged decision trees, this process is similar. For each tree, we compute 
# the sum of the reduction of the loss function across all splits
# We then aggregate this measure across all trees for each feature
# The features with the largest average decrease are considered most important

vip::vip(bag.leaves.cv, num_features = 14, geom = "point")

## CORRELATION MATRIX #########################################################
# Now that we have an idea of what are the important variables let's try to
# plot a correlation chart to see if we can drop some features when we try
# to model the system

# mat is a matrix of data
# ... : further arguments to pass to the native R cor.test function
cor.mtest <- function(mat, ...) {
  mat <- as.matrix(mat)
  n <- ncol(mat)
  p.mat<- matrix(NA, n, n)
  diag(p.mat) <- 0
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      tmp <- cor.test(mat[, i], mat[, j], ...)
      p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
    }
  }
  colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
  p.mat
}

# Matrix of the p-value of the correlation
M <- cor(leaves %>% select(-Species))

col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
p.mat <- cor.mtest(leaves %>% select(-Species))
corrplot(M, method = "color", col = col(200), diag = FALSE, type = "upper",
         order = "hclust",  addCoef.col = "black", tl.col = "black",
         tl.srt = 45)

# I think we can drop the following, that are not the most important ones
# from the analysis and are also highly correlated between each other and
# with other variables as well
M <- cor(leaves %>% select(-Species, -Smoothness, -Average_Contrast,
                           -Maximal_Indentation_Depth, -Average_Intensity))

p.mat <- cor.mtest(leaves %>% select(-Species, -Average_Contrast,
                                     -Maximal_Indentation_Depth,
                                     -Average_Intensity, -Smoothness))
corrplot(M, method = "color", col = col(200), diag = FALSE, type = "upper",
         order = "hclust",  addCoef.col = "black", tl.col = "black",
         tl.srt = 45)

# There are still a couple of significant correlation but the situation looks
# overall much better than before, let's try to learn again the bagged trees
# only with this features and see if the situation improves

bag.leaves.train.drop <- bagging(
  formula = Species ~ .,
  data = leaves.train %>% select(-Smoothness, -Maximal_Indentation_Depth,
                                 -Average_Contrast, -Average_Intensity),
  nbagg = 200,  
  coob = TRUE,
  control = rpart.control(minsplit = 2, cp = 0)
)

1 - bag.leaves.train.drop$err
bag.predict.drop <- predict(bag.leaves.train, newdata = leaves.test)
mean(bag.predict == leaves.test$Species)

env <- foreach:::.foreachGlobals
rm(list = ls(name = env), pos = env)

# Using all the data
start.time <- Sys.time()

bag.leaves.cv.drop <- train(
  Species ~ .,
  data = leaves %>% select(-Smoothness, -Maximal_Indentation_Depth,
                           -Average_Contrast, -Average_Intensity),
  method = "treebag",
  trControl = trainControl(method = "cv", number = 4),
  nbagg = 200,  
  control = rpart.control(minsplit = 2, cp = 0)
)

end.time <- Sys.time()
bag.cv.drop.time.taken <- end.time - start.time
bag.cv.drop.time.taken
bag.leaves.cv.drop

# Yes! It seems improved a little bit overall, and we're also much faster

# RANDOM FOREST ###############################################################
# Random forests are built using the same fundamental principles as decision 
# trees and bagging 
# Random forests help to reduce tree correlation by injecting more randomness 
# into the tree-growing process.29 More specifically, while growing a decision
# tree during the bagging process, random forests perform split-variable 
# randomization where each time a split is to be performed, the search for the 
# split variable is limited to a random subset of of the original features
# When rhe subset is equal to all the set we have bagging

n_features <- length(setdiff(names(leaves), "Species"))

# Train a default random forest model
rf.leaves.train <- ranger(
  Species ~ ., 
  data = leaves.train,
  mtry = floor(sqrt(n_features)), # Number of features considered
  respect.unordered.factors = "order",
  seed = 123
)

# Get OOB accuracy
1 - rf.leaves.train$prediction.error
rf.predict <- predict(rf.leaves, leaves.test)
mean(rf.predict$predictions == leaves.test$Species)

# Using all the data
rf.leaves <- ranger(
  Species ~ ., 
  data = leaves,
  mtry = floor(sqrt(n_features)), # Number of features considered: 3
  respect.unordered.factors = "order",
  seed = 123
)

default_acc <- 1 - rf.leaves$prediction.error
default_acc
## TUNING STRATEGIES
# We have various parameters to tune in a RF:
#  - number of trees: number of trees within the random forest, it need to be 
#    sufficiently large to stabilize the error, a good starting point is 10 
#    times the number of features, however, as we adjust other hyperparameters
#    more or less may be required
#  - number of features to consider: controls the split-variable randomization 
#    feature of random forests helps to balance low tree correlation witj
#    reasonable predictive strength, a good starting points for classification 
#    is the square root of the number of features
#  - tree complexity: node size is probably the most common hyperparameter to
#    control tree complexity and most implementations use the default value is
#    of one for classifications these value tend to produce good results
#  - sampling scheme: The default sampling scheme for random forests is 
#    bootstrapping where 100% of the observations are sampled with replacement
#    however, we can adjust both the sample size and whether to sample with or 
#    without replacement
#  - split rule

hyper_grid <- expand.grid(
  mtry = c(2, 3, 4, 5, 6),
  min.node.size = c(1, 2, 3, 4, 5, 7, 8), 
  replace = c(TRUE, FALSE),                               
  sample.fraction = c(.5, .6, .7, .8, .9, 1),
  acc = NA
)

# Execute full cartesian grid search
for(i in seq_len(nrow(hyper_grid))) {
  # Fit model for ith hyperparameter combination
  fit <- ranger(
    formula         = Species ~ ., 
    data            = leaves, 
    num.trees       = n_features * 10,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$min.node.size[i],
    replace         = hyper_grid$replace[i],
    sample.fraction = hyper_grid$sample.fraction[i],
    verbose         = FALSE,
    seed            = 123,
    respect.unordered.factors = 'order',
  )
  # Exports OOB error 
  hyper_grid$acc[i] <- 1 - fit$prediction.error
}

# Assess top 10 models
hyper_grid %>%
  arrange(desc(acc)) %>%
  mutate(perc_gain = (acc - default_acc) * 100) %>%
  head(10)

# Discarding highly correlated elements we obtain a slight improvement

## FEATURE INTERPRETATION #####################################################
rf.impurity <- ranger(
  formula = Species ~ ., 
  data = leaves %>% select(-Smoothness, -Maximal_Indentation_Depth,
                           -Average_Contrast, -Average_Intensity), 
  num.trees = 200,
  mtry = 5,
  min.node.size = 3,
  sample.fraction = 0.7,
  replace = TRUE,
  importance = "impurity",
  respect.unordered.factors = "order",
  verbose = FALSE,
  seed  = 123
)

# re-run model with permutation-based variable importance
rf.permutation <- ranger(
  formula = Species ~ ., 
  data = leaves %>% select(-Smoothness, -Maximal_Indentation_Depth,
                           -Average_Contrast, -Average_Intensity), 
  num.trees = 200,
  mtry = 5,
  min.node.size = 3,
  sample.fraction = 0.7,
  replace = TRUE,
  importance = "permutation",
  respect.unordered.factors = "order",
  verbose = FALSE,
  seed  = 123
)

var.imp.impurity <- vip::vip(rf.impurity, num_features = 14, geom = "point")
var.imp.permutation <- vip::vip(rf.permutation, num_features = 14, 
                                geom = "point")

gridExtra::grid.arrange(var.imp.impurity, var.imp.permutation, nrow = 1)

# GRADIENT BOOSTING ###########################################################
# Whereas random forests build an ensemble of deep independent trees, GBMs 
# build an ensemble of shallow trees in sequence with each tree learning and 
# improving on the previous one

# Run a basic GBM model
gbm.grid <-  expand.grid(interaction.depth = 3, 
                        n.trees = 500, 
                        shrinkage = 0.1,
                        n.minobsinnode = 10)

tc = trainControl( ## 10-fold CV
  method = "cv",
  # repeats = 5,
  number = 5)

gbm.leaves = train(Species ~.,
                   data = leaves.train, 
                   method = "gbm", 
                   trControl = tc, 
                   tuneGrid = gbm.grid)

gbm.predict = predict(gbm.leaves, leaves.test)
mean(gbm.predict == leaves.test$Species)
gbm.leaves$results$Accuracy

## PARAMETERS #################################################################
# A simple GBM model contains two categories of hyperparameters: boosting 
# hyperparameters and tree-specific hyperparameters
# The two main boosting hyperparameters include: 
#  - number of trees: The total number of trees in the sequence or ensemble, 
#    each tree is grown in sequence to fix up the past tree’s mistakes,  GBMs 
#    often require many trees but since they can easily overfit we must find 
#    the optimal number of trees that minimize the loss function with CV
#  - learning rate: Determines the contribution of each tree on the final 
#    outcome and controls how quickly the algorithm proceeds down the gradient 
#    descent (learns), values range from 0 – 1 with typical values between
#    0.001 – 0.3
# The two main tree hyperparameters in a simple GBM model include:
# - tree depth: Controls the depth of the individual trees, typical values 
#   range from a depth of 3–8 but it is not uncommon to see a tree depth of 1
# - minimum number of observations in terminal nodes: also, controls the 
#   complexity of each tree. Since we tend to use shorter trees this rarely has
#   a large impact on performance. Typical values range from 5 – 15 where
#   higher values help prevent a model from learning relationships which might
#   be highly specific to the particular sample

# Create grid search
hyper.grid <- expand.grid(
  learning.rate = c(0.2, 0.01, 0.001),
  trees = c(200, 500, 1000),
  interaction.depth = c(1, 3, 5),
  n.minobsinnode = c(10, 20),
  acc = NA
)
# Execute grid search
hyper.grid$interaction.depth[i]
for(i in seq_len(nrow(hyper.grid))) {
  # Fit gbm
  gbm.grid <- expand.grid(interaction.depth = hyper.grid$interaction.depth[i], 
                          n.trees = hyper.grid$trees[i], 
                          shrinkage = hyper.grid$learning.rate[i],
                          n.minobsinnode = hyper.grid$n.minobsinnode[i])
  
  tc = trainControl( ## 10-fold CV
    method = "cv",
    # repeats = 5,
    number = 5)
  
  a <- train(Species ~.,
             data = leaves.train, 
             method = "gbm", 
             trControl = tc, 
             tuneGrid = gbm.grid)
  
  
  hyper.grid$acc[i]  <- m$results$Accuracy

}

# results
head(arrange(hyper.grid, acc))

# SUPPORT VECTOR MACHINE ######################################################
# Support vector machines (SVMs) offer a direct approach to binary 
# classification: try to find a hyperplane in some feature space that “best” 
# separates the two classes
# In practice, however, it is difficult (if not impossible) to find a 
# hyperplane to perfectly separate the classes using just the original features
# SVMs overcome this by extending the idea of finding a separating hyperplane 
# in two ways: 
#  (1) loosen what we mean by “perfectly separates”
#  (2) use the so-called kernel trick to enlarge the feature space to the point
#      that perfect separation of classes is (more) likely

# k-MEANS CLUSTERING ##########################################################
# k-means clustering is one of the most commonly used clustering algorithms for 
# partitioning observations into a set of k groups (clusters), where k is
# pre-specified by the analyst
# k-means, like other clustering algorithms, tries to classify observations 
# into mutually exclusive groups, such that observations within the same cluster 
# are as similar as possible (i.e., high intra-class similarity), whereas 
# observations from different clusters are as dissimilar as possible (i.e., 
# low inter-class similarity)
# In k-means clustering, each cluster is represented by its center (i.e, 
# centroid) which corresponds to the mean of the observation values assigned to 
# the cluster
# The procedure used to find these clusters is similar to the k-nearest 
# neighbor (KNN) algorithm; albeit, without the need to predict an average
# response value

## DISTANCES ##################################################################
# The classification of observations into groups requires some method for
# computing the distance or the (dis)similarity between each pair of
# observations which form a distance or dissimilarity or matrix
# There are many approaches to calculating these distances; the choice of 
# distance measure is a critical step in clustering

## CLUSTERING SPECIES #########################################################
# Let’s perform k-means clustering on the features and see if we can identify 
# unique clusters of species without using the response variable
# Here, we declare k = 30 only because we already know there are 30 unique
# species represented in the data
# We also use 30 random starts (nstart = 30)

std.leaves <- leaves
for(i in 3:ncol(leaves)) {
  std.leaves[, i] <- scale(leaves[, i])
}

features <- std.leaves %>% select(-Species)

# Use k-means model with 10 centers and 10 random starts
kmeans.leaves <- kmeans(features, centers = 30, nstart = 30)
kmeans.leaves$cluster
# Print contents of the model output
str(kmeans.leaves)

# Assesting the results
# Create mode function
kmeans.comparison <- data.frame(
  kmeans.leaves$cluster,
  leaves %>% select(Species)
) 

names(kmeans.comparison) <- c('cluster', 'actual')

cluster.species <- kmeans.comparison %>%
  group_by(cluster) %>%
  arrange(cluster) %>%
  summarize (actual = names(which.max(table(actual))))

mode <- cluster.species$actual[kmeans.comparison$cluster]

mean(mode == kmeans.comparison$actual)

## PLOTTING ###################################################################
# Extract cluster centers
kmeans.centers <- kmeans.leaves$centers

colors <- brewer.pal(10, "PuOr") 
colors <- colorRampPalette(colors)(30)
names(colors) <- levels(leaves$Species)
colors.scale <- scale_colour_manual(name = "Species", values = colors)

features.plot <- ggplot(leaves, aes(Entropy, Solidity, 
                        colour = Species)) + geom_point() + colors.scale

features.plot
fviz_cluster(object = kmeans.leaves, data = leaves, 
             choose.vars = c("Entropy", "Solidity"), geom = c("point", "text"))

# HIERARCHICAL CLUSTERING #####################################################
# Hierarchical clustering is an alternative approach to k-means clustering for 
# identifying groups in a data set. In contrast to k-means, hierarchical
# clustering will create a hierarchy of clusters and therefore does not require 
# us to pre-specify the number of clusters
# Furthermore, hierarchical clustering has an added advantage over k-means
# clustering in that its results can be easily visualized using an attractive
# tree-based representation called a dendrogram.

# AGGLOMERATIVE HIERARCHICAL CLUSTERING #######################################
# Dissimilarity matrix
d <- dist(leaves %>% select(-Species), method = "euclidean")

# Hierarchical clustering using Complete Linkage
hc.leaves <- hclust(d, method = "complete" )

hc.leaves <- agnes(leaves %>% select(-Species), method = "complete")

help(agnes)
m <- c( "average", "single", "complete", "ward")
names(m) <- c( "average", "single", "complete", "ward")

ac <- function(x) {
  agnes(leaves %>% select(-Species), method = x)$ac
}

purrr::map_dbl(m, ac)

dend_plot <- fviz_dend(hc.leaves)
dend_data <- attr(dend_plot, "dendrogram")
dend_cuts <- cut(dend_data, h = 30)
fviz_dend(dend_cuts$lower[[2]])

# In order to identify sub-groups (i.e., clusters), we can cut the dendrogram 
# with cutree()
sub_grp <- cutree(hc.leaves, k = 30)

# We can plot the entire dendrogram with fviz_dend and highlight the thirty
# clusters with k = 30
fviz_dend(
  hc.leaves,
  k = 30,
  horiz = TRUE,
  rect = TRUE,
  rect_fill = TRUE,
  rect_border = "jco",
  k_colors = "jco",
  cex = 0.1
)

# However, due to the size of the Ames housing data, the dendrogram is not very 
# legible. Consequently, we may want to zoom into one particular region or
# cluster
# This allows us to see which observations are most similar within a particular
# group

dend_plot <- fviz_dend(hc.leaves) # create full dendogram
dend_data <- attr(dend_plot, "dendrogram") # extract plot info
dend_cuts <- cut(dend_data, h = 30) # cut the dendogram atdesignated height
# Create sub dendrogram plots
p1 <- fviz_dend(dend_cuts$lower[[1]])
p2 <- fviz_dend(dend_cuts$lower[[1]], type = 'circular')

# Side by side plots
gridExtra::grid.arrange(p1, p2, nrow = 1)