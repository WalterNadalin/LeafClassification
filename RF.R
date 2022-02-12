# LIBRARIES ###################################################################
# Helper packages
library(dplyr)        # For data manipulation
library(tidyr)        # For data manipulation
library(ggplot2)      # For awesome graphics
library(RColorBrewer) # For coloring the points on the plots
library(gridExtra)    # For putting plots side by side
library(doParallel)   # For parallel backend to foreach
library(foreach)      # For parallel processing with for loops

# Modeling process packages
library(rsample)     # For resampling procedures
library(rpart)       # Direct engine for decision tree application
library(caret)       # For resampling and model training
library(ipred)       # For fitting bagged decision trees
library(ranger)      # A c++ implementation of random forest 

# Model interpretability packages
library(vip)      # For feature importance
library(corrplot) # For correlation charts

# DATA ########################################################################
leaves <- read.csv(file = '/home/walter/DSSC/I/ML/exam/data/leaf.csv')

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

leaves <- leaves %>% select(-Specimen_Number)

## DROPPING PARAMETERS ########################################################
s = 30
err <- c()
all.err <- c()
for(i in 1:1) {
   fit <- ranger(
    Species ~ ., 
    #data = leaves, 
    data = leaves %>% select(-Smoothness, -Average_Contrast),
    mtry = 3, # Number of features considered: 3
    respect.unordered.factors = "order",
    num.trees = 140,
    min.node.size = 1,
    sample.fraction = 1,
    seed = s
  )
  
  err[i] <- fit$prediction.error
  
  fit.all <- ranger(
    Species ~ ., 
    data = leaves, 
    mtry = 3, # Number of features considered: 3
    respect.unordered.factors = "order",
    num.trees = 140,
    min.node.size = 1,
    sample.fraction = 1,
    seed = s
  )
  
  all.err[i] <- fit.all$prediction.error
}

sum(err > all.err)
length(err > all.err)

leaves <- leaves %>% select(-Smoothness, -Average_Contrast)

### SIMPLE RANDOM SPLITTING ###################################################

# Using rsample package
set.seed(s) # For reproducibility
leaves.split <- initial_split(leaves, prop = 0.8)
leaves.train <- training(leaves.split)
leaves.test <- testing(leaves.split)

# DECISION TREES ##############################################################

# Create a resampling method
cv <- trainControl(
  method = "repeatedcv", 
  number = 5, 
  repeats = 10
)

# Using only the training data
set.seed = s
tree.leaves.train <- train(
  Species ~ .,
  data = leaves.train,
  method = "rpart",
  trControl = cv,
  tuneLength = 20 #  20 different values alpha parameter 
)

tree.error <- min(1 - tree.leaves.train$results$Accuracy)
tree.predict <- predict(tree.leaves.train, newdata = leaves.test)
mean(tree.predict != leaves.test$Species)

# BAGGING #####################################################################

bag.leaves.train <- bagging(
  formula = Species ~ .,
  data = leaves.train,
  nbagg = 200,  
  coob = TRUE,
  control = rpart.control(minsplit = 2, cp = 0),
  seed = s
)
bag.leaves.train$err
bag.predict <- predict(bag.leaves.train, newdata = leaves.test)
mean(bag.predict != leaves.test$Species)

# For plotting
bag.error <- c()
j <- 1;
for (i in seq(from = 1930, to = 2000, by = 10)) {
  fit  <- bagging(
    formula = Species ~ .,
    data = leaves,
    nbagg = i,  
    coob = TRUE,
    control = rpart.control(minsplit = 2, cp = 0),
    seed = s
  )
  
  bag.error[j] <- fit$err
  j <- j + 1
}

bag.error

## CROSS-VALIDATION ###########################################################
bag.leaves.cv.train <- train(
  seed = s,
  Species ~ .,
  data = leaves.train,
  method = "treebag",
  trControl = cv,
  nbagg = 200,  
  control = rpart.control(minsplit = 2, cp = 0)
)

1 - bag.leaves.cv.train$results$Accuracy
bag.predict.cv <- predict(bag.leaves.cv.train, newdata = leaves.test)
mean(bag.predict != leaves.test$Species)

# RANDOM FOREST ###############################################################

rf.leaves.train <- ranger(
  Species ~ ., 
  data = leaves,
  mtry = 3, # Number of features considered
  respect.unordered.factors = "order",
  seed = 1,
  num.trees = 120,
  replace = TRUE,
  min.node.size = 1,
  sample.fraction = 1,
)

default.err <- rf.leaves.train$prediction.error

## TUNING PARAMETERS ##########################################################
# Size
rf.error <- c()
j <- 1
for (i in seq(from = 10, to = 2000, by = 10)) {
  fit <- ranger(
    formula         = Species ~ ., 
    data = leaves,
    num.trees       = i,
    mtry            = 3,
    min.node.size   = 1,
    replace         = TRUE,
    sample.fraction = 1,
    verbose         = FALSE,
    seed            = 4,
    respect.unordered.factors = 'order',
  )
  # Exports OOB error
  rf.error[j] <- fit$prediction.error
  j <- j + 1
}

plot(seq(from = 10, to = 2000, by = 10), rf.error, type = 'l', col = 'red',
     ylab = 'OOB error estimate', xlab = 'Number of trees')
points(seq(from = 10, to = 2000, by = 10), bag.error, type = 'l', col = 'green')
legend(x = 1300, y = 0.39, legend = c('Bag of trees', 'Random forest'), 
       lty = c(1, 1), col = c('green', 'red'))
# abline(h = tree.error, lty = 'dashed', col = 'red')

# Execute full cartesian grid search
hyper_grid <- expand.grid(
  mtry = seq(from = 1, to = 10),
  min.node.size =seq(from = 1, to = 10),
  num.trees = 1500,
  err = NA
)

for(i in seq_len(nrow(hyper_grid))) {
  # Fit model for ith hyperparameter combination
  fit <- ranger(
    formula         = Species ~ ., 
    data = leaves,
    num.trees       = hyper_grid$num.trees[i],
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$min.node.size[i],
    replace         = TRUE,
    sample.fraction = 1,
    verbose         = FALSE,
    seed            = s,
    respect.unordered.factors = 'order',
  )
  # Exports OOB error 
  hyper_grid$err[i] <- fit$prediction.error
}

# Assess top 10 models
hyper_grid %>%
  arrange(err) %>%
  mutate(perc_gain = (default.err - err)) %>%
  head(10)

# Training the random forst
# Train a default random forest model
rf.leaves.train <- ranger(
  Species ~ ., 
  data = leaves.train,
  mtry = 5, # Number of features considered
  respect.unordered.factors = "order",
  seed = s,
  num.trees = 1500,
  replace = TRUE,
  min.node.size = 1,
  sample.fraction = 1,
)

# Get OOB accuracy
rf.leaves.train$prediction.error
rf.predict <- predict(rf.leaves.train, leaves.test)
mean(rf.predict$predictions != leaves.test$Species)

rf_grid <- expand.grid(mtry = 5,
                       min.node.size = 1,
                       splitrule = "gini")

rf.leaves.cv <- train(
  Species ~ .,
  data = leaves.train,# %>% select(-Smoothness, -Average_Contrast, 
  #             -Average_Intensity),
  method = "ranger",
  trControl = cv,
  tuneGrid = rf_grid,
  num.trees = 1500,
  respect.unordered.factors = "order",
  seed = s,
)

1 - rf.leaves.cv$results$Accuracy
rf.predict <- predict(rf.leaves.cv, leaves.test)
mean(rf.predict != leaves.test$Species)

rf.leaves <- ranger(
  Species ~ ., 
  data = leaves,
  mtry = 5, # Number of features considered
  respect.unordered.factors = "order",
  seed = s,
  num.trees = 1500,
  replace = TRUE,
  min.node.size = 1,
  sample.fraction = 1,
)

rf.leaves$prediction.error
