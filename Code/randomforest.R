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
library(mlbench)
library(tuneRanger)
library(mlr) 


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


## FEATURE SELECTION ##########################################################
names(leaves) <- formatted.features

n_features <- length(setdiff(names(leaves), "Species"))
rf.impurity <- ranger(
  formula = Species ~ ., 
  data = leaves, # %>% select(-Smoothness, -Maximal_Indentation_Depth,
  #            -Average_Contrast, -Average_Intensity), 
  num.trees = 500,
  mtry = floor(sqrt(n_features)),
  min.node.size = 1,
  sample.fraction = 1,
  replace = TRUE,
  importance = "impurity",
  respect.unordered.factors = "order",
  verbose = FALSE,
  seed  = 4
)

# re-run model with permutation-based variable importance
rf.permutation <- ranger(
  formula = Species ~ ., 
  data = leaves, # %>% select(-Smoothness, -Maximal_Indentation_Depth,
  #            -Average_Contrast, -Average_Intensity), 
  num.trees = 500,
  mtry = floor(sqrt(n_features)),
  min.node.size = 1,
  sample.fraction = 1,
  replace = TRUE,
  importance = "permutation",
  respect.unordered.factors = "order",
  verbose = FALSE,
  seed  = 4
)

var.imp.impurity <- vip::vip(rf.impurity, num_features = 14, geom = "point")
var.imp.permutation <- vip::vip(rf.permutation, num_features = 14, 
                                geom = "point")

gridExtra::grid.arrange(var.imp.impurity, var.imp.permutation, nrow = 1)

formatted.features <- c("Species", "Eccentricity", "Aspect.ratio", "Elongation",
                        "Solidity", "St.convexity", 
                        "Is.factor", "Max.ind.depth",
                        "Lobedness", "Av.intensity", "Av.contrast", 
                        "Smoothness",  "Third.moment", "Uniformity", "Entropy") 

names(leaves) <- formatted.features
M <- cor(leaves %>% select(Smoothness, Av.intensity, Av.contrast, Third.moment,
                           Solidity, Is.factor, Max.ind.depth, Lobedness))

x11()
col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
p.mat <- cor.mtest(leaves %>% select(-Species))
# corrplot(M, type="upper", order="hclust", tl.col="black", tl.srt=45, diag = FALSE)
corrplot(M, method = "color", col = col(200), diag = FALSE, type = "upper",
         order = "hclust",  addCoef.col = "black", tl.col = "black",
         tl.srt = 45)

names(leaves)  <- c('Species', 'Eccentricity', 'Aspect_Ratio', 
                    'Elongation', 'Solidity', 'Stochastic_Convexity', 
                    'Isoperimetric_Factor', 'Maximal_Indentation_Depth', 'Lobedness', 
                    'Average_Intensity', 'Average_Contrast', 'Smoothness', 
                    'Third_moment', 'Uniformity', 'Entropy')

# RFE
s <- 4
set.seed(s) # For reproducibility
control <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
# run the RFE algorithm
results <- rfe(leaves %>% select(-Species), leaves$Species, sizes=c(1:14),
               rfeControl = control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type = c("g", "o"))

leaves <- leaves %>% select(Species, predictors(results))

### SIMPLE RANDOM SPLITTING ###################################################
# Using rsample package
set.seed(s) # For reproducibility
leaves.split <- initial_split(leaves, prop = 0.7)
leaves.train <- training(leaves.split)
leaves.test <- testing(leaves.split)

# DECISION TREES ##############################################################

# Create a resampling method
cv <- trainControl(
  method = "cv", 
  number = 5
)

# Using only the training data
set.seed = s
tree.leaves.train <- caret::train(
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
  nbagg = 500,  
  coob = TRUE,
  control = rpart.control(minsplit = 2, cp = 0),
  seed = s
)

bag.leaves.train$err
bag.predict <- predict(bag.leaves.train, newdata = leaves.test)
mean(bag.predict != leaves.test$Species)

## CROSS-VALIDATION ###########################################################
bag.leaves.cv.train <- caret::train(
  Species ~ .,
  data = leaves.train,
  method = "treebag",
  trControl = cv,
  nbagg = 500,  
  control = rpart.control(minsplit = 2, cp = 0),
  seed = s
)

1 - bag.leaves.cv.train$results$Accuracy
bag.predict.cv <- predict(bag.leaves.cv.train, newdata = leaves.test)
mean(bag.predict != leaves.test$Species)

# RANDOM FOREST ###############################################################

## TUNING PARAMETERS ##########################################################
# Size

# A mlr task has to be created in order to use the package
# We make an mlr task with the iris dataset here 
# (Classification task with makeClassifTask, Regression Task with makeRegrTask)
set.seed(s)
leaves.task = makeClassifTask(data = leaves, target = "Species")

# Rough Estimation of the Tuning time
estimateTimeTuneRanger(leaves.task)

# Tuning process (takes around 1 minute); Tuning measure is the multiclass brier score
res = tuneRanger(leaves.task, num.trees = 1500, measure = list(multiclass.brier),
                 num.threads = 8, iters = 100, 
                 parameters = list(replace = TRUE, respect.unordered.factors = "order"),
                 tune.parameters = c("mtry", "min.node.size", "sample.fraction"))


# Mean of best 5 % of the results
res
# Model with the new tuned hyperparameters
res$model
# Restart after failing in one of the iterations:
res = restartTuneRanger("./optpath.RData", leaves.task, measure = list(multiclass.brier))

## TRAINING ###################################################################
# Training the random forst
# Train a default random forest model
m = 11
node.size = 2
fraction = 1
rep = TRUE
rf.leaves.train <- ranger(
  Species ~ ., 
  data = leaves.train,
  mtry = m, # Number of features considered
  respect.unordered.factors = "order",
  seed = s,
  num.trees = 1500,
  replace = rep,
  min.node.size = node.size,
  sample.fraction = fraction,
)

# Get OOB accuracy
rf.leaves.train$prediction.error
rf.predict <- predict(rf.leaves.train, leaves.test)
mean(rf.predict$predictions != leaves.test$Species)

rf_grid <- expand.grid(mtry = m,
                       min.node.size = node.size,
                       splitrule = "gini")

rf.leaves.cv <- caret::train(
  Species ~ .,
  data = leaves.train,
  method = "ranger",
  trControl = cv,
  tuneGrid = rf_grid,
  num.trees = 1500,
  sample.fraction = fraction,
  replace = rep,
  respect.unordered.factors = "order",
  seed = s,
)

1 - rf.leaves.cv$results$Accuracy
rf.predict <- predict(rf.leaves.cv, leaves.test)
mean(rf.predict != leaves.test$Species)

rf.leaves <- ranger(
  Species ~ ., 
  data = leaves,
  mtry = m, # Number of features considered
  respect.unordered.factors = "order",
  seed = s,
  num.trees = 1500,
  replace = rep,
  min.node.size = node.size,
  sample.fraction = fraction,
)

rf.leaves$prediction.error

