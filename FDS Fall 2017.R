
# Libraries and data collection
library(tidyverse)
train = read.csv("train.csv")
test = read.csv("test.csv")


#Separating out the Id column for merging it after prediction is done
#Here multiple copies of same variable is maintained for ease of verification and validation of various regression types
testResultLMF = test[1]
testResultLM = test[1]
testResultXB = test[1]
testResultRF = test[1]
testStacking = test[1]

### Removing some Outliers based on plotting patterns identified
# Plotting 'GrLivArea' against SalePrice too see if there are any outliers
#ggplot(train,aes(y=SalePrice,x=GrLivArea))+geom_point()
train <- train[-which(train$GrLivArea > 4000 & train$SalePrice < 300000),]
#ggplot(train,aes(y=SalePrice,x=GarageArea))+geom_point()
train <- train[train$GarageArea < 1200,]

#Since we dont need sale price in the normalization, take a copy of it
SalePrice = train$SalePrice

#Removing the Id and Dependent variables before scalling features
train$SalePrice <- NULL
train$Id <- NULL
test$Id <- NULL

# Data cleaning

#Combining train and test data
full = rbind(train, test)

#Separating integer and string features

#Handling integer features
fullInt <- full %>% select_if(is.numeric)
columns <- colnames(fullInt)

for (colm in columns){
  fullInt[[colm]] <- ifelse(is.na(fullInt[[colm]]), ave(fullInt[[colm]], FUN = function(x) mean(x, na.rm = TRUE)), fullInt[[colm]])
}

#Handling string columns
allCols <- colnames(full)
intCols <- colnames(fullInt)
strCols <- setdiff(allCols, intCols)

fullStrings <- subset.data.frame(full, select = strCols)

# Categorization
# remove columns which are having NA's morethan 80%
columns <- colnames(fullStrings)

totRows = nrow(fullStrings)

for (colm in columns) {
  if (sum(is.na(fullStrings[[colm]]))/totRows >= 0.80){
    fullStrings[[colm]] <- NULL
  }
}

# Normalising the string features
columnsStr <- colnames(fullStrings)
for (colm in columnsStr) {
  fullStrings[[colm]] = c(fullStrings[[colm]])
}

# Replace NA with 0
m <- as.matrix(fullStrings)
y <- which(is.na(m)==TRUE)
m[y] <- 0
fullStrings <- data.frame(m)
print(sum(is.na(fullStrings)))


# After the feature normalization split both test and train data
finalFull <- cbind(fullInt, fullStrings)
finalTrain <- finalFull[1:1454,]
finalTest <- finalFull[1455:2913,]
finalTrain <- cbind(finalTrain, SalePrice)


#Building regression models start here

#Training the model using Multiple linear regression
reg <- lm(finalTrain$SalePrice ~ ., data = finalTrain)
summary(reg)

#try to make our prediction
# @TODO:Handle the warning
pred <- predict(reg, newdata = finalTest)
finalTest$SalePrice <- pred
testResultLM$SalePrice <- finalTest$SalePrice
write.csv(testResultLM, file = 'mypred_LM.csv', row.names = F)

#Score - 0.16012

# Using XGBoost
#install.packages('xgboost')
library(xgboost)

classifier = xgboost(data = as.matrix(finalTrain[-76]), 
                     label = finalTrain$SalePrice,
                     max_depth = 35,
                     subsample = 0.8,
                     eval_metric = "rmse", 
                     eta = 0.8,
                     nthread = 10,
                     nfold = 11, 
                     min_child_weight = 1.9,
                     colsample_bytree = 0.6,
                     nrounds=1600)
summary(classifier)

pred <- predict(classifier, newdata = as.matrix(finalTest))
finalTest$SalePrice <- pred

testResultXB$SalePrice <- finalTest$SalePrice

write.csv(testResultXB, file = 'mypred_XB.csv', row.names = F)

# Score - 0.22064

### Using Random forest

library(randomForest)

model <- randomForest(finalTrain$SalePrice ~.,data = finalTrain, method = "anova", ntree = 20,
                      mtry = 26,replace = F,nodesize = 1,importance = T)

pred <- predict(model, newdata = finalTest)
finalTest$SalePrice <- pred
testResultRF$SalePrice <- finalTest$SalePrice
write.csv(testResultRF, file = 'mypred_RF.csv', row.names = F)

#Score - 0.15444

# Stacking models starts here

# Combining both LM and RF
testStacking$SalePrice = (testResultLM$SalePrice + testResultRF$SalePrice)/2 
write.csv(testStacking, file = 'pred.csv', row.names = F)

#Score - 0.13561



