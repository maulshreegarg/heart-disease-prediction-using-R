#importing dataset
data <- read.csv("heart.csv")
install.packages("GGally") 
install.packages("reshape2")
install.packages("plotly")
install.packages("ggcorrplot")
#importing library
library(reshape2)
library(caret)
library(e1071)
library(randomForest)
library(naivebayes)
library(rpart)
library(class)
library(xgboost)
library(lda)
library(MASS)
library(ggplot2)
library(plotly)
library(ggcorrplot)
library(GGally)
library(dplyr)
library(tidyr)

#data preparation
dim(data)
str(data)
summary(data)
is.na(data)
data %>% distinct()
infinite_columns <- sapply(data, function(col) any(is.infinite(col)))
clean_data <- data[!apply(data[infinite_columns], 1, anyNA), ]
data <- clean_data

#data cleaning and transformation
data <- na.omit(data)
data$Sex <- c(F=1 , M=0)[data$Sex]
data$ChestPainType <- c(ATA=0, NAP=1, ASY= 2)[data$ChestPainType]
data$RestingECG <- c(Normal=0, ST=1)[data$RestingECG]
data$ExerciseAngina <- c(N=0, Y=1)[data$ExerciseAngina]
data$ST_Slope <- c(Up=0, Flat=1, Down=2)[data$ST_Slope]
data$ChestPainType[is.na(data$ChestPainType)] <- median(data$ChestPainType, na.rm = TRUE)
data$Cholesterol[is.na(data$Cholesterol)] <- median(data$Cholesterol, na.rm = TRUE)
data$RestingECG[is.na(data$RestingECG)] <- median(data$RestingECG, na.rm = TRUE)

#finding correlation 
corr_mat <- round(cor(data),2)
# reduce the size of correlation matrix
melted_corr_mat <- melt(corr_mat)
head(melted_corr_mat)
# plotting the correlation heatmap
ggplot(data = melted_corr_mat, aes(x=Var1, y=Var2,fill=value)) + geom_tile()

# Select the six features from the dataset
selected_features <- data %>%
  select("Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "MaxHR")
# Calculate the correlation matrix for the selected features
correlation_matrix <- cor(selected_features)
# Create a ggplot heatmap
heatmap_plot <- ggplot(data = as.data.frame(as.table(correlation_matrix)), aes(Var1, Var2)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0, limits = c(-1, 1)) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Correlation Heatmap of Selected Features",
       x = "Features",
       y = "Features")
# Print the heatmap plot
print(heatmap_plot)

# Create a pair plot using GGally
ggpairs(selected_features , title = "Pair Plot of Heart Data")

#Count plot
ggplot(data, aes(x = HeartDisease)) + geom_bar()

#the no of healthy and sick pepple in both gender
result <- data %>%
  group_by(Sex,HeartDisease) %>%
  summarize(Count = n())
result
plot_ly(data = data, x = ~Sex, color = ~HeartDisease, type = "histogram", barmode = "group", colors = c('#71AEC2', '#D58989')) %>%
  layout(title = "Distribution")

# Pivot the data using dcast
pivot_data <- dcast(data, Sex ~ RestingECG, value.var = "HeartDisease", fun.aggregate = mean)
# Melt the pivoted data
melted_data <- melt(pivot_data, id.vars = "Sex", variable.name = "RestingECG")
# Create a pie chart for each category
pie_chart_list <- lapply(unique(melted_data$RestingECG), function(category) {
  subset_data <- melted_data[melted_data$RestingECG == category, ]
  
  pie_chart <- ggplot(subset_data, aes(x = "", y = value, fill = Sex)) +
    geom_bar(stat = "identity", width = 1) +
    coord_polar("y", start = 0) +
    labs(title = paste("The effect of RestingECG of both sex relations to the disease"),
         x = NULL, y = NULL) +
    theme_void() +
    facet_wrap(~ RestingECG)
  return(pie_chart)
})
# Print the pie charts
print(pie_chart_list)

# Reshape the data
reshaped_data <- as.data.frame(pivot_data)
reshaped_data$Sex <- rownames(reshaped_data)

# Convert the data to long format
library(reshape2)
long_data <- melt(reshaped_data, id.vars = "Sex")

#checking the Distribution of maximum heart rate achieved by HeartDisease
ggplot(data, aes(x = MaxHR, fill = factor(HeartDisease))) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("#EEDD88", "#88CCEE")) +  # Set custom colors
  labs(title = "Distribution of maximum heart rate achieved by HeartDisease")

#checking Distribution of resting blood pressure by disease
ggplot(data, aes(x = as.factor(HeartDisease), y = RestingBP, color = as.factor(HeartDisease))) +
  geom_jitter() +
  labs(title = "Distribution of resting blood pressure by disease") +
  scale_color_manual(values = c("#E41A1C", "#377EB8"))


#effect of fasting blood sugar on disease
res <- data %>%
  group_by(FastingBS,HeartDisease) %>%
  summarize(Count = n())
res
plot_ly(data = data, x = ~FastingBS, color = ~HeartDisease, type = "histogram", barmode = "group", colors = c('#71AEC2', '#D58989')) %>%
  layout(title = "Fasting blood sugar on disease")

#the Average cholesterol for both sexes in relation to the disease
re <- aggregate(Cholesterol ~ HeartDisease + Sex, data = data, FUN = mean)
pivot_result <- reshape(re, idvar = "HeartDisease", timevar = "Sex", direction = "wide")
re

#checking the average disease for each age group 
data$age_cat <- cut(data$Age, breaks = 4)
res1 <- data %>%
  group_by(age_cat) %>%
  summarise(mean_HeartDisease = mean(HeartDisease)) %>%
  arrange(desc(mean_HeartDisease))
res1
data$HeartDisease <- factor(data$HeartDisease)
ggplot(data, aes(x = HeartDisease, fill = HeartDisease)) +
  geom_bar(stat = "count") +
  facet_wrap(~ age_cat, ncol = 1) +
  labs(title = "Distribution of age category by disease")

#checking the average disease for each Cholesterol group
data$Chol_cat=cut(data$Cholesterol,breaks=3)
data$HeartDisease <- as.numeric(as.character(data$HeartDisease))
res2 <- data %>%
  filter(is.numeric(HeartDisease)) %>%
  group_by(Chol_cat) %>%
  summarise(mean_HeartDisease = mean(HeartDisease, na.rm = TRUE)) %>%
  arrange(desc(mean_HeartDisease))
res2
plot_ly(data, x = ~HeartDisease, color = ~HeartDisease) %>%
  add_histogram(
    facet_col = ~Chol_cat,
    barmode = 'group',
    marginal = 'rug'
  ) %>%
  layout(
    title = 'Distribution of Cholesterol categories by disease'
  )

#checking the average disease for each maximum heart rate achieved group
data$MaxHR_cat=cut(data$MaxHR,breaks = 3)
res3 <- data %>%
  group_by(MaxHR_cat) %>%
  summarise(mean_HeartDisease = mean(HeartDisease)) %>%
  arrange(desc(mean_HeartDisease))
res3
data$HeartDisease <- factor(data$HeartDisease)
ggplot(data, aes(x = HeartDisease, fill = HeartDisease)) +
  geom_bar(stat = "count") +
  facet_grid(. ~ MaxHR_cat) +
  labs(title = "Distribution of maximum heart rate achieved by disease") 

#checking What is the average disease for each Oldpeak group
data$Oldpeak_cat=cut(data$Oldpeak,breaks = 2)
data$HeartDisease <- as.numeric(as.character(data$HeartDisease))
res4 <- data %>%
  group_by(Oldpeak_cat) %>%
  summarise(mean_HeartDisease = mean(HeartDisease)) %>%
  arrange(desc(mean_HeartDisease))
res4
plot_ly(data, x = ~HeartDisease, color = ~HeartDisease, type = "histogram") %>%
  layout(barmode = "group", title = "Distribution of Heart Disease")

data  <- subset(data, select = -c(age_cat, Chol_cat, MaxHR_cat, Oldpeak_cat))

#feature selection
predictor <- data[,-c(12)]
correlation <- abs(cor(predictor,data$HeartDisease,use = "everything",
                   method = c("pearson", "kendall", "spearman")))
correlation <- na.omit(correlation)
correlation_filter <- correlation[, 1] > 0.4
features <- names(predictor)[correlation_filter]
features

#splitting training and testing data
set.seed(123)
trainIndex <- createDataPartition(data$HeartDisease, p = 0.7, list = FALSE)
train_data <- data[trainIndex,]
test_data <- data[-trainIndex,]

#logistic regression model
log_model <- glm(HeartDisease ~ ., data=train_data,family = binomial ,model = TRUE, method = "glm.fit")
log_predictions <- predict(log_model,test_data,ype = "response")
log_predictions <- ifelse(is.na(log_predictions), 0, log_predictions) 
log_predictions <- ifelse(log_predictions > 0.5, 1, 0)
plot(test_data$HeartDisease,type="l",col="red")
lines(log_predictions)
log_conf_matrix <- table(test_data$HeartDisease,log_predictions)
log_conf_matrix

#evaluating logistic regression model
log_accuracy <- sum(diag(log_conf_matrix))/sum(log_conf_matrix)
log_accuracy
log_precision <- log_conf_matrix[2,2]/(log_conf_matrix[2,1]+log_conf_matrix[2,2])
log_precision
log_recall <- log_conf_matrix[2,2]/(log_conf_matrix[2,2]+log_conf_matrix[1,2])
log_recall
log_specifi <- log_conf_matrix[1,1]/(log_conf_matrix[1,1]+log_conf_matrix[2,1])
log_specifi
log_f1_score <- 2 * ((log_precision * log_recall) / (log_precision + log_recall))
log_f1_score

#naive bayes model
nb_model <- naiveBayes(HeartDisease ~ ., data = train_data)
nb_predictions <- predict(nb_model, newdata = test_data)
plot(test_data$HeartDisease,type="l",col="red")
lines(nb_predictions)
nb_conf_matrix <- table(test_data$HeartDisease,nb_predictions)
nb_conf_matrix

#evaluating naive bayes model
nb_accuracy <- sum(diag(nb_conf_matrix))/sum(nb_conf_matrix)
nb_accuracy
nb_precision <- nb_conf_matrix[2,2]/(nb_conf_matrix[2,1]+nb_conf_matrix[2,2])
nb_precision
nb_recall <- nb_conf_matrix[2,2]/(nb_conf_matrix[2,2]+nb_conf_matrix[1,2])
nb_recall
nb_specifi <- nb_conf_matrix[1,1]/(nb_conf_matrix[1,1]+nb_conf_matrix[2,1])
nb_specifi
nb_f1_score <- 2 * ((nb_precision * nb_recall) / (nb_precision + nb_recall))
nb_f1_score

#support vector machine  
svm_models <- svm(HeartDisease ~ ., data = train_data)
svm_predictions <- predict(svm_models, newdata = test_data)
svm_predictions <- ifelse(svm_predictions>0.5,1,0)
plot(test_data$HeartDisease,type="l",col="red")
lines(svm_predictions)
order_vec1 <- order(test_data$HeartDisease)
ordered_vec2 <- svm_predictions[order_vec1]
svm_conf_matrix <- table(order_vec1,ordered_vec2)
svm_conf_matrix

#evaluating svm model
svm_accuracy <- sum(diag(svm_conf_matrix))/sum(svm_conf_matrix)
svm_accuracy
svm_precision <- svm_conf_matrix[2,2]/(svm_conf_matrix[2,1]+svm_conf_matrix[2,2])
svm_precision
svm_recall <- svm_conf_matrix[2,2]/(svm_conf_matrix[2,2]+svm_conf_matrix[1,2])
svm_recall
svm_specifi <- svm_conf_matrix[1,1]/(svm_conf_matrix[1,1]+svm_conf_matrix[2,1])
svm_specifi
svm_f1_score <- 2 * ((svm_precision * svm_recall) / (svm_precision + svm_recall))
svm_f1_score

#decision tree
dt_model <- rpart(HeartDisease ~ ., data = train_data, method = "class")
dt_predictions <- predict(dt_model,newdata = test_data, type="class")
plot(test_data$HeartDisease,type="l",col="red")
lines(dt_predictions)
dt_conf_matrix <- table(test_data$HeartDisease,dt_predictions)
dt_conf_matrix

#evaluating decision tree model
dt_accuracy <- sum(diag(dt_conf_matrix))/sum(dt_conf_matrix)
dt_accuracy
dt_precision <- dt_conf_matrix[2,2]/(dt_conf_matrix[2,1]+dt_conf_matrix[2,2])
dt_precision
dt_recall <- dt_conf_matrix[2,2]/(dt_conf_matrix[2,2]+dt_conf_matrix[1,2])
dt_recall
dt_specifi <- dt_conf_matrix[1,1]/(dt_conf_matrix[1,1]+dt_conf_matrix[2,1])
dt_specifi
dt_f1_score <- 2 * ((dt_precision * dt_recall) / (dt_precision + dt_recall))
dt_f1_score

#Check for null values in each column
null_columns <- colSums(is.na(data))
print(null_columns)

#random forest
rf_model <- randomForest(HeartDisease ~ .,data=train_data, na.action = na.roughfix)
rf_predictions <- predict(rf_model, newdata = test_data)
rf_predictions <- ifelse(rf_predictions>0.5,1,0)
plot(test_data$HeartDisease,type="l",col="red")
lines(rf_predictions)
rf_conf_matrix <- table(test_data$HeartDisease,rf_predictions)
rf_conf_matrix

#evaluating random forest
rf_accuracy <- sum(diag(rf_conf_matrix))/sum(rf_conf_matrix)
rf_accuracy
rf_precision <- rf_conf_matrix[2,2]/(rf_conf_matrix[2,1]+rf_conf_matrix[2,2])
rf_precision
rf_recall <- rf_conf_matrix[2,2]/(rf_conf_matrix[2,2]+rf_conf_matrix[1,2])
rf_recall
rf_specifi <- rf_conf_matrix[1,1]/(rf_conf_matrix[1,1]+rf_conf_matrix[2,1])
rf_specifi
rf_f1_score <- 2 * ((rf_precision * rf_recall) / (rf_precision + rf_recall))
rf_f1_score

#knn
knn_model <- knn(train_data[, -1], test_data[, -1], train_data$HeartDisease, k = 2)
knn_predictions <- as.numeric(knn_model)
knn_predictions <- ifelse(knn_predictions==2,1,0)
knn_predictions
plot(test_data$HeartDisease,type="l",col="red")
lines(knn_predictions)
knn_conf_matrix <- table(test_data$HeartDisease,knn_predictions)
knn_conf_matrix

#evaluating knn model
knn_accuracy <- sum(diag(knn_conf_matrix))/sum(knn_conf_matrix)
knn_accuracy
knn_precision <- knn_conf_matrix[2,2]/(knn_conf_matrix[2,1]+knn_conf_matrix[2,2])
knn_precision
knn_recall <- knn_conf_matrix[2,2]/(knn_conf_matrix[2,2]+knn_conf_matrix[1,2])
knn_recall
knn_specifi <- knn_conf_matrix[1,1]/(knn_conf_matrix[1,1]+knn_conf_matrix[2,1])
knn_specifi
knn_f1_score <- 2 * ((knn_precision * knn_recall) / (knn_precision + knn_recall))
knn_f1_score

# Linear Discriminant Analysis (LDA)
lda_model <- lda(HeartDisease ~ ., data = train_data)
lda_predictions <- predict(lda_model, newdata = test_data)$class
plot(test_data$HeartDisease,type="l",col="red")
lines(lda_predictions)
lda_conf_matrix <- table(test_data$HeartDisease,lda_predictions)
lda_conf_matrix

#evaluating lda model
lda_accuracy <- sum(diag(lda_conf_matrix))/sum(lda_conf_matrix)
lda_accuracy
lda_precision <- lda_conf_matrix[2,2]/(lda_conf_matrix[2,1]+lda_conf_matrix[2,2])
lda_precision
lda_recall <- lda_conf_matrix[2,2]/(lda_conf_matrix[2,2]+lda_conf_matrix[1,2])
lda_recall
lda_specifi <- lda_conf_matrix[1,1]/(lda_conf_matrix[1,1]+lda_conf_matrix[2,1])
lda_specifi
lda_f1_score <- 2 * ((lda_precision * lda_recall) / (lda_precision + lda_recall))
lda_f1_score

# Quadratic Discriminant Analysis (QDA)
qda_model <- qda(HeartDisease ~ ., data = train_data)
qda_predictions <- predict(qda_model, newdata = test_data)$class
plot(test_data$HeartDisease,type="l",col="red")
lines(qda_predictions)
qda_conf_matrix <- table(test_data$HeartDisease,qda_predictions)
qda_conf_matrix

#evaluating qda matrix
qda_accuracy <- sum(diag(qda_conf_matrix))/sum(qda_conf_matrix)
qda_accuracy
qda_precision <- qda_conf_matrix[2,2]/(qda_conf_matrix[2,1]+qda_conf_matrix[2,2])
qda_precision
qda_recall <- qda_conf_matrix[2,2]/(qda_conf_matrix[2,2]+qda_conf_matrix[1,2])
qda_recall
qda_specifi <- qda_conf_matrix[1,1]/(qda_conf_matrix[1,1]+qda_conf_matrix[2,1])
qda_specifi
qda_f1_score <- 2 * ((qda_precision * qda_recall) / (qda_precision + qda_recall))
qda_f1_score

#data visualization
#Accuracy
model_names <- c("logistic regression", "naive bayes", "svm", "decision tree", "random forest","knn", "lda", "qda")
accuracies <- c(log_accuracy,nb_accuracy, svm_accuracy, dt_accuracy, rf_accuracy, knn_accuracy, lda_accuracy, qda_accuracy)
accuracy_data <- data.frame(Model = model_names, Accuracy = accuracies)

ggplot(accuracy_data, aes(x = Model, y = Accuracy)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Model Accuracies", x = "Model", y = "Accuracy")

#Precision
model_names <- c("logistic regression", "naive bayes", "svm", "decision tree", "random forest","knn", "lda", "qda")
precision <- c(log_precision,nb_precision, svm_precision, dt_precision, rf_precision, knn_precision, lda_precision, qda_precision)
precision_data <- data.frame(Model = model_names, Precision = precision)

ggplot(precision_data, aes(x = Model, y = Precision)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Model precision", x = "Model", y = "precision")

#recall
model_names <- c("logistic regression", "naive bayes", "svm", "decision tree", "random forest","knn", "lda", "qda")
recall <- c(log_recall,nb_recall, svm_recall, dt_recall, rf_recall, knn_recall, lda_recall, qda_recall)
recall_data <- data.frame(Model = model_names, Recall = recall)

ggplot(recall_data, aes(x = Model, y =  Recall)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Model recall", x = "Model", y = " Recall")

#specificity
model_names <- c("logistic regression", "naive bayes", "svm", "decision tree", "random forest","knn", "lda", "qda")
specificity <- c(log_specifi,nb_specifi, svm_specifi, dt_specifi, rf_specifi, knn_specifi, lda_specifi, qda_specifi)
specificity_data <- data.frame(Model = model_names, Specificity  = specificity )

ggplot(specificity_data, aes(x = Model, y =  Specificity )) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Model specificity", x = "Model", y = "specificity")

#f1 score
model_names <- c("logistic regression", "naive bayes", "svm", "decision tree", "random forest","knn", "lda", "qda")
f1_score <- c(log_f1_score,nb_f1_score, svm_f1_score, dt_f1_score, rf_f1_score, knn_f1_score, lda_f1_score, qda_f1_score)
f1_score_data <- data.frame(Model = model_names, F1_score  = f1_score )

ggplot(f1_score_data, aes(x = Model, y =  f1_score )) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Model f1_score", x = "Model", y = "f1_score")


#features
train_data_features <- subset(train_data, select = c( "ChestPainType", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope","HeartDisease"))
test_data_features <- subset(test_data,select = c("ChestPainType", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope","HeartDisease"))

#logistic regression model with selected features
log_model_f <- glm(HeartDisease ~ ., data = train_data_features,family = binomial ,model = TRUE, method = "glm.fit")
log_predictions_f <- predict(log_model_f,test_data_features,type = "response")
log_predictions_f <- ifelse(is.na(log_predictions_f), 0, log_predictions_f) 
log_predictions_f <- ifelse(log_predictions_f > 0.5, 1, 0)
log_conf_matrix_f <- table(test_data_features$HeartDisease,log_predictions_f)
log_conf_matrix_f

#evaluating logistic regression model
log_accuracy_f <- sum(diag(log_conf_matrix_f))/sum(log_conf_matrix_f)
log_accuracy_f
log_precision_f <- log_conf_matrix_f[2,2]/(log_conf_matrix_f[2,1]+log_conf_matrix_f[2,2])
log_precision_f
log_recall_f <- log_conf_matrix_f[2,2]/(log_conf_matrix_f[2,2]+log_conf_matrix_f[1,2])
log_recall_f
log_specifi_f <- log_conf_matrix_f[1,1]/(log_conf_matrix_f[1,1]+log_conf_matrix_f[2,1])
log_specifi_f
log_f1_score_f <- 2 * ((log_precision_f * log_recall_f) / (log_precision_f + log_recall_f))
log_f1_score_f

#naive bayes model features
nb_model_f <- naiveBayes(HeartDisease ~ ., data = train_data_features)
nb_predictions_f <- predict(nb_model_f, newdata = test_data_features)
nb_conf_matrix_f <- table(test_data_features$HeartDisease,nb_predictions_f)
nb_conf_matrix_f

#evaluating naive bayes model
nb_accuracy_f <- sum(diag(nb_conf_matrix_f))/sum(nb_conf_matrix_f)
nb_accuracy_f
nb_precision_f <- nb_conf_matrix_f[2,2]/(nb_conf_matrix_f[2,1]+nb_conf_matrix_f[2,2])
nb_precision_f
nb_recall_f <- nb_conf_matrix_f[2,2]/(nb_conf_matrix_f[2,2]+nb_conf_matrix_f[1,2])
nb_recall_f
nb_specifi_f <- nb_conf_matrix_f[1,1]/(nb_conf_matrix_f[1,1]+nb_conf_matrix_f[2,1])
nb_specifi_f
nb_f1_score_f <- 2 * ((nb_precision_f * nb_recall_f) / (nb_precision_f + nb_recall_f))
nb_f1_score_f

#support vector machine with features
svm_models_f <- svm(HeartDisease ~ ., data = train_data_features)
svm_predictions_f <- predict(svm_models_f, newdata = test_data_features)
svm_predictions_f <- ifelse(svm_predictions_f>0.5,1,0)
svm_conf_matrix_f <- table(test_data_features$HeartDisease,svm_predictions_f)
svm_conf_matrix_f

#evaluating svm model
svm_accuracy_f <- sum(diag(svm_conf_matrix_f))/sum(svm_conf_matrix_f)
svm_accuracy_f
svm_precision_f <- svm_conf_matrix_f[2,2]/(svm_conf_matrix_f[2,1]+svm_conf_matrix_f[2,2])
svm_precision_f
svm_recall_f <- svm_conf_matrix_f[2,2]/(svm_conf_matrix_f[2,2]+svm_conf_matrix_f[1,2])
svm_recall_f
svm_specifi_f <- svm_conf_matrix_f[1,1]/(svm_conf_matrix_f[1,1]+svm_conf_matrix_f[2,1])
svm_specifi_f
svm_f1_score_f <- 2 * ((svm_precision_f * svm_recall_f) / (svm_precision_f + svm_recall_f))
svm_f1_score_f


#decision tree
dt_model_f <- rpart(HeartDisease ~ ., data = train_data_features, method = "class")
dt_predictions_f <- predict(dt_model_f,newdata = test_data_features, type="class")
dt_predictions_f
dt_conf_matrix_f <- table(test_data_features$HeartDisease,dt_predictions_f)
dt_conf_matrix_f

#evaluating decision tree model
dt_accuracy_f <- sum(diag(dt_conf_matrix_f))/sum(dt_conf_matrix_f)
dt_accuracy_f
dt_precision_f <- dt_conf_matrix_f[2,2]/(dt_conf_matrix_f[2,1]+dt_conf_matrix_f[2,2])
dt_precision_f
dt_recall_f <- dt_conf_matrix_f[2,2]/(dt_conf_matrix_f[2,2]+dt_conf_matrix_f[1,2])
dt_recall_f
dt_specifi_f <- dt_conf_matrix_f[1,1]/(dt_conf_matrix_f[1,1]+dt_conf_matrix_f[2,1])
dt_specifi_f
dt_f1_score_f <- 2 * ((dt_precision_f * dt_recall_f) / (dt_precision_f + dt_recall_f))
dt_f1_score_f

#random forest with features
rf_model_f <- randomForest(HeartDisease ~ .,data=train_data_features)
rf_predictions_f <- predict(rf_model_f, newdata = test_data_features)
rf_predictions_f <- ifelse(rf_predictions_f>0.5,1,0)
rf_conf_matrix_f <- table(test_data_features$HeartDisease,rf_predictions_f)
rf_conf_matrix_f

#evaluating random forest
rf_accuracy_f <- sum(diag(rf_conf_matrix_f))/sum(rf_conf_matrix_f)
rf_accuracy_f
rf_precision_f <- rf_conf_matrix_f[2,2]/(rf_conf_matrix_f[2,1]+rf_conf_matrix_f[2,2])
rf_precision_f
rf_recall_f <- rf_conf_matrix_f[2,2]/(rf_conf_matrix_f[2,2]+rf_conf_matrix_f[1,2])
rf_recall_f
rf_specifi_f <- rf_conf_matrix_f[1,1]/(rf_conf_matrix_f[1,1]+rf_conf_matrix_f[2,1])
rf_specifi_f
rf_f1_score_f <- 2 * ((rf_precision_f * rf_recall_f) / (rf_precision_f + rf_recall_f))
rf_f1_score_f

#knn
knn_model_f <- knn(train_data_features[, -1], test_data_features[, -1], train_data_features$HeartDisease, k = 2)
knn_predictions_f <- as.numeric(knn_model_f)
knn_predictions_f <- ifelse(knn_predictions_f==2,1,0)
knn_predictions_f
knn_conf_matrix_f <- table(test_data_features$HeartDisease,knn_predictions_f)
knn_conf_matrix_f

#evaluating knn model
knn_accuracy_f <- sum(diag(knn_conf_matrix_f))/sum(knn_conf_matrix_f)
knn_accuracy_f
knn_precision_f <- knn_conf_matrix_f[2,2]/(knn_conf_matrix_f[2,1]+knn_conf_matrix_f[2,2])
knn_precision_f
knn_recall_f <- knn_conf_matrix_f[2,2]/(knn_conf_matrix_f[2,2]+knn_conf_matrix_f[1,2])
knn_recall_f
knn_specifi_f <- knn_conf_matrix_f[1,1]/(knn_conf_matrix_f[1,1]+knn_conf_matrix_f[2,1])
knn_specifi_f
knn_f1_score_f <- 2 * ((knn_precision_f * knn_recall_f) / (knn_precision_f + knn_recall_f))
knn_f1_score_f

# Linear Discriminant Analysis (LDA) with selected features
lda_model_f <- lda(HeartDisease ~ ., data = train_data_features)
lda_predictions_f <- predict(lda_model_f, newdata = test_data_features)$class
lda_conf_matrix_f <- table(test_data_features$HeartDisease,lda_predictions_f)
lda_conf_matrix_f

#evaluating lda model
lda_accuracy_f <- sum(diag(lda_conf_matrix_f))/sum(lda_conf_matrix_f)
lda_accuracy_f
lda_precision_f <- lda_conf_matrix_f[2,2]/(lda_conf_matrix_f[2,1]+lda_conf_matrix_f[2,2])
lda_precision_f
lda_recall_f <- lda_conf_matrix_f[2,2]/(lda_conf_matrix_f[2,2]+lda_conf_matrix_f[1,2])
lda_recall_f
lda_specifi_f <- lda_conf_matrix_f[1,1]/(lda_conf_matrix_f[1,1]+lda_conf_matrix_f[2,1])
lda_specifi_f
lda_f1_score_f <- 2 * ((lda_precision_f * lda_recall_f) / (lda_precision_f + lda_recall_f))
lda_f1_score_f

# Quadratic Discriminant Analysis (QDA)
qda_model_f <- qda(HeartDisease ~ ., data = train_data_features)
qda_predictions_f <- predict(qda_model_f, newdata = test_data_features)$class
qda_conf_matrixS_f <- table(test_data_features$HeartDisease,qda_predictions_f)
qda_conf_matrix_f

#evaluating qda matrix
qda_accuracy_f <- sum(diag(qda_conf_matrix_f))/sum(qda_conf_matrix_f)
qda_accuracy_f
qda_precision_f <- qda_conf_matrix_f[2,2]/(qda_conf_matrix_f[2,1]+qda_conf_matrix_f[2,2])
qda_precision_f
qda_recall_f <- qda_conf_matrix_f[2,2]/(qda_conf_matrix_f[2,2]+qda_conf_matrix_f[1,2])
qda_recall_f
qda_specifi_f <- qda_conf_matrix_f[1,1]/(qda_conf_matrix_f[1,1]+qda_conf_matrix_f[2,1])
qda_specifi_f
qda_f1_score_f <- 2 * ((qda_precision_f * qda_recall_f) / (qda_precision_f + qda_recall_f))
qda_f1_score_f

#data visualization
#Accuracy
model_names_f <- c("logistic regression", "naive bayes", "svm", "decision tree", "random forest","knn", "lda", "qda")
accuracies_f <- c(log_accuracy_f,nb_accuracy_f, svm_accuracy_f, dt_accuracy_f, rf_accuracy_f, knn_accuracy_f, lda_accuracy_f, qda_accuracy_f)
accuracy_data_f <- data.frame(Model_f = model_names_f, Accuracy_f = accuracies_f)

ggplot(accuracy_data_f, aes(x = Model_f, y = Accuracy_f)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Model Accuracies", x = "Model", y = "Accuracy")

#Precision
precision_f <- c(log_precision_f,nb_precision_f, svm_precision_f, dt_precision_f, rf_precision_f, knn_precision_f, lda_precision_f, qda_precision_f)
precision_data_f <- data.frame(Model = model_names_f, Precision = precision_f)

ggplot(precision_data_f, aes(x = Model, y = Precision)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Model precision", x = "Model", y = "precision")

#recall
recall_f <- c(log_recall_f,nb_recall_f, svm_recall_f, dt_recall_f, rf_recall_f, knn_recall_f, lda_recall_f, qda_recall_f)
recall_data_f <- data.frame(Model = model_names_f, Recall = recall_f)

ggplot(recall_data, aes(x = Model, y =  Recall)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Model recall", x = "Model", y = " Recall")

#specificity
specificity_f <- c(log_specifi_f,nb_specifi_f, svm_specifi_f, dt_specifi_f, rf_specifi_f, knn_specifi_f, lda_specifi_f, qda_specifi_f)
specificity_data_f <- data.frame(Model = model_names_f, Specificity  = specificity_f )

ggplot(specificity_data_f, aes(x = Model, y =  specificity )) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Model specificity", x = "Model", y = "specificity")

#f1 score
f1_score_f <- c(log_f1_score_f,nb_f1_score_f, svm_f1_score_f, dt_f1_score_f, rf_f1_score_f, knn_f1_score_f, lda_f1_score_f, qda_f1_score_f)
f1_score_data_f <- data.frame(Model = model_names_f, F1_score  = f1_score_f )

ggplot(f1_score_data_f, aes(x = Model, y =  F1_score )) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Model f1_score", x = "Model", y = "f1_score")


#evaluating models
log_eval <- sum(log_accuracy,log_precision,log_recall,log_specifi,log_f1_score)
log_eval_f <- sum(log_accuracy_f,log_precision_f,log_recall_f,log_specifi_f,log_f1_score_f)

nb_eval <- sum(nb_accuracy,nb_precision,nb_specifi,nb_recall,nb_f1_score)
nb_eval_f <- sum(nb_accuracy_f,nb_precision_f,nb_specifi_f,nb_recall_f,nb_f1_score_f)

svm_eval <- sum(svm_accuracy,svm_precision,svm_specifi,svm_recall,svm_f1_score)
svm_eval_f <- sum(svm_accuracy_f,svm_precision_f,svm_specifi_f,svm_recall_f,svm_f1_score_f)

dt_eval <- sum(dt_accuracy,dt_precision,dt_specifi,dt_recall,dt_f1_score)
dt_eval_f <- sum(dt_accuracy_f,dt_precision_f,dt_specifi_f,dt_recall_f,dt_f1_score_f)

rf_eval <- sum(rf_accuracy,rf_precision,rf_specifi,rf_recall,rf_f1_score)
rf_eval_f <- sum(rf_accuracy_f,rf_precision_f,rf_specifi_f,rf_recall_f,rf_f1_score_f)

knn_eval <- sum(knn_accuracy,knn_precision,knn_specifi,knn_recall,knn_f1_score)
knn_eval_f <- sum(knn_accuracy_f,knn_precision_f,knn_specifi_f,knn_recall_f,knn_f1_score_f)

lda_eval <- sum(lda_accuracy,lda_precision,lda_specifi,lda_recall,lda_f1_score)
lda_eval_f <- sum(lda_accuracy_f,lda_precision_f,lda_specifi_f,lda_recall_f,lda_f1_score_f)

qda_eval <- sum(qda_accuracy,qda_precision,qda_specifi,qda_recall,qda_f1_score)
qda_eval_f <- sum(qda_accuracy_f,qda_precision_f,qda_specifi_f,qda_recall_f,qda_f1_score_f)

#visualizing
model_names <- c("logistic regression", "naive bayes", "svm", "decision tree", "random forest","knn", "lda", "qda")
visualizing <- c(log_eval,nb_eval, svm_eval, dt_eval, rf_eval, knn_eval, lda_eval, qda_eval)
visual_data <- data.frame(Model = model_names, Eval = visualizing)

ggplot(visual_data, aes(x = Model, y = Eval)) +
  geom_bar(stat = "identity", fill = "pink") +
  labs(title = "Model evaluation", x = "Model", y = "Evaluation")
  
model_names_f <- c("logistic regression", "naive bayes", "svm", "decision tree", "random forest","knn", "lda", "qda")
visualizing_f <- c(log_eval_f,nb_eval_f, svm_eval_f, dt_eval_f, rf_eval_f, knn_eval_f, lda_eval_f, qda_eval_f)
visual_data_f <- data.frame(Model = model_names_f, Eval = visualizing_f)

ggplot(visual_data_f, aes(x = Model, y = Eval)) +
  geom_bar(stat = "identity", fill = "pink") +
  labs(title = "Model evaluation with selected features", x = "Model", y = "Evaluation")

#knn model works really well with selected features : FastingBS, RestingECG, MaxHR, ExerciseAngina
#linear discriminant analysis works well with all the features 