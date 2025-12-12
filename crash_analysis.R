install.packages("sqldf")
library(dplyr)
library(readr)
library(sqldf)

#load data
person<- read.csv(file.choose())
accident_event <- read.csv(file.choose())
atmosphere_condition <- read.csv(file.choose())
urban_rural <- read.csv(file.choose())
road_condition <- read.csv(file.choose())


type_of_accident <- sqldf("
      SELECT p.ACCIDENT_NO, p.SEX, p.AGE_GROUP, a.EVENT_TYPE_DESC
      FROM person AS p
      LEFT JOIN accident_event AS a
      ON p.ACCIDENT_NO = a.ACCIDENT_NO
      WHERE p.ROAD_USER_TYPE_DESC = 'Drivers'
      ORDER BY p.AGE_GROUP")

# It is interesting how there are up to 17 age groups where 0–4-year-olds are recorded as driving vehicles and being involved in accidents

sqldf("
      SELECT AGE_GROUP, COUNT(*) AS age_group_count
      FROM person
      WHERE ROAD_USER_TYPE_DESC = 'Drivers'
      GROUP BY AGE_GROUP
      ORDER BY age_group_count DESC")

# according to the result, the age group that are likely to involve in the accident as a drivers are 30-39. Now we only interest in this group age.
# Question: how many of them are male, and female? 

sqldf("
      SELECT AGE_GROUP, SEX, COUNT(SEX) AS sex_count
      FROM person
      WHERE ROAD_USER_TYPE_DESC = 'Drivers' AND AGE_GROUP = '30-39' AND SEX IN ('F', 'M')
      GROUP BY SEX")

# So now we are interested in male drivers who are 30-39. Now we want to find out, what type of accident they tend to involve in. 

sqldf("
      SELECT EVENT_TYPE_DESC, COUNT(*) AS accident_type_count
      FROM type_of_accident
      WHERE SEX = 'M' AND AGE_GROUP = '30-39'
      GROUP BY EVENT_TYPE_DESC
      ORDER BY accident_type_count DESC")

# Most male drivers in the group age 30-29 involve in the Collision the most. We want to find out what influence Collision to happen?
#We want to find out which indicator among these highly influence collision
# ==> a profile table include, helmet_belt_worn, asmotsphere condition, rural or urban area, road surface condition. 

View(type_of_accident)
colnames(urban_rural)

male_driver_3039 <- sqldf("
      SELECT HELMET_BELT_WORN, ATMOSPH_COND_DESC, DEG_URBAN_NAME, SURFACE_COND_DESC, a.EVENT_TYPE_DESC
      FROM person AS p
      LEFT JOIN accident_event AS a ON p.ACCIDENT_NO = a.ACCIDENT_NO
      LEFT JOIN atmosphere_condition AS ac ON p.ACCIDENT_NO = ac.ACCIDENT_NO
      LEFT JOIN urban_rural AS u ON p.ACCIDENT_NO = u.ACCIDENT_NO
      LEFT JOIN road_condition AS r ON p.ACCIDENT_NO = r.ACCIDENT_NO
      WHERE SEX = 'M' AND AGE_GROUP = '30-39' AND ROAD_USER_TYPE_DESC = 'Drivers' AND EVENT_TYPE_DESC NOT IN ('Other', 'Not known')
      AND HELMET_BELT_WORN NOT IN (8,9) AND ATMOSPH_COND_DESC <> 'Not known' AND SURFACE_COND_DESC <> 'Unk.'")

View(male_driver_3039)

#How can we answer which features might influence the likelihood of the collision -> EDA
#Since our independent variables are all categorical variables, we will use Chi-square test. 

male_driver_3039$collision_likelihood <- ifelse(male_driver_3039$EVENT_TYPE_DESC == "Collision", 1, 0)

#1 = Collision, 0 = Non-collision (Other type of accident)
#NUll hypothesis (H0)L: The two variables are independent (no relationship)
#Alternative hypothesis (H1): The two variables are not independent (There is a relationship)
#If p-value <0.05, reject the null hypothesis. 

chisq.test(table(male_driver_3039$collision_likelihood, male_driver_3039$ATMOSPH_COND_DESC))
chisq.test(table(male_driver_3039$collision_likelihood, male_driver_3039$DEG_URBAN_NAME))
chisq.test(table(male_driver_3039$collision_likelihood, male_driver_3039$SURFACE_COND_DESC))
chisq.test(table(male_driver_3039$collision_likelihood, male_driver_3039$HELMET_BELT_WORN))

#All four variables show a statistically significant association with collision likelihood. Therefore, we will treat all of them as important
#indicators to predict the likelihood of collision. The objective is to estimate the probability of collision while considering the combined 
#influence of these environmental and behavioral factors

# 1 Set your desired output file path
# (replace with your own folder path if needed)
output_path <- "male_driver_3039.csv"

# 2 Write the dataset to a CSV file
write.csv(male_driver_3039, output_path, row.names = FALSE)

# 3 Check if file is created
file.exists(output_path)

## =======================
## 0) Libraries
## =======================
#install.packages(c("dplyr","caret","pROC","randomForest","smotefamily","xgboost"))
library(dplyr)
library(caret)
library(pROC)
library(randomForest)
library(smotefamily)
library(xgboost)

## =======================
## 1) Preconditions
## =======================
stopifnot(exists("male_driver_3039"))
if (!"collision_likelihood" %in% names(male_driver_3039)) {
  stop("Column 'collision_likelihood' not found. Create it before running this script.")
}

## =======================
## 2) Data prep
## =======================
male_driver_3039 <- male_driver_3039[complete.cases(male_driver_3039[
  , c("HELMET_BELT_WORN","ATMOSPH_COND_DESC","DEG_URBAN_NAME","SURFACE_COND_DESC")
]), ]

X <- model.matrix(
  collision_likelihood ~ HELMET_BELT_WORN + ATMOSPH_COND_DESC +
    DEG_URBAN_NAME + SURFACE_COND_DESC,
  data = male_driver_3039
)
encoded_df <- as.data.frame(X)
encoded_df <- encoded_df[, setdiff(names(encoded_df), "(Intercept)")]
encoded_df$collision_likelihood <- as.numeric(male_driver_3039$collision_likelihood)

set.seed(123)
idx <- sample(1:nrow(encoded_df), 0.7 * nrow(encoded_df))
train <- encoded_df[idx, ]
test  <- encoded_df[-idx, ]

x_train <- subset(train, select = -collision_likelihood)
y_train_num <- train$collision_likelihood
y_train_fac <- factor(train$collision_likelihood, levels = c(0,1))
x_test <- test[, colnames(x_train), drop = FALSE]
y_test_num <- test$collision_likelihood

## =======================
## 3) Helper functions
## =======================
# ---- Helpers: robust metrics + F1-only threshold tuning ----
safe_auc <- function(y_true, y_score) {
  r <- try(roc(y_true, y_score, quiet = TRUE), silent = TRUE)
  if (inherits(r, "try-error")) return(NA_real_)
  as.numeric(auc(r))
}

metrics_bin <- function(y_true_num, y_prob, thr = 0.5) {
  y_pred <- ifelse(y_prob >= thr, 1, 0)
  tp <- sum(y_pred==1 & y_true_num==1)
  tn <- sum(y_pred==0 & y_true_num==0)
  fp <- sum(y_pred==1 & y_true_num==0)
  fn <- sum(y_pred==0 & y_true_num==1)
  
  acc  <- (tp+tn) / (tp+tn+fp+fn)
  prec <- ifelse(tp+fp > 0, tp/(tp+fp), 0)
  rec  <- ifelse(tp+fn > 0, tp/(tp+fn), 0)
  spec <- ifelse(tn+fp > 0, tn/(tn+fp), 0)
  f1   <- ifelse(prec+rec > 0, 2*prec*rec/(prec+rec), 0)
  aucv <- safe_auc(y_true_num, y_prob)
  
  c(Accuracy=acc, Precision=prec, Recall=rec, Specificity=spec, F1=f1, AUC=aucv)
}

tune_threshold <- function(y_true_num, y_prob, grid = seq(0.2, 0.8, by = 0.02)) {
  f1s <- sapply(grid, function(t) {
    y_pred <- ifelse(y_prob >= t, 1, 0)
    tp <- sum(y_pred==1 & y_true_num==1)
    fp <- sum(y_pred==1 & y_true_num==0)
    fn <- sum(y_pred==0 & y_true_num==1)
    prec <- ifelse(tp+fp > 0, tp/(tp+fp), 0)
    rec  <- ifelse(tp+fn > 0, tp/(tp+fn), 0)
    ifelse(prec+rec > 0, 2*prec*rec/(prec+rec), 0)
  })
  grid[ which.max(f1s) ]
}

## =======================
## 4) Baseline models
## =======================
# Logistic
logit_model <- glm(collision_likelihood ~ ., data = train, family = "binomial")
log_prob    <- predict(logit_model, newdata = test, type = "response")

# Threshold tuned by max-F1 (no 'metric' arg)
t_opt <- tune_threshold(y_test_num, log_prob)
log_base_metrics <- metrics_bin(y_test_num, log_prob, thr = t_opt)
print(round(log_base_metrics, 3))

# Random Forest
rf_model <- randomForest(x = x_train, y = y_train_fac, ntree = 200, importance = TRUE)
rf_prob  <- predict(rf_model, newdata = x_test, type = "prob")[, "1"]
rf_base_metrics <- metrics_bin(y_test_num, rf_prob, thr = 0.5)
print(round(rf_base_metrics, 3))


## =======================
## 5) SMOTE using smotefamily
## =======================
# smotefamily::SMOTE expects numeric target and matrix/data.frame predictors
smote_obj <- SMOTE(X = x_train, target = y_train_num, K = 5, dup_size = 0)

# Combine balanced data into one frame
smote_train <- as.data.frame(smote_obj$data)
names(smote_train)[ncol(smote_train)] <- "collision_likelihood"
smote_train$collision_likelihood <- as.numeric(smote_train$collision_likelihood)

x_smote <- subset(smote_train, select = -collision_likelihood)
y_smote_num <- smote_train$collision_likelihood
y_smote_fac <- factor(y_smote_num, levels = c(0,1))

# --- Logistic on SMOTE data
logit_sm <- glm(collision_likelihood ~ ., data = smote_train, family = "binomial")
log_prob_sm <- predict(logit_sm, newdata = test, type = "response")
log_sm_metrics <- metrics_bin(y_test_num, log_prob_sm, thr = t_opt)

# --- Random Forest on SMOTE data
rf_sm <- randomForest(x = x_smote, y = y_smote_fac, ntree = 300, importance = TRUE)
rf_prob_sm <- predict(rf_sm, newdata = x_test, type = "prob")[, "1"]
rf_sm_metrics <- metrics_bin(y_test_num, rf_prob_sm, thr = 0.5)

## =======================
## 6) XGBoost (Gradient Boosting)
## =======================
dtrain <- xgb.DMatrix(data = as.matrix(x_train), label = y_train_num)
dtest  <- xgb.DMatrix(data = as.matrix(x_test))
neg <- sum(y_train_num == 0); pos <- sum(y_train_num == 1)
spw <- if (pos > 0) neg / pos else 1

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  scale_pos_weight = spw,
  max_depth = 4,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_mod  <- xgb.train(params = params, data = dtrain, nrounds = 300, verbose = 0)
xgb_prob <- predict(xgb_mod, dtest)
xgb_metrics <- metrics_bin(y_test_num, xgb_prob, thr = 0.5)

## =======================
## 7) Model comparison
## =======================
res <- rbind(
  Logit_Base  = log_base_metrics,
  RF_Base     = rf_base_metrics,
  Logit_SMOTE = log_sm_metrics,
  RF_SMOTE    = rf_sm_metrics,
  XGBoost     = xgb_metrics
)
print(round(res, 3))

## =======================
## 8) ROC Plots
## =======================
roc_log <- roc(y_test_num, log_prob)
roc_rf  <- roc(y_test_num, rf_prob)
roc_xgb <- roc(y_test_num, xgb_prob)
plot(roc_log,  main = "ROC – Logistic vs RF vs XGBoost")
plot(roc_rf,   add = TRUE, col = "red")
plot(roc_xgb,  add = TRUE, col = "green")
legend("bottomright",
       legend = c(
         paste0("Logit AUC=", round(auc(roc_log),3)),
         paste0("RF AUC=",    round(auc(roc_rf),3)),
         paste0("XGB AUC=",   round(auc(roc_xgb),3))
       ),
       bty="n")

# Variable importance for RF baseline
varImpPlot(rf_model, main = "Variable Importance – RF (Baseline)")
