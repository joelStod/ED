# Load dependencies -------------------------------------------------------
library(randomForest)
library(glmnet)
library(caret)
library(mlbench)
library(caretEnsemble)
library(AppliedPredictiveModeling)
library(plyr)
library(s2dverification)
library(corrplot)
library(sjPlot)
library(MLmetrics)
library(rpart)

# Read in, prep, and select data ------------------------------------------
# To avoid bloat, not all preparation and wrangling is described here, \
# but key elements are.

# The outcome variable is BMI_EDEQ, which is residualized on BMI_Admit
eatDB$BMI_EDEQr<-resid(lm(eatDB$BMI_EDEQ~eatDB$BMI_Admit))

# A predictor variable, Discharge BMI, must also be residualized on BMI_Admit
eatDB$BMI_DCr<-resid(lm(eatDB$BMI_DC~eatDB$BMI_Admit)) 

# Scale ordinal responses in selected cases
scaleMe<-read.table("ColumnNamesToBeZScored.txt",header=TRUE)
eatDB[scaleMe$ColNames]<-scale(eatDB[scaleMe$ColNames])

# Omit variables with missing responses
keep<-NA
for (i in 1:dim(eatDB)[2]) {
  tmp<-sum(is.na(eatDB[,i]))/max(row(eatDB))
  keep<-c(keep,(is.na(tmp) | tmp<0.2)) #return true if # na's <20%
}
keep<-keep[-1]
eatDBc<-eatDB[,keep] #a dataset with at least 40% observation per variable
rm(tmp,keep,i)

# Impute missing data on BMI variable

set.seed(1234)

# Select factors and numeric classes
# Note the variables listed in the array are examples of circular variables
# surviving data wrangling but ultimately omitted from analysis.
eatDBcFN<-eatDBc[,sapply(eatDBc, class) %in% c('numeric','integer','factor')] #remove characters
eatDBc.imputed<-rfImpute(BMI_EDEQr~.,
                         data=eatDBcFN[,
                                       !names(eatDBcFN) %in% c(
                                         "ID", # Subject ID not needed
                                         "Time_Point", # Moot due to earlier selection
                                         "edc_progam", # Program is either Adult or Adol is nuisance variable, covaries with age
                                         "Admit_date", # Date of admission
                                         "DC_date", # Moot
                                         "dx_use2", # Duplicate, reduced version of another variable, dx_cur
                                         "dx_cur",  # Dx depend on BMI
                                         "partial_remiss", # A proxy of dx_cur
                                         "BMI_Admit", # Already encoded by subtraction in residualization
                                         "BMI_DC",  # Replaced by the residuzalized form BMI_DCr
                                         "BMI_EDEQ",# Replaced by teh residualized form BMI_EDEQr
                                         "LLW_BMI", # BMI proxy
                                         "LHW_BMI", # BMI proxy
                                         "time_intake", # Nuisance variable amount of intake to survey
                                         "time_dc" # Nuisance as above
                                       )])

# Be careful and inspect categorical data, the imputation may result in fractions.
save(eatDBc.imputed,file = "eatDBcT2_BMIEDEQr.imputed.Rdata")

# Machine learn on post treatment BMI ----
load(file="eatDBcT2_BMIEDEQr.imputed.Rdata")

inTrain <- createDataPartition(y = eatDBc.imputed$BMI_EDEQr, 
                               p = .75, 
                               list = FALSE) 

train <- eatDBc.imputed[inTrain,]; trainAllVars<-eatDB[inTrain,]
test <- eatDBc.imputed[-inTrain,]; testAllVars<-eatDB[-inTrain,]

save(train,file="train_EDEQr.Rda")
save(trainAllVars, file="trainAllVars_EDEQr.Rda")
save(test,file="test_EDEQr.Rda")
save(testAllVars, file="testAllVars_EDEQr.Rda")

# Learn -----
load("train_EDEQr.Rda")
load("test_EDEQr.Rda")

control <- trainControl(method="repeatedcv",
                        number=10,            # 10-fold cross-validation
                        repeats=10,            # repeat 10-fold CV for 10 times
                        allowParallel=T,
                        index=createMultiFolds(train$BMI_EDEQr, k=10, times=10),
                        savePredictions='final')  

algorithmList <- c('rf', 'svmRadial','glmnet')

library(doParallel)
cluster = makeCluster(detectCores() - 2)
registerDoParallel(cluster)

system.time(
  models.train <- caretList(BMI_EDEQr~., 
                            data=train, 
                            metric='RMSE',
                            trControl=control, 
                            tuneLength=8, 
                            methodList=algorithmList))   

save(models.train, file="BMI_models_EDEQr.train.Rdata")
stopCluster(cluster)
registerDoSEQ()

# Examine model ----
load("BMI_models_EDEQr.train.Rdata")
load("train_EDEQr.Rda"); load("test_EDEQr.Rda")

resamps=resamples(models.train)
summary(resamps)

# Combine models into an ensemble and test their predictions ----
stackControl <- trainControl(method="repeatedcv", number=10, repeats=10, 
                             savePredictions='final')
ens.train <- caretEnsemble(models.train, 
                           metric='RMSE', 
                           trControl=stackControl)
print(ens.train)
plot(ens.train)
summary(ens.train)

var.imp.train <- varImp(ens.train, useModel=TRUE, scale=TRUE)          
plot(var.imp.train$overall)
var.imp.train$cor<-NA
for (x in row.names(var.imp.train)) {
  if (x %in% colnames(train)) {
    if(is.numeric(train[,x])){
      var.imp.train[row.names(var.imp.train)==x,"cor"]<-
        cor(train$BMI_EDEQr,train[,x], method='pearson')
    }
  }
}

SMD<-function(x,y){
  mean1<-mean(x[y])
  mean2<-mean(x[!y])
  poolSD<-sd(x)
  print((mean1-mean2)/poolSD)
}
SMD(train$BMI_EDEQr,train$race_COMBINED==2)
SMD(train$BMI_EDEQr,train$ethnicity_COMBINED==2)
SMD(train$BMI_EDEQr,train$referral_src=="NoTx")

head(var.imp.train[order(-var.imp.train$overall),],n=30)
write.csv(var.imp.train, file="BMI_Time2_EDEQr_18.5_June_2022.csv")

model_preds <- lapply(models.train, predict, newdata=test, type="raw")   
model_preds <- data.frame(model_preds)
model_preds$ensemble <- predict(ens.train, newdata=test, type="raw")
cor.tab<-psych::corr.test(test$BMI_EDEQr,model_preds,method="pearson")
print(cor.tab,short=FALSE)

plot(model_preds$ensemble,test$BMI_EDEQr,ylab="Model Predicted 6-month BMI on EDEQ",
     xlab="Actual EDEQ BMI at 6 Months",pch=16, main="Out of Training Sample Predictions")
abline(lm(test$BMI_EDEQr~model_preds$ensemble),lwd=3,col='darkred')