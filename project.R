#loading data
train=read.csv('Train_data.csv',na.strings=c("",NA),stringsAsFactors = T,sep=',')
test=read.csv('Test_data.csv',na.strings=c("",NA),stringsAsFactors = T)
#missing or empty values
sapply(train,function(x) sum(is.na(x)))
sapply(test,function(x) sum(is.na(x)))
#class imbalance
hist(as.numeric(as.factor(train$Churn)))

#correlation plot
corm=train
corm=corm[-4]#removing state variable
corm=corm[-1]#removing phone number
corm[3:4]=as.integer(unlist(corm[3:4]))
matrix=cor(corm[1:18])
head(print(matrix))
library(corrplot)
corrplot(matrix, method="pie")
#data pre processing
train$international.plan=as.factor(unclass(train$international.plan))
train$voice.mail.plan=as.factor(unclass(train$voice.mail.plan))
train$Churn=as.factor(unclass(train$Churn))
#deleting columns which are not
#useful in prediction(highly correlated)
library(dplyr)
train=select(train,-number.vmail.messages,-total.day.charge,
             -total.night.charge,-total.intl.charge)
#removing columns which are not statistically significant
#reason mentioned in report(to increase accuracy)
train=train[-4]#removinf customer phone number
train=select(train,-state,-area.code,-account.length, -total.day.calls, -total.eve.calls )


#data pre processing on test data
test$international.plan=as.factor(unclass(test$international.plan))

test$voice.mail.plan=as.factor(unclass(test$voice.mail.plan))
test=test[-4]#removing  phone number columns which  not useful


#removing columns which are highly correlated
test=select(test,-number.vmail.messages,-total.day.charge,
            -total.night.charge,-total.intl.charge)
#removing columns which are not statistically significant from trest dat
#as we have removed the same from train data

test=select(test,-state,-area.code,-account.length, -total.day.calls, -total.eve.calls )

# removing outliers and replacing them with quantile 1 and quantile 2
#with respective to their position
boxplot(train)
outlier_removed_data=train
for (value in c('total.eve.minutes','total.day.minutes','total.eve.charge',
                'total.night.minutes','total.night.calls','total.intl.minutes',
                'total.intl.calls')){
qn = quantile(outlier_removed_data[value], c(0.05, 0.95), na.rm = TRUE)
print(qn)
outlier_removed_data[value][outlier_removed_data[value]<qn[1]]=qn[1]
outlier_removed_data[value][outlier_removed_data[value]>qn[2]]=qn[2]
}
boxplot(outlier_removed_data)
#dealing with imbalanced train data
#with outliers
table(train$Churn)
prop.table(table(train$Churn))
install.packages("ROSE")
library(ROSE)
train_data <- ROSE(Churn ~ ., data = train, seed = 1)$data
train_data$Churn=ifelse(train_data$Churn==1,"False","True")

table(train_data$Churn)
prop.table(table(train_data$Churn))
#with out outliers
outlier_removed_data=ROSE(Churn ~ ., data = outlier_removed_data, seed = 1)$data
outlier_removed_data$Churn=ifelse(outlier_removed_data$Churn==1,"False","True")

#function for finding optimum thershold and calculating accuracy
myfunction=function(model){
  print(model)
  #summary of the model
  print(summary(model))
  #predicting classes
  pred_class=predict(model,test[1:10])
   print(table(pred_class,test$Churn))
  #predicting probabilities
  pred=predict(model,test[1:10],type='prob')
  a=data.frame(prob=pred[,2], obs=test$Churn)
  library(pROC)
  #plotting ROC
  modelroc = roc(a$obs,a$prob)
  plot(modelroc, print.auc=TRUE, auc.polygon=T, grid=c(0.1,0.2),grid.col=c('green','red'),max.auc.polygon=TRUE, auc.polygon.col="skyblue", print.thres=TRUE)
  ##adjust optimal cut-off threshold for class probabilities
  threshold = coords(modelroc,x="best",best.method = "closest.topleft")[[1]]
  print("threshold")
  print(threshold)
  #get optimal cutoff threshold
  predCut = factor( ifelse(pred[, 2] < threshold, " False.", " True.") )
  #confusion matrix
  confusionMatrix(test$Churn,predCut)
}


#######logistic regression#####
library(caret)
train_control_log<- trainControl(method="cv", number=10,classProbs = TRUE,summaryFunction = twoClassSummary)
#model with outliers
#no tuning parameters for regressin and classification
Logmodel<- train(Churn~., data=train_data, trControl=train_control_log, method="glm",metric="ROC")
#calling function
myfunction(Logmodel)
######model with out outliers###
Logmodel<- train(Churn~., data=outlier_removed_data, trControl=train_control_log, method="glm",metric="ROC")
#calling function
myfunction(Logmodel)


# #####random forest#####
library(caret)
# define training control
train_control_RF<- trainControl(method="cv", number=10,classProbs = TRUE,summaryFunction = twoClassSummary)
# tuning the model 
newGrid = expand.grid(mtry = c(2,15,20))
#model with outliers

RFmodel<- train(Churn~., data=train_data, trControl=train_control_RF, method="rf",ntree=500,
                metric="ROC",grid=newGrid)
#calling function
myfunction(RFmodel)
####model with out outliers###
RFmodel<- train(Churn~., data=outlier_removed_data, trControl=train_control_log, method="rf",
                ntree=500,metric="ROC",grid=newGrid)
#calling function
myfunction(RFmodel)



#######naive bayes#######
train_control_NB<- trainControl(method="cv", number=10,classProbs = TRUE,summaryFunction = twoClassSummary)
# train the model
nb_grid =data.frame(fL=c(0,1.0,3.0), usekernel = TRUE, adjust=c(0,0.5,2.0))

nb_tune <- data.frame(usekernel =TRUE, fL = 0)
#model with outliers

NBmodel=train(Churn~., data=train_data, trControl=train_control_NB, method="nb",metric="Accuracy",tuneGrid =nb_grid)
#calling function

myfunction(NBmodel)
####model with out outliers###
NBmodel<- train(Churn~., data=outlier_removed_data, trControl=train_control_log, method="nb",metric="ROC",tuneGrid=nb_grid)
#calling function
myfunction(NBmodel)

#explanation:
#The problem is that you have ~800 predictors and will be multiplying a lot of stuff 
#between [0, 1] together and that can cause numerical problems
#So, my guess as tot the problem is that Naive Bayes isn't really the right tool for the job here (e.g. large numbers of predictors).
#Numerically, the probabilities are all going towards zero




####### KNN algorithm ######
train_control_KNN<- trainControl(method="cv", number=10,classProbs = TRUE,summaryFunction = twoClassSummary)
# tuning the model
Knn_grid   = expand.grid(k =50:80)
#model with outliers
KNNmodel=train(Churn~., data=train_data, trControl=train_control_KNN, method="knn",metric="ROC",tuneGrid=Knn_grid)
plot(KNNmodel)
#calling function

myfunction(KNNmodel)

####model with out outliers###
KNNmodel<- train(Churn~., data=outlier_removed_data, trControl=train_control_KNN, method="knn",metric="ROC",tuneGrid=Knn_grid)
#calling function
myfunction(KNNmodel)



