library(moments)
library(DescTools)
library(imputeMissings)
library(ggplot2)
library(purrr)
library(caret)
library(naivebayes)
library(randomForest)


s<-Spotify
str(s)

s<-as.data.frame(s)
s$song_preference<-as.factor(s$song_preference)
s$mode<-as.factor(s$mode)
s$time_signature<-as.factor(s$time_signature)
s$key<-as.factor(s$key)



map(s, ~sum(is.na(.)))
s<-impute(s, method="median/mode")
map(s, ~sum(is.na(.)))

kurtosis(s[c(2,3,4,5,6,8,9,11,12,14)])
skewness(s[c(2,3,4,5,6,8,9,11,12,14)])

s1<-s

s1$acousticness<-sqrt(s1$acousticness)
s1$instrumentalness<-log10(s1$instrumentalness+1)
s1$liveness<-log10(s1$liveness)
s1$speechiness<-log10(s1$speechiness)

#both loudness and duration_ms have severe outliers and are not getting treated
#by using log and sqrt transformations so we will treat the outliers
boxplot(s1$loudness)
boxplot(s1$duration_ms)

#LOUDNESS
outloud<-boxplot(s1$loudness)$out
length(outloud)
s1<-s1[-which(s1$loudness %in% outloud),]
boxplot(outloud)
#DURATION_MS
outduration<-boxplot(s1$duration_ms)$out
length(outduration)
s1<-s1[-which(s1$duration_ms %in% outduration),]
boxplot(outduration)

#RECHECKING THE NORMALITY
skewness(s1[c(2,3,4,5,6,8,9,11,12,14)])
kurtosis(s1[c(2,3,4,5,6,8,9,11,12,14)])

#creating data partition
########################
set.seed(100)
intrain<-createDataPartition(y=s1$song_preference, p=0.8, list=FALSE)
training1<-s1[intrain,]
testing1<-s1[-intrain,]
str(training1)
str(testing1)
########################

model1<-train(data=training1, song_preference~acousticness+danceability+duration_ms+energy+key+liveness+loudness+mode+speechiness+tempo+time_signature+valence, method='glm', family='binomial')
summary(model1)
p1<-predict(model1, newdata = testing1)
confusionMatrix(p1, testing1$song_preference, positive = "1")


#Only continuous + significant IVs.
model1.1<-train(data=training1, song_preference~danceability+energy+loudness+speechiness, method='glm', family='binomial')
summary(model1.1)
p1.1<-predict(model1.1, newdata = testing1)
confusionMatrix(p1.1, testing1$song_preference, positive = "1")

model2.1<-train(data=training1, song_preference~danceability+energy+loudness+speechiness, method='naive_bayes')
p2.1<-predict(model2.1, newdata = testing1)
confusionMatrix(p2.1, testing1$song_preference, positive="1")
#better specificity

#experiment
model3.1<-train(data=training1, song_preference~loudness+speechiness+duration_ms+acousticness+valence, method='glm', family="binomial")
summary(model3.1)
p3.1<-predict(model3.1, newdata = testing1)
confusionMatrix(p3.1, testing1$song_preference, positive="1")


#categorical variables and impact of instrumentalness
model1.2<-train(data=training1, song_preference~key+time_signature+mode, method='glm', family='binomial')
summary(model1.2)
p1.2<-predict(model1.2, newdata = testing1)
confusionMatrix(p1.2, testing1$song_preference, positive = "1")

model2.2<-train(data=training1, song_preference~key+mode+time_signature+instrumentalness, method='glm', family='binomial')
summary(model2.2)
p2.2<-predict(model2.2, newdata = testing1)
confusionMatrix(p2.2, testing1$song_preference, positive = "1")


#DECISION TREES

dt<-train(data=training1, song_preference~danceability+instrumentalness+energy+loudness, method="rpart", parms=list(split="information"))
rpart.plot(dt$finalModel)
predsong<-predict(dt, newdata = testing1)
confusionMatrix(predsong, testing1$song_preference,positive="1")


dt1<-train(data=training1, song_preference~danceability+instrumentalness+energy+loudness, method="rpart", parms=list(split="information"))
rpart.plot(dt1$finalModel)
predsong1<-predict(dt1, newdata = testing1)
confusionMatrix(predsong1, testing1$song_preference,positive="1")


dt2<-train(data=training1, song_preference~acousticness+danceability+duration_ms+energy+instrumentalness+key+liveness+loudness+mode+speechiness+tempo+time_signature+valence, method="rpart")
rpart.plot(dt2$finalModel)
predsong2<-predict(dt2, newdata = testing1)
confusionMatrix(predsong2, testing1$song_preference,positive="1")

dt3<-train(data=training1, song_preference~duration_ms+loudness+speechiness, method="rpart", parms=list(split="information"))
rpart.plot(dt3$finalModel)
predsong3<-predict(dt3, newdata = testing1)
confusionMatrix(predsong3, testing1$song_preference,positive="1")



##########
#RANDOM FORRESTS
##########
rf<-train(song_preference~acousticness+danceability+instrumentalness+duration_ms+energy+key+liveness+loudness+mode+speechiness+tempo+time_signature+valence, data=training1, method="rf")
rf$finalModel
rf
varImp(rf)
predsong_rf<-predict(rf, newdata = testing1)
confusionMatrix(predsong_rf, testing1$song_preference, positive = "1")


rf1<-train(song_preference~acousticness+instrumentalness+danceability+duration_ms+energy+liveness+loudness+speechiness+tempo+valence, data=training1, method="rf")
rf1
rf1$finalModel
varImp(rf1)
predsong_rf1<-predict(rf1, newdata = testing1)
confusionMatrix(predsong_rf1, testing1$song_preference, positive = "1")


rf6<-train(song_preference~acousticness+instrumentalness+danceability+duration_ms+energy+loudness+speechiness+tempo+valence, data=training1, method="rf")
rf6
rf6$finalModel
varImp(rf6)
predsong_rf6<-predict(rf6, newdata = testing1)
confusionMatrix(predsong_rf6, testing1$song_preference, positive = "1")
#######


rf2<-train(song_preference~acousticness+danceability+duration_ms+loudness+speechiness, data=training1, method="rf")
rf2
rf2$finalModel
varImp(rf2)
predsong_rf2<-predict(rf2, newdata = testing1)
confusionMatrix(predsong_rf2, testing1$song_preference, positive = "1")


rf3<-train(song_preference~duration_ms+loudness+speechiness, data=training1, method="rf")
rf3
rf3$finalModel
varImp(rf3)


rf4<-train(song_preference~danceability+instrumentalness+energy+loudness, data=training1, method="rf")
rf4
rf4$finalModel
varImp(rf4)
predsong_rf<-predict(rf4, newdata = testing1)
confusionMatrix(predsong_rf, testing1$song_preference, positive = "1")


rf5<-train(song_preference~acousticness+danceability+duration_ms+energy+loudness+speechiness+tempo+valence, data=training1, method="rf")
rf5
rf5$finalModel
varImp(rf5)
predsong_rf5<-predict(rf5, newdata = testing1)
confusionMatrix(predsong_rf5, testing1$song_preference, positive = "1")




