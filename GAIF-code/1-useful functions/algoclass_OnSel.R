library(methods)

setGeneric("fitting",function(obj,...) standardGeneric("fitting"))
setGeneric("Pred",function(obj,...) standardGeneric("Pred"))
setGeneric("Tuning",function(obj,...) standardGeneric("Tuning"))
setGeneric("DataGen",function(obj,...) standardGeneric("DataGen"))

###ridge - regression
setClass("ridge",slots = list(name="character",alpha="numeric",family="character"), prototype=list(name="RR",alpha=0,family="gaussian") )
setMethod("fitting","ridge",function(obj,X,Y,lambda){
  glmnet(x=X,y=Y,family =obj@family,alpha=obj@alpha,lambda =lambda)
})
setMethod("Pred","ridge",function(obj,model,X_test,lens=0){
  predict(model,X_test)
})
setMethod("Tuning","ridge",function(obj,X,Y){
  ridge=cv.glmnet(x=X,y=Y,type.measure="mse",family = obj@family,alpha=obj@alpha)
  return(ridge$lambda.min)
})
setMethod("DataGen","ridge",function(obj,X,beta,sigma){
  Y=X%*%beta+rnorm(dim(X)[1],0,sigma)
  return(data.frame(x=X,y=Y))
})

###lasso - regression
setClass("lasso",slots = list(name="character",alpha="numeric",family="character"), prototype=list(name="Lasso",alpha=1,family="gaussian") )
setMethod("fitting","lasso",function(obj,X,Y,lambda){
  glmnet(x=X,y=Y,family =obj@family,alpha=obj@alpha,lambda =lambda)
})
setMethod("Pred","lasso",function(obj,model,X_test,lens=0){
  predict(model,X_test)
})
setMethod("Tuning","lasso",function(obj,X,Y,lambda_array=FALSE){
  if (class(lambda_array[1])=='logical')
  {lasso=cv.glmnet(x=X,y=Y,type.measure="mse",family = obj@family,alpha=obj@alpha)}else
  {lasso=cv.glmnet(x=X,y=Y,lambda=lambda_array,type.measure="mse",family = obj@family,alpha=obj@alpha)}
  
  return(lasso$lambda.min)
})
setMethod("DataGen","lasso",function(obj,X,beta,sigma){
  b=c(1,-1,2,-2,rep(0,dim(X)[2]-4))
  Y=X%*%b+rnorm(dim(X)[1],0,sigma)
  return(data.frame(x=X,y=Y))
})


###random forest - regression
setClass("RF",slots = list(name="character"),prototype = list(name="RF"))
setMethod("fitting","RF",function(obj,X,Y,lambda){
  datawork=data.frame(X,y=Y)
  randomForest(y~.,data=datawork,mtry=round(dim(X)[2]/3),ntree=lambda)
})
setMethod("Pred","RF",function(obj,model,X_test,lens=0){
  predict(model,as.data.frame(x=X_test))})
setMethod("Tuning","RF",function(obj,X,Y){
  #lambda_array=seq(15,200,5)
  #Folds=createFolds(1:length(Y),5)
  #datawork=data.frame(X,y=Y)
  #error=sapply(lambda_array, function(lam){
  #  whole_error=sapply(1:5, function(t){
  #    Model=randomForest(y~.,data=datawork[-Folds[[t]],],mtry=round(dim(X)[2]/3),ntree=lam)
  #    error=sum(predict(Model,X[Folds[[t]],])-Y[Folds[[t]]])^2
  #    return(error)
  #  })
  #  mean(whole_error)
  #})
  #ntree=lambda_array[which.min(error)]
  #return(ntree)
  return(500)
})
setMethod("DataGen","RF",function(obj,X,beta,sigma){
  Y=10*X[,1]+7*X[,2]^2+7*exp(X[,3]+2*X[,4]^2)+rnorm(dim(X)[1],sigma)
  return(data.frame(x=X,y=Y))
})

###Neural network - regression
setClass("NN-R",slots = list(name="character",stepmax="numeric",rep="numeric",hidden="numeric"),
         prototype = list(name="NN-R",stepmax=10000,hidden=10))
setMethod("fitting","NN-R",function(obj,X,Y,lambda){
  datawork=data.frame(X,y=Y)
  nnet(y~., data = datawork, size = 10, linout = T, maxit = 2000)
})
setMethod("Pred","NN-R",function(obj,model,X_test,lens=0){
  predict(model,as.data.frame(x=X_test))})
setMethod("Tuning","NN-R",function(obj,X,Y){
  learningrate=0.05
  return(learningrate)})
setMethod("DataGen","NN-R",function(obj,X,beta,sigma){
  Y=X%*%beta+rnorm(dim(X)[1],sigma)
  return(data.frame(x=X,y=Y))
})




###random forest - classification
setClass("RFc",slots = list(name="character"),prototype = list(name="RFc"))
setMethod("fitting","RFc",function(obj,X_rest,Y_rest,lambda){
  randomForest(X_rest,as.factor(Y_rest),mtry=round(3),ntree=lambda)
  #model=randomForest(X_rest,as.factor(Y_rest),mtry=round(dim(X_rest)[2]/3),ntree=200,classwt=c(0.8,0.2))
  #roc(Y_test,predict(model,X_test,type="prob")[,2])
})
setMethod("Pred","RFc",function(obj,model,X_test,lens=0,type='decision'){
  if (type=="class")
  {predraw=as.numeric(as.character(predict(model,X_test)))}else
  {predraw=predict(model,X_test,type="prob")[,2]
  return(predraw)}
})

setMethod("Tuning","RFc",function(obj,X,Y){
  return(500)
})
setMethod("DataGen","RFc",function(obj,X,beta,sigma){
  #Y=apply(X,1,function(i){rbinom(1,1,1/(exp(t(i)%*%beta)+1)) })
  n1=round(dim(X)[1]*sigma)
  d=dim(X)[2]
  #X1=mvrnorm(n1,c(rep(1,d/2),rep(0,d/2)),diag(rep(1,d)))
  X1=mvrnorm(n1,c(rep(1,round(d/4)),rep(0,d-round(d/4))),diag(rep(1/2,d)))+mvrnorm(n1,c(rep(0,round(d/4)),rep(1,d/2-round(d/4)),rep(0,d/2)),diag(rep(1,d)))
  X2=mvrnorm(dim(X)[1]-n1,c(rep(0,d/2),rep(1,d/2)),diag(rep(1,d)))
  X=rbind(X1,X2)
  data=data.frame(x=X,y=c(rep(0,n1),rep(1,dim(X)[1]-n1)))
  data=data[sample(1:dim(data)[1],dim(data)[1]),]
  return(data)
})

###SVM - classification
setClass("SVM",slots = list(name="character"),
         prototype = list(name="SVM"))
setMethod("fitting","SVM",function(obj,X,Y,lambda){
  datawork=data.frame(X,y=as.factor(Y))
  ksvm(y~.,data=datawork,type="C-svc",C=lambda,prob.model=TRUE)
})
setMethod("Pred","SVM",function(obj,model,X_test,lens,type="prob"){
  if (type=="class")
  {predraw=as.numeric(as.character(predict(model,X_test)))}else
  {predraw=predict(model,X_test,type="probabilities")
  return(predraw)}
  })
setMethod("Tuning","SVM",function(obj,X,Y){
  #lambda_list=seq(0.2,4,0.1)
  #datawork=data.frame(X,y=as.factor(Y))
  #error=try(sapply(lambda_list,function(t){
  #  svmmodel=ksvm(y~.,data=datawork,type="C-svc",C=t,cross=5)
  #  return(svmmodel@cross)})
  #)
  #if('try-error' %in% class(error))
  #{return(1)}else
  #{return(lambda_list[min(which.min(error))])}
  #})
  return(1)})
setMethod("DataGen","SVM",function(obj,X,beta,sigma){
  #Y=apply(X,1,function(i){rbinom(1,1,1/(exp(t(i)%*%beta)+1)) })
  n1=round(dim(X)[1]*sigma)
  d=dim(X)[2]
  #X1=mvrnorm(n1,c(rep(1,d/2),rep(0,d/2)),diag(rep(1,d)))
  X1=mvrnorm(n1,c(rep(1,round(d/4)),rep(0,d-round(d/4))),diag(rep(1/2,d)))+mvrnorm(n1,c(rep(0,round(d/4)),rep(1,d/2-round(d/4)),rep(0,d/2)),diag(rep(1,d)))
  X2=mvrnorm(dim(X)[1]-n1,c(rep(0,d/2),rep(1,d/2)),diag(rep(1,d)))
  X=rbind(X1,X2)
  data=data.frame(x=X,y=c(rep(1,n1),rep(-1,dim(X)[1]-n1)))
  data=data[sample(1:dim(data)[1],dim(data)[1]),]
  return(data)
})

###linear regression - regression
setClass("LRs",slots = list(name="character"), prototype=list(name="LR-standard") )
setMethod("fitting","LRs",function(obj,X,Y,lambda){
  if(dim(X)[1]>=dim(X)[2])
  {datawork=data.frame(X,y=Y)
  return(lm(y~.,datawork))}else
  {X_a1=cbind(rep(1,dim(X)[1]),X)
  return(ginv(t(X_a1)%*%X_a1)%*%t(X_a1)%*%Y)}
})
setMethod("Pred","LRs",function(obj,model,X_test,lens=0){

  if(class(model)=="lm")
  {predict(model,as.data.frame(x=X_test))}else
  {X_a1=cbind(rep(1,dim(X_test)[1]),X_test)
  X_a1%*%model}
})
setMethod("Tuning","LRs",function(obj,X,Y){
  return(0)
})
setMethod("DataGen","LRs",function(obj,X,beta,sigma){
  Y=X%*%beta+rnorm(dim(X)[1],0,sigma)
  return(data.frame(x=X,y=Y))
})

###glml - classification
setClass("glml",slots = list(name="character",family="character"), prototype=list(name="GLM",family="binomial") )
setMethod("fitting","glml",function(obj,X,Y,lambda){
  datawork=data.frame(X,y=Y)
  glm(y~.,datawork,family =binomial())
  
})
setMethod("Pred","glml",function(obj,model,X_test,lens=0){
  predict(model,as.data.frame(x=X_test),type="response")})
setMethod("Tuning","glml",function(obj,X,Y){
  
  return(0)
})
setMethod("DataGen","glml",function(obj,X,beta,sigma){
  
  
  proba=apply(X, 1, function(t){
    1/(1+exp(t%*%beta))
  })
  Y=1-sapply(proba,function(t){
    rbinom(1,1,proba)
  })

  return(data.frame(x=X,y=Y))
})


### SVM - regression
setClass("SVM-R",slots = list(name="character"),
         prototype = list(name="SVM-R"))
setMethod("fitting","SVM-R",function(obj,X,Y,lambda){
  datawork=data.frame(X,y=Y)
  ksvm(y~.,data=datawork,C=lambda)
})
setMethod("Pred","SVM-R",function(obj,model,X_test,lens,type="decision"){
  predraw=predict(model,X_test)
  return(predraw)
})
setMethod("Tuning","SVM-R",function(obj,X,Y){
  lambda_list=c(1,2,3,4,5,6,7,8,9,10)
  datawork=data.frame(X,y=Y)
  error=try(sapply(lambda_list,function(t){
    svmmodel=ksvm(y~.,data=datawork,C=t,cross=5)
    return(svmmodel@cross)})
  )
  if('try-error' %in% class(error))
  {return(1)}else
  {return(lambda_list[min(which.min(error))])}
})
#return(1)})
setMethod("DataGen","SVM-R",function(obj,X,beta,sigma){
  Y=10*sin(pi*X[,1]*X[,2])+20*(X[,3]-0.05)^2+10*X[,4]+5*X[,5]+rnorm(dim(X)[1],sigma)
  return(data.frame(x=X,y=Y))
})


### bayes - classification
setClass("nbayes",slots = list(name="character"), prototype=list(name="NBayes") )
setMethod("fitting","nbayes",function(obj,X,Y,lambda){
  datawork=data.frame(X,y=Y)
  model=naiveBayes(y~.,data = datawork)
  
})
setMethod("Pred","nbayes",function(obj,model,X_test,lens=0){
  predict(model,as.data.frame(x=X_test),type="raw")[,2]})
setMethod("Tuning","nbayes",function(obj,X,Y){
  
  return(0)
})
setMethod("DataGen","nbayes",function(obj,X,beta,sigma){return(0)})
  


### lda -classification
setClass("lda",slots = list(name="character"), prototype=list(name="LDA") )
setMethod("fitting","lda",function(obj,X,Y,lambda){
  datawork=data.frame(X,y=Y)
  model=lda(y~.,data = datawork)
  
})
setMethod("Pred","lda",function(obj,model,X_test,lens=0){
  pred=predict(model,as.data.frame(x=X_test))
  return(pred$x)})
setMethod("Tuning","lda",function(obj,X,Y){
  
  return(0)
})
setMethod("DataGen","lda",function(obj,X,beta,sigma){return(0)})


###Neural network - classification
setClass("NN",slots = list(name="character",stepmax="numeric",rep="numeric",hidden="numeric"),
         prototype = list(name="NN",stepmax=10000,hidden=10))
setMethod("fitting","NN",function(obj,X,Y,lambda){
  datawork=data.frame(X,y=Y)
  model=nnet(y~., data = datawork, size = 5,decay=5e-4,entropy=TRUE, maxit = 2000)
  return(model)
})
setMethod("Pred","NN",function(obj,model,X_test,lens=0){
  predict(model,as.data.frame(x=X_test))})
setMethod("Tuning","NN",function(obj,X,Y){
  learningrate=0.05
  return(learningrate)})
setMethod("DataGen","NN",function(obj,X,beta,sigma){
  Y=X%*%beta+rnorm(dim(X)[1],sigma)
  return(data.frame(x=X,y=Y))
})

