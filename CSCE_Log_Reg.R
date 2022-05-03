library(mvtnorm)
library(bayesrules)
library(rstanarm)
library(bayesplot)
library(tidyverse)
library(tidybayes)
library(broom.mixed)

data.file = "C:/Users/eguzm/OneDrive/Documents/heart_failure_clinical_records_dataset .csv"
data = read.csv(data.file,header =TRUE)

y = data$DEATH_EVENT
x1 = data$age
x2 = data$anaemia
x3 = data$creatinine_phosphokinase
x4 = data$diabetes
x5 = data$ejection_fraction
x6 = data$high_blood_pressure
x7 = data$platelets
x8 = data$serum_creatinine
x9 = data$serum_sodium
x10 = data$sex
x11 = data$smoking
x12 = data$time

data_new = data.frame(y,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12)

X <- as.matrix( cbind(rep(1, length(x1)), x1, x2, x3,x4,x5,x6,x7,x8,x9,x10,x11,x12) )

lpy.X <- function(y,X,g=length(y),nu0=1,s20=try(summary(lm(y~-1+X))$sigma^2,silent=TRUE)) {
  n<-dim(X)[1]; p<-dim(X)[2]
  if(p==0) {Hg<-0; s20<-mean(y^2)}
  if(p>0) {Hg<-(g/(g+1))*X%*%solve(t(X)%*%X)%*%t(X)}
  SSRg<- t(y)%*%( diag(1,nrow=n) - Hg)%*%y
  -.5*(n*log(pi)+p*log(1+g)+(nu0+n)*log(nu0*s20+SSRg)-nu0*log(nu0*s20))+lgamma((nu0+n)/2)-lgamma(nu0/2)}


z<-rep(1,dim(X)[2])  # starting with z = all 1's (all terms in model)
lpy.c<-lpy.X(y,X[,z==1,drop=FALSE])
S <- 10000  # number of Monte Carlo iterations
Z<-matrix(NA,S,dim(X)[2])

for(s in 1:S)
{
  for(j in sample(1:dim(X)[2]))
  {
    zp<-z; zp[j] <- 1-zp[j]
    lpy.p<-lpy.X(y,X[,zp==1,drop=FALSE])
    r<- (lpy.p - lpy.c)*(-1)^(zp[j]==0)
    z[j]<-rbinom(1,1,1/(1+exp(-r)))
    if(z[j]==zp[j]) {lpy.c<-lpy.p}
  }
  Z[s,]<-z}

poss.z.vectors <-  unique(Z,MARGIN=1)
z.probs <- rep(0, times= nrow(poss.z.vectors))
for(i in 1:nrow(poss.z.vectors)) {
  z.probs[i] <- sum(apply(Z,1,identical, y=poss.z.vectors[i,]))}
z.probs <- z.probs/sum(z.probs)
cbind(poss.z.vectors, z.probs)[rev(order(z.probs)),]

death_model_full = stan_glm(y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12, data=data_new,
                       family=binomial,prior_intercept = normal(0,4,autoscale = TRUE),
                       prior = normal(0,4,autoscale = TRUE),
                       chains = 4, iter = 10000)

tidy(death_model_full, effects = "fixed", conf.int = TRUE, conf.level = 0.95)

death_model_simp = stan_glm(y~x1+x5+x8+x12, data=data_new,
                            family=binomial,prior_intercept = normal(0,4,autoscale = TRUE),
                            prior = normal(0,4,autoscale = TRUE),
                            chains = 4, iter = 10000)

tidy(death_model_simp, effects = "fixed", conf.int = TRUE, conf.level = 0.95)

death_model_notha_simp = stan_glm(y~x5+x8+x9+x12, data=data_new,
                            family=binomial,prior_intercept = normal(0,4,autoscale = TRUE),
                            prior = normal(0,4,autoscale = TRUE),
                            chains = 4, iter = 10000)

tidy(death_model_notha_simp, effects = "fixed", conf.int = TRUE, conf.level = 0.95)

cv_accuracy_full <- classification_summary_cv(
  model = death_model_full, data = data_new, cutoff = 0.31, k = 10)

cv_accuracy_full$cv

cv_accuracy_simp <- classification_summary_cv(
  model = death_model_simp, data = data_new, cutoff = 0.31, k = 10)

cv_accuracy_simp$cv

cv_accuracy_notha_simp <- classification_summary_cv(
  model = death_model_notha_simp, data = data_new, cutoff = 0.31, k = 10)

cv_accuracy_notha_simp$cv


loo_1 <- loo(death_model_full)
loo_2 <- loo(death_model_simp)
loo_3 <- loo(death_model_notha_simp)
loo_1$estimates
loo_2$estimates
loo_3$estimates





