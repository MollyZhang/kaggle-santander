# Script for exploratory analysis of the santander training set

# clear workspace
rm(list = ls())
library(mgcv)
library(rpart)

# JUMP TO LINE 90 ----------------------------------------------------------

path.to.data <- '~/kaggle/santander/data/train.csv' # change accordingly
data <- read.csv(path.to.data)

dim(data)	# 76020 rows, 371 cols
names(data)

summary(data$TARGET) # Binary: 0 = satisfied; 1 = unsatisfied
table(data$TARGET)


# First see if we can throw out any variables
for (name in names(data)) {
  print(paste(name, class(data[, name])))
}

# All are numeric or int
# Check if any have just a single value--if so, no information content -> remove
remove <- c()
for (name in names(data)) {
  if (length(unique(data[, name])) == 1) {
  	remove <- c(remove, name)
  }
}

remove
remove.inds <- which(names(data) %in% remove)
data <- data[, -remove.inds]

dim(data) # 76020 x 337

# Structure of NAs?
which(is.na(data)) # There are none!  :)


# Any colinearity in the data?
mod <- lm(TARGET ~ ., data = data)
summary(mod)

# Yes, quite a bit... 
# Any perfectly correlated data?  If so remove all but one in each set
# First convert all cols to Z-scores
for (col in 1:length(data)) {
  data[, col] <- scale(data[, col])
}

# Find any identical columns, and remove all but the first

remove <- c()
for (i in 1:336) {
  for (j in (i + 1):337) {
	if (cor(data[, i], data[, j]) == 1) {
	  remove <- c(remove, names(data)[j])
	}
  }
}

remove.inds <- which(names(data) %in% remove)
data <- data[, -remove.inds]

# write simplified data set
write.csv(data, '~/kaggle/santander/data/trainReduced.csv')



data <- read.csv('~/kaggle/santander/data/trainReduced.csv')
data[1:10, 1:10]
data <- data[, -1]
dim(data)	# 76020 x 290

mod <- lm(TARGET ~ ., data = data[, -1])
summary(mod)

# Still some colinearity in the data; remove redundancies:
remove <- names(which(is.na(coef(mod))))
remove.inds <- which(names(data) %in% remove)
data <- data[, -remove.inds]

write.csv(data, '~/kaggle/santander/data/trainReduced.csv')


# START FROM HERE -----------------------------------------------------
data <- read.csv('~/kaggle/santander/data/trainReduced.csv')
dim(data)

mod <- lm(TARGET ~ ., data = data[, -1])
summary(mod)

# Good, no more colinearity!


# Still too many variables to manage easily, use PCA to compress
names(data)
data <- data[, -c(1:2)] # Remove X, ID
dim(data) # 76020 x 234
names(data)

# Do an 80/20 split for prelim testing
n.train <- 0.8 * 76020
set.seed(9)
train.rows <- sample(76020, n.train)

train <- data[train.rows, ]
test <- data[-train.rows, ]


pca <- princomp(train[, -234]) # don't include TARGET
plot(pca, type = 'l')

plot(pca$scores[, 1], pca$scores[, 2], col = as.factor(data$TARGET), pch = 16)
plot(pca$scores[, 1], pca$scores[, 3], col = as.factor(data$TARGET), pch = 16)
plot(pca$scores[, 1], pca$scores[, 4], col = as.factor(data$TARGET), pch = 16)
plot(pca$scores[, 1], pca$scores[, 5], col = as.factor(data$TARGET), pch = 16)
plot(pca$scores[, 1], pca$scores[, 6], col = as.factor(data$TARGET), pch = 16)
plot(pca$scores[, 1], pca$scores[, 7], col = as.factor(data$TARGET), pch = 16)
plot(pca$scores[, 1], pca$scores[, 8], col = as.factor(data$TARGET), pch = 16)
plot(pca$scores[, 1], pca$scores[, 9], col = as.factor(data$TARGET), pch = 16)
plot(pca$scores[, 1], pca$scores[, 10], col = as.factor(data$TARGET), pch = 16)
# no clear separation of TARGET values on any of the first 10 PCs :(

data$TARGET[which(data$TARGET < 0)] <- 0
data$TARGET[which(data$TARGET > 0)] <- 1
table(data$TARGET)
train$TARGET[which(train$TARGET < 0)] <- 0
train$TARGET[which(train$TARGET > 0)] <- 1
test$TARGET[which(test$TARGET < 0)] <- 0
test$TARGET[which(test$TARGET > 0)] <- 1

pca.df <- as.data.frame(pca$scores)
pca.df <- cbind(TARGET = test$TARGET, pca.df)

#mod.pca <- gam(
#  TARGET ~ s(Comp.1, bs = 'cr') + s(Comp.2, bs = 'cr') + s(Comp.3, bs = 'cr') +
#  s(Comp.4, bs = 'cr') + s(Comp.5, bs = 'cr') + s(Comp.6, bs = 'cr') + 
#  s(Comp.7, bs = 'cr') + s(Comp.8, bs = 'cr') + s(Comp.9, bs = 'cr') + 
#  s(Comp.10, bs = 'cr'), 
#  family = binomial, 
#  data = pca.df)
#save(mod.pca, file = '~/kaggle/santander/data/modpca.rda')
load(file = '~/kaggle/santander/data/modpca.rda')
summary(mod.pca) 

par(mfrow = c(3, 4))
plot(mod.pca)

# Try a simpler logistic regression model
mod.log <- glm(TARGET ~ (Comp.1 + Comp.2 + Comp.3 + Comp.4 + Comp.5)^2 + Comp.6 + 
			   Comp.7 + Comp.8 + Comp.9 + Comp.10, 
			   family = binomial, 
			   data = pca.df)
mod.log <- step(mod.log, direction = 'both')			   
summary(mod.log)
par(mfrow = c(2, 2))
plot(mod.log)

# Given our binary response, try a classification tree model
# Initial model:
#rt1 <- rpart(TARGET ~ ., data = test)
#save(rt1, file = '~/kaggle/santander/data/rt1.rda')
load(file = '~/kaggle/santander/data/rt1.rda')
rt1
par(mfrow = c(1, 1))
plot(rt1)
text(rt1)

#rt2 <- rpart(TARGET ~ ., data = train, cp = 0.001)
#save(rt2, file = '~/kaggle/santander/data/rt2.rda')
load(file = '~/kaggle/santander/data/rt2.rda')
plotcp(rt2)

#rt3 <- rpart(TARGET ~ ., data = data, cp = 0.0005)
#save(rt3, file = '~/kaggle/santander/data/rt3.rda')
load(file = '~/kaggle/santander/data/rt3.rda')
plotcp(rt3)
printcp(rt3)
rt3.prune <- prune.rpart(rt3, 0.00095393)
plot(rt3.prune, uniform = T, compress = T)
text(rt3.prune, cex = 0.4)

# See diagnostics:
plot(predict(rt3.prune), 
	 resid(rt3.prune), 
	 xlab = 'Fitted', 
	 ylab = 'Residuals')
qqnorm(resid(rt3.prune))
qqline(resid(rt3.prune))

# Test
rt.test <- predict(rt3.prune, newdata = test)
rt.test[rt.test < 0.5] <- 0
rt.test[rt.test > 0.5] <- 1

(conf.mat <- table(actual = test$TARGET, predicted = rt.test))
(accuracy <- sum(diag(conf.mat)) / sum(conf.mat))
# Pretty bad--misclassifies most unsatisfied customers



# Compare w/ log-reg on raw data
mod.log <- glm(TARGET ~ ., data = train, family = binomial)
mod.log <- update(mod.log, . ~ . - saldo_medio_var33_ult1)
mod.log <- update(mod.log, . ~ . - delta_imp_trasp_var33_in_1y3)
mod.log <- update(mod.log, . ~ . - num_aport_var17_hace3)
mod.log <- update(mod.log, . ~ . - imp_trasp_var17_in_hace3)
mod.log <- update(mod.log, . ~ . - saldo_var13_medio)
# CONTINUE REDUCING MOD HERE
#save(mod.log, file = '~/kaggle/santander/data/modlog.rda')
load(file = '~/kaggle/santander/data/modlog.rda')
summary(mod.log)
max(summary(mod.log)$coef[, 4], na.rm = T)
# IN TESTING: remember that all colinear variables were reduced to a single variable.  Check to see if the colinearity persists in the test data, if not, some kind of consensus should be done






# Find which variables correlate best with TARGET
names(data)

target.cors <- data.frame(variable = names(data)[-236], 
						  target.correlation = numeric(235))
head(target.cors)

for (v in 1:235) {
  var <- as.character(target.cors$variable[v])
  target.cors$target.correlation[v] <- round(cor(data[, var], data$TARGET), 4)
}

target.cors[order(abs(target.cors$target.correlation), decreasing = T), ]