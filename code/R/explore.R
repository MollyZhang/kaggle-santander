# Script for exploratory analysis of the santander training set

# clear workspace
rm(list = ls())

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








# IN TESTING: remember that all colinear variables were reduced to a single variable.  Check to see if the colinearity persists in the test data, if not, some kind of consensus should be done
