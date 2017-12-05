require(data.table)
require(recosystem)
train <- fread(input="train_py.csv")

# split the dataset into the training and validation sets
train31 <- train[1:5000000]
train32 <- train[5000001:7377418]

# Tune the parameters
train_set = data_memory(user_index=train31$msno,item_index=train31$song_id,rating=train31$target)
r = Reco()
opts = r$tune(train_set, opts = list(dim = c(10, 20, 30), lrate = c(0.1, 0.2),
                                     costp_l1 = 0, costq_l1 = 0,
                                     nthread = 1, niter = 10))

# Train
r$train(train_set, opts = c(opts$min, nthread = 4, niter = 20))

# Validation
test_set <- data_memory(user_index=train32$msno,item_index=train32$song_id)
pred_rvec = r$predict(test_set, out_memory())
train32[,predtarget:=pred_rvec]

fastAUC <- function(probs, class) {
     x <- probs
     y <- class
     x1 = x[y==1]; n1 = length(x1); 
     x2 = x[y==0]; n2 = length(x2);
     r = rank(c(x1,x2))  
     auc = (sum(r[1:n1]) - n1*(n1+1)/2) / n1 / n2
     return(auc)
 }

score <- fastAUC(train32$predtarget2,train32$target)

# Prediction
test <- fread(input="test_py.csv")
test_set <- data_memory(user_index=test$msno,item_index=test$song_id)
pred_rvec = r$predict(test_set, out_memory())
test[,target:=pred_rvec]

nor1 <- function(x) {
 	f <- (x - min(x))/(max(x)-min(x))
 	return(f)
 }
test[,target:=nor1(target)]
submission <- test[,c("id","target")]