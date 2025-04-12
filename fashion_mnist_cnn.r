install.packages("this.path")
library(keras3)
library(this.path)

# Get directory of THIS script
script_dir <- this.dir()

# Set working directory to script location
setwd(script_dir)

build_model <- function() {
  keras_model_sequential(name = "fashion_mnist_cnn") %>%
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                  input_shape = c(28, 28, 1)) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 10, activation = "softmax") %>%
    compile(
      optimizer = optimizer_adam(),
      loss = loss_sparse_categorical_crossentropy(),
      metrics = metric_sparse_categorical_accuracy()
    )
}

main <- function() {
  # Load data
  fashion_mnist <- dataset_fashion_mnist()
  x_train <- fashion_mnist$train$x
  y_train <- fashion_mnist$train$y
  x_test <- fashion_mnist$test$x
  y_test <- fashion_mnist$test$y

  # Preprocess
  x_train <- array_reshape(x_train, c(dim(x_train)[1], 28, 28, 1)) / 255
  x_test <- array_reshape(x_test, c(dim(x_test)[1], 28, 28, 1)) / 255

  # Build and train model
  model <- build_model()
  model %>% fit(
    x_train, y_train,
    epochs = 10,
    batch_size = 128,
    validation_split = 0.2
  )
  # Save to script directory
  model %>% save_model("fashion_mnist_cnn.keras")
  saveRDS(x_test, "test_images.rds")
  saveRDS(y_test, "test_labels.rds")
  message("Files saved in: ", getwd())
}

main()