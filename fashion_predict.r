install.packages("tidyr")
library(keras3)
library(ggplot2)
library(dplyr)
library(tidyr)
library(this.path)

# Get directory of THIS script
script_dir <- this.dir()
setwd(script_dir)

class_names <- c(
  "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
)

load_assets <- function() {
  list(
    model = load_model("fashion_mnist_cnn.keras"),
    x_test = readRDS("test_images.rds"),
    y_test = readRDS("test_labels.rds")
  )
}

prepare_image_data <- function(images, actual_labels, predicted_labels) {
  # Convert 4D array to list of data frames
  image_data <- lapply(1:dim(images)[1], function(i) {
    as.data.frame.table(images[i, , , 1]) %>%
      mutate(x = as.numeric(Var2),
             y = 29 - as.numeric(Var1),  # Flip y-axis for proper orientation
             value = Freq,
             label = paste0("Pred: ", class_names[predicted_labels[i] + 1], "\n",
                            "True: ", class_names[actual_labels[i] + 1])) %>%
      select(x, y, value, label)
  })
  
  bind_rows(image_data)
}

plot_predictions <- function(model, x_test, y_test, num_images = 3) {
  # Random sample
  indices <- sample(dim(x_test)[1], num_images)
  images <- x_test[indices, , , , drop = FALSE]
  actual <- y_test[indices]
  
  # Predictions
  predictions <- predict(model, images)
  predicted <- max.col(predictions) - 1
  
  # Prepare data for ggplot
  plot_data <- prepare_image_data(images, actual, predicted)
  
  # Create plot
  ggplot(plot_data, aes(x = x, y = y, fill = value)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "black") +
    facet_wrap(~ label, nrow = 1) +
    theme_void() +
    theme(
      strip.text = element_text(size = 10, face = "bold"),
      legend.position = "none"
    )
}

main <- function() {
  assets <- load_assets()
  print(plot_predictions(assets$model, assets$x_test, assets$y_test))
}

main()