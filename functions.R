library(tidyverse)
library(here)

read_train_data <- function() {
  readr::read_csv(
    file = here("data", "input", "train_data.csv"),
    col_types = cols(
      .default = col_integer(),

      X2  = readr::col_factor(levels = 1:2),
      X3  = readr::col_factor(levels = 0:6),
      X4  = readr::col_factor(levels = 0:3),
      X6  = readr::col_factor(levels = -2:8),
      X7  = readr::col_factor(levels = -2:8),
      X8  = readr::col_factor(levels = -2:8),
      X9  = readr::col_factor(levels = -2:8),
      X10 = readr::col_factor(levels = -2:8),
      X11 = readr::col_factor(levels = -2:8),

      y = readr::col_factor(levels = 0:1)
    )
  ) %>%
    dplyr::select(-id)
}
#read_train_data()

read_test_data <- function() {
  readr::read_csv(
    file = here("data", "input", "test_data.csv"),
    col_types = cols(
      .default = col_integer(),

      X2  = readr::col_factor(levels = 1:2),
      X3  = readr::col_factor(levels = 0:6),
      X4  = readr::col_factor(levels = 0:3),
      X6  = readr::col_factor(levels = -2:8),
      X7  = readr::col_factor(levels = -2:8),
      X8  = readr::col_factor(levels = -2:8),
      X9  = readr::col_factor(levels = -2:8),
      X10 = readr::col_factor(levels = -2:8),
      X11 = readr::col_factor(levels = -2:8)
    )
  )
}
#read_test_data()
