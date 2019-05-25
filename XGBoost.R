library(tidyverse)
library(tidymodels)
library(doParallel)

set.seed(1025)

source("./functions.R")

df.train <- read_train_data()
#df.train <- read_train_data() %>%
#  rsample::initial_split(prop = 1/3, strata = "y") %>%
#  rsample::training()
#summary(df.train)

# 前処理の定義
rec <- recipes::recipe(y ~ ., data = df.train) %>%
  
  recipes::step_log(X1) %>%
  recipes::step_log(X18, X19, X20, X21, X22, X23, offset = 1) %>%
  
  recipes::step_mutate(
    # age_segment
    age_segment = dplyr::case_when(
      X5 <= 20 ~ "lte20",
      X5 <= 30 ~ "lte30",
      X5 <= 40 ~ "lte40",
      X5 <= 50 ~ "lte50",
      X5 <= 60 ~ "lte60",
      X5 <= 70 ~ "lte70",
      X5 <= 80 ~ "lte80",
      T ~ "other"
    ) %>% factor()
  ) %>%
  
  recipes::step_mutate(
    X12_ratio = X12 / X1,
    X13_ratio = X13 / X1,
    X14_ratio = X14 / X1,
    X15_ratio = X15 / X1,
    X16_ratio = X16 / X1,
    X17_ratio = X17 / X1,
    
    X18_ratio = X18 / X1,
    X19_ratio = X19 / X1,
    X20_ratio = X20 / X1,
    X21_ratio = X21 / X1,
    X22_ratio = X22 / X1,
    X23_ratio = X23 / X1,
    
    # X2 ~ X3
    flg_X2_1__X3_3 = ((X2 == "1") & (X3 == "3")) %>% as.integer(),
    flg_X2_2__X3_4 = ((X2 == "2") & (X3 == "4")) %>% as.integer(),
    
    # X2 ~ X4
    flg_X2_1__X4_3 = ((X2 == "1") & (X4 == "3")) %>% as.integer(),
    #    flg_X2_2__X4_0 = (X2 == "2") & (X4 == "0"),
    
    # X3 ~ X4
    flg_X3_2_3__X4_3 = ((X3 %in% c("2", "3")) & (X4 == "3")) %>% as.integer(),
    #    flg_X3_0__X4_1_2 = (X3 == "0") & (X4 %in% c("1", "2")),
    #    flg_X3_1__X4_0 = (X3 == "1") & (X4 == "0"),
    #    flg_X3_4_5_6__X4_3 = (X3 %in% c("4", "5", "6")) & (X4 == "3"),
    
    # X2 ~ age_segment
    flg_X2_1__age_segment_lte60 = ((X2 == "1") & (age_segment == "lte60")) %>% as.integer(),
    flg_X2_2__age_segment_lte40 = ((X2 == "2") & (age_segment == "lte40")) %>% as.integer(),
    flg_X2_2__age_segment_lte70 = ((X2 == "2") & (age_segment == "lte70")) %>% as.integer(),
    #    flg_X2_2__age_segment_lte80 = (X2 == "2") & (age_segment == "lte80"),
    
    # X3 ~ age_segment
    flg_X3_2__age_segment_lte70 = ((X3 == "2") & (age_segment == "lte70")) %>% as.integer(),
    #    flg_X3_3__age_segment_lte80 = (X3 == "3") & (age_segment == "lte80"),
    flg_X3_4__age_segment_lte40 = ((X3 == "4") & (age_segment == "lte40")) %>% as.integer(),
    #    flg_X3_5__age_segment_lte60 = (X3 == "5") & (age_segment == "lte60"),
    #    flg_X3_5__age_segment_lte70 = (X3 == "5") & (age_segment == "lte70"),
    #    flg_X3_6__age_segment_lte40 = (X3 == "6") & (age_segment == "lte40"),
    #    flg_X3_6__age_segment_lte70 = (X3 == "6") & (age_segment == "lte70"),
    
    # X4 ~ age_segment
    #    flg_X4_0__age_segment_lte30_lte50 = (X4 == "0") & (age_segment %in% c("lte30", "lte50")),
    #    flg_X4_0__age_segment_lte40_lte60 = (X4 == "0") & (age_segment %in% c("lte40", "lte60")),
    #    flg_X4_1__age_segment_lte80 = (X4 == "1") & (age_segment == "lte80"),
    flg_X4_3__age_segment_lte30 = ((X4 == "3") & (age_segment == "lte30")) %>% as.integer(),
    flg_X4_3__age_segment_lte60 = ((X4 == "3") & (age_segment == "lte60")) %>% as.integer(),
    
    role = "flags"
  ) %>%
  
  # min-max scaling: 0〜1
  recipes::step_range(all_numeric(), min = 0, max = 1) %>%
  
  # カテゴリ値のダミー変数化
  recipes::step_dummy(all_nominal(), -all_numeric(), -all_outcomes(), role = "dummies")


# ### CV: 最適な木の数を検証 ###
# rec %>%
#   recipes::prep() %>%
#   recipes::juice() %>%
#   {
#     df.data <- (.)
#     y <- as.integer(df.data$y) - 1
#     x <- df.data %>% dplyr::select(-y) %>% as.matrix()
#     xgboost::xgb.cv(
#       param = list(
#         "objective" = "binary:logistic",
#         "eval_metric" = "error"
#       ),
#       data = x,
#       label = y,
#       nfold = 5,
#       eta = 0.1,
# 
#       # max_depth = 6,
#       # colsample_bytree = 0.37,
#       # min_child_weight = 3,
#       # gamma = 0.03,
# 
#       seed = 42,
# #      nrounds = 2000
#       nrounds = 200
#     )
#   }
# # 最適な木の数は 48 本 => trees


# Model 作成
clf <- parsnip::boost_tree(
  mode = "classification",
  learn_rate = 0.1,
  trees = 48,
  mtry = 47,
#  mtry = parsnip::varying(),

  sample_size = round(nrow(df.train) * 0.9),
  # min_n = 2,
  # tree_depth = 6,
  # loss_reduction = 0,
  # sample_size = parsnip::varying(),
  min_n = parsnip::varying(),
  tree_depth = parsnip::varying(),
  loss_reduction = parsnip::varying()
) %>%
  parsnip::set_engine(engine = "xgboost", seed = 42)


# ハイパーパラメータ
grid.params <- dials::grid_regular(
#  dials::mtry %>% dials::range_set(c(43, 47)),       # colsample_bytree

  # # subsample: range の設定だけだとダメだった
  # dials::sample_size %>%
  #   dials::range_set(c(5, 10)),
  # # %>%
  # #   dials::value_set(seq(0.5, 1, 0.1)),

  dials::min_n          %>% dials::range_set(c(2, 2)),          # min_child_weight
  dials::tree_depth     %>% dials::range_set(c(5, 5)),          # max_depth
  dials::loss_reduction %>% dials::range_set(c(0.1, 0.1)),    # gamma
  levels = 3
)
models <- grid.params %>%
  merge(clf, .)


# V-Fold split
cv <- rsample::vfold_cv(data = df.train, v = 5, strata = "y")

cluster <- parallel::detectCores() %>%
  parallel::makeCluster()
#cluster <- parallel::makeCluster(12)
doParallel::registerDoParallel(cluster)

lst.scores <- foreach::foreach(model = models, .packages = c("tidyverse", "recipes")) %dopar% {
  # 予測 & AUC算出
  cv %>%
    dplyr::mutate(
      recipe = purrr::map(splits, ~ recipes::prepper(.x, rec)),
      model.fitted = purrr::map2(splits, recipe, function(splits, recipe) {
        recipe %>%
          recipes::juice() %>%
          parsnip::fit(model, y ~ ., data = .)
      }),
      accuracy = purrr::pmap_dbl(list(splits, recipe, model.fitted), function(splits, recipe, model) {
        print(model.fitted)
        
        rsample::assessment(splits) %>%
          recipes::bake(recipe, .) %>%
          {
            df.baked <- (.)
            df.baked %>%
              dplyr::mutate(
                pred = parsnip::predict_class(model, df.baked)
              )
          } %>%
          yardstick::accuracy(y, pred) %>%
          .$.estimate
      })
    ) %>%
    dplyr::summarise(mean_accuracy = mean(accuracy))
}

doParallel::stopImplicitCluster()


df.param_scores <- grid.params %>%
  dplyr::mutate(
    accuracy = purrr::map_dbl(lst.scores, ~ .$mean_accuracy)
  ) %>%
  dplyr::arrange(desc(accuracy))
df.param_scores %>%
  View



# mtry: 45, accuracy: 0.8188148
# mtry: 40, accuracy: 0.8174445
# mtry: 30, accuracy: 0.8162963
