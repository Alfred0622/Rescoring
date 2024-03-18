# CLM and MLM
## Training
`$ python ./train_LM.py <CLM or MLM>`
如果要改設定，必須去`./config/<CLM or MLM>.yaml`改

## Predict
`$ python ./predict_CLM.py <checkpoint_path> <result_file_name>`
`$ python ./predict_MLM.py <checkpoint_path>`
checkpoint_path 建議用絕對路徑

## 注意
會有一個for_train的選項，目前`predict_CLM.py`我寫在該檔案中 `predict_MLM.py`我寫在`MLM.yaml`中
請注意設定，設成True會decode training file
