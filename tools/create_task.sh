task=$1

mkdir -p ../src/${task}/{model,config,data,utils,tools}
mkdir -p ../src/${task}/data/{aishell,tedlium2}/{noLM,withLM}/{train,dev,test}
mkdir -p ../src/${task}/data/librispeech/{noLM,withLM}/{train,dev_clean,dev_other,test_clean,test_other}
mkdir -p ../src/${task}/data/aishell2/{noLM,withLM}/{train,dev,test_mic,test_ios,test_android}
mkdir -p ../src/${task}/data/csj/{noLM,withLM}/{train,dev,eval1,eval2,eval3}

touch ../src/${task}/utils/__init__.py
touch ../src/${task}/utils/Datasets.py
touch ../src/${task}/utils/CollateFunc.py
touch ../src/${task}/utils/LoadConfig.py

touch ../src/${task}/model/__init__.py