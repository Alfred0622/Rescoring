task=$1

mkdir -p ../src/${task}/{model,config,data,utils,tools}
mkdir -p ../src/${task}/data/{aishell,aishell2,tedlium2,librispeech,csj}/{train,dev,test}/{noLM,withLM}

touch ../src/${task}/utils/__init__.py
touch ../src/${task}/utils/Datasets.py
touch ../src/${task}/utils/CollateFunc.py
touch ../src/${task}/utils/LoadConfig.py

touch ../src/${task}/model/__init__.py