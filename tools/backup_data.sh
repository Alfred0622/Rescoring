dataset=('aishell' 'aishell2' 'csj' 'librispeech' 'tedlium2')
setting=('noLM' 'withLM')
# echo ${dataset[*]}
# echo "${dataset[@]}"
for d in "${dataset[@]}"; do
    for s in "${setting[@]}"; do
        echo $d $s
        if [ "$d" = 'aishell' ] 
        then
            recog_set=('train' 'dev' 'test')
        elif [ "$d" = 'tedlium2' ]
        then
            recog_set=('train' 'dev' 'test')
        elif [ "$d" = 'aishell2' ]
        then
            recog_set=('train' 'dev_ios' 'test_ios' 'test_mic' 'test_android')
        elif [ "$d" = 'csj' ]
        then
            recog_set=('train' 'dev' 'eval1' 'eval2' 'eval3')
        elif [ "$d" = 'librispeech' ]
        then
            recog_set=('train' 'dev_clean' 'dev_other' 'test_clean' 'test_other')
        fi

        echo "${recog_set[@]}"
        for r in "${recog_set[@]}"; do
            mkdir -p /mnt/disk4/Alfred/all_Backup/data/${d}/${s}/${r}
            echo /mnt/disk6/Alfred/Rescoring/data/${d}/data/${s}/${r}/data.json "->" /mnt/disk4/Alfred/all_Backup/data/${d}/${s}/${r}/data.json
            cp /mnt/disk6/Alfred/Rescoring/data/${d}/data/${s}/${r}/data.json /mnt/disk4/Alfred/all_Backup/data/${d}/${s}/${r}/data.json
        done
    done
done