if [ ! -d "genie-toolkit" ]
then
    git clone https://github.com/stanford-oval/genie-toolkit.git
    cd genie-toolkit
    git checkout wip/levenshtein-exp
  
    if test `id -u` = 0 ; then
        su genie-toolkit -c "npm ci"
    else
        npm ci
    fi
    
    cd ..
fi

# if [ ! -d "genie-workdirs" ]
# then
#     git clone https://github.com/stanford-oval/genie-workdirs.git
#     cd genie-workdirs
#     git checkout wip/levenshtein

#     cd ..
# fi

node --experimental_worker $(pwd)/genie-toolkit/dist/tool/genie.js evaluate-file -o output.txt --predictions ${1} --contextual $(pwd)/genie-workdirs/dlgthingtalk/multidomain/eval_fewshot/eval.tsv --thingpedia $(pwd)/genie-workdirs/dlgthingtalk/multidomain/schema.tt --parameter-datasets $(pwd)/genie-workdirs/dlgthingtalk/shared-parameter-datasets.tsv > /dev/null