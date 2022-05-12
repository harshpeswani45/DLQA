for file in ./manual_json/*; do
    #echo "${file##*/}"
    f="$(basename -- $file)"
    filename="${f%.*}"
    # echo $f
    # echo $filename
    python preprocessing.py ./manual_json/$f $filename
    python preprocessing_list.py ./manual_json/$f $filename
    # python helper.py $f $filename
done