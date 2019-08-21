# "data" is a parameter / add_data.py should be placed in the same directory with directory "data"
for dir in "data"/*
do
    label=$(echo $dir | awk 'BEGIN {FS="/"} {print $2}')
    # "ocr_traindata" and "ocr_traindata/dataset.csv" is a parameter
    python add_data.py $dir ocr_traindata --csv_path ocr_traindata/dataset.csv --label $label
done