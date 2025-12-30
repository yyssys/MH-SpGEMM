#!/bin/bash

root_dir="../matrix"
executable="./spgemm"
matrix_list="./16matrix.txt"

if [ ! -x "$executable" ]; then
    echo "Error: Executable $executable not found or not executable."
    exit 1
fi

if [ ! -f "$matrix_list" ]; then
    echo "Error: Matrix list file $matrix_list not found."
    exit 1
fi

total_count=$(grep -c . "$matrix_list")
echo "Total matrices to process: $total_count"

count=0
while IFS= read -r matrix_name || [[ -n "$matrix_name" ]]; do
    ((count++))
    mtx_file="$root_dir/$matrix_name/$matrix_name.mtx"

    if [ -f "$mtx_file" ]; then
        echo "[$count/$total_count] Processing: $mtx_file"
        "$executable" "$mtx_file"
        ret=$?
        if [ $ret -ne 0 ]; then
            echo "Error: Failed to process $mtx_file"
            exit 1
        fi
    else
        echo "Warning: File not found: $mtx_file"
    fi
    sleep 1
done < "$matrix_list"

echo "All listed matrices processed successfully."
