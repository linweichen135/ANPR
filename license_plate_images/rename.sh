ls > filename.txt
idx=1
while read -r filename; do
	if [ $filename != 'rename.sh' ] && [ $filename != 'filename.txt' ] ; then
		echo "Processing $filename ..."
		mv $filename "IMG_$(date +'%Y%m%d')_$(printf '%06d' $idx).png";
		idx=$(expr $idx + 1);
	fi
done < filename.txt
rm filename.txt
