
mkdir data
export str=''
for l in "irish" "english" "basque" "tagalog";  do for f in "train" "test" "dev"; do for (( n=0; n<=9; n++ )); do awk 'BEGIN{FS="\t";} {OFS="\t";} {print $1,"",$2;}' ${l}-10fold/${n}/${f}.uniq > data/${l}_${n}_${f}.tmp; str+="'${l}_${n}_',"; done; done; done

echo $str > fold.txt

# remove first blank lines
for f in data/*.tmp
do
awk '{if (NR==1 && NF==0) next};1' $f > ${f%.tmp}.txt
rm $f
done


