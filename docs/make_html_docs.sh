replace_static() {
    # This function replaces the word "_static" by the word "static" in all html files in current folder
    for file in `ls | grep .html`; do
        sed 's/_static/static/' $file > tmp
        mv tmp $file
    done
   }



# script to make the html documentation
make clean
sphinx-apidoc -f -o source/generated/ ../LambdaZero
make html

# replace _static with static everywhere; folders starting with an underscore (like _static) are ignored by github.io
cd ./build/html/
mv _static/ static/
replace_static
cd ./generated/
replace_static
cd ../../../

# remove all the needless rst files
rm -rf ./source/generated/
