replacestatic() {
    # This function replaces the word "static" by the word "static" in all html files in current folder
    for file in `ls | grep .html`; do
        sed 's/static/static/' $file > tmp
        mv tmp $file
    done
   }



# script to make the html documentation
make clean
sphinx-apidoc -f -o source/generated/ ../LambdaZero
make html

# replace static with static everywhere; folders starting with an underscore (like static) are ignored by github.io
cd ./build/html/
mv static/ static/
replacestatic
cd ./generated/
replacestatic
cd ../../../

# remove all the needless rst files
rm -rf ./source/generated/
