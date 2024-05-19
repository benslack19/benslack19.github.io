#!/bin/bash

###### Usage ########################################################################

# Call this script, using the notebook as an argument from the _posts_drafts directory

# Example: bash nb2md_script.sh 2020-07-28-prior-and-beta.ipynb

# This will create a markdown file from the notebook file.
# Then move the markdown file into the _posts directory, the 
# associated files/figures into the assets directory, then 
# edit the paths in the markdown file to point to the correct path.

####################################################################################


# Create variables, using the notebook name from the argument of the call---------

NB=$1
NB_base=$(echo ${NB} | sed s'|.ipynb||')
NB_md="$NB_base.md"

echo ' '
echo 'File base name: ' ${NB_base}
echo 'Notebook: ' ${NB} 
echo 'Markdown file: ' ${NB_md}
echo ' '

# Convert to markdown format ------------------------------------------------------

jupyter nbconvert --to markdown ${NB}


# Move the associated figures files, in the NB_files folder, to _assets--------------------

mv ${NB_base}_files/ ../assets/${NB_base}_files/

echo 'Moving associated figures ' ${NB_base}_files/ ../_assets/${NB_base}_files/
echo ' '

# Edit path of .png files  ----------------------------------

# I can't move the file first and edit in place with sed
# Better to edit first, create a new file in _posts folder, then delete in current folder

# Find all of the references to .png files in the newly moved markdown script and edit with new path---

# OLD ![png](test_files/test_5_0.png)
# NEW ![png](/assets/test_files/test_5_0.png)

echo 'Before renaming png files'
cat ../_posts/${NB_md} | grep "\[png\](${NB_base}_files"
echo ' '

# sed command was a bit hard to write
# need to use single and double quotes (using comma as separator in sed command
sed 's,'\\[png\]\("$NB_base"_files','\\[png\]\(/assets/"$NB_base"_files',g' ${NB_md} > ../_posts/${NB_md}

echo 'After renaming png files'
cat ../_posts/${NB_md} | grep "\[png\](${NB_base}_files"
echo ' '

# Edit path of .svg files  ---------------------------------------
echo 'Before renaming svg files'
cat ../_posts/${NB_md} | grep "\[svg\](${NB_base}_files"
echo ' '

# sed command was a bit hard to write
# need to use single and double quotes (using comma as separator in sed command
sed 's,'\\[svg\]\("$NB_base"_files','\\[svg\]\(/assets/"$NB_base"_files',g' ${NB_md} > ../_posts/${NB_md}

echo 'After renaming svg files'
cat ../_posts/${NB_md} | grep "\[svg\](${NB_base}_files"
echo ' '


# Move the markdown file to _posts ----------------------------------
echo 'Moved markdown file ' ${NB_md} ../_posts/${NB_md}

# Remove the lines that say `<IPython.core.display.Javascript object>`  (not working)
echo 'Remove random Javascript object text'
# sed '/<IPython.core.display.Javascript object>/d' ./${NB_md} > ./${NB_md}
sed -i '' '/\<IPython.core.display.Javascript object\>/d' ../_posts/${NB_md}
# sed -i '' '/\<IPython.core.display.Javascript object\>/d' ${NB_md}
echo ' '


# Cut out the first line
echo 'Cut out the first line'
# Need to assign to a diff file first. Redirection (>) happens before tail is invoked by the shell.
# https://stackoverflow.com/questions/339483/how-can-i-remove-the-first-line-of-a-text-file-using-bash-sed-script
tail -n +2 ../_posts/${NB_md} > ../_posts/${NB_md}.tmp && mv ../_posts/${NB_md}.tmp ../_posts/${NB_md}
echo ' '

rm ${NB_md}
echo 'Removed markdown file in _posts_drafts folder '
echo ' '

echo "Markdown produced and updated."
echo "cd to parent folder and make sure first line of markdown file is not blank (starts with --- indicating start of YAML)"
echo "Run bundle exec jekyll serve to test page build locally."