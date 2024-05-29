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
echo 'Notebook: ' ${NB} 
echo 'File base name: ' ${NB_base}
echo 'Markdown file: ' ${NB_md}
echo ' '

# Convert to markdown format ------------------------------------------------------
jupyter nbconvert --to markdown ${NB}

# Move the associated figures files, in the NB_files folder, to _assets--------------------
mv ${NB_base}_files/ ../assets/${NB_base}_files/
echo 'Moving associated figures ' ${NB_base}_files/ ../_assets/${NB_base}_files/
echo ' '


# Edit path of .png and .svg files  ----------------------------------
# I can't edit the file paths in place with sed or over-write the markdown.
# Better to edit first, create a new file in _posts folder, then delete in current folder

# Find all of the references to .png and svg files in the newly moved markdown script and edit with new path
# OLD ![png](test_files/test_5_0.png)
# NEW ![png](/assets/test_files/test_5_0.png)

echo 'Before renaming png files, markdown in _post_drafts'
cat ${NB_md} | egrep "svg|png"
echo ' '

# sed -E: Enables extended regular expressions
sed -E 's/!\[(png|svg)\]\(([^)]+)\)/![\1]\(\/assets\/\2)/g' ${NB_md} > ../_posts/${NB_md}

echo 'After renaming png files, markdown in _post'
cat ../_posts/${NB_md} | egrep "svg|png"
echo ' '

# Remove the lines that say `<IPython.core.display.Javascript object>`
echo 'Remove random Javascript object text'
sed -i '' '/\<IPython.core.display.Javascript object\>/d' ../_posts/${NB_md}
echo ' '

rm ${NB_md}
echo 'Removed markdown file in _posts_drafts folder '
echo "Updated markdown produced in _posts folder."
echo "cd to parent folder and run bundle exec jekyll serve to test page build locally."
echo " "
