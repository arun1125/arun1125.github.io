file_name = arg1
date = arg2
output_name = arg3

nbdev_nb2md file_name.ipynb 
python upd_md.py file_name.md
mv file_name_files images/ 
mv file_name_files images/
mv file_name.md _posts/{date}-output_name.md



# OUTCOMES, how you did it

