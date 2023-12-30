default_fname="saves.zip"
fname="${1:-$default_fname}"

wget "https://raw.githubusercontent.com/MarkelZ/robust-models/main/$fname"
unzip "$fname"
rm "$fname"