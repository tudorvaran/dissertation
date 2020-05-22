fis=thesis
if [ "$#" -eq 1 ]; then
  fis="$1"
fi

pdflatex "$fis.tex"


#latex -interaction=nonstopmode "$fis.tex"
#bibtex "$fis.aux"
#latex -interaction=nonstopmode "$fis.tex"
#latex -interaction=nonstopmode "$fis.tex"
#dvips -o "$fis.ps" "$fis.dvi"
#ps2pdf "$fis.ps"