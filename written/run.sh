fis=thesis
if [ "$#" -eq 1 ]; then
  fis="$1"
fi

#bibtex ""
#pdflatex "$fis.tex"


latex -interaction=nonstopmode $fis.tex > /dev/null
set -e
biber $fis
set +e
echo ""
echo ""
latex -interaction=nonstopmode $fis.tex > /dev/null
latex -interaction=nonstopmode $fis.tex
dvips -o $fis.ps $fis.dvi
ps2pdf $fis.ps

#latex -interaction=nonstopmode "$fis.tex"
#bibtex "$fis.aux"
#latex -interaction=nonstopmode "$fis.tex"
#latex -interaction=nonstopmode "$fis.tex"
#dvips -o "$fis.ps" "$fis.dvi"
#ps2pdf "$fis.ps"