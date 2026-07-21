fambench.pdf: fambench.tex references.bib figures/*.pdf
	pdflatex -interaction=nonstopmode fambench.tex
	bibtex fambench
	pdflatex -interaction=nonstopmode fambench.tex
	pdflatex -interaction=nonstopmode fambench.tex

clean:
	rm -f *.aux *.bbl *.blg *.out *.log *.toc
.PHONY: clean
