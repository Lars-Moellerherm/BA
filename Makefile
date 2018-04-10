all: main.pdf

p: Python/auswertung.py Python/vorbereitung.py | Latex/build Python/build
	cd Python/ && python auswertung.py && python vorbereitung.py

l: Latex/content/auswertung.tex Latex/content/diskussion.tex Latex/content/durchfuehrung.tex Latex/content/fehlerrechnung.tex Latex/content/theorie.tex Latex/main.tex Latex/header.tex Latex/lit.bib
	cd Latex/ && latexmk --lualatex --output-directory=build main.tex
	cp Latex/build/main.pdf main.pdf

main.pdf: Latex/content/auswertung.tex Latex/content/diskussion.tex Latex/content/durchfuehrung.tex Latex/content/fehlerrechnung.tex Latex/content/theorie.tex Latex/main.tex Latex/header.tex Latex/lit.bib Python/auswertung.py Python/vorbereitung.py | Latex/build Python/build
	cd Python/ && python auswertung.py && python vorbereitung.py
	cd Latex/ && latexmk --lualatex --output-directory=build main.tex
	cp Latex/build/main.pdf main.pdf

Python/build:
	mkdir -p Python/build

Latex/build:
	mkdir -p Latex/build

clean:
	rm -rf Latex/build
	rm -rf Python/build

.PHONY: all clean
