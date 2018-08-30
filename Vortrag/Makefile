all: build/vortrag.pdf


TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build

n: Notizen.tex | build
	latexmk $(TeXOptions) Notizen.tex

build/vortrag.pdf: FORCE | build
	latexmk $(TeXOptions) Vortrag.tex

FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
