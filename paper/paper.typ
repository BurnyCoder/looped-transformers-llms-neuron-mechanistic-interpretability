// Paper source (Typst, https://typst.app/docs/). Compiled by `uv run python paper/build.py` -> paper/paper.pdf.
// Every number in the Results section is copied from results/Ouro-1.4B/*.json (see reports/ for the mapping).
#set document(title: "Neurons of a Looped Language Model", author: "Libor Burian")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#show link: set text(fill: rgb("#2a78d6"))

#align(center)[
  #text(16pt, weight: "bold")[Neurons of a Looped Language Model: \ Neuron-Level Interpretability and Pre-Registered Causal Steering of Ouro-1.4B]
  #v(4pt)
  Libor Burian
  #v(2pt)
  #text(9pt)[Code: #link("https://github.com/BurnyCoder/looped-transformers-llms-neuron-mechanistic-interpretability")]
]

#include "sections/abstract.typ"
#include "sections/body.typ"

#bibliography("refs.bib", style: "ieee")

#include "sections/ai_statement.typ"
