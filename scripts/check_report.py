"""Static checks on report/report.tex for machines without LaTeX: references, citations,
included files, environment and brace balance.

    python scripts/check_report.py
"""

import os
import re
from collections import Counter

s = open("report/report.tex", encoding="utf-8").read()
labels = set(re.findall(r"\\label\{([^}]+)\}", s))
refs = set(re.findall(r"\\ref\{([^}]+)\}", s))
print("refs without a label:", sorted(refs - labels))
print("labels never referenced:", sorted(labels - refs))
bib = set(re.findall(r"\\bibitem\{([^}]+)\}", s))
cites = {k.strip() for c in re.findall(r"\\cite\{([^}]+)\}", s) for k in c.split(",")}
print("citations without an entry:", sorted(cites - bib))
print("entries never cited:", sorted(bib - cites))
files = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s)
files += [x + ".tex" for x in re.findall(r"\\input\{([^}]+)\}", s)]
print("missing files:", [f for f in files if not os.path.exists("report/" + f)])
begins, ends = Counter(re.findall(r"\\begin\{(\w+\*?)\}", s)), Counter(re.findall(r"\\end\{(\w+\*?)\}", s))
print("unbalanced environments:", dict((begins - ends) + (ends - begins)))
body = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", "", s, flags=re.S)
body = re.sub(r"(?<!\\)%.*", "", body)
plain = body.replace("\\{", "").replace("\\}", "").replace("\\$", "")
print("brace balance (0 is good):", plain.count("{") - plain.count("}"))
print("odd number of $ (False is good):", plain.count("$") % 2 == 1)
print("undefined placeholders:", re.findall(r"\\RESULT\w+", s))
main = body.split("\\begin{thebibliography}")[0]
print("approx. words in main text:", len(re.findall(r"[A-Za-z]{2,}", re.sub(r"\\[a-zA-Z]+", " ", main))))
