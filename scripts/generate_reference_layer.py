from pathlib import Path
import re, json, csv, textwrap, argparse

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path, help="Extracted gp3ml 0.3.0 source directory")
parser.add_argument("out", type=Path, nargs="?", default=Path("."), help="Repository output directory")
args = parser.parse_args()
root = args.root.resolve()
out = args.out.resolve()
(out/'reference').mkdir(parents=True, exist_ok=True)
(out/'tests').mkdir(parents=True, exist_ok=True)
(out/'docs'/'reference').mkdir(parents=True, exist_ok=True)
(out/'docs'/'articles').mkdir(parents=True, exist_ok=True)
(out/'examples').mkdir(parents=True, exist_ok=True)

ns=(root/'NAMESPACE').read_text()
exports=sorted(re.findall(r'^export\(([^)]+)\)$', ns, flags=re.M))
prints=re.findall(r'^S3method\(print,([^)]+)\)$',ns,flags=re.M)
plots=re.findall(r'^S3method\(plot,([^)]+)\)$',ns,flags=re.M)
api=(root/'R'/'api-contracts.R').read_text()
def parse_c(name):
    m=re.search(re.escape(name)+r'\s*<-\s*c\((.*?)\)\s*\n',api,re.S)
    if not m: return []
    return re.findall(r'"([^"]+)"',m.group(1))
stable=parse_c('.gp3ml_contract_baseline_exports')
classes=parse_c('.gp3ml_contract_baseline_classes')
experimental=parse_c('.gp3ml_contract_experimental_exports')
assert set(exports)==set(stable)|set(experimental), (len(exports),len(stable),len(experimental))

# Export inventory
with (out/'reference'/'r_api_inventory.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['name','stability','r_reference_version'])
    for name in exports: w.writerow([name,'stable' if name in stable else 'experimental','0.3.0'])
(out/'reference'/'r_public_classes.json').write_text(json.dumps(classes,indent=2)+"\n")
(out/'reference'/'r_print_methods.json').write_text(json.dumps(sorted(prints),indent=2)+"\n")
(out/'reference'/'r_plot_methods.json').write_text(json.dumps(sorted(plots),indent=2)+"\n")
articles=sorted(p.name for p in (root/'vignettes').glob('*.Rmd'))
(out/'reference'/'r_articles.json').write_text(json.dumps(articles,indent=2)+"\n")
tests=sorted(str(p.relative_to(root)) for p in (root/'tests'/'testthat').glob('*.R'))
(out/'reference'/'r_tests_inventory.json').write_text(json.dumps(tests,indent=2)+"\n")

# Rd examples and reference pages
rd_files=sorted((root/'man').glob('*.Rd'))
example_topics=[]
rd_by_alias={}
for p in rd_files:
    txt=p.read_text(errors='replace')
    aliases=re.findall(r'\\alias\{([^}]+)\}',txt)
    for a in aliases: rd_by_alias[a]=p
    if '\\examples{' in txt and any(a in exports for a in aliases):
        example_topics.extend(a for a in aliases if a in exports)
example_topics=sorted(set(example_topics))
(out/'reference'/'r_example_inventory.json').write_text(json.dumps(example_topics,indent=2)+"\n")

# explicit expect_error contract inventory
failure=[]
for p in sorted((root/'tests'/'testthat').glob('*.R')):
    text=p.read_text(errors='replace')
    # capture line number and compact call neighborhood
    for m in re.finditer(r'expect_error\s*\(',text):
        line=text.count('\n',0,m.start())+1
        segment=text[m.start():m.start()+900]
        # best-effort exported function involved
        fn=''
        for n in exports:
            if re.search(r'\b'+re.escape(n)+r'\s*\(',segment): fn=n; break
        # string matcher in expect_error call
        strings=re.findall(r'"([^"\n]{1,160})"',segment[:700])
        failure.append({'test_file':p.name,'line':line,'export':fn,'expected_fragment':strings[-1] if strings else ''})
(out/'reference'/'r_failure_contracts.json').write_text(json.dumps(failure,indent=2)+"\n")

summary={
 'r_reference_version':'0.3.0','exports':len(exports),'stable_exports':len(stable),'experimental_exports':len(experimental),
 'stable_public_classes':len(classes),'r_source_files':len(list((root/'R').glob('*.R'))),'rd_files':len(rd_files),
 'vignettes':len(articles),'r_test_files':len(tests),'print_methods':len(prints),'plot_methods':len(plots),
 'rd_example_topics':len(example_topics),'explicit_expect_error_contracts':len(failure)
}
(out/'reference'/'reference_summary.json').write_text(json.dumps(summary,indent=2)+"\n")

# Generate source-derived reference docs (concise, no invention)
def one_cmd(txt, cmd):
    m=re.search(r'\\'+cmd+r'\{([^}]*)\}',txt,re.S); return m.group(1).strip() if m else ''
def clean_rd(s):
    s=re.sub(r'\\code\{([^{}]*)\}',r'`\1`',s)
    s=re.sub(r'\\link\{([^{}]*)\}',r'\1',s)
    s=re.sub(r'\\pkg\{([^{}]*)\}',r'\1',s)
    s=re.sub(r'\\emph\{([^{}]*)\}',r'\1',s)
    s=re.sub(r'\\strong\{([^{}]*)\}',r'**\1**',s)
    s=re.sub(r'\\[^\s{]+\{([^{}]*)\}',r'\1',s)
    s=s.replace('\\%','%').replace('\\_','_')
    return re.sub(r'\s+',' ',s).strip()
for name in exports:
    p=rd_by_alias.get(name)
    title=name; desc=''
    usage=''
    if p:
        txt=p.read_text(errors='replace')
        title=clean_rd(one_cmd(txt,'title')) or name
        desc=clean_rd(one_cmd(txt,'description'))
        m=re.search(r'\\usage\{(.*?)\n\}',txt,re.S)
        usage=(m.group(1).strip() if m else '')
    body=f"# `{name}`\n\n**R reference:** gp3ml 0.3.0.\n\n"
    if title and title!=name: body += f"## {title}\n\n"
    if desc: body += desc+"\n\n"
    if usage: body += "## R reference usage\n\n```r\n"+usage+"\n```\n\n"
    body += "The Python implementation is exported as `gp3mlpy.%s`. See the runtime docstring for Python-specific typing and semantic adaptations.\n"%name
    (out/'docs'/'reference'/f'{name}.md').write_text(body)

(out/'docs'/'reference'/'index.md').write_text('# API reference\n\n'+'\n'.join(f'- [`{n}`]({n}.md)' for n in exports)+'\n')

# Article pages preserve source title/prose in a deliberately compact source-derived form.
for p in sorted((root/'vignettes').glob('*.Rmd')):
    text=p.read_text(errors='replace')
    # YAML title
    tm=re.search(r'^title:\s*["\']?(.+?)["\']?\s*$',text,re.M)
    title=tm.group(1) if tm else p.stem
    # remove yaml and R chunks, retain prose headings/paragraphs
    prose=re.sub(r'^---.*?---\s*','',text,flags=re.S)
    prose=re.sub(r'```\{r[^}]*\}.*?```','',prose,flags=re.S)
    prose=re.sub(r'```r.*?```','',prose,flags=re.S)
    prose=prose.strip()
    page=f'# {title}\n\n> Source-derived companion to `gp3ml` 0.3.0 vignette `{p.name}`. R code blocks are omitted here; the Python companion script is under `examples/{p.stem}.py`.\n\n{prose}\n'
    (out/'docs'/'articles'/f'{p.stem}.md').write_text(page)
(out/'docs'/'articles'/'index.md').write_text('# Articles\n\n'+'\n'.join(f'- [{p.stem}]({p.stem}.md)' for p in sorted((root/'vignettes').glob('*.Rmd')))+'\n')
(out/'docs'/'index.md').write_text('# gp3mlpy\n\nPython port of gp3ml 0.3.0 with frozen API and governance contracts.\n\n- [API reference](reference/index.md)\n- [Articles](articles/index.md)\n')

# Runnable article companions: each imports package and verifies frozen API; selected articles execute relevant workflow.
common='''"""Runnable companion generated for the corresponding gp3ml 0.3.0 vignette."""\nimport gp3mlpy as gp\n\nreg = gp.gp3ml_api_contracts()\nassert len(reg.exports) == 127\nassert gp.r_reference_version == "0.3.0"\n'''
for p in sorted((root/'vignettes').glob('*.Rmd')):
    extra=''
    if p.stem in {'participant-generalization','stimulus-generalization','participant-stimulus-generalization','assigned-condition-discrimination','integrated-research-workflow','recording-quality-review'}:
        target={'participant-generalization':'new_participants','stimulus-generalization':'new_stimuli','participant-stimulus-generalization':'new_participants_and_new_stimuli'}.get(p.stem,'new_trials_known_participants')
        workflow='assigned_condition' if p.stem=='assigned-condition-discrimination' else 'recording_quality'
        extra=f'''\ndata = gp.simulate_gazepoint_governed_data(n_participants=12, n_stimuli=4, trials_per_cell=1, seed=17)\ntask = gp.create_gazepoint_synthetic_task(data, workflow="{workflow}", generalization_target="{target}")\nassert task.generalization_target == "{target}"\n'''
    (out/'examples'/f'{p.stem}.py').write_text(common+extra)

print(json.dumps(summary,indent=2))
