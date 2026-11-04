"""Render this repository's limited Markdown syntax into a self-contained offline reader.

The source documents remain authoritative. No network, JavaScript libraries or
third-party page copies are required. Supported syntax is intentionally limited
to the headings, paragraphs, tables, links and fenced code used by these files.
"""
from pathlib import Path
import html
import os
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/offline/index.html'
PAGES=[('algorithms','算法与处理顺序'),('model-card','CNN 模型'),('training','数据与训练'),
       ('data-format','原始数据'),('raspberry-pi','树莓派运行'),('validation','验证与边界')]


def inline(text,source):
    value=html.escape(text)
    def link(match):
        label,url=match.groups()
        if not url.startswith(('http:','https:','#')):
            url=Path(os.path.relpath((source.parent/html.unescape(url)).resolve(),OUT.parent)).as_posix()
        return f'<a href="{html.escape(html.unescape(url),quote=True)}">{label}</a>'
    value=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,value)
    value=re.sub(r'`([^`]+)`',r'<code>\1</code>',value)
    return re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',value)


def render(source):
    lines=source.read_text(encoding='utf-8').splitlines()
    out=[]; paragraph=[]; i=0
    def flush():
        if paragraph:
            out.append('<p>'+inline(' '.join(paragraph),source)+'</p>'); paragraph.clear()
    while i<len(lines):
        line=lines[i]
        if line.startswith('```'):
            flush(); code=[]; i+=1
            while i<len(lines) and not lines[i].startswith('```'):
                code.append(lines[i]); i+=1
            out.append('<pre><code>'+html.escape('\n'.join(code))+'</code></pre>')
        elif line.startswith('|'):
            flush(); rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                row=[x.strip() for x in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+',x) for x in row): rows.append(row)
                i+=1
            out.append('<div class="table"><table>')
            for j,row in enumerate(rows):
                tag='th' if j==0 else 'td'
                out.append('<tr>'+''.join(f'<{tag}>{inline(x,source)}</{tag}>' for x in row)+'</tr>')
            out.append('</table></div>'); continue
        elif line.startswith('#'):
            flush(); depth=min(len(line)-len(line.lstrip('#'))+1,5)
            out.append(f'<h{depth}>'+inline(line.lstrip('#').strip(),source)+f'</h{depth}>')
        elif not line.strip(): flush()
        else: paragraph.append(line)
        i+=1
    flush()
    return '\n'.join(out)


def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    nav=''.join(f'<a href="#{name}">{title}</a>' for name,title in PAGES)
    content='\n'.join(f'<section id="{name}">{render(ROOT/"docs"/(name+".md"))}</section>' for name,_ in PAGES)
    document='''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MLX90640 Thermal Pipeline · 离线说明</title>
<style>
:root{font-family:system-ui,"Microsoft YaHei",sans-serif;color:#172e38;background:#f3f6f5;line-height:1.8}
*{box-sizing:border-box}body{margin:0}main{max-width:1050px;margin:30px auto;padding:36px 48px;background:white;border:1px solid #dce5e2;border-radius:12px}
header{border-top:5px solid #166a63;padding-top:20px}h1{font-size:2.1rem;line-height:1.3}h2{font-size:1.65rem}h3{font-size:1.18rem;margin-top:26px}
p{margin:12px 0}a{color:#086b65;text-underline-offset:.18em}nav{display:flex;gap:10px 22px;flex-wrap:wrap;background:#eaf4ef;padding:15px;margin:24px 0}
section{border-top:1px solid #d5e0dc;margin-top:34px;padding-top:20px;scroll-margin:15px}pre{overflow:auto;background:#f0f4f5;padding:16px;border-radius:6px;font-size:.87rem;line-height:1.6}
code{font-family:Consolas,monospace;overflow-wrap:anywhere}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.92rem}th,td{padding:9px 12px;text-align:left;border-bottom:1px solid #dce5e2}th{background:#edf4f0}
.note{background:#fdf6e9;border-left:4px solid #b58335;padding:12px 17px}.flow{background:#eaf4ef;padding:18px;font-weight:650;color:#134f49;border-radius:6px}.small{font-size:.9rem;color:#526b71}
footer{border-top:1px solid #dce5e2;margin-top:34px;padding-top:15px}@media(max-width:680px){main{margin:0;padding:20px;border:0;border-radius:0}h1{font-size:1.75rem}}
@media print{body,main{background:white;margin:0;border:0}nav{display:none}pre{white-space:pre-wrap}h2,h3{break-after:avoid}}
</style></head><body><main><header><div class="small">SOURCE RELEASE 0.1.0 · 2026-11-04 · MIT</div>
<h1>MLX90640 温度处理完整说明</h1><p>从原始子页、运动自适应频域去交错，到时间滤波与单帧 CNN 重建。</p>
<div class="flow">原始温度 32×24 → 保留当前观测的去交错 → 时间 EMA → CNN 64×48</div>
<p class="small">同时提供先 CNN 后 EMA 的对照；两条顺序共享运动权重。</p>
<div class="note">本页解释活动 v3 实现。64×48 为重建网格；合成误差与静态波动不能代替真实空间分辨率和测温验证。离线阅读无需联网，外部文献链接需联网打开。</div>
</header><nav>'''+nav+'</nav>'+content+'''<footer><a href="../../README.md">仓库 README</a> · <a href="../../THIRD_PARTY_NOTICES.md">来源与许可</a> · <a href="../releasing.md">发布说明</a><p class="small">由 tools/build_offline_docs.py 从仓库说明生成。原始 Markdown 文档为内容来源。</p></footer></main></body></html>'''
    with OUT.open('w',encoding='utf-8',newline='\n') as stream:
        stream.write(document)
    print(f'Generated {OUT.name}: {len(document.encode())} bytes')


if __name__ == '__main__':
    main()
