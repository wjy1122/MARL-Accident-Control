# 论文编译说明

面向本仓库研究主题的中文学术论文草稿，位于 `paper/`。

## 文件

| 文件 | 说明 |
|------|------|
| `main.tex` | 正文（ctexart） |
| `references.bib` | 参考文献 |

## 编译（推荐 XeLaTeX）

在 `paper/` 目录下执行：

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

Windows（PowerShell）若已安装 TeX Live / MiKTeX：

```powershell
cd paper
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

编译成功后生成 `main.pdf`。

## 说明

- 作者、单位、致谢请按实际信息修改。
- 实验表格数据来自仓库正式评估结果（基础 Group Benefit 奖励）；注意力与 Candidate A 完整结果可在实验完成后补充。
- 引用条目为常用相关工作占位，投稿前请按目标期刊/会议格式核对并补全。
